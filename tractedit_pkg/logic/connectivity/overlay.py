"""Cached parcellation contours and their scene lifecycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .. import connectivity
from ...render_batch import batch_render, request_render

if TYPE_CHECKING:
    from ...main_window import MainWindow

logger = logging.getLogger(__name__)


def region_contour_bounds(data, affine):
    """Restrict FURY reslicing to grid-aligned crops for compact atlas IDs."""
    linear = affine[:3, :3]
    if not np.array_equal(linear, np.diag(np.diag(linear))):
        return ()
    highest = int(data.max())
    if not 0 < highest <= 65536:
        return ()
    from scipy.ndimage import find_objects

    return find_objects(data, max_label=highest)

def region_contour_input(data, affine, label, bounds):
    box = bounds[label - 1] if 0 < label <= len(bounds) else None
    if box is None:
        return (data == label).astype(np.uint8), affine
    origin = [max(0, axis.start - 1) for axis in box]
    crop = tuple(slice(start, axis.stop + 1) for start, axis in zip(origin, box))
    placement = affine.copy()
    placement[:3, 3] += affine[:3, :3] @ origin
    return (data[crop] == label).astype(np.uint8), placement

def _hue_to_rgb(hue: float) -> np.ndarray:
    """Convert hue (0-1) to RGB (0-1). Simple HSV to RGB with S=V=1."""
    if hue < 1 / 6:
        return np.array([1.0, hue * 6, 0.0])
    elif hue < 2 / 6:
        return np.array([(2 / 6 - hue) * 6, 1.0, 0.0])
    elif hue < 3 / 6:
        return np.array([0.0, 1.0, (hue - 2 / 6) * 6])
    elif hue < 4 / 6:
        return np.array([0.0, (4 / 6 - hue) * 6, 1.0])
    elif hue < 5 / 6:
        return np.array([(hue - 4 / 6) * 6, 0.0, 1.0])
    else:
        return np.array([1.0, 0.0, (1.0 - hue) * 6])


class ConnectivityOverlayMixin:
    """Overlay operations sharing the connectivity manager owner."""

    def is_parcellation_overlay_cache_current(self) -> bool:
        """Return whether cached actors match the current visible data."""
        mw = self.mw
        if mw.parcellation_data is None or mw.tractogram_data is None:
            return False
        expected_key = (
            self._endpoint_cache_owner(),
            getattr(mw, "_visibility_version", 0),
        )
        return bool(
            getattr(mw, "_parcellation_overlay_cached", False)
            and (
                getattr(mw, "parcellation_region_actors", {})
                or getattr(mw, "_session_deferred_regions", {})
            )
            and getattr(mw, "_parcellation_overlay_cache_key", None) == expected_key
        )

    @batch_render(lambda self: self.mw.vtk_panel)
    def create_parcellation_overlay(self, *, interactive: bool = True) -> bool:
        """
        Creates a 3D parcellation overlay with connected regions highlighted.

        Uses caching to avoid recomputation on toggle. Regions where streamlines
        terminate are colored distinctly.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None:
            connectivity.QMessageBox.warning(
                mw, "No Parcellation", "Please load a parcellation file first."
            )
            return False

        if mw.tractogram_data is None:
            connectivity.QMessageBox.warning(mw, "No Streamlines", "Please load streamlines first.")
            return False

        # If no visible streamlines (e.g. after ROI filtering), just show status - no popup
        if not mw.visible_indices:
            mw.vtk_panel.update_status(
                "No visible streamlines for parcellation overlay"
            )
            return False

        visible_indices = np.fromiter(
            mw.visible_indices,
            dtype=np.int64,
            count=len(mw.visible_indices),
        )
        overlay_cache_key = (
            self._endpoint_cache_owner(),
            getattr(mw, "_visibility_version", 0),
        )

        def report_step(step, message):
            if interactive:
                mw.vtk_panel.update_progress_bar(step, TOTAL_STEPS, visible=True)
                mw.vtk_panel.update_status(message, render=False)
                connectivity.QApplication.processEvents()

        if self.is_parcellation_overlay_cache_current():
            if self._show_parcellation_actors() is False:
                return False
            mw.vtk_panel.update_status(
                f"Parcellation overlay restored ({len(getattr(mw, 'parcellation_connected_labels', []))} regions)"
            )
            return True

        try:
            # Clean up any residual actors from previous computation to prevent crashes
            if interactive:
                self._safe_cleanup_actors()

            TOTAL_STEPS = 5
            report_step(
                0, "Computing parcellation overlay (Step 1/5: Extracting endpoints)..."
            )

            if self._ensure_endpoint_labels(visible_indices) is None:
                return False
            visible_rows = np.isin(
                mw.parcellation_visible_indices,
                visible_indices,
                assume_unique=True,
            )
            start_labels = mw.parcellation_start_labels[visible_rows]
            end_labels = mw.parcellation_end_labels[visible_rows]

            report_step(
                1, "Computing parcellation overlay (Step 2/5: Finding connected regions)..."
            )

            # Get unique connected labels (vectorized)
            all_endpoint_labels = np.concatenate([start_labels, end_labels])
            connected_labels = np.unique(all_endpoint_labels[all_endpoint_labels > 0])
            connected_set = set(connected_labels.tolist())

            logger.info(f"Found {len(connected_labels)} connected regions")
            mw.parcellation_connected_labels = connected_set

            # Vectorized color assignment. Generate colormap colors for all possible labels at once
            try:
                import matplotlib

                cmap = matplotlib.colormaps["tab20"]
                use_cmap = True
            except ImportError:
                use_cmap = False

            # Create lookup table: label -> color index
            label_to_color_idx = {
                label: i % 20 for i, label in enumerate(connected_labels)
            }

            # Pre-compute all colors (vectorized)
            if use_cmap:
                color_array = np.array([cmap(i)[:3] for i in range(20)]) * 255
            else:
                color_array = np.zeros((20, 3))
                for i in range(20):
                    hue = (i * 0.618033988749895) % 1.0
                    color_array[i] = connectivity._hue_to_rgb(hue)
                color_array *= 255

            color_array = color_array.astype(np.uint8)

            # Store color mapping for actors
            label_colors = {}
            for label in connected_labels:
                idx = label_to_color_idx[label]
                label_colors[int(label)] = tuple(color_array[idx].tolist()) + (200,)

            # Store label colors on main window for on-demand actor creation
            mw.parcellation_label_colors = label_colors

            report_step(
                2, "Computing parcellation overlay (Step 3/5: Creating base contour)..."
            )

            # Create VTK actors
            from fury import actor

            # Only colored connected regions are shown.
            mw.parcellation_overlay_actor = None  # No background actor

            # Create colored contours for connected regions (limit for performance)
            mw.parcellation_region_actors = {}  # {label: actor}
            max_regions = min(len(connected_labels), 20)  # Reduced limit for stability

            # Compute all region sizes in ONE pass using bincount
            all_region_sizes = np.bincount(mw.parcellation_data.ravel())

            # Build region_sizes list from precomputed bincount
            region_sizes = [
                (label, all_region_sizes[label] if label < len(all_region_sizes) else 0)
                for label in connected_labels
            ]
            region_sizes.sort(key=lambda x: x[1], reverse=True)

            region_visibility = getattr(mw, "parcellation_region_visibility", {})
            regions = [
                label for i, (label, size) in enumerate(region_sizes)
                if size >= 50 and (i < max_regions or region_visibility.get(label))
            ]
            report_step(
                3, f"Computing parcellation overlay (Step 4/5: Creating "
                f"{len(regions)} region contours)..."
            )

            bounds = connectivity.region_contour_bounds(mw.parcellation_data, mw.parcellation_affine) if regions else ()
            for label in regions:
                color = label_colors[int(label)]
                rgb_color = (color[0] / 255, color[1] / 255, color[2] / 255)

                try:
                    # Create binary mask for this region
                    region_mask, placement = connectivity.region_contour_input(
                        mw.parcellation_data, mw.parcellation_affine, label, bounds,
                    )

                    region_actor = actor.contour_from_roi(
                        region_mask,
                        affine=placement,
                        color=rgb_color,
                        opacity=0.85,  # High opacity for intersected regions
                    )
                    mw.parcellation_region_actors[int(label)] = region_actor
                except (ValueError, RuntimeError) as e:
                    if not interactive:
                        raise
                    logger.warning(f"Failed to create contour for label {label}: {e}")

            # Store set of labels with pre-created actors (main labels)
            mw.parcellation_main_labels = set(mw.parcellation_region_actors.keys())

            # Add all actors to scene, respecting visibility settings
            if mw.vtk_panel and mw.vtk_panel.scene:
                # Only add background actor if it exists
                if mw.parcellation_overlay_actor is not None:
                    mw.vtk_panel.scene.add(mw.parcellation_overlay_actor)

                for label, region_actor in mw.parcellation_region_actors.items():
                    # Only add to scene if marked visible (default True for new regions)
                    if region_visibility.get(label, True):
                        mw.vtk_panel.scene.add(region_actor)

            # Mark as cached
            mw._parcellation_overlay_cached = True
            mw._parcellation_overlay_visible = True
            mw._parcellation_overlay_cache_key = overlay_cache_key

            if not interactive:
                return True
            mw.vtk_panel.update_progress_bar(TOTAL_STEPS, TOTAL_STEPS, visible=True)
            connectivity.QApplication.processEvents()

            # Hide progress bar after short delay
            mw.vtk_panel.update_progress_bar(0, 0, visible=False)

            mw.vtk_panel.update_status(
                f"Parcellation overlay: {len(connected_labels)} connected regions "
                f"({len(mw.parcellation_region_actors)} displayed)"
            )

            # Update data panel to show connected regions
            try:
                mw._update_data_panel_display()
            except RuntimeError as e:
                if "has been deleted" in str(e):
                    logger.debug("Tree widget item deleted during update, retrying...")
                    # Small delay then retry
                    connectivity.QApplication.processEvents()
                    try:
                        mw._update_data_panel_display()
                    except RuntimeError:
                        logger.warning("Could not update data panel display")
                else:
                    raise

            # Ensure VTK is ready before allowing interaction
            try:
                # Non-blocking delay for VTK state stabilization
                from PyQt6.QtCore import QTimer

                QTimer.singleShot(100, lambda: self._finalize_render(mw))
            except RuntimeError:
                logger.debug(
                    "VTK render or timer setup failed during overlay creation."
                )

            return True

        except (ValueError, IndexError, TypeError, RuntimeError) as e:
            mw.vtk_panel.update_progress_bar(0, 0, visible=False)
            if not interactive:
                raise
            logger.error(f"Error creating parcellation overlay: {e}", exc_info=True)
            connectivity.QMessageBox.critical(
                mw, "Overlay Error", f"Failed to create parcellation overlay:\n{e}"
            )
            return False

    def _finalize_render(self, mw: "MainWindow") -> None:
        """
        Performs final render after non-blocking delay.

        Called by QTimer.singleShot to complete VTK state stabilization
        without blocking the UI thread.
        """
        try:
            if mw.vtk_panel and mw.vtk_panel.render_window:
                mw.vtk_panel.render_window.Render()
                connectivity.QApplication.processEvents()
        except RuntimeError:
            logger.debug("VTK render failed during finalize_render.")

    def _safe_cleanup_actors(self) -> None:
        """
        Safely cleans up any residual parcellation actors before recomputing.
        Prevents crashes from stale VTK state.
        """
        mw = self.mw
        mw._session_deferred_regions = {}

        if not mw.vtk_panel or not mw.vtk_panel.scene:
            return

        # Remove main parcellation actor from scene if it exists
        if hasattr(mw, "parcellation_overlay_actor") and mw.parcellation_overlay_actor:
            try:
                mw.vtk_panel.scene.rm(mw.parcellation_overlay_actor)
            except (ValueError, RuntimeError):
                logger.debug("Failed to remove parcellation overlay actor.")
            mw.parcellation_overlay_actor = None

        # Remove all region actors from scene
        if hasattr(mw, "parcellation_region_actors") and mw.parcellation_region_actors:
            for actor in list(mw.parcellation_region_actors.values()):
                try:
                    mw.vtk_panel.scene.rm(actor)
                except (ValueError, RuntimeError):
                    logger.debug("Failed to remove parcellation region actor.")
            mw.parcellation_region_actors = {}

        # Reset cache flags
        mw._parcellation_overlay_cached = False
        mw._parcellation_overlay_visible = False
        mw._parcellation_overlay_cache_key = None

        request_render(mw.vtk_panel, mw.vtk_panel.render_window)

    def _show_parcellation_actors(self) -> bool:
        """Shows cached parcellation actors (adds them back to scene)."""
        mw = self.mw
        if not mw.vtk_panel or not mw.vtk_panel.scene:
            return False

        if getattr(mw, "_session_deferred_regions", {}):
            from ..session_manager import materialize_session_regions

            labels = [
                label
                for label, presentation in mw._session_deferred_regions.items()
                if mw.parcellation_region_visibility.get(label, True)
                and presentation["visible"]
            ]
            if not materialize_session_regions(mw, labels):
                return False

        if hasattr(mw, "parcellation_overlay_actor") and mw.parcellation_overlay_actor:
            try:
                mw.vtk_panel.scene.add(mw.parcellation_overlay_actor)
            except (ValueError, RuntimeError):
                logger.debug("Failed to add parcellation overlay actor to scene.")

        # Only add region actors that are marked as visible
        if hasattr(mw, "parcellation_region_actors"):
            region_visibility = getattr(mw, "parcellation_region_visibility", {})
            for label, actor in mw.parcellation_region_actors.items():
                # Default to visible if not explicitly hidden
                if region_visibility.get(label, True):
                    try:
                        mw.vtk_panel.scene.add(actor)
                    except (ValueError, RuntimeError):
                        logger.debug("Failed to add region %s actor to scene.", label)

        request_render(mw.vtk_panel, mw.vtk_panel.render_window)
        mw._parcellation_overlay_visible = True
        return True

    def _hide_parcellation_actors(self, render: bool = True) -> None:
        """Hides parcellation actors (removes from scene but keeps cached)."""
        mw = self.mw
        for presentation in getattr(mw, "_session_deferred_regions", {}).values():
            presentation["attached"] = False
        if not mw.vtk_panel or not mw.vtk_panel.scene:
            return

        actor_removed = False
        if hasattr(mw, "parcellation_overlay_actor") and mw.parcellation_overlay_actor:
            try:
                mw.vtk_panel.scene.rm(mw.parcellation_overlay_actor)
                actor_removed = True
            except (ValueError, RuntimeError):
                logger.debug("Failed to remove parcellation overlay from scene.")

        if hasattr(mw, "parcellation_region_actors"):
            for actor in mw.parcellation_region_actors.values():
                try:
                    mw.vtk_panel.scene.rm(actor)
                    actor_removed = True
                except (ValueError, RuntimeError):
                    logger.debug("Failed to remove parcellation region actor.")

        if render and actor_removed:
            request_render(mw.vtk_panel, mw.vtk_panel.render_window)
        mw._parcellation_overlay_visible = False

    @batch_render(lambda self: self.mw.vtk_panel)
    def remove_parcellation_overlay(self) -> None:
        """
        Hides the parcellation overlay from the 3D scene.

        Keeps actors cached so the overlay can be quickly restored without
        recomputation. Use _clear_parcellation() to fully clear all data.
        """
        mw = self.mw

        # Hide actors from scene (keeping them cached)
        self._hide_parcellation_actors()

        mw._parcellation_overlay_visible = False

        mw.vtk_panel.update_status("Parcellation overlay hidden")

    def _invalidate_overlay_cache(self, render: bool = True) -> None:
        """Clear visibility-dependent parcellation actors."""
        mw = self.mw
        mw._session_deferred_regions = {}
        self._hide_parcellation_actors(render=render)
        mw.parcellation_overlay_actor = None
        mw.parcellation_region_actors = {}
        mw.parcellation_connected_labels = set()
        mw.parcellation_main_labels = set()
        mw.parcellation_label_colors = {}
        mw._parcellation_overlay_cached = False
        mw._parcellation_overlay_visible = False
        mw._parcellation_overlay_cache_key = None

    def toggle_region_visibility(
        self, label: int, visible: bool, batch_mode: bool = False
    ) -> None:
        """
        Toggles visibility of an individual parcellation region.

        Supports on-demand actor creation for regions beyond the initial limit.

        Args:
            label: The parcellation label ID.
            visible: True to show, False to hide.
            batch_mode: If True, skip render and status update (for bulk operations).
        """
        ##TODO - to refactor

        mw = self.mw

        if not hasattr(mw, "parcellation_region_actors"):
            mw.parcellation_region_actors = {}

        # Don't create actors if the main overlay is not visible
        overlay_visible = getattr(mw, "_parcellation_overlay_visible", False)

        if (
            visible
            and overlay_visible
            and label in getattr(mw, "_session_deferred_regions", {})
        ):
            from ..session_manager import materialize_session_regions

            if not materialize_session_regions(mw, [label]):
                return

        # If toggling ON and no actor exists, create on-demand
        if visible and label not in mw.parcellation_region_actors:
            # Only create on-demand if the overlay is currently visible
            if not overlay_visible:
                logger.debug(
                    f"Skipping on-demand actor creation for {label} - overlay not visible"
                )
                return

            # Check if this is a connected region (has streamline endpoints)
            connected_labels = getattr(mw, "parcellation_connected_labels", set())
            if label in connected_labels:
                logger.info(f"Creating on-demand actor for region {label}")
                success = self._create_region_actor_on_demand(label)
                if not success:
                    logger.warning(f"Failed to create actor for region {label}")
                    return
            else:
                logger.debug(f"Region {label} is not a connected region")
                return

        # Now toggle visibility
        if label not in mw.parcellation_region_actors:
            logger.debug(f"Region {label} has no actor to toggle")
            return

        region_actor = mw.parcellation_region_actors[label]

        # Only manipulate scene if overlay is visible
        if overlay_visible and mw.vtk_panel and mw.vtk_panel.scene:
            try:
                if visible:
                    mw.vtk_panel.scene.add(region_actor)
                else:
                    mw.vtk_panel.scene.rm(region_actor)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Failed to toggle region {label} visibility: {e}")

        # Only update status if not in batch mode and overlay is visible
        if not batch_mode and overlay_visible:
            region_name = mw.parcellation_labels.get(label, f"Region_{label}")
            status = "shown" if visible else "hidden"
            mw.vtk_panel.update_status(f"{region_name} {status}")

    def _create_region_actor_on_demand(self, label: int) -> bool:
        """
        Creates a single region actor on-demand.

        Used for regions beyond the initial 20-region limit that are
        toggled ON by the user.

        Args:
            label: The parcellation label ID.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None:
            return False

        try:
            from fury import actor

            # Get color from stored label colors, or generate one
            label_colors = getattr(mw, "parcellation_label_colors", {})
            if label in label_colors:
                color = label_colors[label]
                rgb_color = (color[0] / 255, color[1] / 255, color[2] / 255)
            else:
                # Generate a color using golden ratio cycling
                connected_labels = list(
                    getattr(mw, "parcellation_connected_labels", set())
                )
                try:
                    idx = connected_labels.index(label)
                except ValueError:
                    idx = label  # Use label as index if not found
                hue = (idx * 0.618033988749895) % 1.0
                rgb = connectivity._hue_to_rgb(hue)
                rgb_color = tuple(rgb.tolist())

                # Store the color for future use
                if not hasattr(mw, "parcellation_label_colors"):
                    mw.parcellation_label_colors = {}
                mw.parcellation_label_colors[label] = (
                    int(rgb_color[0] * 255),
                    int(rgb_color[1] * 255),
                    int(rgb_color[2] * 255),
                    200,
                )

            # Create binary mask for this region
            bounds = connectivity.region_contour_bounds(mw.parcellation_data, mw.parcellation_affine)
            region_mask, placement = connectivity.region_contour_input(
                mw.parcellation_data, mw.parcellation_affine, label, bounds,
            )

            # Check region size
            size = np.sum(region_mask)
            if size < 50:
                logger.warning(f"Region {label} is too small ({size} voxels)")
                return False

            # Create contour actor
            region_actor = actor.contour_from_roi(
                region_mask,
                affine=placement,
                color=rgb_color,
                opacity=0.85,
            )

            # Store the actor
            mw.parcellation_region_actors[label] = region_actor

            logger.info(f"Created on-demand actor for region {label}")
            return True

        except (ValueError, RuntimeError) as e:
            logger.error(f"Error creating on-demand actor for region {label}: {e}")
            return False
