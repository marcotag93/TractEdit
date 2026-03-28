# -*- coding: utf-8 -*-

"""
Streamlines manager for TractEdit visualization.

Handles streamline visualization including actor creation, color mapping,
scalar bar management, and highlight actors for selected streamlines.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import numpy as np
import vtk
from fury import actor, colormap
from vtk.util import numpy_support

from ..utils import ColorMode, MAX_HIGHLIGHT_STREAMLINES

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


# ============================================================================
# Streamlines Manager Class
# ============================================================================


class StreamlinesManager:
    """
    Manages streamline visualization operations.

    Handles actor creation, color mapping, scalar bar display,
    and highlight actors for selected streamlines.
    """

    # Threshold for incremental vs full rebuild
    INCREMENTAL_THRESHOLD = 50

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        """
        Initialize the streamlines manager.

        Args:
            vtk_panel: Reference to the parent VTKPanel instance.
        """
        self.panel = vtk_panel
        # Cache for incremental highlight updates
        self._last_selected_indices: Set[int] = set()
        # Deferred update state
        self._pending_highlight_update: bool = False
        self._update_timer = None

    def update_highlight(self) -> None:
        """Updates the actor for highlighted/selected streamlines."""
        if not self.panel.scene:
            return

        # Check prerequisites early
        if (
            not self.panel.main_window
            or not hasattr(self.panel.main_window, "selected_streamline_indices")
            or not self.panel.main_window.tractogram_data
        ):
            # Remove any existing highlight actor
            if self.panel.highlight_actor is not None:
                try:
                    self.panel.scene.rm(self.panel.highlight_actor)
                except (ValueError, AttributeError):
                    pass
                self.panel.highlight_actor = None
            if self.panel.main_window:
                self.panel.main_window._update_action_states()
            return

        selected_indices: Set[int] = self.panel.main_window.selected_streamline_indices

        # EARLY EXIT: If selection is empty, just remove existing actor and return
        if not selected_indices:
            if self.panel.highlight_actor is not None:
                try:
                    self.panel.scene.rm(self.panel.highlight_actor)
                except (ValueError, AttributeError):
                    pass
                self.panel.highlight_actor = None
            self._last_selected_indices = set()
            if self.panel.main_window:
                self.panel.main_window._update_action_states()
            return

        # SAFETY GUARD: Building a VTK highlight actor allocates a flat point
        # buffer that scales with |selection| × avg_points_per_streamline.
        if len(selected_indices) > MAX_HIGHLIGHT_STREAMLINES:
            logger.warning(
                "update_highlight skipped: selection size %d exceeds "
                "MAX_HIGHLIGHT_STREAMLINES=%d. Highlight actor not built.",
                len(selected_indices),
                MAX_HIGHLIGHT_STREAMLINES,
            )
            if self.panel.main_window:
                self.panel.main_window._update_action_states()
            return

        # Safely Remove Existing Highlight Actor
        if self.panel.highlight_actor is not None:
            try:
                self.panel.scene.rm(self.panel.highlight_actor)
            except (ValueError, AttributeError):
                pass
            except (RuntimeError, AttributeError) as e:
                logger.error(
                    f"  Error removing highlight actor: {e}. Proceeding cautiously."
                )
            finally:
                self.panel.highlight_actor = None

        tractogram = self.panel.main_window.tractogram_data

        # Create new actor only if there's a valid selection
        if selected_indices:
            tractogram_len = len(tractogram)
            valid_indices_arr = np.array(
                [idx for idx in selected_indices if 0 <= idx < tractogram_len],
                dtype=np.int64,
            )

            if len(valid_indices_arr) == 0:
                if self.panel.main_window:
                    self.panel.main_window._update_action_states()
                return

            # Build highlight actor - use vectorized VTK polydata path when
            # ArraySequence internals are available (same approach as the main
            # actor fast path), falling back to FURY actor.line() otherwise.
            has_array_sequence = (
                hasattr(tractogram, "_data")
                and hasattr(tractogram, "_offsets")
                and hasattr(tractogram, "_lengths")
            )

            try:
                if has_array_sequence:
                    # FAST PATH: Vectorized VTK polydata construction.
                    # Bypasses FURY's Python-loop actor.line() entirely
                    src_data = tractogram._data
                    src_offsets = tractogram._offsets
                    src_lengths = tractogram._lengths

                    # Gather lengths via vectorized fancy indexing (no loop)
                    hl_lengths = src_lengths[valid_indices_arr]

                    # Filter out zero-length streamlines
                    valid_mask = hl_lengths > 0
                    hl_indices = valid_indices_arr[valid_mask]
                    hl_lengths = hl_lengths[valid_mask]

                    if len(hl_indices) == 0:
                        if self.panel.main_window:
                            self.panel.main_window._update_action_states()
                        return

                    # Compute cumulative destination offsets (N+1 entries)
                    dst_offsets_arr = np.empty(len(hl_lengths) + 1, dtype=np.int64)
                    dst_offsets_arr[0] = 0
                    np.cumsum(hl_lengths, out=dst_offsets_arr[1:])
                    total_points = int(dst_offsets_arr[-1])

                    # Allocate flat output buffer and copy only the selected
                    # streamlines' data via sequential contiguous memcpy.
                    # This is faster than scatter-gather for typical selection sizes
                    flat_points = np.empty((total_points, 3), dtype=np.float32)
                    src_starts = src_offsets[hl_indices].astype(np.int64)
                    dst_starts = dst_offsets_arr[:-1].astype(np.int64)
                    sub_lengths = hl_lengths.astype(np.int64)
                    for i in range(len(hl_indices)):
                        s = int(src_starts[i])
                        d = int(dst_starts[i])
                        n = int(sub_lengths[i])
                        flat_points[d : d + n] = src_data[s : s + n]

                    # Build VTK polydata (yellow) and actor directly
                    poly_data = self._build_polydata_direct(
                        flat_points,
                        hl_lengths,
                        dst_offsets_arr,
                        (1.0, 1.0, 0.0),  # yellow
                        len(hl_indices),
                    )
                    hl_mapper = vtk.vtkPolyDataMapper()
                    hl_mapper.SetInputData(poly_data)
                    hl_mapper.ScalarVisibilityOn()
                    hl_mapper.SetScalarModeToUsePointFieldData()
                    hl_mapper.SelectColorArray("colors")
                    hl_mapper.Update()
                    hl_actor = vtk.vtkActor()
                    hl_actor.SetMapper(hl_mapper)
                    hl_actor.GetProperty().SetLineWidth(6)
                    hl_actor.GetProperty().SetOpacity(1.0)
                    self.panel.highlight_actor = hl_actor

                else:
                    # SLOW PATH: FURY fallback for non-ArraySequence data
                    selected_sl_data_list = []
                    for idx in valid_indices_arr:
                        try:
                            sl = tractogram[idx]
                            if sl is not None and len(sl) > 0:
                                selected_sl_data_list.append(
                                    sl.astype(np.float32, copy=False)
                                )
                        except (IndexError, ValueError, TypeError):
                            logger.debug(
                                "Failed to extract streamline at index %d.", idx
                            )

                    if not selected_sl_data_list:
                        if self.panel.main_window:
                            self.panel.main_window._update_action_states()
                        return

                    self.panel.highlight_actor = actor.line(
                        selected_sl_data_list,
                        colors=(1, 1, 0),
                        linewidth=6,
                        opacity=1.0,
                    )

                self.panel.scene.add(self.panel.highlight_actor)

                if (
                    self.panel.main_window
                    and not self.panel.main_window.bundle_is_visible
                ):
                    self.panel.highlight_actor.SetVisibility(0)

                if self.panel.highlight_actor:
                    try:
                        mapper = self.panel.highlight_actor.GetMapper()
                        if mapper:
                            mapper.SetResolveCoincidentTopologyToPolygonOffset()
                            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                                -1, -15
                            )
                    except (RuntimeError, AttributeError) as e:
                        logger.warning(
                            f"Warning: Could not apply mapper offset to highlight_actor: {e}"
                        )

            except (RuntimeError, ValueError, AttributeError) as e:
                logger.error(f"Error creating highlight actor:", exc_info=True)
                self.panel.highlight_actor = None

        # Update UI action states
        if self.panel.main_window:
            self.panel.main_window._update_action_states()

        # Cache current selection for future comparison
        self._last_selected_indices = (
            selected_indices.copy() if selected_indices else set()
        )

    def schedule_highlight_update(self, delay_ms: int = 30) -> None:
        """
        Schedule a deferred highlight update with debouncing.

        Multiple calls within the delay period will be coalesced into
        a single update, preventing redundant rebuilds during rapid
        selection operations.

        Args:
            delay_ms: Delay in milliseconds before executing the update.
        """
        from PyQt6.QtCore import QTimer

        self._pending_highlight_update = True

        # Cancel existing timer if any
        if self._update_timer is not None:
            self._update_timer.stop()
            self._update_timer.deleteLater()

        # Create new timer
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._execute_deferred_highlight)
        self._update_timer.start(delay_ms)

    def _execute_deferred_highlight(self) -> None:
        """Execute the deferred highlight update."""
        self._pending_highlight_update = False
        self._update_timer = None
        self.update_highlight()

    def has_selection_changed_significantly(self) -> bool:
        """
        Check if the selection has changed significantly enough to warrant rebuild.

        Returns:
            True if selection changed by more than INCREMENTAL_THRESHOLD items.
        """
        if not self.panel.main_window:
            return True

        current = self.panel.main_window.selected_streamline_indices or set()
        added = current - self._last_selected_indices
        removed = self._last_selected_indices - current

        return len(added) + len(removed) > self.INCREMENTAL_THRESHOLD

    def update_roi_highlight_actor(self) -> None:
        """
        Updates the Red highlight actor based on ROI 'Select' indices.
        """
        if not self.panel.scene:
            return

        # Remove existing
        if self.panel.roi_highlight_actor:
            try:
                self.panel.scene.rm(self.panel.roi_highlight_actor)
            except (ValueError, RuntimeError):
                logger.debug("Failed to remove ROI highlight actor from scene.")
            self.panel.roi_highlight_actor = None

        if not self.panel.main_window or not self.panel.main_window.tractogram_data:
            return

        indices = getattr(self.panel.main_window, "roi_highlight_indices", set())
        if not indices:
            return

        # Get actual streamline data - optimized for ArraySequence
        tractogram = self.panel.main_window.tractogram_data
        tractogram_len = len(tractogram)

        valid_indices_arr = np.array(
            [idx for idx in indices if 0 <= idx < tractogram_len], dtype=np.int64
        )

        if len(valid_indices_arr) == 0:
            return

        streamlines_list = []
        has_array_sequence = (
            hasattr(tractogram, "_data")
            and hasattr(tractogram, "_offsets")
            and hasattr(tractogram, "_lengths")
        )

        if has_array_sequence:
            src_data = tractogram._data
            src_offsets = tractogram._offsets
            src_lengths = tractogram._lengths

            for idx in valid_indices_arr:
                length = src_lengths[idx]
                if length > 0:
                    start = src_offsets[idx]
                    end = start + length
                    streamlines_list.append(
                        src_data[start:end].astype(np.float32, copy=False)
                    )
        else:
            for idx in valid_indices_arr:
                try:
                    sl = tractogram[idx]
                    if sl is not None and len(sl) > 0:
                        streamlines_list.append(sl.astype(np.float32, copy=False))
                except (IndexError, ValueError, TypeError):
                    logger.debug("Failed to extract streamline at index %d.", idx)

        if not streamlines_list:
            return

        try:
            # Create Red Actor (1, 0, 0)
            self.panel.roi_highlight_actor = actor.line(
                streamlines_list,
                colors=(1, 0, 0),
                linewidth=4,
                opacity=1.0,
                depth_cue=False,  # Make it pop out
            )
            self.panel.scene.add(self.panel.roi_highlight_actor)

            # Use mapper offset to ensure it draws ON TOP of the grey/yellow bundles
            mapper = self.panel.roi_highlight_actor.GetMapper()
            if mapper:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                    -2, -20
                )  # Stronger offset than yellow

            self.panel.render_window.Render()

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error(f"Error creating ROI highlight actor: {e}")

    def update_invert_contour(self) -> None:
        """Build or remove the cyan keeper-contour actor for inversion mode.

        Renders the pre-inversion keeper streamlines in cyan (#05B8CC) to give
        the user spatial feedback about which streamlines will *survive* the
        deletion.  The keeper set is always small (built with the sphere tool),
        so the simple per-streamline loop used by ``update_roi_highlight_actor``
        is perfectly adequate — no vectorised polydata path is needed.

        The actor is only built when ``_inversion_active`` is True and
        ``_inversion_keeper_indices`` is non-empty; otherwise any existing
        contour actor is removed and the method returns.
        """
        if not self.panel.scene:
            return

        # Remove any existing contour actor before rebuilding.
        if self.panel.invert_contour_actor is not None:
            try:
                self.panel.scene.rm(self.panel.invert_contour_actor)
            except (ValueError, AttributeError):
                pass
            self.panel.invert_contour_actor = None

        mw = self.panel.main_window
        if not mw:
            return

        keeper_indices = mw._inversion_keeper_indices
        if not keeper_indices or not mw._inversion_active:
            return

        tractogram = mw.tractogram_data
        if not tractogram:
            return

        tractogram_len = len(tractogram)
        valid_indices_arr = np.array(
            [idx for idx in keeper_indices if 0 <= idx < tractogram_len],
            dtype=np.int64,
        )
        if len(valid_indices_arr) == 0:
            return

        # Collect point arrays for each keeper streamline.
        streamlines_list = []
        has_array_sequence = (
            hasattr(tractogram, "_data")
            and hasattr(tractogram, "_offsets")
            and hasattr(tractogram, "_lengths")
        )

        if has_array_sequence:
            src_data = tractogram._data
            src_offsets = tractogram._offsets
            src_lengths = tractogram._lengths
            for idx in valid_indices_arr:
                length = src_lengths[idx]
                if length > 0:
                    start = src_offsets[idx]
                    streamlines_list.append(
                        src_data[start : start + length].astype(np.float32, copy=False)
                    )
        else:
            for idx in valid_indices_arr:
                try:
                    sl = tractogram[idx]
                    if sl is not None and len(sl) > 0:
                        streamlines_list.append(sl.astype(np.float32, copy=False))
                except (IndexError, ValueError, TypeError):
                    logger.debug(
                        "Failed to extract keeper streamline at index %d.", idx
                    )

        if not streamlines_list:
            return

        try:
            # Cyan #05B8CC → VTK float (0.020, 0.722, 0.800).
            self.panel.invert_contour_actor = actor.line(
                streamlines_list,
                colors=(0.020, 0.722, 0.800),
                linewidth=8,
                opacity=1.0,
                depth_cue=False,
            )
            self.panel.scene.add(self.panel.invert_contour_actor)

            # Polygon offset ensures the contour renders on top of the main
            # bundle and the yellow highlight (same technique as the ROI actor).
            mapper = self.panel.invert_contour_actor.GetMapper()
            if mapper:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -20)

            # Honour bundle-level visibility toggled by the user.
            if not mw.bundle_is_visible:
                self.panel.invert_contour_actor.SetVisibility(0)

            self.panel.render_window.Render()
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Error creating invert contour actor: %s", e)
            self.panel.invert_contour_actor = None

    def calculate_scalar_colors(
        self,
        streamlines_gen,
        scalar_gen,
        vmin: float,
        vmax: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates vertex colors based on a list of scalar arrays per streamline,
        using the provided vmin and vmax for the colormap range.
        Returns a single concatenated (TotalPoints, 3) numpy array for FURY.

        Args:
            streamlines_gen: Generator/iterable of streamline arrays.
            scalar_gen: Generator/iterable of scalar arrays (one per streamline).
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.

        Returns:
            Dictionary with 'colors', 'opacity', 'linewidth', 'lut' keys, or None.
        """

        # Create the LUT
        try:
            lut = vtk.vtkLookupTable()
            table_min = vmin - 0.5 if vmin == vmax else vmin
            table_max = vmax + 0.5 if vmin == vmax else vmax

            lut.SetTableRange(table_min, table_max)
            lut.SetHueRange(0.667, 0.0)  # Blue to Red (standard)
            lut.Build()
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error creating scalar LUT: {e}. Defaulting to grey.")
            return None  # Fallback to grey

        # Build the list of color arrays (one per *non-empty* streamline)
        default_color_rgb = np.array([128, 128, 128], dtype=np.uint8)
        vertex_colors_list: List[np.ndarray] = []  # List to hold individual np.arrays
        rgb_output: List[float] = [0.0, 0.0, 0.0]

        for sl, sl_scalars in zip(streamlines_gen, scalar_gen):
            num_points = len(sl) if sl is not None else 0
            if num_points == 0:  # Skip empty streamlines
                continue

            sl_colors_rgb = np.empty((num_points, 3), dtype=np.uint8)

            # Check if this (non-empty) streamline has valid scalar data
            has_valid_scalar_for_this_sl = False
            if (
                sl_scalars is not None
                and hasattr(sl_scalars, "size")
                and len(sl_scalars) == num_points
            ):
                try:
                    for j in range(num_points):
                        lut.GetColor(sl_scalars[j], rgb_output)
                        sl_colors_rgb[j] = [int(c * 255) for c in rgb_output]
                    has_valid_scalar_for_this_sl = True
                except (TypeError, ValueError, IndexError):
                    has_valid_scalar_for_this_sl = False  # e.g., non-numeric data

            if not has_valid_scalar_for_this_sl:
                sl_colors_rgb[:] = default_color_rgb  # Fill with default color

            vertex_colors_list.append(sl_colors_rgb)

        if not vertex_colors_list:
            return None

        # Concatenate all color arrays into one big array
        try:
            concatenated_colors = np.concatenate(vertex_colors_list, axis=0)
        except ValueError as ve:
            logger.error(f"Failed to concatenate color arrays: {ve}")
            return None  # Fallback to grey

        return {
            "colors": concatenated_colors,
            "opacity": 0.8,
            "linewidth": 3,
            "lut": lut,
        }

    def update_scalar_bar(self, lut: vtk.vtkLookupTable, title: str) -> None:
        """Creates or updates the scalar bar actor with improved UX, title positioning, and precision."""
        if not self.panel.scene:
            return

        if self.panel.scalar_bar_actor is None:
            self.panel.scalar_bar_actor = vtk.vtkScalarBarActor()
            self.panel.scalar_bar_actor.SetOrientationToVertical()

            # Size and Positioning
            self.panel.scalar_bar_actor.SetWidth(0.04)
            self.panel.scalar_bar_actor.SetHeight(0.35)

            # Position: (X, Y) normalized viewport coordinates
            self.panel.scalar_bar_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            self.panel.scalar_bar_actor.SetPosition(0.94, 0.32)

            # Typography
            title_prop = vtk.vtkTextProperty()
            title_prop.SetColor(0.9, 0.9, 0.9)
            title_prop.SetFontSize(16)
            title_prop.SetFontFamilyToArial()
            title_prop.ItalicOn()
            title_prop.ShadowOn()
            title_prop.SetJustificationToRight()

            # Label Styling
            label_prop = vtk.vtkTextProperty()
            label_prop.SetColor(0.8, 0.8, 0.8)
            label_prop.SetFontSize(10)
            label_prop.SetFontFamilyToArial()
            label_prop.ShadowOff()

            self.panel.scalar_bar_actor.SetTitleTextProperty(title_prop)
            self.panel.scalar_bar_actor.SetLabelTextProperty(label_prop)

            # Formatting
            self.panel.scalar_bar_actor.SetNumberOfLabels(5)
            self.panel.scalar_bar_actor.SetLabelFormat("%-.4g")
            self.panel.scalar_bar_actor.UnconstrainedFontSizeOn()
            self.panel.scalar_bar_actor.SetTextPositionToPrecedeScalarBar()
            self.panel.scene.add(self.panel.scalar_bar_actor)

        # Update Data
        self.panel.scalar_bar_actor.SetLookupTable(lut)
        self.panel.scalar_bar_actor.SetTitle(title + "\n")
        self.panel.scalar_bar_actor.SetVisibility(1)
        self.panel.scalar_bar_actor.Modified()

    def get_streamline_actor_params(self) -> Dict[str, Any]:
        """
        Determines parameters for the main streamlines actor using STRIDE.

        Optimized to avoid double-copying data by using simple lists instead
        of rebuilding ArraySequence containers. Uses ArraySequence internals
        for batch extraction when available.

        Returns:
            Dictionary with streamline visualization parameters.
        """
        # Get opacity from MainWindow state
        current_opacity = getattr(self.panel.main_window, "bundle_opacity", 1.0)

        # Default parameters
        params: Dict[str, Any] = {
            "colors": (0.8, 0.8, 0.8),
            "opacity": current_opacity,
            "linewidth": 1,
        }

        if not self.panel.main_window or not self.panel.main_window.tractogram_data:
            return params

        tractogram = self.panel.main_window.tractogram_data
        visible_indices = self.panel.main_window.visible_indices

        if not visible_indices:
            params["streamlines_list"] = []
            return params

        # Convert to numpy array for efficient slicing (no sorting needed)
        visible_indices_arr = np.fromiter(visible_indices, dtype=np.int64)

        # Apply STRIDE (Skip Logic)
        stride = self.panel.main_window.render_stride
        subset_indices = visible_indices_arr[::stride]  # Only pick 1 every N

        if len(subset_indices) == 0:
            params["streamlines_list"] = []
            return params

        # FAST PATH: Vectorized batch extraction using AOT kernel
        has_array_sequence = (
            hasattr(tractogram, "_data")
            and hasattr(tractogram, "_offsets")
            and hasattr(tractogram, "_lengths")
        )

        if has_array_sequence:
            src_data = tractogram._data
            src_offsets = tractogram._offsets
            src_lengths = tractogram._lengths

            # Gather lengths for the subset (vectorized fancy indexing)
            subset_lengths = src_lengths[subset_indices]

            # Filter out zero-length streamlines
            valid_mask = subset_lengths > 0
            subset_indices = subset_indices[valid_mask]
            subset_lengths = subset_lengths[valid_mask]

            if len(subset_indices) == 0:
                params["streamlines_list"] = []
                return params

            # Compute destination offsets via cumsum
            dst_offsets = np.empty(len(subset_lengths) + 1, dtype=np.int64)
            dst_offsets[0] = 0
            np.cumsum(subset_lengths, out=dst_offsets[1:])
            total_points = int(dst_offsets[-1])

            # Gather source offsets for the strided subset
            src_starts = src_offsets[subset_indices].astype(np.int64)
            sub_lengths = subset_lengths.astype(np.int64)

            # Build flat float32 output buffer via vectorised scatter-gather.
            #
            #    The AOT copy_streamlines_chunk kernel requires float64 arrays,
            #    which previously forced a full-tractogram dtype conversion
            #    (np.ascontiguousarray(src_data, dtype=np.float64)) before a
            #    single point was copied.
            #
            #      seg_ids[j]          : which subset-segment point j belongs to
            #      local_offsets[j]    : offset of point j within that segment
            #      src_point_indices[j]: absolute row in src_data for point j
            seg_ids = np.repeat(
                np.arange(len(sub_lengths), dtype=np.int64), sub_lengths
            )
            local_offsets = (
                np.arange(total_points, dtype=np.int64) - dst_offsets[seg_ids]
            )
            src_point_indices = src_starts[seg_ids] + local_offsets

            # Fancy-index the float32 source directly — no full-tractogram
            # float64 conversion required.
            flat_points_f32 = np.ascontiguousarray(
                src_data[src_point_indices], dtype=np.float32
            )

            # Store flat buffer + metadata for direct VTK construction
            params["_flat_points"] = flat_points_f32
            params["_subset_lengths"] = subset_lengths
            params["_dst_offsets"] = dst_offsets
            params["_num_streamlines"] = len(subset_indices)
            params["_subset_indices"] = subset_indices

            # Build list view for backward compatibility with FURY path
            # (zero-copy slicing into the flat buffer)
            visible_streamlines_list = [
                flat_points_f32[dst_offsets[i] : dst_offsets[i + 1]]
                for i in range(len(subset_indices))
            ]
        else:
            # SLOW PATH: Fall back to individual indexing
            visible_streamlines_list = [
                tractogram[i].astype(np.float32, copy=False) for i in subset_indices
            ]

        # Coloring Logic
        current_mode = self.panel.main_window.current_color_mode

        if current_mode == ColorMode.ORIENTATION:
            if "_flat_points" in params:
                # FAST PATH: Vectorized orientation color computation.
                # Replicates FURY's colormap.line_colors() + orient2rgb()
                flat_pts = params["_flat_points"]
                sub_lengths = params["_subset_lengths"]
                offsets = params["_dst_offsets"]

                # Extract first and last point of each streamline
                first_idx = offsets[:-1].astype(np.intp)
                last_idx = (offsets[1:] - 1).astype(np.intp)
                first_pts = flat_pts[first_idx]  # (N, 3)
                last_pts = flat_pts[last_idx]  # (N, 3)

                # Vectorized DEC orientation: |normalize(last - first)|
                directions = last_pts - first_pts  # (N, 3)
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)  # guard against zero-length
                orient_colors = np.abs(directions / norms)  # (N, 3) float in [0, 1]

                # Expand per-streamline colors to per-point
                colors_per_point = np.repeat(
                    orient_colors.astype(np.float32), sub_lengths, axis=0
                )
                params["colors"] = colors_per_point
            else:
                # SLOW PATH: FURY's list-based orientation coloring
                params["colors"] = colormap.line_colors(visible_streamlines_list)
            params["opacity"] = current_opacity

        elif current_mode == ColorMode.SCALAR:
            active_scalar = self.panel.main_window.active_scalar_name
            scalar_seq = None
            is_per_streamline = False

            # Try generic dictionary
            if (
                self.panel.main_window.scalar_data_per_point
                and active_scalar in self.panel.main_window.scalar_data_per_point
            ):
                scalar_seq = self.panel.main_window.scalar_data_per_point[active_scalar]

            # Try TRX 'Data Per Vertex' (dpv)
            elif (
                hasattr(tractogram, "data_per_vertex")
                and active_scalar in tractogram.data_per_vertex
            ):
                scalar_seq = tractogram.data_per_vertex[active_scalar]

            # Try TRX 'Data Per Streamline' (dps)
            elif (
                hasattr(tractogram, "data_per_streamline")
                and active_scalar in tractogram.data_per_streamline
            ):
                scalar_seq = tractogram.data_per_streamline[active_scalar]
                is_per_streamline = True

            if scalar_seq is not None:
                try:
                    flat_scalars = None

                    if is_per_streamline:
                        # Extract values for the subset
                        subset_values = [scalar_seq[i] for i in subset_indices]

                        # Calculate lengths manually
                        lengths = [len(sl) for sl in visible_streamlines_list]

                        # Repeat value N times for each streamline
                        flat_scalars = np.repeat(subset_values, lengths)
                    else:
                        subset_arrays = [scalar_seq[i] for i in subset_indices]
                        flat_scalars = np.concatenate(subset_arrays)

                    # Vectorized Color Mapping
                    vmin = self.panel.main_window.scalar_min_val
                    vmax = self.panel.main_window.scalar_max_val

                    # Create LUT
                    lut = vtk.vtkLookupTable()
                    table_min = vmin - 0.5 if vmin == vmax else vmin
                    table_max = vmax + 0.5 if vmin == vmax else vmax
                    lut.SetTableRange(table_min, table_max)
                    lut.SetHueRange(0.667, 0.0)  # Blue to Red
                    lut.Build()

                    # Convert scalars to VTK array (explicit float32)
                    vtk_scalars = numpy_support.numpy_to_vtk(
                        flat_scalars.astype(np.float32),
                        deep=True,
                        array_type=vtk.VTK_FLOAT,
                    )

                    # Map Scalars (Returns vtkUnsignedCharArray)
                    vtk_colors = lut.MapScalars(
                        vtk_scalars, vtk.VTK_COLOR_MODE_DEFAULT, -1
                    )

                    # Convert back to Numpy and strip Alpha
                    colors_flat = numpy_support.vtk_to_numpy(vtk_colors)
                    colors_rgb = colors_flat[:, :3]

                    params["colors"] = colors_rgb
                    params["lut"] = lut
                    params["opacity"] = current_opacity

                except (ValueError, IndexError, TypeError, RuntimeError) as e:
                    logger.error(
                        f"Error during vectorized scalar calculation: {e}",
                        exc_info=True,
                    )
                    params["colors"] = (0.5, 0.5, 0.5)

        params["streamlines_list"] = visible_streamlines_list
        return params

    def _build_polydata_direct(
        self,
        flat_points: np.ndarray,
        lengths: np.ndarray,
        dst_offsets: np.ndarray,
        colors: Any,
        num_streamlines: int,
    ) -> "vtk.vtkPolyData":
        """Build vtkPolyData directly from flat arrays, bypassing FURY.

        Constructs the VTK rendering data structure using fully vectorized
        numpy operations.

        Args:
            flat_points: ``(TotalPoints, 3)`` float32 array of all vertices.
            lengths: ``(N,)`` array of per-streamline point counts.
            dst_offsets: ``(N+1,)`` cumulative offset array (int64).
            colors: Per-point colors as ``(TotalPoints, 3)`` float32 or
                uint8 array, or a uniform RGB tuple ``(r, g, b)`` with
                values in [0, 1].
            num_streamlines: Number of streamlines (used for validation).

        Returns:
            Fully configured ``vtkPolyData`` with points, lines, and
            per-point color scalars.
        """
        total_points = len(flat_points)

        # Points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(
            numpy_support.numpy_to_vtk(
                np.ascontiguousarray(flat_points, dtype=np.float32),
                deep=True,
                array_type=vtk.VTK_FLOAT,
            )
        )

        # Cell array (fully vectorized) 
        # Connectivity is identity: [0, 1, 2, ..., total_points - 1]
        connectivity = np.arange(total_points, dtype=np.int64)
        offsets = np.ascontiguousarray(dst_offsets, dtype=np.int64)

        vtk_offsets = numpy_support.numpy_to_vtk(
            offsets, deep=True, array_type=vtk.VTK_ID_TYPE
        )
        vtk_conn = numpy_support.numpy_to_vtk(
            connectivity, deep=True, array_type=vtk.VTK_ID_TYPE
        )

        cell_array = vtk.vtkCellArray()
        cell_array.SetData(vtk_offsets, vtk_conn)

        # Assemble PolyData
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(cell_array)

        # Colors
        # Handle three input formats:
        #   - tuple (r, g, b) float [0,1] → uniform color for all points
        #   - ndarray float32 [0,1]       → per-point, needs 255× conversion
        #   - ndarray uint8 [0,255]        → per-point, ready to use
        if isinstance(colors, tuple):
            color_arr = np.empty((total_points, 3), dtype=np.uint8)
            color_arr[:] = [int(c * 255) for c in colors]
        elif isinstance(colors, np.ndarray) and colors.dtype == np.uint8:
            color_arr = np.ascontiguousarray(colors)
        elif isinstance(colors, np.ndarray):
            color_arr = (np.ascontiguousarray(colors) * 255).astype(np.uint8)
        else:
            # Fallback: default grey
            color_arr = np.full((total_points, 3), 204, dtype=np.uint8)

        vtk_colors = numpy_support.numpy_to_vtk(
            color_arr, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_colors.SetName("colors")
        poly_data.GetPointData().SetScalars(vtk_colors)

        return poly_data

    def update_main_streamlines_actor(self, force: bool = False) -> None:
        """
        Recreates the main actor using the skipped/strided dataset.
        Handles both Line and Tube visualization.

        Uses a rebuild guard to skip redundant rebuilds when no tracked
        state has changed (visibility, stride, color mode, scalar,
        tube mode, opacity).

        Args:
            force: If True, bypasses the rebuild guard and forces a full
                   actor rebuild regardless of cached state.
        """
        try:
            if not self.panel.scene:
                return

            # Rebuild guard: skip if nothing changed
            if not force and self.panel.main_window:
                mw = self.panel.main_window
                current_state = (
                    mw._visibility_version,
                    mw.render_stride,
                    mw.current_color_mode,
                    mw.active_scalar_name,
                    getattr(mw, "render_as_tubes", False),
                    getattr(mw, "bundle_opacity", 1.0),
                )
                cached_state = (
                    mw._last_visibility_version,
                    mw._last_render_stride,
                    mw._last_color_mode,
                    mw._last_active_scalar,
                    mw._last_tube_mode,
                    mw._last_bundle_opacity,
                )
                if current_state == cached_state and self.panel.streamlines_actor is not None:
                    logger.debug(
                        "Rebuild guard: skipping redundant actor rebuild "
                        "(visibility_version=%d, stride=%d)",
                        mw._visibility_version,
                        mw.render_stride,
                    )
                    return  # Nothing changed, skip rebuild

                # Update cache for next comparison
                (
                    mw._last_visibility_version,
                    mw._last_render_stride,
                    mw._last_color_mode,
                    mw._last_active_scalar,
                    mw._last_tube_mode,
                    mw._last_bundle_opacity,
                ) = current_state

            # Remove old actor
            if self.panel.streamlines_actor is not None:
                self.panel.scene.rm(self.panel.streamlines_actor)
                self.panel.streamlines_actor = None

            if not self.panel.main_window or not self.panel.main_window.tractogram_data:
                if self.panel.scalar_bar_actor:
                    self.panel.scalar_bar_actor.SetVisibility(0)
                self.panel.render_window.Render()
                return

            # Get Params (includes subsampled list and flat buffer metadata)
            params = self.get_streamline_actor_params()
            sl_list = params.get("streamlines_list", [])

            current_mode = self.panel.main_window.current_color_mode
            active_scalar = self.panel.main_window.active_scalar_name

            if current_mode == ColorMode.SCALAR and "lut" in params and active_scalar:
                self.update_scalar_bar(params["lut"], active_scalar)
            else:
                if self.panel.scalar_bar_actor:
                    self.panel.scalar_bar_actor.SetVisibility(0)

            if not sl_list:
                self.panel.render_window.Render()
                return

            # Determine rendering mode
            use_tubes = getattr(self.panel.main_window, "render_as_tubes", False)
            has_flat_buffer = "_flat_points" in params

            # Visual properties
            colors = params.get("colors", (0.8, 0.8, 0.8))
            opacity = params.get("opacity", 0.5)
            line_width = params.get("linewidth", 2)
            num_streamlines = len(sl_list)

            # Large dataset flag for tube radius only (line width stays
            # constant to avoid visual quality shifts when the rendered
            # count crosses a threshold during deletions/undo).
            LARGE_DATASET_THRESHOLD = 19000
            is_large_dataset = num_streamlines > LARGE_DATASET_THRESHOLD

            try:
                if has_flat_buffer and not use_tubes:
                    # ==========================================================
                    # FAST PATH: Direct VTK construction (bypasses FURY)
                    # ==========================================================
                    flat_points = params["_flat_points"]
                    lengths = params["_subset_lengths"]
                    dst_offsets = params["_dst_offsets"]
                    n_sl = params["_num_streamlines"]

                    poly_data = self._build_polydata_direct(
                        flat_points, lengths, dst_offsets, colors, n_sl
                    )

                    # Build mapper (matching FURY's internal pattern)
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(poly_data)
                    mapper.ScalarVisibilityOn()
                    mapper.SetScalarModeToUsePointFieldData()
                    mapper.SelectColorArray("colors")
                    mapper.Update()

                    # Build actor
                    self.panel.streamlines_actor = vtk.vtkActor()
                    self.panel.streamlines_actor.SetMapper(mapper)
                    self.panel.streamlines_actor.GetProperty().SetLineWidth(
                        line_width
                    )
                    self.panel.streamlines_actor.GetProperty().SetOpacity(opacity)

                elif use_tubes:
                    tube_radius = 0.15 if not is_large_dataset else 0.10

                    if has_flat_buffer:
                        # ======================================================
                        # FAST PATH: Direct VTK tube construction (bypasses FURY)
                        # ======================================================
                        flat_points = params["_flat_points"]
                        lengths = params["_subset_lengths"]
                        dst_offsets = params["_dst_offsets"]
                        n_sl = params["_num_streamlines"]

                        # Build line polydata using existing vectorized builder
                        poly_data = self._build_polydata_direct(
                            flat_points, lengths, dst_offsets, colors, n_sl
                        )

                        # Compute normals (matches FURY's streamtube pipeline)
                        normals = vtk.vtkPolyDataNormals()
                        normals.SetInputData(poly_data)
                        normals.ComputeCellNormalsOn()
                        normals.ComputePointNormalsOn()
                        normals.ConsistencyOn()
                        normals.AutoOrientNormalsOn()
                        normals.Update()

                        # Apply tube filter (C++ geometry generation)
                        tube_filter = vtk.vtkTubeFilter()
                        tube_filter.SetInputConnection(
                            normals.GetOutputPort()
                        )
                        tube_filter.SetNumberOfSides(9)
                        tube_filter.SetRadius(tube_radius)
                        tube_filter.CappingOn()
                        tube_filter.Update()

                        # Build mapper
                        mapper = vtk.vtkPolyDataMapper()
                        mapper.SetInputConnection(
                            tube_filter.GetOutputPort()
                        )
                        mapper.ScalarVisibilityOn()
                        mapper.SetScalarModeToUsePointFieldData()
                        mapper.SelectColorArray("colors")
                        mapper.Update()

                        # Build actor
                        self.panel.streamlines_actor = vtk.vtkActor()
                        self.panel.streamlines_actor.SetMapper(mapper)
                        prop = self.panel.streamlines_actor.GetProperty()
                        prop.SetInterpolationToPhong()
                        prop.BackfaceCullingOn()
                        prop.SetOpacity(opacity)

                    else:
                        # ======================================================
                        # SLOW PATH: FURY fallback (non-ArraySequence data)
                        # ======================================================
                        self.panel.streamlines_actor = actor.streamtube(
                            sl_list,
                            colors=colors,
                            opacity=opacity,
                            linewidth=tube_radius,
                            lod=False,
                        )
                else:
                    # ==========================================================
                    # SLOW PATH: Line rendering via FURY (fallback)
                    # ==========================================================
                    self.panel.streamlines_actor = actor.line(
                        sl_list,
                        colors=colors,
                        opacity=opacity,
                        linewidth=line_width,
                        lod=False,
                    )
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.error(
                    f"Error creating streamline actor (Tubes={use_tubes}): {e}"
                )
                self.panel.update_status(f"Error rendering streamlines: {e}")
                return

            self.panel.scene.add(self.panel.streamlines_actor)

            # Apply visibility toggle
            if not self.panel.main_window.bundle_is_visible:
                self.panel.streamlines_actor.SetVisibility(0)

            self.update_highlight()
            self.panel.render_window.Render()

        except (RuntimeError, ValueError, IndexError, AttributeError, TypeError) as e:
            logger.error(f"Error in update_main_streamlines_actor: {e}", exc_info=True)
            self.panel.update_status("Error updating visualization.")
