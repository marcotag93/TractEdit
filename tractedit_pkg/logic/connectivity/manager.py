"""Connectivity data ownership, endpoint caches and region logic."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Dict, Tuple, Any, Iterable

import numpy as np
import nibabel as nib
from PyQt6.QtCore import Qt

from .. import connectivity
from ...render_batch import batch_render
from ...file_dialogs import choose_input
from ...reference_grid import mask_spatial_metadata, validate_volume_geometry
from ...utils import signals_blocked
from .export import ConnectivityExportMixin
from .overlay import ConnectivityOverlayMixin

if TYPE_CHECKING:
    from contextlib import ExitStack
    from ...main_window import MainWindow

logger = logging.getLogger(__name__)


_ATLAS_READ_BLOCK_BYTES = 4 * 1024 * 1024


def _has_valid_endpoints(streamline: Any) -> bool:
    if streamline is None:
        return False
    coordinates = np.asarray(streamline)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) < 2:
        return False
    if not np.all(np.isfinite(coordinates[[0, -1]])):
        return False
    return True

def _read_atlas_c_int32(dataobj) -> np.ndarray:
    """Read a file proxy sequentially into the native kernel's final layout."""
    if not isinstance(dataobj, nib.arrayproxy.ArrayProxy) or len(dataobj.shape) != 3:
        return np.asarray(dataobj, dtype=np.int32, order="C")
    from nibabel.openers import ImageOpener

    shape = dataobj.shape
    axis = 2 if dataobj.order == "F" else 0
    plane_size = int(np.prod(shape)) // shape[axis]
    # Scaling can promote stored integers to float64. A single plane is the
    # minimum sequential unit, even if an unusually large plane exceeds budget.
    width = max(
        1, connectivity._ATLAS_READ_BLOCK_BYTES // (plane_size * max(8, dataobj.dtype.itemsize))
    )
    result = np.empty(shape, dtype=np.int32, order="C")
    spec = (shape, dataobj.dtype, dataobj.offset, dataobj.slope, dataobj.inter)
    # One explicitly scoped stream avoids repeated decompression for .nii.gz.
    with ImageOpener(dataobj.file_like) as stream:
        proxy = nib.arrayproxy.ArrayProxy(stream, spec, mmap=False, order=dataobj.order)
        for start in range(0, shape[axis], width):
            slices = [slice(None)] * 3
            slices[axis] = slice(start, min(start + width, shape[axis]))
            slices = tuple(slices)
            result[slices] = proxy[slices]
    return result


class ConnectivityManager(ConnectivityExportMixin, ConnectivityOverlayMixin):
    """
    Manages connectivity matrix computation for TractEdit.

    This class handles:
    - Loading FreeSurfer parcellation files
    - Computing endpoint-based structural connectivity matrices
    - Exporting matrices in various formats (CSV, NPY)
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the connectivity manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def _invert_parcellation_affine(self) -> Optional[np.ndarray]:
        """
        Safely computes the inverse of the parcellation affine matrix.

        Returns:
            The inverse affine matrix, or None if the matrix is singular.
        """
        try:
            affine = np.asarray(self.mw.parcellation_affine, dtype=np.float64)
            if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
                raise np.linalg.LinAlgError
            inverse = np.linalg.inv(affine)
            if not np.all(np.isfinite(inverse)):
                raise np.linalg.LinAlgError
            return inverse
        except (TypeError, ValueError, np.linalg.LinAlgError):
            logger.error(
                "Parcellation affine matrix is singular, cannot compute inverse."
            )
            connectivity.QMessageBox.critical(
                self.mw,
                "Error",
                "Invalid parcellation affine matrix (singular). "
                "The parcellation file may be corrupted.",
            )
            return None

    def _endpoint_cache_owner(self) -> Tuple[int, int, int, bytes, int, int, int]:
        """Own labels by data versions/identities and the affine's values.

        Volume and streamline edits require their data version to advance.
        The small 4x4 affine is snapshotted directly to detect in-place edits
        without scanning or copying either large dataset.
        """
        mw = self.mw
        tractogram = mw.tractogram_data
        return (
            getattr(mw, "_parcellation_data_version", 0),
            id(mw.parcellation_data),
            id(mw.parcellation_affine),
            np.asarray(mw.parcellation_affine, dtype=np.float64).tobytes(),
            getattr(mw, "_tractogram_data_version", 0),
            id(tractogram),
            len(tractogram),
        )

    def _reset_endpoint_cache(self) -> None:
        """Clear all endpoint-derived parcellation state."""
        mw = self.mw
        mw.parcellation_region_intersection_cache = {}
        mw.parcellation_start_labels = None
        mw.parcellation_end_labels = None
        mw.parcellation_visible_indices = None
        mw._parcellation_endpoint_labels_computed = None
        mw._parcellation_endpoint_cache_key = None

    def _ensure_endpoint_labels(self, indices: Iterable[int]) -> Optional[np.ndarray]:
        """Compute labels only for uncached streamline indices."""
        mw = self.mw
        if mw.parcellation_data is None or mw.tractogram_data is None:
            return None

        owner = self._endpoint_cache_owner()
        if getattr(mw, "_parcellation_endpoint_cache_key", None) != owner:
            self._reset_endpoint_cache()

        if isinstance(indices, np.ndarray):
            requested = np.asarray(indices, dtype=np.int64).reshape(-1)
        else:
            requested = np.fromiter(indices, dtype=np.int64)

        n_streamlines = len(mw.tractogram_data)
        if requested.size and (
            np.any(requested < 0) or np.any(requested >= n_streamlines)
        ):
            raise IndexError("Streamline index outside tractogram bounds.")

        computed = getattr(mw, "_parcellation_endpoint_labels_computed", None)
        if computed is None or computed.shape != (n_streamlines,):
            mw.parcellation_start_labels = np.empty(0, dtype=np.int32)
            mw.parcellation_end_labels = np.empty(0, dtype=np.int32)
            computed = np.zeros(n_streamlines, dtype=np.uint8)
            mw._parcellation_endpoint_labels_computed = computed
            mw.parcellation_visible_indices = np.empty(0, dtype=np.int64)

        missing = requested[computed[requested] == 0]
        if missing.size:
            valid_positions = []
            start_points = []
            end_points = []
            for position, streamline_index in enumerate(missing):
                streamline = mw.tractogram_data[int(streamline_index)]
                if connectivity._has_valid_endpoints(streamline):
                    valid_positions.append(position)
                    start_points.append(streamline[0])
                    end_points.append(streamline[-1])

            inv_affine = self._invert_parcellation_affine()
            if inv_affine is None:
                return None
            start_labels = np.zeros(len(missing), dtype=np.int32)
            end_labels = np.zeros(len(missing), dtype=np.int32)
            if valid_positions:
                valid_start, valid_end = connectivity._compute_endpoint_labels(
                    np.ascontiguousarray(start_points, dtype=np.float64),
                    np.ascontiguousarray(end_points, dtype=np.float64),
                    np.ascontiguousarray(inv_affine[:3, :3], dtype=np.float64),
                    np.ascontiguousarray(inv_affine[:3, 3], dtype=np.float64),
                    mw.parcellation_data,
                    np.array(mw.parcellation_data.shape, dtype=np.int64),
                )
                start_labels[valid_positions] = valid_start
                end_labels[valid_positions] = valid_end
            invalid_count = len(missing) - len(valid_positions)
            if invalid_count:
                logger.warning(
                    "Ignored %d degenerate streamline(s) during endpoint labeling.",
                    invalid_count,
                )
            mw.parcellation_start_labels = np.concatenate(
                (mw.parcellation_start_labels, start_labels)
            )
            mw.parcellation_end_labels = np.concatenate(
                (mw.parcellation_end_labels, end_labels)
            )
            # Distinguish degenerate fibers from valid unlabeled endpoints.
            computed[missing] = 2
            computed[missing[valid_positions]] = 1
            mw.parcellation_visible_indices = np.concatenate(
                (mw.parcellation_visible_indices, missing)
            )

        mw._parcellation_endpoint_cache_key = owner
        return missing

    def ensure_region_intersections(
        self,
        labels: Iterable[int],
        indices: Iterable[int],
    ) -> bool:
        """Ensure region caches cover the requested streamline indices."""
        mw = self.mw
        try:
            newly_computed = self._ensure_endpoint_labels(indices)
            if newly_computed is None:
                return False

            requested_labels = {int(label) for label in labels}
            cached_labels = set(mw.parcellation_region_intersection_cache)
            computed_indices = mw.parcellation_visible_indices
            start_labels = mw.parcellation_start_labels
            end_labels = mw.parcellation_end_labels

            if newly_computed.size:
                new_start_labels = start_labels[-len(newly_computed) :]
                new_end_labels = end_labels[-len(newly_computed) :]
                for label in cached_labels:
                    matches = (new_start_labels == label) | (new_end_labels == label)
                    mw.parcellation_region_intersection_cache[label].update(
                        map(int, newly_computed[matches])
                    )

            for label in requested_labels - cached_labels:
                matches = (start_labels == label) | (end_labels == label)
                mw.parcellation_region_intersection_cache[label] = set(
                    map(int, computed_indices[matches])
                )
            return True
        except (ValueError, IndexError, TypeError) as error:
            logger.error("Error computing parcellation intersections: %s", error)
            return False

    @batch_render(lambda self: self.mw.vtk_panel)
    def load_parcellation(self, file_path: Optional[str] = None) -> bool:
        """
        Loads a FreeSurfer parcellation/segmentation NIfTI file.

        Args:
            file_path: Optional path to parcellation file. If None, opens dialog.

        Returns:
            True if loading was successful, False otherwise.
        """
        mw = self.mw

        # Get file path via dialog if not provided
        if not file_path:
            file_path, _ = choose_input(
                connectivity.QFileDialog.getOpenFileName,
                mw,
                "Load FreeSurfer Parcellation",
                "",
                "NIfTI Files (*.nii *.nii.gz);;All Files (*)",
            )

        if not file_path:
            return False

        try:
            mw.vtk_panel.update_progress_status("Loading parcellation...")
            connectivity.QApplication.processEvents()

            # Load NIfTI
            img = nib.load(file_path)
            try:
                validate_volume_geometry(img.dataobj, img.affine, "Parcellation")
            except ValueError as error:
                connectivity.QMessageBox.warning(mw, "Parcellation Error", str(error))
                return False
            # The compiled endpoint kernel requires C-contiguous int32 data.
            data = connectivity._read_atlas_c_int32(img.dataobj)
            affine = img.affine.astype(np.float64)
            spatial = mask_spatial_metadata(img)

            previous_region_states = getattr(mw, "parcellation_region_states", {})
            had_active_region_filters = any(
                state.get("include") or state.get("exclude")
                for state in previous_region_states.values()
            )
            self.invalidate_parcellation_cache(
                clear_region_states=True,
                clear_region_visibility=True,
            )
            mw._parcellation_data_version = (
                getattr(mw, "_parcellation_data_version", 0) + 1
            )

            # Store parcellation data
            mw.parcellation_data = data
            mw.parcellation_affine = affine
            mw.parcellation_spatial = spatial
            mw.parcellation_path = file_path

            # Try to load label names from FreeSurfer LUT
            mw.parcellation_labels = connectivity.parse_freesurfer_lut()

            # Add any labels found in parcellation but not in LUT
            unique_labels = np.unique(data)
            mw._parcellation_label_count = np.count_nonzero(unique_labels)
            for label in unique_labels:
                if label not in mw.parcellation_labels:
                    mw.parcellation_labels[label] = f"Region_{label}"

            roi_manager = vars(mw).get("roi_manager")
            if roi_manager is not None and had_active_region_filters:
                roi_manager.apply_logic_filters()

            # Update UI
            n_regions = len(unique_labels)
            mw.vtk_panel.update_status(
                f"Parcellation loaded: {os.path.basename(file_path)} "
                f"({n_regions} regions)"
            )
            mw._update_action_states()
            mw._update_bundle_info_display()
            mw._update_data_panel_display()

            logger.info(f"Loaded parcellation: {file_path} with {n_regions} regions")
            return True

        except (OSError, ValueError) as e:
            logger.error(f"Error loading parcellation: {e}", exc_info=True)
            connectivity.QMessageBox.critical(
                mw, "Load Error", f"Could not load parcellation file:\n{e}"
            )
            return False

    def _reset_parcellation_state(self) -> None:
        """Clear parcellation data, caches, and menu toggle state."""
        mw = self.mw

        self.invalidate_parcellation_cache(
            clear_region_states=True,
            clear_region_visibility=True,
        )
        mw._parcellation_data_version = getattr(mw, "_parcellation_data_version", 0) + 1

        # Clear parcellation data
        mw.parcellation_data = None
        mw._parcellation_label_count = None
        mw.parcellation_affine = None
        mw.parcellation_spatial = None
        mw.parcellation_path = None
        mw.parcellation_labels = {}
        mw.parcellation_connected_labels = set()
        mw.parcellation_region_visibility = {}
        mw.parcellation_main_labels = set()
        mw.parcellation_label_colors = {}

        # Clear cache flags and region filter data
        mw._parcellation_overlay_cached = False
        mw._parcellation_overlay_visible = False
        mw.parcellation_region_states = {}
        mw.parcellation_region_intersection_cache = {}
        mw.parcellation_start_labels = None
        mw.parcellation_end_labels = None
        mw.parcellation_visible_indices = None

        # Update menu action state
        if getattr(mw, "view_parcellation_action", None) is not None:
            mw.view_parcellation_action.setChecked(False)

    @batch_render(lambda self: self.mw.vtk_panel)
    def remove_parcellation(self) -> None:
        """
        Completely removes all parcellation data and frees the cache.

        Clears all parcellation-related data structures and updates the UI.
        """
        mw = self.mw

        self._reset_parcellation_state()

        # Re-apply logic filters with skip protection against RAM exhaustion
        mw.roi_manager._apply_filters_with_skip_protection()

        # Update UI
        mw._update_action_states()
        mw._update_data_panel_display()

        mw.vtk_panel.update_status("Parcellation removed and cache cleared.")
        logger.info("Parcellation removed and cache cleared")

    def compute_connectivity_matrix(self) -> Optional[Dict[str, Any]]:
        """
        Computes the structural connectivity matrix from visible streamlines.

        Uses streamline endpoints to determine connections between parcellation
        regions. The matrix is symmetric and includes self-connections.

        Returns:
            Dictionary with 'matrix', 'labels', 'label_names', 'counts' or None on error.
        """
        mw = self.mw

        # Validate data
        if mw.tractogram_data is None:
            connectivity.QMessageBox.warning(
                mw, "No Streamlines", "Please load a streamline bundle first."
            )
            return None

        if mw.parcellation_data is None:
            connectivity.QMessageBox.warning(
                mw, "No Parcellation", "Please load a parcellation file first."
            )
            return None

        if not mw.visible_indices:
            connectivity.QMessageBox.warning(
                mw, "No Visible Streamlines", "No streamlines are currently visible."
            )
            return None

        try:
            # Setup progress dialog
            progress = connectivity.QProgressDialog(
                "Computing connectivity matrix...", "Cancel", 0, 100, mw
            )
            progress.setWindowTitle("Connectivity Matrix")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            connectivity.QApplication.processEvents()

            # Extract visible streamlines endpoints
            visible_indices = np.fromiter(mw.visible_indices, dtype=np.int64)
            progress.setLabelText("Computing endpoint labels...")
            progress.setValue(10)
            connectivity.QApplication.processEvents()
            if progress.wasCanceled():
                return None
            if self._ensure_endpoint_labels(visible_indices) is None:
                return None
            rows = np.isin(mw.parcellation_visible_indices, visible_indices, assume_unique=True)
            start_labels = mw.parcellation_start_labels[rows]
            end_labels = mw.parcellation_end_labels[rows]
            n_streamlines = int(np.count_nonzero(
                mw._parcellation_endpoint_labels_computed[visible_indices] == 1
            ))

            progress.setLabelText("Building connectivity matrix...")
            progress.setValue(70)
            connectivity.QApplication.processEvents()

            if progress.wasCanceled():
                return None

            # Get unique labels (excluding 0 = Unknown unless specifically needed)
            all_labels = np.union1d(start_labels, end_labels)
            nonzero_labels = all_labels[all_labels > 0]

            if len(nonzero_labels) == 0:
                connectivity.QMessageBox.warning(
                    mw,
                    "No Connections",
                    "No streamline endpoints fall within labeled regions.",
                )
                return None

            # Count each pair symmetrically and each diagonal connection once.
            n_labels = len(nonzero_labels)
            matrix = np.zeros((n_labels, n_labels), dtype=np.int32)
            paired = (start_labels > 0) & (end_labels > 0)
            starts = np.searchsorted(nonzero_labels, start_labels[paired])
            ends = np.searchsorted(nonzero_labels, end_labels[paired])
            np.add.at(matrix, (starts, ends), 1)
            off_diagonal = starts != ends
            np.add.at(matrix, (ends[off_diagonal], starts[off_diagonal]), 1)
            progress.setLabelText("Finalizing...")
            progress.setValue(90)
            connectivity.QApplication.processEvents()

            # Get label names
            label_names = [
                mw.parcellation_labels.get(int(label), f"Region_{label}")
                for label in nonzero_labels
            ]

            # Compute connection counts per region
            connection_counts = np.sum(matrix, axis=1)

            progress.setValue(100)
            progress.close()

            result = {
                "matrix": matrix,
                "labels": nonzero_labels.tolist(),
                "label_names": label_names,
                "counts": connection_counts.tolist(),
                "n_streamlines": n_streamlines,
                "n_regions": n_labels,
            }

            logger.info(
                f"Computed connectivity matrix: {n_labels}x{n_labels} from {n_streamlines} streamlines"
            )

            return result

        except (ValueError, IndexError, TypeError) as e:
            logger.error(f"Error computing connectivity matrix: {e}", exc_info=True)
            connectivity.QMessageBox.critical(
                mw, "Computation Error", f"Failed to compute connectivity matrix:\n{e}"
            )
            return None

    def stage_bundle_replacement(self, rollback: ExitStack) -> None:
        mw = self.mw
        names = (
            "_session_deferred_regions", "_parcellation_overlay_visible",
            "_parcellation_overlay_cached", "_parcellation_overlay_cache_key",
            "_parcellation_endpoint_cache_key", "_parcellation_endpoint_labels_computed",
            "parcellation_overlay_actor", "parcellation_region_actors",
            "parcellation_connected_labels", "parcellation_main_labels",
            "parcellation_label_colors", "parcellation_region_states",
            "parcellation_start_labels", "parcellation_end_labels",
            "parcellation_visible_indices", "parcellation_region_intersection_cache",
        )
        previous = {name: getattr(mw, name, None) for name in names}
        scene = mw.vtk_panel.scene
        attached = [
            actor for actor in (
                mw.parcellation_overlay_actor, *mw.parcellation_region_actors.values()
            )
            if actor is not None and scene.HasViewProp(actor)
        ]

        def restore():
            self._invalidate_overlay_cache(render=False)
            for name, value in previous.items():
                setattr(mw, name, value)
            for actor in attached:
                scene.add(actor)
            mw.vtk_panel.update_progress_bar(0, 0, visible=False)

        rollback.callback(restore)
        self.invalidate_parcellation_cache(clear_region_states=True, render=False)
        if previous["_parcellation_overlay_visible"]:
            if not self.create_parcellation_overlay(interactive=False):
                raise RuntimeError("Could not rebuild the parcellation overlay.")

    def invalidate_parcellation_cache(
        self,
        clear_region_states: bool = False,
        clear_region_visibility: bool = False,
        render: bool = True,
    ) -> None:
        """Clear caches owned by the current atlas and tractogram."""
        mw = self.mw
        self._invalidate_overlay_cache(render=render)
        self._reset_endpoint_cache()
        if clear_region_states:
            mw.parcellation_region_states = {}
        if clear_region_visibility:
            mw.parcellation_region_visibility = {}

    def compute_region_intersection(self, label: int) -> bool:
        """
        Computes which streamlines pass through a parcellation region.

        Extends the endpoint cache for currently visible streamlines as needed.

        Args:
            label: The parcellation region label ID.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None or mw.tractogram_data is None:
            return False

        indices = getattr(mw, "visible_indices", range(len(mw.tractogram_data)))
        if self.ensure_region_intersections([label], indices):
            region_name = mw.parcellation_labels.get(label, f"Region_{label}")
            logger.info(
                "Region %s: %d streamlines",
                region_name,
                len(mw.parcellation_region_intersection_cache[label]),
            )
            return True
        return False

    @batch_render(lambda self: self.mw.vtk_panel)
    def recalculate_all_intersections(self) -> bool:
        """
        Recalculates all parcellation intersections based on current visible streamlines.

        This clears the endpoint labels cache, recomputes them from current visible
        streamlines, clears the intersection cache, and re-applies any active filters.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None or mw.tractogram_data is None:
            mw.vtk_panel.update_status("No parcellation or streamlines loaded")
            return False

        if not mw.visible_indices:
            mw.vtk_panel.update_status("No visible streamlines for intersection")
            return False

        try:
            mw.vtk_panel.update_progress_status("Recalculating parcellation intersections...")
            connectivity.QApplication.processEvents()

            self.invalidate_parcellation_cache(clear_region_states=True)
            mw.roi_manager.apply_logic_filters()

            if not self.create_parcellation_overlay():
                return False

            connected_labels = mw.parcellation_connected_labels
            n_streamlines = len(mw.visible_indices)

            mw.vtk_panel.update_status(
                f"Parcellation intersections recalculated ({len(connected_labels)} regions, "
                f"{n_streamlines} streamlines)"
            )

            # Update data panel
            mw._update_data_panel_display()

            logger.info(
                f"Recalculated parcellation intersections: {len(connected_labels)} regions, "
                f"{n_streamlines} streamlines"
            )
            return True

        except (ValueError, IndexError, TypeError, RuntimeError) as e:
            logger.error(f"Error recalculating intersections: {e}", exc_info=True)
            mw.vtk_panel.update_status(f"Error recalculating intersections: {e}")
            return False

    @batch_render(lambda self: self.mw.vtk_panel)
    def set_region_logic_mode(self, label: int, mode: str) -> None:
        """
        Sets the logic mode for a parcellation region.

        Modes: 'none', 'include', 'exclude'
        - Include: Applies to THIS specific region (cumulative with others)
        - Exclude: Applies to THIS specific region (can switch from include)
        - None: Clears filter for THIS specific region only

        Args:
            label: The parcellation region label ID.
            mode: The logic mode to set.
        """
        mw = self.mw

        # Only allow filter operations when parcellation overlay is visible
        overlay_visible = getattr(mw, "_parcellation_overlay_visible", False)
        if not overlay_visible:
            mw.vtk_panel.update_status(
                "Enable parcellation overlay first to use region filters"
            )
            return

        # Initialize state dict if not exists
        if not hasattr(mw, "parcellation_region_states"):
            mw.parcellation_region_states = {}

        # Initialize intersection cache if not exists
        if not hasattr(mw, "parcellation_region_intersection_cache"):
            mw.parcellation_region_intersection_cache = {}

        region_name = mw.parcellation_labels.get(label, f"Region_{label}")

        if mode == "none":
            # Clear filter for THIS specific region only
            if label in mw.parcellation_region_states:
                del mw.parcellation_region_states[label]
            mw.vtk_panel.update_status(f"{region_name}: filter removed")

        elif mode == "include":
            if label not in mw.parcellation_region_states:
                mw.parcellation_region_states[label] = {
                    "include": False,
                    "exclude": False,
                }

            # Set this region to include (clear exclude if set)
            mw.parcellation_region_states[label]["include"] = True
            mw.parcellation_region_states[label]["exclude"] = False

            # Compute intersection if not cached
            if label not in mw.parcellation_region_intersection_cache:
                success = self.compute_region_intersection(label)
                if not success:
                    mw.parcellation_region_states[label]["include"] = False
                    return

            mw.vtk_panel.update_status(f"{region_name}: set as INCLUDE")

        elif mode == "exclude":
            if label not in mw.parcellation_region_states:
                mw.parcellation_region_states[label] = {
                    "include": False,
                    "exclude": False,
                }

            # Set this region to exclude (clear include if set)
            mw.parcellation_region_states[label]["include"] = False
            mw.parcellation_region_states[label]["exclude"] = True

            # Compute intersection if not cached
            if label not in mw.parcellation_region_intersection_cache:
                success = self.compute_region_intersection(label)
                if not success:
                    mw.parcellation_region_states[label]["exclude"] = False
                    return

            mw.vtk_panel.update_status(f"{region_name}: set as EXCLUDE")

        # Apply filters with skip protection against RAM exhaustion
        mw.roi_manager._apply_filters_with_skip_protection()

        # Update data panel with signals blocked to prevent cascading callbacks
        if mw.data_tree_widget:
            with signals_blocked(mw.data_tree_widget):
                mw._update_data_panel_display()
