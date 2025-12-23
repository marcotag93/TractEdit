# -*- coding: utf-8 -*-

"""
ROI Manager for TractEdit application.

Handles ROI-related operations including:
- Computing ROI-streamline intersections
- Fast sphere and rectangle intersection updates
- Applying logic filters (include/exclude)
- ROI actions (rename, color change, save, remove)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Set, Optional, List

import numpy as np
import nibabel as nib
from numba import njit, prange
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QInputDialog,
    QFileDialog,
    QMessageBox,
)

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


@njit(nogil=True)
def _check_streamline_roi_intersection(
    streamline: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    roi_data: np.ndarray,
    dims: np.ndarray,
) -> bool:
    """
    Numba-optimized check if a single streamline intersects with ROI.

    Args:
        streamline: (N, 3) array of streamline points in world space.
        R: (3, 3) rotation part of inverse affine.
        T: (3,) translation part of inverse affine.
        roi_data: 3D ROI volume.
        dims: (3,) array of volume dimensions.

    Returns:
        True if streamline intersects ROI.
    """
    n_pts = streamline.shape[0]

    for i in range(n_pts):
        # Apply inverse affine: vox = point @ R.T + T
        vx = (
            streamline[i, 0] * R[0, 0]
            + streamline[i, 1] * R[1, 0]
            + streamline[i, 2] * R[2, 0]
            + T[0]
        )
        vy = (
            streamline[i, 0] * R[0, 1]
            + streamline[i, 1] * R[1, 1]
            + streamline[i, 2] * R[2, 1]
            + T[1]
        )
        vz = (
            streamline[i, 0] * R[0, 2]
            + streamline[i, 1] * R[1, 2]
            + streamline[i, 2] * R[2, 2]
            + T[2]
        )

        # Round to nearest integer
        ix = int(np.round(vx))
        iy = int(np.round(vy))
        iz = int(np.round(vz))

        # Bounds check
        if (
            ix >= 0
            and ix < dims[0]
            and iy >= 0
            and iy < dims[1]
            and iz >= 0
            and iz < dims[2]
        ):
            if roi_data[ix, iy, iz] > 0:
                return True

    return False


class ROIManager:
    """
    Manages ROI operations and streamline intersection logic.

    This class handles:
    - Computing ROI-streamline intersections (broad and narrow phase)
    - Fast updates for sphere and rectangle ROIs during interaction
    - Logic mode management (select, include, exclude)
    - Applying filters based on ROI logic modes
    - ROI actions (color change, rename, save, remove)
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the ROI manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def compute_roi_intersection(self, roi_path: str) -> bool:
        """
        Computes intersections using a Broad Phase (Bounding Box) filter
        followed by a Narrow Phase (Voxel Grid) check.
        """
        mw = self.mw

        if not mw.tractogram_data or roi_path not in mw.roi_layers:
            return False

        # Ensure we have bounding boxes
        if mw.streamline_bboxes is None:
            mw.streamline_bboxes = np.array(
                [[np.min(sl, axis=0), np.max(sl, axis=0)] for sl in mw.tractogram_data]
            )

        total_fibers = len(mw.tractogram_data)
        mw.vtk_panel.update_status(
            f"Computing intersection: {os.path.basename(roi_path)}..."
        )
        mw.vtk_panel.update_progress_bar(0, total_fibers, visible=True)
        QApplication.processEvents()

        try:
            roi_data = mw.roi_layers[roi_path]["data"]
            roi_affine = mw.roi_layers[roi_path]["affine"]
            inv_affine = mw.roi_layers[roi_path]["inv_affine"]
            dims = roi_data.shape

            # BROAD PHASE: Bounding Box Filter
            roi_indices = np.argwhere(roi_data > 0)

            if roi_indices.size == 0:
                mw.roi_intersection_cache[roi_path] = set()
                mw.vtk_panel.update_status(f"ROI is empty. Found 0.")
                return True

            v_min = np.min(roi_indices, axis=0)
            v_max = np.max(roi_indices, axis=0) + 1

            # Create the 8 corners of the ROI BBox in Voxel Space
            corners_vox = np.array(
                [
                    [v_min[0], v_min[1], v_min[2]],
                    [v_min[0], v_min[1], v_max[2]],
                    [v_min[0], v_max[1], v_min[2]],
                    [v_min[0], v_max[1], v_max[2]],
                    [v_max[0], v_min[1], v_min[2]],
                    [v_max[0], v_min[1], v_max[2]],
                    [v_max[0], v_max[1], v_min[2]],
                    [v_max[0], v_max[1], v_max[2]],
                ]
            )

            # Transform ROI Voxel Corners -> World Space to get World AABB
            corners_world = nib.affines.apply_affine(roi_affine, corners_vox)
            roi_world_min = np.min(corners_world, axis=0)
            roi_world_max = np.max(corners_world, axis=0)

            # Add small padding/tolerance
            tolerance = 2.0
            roi_world_min -= tolerance
            roi_world_max += tolerance

            overlap_mask = np.all(
                mw.streamline_bboxes[:, 1] >= roi_world_min, axis=1
            ) & np.all(mw.streamline_bboxes[:, 0] <= roi_world_max, axis=1)

            candidate_indices = np.where(overlap_mask)[0]

            # NARROW PHASE: Numba-optimized Voxel Grid Check
            intersecting: Set[int] = set()

            # Pre-fetch affine components for Numba (ensure contiguous float64)
            T = np.ascontiguousarray(inv_affine[:3, 3], dtype=np.float64)
            R = np.ascontiguousarray(inv_affine[:3, :3], dtype=np.float64)
            dims_arr = np.array(dims[:3], dtype=np.int64)

            # Ensure ROI data is contiguous for Numba
            roi_data_c = np.ascontiguousarray(roi_data)

            n_candidates = len(candidate_indices)
            for i, idx in enumerate(candidate_indices):
                if i % 500 == 0:
                    mw.vtk_panel.update_progress_bar(i, n_candidates, visible=True)
                    QApplication.processEvents()

                sl = mw.tractogram_data[idx]

                # Use Numba-optimized check
                sl_c = np.ascontiguousarray(sl, dtype=np.float64)
                if _check_streamline_roi_intersection(sl_c, R, T, roi_data_c, dims_arr):
                    intersecting.add(idx)

            mw.roi_intersection_cache[roi_path] = intersecting
            mw.vtk_panel.update_status(
                f"Intersection done. Found {len(intersecting)} "
                f"(Candidates: {len(candidate_indices)})."
            )
            return True

        except Exception as e:
            logger.warning(f"Intersection Error: {e}")
            mw.vtk_panel.update_status("Intersection failed.")
            return False

        finally:
            mw.vtk_panel.update_progress_bar(0, 0, visible=False)

    def update_sphere_roi_intersection(
        self, roi_name: str, center: np.ndarray, radius: float
    ) -> None:
        """
        Fast update of ROI intersection for spherical ROIs during interaction.
        Bypasses the slow voxel grid check and uses geometric distance check.
        """
        mw = self.mw

        if not mw.vtk_panel:
            return

        # Fast Geometric Check
        intersecting_indices = mw.vtk_panel._find_streamlines_in_radius(
            center, radius, check_all=True
        )

        # Update Cache
        mw.roi_intersection_cache[roi_name] = intersecting_indices

        # Apply Filters
        self.apply_logic_filters()

    def update_rectangle_roi_intersection(
        self, roi_name: str, min_point: np.ndarray, max_point: np.ndarray
    ) -> None:
        """
        Fast update of ROI intersection for rectangular ROIs during interaction.
        """
        mw = self.mw

        if not mw.vtk_panel:
            return

        # Fast Geometric Check
        intersecting_indices = mw.vtk_panel._find_streamlines_in_box(
            min_point, max_point, check_all=True
        )

        # Update Cache
        mw.roi_intersection_cache[roi_name] = intersecting_indices

        # Apply Filters
        self.apply_logic_filters()

    def set_roi_logic_mode(self, roi_path: str, mode: str) -> None:
        """
        Sets the logic mode for an ROI, ensuring mutual exclusivity.
        Modes: 'none', 'select', 'include', 'exclude'
        """
        mw = self.mw

        if roi_path not in mw.roi_states:
            return

        # Reset all flags
        for f in ["select", "include", "exclude"]:
            mw.roi_states[roi_path][f] = False

        # Set new flag (unless mode is 'none')
        if mode != "none":
            mw.roi_states[roi_path][mode] = True

            # Compute intersection if needed
            if roi_path not in mw.roi_intersection_cache:
                success = self.compute_roi_intersection(roi_path)
                if not success:
                    mw.roi_states[roi_path][mode] = False  # Revert on failure

        # Refresh Visuals
        self.update_roi_visual_selection()
        self.apply_logic_filters()

        # Refresh Panel Text (to show [TAG])
        mw._update_data_panel_display()

    def update_roi_visual_selection(self) -> None:
        """Updates the visual highlight for ROIs with 'select' mode active."""
        mw = self.mw

        active_selects = [p for p, s in mw.roi_states.items() if s["select"]]
        combined: Set[int] = set()
        for p in active_selects:
            combined.update(mw.roi_intersection_cache.get(p, set()))
        mw.roi_highlight_indices = combined
        if mw.vtk_panel:
            mw.vtk_panel.update_roi_highlight_actor()

    def apply_logic_filters(self) -> None:
        """Applies include/exclude logic filters to streamline visibility."""
        mw = self.mw

        if not hasattr(mw, "manual_visible_indices"):
            mw.manual_visible_indices = (
                set(range(len(mw.tractogram_data))) if mw.tractogram_data else set()
            )

        # Start with manual state
        final_indices = mw.manual_visible_indices.copy()

        # ========== ROI FILTERS ==========
        # Apply ROI Includes
        active_includes = [p for p, s in mw.roi_states.items() if s["include"]]
        for p in active_includes:
            roi_indices = mw.roi_intersection_cache.get(p, set())
            final_indices.intersection_update(roi_indices)

        # Apply ROI Excludes
        active_excludes = [p for p, s in mw.roi_states.items() if s["exclude"]]
        for p in active_excludes:
            excl = mw.roi_intersection_cache.get(p, set())
            final_indices.difference_update(excl)

        # ========== PARCELLATION REGION FILTERS ==========
        parc_states = getattr(mw, "parcellation_region_states", {})
        parc_cache = getattr(mw, "parcellation_region_intersection_cache", {})

        # Apply Parcellation Region Includes (cumulative - must pass through ALL)
        parc_includes = [l for l, s in parc_states.items() if s.get("include")]
        for label in parc_includes:
            region_indices = parc_cache.get(label, set())
            final_indices.intersection_update(region_indices)

        # Apply Parcellation Region Excludes
        parc_excludes = [l for l, s in parc_states.items() if s.get("exclude")]
        for label in parc_excludes:
            excl = parc_cache.get(label, set())
            final_indices.difference_update(excl)

        mw.visible_indices = final_indices

        # Handle empty result - show warning but allow recovery ##TODO - think this might go, it's not necessary anymore
        if not final_indices and mw.tractogram_data:
            logger.warning(
                "All streamlines filtered out. Remove filters to restore visibility."
            )
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    "No streamlines match current filters - remove filters to restore"
                )

        if mw.vtk_panel:
            mw.vtk_panel.update_main_streamlines_actor()
        mw._update_bundle_info_display()

    def change_roi_color_action(self, path: str) -> None:
        """Opens a color picker and updates the ROI layer color."""
        mw = self.mw

        color = QColorDialog.getColor()

        if color.isValid():
            rgb_normalized = (color.redF(), color.greenF(), color.blueF())

            if path in mw.roi_layers:
                mw.roi_layers[path]["color"] = rgb_normalized

            if mw.vtk_panel:
                mw.vtk_panel.set_roi_layer_color(path, rgb_normalized)
                mw.vtk_panel.update_status(
                    f"Updated color for {os.path.basename(path)}"
                )

            mw._update_data_panel_display()

    def rename_roi_action(self, old_path: str) -> None:
        """Renames an ROI layer."""
        mw = self.mw

        try:
            # Normalize path to ensure matching
            if old_path not in mw.roi_layers:
                norm_path = os.path.normpath(old_path)
                if norm_path in mw.roi_layers:
                    old_path = norm_path
                else:
                    logger.warning(f"ROI not found: {old_path}")
                    return

            current_name = os.path.basename(old_path)
            new_name, ok = QInputDialog.getText(
                mw, "Rename ROI", "Enter new name:", text=current_name
            )

            if not ok or not new_name or new_name == current_name:
                return

            # Create new path with new name
            old_dir = os.path.dirname(old_path) if os.path.dirname(old_path) else ""
            new_path = os.path.join(old_dir, new_name) if old_dir else new_name

            # Update all dictionaries
            # roi_layers - store display_name for custom naming
            layer_data = mw.roi_layers.pop(old_path)
            layer_data["display_name"] = new_name
            mw.roi_layers[new_path] = layer_data

            # roi_visibility
            if old_path in mw.roi_visibility:
                mw.roi_visibility[new_path] = mw.roi_visibility.pop(old_path)

            # roi_opacities
            if old_path in mw.roi_opacities:
                mw.roi_opacities[new_path] = mw.roi_opacities.pop(old_path)

            # roi_states
            if old_path in mw.roi_states:
                mw.roi_states[new_path] = mw.roi_states.pop(old_path)

            # roi_intersection_cache
            if old_path in mw.roi_intersection_cache:
                mw.roi_intersection_cache[new_path] = mw.roi_intersection_cache.pop(
                    old_path
                )

            # Update current_drawing_roi if it was the renamed ROI
            if mw.current_drawing_roi == old_path:
                mw.current_drawing_roi = new_path

            # Update VTK panel
            if mw.vtk_panel:
                if old_path in mw.vtk_panel.roi_slice_actors:
                    mw.vtk_panel.roi_slice_actors[new_path] = (
                        mw.vtk_panel.roi_slice_actors.pop(old_path)
                    )

                if old_path in mw.vtk_panel.sphere_params_per_roi:
                    mw.vtk_panel.sphere_params_per_roi[new_path] = (
                        mw.vtk_panel.sphere_params_per_roi.pop(old_path)
                    )

                if old_path in mw.vtk_panel.rectangle_params_per_roi:
                    mw.vtk_panel.rectangle_params_per_roi[new_path] = (
                        mw.vtk_panel.rectangle_params_per_roi.pop(old_path)
                    )

                mw.vtk_panel.update_status(f"Renamed: {current_name} -> {new_name}")

            mw._update_data_panel_display()
            mw._update_bundle_info_display()

        except Exception as e:
            logger.error(f"Error renaming ROI: {e}", exc_info=True)
            QMessageBox.warning(mw, "Rename Error", f"Failed to rename ROI: {e}")

    def save_roi_action(self, roi_path: str) -> None:
        """Saves the specified ROI to a NIfTI file."""
        mw = self.mw

        if roi_path not in mw.roi_layers:
            QMessageBox.warning(mw, "Save Error", "ROI not found.")
            return

        roi_layer = mw.roi_layers[roi_path]
        roi_data = roi_layer["data"]
        roi_affine = roi_layer["affine"]

        # Use display_name if available (set by rename), otherwise use path basename
        default_name = roi_layer.get("display_name", os.path.basename(roi_path))

        # Ensure proper NIfTI extension
        if not default_name.endswith((".nii", ".nii.gz")):
            default_name += ".nii.gz"

        save_path, _ = QFileDialog.getSaveFileName(
            mw, "Save ROI", default_name, "NIfTI Files (*.nii *.nii.gz)"
        )

        if not save_path:
            return

        try:
            nib.save(nib.Nifti1Image(roi_data, roi_affine), save_path)
            if mw.vtk_panel:
                mw.vtk_panel.update_status(f"Saved ROI to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving ROI: {e}", exc_info=True)
            QMessageBox.critical(mw, "Save Error", f"Failed to save ROI: {e}")

    def remove_roi_layer_action(self, path: str) -> None:
        """Removes a specific ROI layer."""
        mw = self.mw

        if path not in mw.roi_layers:
            return

        # Remove from VTK
        if mw.vtk_panel:
            mw.vtk_panel.remove_roi_layer(path)

        # Remove from data structures
        del mw.roi_layers[path]

        if path in mw.roi_visibility:
            del mw.roi_visibility[path]

        if path in mw.roi_opacities:
            del mw.roi_opacities[path]

        if path in mw.roi_states:
            del mw.roi_states[path]

        if path in mw.roi_intersection_cache:
            del mw.roi_intersection_cache[path]

        if mw.current_drawing_roi == path:
            mw.current_drawing_roi = None

        # Refresh
        self.apply_logic_filters()
        mw._update_data_panel_display()
        mw._update_bundle_info_display()

        if mw.vtk_panel:
            mw.vtk_panel.update_status(f"Removed ROI: {os.path.basename(path)}")
