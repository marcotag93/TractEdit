# -*- coding: utf-8 -*-

"""
ROI Manager for TractEdit application.

Handles ROI-related operations including:
- Computing ROI-streamline intersections
- Fast sphere and rectangle intersection updates
- Applying logic filters (include/exclude)
- ROI actions (rename, color change, save, remove)
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, Set

import numpy as np
import nibabel as nib
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QInputDialog,
    QFileDialog,
    QMessageBox,
)

from ..utils import AUTO_SKIP_THRESHOLD, TARGET_RENDER_COUNT
from ..transactional_io import transactional_save

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# AOT-compiled functions (imported from pre-compiled extension)
# ============================================================================


# _check_streamline_roi_intersection — AOT-compiled (see _numba_aot/build_aot.py)
from tractedit_pkg._numba_aot import (  # noqa: E402
    check_streamline_roi_intersection as _check_streamline_roi_intersection,
)


# ============================================================================
# ROI Manager Class
# ============================================================================


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

        parametric = self._compute_parametric_intersection(roi_path)
        if parametric is not None:
            mw.roi_intersection_cache[roi_path] = parametric
            mw.vtk_panel.update_status(f"Intersection done. Found {len(parametric)}.")
            return True

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

            # BROAD PHASE: Bounding Box Filter
            foreground = roi_data > 0
            roi_indices = np.argwhere(foreground)

            if roi_indices.size == 0:
                mw.roi_intersection_cache[roi_path] = set()
                mw.vtk_panel.update_status("ROI is empty. Found 0.")
                return True

            if roi_data.dtype == np.uint8:
                roi_data_c = np.ascontiguousarray(roi_data)
            else:
                roi_data_c = np.ascontiguousarray(foreground).view(np.uint8)
            del foreground

            # The native query rounds to the nearest voxel center. Include
            # the entire half-voxel cell before applying rotation/shear.
            v_min = np.min(roi_indices, axis=0) - 0.5
            v_max = np.max(roi_indices, axis=0) + 0.5

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

            # NARROW PHASE: AOT-optimized Voxel Grid Check
            intersecting: Set[int] = set()

            # Pre-fetch affine components for AOT kernel (ensure contiguous float64)
            T = np.ascontiguousarray(inv_affine[:3, 3], dtype=np.float64)
            # The compiled kernel multiplies world row vectors by R.
            R = np.ascontiguousarray(inv_affine[:3, :3].T, dtype=np.float64)
            dims_arr = np.array(roi_data.shape[:3], dtype=np.int64)

            n_candidates = len(candidate_indices)
            for i, idx in enumerate(candidate_indices):
                if i % 500 == 0:
                    mw.vtk_panel.update_progress_bar(i, n_candidates, visible=True)
                    QApplication.processEvents()

                sl = mw.tractogram_data[idx]

                # Use AOT-optimized check
                sl_c = np.ascontiguousarray(sl, dtype=np.float64)
                if _check_streamline_roi_intersection(sl_c, R, T, roi_data_c, dims_arr):
                    intersecting.add(idx)

            mw.roi_intersection_cache[roi_path] = intersecting
            mw.vtk_panel.update_status(
                f"Intersection done. Found {len(intersecting)} "
                f"(Candidates: {len(candidate_indices)})."
            )
            return True

        except (ValueError, IndexError, TypeError, RuntimeError) as e:
            logger.warning(f"Intersection Error: {e}")
            mw.vtk_panel.update_status("Intersection failed.")
            return False

        finally:
            mw.vtk_panel.update_progress_bar(0, 0, visible=False)

    def _compute_parametric_intersection(self, roi_name: str) -> Optional[Set[int]]:
        """Return live analytic ROI membership, or None for voxel-mask ROIs."""
        mw = self.mw
        panel = mw.vtk_panel
        if panel is None:
            return None

        sphere_registry = getattr(panel, "sphere_params_per_roi", None)
        sphere_params = (
            sphere_registry.get(roi_name)
            if isinstance(sphere_registry, Mapping)
            else None
        )
        if sphere_params:
            center = np.asarray(sphere_params.get("center"), dtype=np.float64)
            radius = float(sphere_params.get("radius", 0.0))
            if center.shape == (3,) and np.all(np.isfinite(center)) and radius > 0:
                return panel._find_streamlines_in_radius(center, radius, check_all=True)

        rectangle_registry = getattr(panel, "rectangle_params_per_roi", None)
        rectangle_params = (
            rectangle_registry.get(roi_name)
            if isinstance(rectangle_registry, Mapping)
            else None
        )
        if rectangle_params:
            voxel_min = rectangle_params.get("voxel_min")
            voxel_max = rectangle_params.get("voxel_max")
            if voxel_min is not None and voxel_max is not None:
                affine = mw.roi_layers[roi_name]["affine"]
                return panel._find_streamlines_in_oriented_box(
                    affine,
                    np.asarray(voxel_min, dtype=np.int64),
                    np.asarray(voxel_max, dtype=np.int64),
                    check_all=True,
                )

        return None

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
        self,
        roi_name: str,
        min_point: Optional[np.ndarray] = None,
        max_point: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fast update of ROI intersection for rectangular ROIs during interaction.
        """
        mw = self.mw

        if not mw.vtk_panel:
            return

        intersecting_indices = self._compute_parametric_intersection(roi_name)
        if intersecting_indices is None:
            if min_point is not None and max_point is not None:
                intersecting_indices = mw.vtk_panel._find_streamlines_in_box(
                    min_point, max_point, check_all=True
                )
            elif not self.compute_roi_intersection(roi_name):
                return
            else:
                intersecting_indices = mw.roi_intersection_cache[roi_name]

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
        self._apply_filters_with_skip_protection()

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

        # Collect active filters first to avoid unnecessary work
        active_includes = [p for p, s in mw.roi_states.items() if s["include"]]
        active_excludes = [p for p, s in mw.roi_states.items() if s["exclude"]]

        # Collect parcellation filters
        parc_states = getattr(mw, "parcellation_region_states", {})
        parc_cache = getattr(mw, "parcellation_region_intersection_cache", {})
        parc_includes = [
            label for label, state in parc_states.items() if state.get("include")
        ]
        parc_excludes = [
            label for label, state in parc_states.items() if state.get("exclude")
        ]
        connectivity_manager = vars(mw).get("connectivity_manager")

        # OPTIMIZATION: Only copy if we have filters to apply
        has_filters = (
            active_includes or active_excludes or parc_includes or parc_excludes
        )

        if has_filters:
            # Copy only when modification is needed
            final_indices = mw.manual_visible_indices.copy()

            # Apply ROI Includes
            for p in active_includes:
                roi_indices = mw.roi_intersection_cache.get(p, set())
                final_indices.intersection_update(roi_indices)

            # Apply ROI Excludes
            for p in active_excludes:
                excl = mw.roi_intersection_cache.get(p, set())
                final_indices.difference_update(excl)

            if (
                connectivity_manager is not None
                and (parc_includes or parc_excludes)
                and connectivity_manager.ensure_region_intersections(
                    parc_includes + parc_excludes,
                    final_indices,
                )
            ):
                parc_cache = mw.parcellation_region_intersection_cache

            # Apply Parcellation Region Includes
            for label in parc_includes:
                region_indices = parc_cache.get(label, set())
                final_indices.intersection_update(region_indices)

            # Apply Parcellation Region Excludes
            for label in parc_excludes:
                excl = parc_cache.get(label, set())
                final_indices.difference_update(excl)
        else:
            # No filters active, use manual state directly (no copy needed)
            final_indices = mw.manual_visible_indices

        mw.visible_indices = final_indices
        mw._visibility_version += 1

        # Invalidate visible array cache since visibility changed
        if mw.vtk_panel and hasattr(mw.vtk_panel, "selection_manager"):
            mw.vtk_panel.selection_manager.invalidate_visible_cache()

        # Handle empty result - show warning but allow recovery
        if not final_indices and mw.tractogram_data:
            logger.warning(
                "All streamlines filtered out. Remove filters to restore visibility."
            )
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    "No streamlines match current filters - remove filters to restore"
                )

        # Re-render with current stride (do NOT recalculate skip level).
        # The rebuild guard handles deduplication.
        if mw.vtk_panel:
            mw.vtk_panel.update_main_streamlines_actor()
        mw._update_bundle_info_display()

    def _apply_filters_with_skip_protection(self) -> None:
        """Applies logic filters with automatic skip/stride recalculation.

        For tractograms at or above AUTO_SKIP_THRESHOLD, the
        user's manual skip-disable override is cleared and a conservative
        pre-stride is applied before the filter pass to prevent a transient
        stride-1 render from exhausting RAM.
        """
        mw = self.mw

        if mw.tractogram_data is not None:
            try:
                total = len(mw.tractogram_data)
            except TypeError:
                total = 0

            if total >= AUTO_SKIP_THRESHOLD:
                mw._skip_user_disabled = False
                approx_visible = len(mw.manual_visible_indices)
                if approx_visible > TARGET_RENDER_COUNT:
                    mw.render_stride = max(1, approx_visible // TARGET_RENDER_COUNT)
                else:
                    mw.render_stride = 1

        self.apply_logic_filters()
        mw._auto_calculate_skip_level()

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
            if old_path not in mw.roi_layers:
                old_identity = os.path.normcase(
                    os.path.abspath(os.path.normpath(old_path))
                )
                matching_path = next(
                    (
                        path
                        for path in mw.roi_layers
                        if os.path.normcase(os.path.abspath(os.path.normpath(path)))
                        == old_identity
                    ),
                    None,
                )
                if matching_path is None:
                    logger.warning(f"ROI not found: {old_path}")
                    return
                old_path = matching_path

            current_name = os.path.basename(old_path)
            new_name, ok = QInputDialog.getText(
                mw, "Rename ROI", "Enter new name:", text=current_name
            )

            if not ok or not new_name.strip() or new_name == current_name:
                return

            old_dir = os.path.dirname(old_path) if os.path.dirname(old_path) else ""
            new_path = os.path.normpath(
                os.path.join(old_dir, new_name) if old_dir else new_name
            )

            new_identity = os.path.normcase(os.path.abspath(new_path))
            has_collision = any(
                path != old_path
                and os.path.normcase(os.path.abspath(os.path.normpath(path)))
                == new_identity
                for path in mw.roi_layers
            )
            if has_collision:
                QMessageBox.warning(
                    mw,
                    "Rename ROI",
                    f"An ROI named '{new_name}' already exists.",
                )
                return

            if new_path == old_path:
                return

            # History uses layer keys, so a destination retained by another
            # layer's history must not become an alias for this layer.
            for stack_name in ("unified_undo_stack", "unified_redo_stack"):
                for action in getattr(mw, stack_name, ()):
                    history_path = action.get("roi_name")
                    if (
                        history_path
                        and history_path != old_path
                        and os.path.normcase(os.path.abspath(os.path.normpath(history_path)))
                        == new_identity
                    ):
                        QMessageBox.warning(
                            mw,
                            "Rename ROI",
                            f"The name '{new_name}' is still referenced by another ROI's history.",
                        )
                        return

            layer_data = mw.roi_layers[old_path]
            layer_data["display_name"] = new_name

            mw.roi_layers[new_path] = layer_data
            mw.roi_layers.pop(old_path)

            if old_path in mw.roi_visibility:
                mw.roi_visibility[new_path] = mw.roi_visibility.pop(old_path)
            if old_path in mw.roi_opacities:
                mw.roi_opacities[new_path] = mw.roi_opacities.pop(old_path)
            if old_path in mw.roi_states:
                mw.roi_states[new_path] = mw.roi_states.pop(old_path)
            if old_path in mw.roi_intersection_cache:
                mw.roi_intersection_cache[new_path] = mw.roi_intersection_cache.pop(
                    old_path
                )

            if mw.vtk_panel:
                actor_mappings = (
                    mw.vtk_panel.roi_slice_actors,
                    mw.vtk_panel.sphere_params_per_roi,
                    mw.vtk_panel.rectangle_params_per_roi,
                )
                for mapping in actor_mappings:
                    if old_path in mapping:
                        mapping[new_path] = mapping.pop(old_path)

            for stack_name in ("unified_undo_stack", "unified_redo_stack"):
                for action in getattr(mw, stack_name, ()):
                    if action.get("roi_name") == old_path:
                        action["roi_name"] = new_path

            if mw.current_drawing_roi == old_path:
                mw.current_drawing_roi = new_path

            if mw.vtk_panel:
                mw.vtk_panel.update_status(f"Renamed: {current_name} -> {new_name}")

            mw._update_data_panel_display()
            mw._update_bundle_info_display()

        except (KeyError, ValueError, AttributeError, RuntimeError) as e:
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
            image = nib.Nifti1Image(roi_data, roi_affine)
            transactional_save(save_path, lambda path: nib.save(image, path))
            if mw.vtk_panel:
                mw.vtk_panel.update_status(f"Saved ROI to: {save_path}")
        except (OSError, ValueError) as e:
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

        # Removal cannot be undone. Drop only this layer's obsolete snapshots
        # before its key can be reused, retaining other actions and their order.
        for stack_name in ("unified_undo_stack", "unified_redo_stack"):
            stack = getattr(mw, stack_name, None)
            if stack is not None:
                stack[:] = [action for action in stack if action.get("roi_name") != path]
        mw._update_action_states()

        if path in mw.roi_visibility:
            del mw.roi_visibility[path]

        if path in mw.roi_opacities:
            del mw.roi_opacities[path]

        if path in mw.roi_states:
            del mw.roi_states[path]

        if path in mw.roi_intersection_cache:
            del mw.roi_intersection_cache[path]

        # Check if we need to reset drawing modes
        was_current_drawing_roi = mw.current_drawing_roi == path

        if mw.current_drawing_roi == path:
            mw.current_drawing_roi = None

        # Reset all drawing modes if:
        # 1. No ROIs left, OR
        # 2. The current drawing ROI was removed and any drawing mode is active
        should_reset_drawing = False
        if not mw.roi_layers:
            should_reset_drawing = True
        elif was_current_drawing_roi:
            # Check if any drawing mode is active
            is_any_mode_active = (
                getattr(mw, "is_drawing_mode", False)
                or getattr(mw, "is_eraser_mode", False)
                or getattr(mw, "is_sphere_mode", False)
                or getattr(mw, "is_rectangle_mode", False)
            )
            if is_any_mode_active:
                should_reset_drawing = True

        if should_reset_drawing and hasattr(mw, "drawing_modes_manager"):
            mw.drawing_modes_manager.reset_all_drawing_modes()

        # Refresh with skip protection to prevent RAM exhaustion
        self._apply_filters_with_skip_protection()
        mw._update_data_panel_display()
        mw._update_bundle_info_display()

        if mw.vtk_panel:
            mw.vtk_panel.update_status(f"Removed ROI: {os.path.basename(path)}")
