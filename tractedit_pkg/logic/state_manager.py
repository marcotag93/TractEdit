# -*- coding: utf-8 -*-

"""
State Manager for TractEdit application.

Handles application state operations including:
- Unified undo/redo for all operations (streamline deletions and ROI modifications)
- Selection operations (clear, delete)
- Radius adjustment
- Color mode management
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Set

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from ..render_batch import batch_render, request_render
from ..utils import (
    AUTO_SKIP_THRESHOLD,
    ColorMode,
    MAX_STACK_LEVELS,
    MIN_SELECTION_RADIUS,
    RADIUS_INCREMENT,
)

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# Action Type
# ============================================================================


class ActionType(Enum):
    """Enum defining the types of undoable actions.

    This enables unified undo/redo across all modes without requiring
    the user to be in a specific mode to undo a particular action type.
    """

    STREAMLINE_DELETION = auto()
    ROI_MODIFICATION = auto()
    POLYLINE_EDIT = auto()


_ROI_HISTORY_SCAN_ITEMS = 262144


def retain_polyline_history(mw) -> None:
    for name in ("unified_undo_stack", "unified_redo_stack"):
        retained = [
            action for action in getattr(mw, name, ())
            if action.get("action_type") == ActionType.POLYLINE_EDIT
        ]
        setattr(mw, name, retained)


def _different_elements(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.dtype != second.dtype or first.shape != second.shape:
        raise ValueError("ROI history arrays must have matching shape and dtype.")
    if first.dtype.kind == "f":
        unsigned_dtype = np.dtype(f"u{first.dtype.itemsize}")
        return np.not_equal(
            first.reshape(-1).view(unsigned_dtype),
            second.reshape(-1).view(unsigned_dtype),
        )
    if first.dtype.kind == "c":
        first_bytes = np.ascontiguousarray(first).view(np.uint8)
        second_bytes = np.ascontiguousarray(second).view(np.uint8)
        first_bytes = first_bytes.reshape(-1, first.dtype.itemsize)
        second_bytes = second_bytes.reshape(-1, second.dtype.itemsize)
        return np.any(first_bytes != second_bytes, axis=1)
    return np.not_equal(first, second).reshape(-1)


def _parameters_equal(first: Any, second: Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
            return False
        return (
            first.dtype == second.dtype
            and first.shape == second.shape
            and not np.any(_different_elements(first.ravel(), second.ravel()))
        )
    if isinstance(first, dict) or isinstance(second, dict):
        if not isinstance(first, dict) or not isinstance(second, dict):
            return False
        return first.keys() == second.keys() and all(
            _parameters_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            return False
        return all(_parameters_equal(a, b) for a, b in zip(first, second))
    return first == second


# ============================================================================
# State Manager Class
# ============================================================================


class StateManager:
    """
    Manages application state operations.

    This class handles:
    - Unified undo/redo for streamline deletions and ROI modifications
    - Selection operations (clear, delete)
    - Camera reset
    - Radius adjustment
    - Color mode management

    The unified undo/redo system maintains a single chronological history
    of all operations, allowing users to undo/redo in order regardless
    of the operation type or current mode.
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the state manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def perform_undo(self) -> None:
        self._replay_history(undo=True)

    def perform_redo(self) -> None:
        self._replay_history(undo=False)

    def _replay_history(self, *, undo: bool) -> None:
        mw = self.mw
        polylines = getattr(mw, "mesh_polyline_manager", None)
        if polylines is not None and polylines.active is True:
            polylines.step_draft(undo=undo)
            return
        source = mw.unified_undo_stack if undo else mw.unified_redo_stack
        operation = "undo" if undo else "redo"
        if not source:
            if mw.vtk_panel:
                mw.vtk_panel.update_status(f"Nothing to {operation}.")
            return
        if source[-1].get("action_type") == ActionType.POLYLINE_EDIT:
            mw.mesh_polyline_manager.replay_history(undo=undo)
            return

        action = source.pop()
        action_type = action.get("action_type")
        if action_type == ActionType.ROI_MODIFICATION:
            destination = mw.unified_redo_stack if undo else mw.unified_undo_stack
            self._apply_roi_action(action, destination, "undone" if undo else "redone")
        elif action_type == ActionType.STREAMLINE_DELETION:
            self._replay_streamline_action(action, undo=undo)
        else:
            logger.warning("Unknown action type in %s stack: %s", operation, action_type)
        mw._update_action_states()

    @batch_render(lambda self: self.mw.vtk_panel)
    def _replay_streamline_action(self, action: Dict[str, Any], *, undo: bool) -> None:
        mw = self.mw
        deleted_indices = action.get("deleted_indices", set())
        if not deleted_indices:
            return

        destination = mw.unified_redo_stack if undo else mw.unified_undo_stack
        destination.append({
            "action_type": ActionType.STREAMLINE_DELETION,
            "deleted_indices": deleted_indices.copy(),
        })
        if len(destination) > MAX_STACK_LEVELS:
            destination.pop(0)

        if undo:
            mw.manual_visible_indices.update(deleted_indices)
            force_auto_skip = self._should_force_auto_skip_for_undo_redo()
            if force_auto_skip:
                mw._skip_user_disabled = False
            mw.roi_manager.apply_logic_filters(recalculate_skip=force_auto_skip)
            status = "Undone: Restored"
        else:
            mw.manual_visible_indices.difference_update(deleted_indices)
            mw.roi_manager.apply_logic_filters()
            status = "Redone: Deleted"

        if mw.vtk_panel:
            mw.vtk_panel.update_status(f"{status} {len(deleted_indices)} streamline(s).")

    def _should_force_auto_skip_for_undo_redo(self) -> bool:
        """Return True when undo/redo should force auto-skip protection."""
        tractogram_data = self.mw.tractogram_data
        if tractogram_data is None:
            return False

        try:
            total_streamlines = len(tractogram_data)
        except TypeError:
            return False

        return total_streamlines >= AUTO_SKIP_THRESHOLD

    def _roi_parameters(self, roi_name: str) -> tuple[Any, Any]:
        panel = self.mw.vtk_panel
        if panel is None:
            return None, None
        sphere = getattr(panel, "sphere_params_per_roi", {}).get(roi_name)
        rectangle = getattr(panel, "rectangle_params_per_roi", {}).get(roi_name)
        return deepcopy(sphere), deepcopy(rectangle)

    def _capture_roi_baseline(self, roi_name: str) -> Optional[Dict[str, Any]]:
        if roi_name not in self.mw.roi_layers:
            return None
        data = self.mw.roi_layers[roi_name]["data"]
        sphere, rectangle = self._roi_parameters(roi_name)
        return {
            "data_snapshot": data.copy(),
            "sphere_params": sphere,
            "rectangle_params": rectangle,
        }

    def _materialize_roi_action(
        self,
        roi_name: str,
        baseline: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if roi_name not in self.mw.roi_layers:
            return None
        data = self.mw.roi_layers[roi_name]["data"]
        old_sphere = baseline.get("sphere_params")
        old_rectangle = baseline.get("rectangle_params")
        sphere, rectangle = self._roi_parameters(roi_name)
        parameters_changed = not _parameters_equal(
            old_sphere, sphere
        ) or not _parameters_equal(old_rectangle, rectangle)

        old_snapshot = baseline.get("data_snapshot")
        if old_snapshot is None:
            return None
        patch_item_bytes = np.dtype(np.int64).itemsize + data.dtype.itemsize
        use_snapshot = False
        if data.flags.c_contiguous:
            different = _different_elements(old_snapshot.ravel(), data.reshape(-1))
            changed_count = int(np.count_nonzero(different))
            use_snapshot = changed_count * patch_item_bytes >= data.nbytes
            changed_indices = (
                np.empty(0, dtype=np.int64)
                if use_snapshot
                else np.flatnonzero(different).astype(np.int64, copy=False)
            )
        else:
            changed_parts = []
            changed_count = 0
            for start in range(0, data.size, _ROI_HISTORY_SCAN_ITEMS):
                stop = min(start + _ROI_HISTORY_SCAN_ITEMS, data.size)
                previous = np.asarray(old_snapshot.flat[start:stop])
                current = np.asarray(data.flat[start:stop])
                local = np.flatnonzero(_different_elements(previous, current))
                if local.size:
                    changed_count += len(local)
                    if changed_count * patch_item_bytes >= data.nbytes:
                        use_snapshot = True
                        changed_parts.clear()
                        break
                    changed_parts.append(local.astype(np.int64) + start)
            changed_indices = (
                np.concatenate(changed_parts)
                if changed_parts
                else np.empty(0, dtype=np.int64)
            )

        if changed_count == 0 and not parameters_changed:
            return None
        action = {
            "action_type": ActionType.ROI_MODIFICATION,
            "roi_name": roi_name,
            "sphere_params": deepcopy(old_sphere),
            "rectangle_params": deepcopy(old_rectangle),
        }
        if not use_snapshot:
            old_values = np.asarray(old_snapshot.flat[changed_indices]).copy()
            action.update(
                {
                    "voxel_indices": changed_indices,
                    "voxel_values": old_values,
                    "data_shape": data.shape,
                }
            )
        else:
            action["data_snapshot"] = old_snapshot
        return action

    @contextmanager
    def roi_modification(self, roi_name: str) -> Iterator[None]:
        """Record one exact, memory-bounded ROI modification."""
        baseline = self._capture_roi_baseline(roi_name)
        try:
            yield
        finally:
            if baseline is not None:
                action = self._materialize_roi_action(roi_name, baseline)
                if action is not None:
                    self.mw.unified_undo_stack.append(action)
                    self.mw.unified_redo_stack.clear()
                    if len(self.mw.unified_undo_stack) > MAX_STACK_LEVELS:
                        self.mw.unified_undo_stack.pop(0)

    @batch_render(lambda self: self.mw.vtk_panel)
    def _apply_roi_action(
        self,
        action: Dict[str, Any],
        destination_stack: list[Dict[str, Any]],
        operation: str,
    ) -> None:
        mw = self.mw
        roi_name = action.get("roi_name")
        if not roi_name or roi_name not in mw.roi_layers:
            if roi_name and mw.vtk_panel:
                mw.vtk_panel.update_status(
                    f"ROI {roi_name} no longer exists, skipping {operation}."
                )
            return

        data = mw.roi_layers[roi_name]["data"]
        sphere, rectangle = self._roi_parameters(roi_name)
        inverse = {
            "action_type": ActionType.ROI_MODIFICATION,
            "roi_name": roi_name,
            "sphere_params": sphere,
            "rectangle_params": rectangle,
        }

        if "data_snapshot" in action:
            inverse["data_snapshot"] = data.copy()
            data[:] = action["data_snapshot"]
        else:
            indices = action.get("voxel_indices")
            values = action.get("voxel_values")
            if indices is None or values is None:
                return
            inverse.update(
                {
                    "voxel_indices": indices.copy(),
                    "voxel_values": np.asarray(data.flat[indices]).copy(),
                    "data_shape": data.shape,
                }
            )
            data.flat[indices] = values

        destination_stack.append(inverse)
        if len(destination_stack) > MAX_STACK_LEVELS:
            destination_stack.pop(0)
        self._restore_roi_parameters(
            roi_name,
            action.get("sphere_params"),
            action.get("rectangle_params"),
        )

        if mw.vtk_panel:
            roi_affine = mw.roi_layers[roi_name]["affine"]
            mw.vtk_panel.update_roi_layer(roi_name, data, roi_affine)
            mw.vtk_panel.update_status(
                f"ROI operation {operation} on {os.path.basename(roi_name)}"
            )
        if (action.get("sphere_params") is not None
                or action.get("rectangle_params") is not None
                or any(mw.roi_states.get(roi_name, {}).values())):
            mw.roi_manager.compute_roi_intersection(roi_name)
        else:
            mw.roi_intersection_cache.pop(roi_name, None)
        mw.roi_manager.update_roi_visual_selection()
        mw.roi_manager.apply_logic_filters()

    def _restore_roi_parameters(
        self,
        roi_name: str,
        sphere: Any,
        rectangle: Any,
    ) -> None:
        panel = self.mw.vtk_panel
        if panel is None:
            return
        sphere_parameters = getattr(panel, "sphere_params_per_roi", {})
        rectangle_parameters = getattr(panel, "rectangle_params_per_roi", {})
        if sphere is None:
            sphere_parameters.pop(roi_name, None)
        else:
            sphere_parameters[roi_name] = deepcopy(sphere)
        if rectangle is None:
            rectangle_parameters.pop(roi_name, None)
        else:
            rectangle_parameters[roi_name] = deepcopy(rectangle)

    def save_streamline_deletion_for_undo(self, deleted_indices: Set[int]) -> None:
        """
        Saves a streamline deletion to the unified undo stack.

        Args:
            deleted_indices: Set of streamline indices that were deleted.
        """
        mw = self.mw

        if not deleted_indices:
            return

        action = {
            "action_type": ActionType.STREAMLINE_DELETION,
            "deleted_indices": deleted_indices.copy(),
        }
        mw.unified_undo_stack.append(action)

        # Clear redo stack (new action invalidates redo history)
        mw.unified_redo_stack.clear()

        # Limit stack size
        if len(mw.unified_undo_stack) > MAX_STACK_LEVELS:
            mw.unified_undo_stack.pop(0)

    @batch_render(lambda self: self.mw.vtk_panel)
    def perform_clear_selection(self) -> None:
        """Clear the current streamline selection and exit inversion mode."""
        mw = self.mw

        if mw.vtk_panel:
            mw.vtk_panel.update_radius_actor(visible=False)

        if mw.selected_streamline_indices:
            mw.selected_streamline_indices = set()
            mapping = getattr(mw, "surface_mapping_manager", None)
            if mapping is not None:
                mapping.selection_changed()

            # Exit inversion mode and remove the cyan keeper contour.
            mw._inversion_active = False
            mw._inversion_keeper_indices = set()
            if mw.vtk_panel:
                mw.vtk_panel.clear_invert_contour()
                mw.vtk_panel.update_highlight()
                mw.vtk_panel.update_status("Selection cleared.")
        elif mw.vtk_panel:
            mw.vtk_panel.update_status("Clear: No active selection.")

        mw._update_action_states()

    def perform_reset_camera(self) -> None:
        """
        Resets the 3D camera view to a Front Coronal orientation.
        Centers the view and aligns it with the Y-axis (Anterior-Posterior).
        """
        mw = self.mw

        if not mw.vtk_panel or not mw.vtk_panel.scene:
            return

        # Standard view reset
        mw.vtk_panel.scene.reset_camera()

        # Get the camera and current parameters
        cam = mw.vtk_panel.scene.GetActiveCamera()
        fp = cam.GetFocalPoint()
        dist = cam.GetDistance()

        # Re-orient to Front Coronal (Anterior View)
        cam.SetPosition(fp[0], fp[1] + dist, fp[2])
        cam.SetFocalPoint(fp[0], fp[1], fp[2])
        cam.SetViewUp(0, 0, 1)

        # Finalize update
        mw.vtk_panel.scene.reset_clipping_range()
        mw.vtk_panel.update_status("Camera reset (Front Coronal).", render=False)
        if mw.vtk_panel.render_window:
            request_render(mw.vtk_panel, mw.vtk_panel.render_window)

    @batch_render(lambda self: self.mw.vtk_panel)
    def perform_delete_selection(self) -> None:
        """Delete the currently selected streamlines and exit inversion mode."""
        mw = self.mw

        if not mw.selected_streamline_indices:
            return

        # Save to unified undo stack.
        to_delete = mw.selected_streamline_indices.copy()
        self.save_streamline_deletion_for_undo(to_delete)

        # Update MANUAL state.
        mw.manual_visible_indices.difference_update(to_delete)

        mw.selected_streamline_indices = set()
        mapping = getattr(mw, "surface_mapping_manager", None)
        if mapping is not None:
            mapping.selection_changed()

        # Exit inversion mode and remove the cyan keeper contour.
        mw._inversion_active = False
        mw._inversion_keeper_indices = set()
        if mw.vtk_panel:
            mw.vtk_panel.clear_invert_contour()

        mw.roi_manager.apply_logic_filters(recalculate_skip=True)

        mw._update_action_states()

        # Hide selection sphere after deletion.
        if mw.vtk_panel:
            mw.vtk_panel.update_radius_actor(visible=False)
            mw.vtk_panel.update_status(f"Deleted {len(to_delete)} streamline(s).")

    def increase_radius(self) -> None:
        self._adjust_radius(RADIUS_INCREMENT)

    def decrease_radius(self) -> None:
        self._adjust_radius(-RADIUS_INCREMENT)

    @batch_render(lambda self: self.mw.vtk_panel)
    def _adjust_radius(self, delta: float) -> None:
        mw = self.mw
        if not mw.tractogram_data:
            return

        radius = mw.selection_radius_3d + delta
        mw.selection_radius_3d = max(MIN_SELECTION_RADIUS, radius) if delta < 0 else radius
        direction = "decreased" if delta < 0 else "increased"
        if mw.vtk_panel:
            mw.vtk_panel.update_status(
                f"Selection radius {direction} to {mw.selection_radius_3d:.1f}mm."
            )
            if mw.vtk_panel.radius_actor and mw.vtk_panel.radius_actor.GetVisibility():
                center = mw.vtk_panel.radius_actor.GetCenter()
                mw.vtk_panel.update_radius_actor(
                    center_point=center, radius=mw.selection_radius_3d, visible=True
                )

    @batch_render(lambda self: self.mw.vtk_panel)
    def hide_sphere(self) -> None:
        """Hides the selection sphere."""
        mw = self.mw

        if mw.vtk_panel:
            mw.vtk_panel.update_radius_actor(visible=False)
            mw.vtk_panel.update_status("Selection sphere hidden.")

    @batch_render(lambda self: self.mw.vtk_panel)
    def set_color_mode(self, mode: ColorMode) -> None:
        """Sets the streamline coloring mode and triggers VTK update."""
        mw = self.mw

        if not isinstance(mode, ColorMode):
            return
        if not mw.tractogram_data:
            mw.color_default_action.setChecked(True)
            return

        # Handle scalar toolbar visibility
        if mw.current_color_mode != mode:
            if mode == ColorMode.SCALAR:
                if not mw.active_scalar_name:
                    QMessageBox.warning(
                        mw,
                        "Coloring Error",
                        "No active scalar data loaded for streamlines.",
                    )
                    if mw.current_color_mode == ColorMode.DEFAULT:
                        mw.color_default_action.setChecked(True)
                    elif mw.current_color_mode == ColorMode.ORIENTATION:
                        mw.color_orientation_action.setChecked(True)
                    return

                # Calculate range in scalar mode
                if not mw.scalar_range_initialized:
                    mw.scalar_manager.update_scalar_data_range()
                    mw.scalar_range_initialized = True

                if mw.scalar_toolbar:
                    mw.scalar_toolbar.setVisible(True)

            elif mode == ColorMode.DEFAULT or mode == ColorMode.ORIENTATION:
                if mw.scalar_toolbar:
                    mw.scalar_toolbar.setVisible(False)

                mw.bundle_is_visible = True

            mw.current_color_mode = mode
            if mw.vtk_panel:
                mw.vtk_panel.update_main_streamlines_actor()
                mw.vtk_panel.update_status(
                    f"Streamline color mode changed to {mode.name}."
                )

        # Ensure toolbar visibility
        if mw.scalar_toolbar:
            is_scalar = mode == ColorMode.SCALAR and bool(mw.active_scalar_name)
            mw.scalar_toolbar.setVisible(is_scalar)
