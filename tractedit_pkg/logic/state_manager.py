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
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from ..utils import (
    AUTO_SKIP_THRESHOLD,
    ColorMode,
    MAX_STACK_LEVELS,
    MIN_SELECTION_RADIUS,
    RADIUS_INCREMENT,
    TARGET_RENDER_COUNT,
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
    SELECTION_CHANGE = auto()


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
        """
        Performs undo operation from the unified stack.

        Dispatches to the appropriate undo handler based on the action type
        stored in the action record, not the current UI mode.
        """
        mw = self.mw

        if not mw.unified_undo_stack:
            if mw.vtk_panel:
                mw.vtk_panel.update_status("Nothing to undo.")
            return

        action = mw.unified_undo_stack.pop()
        action_type = action.get("action_type")

        if action_type == ActionType.ROI_MODIFICATION:
            self._undo_roi_action(action)
        elif action_type == ActionType.STREAMLINE_DELETION:
            self._undo_streamline_action(action)
        elif action_type == ActionType.SELECTION_CHANGE:
            self._undo_selection_action(action)
        else:
            logger.warning(f"Unknown action type in undo stack: {action_type}")

        mw._update_action_states()

    def perform_redo(self) -> None:
        """
        Performs redo operation from the unified stack.

        Dispatches to the appropriate redo handler based on the action type
        stored in the action record, not the current UI mode.
        """
        mw = self.mw

        if not mw.unified_redo_stack:
            if mw.vtk_panel:
                mw.vtk_panel.update_status("Nothing to redo.")
            return

        action = mw.unified_redo_stack.pop()
        action_type = action.get("action_type")

        if action_type == ActionType.ROI_MODIFICATION:
            self._redo_roi_action(action)
        elif action_type == ActionType.STREAMLINE_DELETION:
            self._redo_streamline_action(action)
        elif action_type == ActionType.SELECTION_CHANGE:
            self._redo_selection_action(action)
        else:
            logger.warning(f"Unknown action type in redo stack: {action_type}")

        mw._update_action_states()

    def _undo_streamline_action(self, action: Dict[str, Any]) -> None:
        """Undo a streamline deletion, restoring visibility and re-applying skip.

        After restoring deleted streamlines, the skip level is always
        recalculated via ``_auto_calculate_skip_level()``.  For tractograms
        at or above ``AUTO_SKIP_THRESHOLD``, the user's manual
        skip-disable override is also cleared and a conservative pre-stride
        is applied before the ROI filter pass to prevent a transient
        stride-1 render from exhausting RAM.

        Args:
            action: The action record containing deleted_indices.
        """
        mw = self.mw

        deleted_indices = action.get("deleted_indices", set())
        if not deleted_indices:
            return

        redo_action = {
            "action_type": ActionType.STREAMLINE_DELETION,
            "deleted_indices": deleted_indices.copy(),
        }
        mw.unified_redo_stack.append(redo_action)
        if len(mw.unified_redo_stack) > MAX_STACK_LEVELS:
            mw.unified_redo_stack.pop(0)

        mw.manual_visible_indices.update(deleted_indices)

        force_auto_skip = self._should_force_auto_skip_for_undo_redo()
        if force_auto_skip:
            mw._skip_user_disabled = False

            approx_visible = len(mw.manual_visible_indices)
            if approx_visible > TARGET_RENDER_COUNT:
                mw.render_stride = max(1, approx_visible // TARGET_RENDER_COUNT)
            else:
                mw.render_stride = 1

        mw.roi_manager.apply_logic_filters()
        mw._auto_calculate_skip_level()

        if mw.vtk_panel:
            mw.vtk_panel.update_status(
                f"Undone: Restored {len(deleted_indices)} streamline(s)."
            )

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

    def _redo_streamline_action(self, action: Dict[str, Any]) -> None:
        """
        Redoes a streamline deletion action.

        Args:
            action: The action record containing deleted_indices.
        """
        mw = self.mw

        deleted_indices = action.get("deleted_indices", set())
        if not deleted_indices:
            return

        # Create undo action to allow re-undoing
        undo_action = {
            "action_type": ActionType.STREAMLINE_DELETION,
            "deleted_indices": deleted_indices.copy(),
        }
        mw.unified_undo_stack.append(undo_action)

        # Limit undo stack size
        if len(mw.unified_undo_stack) > MAX_STACK_LEVELS:
            mw.unified_undo_stack.pop(0)

        # Re-delete the streamlines
        mw.manual_visible_indices.difference_update(deleted_indices)
        mw.roi_manager.apply_logic_filters()

        if mw.vtk_panel:
            mw.vtk_panel.update_status(
                f"Redone: Deleted {len(deleted_indices)} streamline(s)."
            )

    def _undo_roi_action(self, action: Dict[str, Any]) -> None:
        """
        Undoes an ROI modification action.

        Args:
            action: The action record containing ROI state snapshot.
        """
        mw = self.mw

        roi_name = action.get("roi_name")
        old_data = action.get("data_snapshot")
        old_sphere_params = action.get("sphere_params")
        old_rectangle_params = action.get("rectangle_params")

        if not roi_name or old_data is None:
            return

        # Check if ROI still exists
        if roi_name not in mw.roi_layers:
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    f"ROI {roi_name} no longer exists, skipping undo."
                )
            return

        # Save current state for redo
        current_data = mw.roi_layers[roi_name]["data"].copy()

        current_sphere_params = None
        if mw.vtk_panel and hasattr(mw.vtk_panel, "sphere_params_per_roi"):
            if roi_name in mw.vtk_panel.sphere_params_per_roi:
                current_sphere_params = mw.vtk_panel.sphere_params_per_roi[
                    roi_name
                ].copy()

        current_rectangle_params = None
        if mw.vtk_panel and hasattr(mw.vtk_panel, "rectangle_params_per_roi"):
            if roi_name in mw.vtk_panel.rectangle_params_per_roi:
                current_rectangle_params = mw.vtk_panel.rectangle_params_per_roi[
                    roi_name
                ].copy()

        redo_action = {
            "action_type": ActionType.ROI_MODIFICATION,
            "roi_name": roi_name,
            "data_snapshot": current_data,
            "sphere_params": current_sphere_params,
            "rectangle_params": current_rectangle_params,
        }
        mw.unified_redo_stack.append(redo_action)

        # Limit redo stack size
        if len(mw.unified_redo_stack) > MAX_STACK_LEVELS:
            mw.unified_redo_stack.pop(0)

        # Restore old state
        mw.roi_layers[roi_name]["data"][:] = old_data

        # Restore sphere params
        if mw.vtk_panel:
            if old_sphere_params:
                mw.vtk_panel.sphere_params_per_roi[roi_name] = old_sphere_params
            elif roi_name in mw.vtk_panel.sphere_params_per_roi:
                del mw.vtk_panel.sphere_params_per_roi[roi_name]

        # Restore rectangle params
        if mw.vtk_panel:
            if old_rectangle_params:
                mw.vtk_panel.rectangle_params_per_roi[roi_name] = old_rectangle_params
            elif roi_name in mw.vtk_panel.rectangle_params_per_roi:
                del mw.vtk_panel.rectangle_params_per_roi[roi_name]

        # Update visualization
        if mw.vtk_panel:
            roi_affine = mw.roi_layers[roi_name]["affine"]
            mw.vtk_panel.update_roi_layer(roi_name, old_data, roi_affine)
            mw.vtk_panel.update_status(
                f"ROI operation undone on {os.path.basename(roi_name)}"
            )

        # Re-calculate Intersection and Logic
        mw.roi_manager.compute_roi_intersection(roi_name)
        mw.roi_manager.update_roi_visual_selection()
        mw.roi_manager.apply_logic_filters()

    def _redo_roi_action(self, action: Dict[str, Any]) -> None:
        """
        Redoes an ROI modification action.

        Args:
            action: The action record containing ROI state snapshot.
        """
        mw = self.mw

        roi_name = action.get("roi_name")
        redo_data = action.get("data_snapshot")
        redo_sphere_params = action.get("sphere_params")
        redo_rectangle_params = action.get("rectangle_params")

        if not roi_name or redo_data is None:
            return

        # Check if ROI still exists
        if roi_name not in mw.roi_layers:
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    f"ROI {roi_name} no longer exists, skipping redo."
                )
            return

        # Save current state to undo stack
        current_data = mw.roi_layers[roi_name]["data"].copy()

        current_sphere_params = None
        if mw.vtk_panel and hasattr(mw.vtk_panel, "sphere_params_per_roi"):
            if roi_name in mw.vtk_panel.sphere_params_per_roi:
                current_sphere_params = mw.vtk_panel.sphere_params_per_roi[
                    roi_name
                ].copy()

        current_rectangle_params = None
        if mw.vtk_panel and hasattr(mw.vtk_panel, "rectangle_params_per_roi"):
            if roi_name in mw.vtk_panel.rectangle_params_per_roi:
                current_rectangle_params = mw.vtk_panel.rectangle_params_per_roi[
                    roi_name
                ].copy()

        undo_action = {
            "action_type": ActionType.ROI_MODIFICATION,
            "roi_name": roi_name,
            "data_snapshot": current_data,
            "sphere_params": current_sphere_params,
            "rectangle_params": current_rectangle_params,
        }
        mw.unified_undo_stack.append(undo_action)

        # Limit undo stack size
        if len(mw.unified_undo_stack) > MAX_STACK_LEVELS:
            mw.unified_undo_stack.pop(0)

        # Restore redo state
        mw.roi_layers[roi_name]["data"][:] = redo_data

        # Restore sphere params
        if mw.vtk_panel:
            if redo_sphere_params:
                mw.vtk_panel.sphere_params_per_roi[roi_name] = redo_sphere_params
            elif roi_name in mw.vtk_panel.sphere_params_per_roi:
                del mw.vtk_panel.sphere_params_per_roi[roi_name]

        # Restore rectangle params
        if mw.vtk_panel:
            if redo_rectangle_params:
                mw.vtk_panel.rectangle_params_per_roi[roi_name] = redo_rectangle_params
            elif roi_name in mw.vtk_panel.rectangle_params_per_roi:
                del mw.vtk_panel.rectangle_params_per_roi[roi_name]

        # Update visualization
        if mw.vtk_panel:
            roi_affine = mw.roi_layers[roi_name]["affine"]
            mw.vtk_panel.update_roi_layer(roi_name, redo_data, roi_affine)
            mw.vtk_panel.update_status(
                f"ROI operation redone on {os.path.basename(roi_name)}"
            )

        # Re-calculate Intersection and Logic
        mw.roi_manager.compute_roi_intersection(roi_name)
        mw.roi_manager.update_roi_visual_selection()
        mw.roi_manager.apply_logic_filters()

    def save_selection_change_for_undo(
        self, added_indices: Set[int], removed_indices: Set[int]
    ) -> None:
        """Save a sphere selection change to the unified undo stack.

        Called by :meth:`~tractedit_pkg.visualization.selection.SelectionManager.apply_selection`
        **before** mutating ``selected_streamline_indices``, so the delta is
        always recorded against the pre-operation state.

        The record stores only the net delta (which indices were added / removed),
        not a full snapshot, keeping memory consumption proportional to the
        sphere selection size rather than the total tractogram size.

        Args:
            added_indices: Indices that will be added to the selection
                (empty set for a deselect-in-sphere operation).
            removed_indices: Indices that will be removed from the selection
                (empty set for an add-only operation).
        """
        mw = self.mw

        if not added_indices and not removed_indices:
            return

        action = {
            "action_type": ActionType.SELECTION_CHANGE,
            "added_indices": added_indices.copy(),
            "removed_indices": removed_indices.copy(),
        }
        mw.unified_undo_stack.append(action)

        # Any new action invalidates the redo history.
        mw.unified_redo_stack.clear()

        if len(mw.unified_undo_stack) > MAX_STACK_LEVELS:
            mw.unified_undo_stack.pop(0)

    def _undo_selection_action(self, action: Dict[str, Any]) -> None:
        """Undo a sphere selection change, reversing the recorded delta.

        - Indices that were *added* are removed from the current selection.
        - Indices that were *removed* are re-added to the current selection.

        Both operations use a single C-level ``set`` method, consistent with
        the performance approach used in
        :meth:`~tractedit_pkg.visualization.selection.SelectionManager.apply_selection`.

        If inversion mode is active when this method is called, it is exited
        first by restoring the pre-inversion keeper set.  This is necessary
        because the undo delta was recorded against the pre-inversion state:
        applying it directly to the inverted selection would produce a
        semantically incorrect result and leave ``_inversion_active`` set,
        causing subsequent ``S`` / ``D`` operations to misbehave silently.

        Args:
            action: The action record produced by
                :meth:`save_selection_change_for_undo`.
        """
        mw = self.mw

        added_indices: Set[int] = action.get("added_indices", set())
        removed_indices: Set[int] = action.get("removed_indices", set())

        if not added_indices and not removed_indices:
            return

        # Push the inverse delta onto the redo stack so the operation can be
        # replayed.  The redo record is the mirror image of the undo record.
        redo_action = {
            "action_type": ActionType.SELECTION_CHANGE,
            "added_indices": added_indices.copy(),
            "removed_indices": removed_indices.copy(),
        }
        mw.unified_redo_stack.append(redo_action)
        if len(mw.unified_redo_stack) > MAX_STACK_LEVELS:
            mw.unified_redo_stack.pop(0)

        current_selection: Set[int] = mw.selected_streamline_indices
        if current_selection is None:
            current_selection = set()
            mw.selected_streamline_indices = current_selection

        # If inversion mode is active, restore the pre-inversion (keeper) state
        # before applying the delta.  The keeper set is the selection that
        # existed just before ``I`` was pressed — i.e., the correct baseline
        # for reversing the recorded ``S`` / ``Shift+S`` delta.
        if mw._inversion_active:
            current_selection.clear()
            current_selection.update(mw._inversion_keeper_indices)
            mw._inversion_active = False
            mw._inversion_keeper_indices = set()
            if mw.vtk_panel:
                mw.vtk_panel.clear_invert_contour()

        # Reverse the original delta with single C-level set operations.
        current_selection.difference_update(added_indices)
        current_selection.update(removed_indices)

        n_changed = len(added_indices) + len(removed_indices)
        total = len(current_selection)

        if mw.vtk_panel:
            mw.vtk_panel.update_highlight()
            mw.vtk_panel.update_status(
                f"Undone: Selection change ({n_changed:,} streamline(s)). "
                f"Total selected: {total:,}"
            )

    def _redo_selection_action(self, action: Dict[str, Any]) -> None:
        """Redo a sphere selection change, re-applying the original delta.

        - Indices that were originally *added* are re-added.
        - Indices that were originally *removed* are re-removed.

        As with :meth:`_undo_selection_action`, any active inversion mode is
        exited before the delta is applied to ensure a consistent baseline.

        Args:
            action: The action record produced by
                :meth:`save_selection_change_for_undo`.
        """
        mw = self.mw

        added_indices: Set[int] = action.get("added_indices", set())
        removed_indices: Set[int] = action.get("removed_indices", set())

        if not added_indices and not removed_indices:
            return

        # Push the inverse delta back onto the undo stack.
        undo_action = {
            "action_type": ActionType.SELECTION_CHANGE,
            "added_indices": added_indices.copy(),
            "removed_indices": removed_indices.copy(),
        }
        mw.unified_undo_stack.append(undo_action)
        if len(mw.unified_undo_stack) > MAX_STACK_LEVELS:
            mw.unified_undo_stack.pop(0)

        current_selection: Set[int] = mw.selected_streamline_indices
        if current_selection is None:
            current_selection = set()
            mw.selected_streamline_indices = current_selection

        # Exit inversion mode if active, for the same reason as in
        # ``_undo_selection_action``: the redo delta must be applied to a
        # clean, non-inverted selection baseline.
        if mw._inversion_active:
            current_selection.clear()
            current_selection.update(mw._inversion_keeper_indices)
            mw._inversion_active = False
            mw._inversion_keeper_indices = set()
            if mw.vtk_panel:
                mw.vtk_panel.clear_invert_contour()

        # Re-apply the original delta with single C-level set operations.
        current_selection.update(added_indices)
        current_selection.difference_update(removed_indices)

        n_changed = len(added_indices) + len(removed_indices)
        total = len(current_selection)

        if mw.vtk_panel:
            mw.vtk_panel.update_highlight()
            mw.vtk_panel.update_status(
                f"Redone: Selection change ({n_changed:,} streamline(s)). "
                f"Total selected: {total:,}"
            )

    def save_roi_state_for_undo(self, roi_name: str) -> None:
        """
        Saves the current ROI state to the unified undo stack before modification.

        Args:
            roi_name: The name/path of the ROI being modified.
        """
        mw = self.mw

        if roi_name not in mw.roi_layers:
            return

        # Create a deep copy of the ROI data
        roi_data_copy = mw.roi_layers[roi_name]["data"].copy()

        # Save sphere params if available
        sphere_params = None
        if mw.vtk_panel and hasattr(mw.vtk_panel, "sphere_params_per_roi"):
            if roi_name in mw.vtk_panel.sphere_params_per_roi:
                sphere_params = mw.vtk_panel.sphere_params_per_roi[roi_name].copy()

        # Save rectangle params if available
        rectangle_params = None
        if mw.vtk_panel and hasattr(mw.vtk_panel, "rectangle_params_per_roi"):
            if roi_name in mw.vtk_panel.rectangle_params_per_roi:
                rectangle_params = mw.vtk_panel.rectangle_params_per_roi[
                    roi_name
                ].copy()

        # Save to unified undo stack with action type
        action = {
            "action_type": ActionType.ROI_MODIFICATION,
            "roi_name": roi_name,
            "data_snapshot": roi_data_copy,
            "sphere_params": sphere_params,
            "rectangle_params": rectangle_params,
        }
        mw.unified_undo_stack.append(action)

        # Clear redo stack (new action invalidates redo history)
        mw.unified_redo_stack.clear()

        # Limit stack size
        if len(mw.unified_undo_stack) > MAX_STACK_LEVELS:
            mw.unified_undo_stack.pop(0)

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

    def perform_clear_selection(self) -> None:
        """Clear the current streamline selection and exit inversion mode."""
        mw = self.mw

        if mw.vtk_panel:
            mw.vtk_panel.update_radius_actor(visible=False)

        if mw.selected_streamline_indices:
            mw.selected_streamline_indices = set()

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
        if mw.vtk_panel.render_window:
            mw.vtk_panel.render_window.Render()

        mw.vtk_panel.update_status("Camera reset (Front Coronal).")

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

        # Exit inversion mode and remove the cyan keeper contour.
        mw._inversion_active = False
        mw._inversion_keeper_indices = set()
        if mw.vtk_panel:
            mw.vtk_panel.clear_invert_contour()

        mw.roi_manager.apply_logic_filters()

        # Recalculate stride so the now-smaller visible set is rendered at the correct density.
        mw._auto_calculate_skip_level()

        mw._update_action_states()

        # Hide selection sphere after deletion.
        if mw.vtk_panel:
            mw.vtk_panel.update_radius_actor(visible=False)
            mw.vtk_panel.update_status(f"Deleted {len(to_delete)} streamline(s).")

    def increase_radius(self) -> None:
        """Increases the selection radius."""
        mw = self.mw

        if not mw.tractogram_data:
            return

        mw.selection_radius_3d += RADIUS_INCREMENT
        if mw.vtk_panel:
            mw.vtk_panel.update_status(
                f"Selection radius increased to {mw.selection_radius_3d:.1f}mm."
            )
            if mw.vtk_panel.radius_actor and mw.vtk_panel.radius_actor.GetVisibility():
                center = mw.vtk_panel.radius_actor.GetCenter()
                mw.vtk_panel.update_radius_actor(
                    center_point=center, radius=mw.selection_radius_3d, visible=True
                )

    def decrease_radius(self) -> None:
        """Decreases the selection radius."""
        mw = self.mw

        if not mw.tractogram_data:
            return

        new_radius = mw.selection_radius_3d - RADIUS_INCREMENT
        mw.selection_radius_3d = max(MIN_SELECTION_RADIUS, new_radius)
        if mw.vtk_panel:
            mw.vtk_panel.update_status(
                f"Selection radius decreased to {mw.selection_radius_3d:.1f}mm."
            )
            if mw.vtk_panel.radius_actor and mw.vtk_panel.radius_actor.GetVisibility():
                center = mw.vtk_panel.radius_actor.GetCenter()
                mw.vtk_panel.update_radius_actor(
                    center_point=center, radius=mw.selection_radius_3d, visible=True
                )

    def hide_sphere(self) -> None:
        """Hides the selection sphere."""
        mw = self.mw

        if mw.vtk_panel:
            mw.vtk_panel.update_radius_actor(visible=False)
            mw.vtk_panel.update_status("Selection sphere hidden.")

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
                    mw._update_scalar_data_range()
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
