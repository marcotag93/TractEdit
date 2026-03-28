# -*- coding: utf-8 -*-

"""
Tests for the StateManager class.

Covers undo/redo operations for streamline deletions and ROI modifications,
stack size limits, radius adjustment, and selection clearing.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from tractedit_pkg.logic.state_manager import StateManager, ActionType
from tractedit_pkg.utils import (
    MAX_STACK_LEVELS,
    DEFAULT_SELECTION_RADIUS,
    MIN_SELECTION_RADIUS,
    RADIUS_INCREMENT,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_main_window():
    """Creates a mock MainWindow with the minimum attributes for StateManager."""
    mw = MagicMock()

    # Core state
    mw.unified_undo_stack = []
    mw.unified_redo_stack = []
    mw.manual_visible_indices = {0, 1, 2, 3, 4}
    mw.visible_indices = {0, 1, 2, 3, 4}
    mw.selected_streamline_indices = set()
    mw.selection_radius_3d = DEFAULT_SELECTION_RADIUS
    mw.tractogram_data = MagicMock()  # Non-None to indicate loaded data
    mw._skip_user_disabled = False

    # ROI state
    mw.roi_layers = {}
    mw.roi_states = {}
    mw.roi_intersection_cache = {}

    # VTK panel (mocked)
    mw.vtk_panel = MagicMock()
    mw.vtk_panel.radius_actor = MagicMock()
    mw.vtk_panel.radius_actor.GetVisibility.return_value = False

    return mw


@pytest.fixture
def state_manager(mock_main_window):
    """Creates a StateManager bound to a mock MainWindow."""
    sm = StateManager(mock_main_window)
    # Also wire up the roi_manager.apply_logic_filters mock
    mock_main_window.roi_manager = MagicMock()
    return sm


# ============================================================================
# Streamline Undo/Redo Tests
# ============================================================================


class TestStreamlineUndoRedo:
    """Tests for streamline deletion undo/redo."""

    def test_save_streamline_deletion(self, state_manager, mock_main_window):
        """Saving a deletion pushes to undo stack and clears redo stack."""
        mock_main_window.unified_redo_stack = [{"dummy": True}]

        state_manager.save_streamline_deletion_for_undo({1, 2})

        assert len(mock_main_window.unified_undo_stack) == 1
        action = mock_main_window.unified_undo_stack[0]
        assert action["action_type"] == ActionType.STREAMLINE_DELETION
        assert action["deleted_indices"] == {1, 2}
        # Redo stack should be cleared on new action
        assert len(mock_main_window.unified_redo_stack) == 0

    def test_save_empty_deletion_is_noop(self, state_manager, mock_main_window):
        """Saving an empty deletion set does nothing."""
        state_manager.save_streamline_deletion_for_undo(set())

        assert len(mock_main_window.unified_undo_stack) == 0

    def test_undo_streamline_deletion(self, state_manager, mock_main_window):
        """Undoing a deletion restores deleted indices to manual_visible_indices."""
        deleted = {1, 3}
        mock_main_window.manual_visible_indices = {0, 2, 4}

        # Push action onto undo stack
        mock_main_window.unified_undo_stack.append({
            "action_type": ActionType.STREAMLINE_DELETION,
            "deleted_indices": deleted.copy(),
        })

        state_manager.perform_undo()

        # Deleted indices should be restored
        assert 1 in mock_main_window.manual_visible_indices
        assert 3 in mock_main_window.manual_visible_indices
        # Redo stack should have the action
        assert len(mock_main_window.unified_redo_stack) == 1
        # apply_logic_filters should be called
        mock_main_window.roi_manager.apply_logic_filters.assert_called()

    def test_redo_streamline_deletion(self, state_manager, mock_main_window):
        """Redoing a deletion removes indices from manual_visible_indices."""
        deleted = {1, 3}
        mock_main_window.manual_visible_indices = {0, 1, 2, 3, 4}

        # Push action onto redo stack
        mock_main_window.unified_redo_stack.append({
            "action_type": ActionType.STREAMLINE_DELETION,
            "deleted_indices": deleted.copy(),
        })

        state_manager.perform_redo()

        assert 1 not in mock_main_window.manual_visible_indices
        assert 3 not in mock_main_window.manual_visible_indices
        assert len(mock_main_window.unified_undo_stack) == 1

    def test_undo_empty_stack(self, state_manager, mock_main_window):
        """Undoing with empty stack shows status message."""
        state_manager.perform_undo()

        mock_main_window.vtk_panel.update_status.assert_called_with(
            "Nothing to undo."
        )

    def test_redo_empty_stack(self, state_manager, mock_main_window):
        """Redoing with empty stack shows status message."""
        state_manager.perform_redo()

        mock_main_window.vtk_panel.update_status.assert_called_with(
            "Nothing to redo."
        )

    def test_undo_preserves_skip_override_for_small_tractogram(
        self, state_manager, mock_main_window
    ):
        """Undo keeps user skip OFF for tractograms with <= 250k streamlines."""
        deleted = {1, 3}
        mock_main_window.manual_visible_indices = {0, 2, 4}
        mock_main_window.tractogram_data = range(250000)
        mock_main_window._skip_user_disabled = True

        mock_main_window.unified_undo_stack.append(
            {
                "action_type": ActionType.STREAMLINE_DELETION,
                "deleted_indices": deleted.copy(),
            }
        )

        state_manager.perform_undo()

        assert mock_main_window._skip_user_disabled is True
        mock_main_window._auto_calculate_skip_level.assert_not_called()

    def test_undo_forces_auto_skip_for_large_tractogram(
        self, state_manager, mock_main_window
    ):
        """Undo forces auto-skip protection for tractograms with > 250k streamlines."""
        deleted = {1, 3}
        mock_main_window.manual_visible_indices = {0, 2, 4}
        mock_main_window.tractogram_data = range(250001)
        mock_main_window._skip_user_disabled = True

        mock_main_window.unified_undo_stack.append(
            {
                "action_type": ActionType.STREAMLINE_DELETION,
                "deleted_indices": deleted.copy(),
            }
        )

        state_manager.perform_undo()

        assert mock_main_window._skip_user_disabled is False
        mock_main_window._auto_calculate_skip_level.assert_called_once()


# ============================================================================
# Stack Size Limit Tests
# ============================================================================


class TestStackSizeLimits:
    """Tests that undo/redo stacks respect MAX_STACK_LEVELS."""

    def test_undo_stack_limit(self, state_manager, mock_main_window):
        """Undo stack does not exceed MAX_STACK_LEVELS."""
        for i in range(MAX_STACK_LEVELS + 5):
            state_manager.save_streamline_deletion_for_undo({i})

        assert len(mock_main_window.unified_undo_stack) == MAX_STACK_LEVELS

    def test_oldest_action_evicted(self, state_manager, mock_main_window):
        """Oldest action is evicted when stack overflows."""
        for i in range(MAX_STACK_LEVELS + 1):
            state_manager.save_streamline_deletion_for_undo({i})

        # The first action (index 0) should have been evicted
        first_action = mock_main_window.unified_undo_stack[0]
        assert 0 not in first_action["deleted_indices"]
        assert 1 in first_action["deleted_indices"]


# ============================================================================
# Selection Tests
# ============================================================================


class TestSelectionOperations:
    """Tests for selection clearing and deletion."""

    def test_clear_selection(self, state_manager, mock_main_window):
        """Clearing selection empties selected_streamline_indices."""
        mock_main_window.selected_streamline_indices = {0, 1, 2}

        state_manager.perform_clear_selection()

        assert mock_main_window.selected_streamline_indices == set()
        mock_main_window.vtk_panel.update_highlight.assert_called_once()

    def test_clear_empty_selection(self, state_manager, mock_main_window):
        """Clearing when nothing is selected shows appropriate message."""
        mock_main_window.selected_streamline_indices = set()

        state_manager.perform_clear_selection()

        mock_main_window.vtk_panel.update_status.assert_called_with(
            "Clear: No active selection."
        )

    def test_delete_selection(self, state_manager, mock_main_window):
        """Deleting selection removes indices and pushes to undo stack."""
        mock_main_window.selected_streamline_indices = {1, 2}
        mock_main_window.manual_visible_indices = {0, 1, 2, 3, 4}

        state_manager.perform_delete_selection()

        # Deleted indices removed from manual_visible_indices
        assert 1 not in mock_main_window.manual_visible_indices
        assert 2 not in mock_main_window.manual_visible_indices
        # Selection should be cleared after deletion
        assert mock_main_window.selected_streamline_indices == set()
        # Undo stack should have the action
        assert len(mock_main_window.unified_undo_stack) == 1

    def test_delete_empty_selection_is_noop(self, state_manager, mock_main_window):
        """Deleting with no selection does nothing."""
        mock_main_window.selected_streamline_indices = set()

        state_manager.perform_delete_selection()

        assert len(mock_main_window.unified_undo_stack) == 0


# ============================================================================
# Radius Tests
# ============================================================================


class TestRadiusAdjustment:
    """Tests for selection radius increase/decrease."""

    def test_increase_radius(self, state_manager, mock_main_window):
        """Increasing radius adds RADIUS_INCREMENT."""
        initial = mock_main_window.selection_radius_3d

        state_manager.increase_radius()

        assert mock_main_window.selection_radius_3d == initial + RADIUS_INCREMENT

    def test_decrease_radius(self, state_manager, mock_main_window):
        """Decreasing radius subtracts RADIUS_INCREMENT."""
        initial = mock_main_window.selection_radius_3d

        state_manager.decrease_radius()

        expected = max(MIN_SELECTION_RADIUS, initial - RADIUS_INCREMENT)
        assert mock_main_window.selection_radius_3d == expected

    def test_radius_cannot_go_below_minimum(self, state_manager, mock_main_window):
        """Radius never drops below MIN_SELECTION_RADIUS."""
        mock_main_window.selection_radius_3d = MIN_SELECTION_RADIUS

        state_manager.decrease_radius()

        assert mock_main_window.selection_radius_3d == MIN_SELECTION_RADIUS

    def test_radius_no_change_without_tractogram(
        self, state_manager, mock_main_window
    ):
        """Radius operations do nothing when no tractogram is loaded."""
        mock_main_window.tractogram_data = None
        initial = mock_main_window.selection_radius_3d

        state_manager.increase_radius()
        assert mock_main_window.selection_radius_3d == initial

        state_manager.decrease_radius()
        assert mock_main_window.selection_radius_3d == initial


# ============================================================================
# ROI Undo/Redo Tests
# ============================================================================


class TestROIUndoRedo:
    """Tests for ROI modification undo/redo."""

    def test_save_roi_state(self, state_manager, mock_main_window):
        """Saving ROI state creates a snapshot on the undo stack."""
        roi_data = np.ones((10, 10, 10), dtype=np.uint8)
        mock_main_window.roi_layers["test_roi"] = {
            "data": roi_data,
            "affine": np.eye(4),
        }
        mock_main_window.vtk_panel.sphere_params_per_roi = {}
        mock_main_window.vtk_panel.rectangle_params_per_roi = {}

        state_manager.save_roi_state_for_undo("test_roi")

        assert len(mock_main_window.unified_undo_stack) == 1
        action = mock_main_window.unified_undo_stack[0]
        assert action["action_type"] == ActionType.ROI_MODIFICATION
        assert action["roi_name"] == "test_roi"
        # Data snapshot should be a copy, not the same reference
        assert action["data_snapshot"] is not roi_data
        np.testing.assert_array_equal(action["data_snapshot"], roi_data)

    def test_save_roi_state_nonexistent_roi(
        self, state_manager, mock_main_window
    ):
        """Saving state for a non-existent ROI does nothing."""
        state_manager.save_roi_state_for_undo("nonexistent_roi")

        assert len(mock_main_window.unified_undo_stack) == 0

    def test_undo_roi_modification(self, state_manager, mock_main_window):
        """Undoing an ROI modification restores the previous data snapshot."""
        roi_data = np.ones((10, 10, 10), dtype=np.uint8)
        old_data = np.zeros((10, 10, 10), dtype=np.uint8)
        mock_main_window.roi_layers["test_roi"] = {
            "data": roi_data,
            "affine": np.eye(4),
        }
        mock_main_window.vtk_panel.sphere_params_per_roi = {}
        mock_main_window.vtk_panel.rectangle_params_per_roi = {}

        mock_main_window.unified_undo_stack.append({
            "action_type": ActionType.ROI_MODIFICATION,
            "roi_name": "test_roi",
            "data_snapshot": old_data.copy(),
            "sphere_params": None,
            "rectangle_params": None,
        })

        state_manager.perform_undo()

        # ROI data should be restored to old_data
        np.testing.assert_array_equal(
            mock_main_window.roi_layers["test_roi"]["data"], old_data
        )
        # Redo stack should have the current (modified) state
        assert len(mock_main_window.unified_redo_stack) == 1

    def test_undo_roi_for_deleted_roi(self, state_manager, mock_main_window):
        """Undoing an ROI action when the ROI no longer exists skips gracefully."""
        mock_main_window.unified_undo_stack.append({
            "action_type": ActionType.ROI_MODIFICATION,
            "roi_name": "deleted_roi",
            "data_snapshot": np.zeros((5, 5, 5)),
            "sphere_params": None,
            "rectangle_params": None,
        })

        state_manager.perform_undo()

        mock_main_window.vtk_panel.update_status.assert_called()
        # Redo stack should NOT get an entry for non-existent ROI
        assert len(mock_main_window.unified_redo_stack) == 0


# ============================================================================
# Action Type Tests
# ============================================================================


class TestActionType:
    """Tests for the ActionType enum."""

    def test_action_types_exist(self):
        """Both action types are defined."""
        assert ActionType.STREAMLINE_DELETION is not None
        assert ActionType.ROI_MODIFICATION is not None

    def test_action_types_distinct(self):
        """Action types have different values."""
        assert ActionType.STREAMLINE_DELETION != ActionType.ROI_MODIFICATION
