# -*- coding: utf-8 -*-

"""
Tests for the ROIManager logic filter operations.

Covers include/exclude filter application, parcellation region filters,
and the interaction between multiple simultaneous filters.
"""

from unittest.mock import MagicMock, PropertyMock
import numpy as np
import pytest

from tractedit_pkg.logic.roi_manager import ROIManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_main_window():
    """Creates a mock MainWindow with ROI-related state."""
    mw = MagicMock()

    # Core streamline state
    mw.tractogram_data = MagicMock()
    mw.tractogram_data.__len__ = MagicMock(return_value=10)
    mw.manual_visible_indices = set(range(10))
    mw.visible_indices = set(range(10))

    # ROI state
    mw.roi_layers = {}
    mw.roi_states = {}
    mw.roi_intersection_cache = {}

    # Parcellation state
    mw.parcellation_region_states = {}
    mw.parcellation_region_intersection_cache = {}

    # VTK panel
    mw.vtk_panel = MagicMock()

    return mw


@pytest.fixture
def roi_manager(mock_main_window):
    """Creates an ROIManager bound to a mock MainWindow."""
    return ROIManager(mock_main_window)


# ============================================================================
# Logic Filter Tests
# ============================================================================


class TestApplyLogicFilters:
    """Tests for the apply_logic_filters method."""

    def test_no_filters_uses_manual_visible(
        self, roi_manager, mock_main_window
    ):
        """Without filters, visible_indices equals manual_visible_indices."""
        mock_main_window.manual_visible_indices = {0, 1, 2, 3}

        roi_manager.apply_logic_filters()

        assert mock_main_window.visible_indices == {0, 1, 2, 3}

    def test_include_filter_intersects(self, roi_manager, mock_main_window):
        """Include filter keeps only intersecting streamlines."""
        mock_main_window.manual_visible_indices = {0, 1, 2, 3, 4}
        mock_main_window.roi_states = {
            "roi_a": {"select": False, "include": True, "exclude": False}
        }
        mock_main_window.roi_intersection_cache = {
            "roi_a": {1, 2, 3, 7}  # Only 1, 2, 3 overlap with manual
        }

        roi_manager.apply_logic_filters()

        assert mock_main_window.visible_indices == {1, 2, 3}

    def test_exclude_filter_removes(self, roi_manager, mock_main_window):
        """Exclude filter removes intersecting streamlines."""
        mock_main_window.manual_visible_indices = {0, 1, 2, 3, 4}
        mock_main_window.roi_states = {
            "roi_a": {"select": False, "include": False, "exclude": True}
        }
        mock_main_window.roi_intersection_cache = {
            "roi_a": {2, 3}
        }

        roi_manager.apply_logic_filters()

        assert mock_main_window.visible_indices == {0, 1, 4}

    def test_include_then_exclude(self, roi_manager, mock_main_window):
        """Include followed by exclude applies both filters."""
        mock_main_window.manual_visible_indices = set(range(10))
        mock_main_window.roi_states = {
            "roi_inc": {"select": False, "include": True, "exclude": False},
            "roi_exc": {"select": False, "include": False, "exclude": True},
        }
        mock_main_window.roi_intersection_cache = {
            "roi_inc": {0, 1, 2, 3, 4, 5},  # Keep only 0-5
            "roi_exc": {4, 5},  # Then remove 4, 5
        }

        roi_manager.apply_logic_filters()

        assert mock_main_window.visible_indices == {0, 1, 2, 3}

    def test_multiple_includes_intersect(self, roi_manager, mock_main_window):
        """Multiple include filters intersect (AND logic)."""
        mock_main_window.manual_visible_indices = set(range(10))
        mock_main_window.roi_states = {
            "roi_a": {"select": False, "include": True, "exclude": False},
            "roi_b": {"select": False, "include": True, "exclude": False},
        }
        mock_main_window.roi_intersection_cache = {
            "roi_a": {0, 1, 2, 3},
            "roi_b": {2, 3, 4, 5},
        }

        roi_manager.apply_logic_filters()

        # Intersection of {0,1,2,3} and {2,3,4,5} = {2,3}
        assert mock_main_window.visible_indices == {2, 3}

    def test_select_mode_does_not_filter(self, roi_manager, mock_main_window):
        """Select mode does not affect visibility."""
        mock_main_window.manual_visible_indices = set(range(10))
        mock_main_window.roi_states = {
            "roi_a": {"select": True, "include": False, "exclude": False}
        }
        mock_main_window.roi_intersection_cache = {
            "roi_a": {0, 1, 2}
        }

        roi_manager.apply_logic_filters()

        assert mock_main_window.visible_indices == set(range(10))

    def test_empty_include_cache_clears_all(
        self, roi_manager, mock_main_window
    ):
        """Include filter with empty cache results in no visible streamlines."""
        mock_main_window.manual_visible_indices = set(range(10))
        mock_main_window.roi_states = {
            "roi_a": {"select": False, "include": True, "exclude": False}
        }
        mock_main_window.roi_intersection_cache = {
            "roi_a": set()  # No intersections
        }

        roi_manager.apply_logic_filters()

        assert mock_main_window.visible_indices == set()

    def test_filters_do_not_mutate_manual_visible(
        self, roi_manager, mock_main_window
    ):
        """Filters should not modify manual_visible_indices."""
        original_manual = {0, 1, 2, 3, 4}
        mock_main_window.manual_visible_indices = original_manual.copy()
        mock_main_window.roi_states = {
            "roi_a": {"select": False, "include": False, "exclude": True}
        }
        mock_main_window.roi_intersection_cache = {
            "roi_a": {2, 3}
        }

        roi_manager.apply_logic_filters()

        assert mock_main_window.manual_visible_indices == original_manual


# ============================================================================
# Parcellation Filter Tests
# ============================================================================


class TestParcellationFilters:
    """Tests for parcellation region include/exclude filters."""

    def test_parcellation_include(self, roi_manager, mock_main_window):
        """Parcellation include filters streamlines by region."""
        mock_main_window.manual_visible_indices = set(range(10))
        mock_main_window.parcellation_region_states = {
            42: {"include": True, "exclude": False}
        }
        mock_main_window.parcellation_region_intersection_cache = {
            42: {0, 1, 2}
        }

        roi_manager.apply_logic_filters()

        assert mock_main_window.visible_indices == {0, 1, 2}

    def test_parcellation_exclude(self, roi_manager, mock_main_window):
        """Parcellation exclude removes streamlines from visibility."""
        mock_main_window.manual_visible_indices = set(range(10))
        mock_main_window.parcellation_region_states = {
            42: {"include": False, "exclude": True}
        }
        mock_main_window.parcellation_region_intersection_cache = {
            42: {0, 1, 2}
        }

        roi_manager.apply_logic_filters()

        expected = set(range(3, 10))
        assert mock_main_window.visible_indices == expected

    def test_combined_roi_and_parcellation_filters(
        self, roi_manager, mock_main_window
    ):
        """ROI and parcellation filters combine correctly."""
        mock_main_window.manual_visible_indices = set(range(10))
        mock_main_window.roi_states = {
            "roi_a": {"select": False, "include": True, "exclude": False}
        }
        mock_main_window.roi_intersection_cache = {
            "roi_a": {0, 1, 2, 3, 4, 5}
        }
        mock_main_window.parcellation_region_states = {
            42: {"include": False, "exclude": True}
        }
        mock_main_window.parcellation_region_intersection_cache = {
            42: {4, 5}
        }

        roi_manager.apply_logic_filters()

        # ROI include keeps 0-5, parcellation exclude removes 4,5
        assert mock_main_window.visible_indices == {0, 1, 2, 3}
