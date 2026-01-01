# -*- coding: utf-8 -*-
"""
Unit tests for tractedit_pkg/utils.py.

Tests utility functions, constants, and enums used throughout TractEdit.
"""

import pytest
import numpy as np
from datetime import datetime

from tractedit_pkg.utils import (
    ColorMode,
    get_formatted_datetime,
    get_asset_path,
    format_tuple,
    MAX_STACK_LEVELS,
    DEFAULT_SELECTION_RADIUS,
    MIN_SELECTION_RADIUS,
    RADIUS_INCREMENT,
    SLIDER_PRECISION,
    ROI_COLORS,
)


class TestColorModeEnum:
    """Tests for the ColorMode enumeration."""

    def test_color_mode_default_value(self):
        """Verify DEFAULT mode has expected integer value."""
        assert ColorMode.DEFAULT.value == 0

    def test_color_mode_orientation_value(self):
        """Verify ORIENTATION mode has expected integer value."""
        assert ColorMode.ORIENTATION.value == 1

    def test_color_mode_scalar_value(self):
        """Verify SCALAR mode has expected integer value."""
        assert ColorMode.SCALAR.value == 2

    def test_color_mode_count(self):
        """Verify all three color modes exist."""
        modes = list(ColorMode)
        assert len(modes) == 3

    def test_color_mode_comparison(self):
        """Test that enum members can be compared correctly."""
        assert ColorMode.DEFAULT != ColorMode.ORIENTATION
        assert ColorMode.DEFAULT == ColorMode.DEFAULT


class TestConstants:
    """Tests for module constants."""

    def test_max_stack_levels_positive(self):
        """MAX_STACK_LEVELS should be a positive integer."""
        assert isinstance(MAX_STACK_LEVELS, int)
        assert MAX_STACK_LEVELS > 0

    def test_default_selection_radius_reasonable(self):
        """DEFAULT_SELECTION_RADIUS should be within expected range."""
        assert isinstance(DEFAULT_SELECTION_RADIUS, float)
        assert 0.5 <= DEFAULT_SELECTION_RADIUS <= 50.0

    def test_min_selection_radius_positive(self):
        """MIN_SELECTION_RADIUS should be positive and less than default."""
        assert MIN_SELECTION_RADIUS > 0
        assert MIN_SELECTION_RADIUS <= DEFAULT_SELECTION_RADIUS

    def test_radius_increment_positive(self):
        """RADIUS_INCREMENT should be a small positive value."""
        assert RADIUS_INCREMENT > 0
        assert RADIUS_INCREMENT <= 5.0

    def test_slider_precision_reasonable(self):
        """SLIDER_PRECISION should provide sufficient granularity."""
        assert isinstance(SLIDER_PRECISION, int)
        assert SLIDER_PRECISION >= 100

    def test_roi_colors_format(self):
        """ROI_COLORS should contain valid RGB tuples."""
        assert len(ROI_COLORS) > 0
        for color in ROI_COLORS:
            assert len(color) == 3
            for component in color:
                assert 0.0 <= component <= 1.0


class TestGetFormattedDatetime:
    """Tests for the get_formatted_datetime function."""

    def test_returns_string(self):
        """Function should return a string."""
        result = get_formatted_datetime()
        assert isinstance(result, str)

    def test_contains_date_separators(self):
        """Result should contain date separators (/ or -)."""
        result = get_formatted_datetime()
        assert '/' in result or '-' in result

    def test_contains_time_separators(self):
        """Result should contain time separators (:)."""
        result = get_formatted_datetime()
        assert ':' in result

    def test_not_empty(self):
        """Result should not be empty."""
        result = get_formatted_datetime()
        assert len(result) > 0


class TestGetAssetPath:
    """Tests for the get_asset_path function."""

    def test_returns_string(self):
        """Function should return a string path."""
        result = get_asset_path("test_asset.png")
        assert isinstance(result, str)

    def test_contains_assets_directory(self):
        """Path should include 'assets' directory."""
        result = get_asset_path("test_asset.png")
        assert 'assets' in result

    def test_ends_with_asset_name(self):
        """Path should end with the requested asset name."""
        asset_name = "test_image.png"
        result = get_asset_path(asset_name)
        assert result.endswith(asset_name)


class TestFormatTuple:
    """Tests for the format_tuple function."""

    def test_simple_tuple(self):
        """Format a simple tuple with default precision."""
        result = format_tuple((1.0, 2.0, 3.0))
        assert result == "(1.00, 2.00, 3.00)"

    def test_custom_precision(self):
        """Format with custom precision."""
        result = format_tuple((1.5555, 2.6666), precision=3)
        assert result == "(1.556, 2.667)"

    def test_list_input(self):
        """Format a list (should work like tuple)."""
        result = format_tuple([1.0, 2.0])
        assert result == "(1.00, 2.00)"

    def test_integer_values(self):
        """Format integers as floats."""
        result = format_tuple((1, 2, 3))
        assert result == "(1.00, 2.00, 3.00)"

    def test_single_element(self):
        """Format single-element tuple."""
        result = format_tuple((5.0,))
        assert result == "(5.00)"

    def test_non_tuple_input(self):
        """Non-tuple input should return string representation."""
        result = format_tuple("not a tuple")
        assert result == "not a tuple"

    def test_numpy_array(self):
        """NumPy arrays are not tuples, should return str()."""
        arr = np.array([1.0, 2.0, 3.0])
        result = format_tuple(arr)
        # NumPy arrays don't pass isinstance(data, (list, tuple))
        assert isinstance(result, str)

    def test_empty_tuple(self):
        """Empty tuple should format correctly."""
        result = format_tuple(())
        assert result == "()"

    def test_negative_values(self):
        """Format negative values correctly."""
        result = format_tuple((-1.5, 2.5, -3.5))
        assert result == "(-1.50, 2.50, -3.50)"

    def test_zero_precision(self):
        """Format with zero precision (integers)."""
        result = format_tuple((1.6, 2.4), precision=0)
        assert result == "(2, 2)"
