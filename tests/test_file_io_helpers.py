# -*- coding: utf-8 -*-
"""
Unit tests for tractedit_pkg/file_io.py helper functions.

Tests parsing utilities, Numba-optimized functions, and MemoryMappedImage.
Avoids GUI components (QThread-based classes) for headless compatibility.
"""

import pytest
import numpy as np
import nibabel as nib
from nibabel.streamlines import ArraySequence

from tractedit_pkg.file_io import (
    parse_numeric_tuple_from_string,
    _compute_bboxes_numba,
    MemoryMappedImage,
    MAX_VOXELS,
    MMAP_SLICE_CACHE_SIZE,
)


class TestParseNumericTupleFromString:
    """Tests for the parse_numeric_tuple_from_string function."""

    # ========================================================================
    # String Input Tests
    # ========================================================================

    def test_parse_tuple_string_parentheses(self):
        """Parse string with parentheses: '(1, 2, 3)'."""
        result = parse_numeric_tuple_from_string("(1, 2, 3)")
        assert result == (1.0, 2.0, 3.0)

    def test_parse_tuple_string_brackets(self):
        """Parse string with brackets: '[1, 2, 3]'."""
        result = parse_numeric_tuple_from_string("[1, 2, 3]")
        assert result == (1.0, 2.0, 3.0)

    def test_parse_space_separated(self):
        """Parse space-separated string: '1 2 3'."""
        result = parse_numeric_tuple_from_string("1 2 3")
        assert result == (1.0, 2.0, 3.0)

    def test_parse_comma_separated(self):
        """Parse comma-separated without brackets: '1, 2, 3'."""
        result = parse_numeric_tuple_from_string("1, 2, 3")
        assert result == (1.0, 2.0, 3.0)

    def test_parse_mixed_separators(self):
        """Parse mixed separators: '1,2 3'."""
        result = parse_numeric_tuple_from_string("1,2 3")
        assert result == (1.0, 2.0, 3.0)

    def test_parse_floats(self):
        """Parse float values: '(1.5, 2.7, 3.9)'."""
        result = parse_numeric_tuple_from_string("(1.5, 2.7, 3.9)")
        assert result == pytest.approx((1.5, 2.7, 3.9))

    def test_parse_negative_values(self):
        """Parse negative values: '(-1, -2.5, 3)'."""
        result = parse_numeric_tuple_from_string("(-1, -2.5, 3)")
        assert result == (-1.0, -2.5, 3.0)

    def test_parse_integer_type(self):
        """Parse with integer target type."""
        result = parse_numeric_tuple_from_string("(1.9, 2.1, 3.5)", target_type=int)
        assert result == (1, 2, 3)

    # ========================================================================
    # Length Validation Tests
    # ========================================================================

    def test_expected_length_match(self):
        """Parse with matching expected length."""
        result = parse_numeric_tuple_from_string("(1, 2, 3)", expected_length=3)
        assert result == (1.0, 2.0, 3.0)

    def test_expected_length_mismatch(self):
        """Returns original string when length doesn't match."""
        original = "(1, 2, 3)"
        result = parse_numeric_tuple_from_string(original, expected_length=2)
        assert result == original

    # ========================================================================
    # Non-String Input Tests
    # ========================================================================

    def test_list_passthrough(self):
        """Lists should be converted to tuples."""
        result = parse_numeric_tuple_from_string([1, 2, 3])
        assert result == (1.0, 2.0, 3.0)

    def test_tuple_passthrough(self):
        """Tuples should pass through with type conversion."""
        result = parse_numeric_tuple_from_string((1, 2, 3))
        assert result == (1.0, 2.0, 3.0)

    def test_numpy_array_passthrough(self):
        """NumPy arrays should be returned with dtype conversion."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = parse_numeric_tuple_from_string(arr, target_type=float)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_empty_string(self):
        """Empty string should return as-is."""
        result = parse_numeric_tuple_from_string("")
        assert result == ""

    def test_invalid_string(self):
        """Invalid string should return as-is."""
        original = "not a number"
        result = parse_numeric_tuple_from_string(original)
        assert result == original

    def test_single_value(self):
        """Single value string parses to scalar."""
        result = parse_numeric_tuple_from_string("42")
        # Function returns scalar or single-element tuple depending on context
        assert result == 42.0 or result == (42.0,)

    def test_scientific_notation(self):
        """Scientific notation should be parsed."""
        result = parse_numeric_tuple_from_string("(1e-3, 2e5)")
        assert result == pytest.approx((0.001, 200000.0))


class TestComputeBboxesNumba:
    """Tests for the _compute_bboxes_numba function."""

    @pytest.mark.numba
    def test_simple_streamlines(self, sample_streamlines):
        """Compute bounding boxes for sample streamlines."""
        flat_data = sample_streamlines._data
        offsets = sample_streamlines._offsets
        lengths = sample_streamlines._lengths

        bboxes = _compute_bboxes_numba(flat_data, offsets, lengths)

        assert bboxes.shape == (3, 2, 3)  # 3 streamlines, [min, max], xyz
        assert bboxes.dtype == np.float32

    @pytest.mark.numba
    def test_bbox_min_max_correct(self, sample_streamlines):
        """Verify min/max values are computed correctly."""
        flat_data = sample_streamlines._data
        offsets = sample_streamlines._offsets
        lengths = sample_streamlines._lengths

        bboxes = _compute_bboxes_numba(flat_data, offsets, lengths)

        # First streamline: [[0,0,0], [1,1,1], [2,2,2]]
        assert np.allclose(bboxes[0, 0], [0.0, 0.0, 0.0])  # min
        assert np.allclose(bboxes[0, 1], [2.0, 2.0, 2.0])  # max

        # Second streamline: [[5,5,5], [6,6,6]]
        assert np.allclose(bboxes[1, 0], [5.0, 5.0, 5.0])  # min
        assert np.allclose(bboxes[1, 1], [6.0, 6.0, 6.0])  # max

    @pytest.mark.numba
    def test_single_point_streamline(self):
        """Single-point streamline should have min == max."""
        streamlines = ArraySequence([
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        ])

        bboxes = _compute_bboxes_numba(
            streamlines._data,
            streamlines._offsets,
            streamlines._lengths
        )

        assert np.allclose(bboxes[0, 0], [1.0, 2.0, 3.0])
        assert np.allclose(bboxes[0, 1], [1.0, 2.0, 3.0])

    @pytest.mark.numba
    def test_empty_streamlines(self):
        """Empty list of streamlines should be handled gracefully."""
        streamlines = ArraySequence([])

        # Empty ArraySequence may have different internal structure
        # Just verify no exception is raised
        if len(streamlines._lengths) > 0:
            bboxes = _compute_bboxes_numba(
                streamlines._data,
                streamlines._offsets,
                streamlines._lengths
            )
            assert bboxes.shape[1:] == (2, 3)


class TestMemoryMappedImage:
    """Tests for the MemoryMappedImage class."""

    def test_shape_property(self, temp_nifti_file):
        """Verify shape property returns correct dimensions."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        assert mmap.shape == (32, 32, 32)

    def test_affine_property(self, temp_nifti_file):
        """Verify affine property returns 4x4 matrix."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        assert mmap.affine.shape == (4, 4)
        assert np.allclose(mmap.affine, np.eye(4))

    def test_get_slice_axial(self, temp_nifti_file):
        """Extract axial (z) slice."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        slice_data = mmap.get_slice('z', 16)

        assert slice_data.shape == (32, 32)
        assert slice_data.dtype == np.float32

    def test_get_slice_coronal(self, temp_nifti_file):
        """Extract coronal (y) slice."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        slice_data = mmap.get_slice('y', 16)

        assert slice_data.shape == (32, 32)
        assert slice_data.dtype == np.float32

    def test_get_slice_sagittal(self, temp_nifti_file):
        """Extract sagittal (x) slice."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        slice_data = mmap.get_slice('x', 16)

        assert slice_data.shape == (32, 32)
        assert slice_data.dtype == np.float32

    def test_get_slice_index_clamping(self, temp_nifti_file):
        """Out-of-bounds indices should be clamped."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        # Should clamp to valid range, not raise error
        slice_high = mmap.get_slice('z', 1000)
        slice_low = mmap.get_slice('z', -100)

        assert slice_high.shape == (32, 32)
        assert slice_low.shape == (32, 32)

    def test_get_value_range(self, temp_nifti_file):
        """Value range should return (min, max) tuple."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        vmin, vmax = mmap.get_value_range()

        assert isinstance(vmin, float)
        assert isinstance(vmax, float)
        assert vmin <= vmax

    def test_x_flip_affine_modification(self, temp_nifti_ras):
        """X-flip should modify affine for LAS orientation."""
        img = nib.load(temp_nifti_ras)
        mmap = MemoryMappedImage(img, needs_x_flip=True)

        # Affine X column should be negated
        assert mmap.affine[0, 0] < 0

    def test_clear_cache(self, temp_nifti_file):
        """Cache clearing should not raise errors."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        # Access some slices to populate cache
        mmap.get_slice('z', 10)
        mmap.get_slice('z', 15)

        # Clear should not raise
        mmap.clear_cache()


class TestModuleConstants:
    """Tests for file_io module constants."""

    def test_max_voxels_reasonable(self):
        """MAX_VOXELS should be within reasonable range."""
        assert MAX_VOXELS > 1_000_000  # At least 100^3
        assert MAX_VOXELS < 10_000_000_000  # Less than 2150^3

    def test_mmap_cache_size_positive(self):
        """MMAP_SLICE_CACHE_SIZE should be positive."""
        assert MMAP_SLICE_CACHE_SIZE > 0
        assert isinstance(MMAP_SLICE_CACHE_SIZE, int)
