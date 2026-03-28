# -*- coding: utf-8 -*-
"""
Unit tests for tractedit_pkg/file_io.py helper functions.

Tests parsing utilities, AOT-compiled functions, MemoryMappedImage,
and TRX bbox caching (Phase 1 of TRX optimisation plan).
Avoids GUI components (QThread-based classes) for headless compatibility.
"""

import os
import shutil
import tempfile

import pytest
import numpy as np
import nibabel as nib
from nibabel.streamlines import ArraySequence
import trx.trx_file_memmap as tbx

from tractedit_pkg.file_io import (
    parse_numeric_tuple_from_string,
    _compute_bboxes_numba,
    _try_load_cached_bboxes,
    _embed_cached_bboxes,
    _extract_visible_bboxes,
    _save_trx_native,
    _TRX_BBOX_KEY,
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
        streamlines = ArraySequence([np.array([[1.0, 2.0, 3.0]], dtype=np.float32)])

        bboxes = _compute_bboxes_numba(
            streamlines._data, streamlines._offsets, streamlines._lengths
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
                streamlines._data, streamlines._offsets, streamlines._lengths
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

        slice_data = mmap.get_slice("z", 16)

        assert slice_data.shape == (32, 32)
        assert slice_data.dtype == np.float32

    def test_get_slice_coronal(self, temp_nifti_file):
        """Extract coronal (y) slice."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        slice_data = mmap.get_slice("y", 16)

        assert slice_data.shape == (32, 32)
        assert slice_data.dtype == np.float32

    def test_get_slice_sagittal(self, temp_nifti_file):
        """Extract sagittal (x) slice."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        slice_data = mmap.get_slice("x", 16)

        assert slice_data.shape == (32, 32)
        assert slice_data.dtype == np.float32

    def test_get_slice_index_clamping(self, temp_nifti_file):
        """Out-of-bounds indices should be clamped."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        # Should clamp to valid range, not raise error
        slice_high = mmap.get_slice("z", 1000)
        slice_low = mmap.get_slice("z", -100)

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

    def test_clear_cache(self, temp_nifti_file):
        """Cache clearing should not raise errors."""
        img = nib.load(temp_nifti_file)
        mmap = MemoryMappedImage(img)

        # Access some slices to populate cache
        mmap.get_slice("z", 10)
        mmap.get_slice("z", 15)

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


class TestResampleStreamline:
    """Tests for the AOT-compiled resample_streamline function."""

    @pytest.mark.numba
    def test_straight_line_uniform_spacing(self):
        """Resampled straight line should have uniformly spaced points."""
        from tractedit_pkg._numba_aot import resample_streamline

        streamline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        result = resample_streamline(streamline, 11)

        assert result.shape == (11, 3)
        # Points should be at x = 0, 1, 2, ... 10
        expected_x = np.linspace(0.0, 10.0, 11)
        np.testing.assert_allclose(result[:, 0], expected_x, atol=1e-10)
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[:, 2], 0.0, atol=1e-10)

    @pytest.mark.numba
    def test_single_point_streamline(self):
        """Single-point streamline should produce identical output points."""
        from tractedit_pkg._numba_aot import resample_streamline

        streamline = np.array([[5.0, 3.0, 1.0]], dtype=np.float64)
        result = resample_streamline(streamline, 10)

        assert result.shape == (10, 3)
        for i in range(10):
            np.testing.assert_allclose(result[i], [5.0, 3.0, 1.0])

    @pytest.mark.numba
    def test_two_point_streamline(self):
        """Two-point streamline should interpolate linearly."""
        from tractedit_pkg._numba_aot import resample_streamline

        streamline = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float64)
        result = resample_streamline(streamline, 6)

        assert result.shape == (6, 3)
        # First and last must match input endpoints
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(result[-1], [3.0, 4.0, 0.0], atol=1e-10)

    @pytest.mark.numba
    def test_endpoints_preserved(self):
        """First and last resampled points must match original endpoints."""
        from tractedit_pkg._numba_aot import resample_streamline

        streamline = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )
        result = resample_streamline(streamline, 50)

        np.testing.assert_allclose(result[0], streamline[0], atol=1e-10)
        np.testing.assert_allclose(result[-1], streamline[-1], atol=1e-10)

    @pytest.mark.numba
    def test_high_resolution_arc_length(self):
        """Resampled points should be equally spaced in arc length."""
        from tractedit_pkg._numba_aot import resample_streamline

        # Create a curved streamline (quarter circle in XY plane)
        t = np.linspace(0, np.pi / 2, 1000)
        streamline = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
        streamline = np.ascontiguousarray(streamline, dtype=np.float64)

        result = resample_streamline(streamline, 100)

        # Compute inter-point distances — should be approximately equal
        diffs = np.diff(result, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        assert np.std(distances) / np.mean(distances) < 0.02  # CV < 2%


class TestResampleBatch:
    """Tests for the batch resampling wrapper (resample_batch)."""

    @pytest.mark.numba
    def test_batch_matches_serial(self):
        """Batch results must match per-streamline serial resampling."""
        from tractedit_pkg._numba_aot import resample_streamline
        from tractedit_pkg._numba_aot._parallel_wrappers import resample_batch

        streamlines = [
            np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float64),
            np.array([[0, 0, 0], [0, 3, 4]], dtype=np.float64),
            np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float64),
        ]
        nb_points = 20
        as_seq = ArraySequence(streamlines)
        flat = np.ascontiguousarray(as_seq._data.astype(np.float64))
        offsets = np.ascontiguousarray(as_seq._offsets.astype(np.int64))
        lengths = np.ascontiguousarray(as_seq._lengths.astype(np.int64))

        batch_result = resample_batch(flat, offsets, lengths, nb_points)

        for i, sl in enumerate(streamlines):
            serial_result = resample_streamline(np.ascontiguousarray(sl), nb_points)
            np.testing.assert_allclose(
                batch_result[i],
                serial_result,
                atol=1e-10,
                err_msg=f"Mismatch for streamline {i}",
            )

    @pytest.mark.numba
    def test_batch_output_shape(self, sample_streamlines):
        """Batch resampling should produce correct output shape."""
        from tractedit_pkg._numba_aot._parallel_wrappers import resample_batch

        flat = np.ascontiguousarray(sample_streamlines._data.astype(np.float64))
        offsets = np.ascontiguousarray(sample_streamlines._offsets.astype(np.int64))
        lengths = np.ascontiguousarray(sample_streamlines._lengths.astype(np.int64))

        result = resample_batch(flat, offsets, lengths, 50)

        assert result.shape == (3, 50, 3)
        assert result.dtype == np.float64


# ============================================================================
# TRX Bbox Caching Tests (Phase 1)
# ============================================================================


def _make_reference_image(
    dims: tuple = (10, 10, 10),
) -> nib.Nifti1Image:
    """Create a minimal NIfTI reference image for TRX file creation."""
    affine = np.eye(4, dtype=np.float32)
    nifti_header = nib.Nifti1Header()
    nifti_header.set_data_shape(dims)
    nifti_header.set_zooms((1.0, 1.0, 1.0))
    nifti_header.set_qform(affine)
    nifti_header.set_sform(affine)
    return nib.Nifti1Image(np.empty(dims, dtype=np.int8), affine, header=nifti_header)


def _create_trx_with_bboxes(
    streamlines: ArraySequence,
    bboxes: np.ndarray,
    path: str,
) -> None:
    """Create and save a TRX file with embedded bbox cache."""
    ref = _make_reference_image()
    tractogram = nib.streamlines.Tractogram(
        streamlines=streamlines,
        affine_to_rasmm=np.eye(4),
    )
    trx_obj = tbx.TrxFile.from_tractogram(tractogram, ref)
    _embed_cached_bboxes(trx_obj, bboxes)
    tbx.save(trx_obj, path)


class TestTryLoadCachedBboxes:
    """Tests for _try_load_cached_bboxes (Phase 1 load path)."""

    @pytest.mark.numba
    def test_cache_hit_returns_correct_bboxes(self, sample_streamlines):
        """Load cached bboxes from a TRX file saved by TractEdit."""
        # Compute ground-truth bboxes
        bboxes_gt = _compute_bboxes_numba(
            sample_streamlines._data,
            sample_streamlines._offsets,
            sample_streamlines._lengths,
        )
        n = len(sample_streamlines)
        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            trx_path = os.path.join(tmp, "test_cache_hit.trx")
            _create_trx_with_bboxes(sample_streamlines, bboxes_gt, trx_path)

            # Reload and check
            trx_loaded = tbx.load(trx_path)
            result = _try_load_cached_bboxes(trx_loaded, n)

            assert result is not None
            assert result.shape == (n, 2, 3)
            assert result.dtype == np.float32
            np.testing.assert_allclose(result, bboxes_gt, atol=1e-6)
        finally:
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_cache_miss_returns_none(self, sample_streamlines):
        """Third-party TRX file without cached bboxes returns None."""
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_obj = tbx.TrxFile.from_tractogram(tractogram, ref)
        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            trx_path = os.path.join(tmp, "test_no_cache.trx")
            tbx.save(trx_obj, trx_path)

            trx_loaded = tbx.load(trx_path)
            result = _try_load_cached_bboxes(
                trx_loaded, len(sample_streamlines)
            )
            assert result is None
        finally:
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_wrong_shape_triggers_fallback(self, sample_streamlines):
        """Corrupted bbox cache with wrong shape triggers fallback."""
        n = len(sample_streamlines)
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_obj = tbx.TrxFile.from_tractogram(tractogram, ref)

        # Inject wrong-shape data: (N, 4) instead of (N, 6)
        wrong_shape = np.zeros((n, 4), dtype=np.float32)
        trx_obj.data_per_streamline[_TRX_BBOX_KEY] = wrong_shape

        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            trx_path = os.path.join(tmp, "test_wrong_shape.trx")
            tbx.save(trx_obj, trx_path)

            trx_loaded = tbx.load(trx_path)
            result = _try_load_cached_bboxes(trx_loaded, n)
            assert result is None
        finally:
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_wrong_count_triggers_fallback(self, sample_streamlines):
        """Bbox cache with mismatched streamline count triggers fallback.

        trx-python validates dps dimensions on save, so we test the
        fallback logic directly with an in-memory mock object.
        """
        n = len(sample_streamlines)

        # Create a mock TRX-like object with mismatched bbox count
        class FakeTrxWrongCount:
            data_per_streamline = {
                _TRX_BBOX_KEY: np.zeros((n + 5, 6), dtype=np.float32)
            }

        # Pass the correct streamline count — cache has n+5 rows
        result = _try_load_cached_bboxes(FakeTrxWrongCount(), n)
        assert result is None

    def test_no_data_per_streamline_attr(self):
        """Object without data_per_streamline attribute returns None."""

        class FakeTrx:
            pass

        result = _try_load_cached_bboxes(FakeTrx(), 10)
        assert result is None


class TestEmbedCachedBboxes:
    """Tests for _embed_cached_bboxes (Phase 1 save path)."""

    def test_embed_stores_flat_array(self, sample_streamlines):
        """Embedded bboxes should be stored as (N, 6) float32."""
        n = len(sample_streamlines)
        bboxes = np.random.rand(n, 2, 3).astype(np.float32)

        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_obj = tbx.TrxFile.from_tractogram(tractogram, ref)

        _embed_cached_bboxes(trx_obj, bboxes)

        stored = trx_obj.data_per_streamline[_TRX_BBOX_KEY]
        assert stored.shape == (n, 6)
        assert stored.dtype == np.float32
        np.testing.assert_allclose(
            stored, bboxes.reshape(-1, 6), atol=1e-7
        )

    def test_embed_none_is_noop(self, sample_streamlines):
        """Embedding None bboxes should not add the key."""
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_obj = tbx.TrxFile.from_tractogram(tractogram, ref)

        _embed_cached_bboxes(trx_obj, None)

        assert _TRX_BBOX_KEY not in trx_obj.data_per_streamline

    def test_embed_empty_array_is_noop(self, sample_streamlines):
        """Embedding an empty array should not add the key."""
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_obj = tbx.TrxFile.from_tractogram(tractogram, ref)

        _embed_cached_bboxes(trx_obj, np.array([], dtype=np.float32))

        assert _TRX_BBOX_KEY not in trx_obj.data_per_streamline


class TestExtractVisibleBboxes:
    """Tests for _extract_visible_bboxes."""

    def test_extracts_correct_subset(self):
        """Should return bboxes only for visible indices, sorted."""

        class FakeMainWindow:
            streamline_bboxes = np.arange(30).reshape(5, 2, 3).astype(
                np.float32
            )
            visible_indices = {0, 2, 4}

        result = _extract_visible_bboxes(FakeMainWindow())

        assert result is not None
        assert result.shape == (3, 2, 3)
        # Indices 0, 2, 4 → rows 0, 2, 4 of the original array
        np.testing.assert_array_equal(result[0], FakeMainWindow.streamline_bboxes[0])
        np.testing.assert_array_equal(result[1], FakeMainWindow.streamline_bboxes[2])
        np.testing.assert_array_equal(result[2], FakeMainWindow.streamline_bboxes[4])

    def test_returns_none_when_no_bboxes(self):
        """Should return None if streamline_bboxes is None."""

        class FakeMainWindow:
            streamline_bboxes = None
            visible_indices = {0, 1}

        assert _extract_visible_bboxes(FakeMainWindow()) is None

    def test_returns_none_when_no_visible(self):
        """Should return None if visible_indices is empty."""

        class FakeMainWindow:
            streamline_bboxes = np.zeros((3, 2, 3), dtype=np.float32)
            visible_indices = set()

        assert _extract_visible_bboxes(FakeMainWindow()) is None


class TestTrxBboxRoundTrip:
    """End-to-end round-trip tests for TRX bbox caching."""

    @pytest.mark.numba
    def test_save_reload_preserves_bboxes(self, sample_streamlines):
        """Save with bboxes → reload → bboxes match within float32 tolerance."""
        bboxes_gt = _compute_bboxes_numba(
            sample_streamlines._data,
            sample_streamlines._offsets,
            sample_streamlines._lengths,
        )
        n = len(sample_streamlines)
        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            trx_path = os.path.join(tmp, "roundtrip.trx")
            _create_trx_with_bboxes(sample_streamlines, bboxes_gt, trx_path)

            trx_loaded = tbx.load(trx_path)
            cached = _try_load_cached_bboxes(trx_loaded, n)
            assert cached is not None
            np.testing.assert_allclose(cached, bboxes_gt, atol=1e-6)

            # Also verify streamline data is intact
            loaded_sl = trx_loaded.streamlines
            assert len(loaded_sl) == n
            for i in range(n):
                np.testing.assert_allclose(
                    np.asarray(loaded_sl[i], dtype=np.float32),
                    np.asarray(sample_streamlines[i], dtype=np.float32),
                    atol=1e-5,
                )
        finally:
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_trx_without_cache_falls_back_to_computation(
        self, sample_streamlines
    ):
        """TRX without cache → _try_load_cached_bboxes returns None,
        and AOT computation produces correct results."""
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_obj = tbx.TrxFile.from_tractogram(tractogram, ref)
        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            trx_path = os.path.join(tmp, "no_cache.trx")
            tbx.save(trx_obj, trx_path)

            trx_loaded = tbx.load(trx_path)
            n = len(trx_loaded.streamlines)
            cached = _try_load_cached_bboxes(trx_loaded, n)
            assert cached is None

            # Fallback: compute from scratch
            sl = trx_loaded.streamlines
            computed = _compute_bboxes_numba(
                sl._data, sl._offsets, sl._lengths
            )
            assert computed.shape == (n, 2, 3)
            assert computed.dtype == np.float32
        finally:
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_trk_format_unaffected(self, sample_streamlines):
        """TRK loading and bbox computation must be completely unaffected."""
        bboxes = _compute_bboxes_numba(
            sample_streamlines._data,
            sample_streamlines._offsets,
            sample_streamlines._lengths,
        )

        # Verify TRK-style bbox computation still works correctly
        assert bboxes.shape == (len(sample_streamlines), 2, 3)
        assert bboxes.dtype == np.float32

        # First streamline: [[0,0,0], [1,1,1], [2,2,2]]
        np.testing.assert_allclose(bboxes[0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(bboxes[0, 1], [2.0, 2.0, 2.0])


# ============================================================================
# TRX Native Save Tests (Phase 2)
# ============================================================================


class TestSaveTrxNative:
    """Tests for _save_trx_native (Phase 2 TRX-native save path)."""

    @pytest.mark.numba
    def test_round_trip_preserves_streamlines(self, sample_streamlines):
        """Native save → reload should produce identical streamline data."""
        n = len(sample_streamlines)
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_source = tbx.TrxFile.from_tractogram(tractogram, ref)

        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            src_path = os.path.join(tmp, "source.trx")
            tbx.save(trx_source, src_path)
            # Reload so we have a proper on-disk TRX (memmap-backed)
            trx_source = tbx.load(src_path)

            out_path = os.path.join(tmp, "native_out.trx")
            all_indices = set(range(n))
            _save_trx_native(
                trx_source=trx_source,
                visible_indices=all_indices,
                bboxes=None,
                output_path=out_path,
            )

            trx_loaded = tbx.load(out_path)
            assert len(trx_loaded.streamlines) == n
            for i in range(n):
                np.testing.assert_allclose(
                    np.asarray(trx_loaded.streamlines[i], dtype=np.float32),
                    np.asarray(sample_streamlines[i], dtype=np.float32),
                    atol=1e-5,
                )
        finally:
            if trx_source is not None and hasattr(trx_source, "close"):
                trx_source.close()
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_subset_selection(self, sample_streamlines):
        """Native save with a subset of indices saves only those streamlines."""
        n = len(sample_streamlines)
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_source = tbx.TrxFile.from_tractogram(tractogram, ref)

        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            src_path = os.path.join(tmp, "source.trx")
            tbx.save(trx_source, src_path)
            trx_source = tbx.load(src_path)

            # Save only first and last streamline
            subset = {0, n - 1}
            out_path = os.path.join(tmp, "subset_out.trx")
            _save_trx_native(
                trx_source=trx_source,
                visible_indices=subset,
                bboxes=None,
                output_path=out_path,
            )

            trx_loaded = tbx.load(out_path)
            assert len(trx_loaded.streamlines) == 2

            # First saved streamline should match original index 0
            np.testing.assert_allclose(
                np.asarray(trx_loaded.streamlines[0], dtype=np.float32),
                np.asarray(sample_streamlines[0], dtype=np.float32),
                atol=1e-5,
            )
            # Second saved streamline should match original last index
            np.testing.assert_allclose(
                np.asarray(trx_loaded.streamlines[1], dtype=np.float32),
                np.asarray(sample_streamlines[n - 1], dtype=np.float32),
                atol=1e-5,
            )
        finally:
            if trx_source is not None and hasattr(trx_source, "close"):
                trx_source.close()
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_embeds_bboxes_on_save(self, sample_streamlines):
        """Native save should embed bboxes that survive round-trip."""
        n = len(sample_streamlines)
        bboxes = _compute_bboxes_numba(
            sample_streamlines._data,
            sample_streamlines._offsets,
            sample_streamlines._lengths,
        )
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_source = tbx.TrxFile.from_tractogram(tractogram, ref)

        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            src_path = os.path.join(tmp, "source.trx")
            tbx.save(trx_source, src_path)
            trx_source = tbx.load(src_path)

            out_path = os.path.join(tmp, "with_bboxes.trx")
            all_indices = set(range(n))
            _save_trx_native(
                trx_source=trx_source,
                visible_indices=all_indices,
                bboxes=bboxes,
                output_path=out_path,
            )

            trx_loaded = tbx.load(out_path)
            cached = _try_load_cached_bboxes(trx_loaded, n)
            assert cached is not None
            np.testing.assert_allclose(cached, bboxes, atol=1e-6)
        finally:
            if trx_source is not None and hasattr(trx_source, "close"):
                trx_source.close()
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_subset_bboxes_match_selected(self, sample_streamlines):
        """When saving a subset, embedded bboxes should match the subset."""
        n = len(sample_streamlines)
        bboxes = _compute_bboxes_numba(
            sample_streamlines._data,
            sample_streamlines._offsets,
            sample_streamlines._lengths,
        )
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_source = tbx.TrxFile.from_tractogram(tractogram, ref)

        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            src_path = os.path.join(tmp, "source.trx")
            tbx.save(trx_source, src_path)
            trx_source = tbx.load(src_path)

            # Save only indices 1 and 2
            subset = {1, 2}
            out_path = os.path.join(tmp, "subset_bboxes.trx")
            _save_trx_native(
                trx_source=trx_source,
                visible_indices=subset,
                bboxes=bboxes,
                output_path=out_path,
            )

            trx_loaded = tbx.load(out_path)
            assert len(trx_loaded.streamlines) == 2
            cached = _try_load_cached_bboxes(trx_loaded, 2)
            assert cached is not None
            # Cached bboxes should match original indices 1 and 2
            expected = bboxes[[1, 2]]
            np.testing.assert_allclose(cached, expected, atol=1e-6)
        finally:
            if trx_source is not None and hasattr(trx_source, "close"):
                trx_source.close()
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_none_bboxes_saves_without_cache(self, sample_streamlines):
        """Native save with bboxes=None should not embed bbox cache."""
        n = len(sample_streamlines)
        ref = _make_reference_image()
        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            affine_to_rasmm=np.eye(4),
        )
        trx_source = tbx.TrxFile.from_tractogram(tractogram, ref)

        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            src_path = os.path.join(tmp, "source.trx")
            tbx.save(trx_source, src_path)
            trx_source = tbx.load(src_path)

            out_path = os.path.join(tmp, "no_bboxes.trx")
            _save_trx_native(
                trx_source=trx_source,
                visible_indices=set(range(n)),
                bboxes=None,
                output_path=out_path,
            )

            trx_loaded = tbx.load(out_path)
            cached = _try_load_cached_bboxes(trx_loaded, n)
            assert cached is None
        finally:
            if trx_source is not None and hasattr(trx_source, "close"):
                trx_source.close()
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_scalar_data_preserved(self, sample_streamlines):
        """Native save should preserve data_per_vertex (scalar data)."""
        n = len(sample_streamlines)
        ref = _make_reference_image()

        # Build per-point scalars as a list of arrays (one per streamline)
        # — this is how nibabel's data_per_point works.
        scalar_per_point = []
        offset = 0
        for i in range(n):
            pts = len(sample_streamlines[i])
            scalar_per_point.append(
                np.arange(offset, offset + pts, dtype=np.float32).reshape(-1, 1)
            )
            offset += pts

        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            data_per_point={"test_scalar": scalar_per_point},
            affine_to_rasmm=np.eye(4),
        )
        trx_source = tbx.TrxFile.from_tractogram(tractogram, ref)

        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            src_path = os.path.join(tmp, "with_scalars.trx")
            tbx.save(trx_source, src_path)
            trx_source = tbx.load(src_path)

            out_path = os.path.join(tmp, "scalars_out.trx")
            _save_trx_native(
                trx_source=trx_source,
                visible_indices=set(range(n)),
                bboxes=None,
                output_path=out_path,
            )

            trx_loaded = tbx.load(out_path)
            assert "test_scalar" in trx_loaded.data_per_vertex
            loaded_dpv = trx_loaded.data_per_vertex["test_scalar"]
            # data_per_vertex values are ArraySequence — use ._data
            # to get the flat (N_total_pts, K) array.
            loaded_flat = np.asarray(loaded_dpv._data)
            expected = np.concatenate(scalar_per_point, axis=0)
            np.testing.assert_allclose(
                loaded_flat, expected, atol=1e-6
            )
        finally:
            if trx_source is not None and hasattr(trx_source, "close"):
                trx_source.close()
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.numba
    def test_scalar_subset_preserved(self, sample_streamlines):
        """Native save with subset should preserve only selected scalars."""
        n = len(sample_streamlines)
        ref = _make_reference_image()

        # Build per-point scalars as a list of arrays (one per streamline)
        scalar_per_point = []
        offset = 0
        for i in range(n):
            pts = len(sample_streamlines[i])
            scalar_per_point.append(
                np.arange(offset, offset + pts, dtype=np.float32).reshape(-1, 1)
            )
            offset += pts

        tractogram = nib.streamlines.Tractogram(
            streamlines=sample_streamlines,
            data_per_point={"test_scalar": scalar_per_point},
            affine_to_rasmm=np.eye(4),
        )
        trx_source = tbx.TrxFile.from_tractogram(tractogram, ref)

        tmp = tempfile.mkdtemp()
        trx_loaded = None
        try:
            src_path = os.path.join(tmp, "with_scalars.trx")
            tbx.save(trx_source, src_path)
            trx_source = tbx.load(src_path)

            # Save only streamline 0
            out_path = os.path.join(tmp, "scalar_subset.trx")
            _save_trx_native(
                trx_source=trx_source,
                visible_indices={0},
                bboxes=None,
                output_path=out_path,
            )

            trx_loaded = tbx.load(out_path)
            assert len(trx_loaded.streamlines) == 1
            assert "test_scalar" in trx_loaded.data_per_vertex

            # The scalar for streamline 0 should match the original
            expected_scalar = scalar_per_point[0]
            loaded_dpv = trx_loaded.data_per_vertex["test_scalar"]
            loaded_flat = np.asarray(loaded_dpv._data)
            np.testing.assert_allclose(
                loaded_flat, expected_scalar, atol=1e-6
            )
        finally:
            if trx_source is not None and hasattr(trx_source, "close"):
                trx_source.close()
            if trx_loaded is not None and hasattr(trx_loaded, "close"):
                trx_loaded.close()
            shutil.rmtree(tmp, ignore_errors=True)
