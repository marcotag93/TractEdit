"""Regression tests for RGB scaling and lazy volume selection."""

import nibabel as nib
import numpy as np
import pytest

from tractedit_pkg.image_volume import rgb_display_values, select_image_volume


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0.0, 0.5, 1.0], [0, 128, 255]),
        ([-0.01, 0.5, 1.0], [0, 128, 255]),
        ([0.0, 0.5, 1.01], [0, 128, 255]),
        ([-10.0, 128.0, 300.0], [0, 128, 255]),
        ([np.nan, np.inf, -np.inf], [0, 255, 0]),
        ([0, 1, 0], [0, 255, 0]),
    ],
)
def test_rgb_scaling(values, expected):
    source = np.array(values)
    original = source.copy()
    np.testing.assert_array_equal(rgb_display_values(source), expected)
    np.testing.assert_array_equal(source, original)


def test_volume_selection_defers_and_limits_reads():
    data = np.arange(6 * 8 * 10 * 4, dtype=np.int16).reshape(6, 8, 10, 4)

    class RecordingProxy:
        shape = data.shape

        def __init__(self):
            self.reads = []

        def __getitem__(self, key):
            self.reads.append(key)
            return data[key]

    proxy = RecordingProxy()
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    image = nib.Nifti1Image(proxy, affine, nib.Nifti1Header())
    selected = select_image_volume(image, 2)
    selected = nib.as_closest_canonical(selected)
    assert proxy.reads == []
    sampled = selected.dataobj[::2, ::2, ::2]
    assert proxy.reads == [(slice(None, None, 2),) * 3 + (2,)]
    np.testing.assert_array_equal(sampled, data[::2, ::2, ::2, 2])
    np.testing.assert_array_equal(selected.affine, affine)
    np.testing.assert_array_equal(selected.get_fdata(), data[..., 2])
    assert selected.shape == data.shape[:3]
    assert image.shape == data.shape


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_selected_volume_preserves_scaling_and_orientation(tmp_path, suffix):
    data = np.arange(3 * 4 * 5 * 2, dtype=np.int16).reshape(3, 4, 5, 2)
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    image = nib.Nifti1Image(data, affine)
    image.header.set_slope_inter(2.0, 10.0)
    path = tmp_path / f"series{suffix}"
    nib.save(image, path)
    source = nib.load(path)
    selected = nib.as_closest_canonical(select_image_volume(source, 1))
    expected = nib.as_closest_canonical(source.slicer[..., 1])
    np.testing.assert_array_equal(selected.get_fdata(), expected.get_fdata())
    np.testing.assert_array_equal(selected.affine, expected.affine)
