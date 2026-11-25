"""Explicit display selection for scalar NIfTI series and RGB images."""

import numpy as np
from nibabel.fileslice import canonical_slicers

from .reference_grid import validate_affine


class _VolumeProxy:
    """Expose one volume while forwarding spatial slices to the source."""

    def __init__(self, source, volume_index):
        self._source = source
        self._volume_index = volume_index
        self.shape = source.shape[:3]

    def __getitem__(self, key):
        spatial_key = canonical_slicers(key, self.shape)
        return self._source[spatial_key + (self._volume_index,)]

    def __array__(self, dtype=None, copy=None):
        values = np.asarray(self[...], dtype=dtype)
        return values.copy() if copy else values


def validate_display_selection(selection):
    if (
        not isinstance(selection, dict)
        or set(selection) != {"volume_index", "rgb"}
        or type(selection["volume_index"]) is not int
        or selection["volume_index"] < 0
        or type(selection["rgb"]) is not bool
    ):
        raise ValueError("Invalid anatomical volume selection.")


def select_image_volume(image, volume_index=0, rgb=False):
    """Select a volume without reading its voxel data."""
    validate_display_selection({"volume_index": volume_index, "rgb": rgb})
    validate_affine(image.affine, "Anatomical image")
    shape = image.shape
    if len(shape) not in (3, 4) or any(size < 1 for size in shape):
        raise ValueError("Anatomical images must have 3 or 4 non-empty dimensions.")
    count = shape[3] if len(shape) == 4 else 1
    if volume_index >= count:
        raise ValueError(f"Volume index {volume_index} is outside {count} volumes.")
    if rgb:
        if len(shape) != 4 or count != 3:
            raise ValueError("RGB display requires exactly three components.")
        return image
    if len(shape) == 3:
        return image
    return image.__class__(
        _VolumeProxy(image.dataobj, volume_index), image.affine, image.header
    )


def rgb_display_values(data):
    """Map normalized or byte-range RGB to display bytes without editing data."""
    values = np.asarray(data)
    finite = values[np.isfinite(values)]
    # Allow small overshoots in normalized floating-point images.
    normalized = (
        finite.size
        and finite.max() <= (
            1.05 if np.issubdtype(values.dtype, np.floating) else 1
        )
    )
    scale = 255.0 if normalized else 1.0
    return np.rint(
        np.clip(np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0) * scale, 0, 255)
    ).astype(np.uint8)
