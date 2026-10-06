"""Lightweight tractogram input checks shared by GUI and headless readers."""

from __future__ import annotations

from typing import Any

import numpy as np


def packed_streamline_arrays(streamlines: Any, *, allow_singleton: bool = False):
    """Validate ArraySequence layout without copying its coordinate buffer."""
    if not all(
        hasattr(streamlines, attribute)
        for attribute in ("_data", "_offsets", "_lengths")
    ):
        return None
    data = np.asarray(streamlines._data)
    offsets = np.asarray(streamlines._offsets)
    lengths = np.asarray(streamlines._lengths)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(
            f"Streamline coordinates must have shape (N, 3), got {data.shape}."
        )
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Streamline coordinates must be numeric.")
    if offsets.ndim != 1 or lengths.ndim != 1:
        raise ValueError("Streamline offsets and lengths must be one-dimensional.")
    if len(offsets) != len(lengths) or len(lengths) != len(streamlines):
        raise ValueError("Streamline offsets and lengths are inconsistent.")
    if len(lengths) == 0:
        raise ValueError("Tractogram is empty and contains no streamlines.")
    if not np.issubdtype(offsets.dtype, np.integer) or not np.issubdtype(
        lengths.dtype, np.integer
    ):
        raise ValueError("Streamline offsets and lengths must be integers.")
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if np.any(offsets < 0) or np.any(offsets > len(data)):
        raise ValueError("A streamline offset is outside the coordinate buffer.")
    if np.any(lengths < 0) or np.any(lengths > len(data) - offsets):
        raise ValueError("A streamline length exceeds its coordinate buffer.")
    ends = offsets + lengths
    if len(offsets) > 1 and np.any(offsets[1:] < ends[:-1]):
        raise ValueError("Streamline offsets overlap or are not ordered.")
    empty = np.flatnonzero(lengths == 0)
    if empty.size:
        raise ValueError(
            f"Tractogram contains an empty streamline at index {empty[0]}."
        )
    singleton = np.flatnonzero(lengths == 1)
    if singleton.size and not allow_singleton:
        raise ValueError(
            f"Tractogram contains a single-point streamline at index {singleton[0]}."
        )
    return data, offsets, lengths


def array_is_finite(array: np.ndarray, block_bytes: int = 4 * 1024 * 1024) -> bool:
    values = np.asarray(array)
    if not np.issubdtype(values.dtype, np.number):
        return False
    block_items = max(1, block_bytes // max(1, values.dtype.itemsize))
    iterator = np.nditer(
        values,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=block_items,
    )
    return all(np.all(np.isfinite(chunk)) for chunk in iterator)


def validate_tractogram_data(
    streamlines: Any,
    data_per_point: dict[str, Any] | None = None,
    data_per_streamline: dict[str, Any] | None = None,
    *,
    allow_singleton: bool = False,
) -> np.ndarray:
    """Reject malformed geometry and metadata before scientific publication."""
    packed = packed_streamline_arrays(streamlines, allow_singleton=allow_singleton)
    if packed is not None:
        data, _, lengths = packed
        if not array_is_finite(data):
            raise ValueError("Tractogram contains non-finite coordinates.")
    else:
        lengths_list = []
        for index, streamline in enumerate(streamlines):
            coordinates = np.asarray(streamline)
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError(
                    f"Streamline {index} must have shape (N, 3), "
                    f"got {coordinates.shape}."
                )
            if len(coordinates) == 0:
                raise ValueError(
                    f"Tractogram contains an empty streamline at index {index}."
                )
            if len(coordinates) == 1 and not allow_singleton:
                raise ValueError(
                    f"Tractogram contains a single-point streamline at index {index}."
                )
            if not array_is_finite(coordinates):
                raise ValueError(
                    f"Tractogram contains non-finite coordinates at index {index}."
                )
            lengths_list.append(len(coordinates))
        lengths = np.asarray(lengths_list, dtype=np.int64)
    if len(lengths) == 0:
        raise ValueError("Tractogram is empty and contains no streamlines.")
    for name, values in (data_per_point or {}).items():
        if not hasattr(values, "_lengths") and len(values) == int(lengths.sum()):
            if not array_is_finite(np.asarray(values)):
                raise ValueError(f"Point metadata {name!r} contains non-finite values.")
            continue
        if len(values) != len(lengths):
            raise ValueError(f"Point metadata {name!r} does not match streamlines.")
        packed_lengths = getattr(values, "_lengths", None)
        if packed_lengths is not None and not np.array_equal(packed_lengths, lengths):
            raise ValueError(
                f"Point metadata {name!r} does not match streamline lengths."
            )
        for index, item in enumerate(values):
            if len(item) != lengths[index] or not array_is_finite(np.asarray(item)):
                raise ValueError(
                    f"Point metadata {name!r} is malformed at index {index}."
                )
    for name, values in (data_per_streamline or {}).items():
        if len(values) != len(lengths):
            raise ValueError(
                f"Streamline metadata {name!r} does not match streamlines."
            )
        if not array_is_finite(np.asarray(values)):
            raise ValueError(
                f"Streamline metadata {name!r} contains non-finite values."
            )
    return lengths
