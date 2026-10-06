# -*- coding: utf-8 -*-
"""Tractogram metadata capability, validation, and VTK conversion helpers."""

import logging
from typing import Any, Dict, Mapping, Tuple

import nibabel as nib
import numpy as np


logger = logging.getLogger(__name__)


FORMAT_METADATA_CAPABILITIES = {
    ".trk": frozenset({"data_per_point", "data_per_streamline"}),
    ".tck": frozenset(),
    ".trx": frozenset(
        {"data_per_point", "data_per_streamline", "groups", "data_per_group"}
    ),
    ".vtk": frozenset({"data_per_point", "data_per_streamline"}),
    ".vtp": frozenset({"data_per_point", "data_per_streamline"}),
}


def ensure_metadata_supported(
    extension: str,
    data_per_point: Mapping[str, Any] | None = None,
    data_per_streamline: Mapping[str, Any] | None = None,
    groups: Mapping[str, Any] | None = None,
    data_per_group: Mapping[str, Any] | None = None,
) -> None:
    """Reject metadata that the destination format cannot represent."""
    try:
        capabilities = FORMAT_METADATA_CAPABILITIES[extension.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported tractogram extension: {extension}") from exc

    payloads = {
        "data_per_point": data_per_point,
        "data_per_streamline": data_per_streamline,
        "groups": groups,
        "data_per_group": data_per_group,
    }
    unsupported = [
        name
        for name, payload in payloads.items()
        if payload is not None and len(payload) > 0 and name not in capabilities
    ]
    if unsupported:
        format_name = extension.removeprefix(".").upper()
        fields = ", ".join(unsupported)
        raise ValueError(
            f"{format_name} format cannot preserve metadata fields: {fields}"
        )


def _numeric_vtk_arrays(
    attributes: Any,
    expected_tuples: int,
    association: str,
) -> Dict[str, np.ndarray]:
    from vtk.util import numpy_support

    arrays = {}
    for index in range(attributes.GetNumberOfArrays()):
        fallback_name = "Scalar" if association == "point" else "Property"
        name = attributes.GetArrayName(index) or f"{fallback_name}_{index}"
        vtk_array = attributes.GetArray(index)
        if vtk_array is None:
            logger.warning(
                "Skipping unsupported nonnumeric VTK %s-data array '%s'.",
                association,
                name,
            )
            continue
        if vtk_array.GetNumberOfTuples() != expected_tuples:
            raise ValueError(
                f"VTK {association}-data array '{name}' has "
                f"{vtk_array.GetNumberOfTuples()} tuples; expected {expected_tuples}."
            )
        if name in arrays:
            raise ValueError(f"Duplicate VTK {association}-data array name: {name}")
        arrays[name] = numpy_support.vtk_to_numpy(vtk_array)
    return arrays


def _array_sequence_from_flat_data(
    data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> nib.streamlines.ArraySequence:
    sequence = nib.streamlines.ArraySequence()
    sequence._data = data
    sequence._offsets = offsets
    sequence._lengths = lengths
    return sequence


def extract_vtk_metadata(
    poly_data: Any,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> Tuple[
    Dict[str, nib.streamlines.ArraySequence],
    Dict[str, np.ndarray],
]:
    """Extract numeric VTK point and line-cell arrays in streamline order."""
    point_arrays = _numeric_vtk_arrays(
        poly_data.GetPointData(), poly_data.GetNumberOfPoints(), "point"
    )
    streamline_offsets = np.asarray(offsets[:-1], dtype=np.intp).copy()
    streamline_lengths = np.asarray(lengths, dtype=np.intp).copy()
    data_per_point = {
        name: _array_sequence_from_flat_data(
            values[connectivity], streamline_offsets, streamline_lengths
        )
        for name, values in point_arrays.items()
    }

    cell_arrays = _numeric_vtk_arrays(
        poly_data.GetCellData(), poly_data.GetNumberOfCells(), "cell"
    )
    line_start = poly_data.GetNumberOfVerts()
    line_stop = line_start + poly_data.GetNumberOfLines()
    data_per_streamline = {
        name: values[line_start:line_stop].copy()
        for name, values in cell_arrays.items()
    }
    return data_per_point, data_per_streamline


def _as_numeric_vtk_array(values: Any, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim not in (1, 2):
        raise ValueError(
            f"Metadata field '{field_name}' must have one or two dimensions."
        )
    if array.dtype.kind not in "iuf":
        raise ValueError(f"Metadata field '{field_name}' must be numeric.")
    return np.ascontiguousarray(array)


def add_vtk_metadata(
    poly_data: Any,
    tractogram: nib.streamlines.Tractogram,
) -> None:
    """Attach per-point and per-streamline metadata to VTK polydata."""
    if not tractogram.data_per_point and not tractogram.data_per_streamline:
        return

    from vtk.util import numpy_support

    streamlines = tractogram.streamlines
    if hasattr(streamlines, "_lengths"):
        lengths = np.asarray(streamlines._lengths)
    else:
        lengths = np.asarray([len(streamline) for streamline in streamlines])
    total_points = int(lengths.sum())
    streamline_count = len(lengths)

    for name, sequence in tractogram.data_per_point.items():
        if len(sequence) != streamline_count:
            raise ValueError(
                f"Per-point metadata field '{name}' has {len(sequence)} items; "
                f"expected {streamline_count}."
            )
        if hasattr(sequence, "_data") and hasattr(sequence, "_lengths"):
            if not np.array_equal(sequence._lengths, lengths):
                raise ValueError(
                    f"Per-point metadata field '{name}' does not match point counts."
                )
            flat = sequence._data
        else:
            values = [np.asarray(item) for item in sequence]
            if any(
                len(item) != length
                for item, length in zip(values, lengths, strict=True)
            ):
                raise ValueError(
                    f"Per-point metadata field '{name}' does not match point counts."
                )
            flat = (
                np.concatenate(values, axis=0)
                if values
                else np.empty((0,), dtype=np.float32)
            )
        flat = _as_numeric_vtk_array(flat, name)
        if len(flat) != total_points:
            raise ValueError(
                f"Per-point metadata field '{name}' has {len(flat)} tuples; "
                f"expected {total_points}."
            )
        vtk_array = numpy_support.numpy_to_vtk(flat, deep=True)
        vtk_array.SetName(name)
        poly_data.GetPointData().AddArray(vtk_array)

    for name, values in tractogram.data_per_streamline.items():
        array = _as_numeric_vtk_array(values, name)
        if len(array) != streamline_count:
            raise ValueError(
                f"Per-streamline metadata field '{name}' has {len(array)} items; "
                f"expected {streamline_count}."
            )
        vtk_array = numpy_support.numpy_to_vtk(array, deep=True)
        vtk_array.SetName(name)
        poly_data.GetCellData().AddArray(vtk_array)
