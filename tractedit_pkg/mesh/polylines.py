# -*- coding: utf-8 -*-
"""Ordered annotations in scanner-RAS millimetres and line exports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ..transactional_io import staged_output

if TYPE_CHECKING:
    import vtk
    from ..reference_grid import ReferenceGrid

MAX_POLYLINES = 10000
MAX_POLYLINE_POINTS = 1000000
DEFAULT_POLYLINE_COLOR = (1.0, 0.75, 0.0)
DEFAULT_LINE_WIDTH = 3.0
DEFAULT_POINT_SIZE = 7.0
POLYLINE_STYLE_FIELDS = {"line_width", "point_size", "show_points"}


def validated_style(
    line_width=DEFAULT_LINE_WIDTH, point_size=DEFAULT_POINT_SIZE, show_points=True,
) -> dict:
    for name, value, maximum in (
        ("line_width", line_width, 10), ("point_size", point_size, 20),
    ):
        if (
            type(value) not in (int, float)
            or not np.isfinite(value)
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"Polyline {name} must be finite and in [1, {maximum}]")
    if type(show_points) is not bool:
        raise ValueError("Polyline show_points must be boolean")
    return {
        "line_width": float(line_width), "point_size": float(point_size),
        "show_points": show_points,
    }


def validated_name(name) -> str:
    if not isinstance(name, str) or not name.strip() or len(name) > 4096:
        raise ValueError("Invalid polyline name")
    return name


def normalize_legacy_polylines(records) -> list[dict]:
    validate_polylines(records, legacy=True)
    return [
        dict(
            record, points=validated_points(record["points"]),
            **validated_style(show_points=False),
        ) for record in records
    ]


def validated_points(points, *, allow_duplicate_adjacent=False) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1:] != (3,)
        or not 2 <= len(values) <= MAX_POLYLINE_POINTS
        or not np.isfinite(values).all()
        or (not allow_duplicate_adjacent and np.any(np.all(values[1:] == values[:-1], axis=1)))
    ):
        raise ValueError("A polyline needs distinct adjacent finite 3D vertices")
    return np.ascontiguousarray(values)


def validated_color(color) -> tuple[float, float, float]:
    values = np.asarray(color, dtype=np.float64)
    if values.shape != (3,) or not np.all(
        np.isfinite(values) & (values >= 0) & (values <= 1)
    ):
        raise ValueError("Polyline color must contain three values in [0, 1]")
    return tuple(float(value) for value in values)


def validate_polylines(records, *, legacy=False) -> None:
    if not isinstance(records, list) or len(records) > MAX_POLYLINES:
        raise ValueError("Invalid polyline count")
    identities = set()
    total = 0
    for record in records:
        extra = set()
        if isinstance(record, dict) and "region" in record and not legacy:
            region = record["region"]
            if (not isinstance(region, dict)
                    or set(region) - {"reference_space"} != {"faces", "loop", "digest"}
                    or ("reference_space" in region and (
                        not isinstance(region["reference_space"], str)
                        or len(region["reference_space"]) > 4096
                        or not region["reference_space"].isprintable()
                    ))):
                raise ValueError("Invalid surface polygon fields")
            faces, loop = region["faces"], region["loop"]
            for ids in (faces, loop):
                if (not isinstance(ids, np.ndarray) or ids.ndim != 1
                        or ids.dtype.kind not in "iu" or not ids.size or ids.min() < 0):
                    raise ValueError("Invalid surface polygon indices")
            if (not isinstance(region["digest"], str) or len(region["digest"]) != 64
                    or len(np.unique(faces)) != len(faces) or len(loop) < 4
                    or loop[0] != loop[-1] or len(np.unique(loop)) != len(loop)-1
                    or len(loop) != len(record.get("points", ()))):
                raise ValueError("Invalid surface polygon boundary or source")
            extra.add("region")
        if not isinstance(record, dict) or set(record) - extra != {
            "id",
            "name",
            "points",
            "color",
            "visible",
        } | (set() if legacy else POLYLINE_STYLE_FIELDS):
            raise ValueError("Invalid polyline fields")
        identity = record["id"]
        if (
            not isinstance(identity, str)
            or str(UUID(identity)) != identity
            or identity in identities
        ):
            raise ValueError("Invalid or duplicate polyline identity")
        identities.add(identity)
        validated_name(record["name"])
        if not legacy:
            validated_style(**{key: record[key] for key in POLYLINE_STYLE_FIELDS})
            points = record["points"]
            if (
                not isinstance(points, np.ndarray) or points.dtype != np.float64
                or not points.flags.c_contiguous
            ):
                raise ValueError("Polyline points must be C-contiguous float64")
        if type(record["visible"]) is not bool:
            raise ValueError("Invalid polyline visibility")
        points = validated_points(
            record["points"], allow_duplicate_adjacent="region" in record,
        )
        total += len(points)
        if total > MAX_POLYLINE_POINTS:
            raise ValueError("Polyline vertices exceed the layer limit")
        validated_color(record["color"])


def load_polyline_vtp(path, *, space="from_file", cancel_check=None, mark_default_color=False):
    from vtk.util.numpy_support import vtk_to_numpy

    from ..mesh_io import MeshLoadCancelled, load_mesh
    from .polyline_csv import PolylineCSVCancelled
    from .model import resolve_mesh_coordinates

    try:
        result = load_mesh(path, cancel_check=cancel_check)
    except MeshLoadCancelled as error:
        raise PolylineCSVCancelled() from error
    data = result["polydata"]
    lines = data.GetLines()
    count = data.GetNumberOfLines()
    if not count or data.GetNumberOfPolys() or data.GetNumberOfStrips():
        raise ValueError("Choose a VTP containing line geometry without surfaces")
    if count > MAX_POLYLINES or lines.GetNumberOfConnectivityIds() > MAX_POLYLINE_POINTS:
        raise ValueError("Polyline container capacity exceeded")
    coordinates = resolve_mesh_coordinates(result["space_metadata"], space)
    offsets = vtk_to_numpy(lines.GetOffsetsArray())
    ids = vtk_to_numpy(lines.GetConnectivityArray())
    points = vtk_to_numpy(data.GetPoints().GetData())
    colors = data.GetPointData().GetArray("Colors")
    rgb = None
    if colors and colors.GetNumberOfComponents() in (3, 4):
        values = vtk_to_numpy(colors)
        if values.dtype == np.uint8 and len(values) == len(points):
            rgb = values
    records = []
    for index in range(count):
        if cancel_check and cancel_check():
            raise PolylineCSVCancelled()
        vertices = ids[offsets[index]:offsets[index + 1]]
        trace = points[vertices].astype(np.float64, copy=False)
        if coordinates["coordinate_system"] == "LPS":
            trace[:, :2] *= -1
        name = Path(path).stem
        records.append({
            "name": name if count == 1 else f"{name} ({index + 1})",
            "points": validated_points(trace),
            "color": (DEFAULT_POLYLINE_COLOR if rgb is None else
                      validated_color(rgb[vertices[0], :3] / 255)),
            "visible": True, **validated_style(),
        })
        if mark_default_color and rgb is None:
            records[-1]["auto_color"] = True
    return records


def polyline_polydata(points, color) -> vtk.vtkPolyData:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    values = validated_points(points)
    rgb = np.rint(np.asarray(validated_color(color)) * 255).astype(np.uint8)
    vertices = vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(values, deep=True))
    line = vtk.vtkCellArray()
    dtype = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
    line.SetData(
        numpy_to_vtkIdTypeArray(np.array([0, len(values)], dtype=dtype), deep=True),
        numpy_to_vtkIdTypeArray(np.arange(len(values), dtype=dtype), deep=True),
    )
    data = vtk.vtkPolyData()
    data.SetPoints(vertices)
    data.SetLines(line)
    colors = numpy_to_vtk(np.tile(rgb, (len(values), 1)), deep=True)
    colors.SetName("Colors")
    data.GetPointData().SetScalars(colors)
    space = vtk.vtkStringArray()
    space.SetName("SPACE")
    space.InsertNextValue("RAS")
    data.GetFieldData().AddArray(space)
    return data


def protected_export_path(path, protected_paths):
    if Path(path).is_symlink():
        raise ValueError("Polyline export cannot replace a symbolic link")
    destination = Path(path).resolve()
    for source in protected_paths:
        source = Path(source).resolve()
        if destination == source or (
            destination.exists()
            and source.exists()
            and os.path.samefile(destination, source)
        ):
            raise ValueError("Polyline export cannot overwrite a loaded source")
    return destination


def save_polyline(
    path, points, color, *, protected_paths=(), csv_profile="xyz", header=True,
    name=None, line_width=DEFAULT_LINE_WIDTH, point_size=DEFAULT_POINT_SIZE,
    show_points=True, visible=True, cancel_check=None,
    reference_grid: ReferenceGrid | None = None,
) -> None:
    """Write line topology without moving vertices or modifying source files."""
    destination = protected_export_path(path, protected_paths)
    extension = destination.suffix.lower()
    if extension == ".csv":
        from .polyline_csv import save_polyline_csv

        save_polyline_csv(
            destination, points, color, profile=csv_profile, header=header,
            name=name, line_width=line_width, point_size=point_size,
            show_points=show_points, visible=visible,
            protected_paths=protected_paths, cancel_check=cancel_check,
        )
        return
    if extension not in (".vtp", ".vtk", ".obj", ".ply", ".trk", ".trx"):
        raise ValueError(
            "Supported polyline formats: VTP, VTK, OBJ, PLY, CSV, TRK and TRX"
        )
    values = validated_points(points)
    color = validated_color(color)
    from .polyline_csv import _checkpoint

    _checkpoint(cancel_check)
    with staged_output(destination) as staged:
        if extension in (".trk", ".trx"):
            import nibabel as nib

            from ..file_io import (
                _prepare_trk_header, _prepare_trx_header, _save_tractogram_file,
            )
            from ..reference_grid import ReferenceGrid

            if np.max(np.abs(values)) > np.finfo(np.float32).max:
                raise ValueError("Polyline coordinates exceed float32 range")
            grid = reference_grid or ReferenceGrid.from_points(
                values, provenance="synthetic:polyline-bounds", include_origin=False,
            )
            maximum = 32767 if extension == ".trk" else 65535
            if any(size > maximum for size in grid.shape):
                raise ValueError(
                    "Polyline reference dimensions exceed the format limit"
                )
            prepare = (
                _prepare_trk_header if extension == ".trk" else _prepare_trx_header
            )
            header = prepare({}, 1, reference_grid=grid)
            tractogram = nib.streamlines.Tractogram(
                [values], affine_to_rasmm=np.eye(4),
            )
            _save_tractogram_file(
                tractogram, header, str(staged), extension, _transactional=False,
            )
        elif extension in (".vtp", ".vtk"):
            import vtk

            data = polyline_polydata(values, color)
            writer = (
                vtk.vtkXMLPolyDataWriter()
                if extension == ".vtp"
                else vtk.vtkPolyDataWriter()
            )
            writer.SetFileName(str(staged))
            writer.SetInputData(data)
            if extension == ".vtk":
                writer.SetFileTypeToBinary()
            if writer.Write() != 1 or writer.GetErrorCode():
                raise OSError("Could not write polyline")
        elif extension == ".obj":
            with staged.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write("# TractEdit polyline; SPACE=RAS; units=mm\n")
                rgb = " ".join(format(v, ".17g") for v in color)
                for index, point in enumerate(values):
                    if index % 4096 == 0:
                        _checkpoint(cancel_check)
                    xyz = " ".join(format(v, ".17g") for v in point)
                    stream.write(f"v {xyz} {rgb}\n")
                stream.write(
                    "l " + " ".join(str(i + 1) for i in range(len(values))) + "\n"
                )
        else:
            rgb = " ".join(str(int(v)) for v in np.rint(np.array(color) * 255))
            with staged.open("w", encoding="ascii", newline="\n") as stream:
                stream.write(
                    "ply\nformat ascii 1.0\n"
                    "comment TractEdit polyline; SPACE=RAS; units=mm\n"
                    f"element vertex {len(values)}\n"
                    "property double x\nproperty double y\nproperty double z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    f"element edge {len(values) - 1}\n"
                    "property int vertex1\nproperty int vertex2\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "end_header\n"
                )
                for index, point in enumerate(values):
                    if index % 4096 == 0:
                        _checkpoint(cancel_check)
                    xyz = " ".join(format(v, ".17g") for v in point)
                    stream.write(f"{xyz} {rgb}\n")
                for index in range(len(values) - 1):
                    if index % 4096 == 0:
                        _checkpoint(cancel_check)
                    stream.write(f"{index} {index + 1} {rgb}\n")
        _checkpoint(cancel_check)
        protected_export_path(destination, protected_paths)
