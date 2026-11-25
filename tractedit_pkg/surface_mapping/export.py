# -*- coding: utf-8 -*-
"""Lossless RAS surface analysis exports, staged as a single output set."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from .. import __version__
from .models import EndpointStatus, IntersectionClass, _check_cancel
from ..transactional_io import staged_output_set

if TYPE_CHECKING:
    from .process import SurfaceMappingBundle

EXPORT_SCHEMA = 1
BLOCK_ITEMS = 4096
BLOCK_BYTES = 4 * 1024 * 1024


def output_paths(
    prefix: str | Path,
    bundle: SurfaceMappingBundle,
    scientific: bool = True,
    figure: bool = False,
) -> dict[str, Path]:
    base = Path(prefix).absolute()
    if not base.name.strip() or base.name in (".", ".."):
        raise ValueError("Choose a nonempty output prefix")
    if any(char in base.name for char in '<>:"/\\|?*') or base.name.endswith(
        (" ", ".")
    ):
        raise ValueError("Invalid output prefix")
    if not scientific and not figure:
        raise ValueError("Choose scientific data, a figure, or both")
    suffixes = {}
    if scientific:
        suffixes.update(
            surface="_surface.vtp", points="_points.vtp", scope="_scope.csv"
        )
        if bundle.result.endpoints is not None:
            suffixes["associations"] = "_associations.csv"
        else:
            suffixes["events"] = "_events.csv"
        if bundle.result.intervals is not None and len(bundle.result.intervals):
            suffixes.update(
                overlaps="_overlaps.vtp",
                intervals="_intervals.csv",
                interval_cells="_interval_cells.csv",
            )
    if figure:
        suffixes.update(figure="_figure.png", figure_metadata="_figure.json")
    if scientific:
        suffixes["provenance"] = "_provenance.json"
    paths = {
        role: base.with_name(base.name + suffix) for role, suffix in suffixes.items()
    }
    companion = getattr(bundle, "companion", None)
    if companion is not None and scientific:
        companion_paths = output_paths(
            base.with_name(base.name + "_surface_intersections"), companion
        )
        paths.update(
            ("surface_intersections_" + role, path)
            for role, path in companion_paths.items()
        )
    return paths


def validate_destinations(paths, sources, overwrite=False) -> None:
    for path in paths.values():
        if path.is_symlink():
            raise ValueError(f"Output must be a regular file: {path}")
    destinations = [path.resolve() for path in paths.values()]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Output paths collide")
    source_paths = [
        Path(record["path"] if isinstance(record, dict) else record).resolve()
        for record in sources.values()
    ]
    for destination in destinations:
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"Destination directory does not exist: {destination.parent}"
            )
        if destination.exists() and not destination.is_file():
            raise ValueError(f"Output must be a regular file: {destination}")
        if any(
            destination == source
            or destination.exists()
            and source.exists()
            and os.path.samefile(destination, source)
            for source in source_paths
        ):
            raise ValueError(f"An export cannot overwrite a source: {destination}")
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Output files already exist: " + ", ".join(existing))


def _digest(path, cancel_check=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(BLOCK_BYTES):
            _check_cancel(cancel_check)
            digest.update(block)
    return digest.hexdigest()


def _array_digest(values, cancel_check):
    digest = hashlib.sha256()
    for start in range(0, len(values), BLOCK_ITEMS):
        _check_cancel(cancel_check)
        digest.update(
            np.ascontiguousarray(values[start : start + BLOCK_ITEMS]).tobytes()
        )
    return {
        "sha256": digest.hexdigest(),
        "dtype": values.dtype.str,
        "shape": list(values.shape),
    }


def _scope_digest(streamlines, ids, cancel_check):
    if streamlines is None:
        return None
    digest = hashlib.sha256()
    for original_id in ids:
        _check_cancel(cancel_check)
        line = streamlines[int(original_id)]
        digest.update(np.asarray([original_id, len(line)], dtype="<i8").tobytes())
        digest.update(line.dtype.str.encode("ascii"))
        for start in range(0, len(line), BLOCK_ITEMS):
            _check_cancel(cancel_check)
            digest.update(
                np.ascontiguousarray(line[start : start + BLOCK_ITEMS]).tobytes()
            )
    return {
        "sha256": digest.hexdigest(),
        "policy": "Sorted original int64 ID and point count (little endian), coordinate dtype string and C-order original coordinate bytes per requested streamline; includes original intermediate points",
    }


def _add_array(attributes, name, values):
    from vtkmodules.util.numpy_support import numpy_to_vtk

    values = np.asarray(values)
    if values.dtype.kind == "b":
        values = values.astype(np.uint8)
    array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    array.SetName(name)
    attributes.AddArray(array)


def _field(data, name, value):
    from vtkmodules.vtkCommonCore import vtkStringArray

    array = vtkStringArray()
    array.SetName(name)
    array.InsertNextValue(value)
    data.GetFieldData().AddArray(array)


def _polydata(points, cell_size):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    data, vtk_points, cells = vtkPolyData(), vtkPoints(), vtkCellArray()
    vtk_points.SetData(
        numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True)
    )
    data.SetPoints(vtk_points)
    if not cell_size:
        return data
    count = len(points)
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            np.arange(0, count + 1, cell_size, dtype=np.int64), deep=True
        ),
        numpy_to_vtkIdTypeArray(np.arange(count, dtype=np.int64), deep=True),
    )
    (data.SetVerts if cell_size == 1 else data.SetLines)(cells)
    return data


def _write_vtp(path, data, cancel_check=None):
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

    _field(data, "SPACE", "RAS")
    _field(data, "units", "mm")
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    writer.SetHeaderTypeToUInt64()
    if cancel_check is not None:
        writer.AddObserver(
            "ProgressEvent", lambda *_: writer.SetAbortExecute(bool(cancel_check()))
        )
    if writer.Write() != 1 or writer.GetErrorCode():
        _check_cancel(cancel_check)
        raise OSError(f"Could not write VTP: {path.name}")
    _check_cancel(cancel_check)


def _table_columns(dtype):
    columns = []
    for name in dtype.names:
        field = dtype.fields[name][0]
        if field.shape:
            columns.extend(
                (f"{name}_{axis}", name, i)
                for i, axis in enumerate("xyz"[: field.shape[0]])
            )
        else:
            columns.append((name, name, None))
    return columns


def _write_table(path, table, cancel_check):
    columns = _table_columns(table.dtype)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", *(name for name, _, _ in columns)])
        for start in range(0, len(table), BLOCK_ITEMS):
            _check_cancel(cancel_check)
            for index in range(start, min(start + BLOCK_ITEMS, len(table))):
                row = table[index]
                writer.writerow(
                    [
                        index,
                        *(
                            (
                                row[name].item()
                                if component is None
                                else row[name][component].item()
                            )
                            for _, name, component in columns
                        ),
                    ]
                )


def _surface_data(bundle):
    from vtkmodules.vtkCommonDataModel import vtkPolyData

    geometry, result = bundle.geometry, bundle.result
    data = vtkPolyData()
    data.ShallowCopy(geometry.polydata)
    _add_array(data.GetCellData(), "triangle_id", geometry.triangle_ids)
    _add_array(data.GetCellData(), "source_cell_id", geometry.source_cell_ids)
    _add_array(data.GetCellData(), "area_mm2", geometry.areas_mm2)
    _add_array(data.GetPointData(), "vertex_area_mm2", geometry.vertex_areas_mm2)
    if result.endpoints is not None:
        for name in (
            "endpoint_contributions",
            "endpoint_density",
            "endpoint_density_valid",
        ):
            _add_array(data.GetPointData(), name, getattr(result, name))
        data.GetPointData().SetActiveScalars("endpoint_contributions")
        _field(data, "endpoint_contributions_units", "endpoint contributions")
        _field(data, "endpoint_density_units", "endpoint contributions/mm^2")
    return data


def _point_data(result, cancel_check):
    table = result.endpoints if result.endpoints is not None else result.intersections
    # Two passes bound temporary masks even for disk-backed event tables.
    blocks = []
    for start in range(0, len(table), BLOCK_ITEMS):
        _check_cancel(cancel_check)
        block = table[start : start + BLOCK_ITEMS]
        valid = np.isfinite(block["position_ras"]).all(axis=1)
        if result.endpoints is not None:
            valid &= block["status"] == EndpointStatus.ASSIGNED
        else:
            valid &= block["class"] != IntersectionClass.INVALID
        blocks.append(np.flatnonzero(valid).astype(np.int64) + start)
    ids = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
    data = _polydata(table["position_ras"][ids], 1)
    _add_array(data.GetPointData(), "record_id", ids)
    for name in table.dtype.names:
        if name != "position_ras":
            _check_cancel(cancel_check)
            _add_array(data.GetPointData(), name, table[name][ids])
    return data


def _overlap_data(result, cancel_check, streamlines):
    from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonDataModel import vtkCellArray

    table = result.intervals
    offsets = np.zeros(len(table) + 1, dtype=np.int64)
    for index, row in enumerate(table):
        if index % BLOCK_ITEMS == 0:
            _check_cancel(cancel_check)
        extra = max(0, int(row["end_segment_id"] - row["start_segment_id"]))
        if extra and streamlines is None:
            raise ValueError(
                "Original streamline vertices are required for multi-segment overlaps"
            )
        offsets[index + 1] = offsets[index] + 2 + extra
    positions = np.empty((offsets[-1], 3), dtype=np.float64)
    for index, row in enumerate(table):
        _check_cancel(cancel_check)
        start, end = offsets[index : index + 2]
        positions[start], positions[end - 1] = (
            row["start_position_ras"],
            row["end_position_ras"],
        )
        if end - start > 2:
            positions[start + 1 : end - 1] = streamlines[int(row["streamline_id"])][
                int(row["start_segment_id"]) + 1 : int(row["end_segment_id"]) + 1
            ]
    data = _polydata(positions, 0)
    lines = vtkCellArray()
    lines.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(np.arange(len(positions), dtype=np.int64), deep=True),
    )
    data.SetLines(lines)
    _add_array(data.GetCellData(), "record_id", np.arange(len(table), dtype=np.int64))
    for name in table.dtype.names:
        _check_cancel(cancel_check)
        _add_array(data.GetCellData(), name, table[name])
    _field(
        data,
        "geometry_role",
        "coplanar surface overlap intervals; not original streamlines",
    )
    _field(data, "cell_relation", "interval_id joins the interval_cells CSV")
    return data


def _array_definitions(bundle):
    definitions = {
        "triangle_id": {
            "association": "CellData(surface), PointData(points)",
            "meaning": "Zero-based derived triangulation ID, before degenerate exclusion",
        },
        "source_cell_id": {
            "association": "CellData(surface), PointData(points)",
            "meaning": "Zero-based original VTK cell ID, including vertex/line cell offsets",
        },
        "record_id": {
            "association": "PointData(points), CellData(overlaps)",
            "meaning": "Zero-based row in the corresponding CSV",
        },
        "streamline_id": {
            "association": "PointData(points), CellData(overlaps)",
            "meaning": "Original zero-based storage ID, int64",
        },
        "vertex_area_mm2": {
            "association": "PointData(surface)",
            "meaning": "One third of each incident valid triangle area, mm^2",
        },
        "area_mm2": {
            "association": "CellData(surface)",
            "meaning": "Valid triangle area, mm^2",
        },
    }
    if bundle.result.endpoints is not None:
        definitions.update(
            {
                "endpoint_contributions": {
                    "association": "PointData(surface)",
                    "meaning": "Unit assigned-endpoint weights distributed barycentrically",
                },
                "endpoint_density": {
                    "association": "PointData(surface)",
                    "meaning": "endpoint contributions/mm^2; no smoothing; not axonal density",
                },
                "endpoint_density_valid": {
                    "association": "PointData(surface)",
                    "meaning": "uint8: 1 for positive vertex area, 0 for masked zero area",
                },
            }
        )
    tables = (
        ("associations", bundle.result.endpoints),
        ("events", bundle.result.intersections),
        ("intervals", bundle.result.intervals),
        ("interval_cells", bundle.result.interval_cells),
    )
    for role, table in tables:
        if table is None:
            continue
        for name in table.dtype.names:
            definitions.setdefault(
                name,
                {
                    "association": (
                        "PointData(points)"
                        if role in ("associations", "events")
                        else (
                            "CellData(overlaps)"
                            if role == "intervals"
                            else "CSV relation"
                        )
                    )
                },
            )
            if name == "position_ras":
                definitions[name]["association"] = "Points(points), CSV"
            definitions[name].setdefault(
                "meaning",
                {
                    "endpoint": "Storage endpoint 0/1; not biological direction",
                    "status": "EndpointStatus enumeration",
                    "class": "IntersectionClass enumeration; overlaps are intervals",
                    "barycentric": "Three dimensionless weights on the representative triangle",
                    "distance_mm": "Actual endpoint-to-surface distance, mm",
                    "angle_deg": "Acute unsigned incidence to geometric normal, degrees",
                    "normal_nonunique": "Incident faces have different normals",
                    "segment_id": "Original zero-based segment ID",
                    "segment_t": "Dimensionless local segment parameter in [0,1]",
                }.get(
                    name,
                    (
                        "RAS mm"
                        if "position_ras" in name
                        else (
                            "Curvilinear distance in mm"
                            if "arc_length_mm" in name
                            else "Original segment/triangle/cell or interval identifier/parameter; see table schema"
                        )
                    ),
                ),
            )
            definitions[name]["dtype"] = str(table.dtype.fields[name][0])
    for name, definition in definitions.items():
        if name == "endpoint_density_valid":
            definition["dtype"] = "uint8"
        elif name in (
            "endpoint_contributions",
            "endpoint_density",
            "vertex_area_mm2",
            "area_mm2",
        ):
            definition["dtype"] = "float64"
    return definitions


def _write_json(path, document):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class StagedSurfaceMappingExport:
    def __init__(self, context, paths, sources, overwrite, cancel_check):
        self._context = context
        self.paths, self.sources = paths, sources
        self.overwrite, self.cancel_check = overwrite, cancel_check
        self.source_stamps = {}

    def close(self):
        if self._context is not None:
            context, self._context = self._context, None
            error = RuntimeError("Surface mapping export discarded")
            context.__exit__(type(error), error, error.__traceback__)

    def commit(self, current_check: Callable[[], bool] = lambda: True):
        try:
            _check_cancel(self.cancel_check)
            if not current_check():
                raise ValueError(
                    "Recompute required: surface mapping is no longer current"
                )
            validate_destinations(self.paths, self.sources, self.overwrite)
            from ..session.io import source_stamp

            if any(
                source_stamp(path) != stamp
                for path, stamp in self.source_stamps.items()
            ):
                raise ValueError(
                    "An input file changed while exporting the surface mapping"
                )
            context, self._context = self._context, None
            if context is None:
                raise RuntimeError("Export has already been completed or discarded")
            context.__exit__(None, None, None)
            return self.paths
        finally:
            self.close()


def stage_surface_mapping(
    prefix,
    bundle,
    sources,
    *,
    scientific=True,
    figure=None,
    presentation=None,
    overwrite=False,
    cancel_check=None,
    streamlines=None,
    _transaction=None,
    _staged=None,
    _records=None,
    _scope=None,
):
    """Write detached data on a worker; the GUI validates ownership before commit.

    ``figure`` is an independent (vtkImageData, metadata) pair captured on the
    GUI thread. The caller keeps the bundle leased until staging finishes.
    """
    _check_cancel(cancel_check)
    paths = output_paths(prefix, bundle, scientific, figure is not None)
    validate_destinations(paths, sources, overwrite)
    if _transaction is None:
        context = staged_output_set(list(paths.values()))
        staged = context.__enter__()
        transaction = StagedSurfaceMappingExport(
            context, paths, sources, overwrite, cancel_check
        )
    else:
        transaction, staged = _transaction, _staged
    try:
        companion = getattr(bundle, "companion", None)
        if _records is None:
            records = {}
            from ..session.io import checked_fingerprint

            for role, source in sources.items():
                path = source["path"] if isinstance(source, dict) else source
                receipt, stamp = checked_fingerprint(
                    path, lambda: bool(cancel_check and cancel_check())
                )
                if (
                    path in transaction.source_stamps
                    and transaction.source_stamps[path] != stamp
                ):
                    raise ValueError(
                        "An input file changed while exporting the surface mapping"
                    )
                records[role] = receipt
                transaction.source_stamps[path] = stamp
        else:
            records = _records
        ids = bundle.result.request.streamline_ids
        scope_digest = (
            _scope[1] if _scope is not None and np.array_equal(ids, _scope[0])
            else _scope_digest(streamlines, ids, cancel_check)
        )
        if companion is not None and scientific:
            base = Path(prefix)
            stage_surface_mapping(
                base.with_name(base.name + "_surface_intersections"),
                companion, sources, presentation=presentation,
                overwrite=overwrite, cancel_check=cancel_check,
                streamlines=streamlines, _transaction=transaction, _staged=staged,
                _records=records, _scope=(ids, scope_digest),
            )
        geometry = bundle.geometry
        geometry_record = {
            "epsilon_mm": geometry.epsilon_mm,
            "diagnostics": asdict(geometry.diagnostics),
            "arrays": {
                name: _array_digest(getattr(geometry, name), cancel_check)
                for name in (
                    "points_ras",
                    "triangles",
                    "triangle_ids",
                    "source_cell_ids",
                )
            },
        }
        if scientific:
            for role, builder in (
                ("surface", lambda: _surface_data(bundle)),
                ("points", lambda: _point_data(bundle.result, cancel_check)),
                (
                    "overlaps",
                    lambda: _overlap_data(bundle.result, cancel_check, streamlines),
                ),
            ):
                if role in paths:
                    _check_cancel(cancel_check)
                    data = builder()
                    _write_vtp(staged[paths[role]], data, cancel_check)
                    del data
            for role, name in (
                ("associations", "endpoints"),
                ("events", "intersections"),
                ("intervals", "intervals"),
                ("interval_cells", "interval_cells"),
            ):
                if role in paths:
                    _write_table(
                        staged[paths[role]], getattr(bundle.result, name), cancel_check
                    )
            scope = bundle.result.request.streamline_ids
            _write_table(
                staged[paths["scope"]],
                scope.view(np.dtype([("streamline_id", "i8")])),
                cancel_check,
            )
        request = bundle.result.request.metadata()
        request["streamline_ids"] = (
            ids.tolist()
            if len(ids) <= BLOCK_ITEMS
            else {
                "file": paths["scope"].name if scientific else None,
                "count": len(ids),
                **_array_digest(ids, cancel_check),
            }
        )
        manifest = {
            "format": "tractedit-surface-mapping",
            "schema": EXPORT_SCHEMA,
            "software_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "space": "RAS",
            "coordinate_units": "mm",
            "index_base": 0,
            "missing_identifiers": -1,
            "missing_floating_values": "NaN",
            "request": request,
            "sources": records,
            "source_hash_policy": "File bytes verified at the export boundary; live scientific geometry and requested streamline coordinates are fingerprinted separately",
            "tractogram_scope": scope_digest,
            "geometry": geometry_record,
            "diagnostics": asdict(bundle.result.diagnostics),
            "arrays": _array_definitions(bundle),
            "endpoint_status": {item.name: int(item) for item in EndpointStatus},
            "intersection_class": {item.name: int(item) for item in IntersectionClass},
            "quantitative_policy": "Only ASSIGNED endpoints contribute barycentrically. BEYOND_DISTANCE, AMBIGUOUS and INVALID are excluded. Only CROSSING points enter crossing counts; TOUCH, COPLANAR_OVERLAP, AMBIGUOUS and INVALID are QC.",
            "counter_convention": "Two storage endpoints per requested streamline. Endpoint contributions count endpoints, not unique streamlines. crossing_events counts points; crossing_streamlines counts unique original IDs. coplanar_intervals includes ambiguous intervals; ambiguous_events includes points and intervals. No count represents axons.",
            "presentation": presentation or {},
            "files": [],
        }
        if companion is not None:
            manifest["additional_mapping"] = {
                "parameters": asdict(companion.result.request.parameters),
                "diagnostics": asdict(companion.result.diagnostics),
                "request_id": companion.result.request.request_id,
                "scope": "Same original streamline IDs as the primary mapping",
                "provenance": (
                    paths["surface_intersections_provenance"].name
                    if scientific else None
                ),
            }
        if figure is not None:
            from vtkmodules.vtkIOImage import vtkPNGWriter

            image, metadata = figure
            writer = vtkPNGWriter()
            writer.SetInputData(image)
            writer.SetFileName(str(staged[paths["figure"]]))
            writer.Write()
            if writer.GetErrorCode():
                raise OSError("Could not encode the surface mapping PNG")
            figure_record = {
                **metadata,
                "request_id": request["request_id"],
                "provenance": paths["provenance"].name if scientific else None,
                "png_sha256": _digest(staged[paths["figure"]], cancel_check),
            }
            if scientific:
                _write_json(staged[paths["figure_metadata"]], figure_record)
        manifest_role = "provenance" if scientific else "figure_metadata"
        for role, path in paths.items():
            if role != manifest_role:
                manifest["files"].append(
                    {
                        "role": role,
                        "name": path.name,
                        "sha256": _digest(staged[path], cancel_check),
                        "bytes": staged[path].stat().st_size,
                    }
                )
        manifest["manifest_checksum_policy"] = (
            "All payload files are SHA-256 checked; the provenance manifest excludes itself to avoid a circular hash."
        )
        _write_json(
            staged[paths[manifest_role]],
            manifest if scientific else {**figure_record, "provenance": manifest},
        )
        _check_cancel(cancel_check)
        return transaction
    except BaseException:
        transaction.close()
        raise


def export_surface_mapping(
    prefix, bundle, sources, *, current_check=lambda: True, **options
):
    if not current_check():
        raise ValueError("Recompute required: surface mapping is no longer current")
    transaction = stage_surface_mapping(prefix, bundle, sources, **options)
    return transaction.commit(current_check)
