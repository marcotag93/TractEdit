# -*- coding: utf-8 -*-
"""Cancellable surface analysis with file snapshots and one isolated process."""

import json
import shutil
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ..process_isolation import run_isolated_process
from .models import (
    ENDPOINT_DTYPE,
    QUERY_CHUNK_ITEMS,
    EndpointStatus,
    IntersectionClass,
    SurfaceGeometry,
    SurfaceGeometryDiagnostics,
    SurfaceGeometryKey,
    SurfaceMappingDiagnostics,
    SurfaceMappingParameters,
    SurfaceMappingRequest,
    SurfaceMappingResult,
    _check_cancel,
    mapping_source,
    prepare_surface_geometry,
)
from .queries import flat_streamline_source

TRANSFER_BYTES = 4 * 1024 * 1024
MARKER_LIMIT = 50_000
INTERVAL_LIMIT = 10_000
_GEOMETRY_ARRAYS = (
    "points_ras",
    "triangles",
    "triangle_ids",
    "source_cell_ids",
    "normals",
    "areas_mm2",
    "vertex_areas_mm2",
    "barycentric_tolerances",
)
_RESULT_ARRAYS = (
    "endpoints",
    "intersections",
    "intervals",
    "interval_cells",
    "endpoint_contributions",
    "endpoint_density",
    "endpoint_density_valid",
)


def _save_array(path: Path, source: np.ndarray, cancel: Callable[[], bool]) -> None:
    _check_cancel(cancel)
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=source.dtype,
        shape=source.shape,
    )
    try:
        step = max(
            1,
            TRANSFER_BYTES
            // max(1, source.dtype.itemsize * int(np.prod(source.shape[1:]))),
        )
        for start in range(0, len(source), step):
            _check_cancel(cancel)
            target[start : start + step] = source[start : start + step]
        target.flush()
    finally:
        target._mmap.close()


def _polydata(points, cells):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    data, vtk_points = vtkPolyData(), vtkPoints()
    data._surface_mapping_buffers = (
        points,
        *[array for pair in cells for array in pair],
    )
    vtk_points.SetData(numpy_to_vtk(points))
    data.SetPoints(vtk_points)
    for setter, (offsets, indices) in zip(
        (data.SetVerts, data.SetLines, data.SetPolys, data.SetStrips),
        cells,
    ):
        array = vtkCellArray()
        array.SetData(
            numpy_to_vtkIdTypeArray(offsets), numpy_to_vtkIdTypeArray(indices)
        )
        setter(array)
    return data


def _terminal_pair(line, end, cancel):
    length = len(line)
    index = 0 if end == 0 else length - 1
    point = np.asarray(line[index], dtype=np.float64)
    if length < 2:
        return point, point
    if isinstance(line, np.ndarray):
        neighbor = line[1 if end == 0 else length - 2]
        if (
            float(neighbor[0]) != float(point[0])
            or float(neighbor[1]) != float(point[1])
            or float(neighbor[2]) != float(point[2])
        ):
            return point, np.asarray(neighbor, dtype=np.float64)
    for start in range(1, length, QUERY_CHUNK_ITEMS):
        _check_cancel(cancel)
        stop = min(start + QUERY_CHUNK_ITEMS, length)
        block = np.asarray(
            line[start:stop]
            if end == 0
            else line[length - stop : length - start][::-1]
        )
        distinct = np.flatnonzero(np.any(block != point, axis=1))
        if len(distinct):
            return point, block[distinct[0]]
    return point, point


def _stage_flat_points(
    points, ids, offsets, data, src_offsets, src_lengths, endpoint_mode, cancel
):
    if endpoint_mode:
        buf = np.empty((QUERY_CHUNK_ITEMS, 4, 3), dtype=np.float64)
        for c_start in range(0, len(ids), QUERY_CHUNK_ITEMS):
            _check_cancel(cancel)
            c_stop = min(c_start + QUERY_CHUNK_ITEMS, len(ids))
            chunk_ids = ids[c_start:c_stop]
            chunk_lens = src_lengths[chunk_ids].astype(np.int64, copy=False)
            valid_mask = chunk_lens > 0
            if not np.any(valid_mask):
                continue
            if not np.all(valid_mask):
                chunk_ids = chunk_ids[valid_mask]
                chunk_lens = chunk_lens[valid_mask]
            chunk_offs = src_offsets[chunk_ids].astype(np.int64, copy=False)
            out = buf[: len(chunk_ids)]
            out[:, 0] = data[chunk_offs]
            out[:, 1] = data[chunk_offs + np.minimum(chunk_lens - 1, 1)]
            out[:, 2] = data[chunk_offs + np.maximum(chunk_lens - 2, 0)]
            out[:, 3] = data[chunk_offs + chunk_lens - 1]
            dup0 = (chunk_lens > 2) & ~np.any(out[:, 1] != out[:, 0], axis=1)
            if np.any(dup0):
                for k in np.flatnonzero(dup0):
                    off_k = int(chunk_offs[k])
                    len_k = int(chunk_lens[k])
                    _, out[k, 1] = _terminal_pair(
                        data[off_k : off_k + len_k], 0, cancel
                    )
            dup1 = (chunk_lens > 2) & ~np.any(out[:, 2] != out[:, 3], axis=1)
            if np.any(dup1):
                for k in np.flatnonzero(dup1):
                    off_k = int(chunk_offs[k])
                    len_k = int(chunk_lens[k])
                    _, out[k, 2] = _terminal_pair(
                        data[off_k : off_k + len_k], 1, cancel
                    )
            points[int(offsets[c_start]) : int(offsets[c_stop])] = out.reshape(-1, 3)
        return
    step = max(1, TRANSFER_BYTES // (3 * 8))
    gather_limit = QUERY_CHUNK_ITEMS * 16
    for c_start in range(0, len(ids), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel)
        c_stop = min(c_start + QUERY_CHUNK_ITEMS, len(ids))
        out_start = int(offsets[c_start])
        out_stop = int(offsets[c_stop])
        total_pts = out_stop - out_start
        if not total_pts:
            continue
        chunk_ids = ids[c_start:c_stop]
        chunk_lens = src_lengths[chunk_ids].astype(np.int64, copy=False)
        valid_mask = chunk_lens > 0
        if not np.all(valid_mask):
            chunk_ids = chunk_ids[valid_mask]
            chunk_lens = chunk_lens[valid_mask]
        chunk_offs = src_offsets[chunk_ids].astype(np.int64, copy=False)
        if len(chunk_offs) == 1 or bool(
            np.all(chunk_offs[1:] == chunk_offs[:-1] + chunk_lens[:-1])
        ):
            base_src = int(chunk_offs[0])
            for rel in range(0, total_pts, step):
                _check_cancel(cancel)
                end_rel = min(rel + step, total_pts)
                points[out_start + rel : out_start + end_rel] = data[
                    base_src + rel : base_src + end_rel
                ]
        elif total_pts <= gather_limit:
            local_offsets = np.empty(len(chunk_lens), dtype=np.int64)
            local_offsets[0] = 0
            if len(chunk_lens) > 1:
                np.cumsum(chunk_lens[:-1], out=local_offsets[1:])
            gather = np.arange(total_pts, dtype=np.int64) + np.repeat(
                chunk_offs - local_offsets, chunk_lens
            )
            points[out_start:out_stop] = data[gather]
        else:
            dst = out_start
            for off_k, len_k in zip(chunk_offs.tolist(), chunk_lens.tolist()):
                for block in range(0, len_k, QUERY_CHUNK_ITEMS):
                    _check_cancel(cancel)
                    stop_k = min(len_k, block + QUERY_CHUNK_ITEMS)
                    points[dst + block : dst + stop_k] = data[
                        off_k + block : off_k + stop_k
                    ]
                dst += len_k


def _stage(folder, layer, request, streamlines, cancel):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    source = mapping_source(layer, cancel)
    _save_array(
        folder / "mesh-points.npy", vtk_to_numpy(source.GetPoints().GetData()), cancel
    )
    for name, cells in zip(
        ("verts", "lines", "polys", "strips"),
        (source.GetVerts(), source.GetLines(), source.GetPolys(), source.GetStrips()),
    ):
        for suffix, array in (
            ("offsets", cells.GetOffsetsArray()),
            ("indices", cells.GetConnectivityArray()),
        ):
            _save_array(
                folder / f"{name}-{suffix}.npy",
                vtk_to_numpy(array).astype(np.int64, copy=False),
                cancel,
            )
    _save_array(folder / "ids.npy", request.streamline_ids, cancel)
    metadata = request.metadata()
    metadata["line_count"] = len(streamlines)
    (folder / "request.json").write_text(json.dumps(metadata), "utf-8")
    ids = request.streamline_ids
    lengths = np.zeros(len(ids), dtype=np.int64)
    endpoint_mode = request.parameters.mode == "endpoint_proximity"
    flat = flat_streamline_source(streamlines)
    if flat is not None:
        data, src_offsets, src_lengths, _ = flat
        for start in range(0, len(ids), QUERY_CHUNK_ITEMS):
            _check_cancel(cancel)
            stop = min(start + QUERY_CHUNK_ITEMS, len(ids))
            block_lens = src_lengths[ids[start:stop]].astype(np.int64, copy=False)
            if endpoint_mode:
                lengths[start:stop] = np.where(block_lens > 0, 4, 0)
            else:
                lengths[start:stop] = np.maximum(block_lens, 0)
    else:
        for index, original_id in enumerate(ids):
            _check_cancel(cancel)
            line = streamlines[int(original_id)]
            if len(line.shape) == 2 and line.shape[1] == 3 and len(line):
                lengths[index] = 4 if endpoint_mode else len(line)
    offsets = np.empty(len(ids) + 1, dtype=np.int64)
    offsets[0] = 0
    if len(ids):
        np.cumsum(lengths, out=offsets[1:])
    _save_array(folder / "offsets.npy", offsets, cancel)
    points = np.lib.format.open_memmap(
        folder / "line-points.npy",
        mode="w+",
        dtype=np.float64,
        shape=(int(offsets[-1]), 3),
    )
    try:
        if flat is not None:
            _stage_flat_points(
                points,
                ids,
                offsets,
                data,
                src_offsets,
                src_lengths,
                endpoint_mode,
                cancel,
            )
        else:
            for index, original_id in enumerate(ids):
                _check_cancel(cancel)
                if not lengths[index]:
                    continue
                line = streamlines[int(original_id)]
                start = int(offsets[index])
                if endpoint_mode:
                    first, first_inner = _terminal_pair(line, 0, cancel)
                    last, last_inner = _terminal_pair(line, 1, cancel)
                    points[start : start + 4] = (first, first_inner, last_inner, last)
                else:
                    for block in range(0, len(line), QUERY_CHUNK_ITEMS):
                        _check_cancel(cancel)
                        stop = min(len(line), block + QUERY_CHUNK_ITEMS)
                        points[start + block : start + stop] = line[
                            block : block + QUERY_CHUNK_ITEMS
                        ]
        points.flush()
    finally:
        points._mmap.close()


class _Snapshots:
    def __init__(self, ids, offsets, points, count):
        self.ids, self.offsets, self.points, self.count = ids, offsets, points, count

    def __len__(self):
        return self.count

    def __getitem__(self, original_id):
        index = np.searchsorted(self.ids, original_id)
        if index == len(self.ids) or self.ids[index] != original_id:
            raise IndexError("Streamline is outside the frozen scope")
        return self.points[self.offsets[index] : self.offsets[index + 1]]


def _read_request(folder):
    document = json.loads((folder / "request.json").read_text("utf-8"))
    count = document.pop("line_count")
    document.pop("algorithm_version")
    key = document["geometry_key"]
    key["geometry_revision"] = tuple(map(tuple, key["geometry_revision"]))
    key["source_to_ras"] = tuple(key["source_to_ras"])
    document["geometry_key"] = SurfaceGeometryKey(**key)
    document["parameters"] = SurfaceMappingParameters(**document["parameters"])
    document["streamline_ids"] = np.load(folder / "ids.npy", allow_pickle=False)
    return SurfaceMappingRequest(**document), count


def _sample_points(table, field, classes, cancel):
    """Sample markers and collect unique IDs from a streamline-ID-sorted table."""
    counts = np.zeros(len(classes), dtype=np.int64)
    for start in range(0, len(table), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel)
        block = table[field][start : start + QUERY_CHUNK_ITEMS]
        for index, label in enumerate(classes):
            counts[index] += np.count_nonzero(block == label)
    count = int(counts.sum())
    quotas = counts.copy()
    if count > MARKER_LIMIT:
        quotas = (counts > 0).astype(np.int64)
        remaining = MARKER_LIMIT - int(quotas.sum())
        weights = counts - quotas
        shares = weights * remaining
        quotas += shares // weights.sum()
        extra = MARKER_LIMIT - int(quotas.sum())
        order = np.argsort(-(shares % weights.sum()), kind="stable")
        quotas[order[:extra]] += 1
    positions, labels, associated = [], [], []
    seen = np.zeros(len(classes), dtype=np.int64)
    last_associated = -1
    for start in range(0, len(table), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel)
        block = table[start : start + QUERY_CHUNK_ITEMS]
        selected = np.zeros(len(block), dtype=bool)
        for index, label in enumerate(classes):
            if not counts[index]:
                continue
            keep = np.flatnonzero(block[field] == label)
            ranks = seen[index] + np.arange(len(keep))
            chosen = keep[
                (ranks + 1) * quotas[index] // counts[index]
                != ranks * quotas[index] // counts[index]
            ]
            selected[chosen] = True
            seen[index] += len(keep)
        chosen = np.flatnonzero(selected)
        positions.append(block["position_ras"][chosen])
        labels.append(block[field][chosen])
        ids = block["streamline_id"][block[field] == 0]
        if len(ids):
            unique = np.r_[ids[0] != last_associated, ids[1:] != ids[:-1]]
            if np.any(unique):
                associated.append(ids[unique])
            last_associated = int(ids[-1])
    return (
        np.concatenate(positions) if positions else np.empty((0, 3)),
        np.concatenate(labels) if labels else np.empty(0, dtype=np.uint8),
        (
            np.concatenate(associated)
            if associated
            else np.empty(0, dtype=np.int64)
        ),
        count,
    )


def _sample_overlaps(intervals, lines, cancel):
    if intervals is None or not len(intervals):
        return np.empty((0, 2, 3)), 0
    total = 0
    for start in range(0, len(intervals), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel)
        block = intervals[start : start + QUERY_CHUNK_ITEMS]
        total += int(np.sum(block["end_segment_id"] - block["start_segment_id"] + 1))
    stride = max(1, int(np.ceil(total / INTERVAL_LIMIT)))
    samples = []
    seen = 0
    for interval in intervals:
        _check_cancel(cancel)
        first, last = int(interval["start_segment_id"]), int(interval["end_segment_id"])
        line = lines[int(interval["streamline_id"])]
        offset = (-seen) % stride
        for segment in range(first + offset, last + 1, stride):
            _check_cancel(cancel)
            p = interval["start_position_ras"] if segment == first else line[segment]
            q = interval["end_position_ras"] if segment == last else line[segment + 1]
            samples.append(np.array([p, q]))
        seen += last - first + 1
    return np.array(samples, dtype=np.float64).reshape(-1, 2, 3), total


def _compute(directory, cancelled, channel):
    folder = Path(directory)
    maps = []
    result = None

    def load(name):
        array = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        maps.append(array)
        return array

    def phase(name):
        channel.send(name)
        _check_cancel(cancelled.is_set)

    try:
        request, count = _read_request(folder)
        source = _polydata(
            load("mesh-points"),
            [
                (load(f"{name}-offsets"), load(f"{name}-indices"))
                for name in ("verts", "lines", "polys", "strips")
            ],
        )
        layer = {
            "id": request.mesh_id,
            "polydata": source,
            "source_to_ras": np.array(request.geometry_key.source_to_ras).reshape(4, 4),
        }
        phase("geometry")
        geometry = prepare_surface_geometry(
            layer, cancel_check=cancelled.is_set, progress=phase
        )
        geometry.key = request.geometry_key
        phase("locator")
        geometry.get_locator()
        phase("query")
        lines = _Snapshots(load("ids"), load("offsets"), load("line-points"), count)
        output = folder / "output"
        output.mkdir()
        if request.parameters.mode == "endpoint_proximity":
            from .endpoints import map_endpoints

            required = 2 * len(request.streamline_ids) * ENDPOINT_DTYPE.itemsize
            required += sum(getattr(geometry, name).nbytes for name in _GEOMETRY_ARRAYS)
            required += len(geometry.points_ras) * 17
            if shutil.disk_usage(folder).free < required:
                raise OSError("Insufficient temporary disk space for surface mapping")
            records = np.lib.format.open_memmap(
                output / "endpoints.npy",
                mode="w+",
                dtype=ENDPOINT_DTYPE,
                shape=(2 * len(request.streamline_ids),),
            )
            maps.append(records)
            result = map_endpoints(
                geometry,
                request,
                lines,
                cancel_check=cancelled.is_set,
                endpoint_records=records,
                progress=phase,
            )
        else:
            from .intersections import map_intersections

            result = map_intersections(
                geometry,
                request,
                lines,
                output_directory=folder,
                cancel_check=cancelled.is_set,
                progress=phase,
            )
        phase("presentation")
        for name in _GEOMETRY_ARRAYS:
            _save_array(
                output / f"{name}.npy", getattr(geometry, name), cancelled.is_set
            )
        result_names = []
        for name in _RESULT_ARRAYS:
            array = getattr(result, name)
            if array is not None:
                target = output / f"{name}.npy"
                # NumPy resolves Path filenames, including Windows 8.3 names.
                # Never truncate an output that is already mapped by this worker.
                if (
                    isinstance(array, np.memmap)
                    and Path(array.filename).resolve() == target.resolve()
                ):
                    array.flush()
                else:
                    _save_array(target, array, cancelled.is_set)
                result_names.append(name)
        table = (
            result.endpoints if result.endpoints is not None else result.intersections
        )
        field = "status" if result.endpoints is not None else "class"
        classes = (
            (EndpointStatus.ASSIGNED, EndpointStatus.AMBIGUOUS)
            if field == "status"
            else (
                IntersectionClass.CROSSING,
                IntersectionClass.TOUCH,
                IntersectionClass.AMBIGUOUS,
            )
        )
        positions, labels, ids, marker_count = _sample_points(
            table, field, classes, cancelled.is_set
        )
        intervals = result.intervals
        overlap_count = 0 if intervals is None else len(intervals)
        overlaps, overlap_segment_count = _sample_overlaps(
            intervals, lines, cancelled.is_set
        )
        for name, array in (
            ("marker_points", positions),
            ("marker_classes", labels),
            ("associated_ids", ids),
            ("overlap_points", overlaps),
        ):
            _save_array(output / f"{name}.npy", array, cancelled.is_set)
        document = {
            "diagnostics": asdict(result.diagnostics),
            "geometry_diagnostics": asdict(geometry.diagnostics),
            "epsilon_mm": geometry.epsilon_mm,
            "result_arrays": result_names,
            "marker_count": marker_count,
            "overlap_count": overlap_count,
            "overlap_segment_count": overlap_segment_count,
        }
        for name in ("endpoint_contributions", "endpoint_density"):
            values = getattr(result, name)
            finite = values[np.isfinite(values)] if values is not None else []
            document[name + "_range"] = (
                [float(np.min(finite)), float(np.max(finite))]
                if len(finite)
                else [0.0, 1.0]
            )
        _check_cancel(cancelled.is_set)
        (output / "complete.json").write_text(json.dumps(document), "utf-8")
    except Exception as error:
        (folder / "error.json").write_text(
            json.dumps(str(error)[:4096] or type(error).__name__), "utf-8"
        )
    finally:
        if result is not None:
            result.close()
        for array in maps:
            array._mmap.close()
        channel.close()


class _Storage:
    def __init__(self, directory):
        self.directory, self.maps = directory, []

    def load(self, name):
        array = np.load(
            Path(self.directory.name) / "output" / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.maps.append(array)
        return array

    def close(self):
        for array in self.maps:
            array._mmap.close()
        self.maps.clear()
        self.directory.cleanup()


@dataclass
class SurfaceMappingBundle:
    result: SurfaceMappingResult
    geometry: SurfaceGeometry
    marker_points: np.ndarray
    marker_classes: np.ndarray
    associated_ids: np.ndarray
    overlap_points: np.ndarray
    marker_count: int
    overlap_count: int
    overlap_segment_count: int
    contribution_range: tuple[float, float]
    density_range: tuple[float, float]
    companion: "SurfaceMappingBundle | None" = field(default=None, repr=False)
    _leases: int = field(default=0, init=False, repr=False)
    _close_requested: bool = field(default=False, init=False, repr=False)

    def close(self):
        self._close_requested = True
        if not self._leases:
            self.result.close()
            if self.companion is not None:
                self.companion.close()

    @contextmanager
    def lease(self):
        if self._close_requested:
            raise RuntimeError("Surface mapping has already been released")
        self._leases += 1
        try:
            yield self
        finally:
            self._leases -= 1
            if not self._leases and self._close_requested:
                self.result.close()
                if self.companion is not None:
                    self.companion.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _restore(directory, request):
    storage = _Storage(directory)
    try:
        folder = Path(directory.name) / "output"
        document = json.loads((folder / "complete.json").read_text("utf-8"))
        arrays = {name: storage.load(name) for name in _GEOMETRY_ARRAYS}
        triangle_count = len(arrays["triangles"])
        empty = (np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64))
        polydata = _polydata(
            arrays["points_ras"],
            [
                empty,
                empty,
                (
                    np.arange(triangle_count + 1, dtype=np.int64) * 3,
                    arrays["triangles"].reshape(-1),
                ),
                empty,
            ],
        )
        geometry = SurfaceGeometry(
            key=request.geometry_key,
            **arrays,
            epsilon_mm=document["epsilon_mm"],
            diagnostics=SurfaceGeometryDiagnostics(**document["geometry_diagnostics"]),
            polydata=polydata,
        )
        result = SurfaceMappingResult(
            request,
            SurfaceMappingDiagnostics(**document["diagnostics"]),
            geometry.diagnostics,
            _storage=storage,
            **{name: storage.load(name) for name in document["result_arrays"]},
        )
        return SurfaceMappingBundle(
            result,
            geometry,
            *(
                storage.load(name)
                for name in (
                    "marker_points",
                    "marker_classes",
                    "associated_ids",
                    "overlap_points",
                )
            ),
            document["marker_count"],
            document["overlap_count"],
            document["overlap_segment_count"],
            tuple(document["endpoint_contributions_range"]),
            tuple(document["endpoint_density_range"]),
        )
    except BaseException:
        storage.close()
        raise


def run_surface_mapping(
    layer: dict,
    request: SurfaceMappingRequest,
    streamlines: Sequence[np.ndarray],
    cancel_check: Callable[[], bool],
    progress: Callable[[str], None] | None = None,
    *,
    temporary_parent: str | Path | None = None,
) -> SurfaceMappingBundle:
    """Stage on the coordinating worker; return only complete mapped results.

    The caller must retain the TRX backing owner until this function returns.
    The child owns all native analysis and is terminated after a 250 ms cancel
    grace period. Mappings close before Windows temporary directories retire.
    """
    if request.parameters.mode == "both":
        parameters = request.parameters
        bundle = run_surface_mapping(
            layer,
            replace(request, parameters=replace(parameters, mode="endpoint_proximity")),
            streamlines, cancel_check, progress, temporary_parent=temporary_parent,
        )
        try:
            bundle.companion = run_surface_mapping(
                layer,
                replace(
                    request,
                    parameters=replace(parameters, mode="surface_intersections"),
                ),
                streamlines, cancel_check, progress, temporary_parent=temporary_parent,
            )
            _check_cancel(cancel_check)
            return bundle
        except BaseException:
            bundle.close()
            raise
    directory = TemporaryDirectory(prefix="tractedit-surface-", dir=temporary_parent)
    transferred = False
    try:
        if progress:
            progress("staging")
        _check_cancel(cancel_check)
        _stage(Path(directory.name), layer, request, streamlines, cancel_check)
        _check_cancel(cancel_check)
        if progress:
            progress("spawn")
        from ..mesh.process import start_process

        exitcode = run_isolated_process(
            _compute, (directory.name,), cancel_check=lambda: _check_cancel(cancel_check),
            progress=progress, channel=True, cooperative=True, start=start_process,
        )
        error = Path(directory.name) / "error.json"
        if error.exists():
            raise RuntimeError(json.loads(error.read_text("utf-8")))
        if exitcode:
            raise RuntimeError(
                f"Surface mapping process exited with code {exitcode}"
            )
        for path in Path(directory.name).iterdir():
            if path.is_file():
                path.unlink()
        if progress:
            progress("restore")
        bundle = _restore(directory, request)
        transferred = True
        return bundle
    finally:
        if not transferred:
            directory.cleanup()
