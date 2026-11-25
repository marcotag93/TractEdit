# -*- coding: utf-8 -*-
"""Original-segment surface events with coplanar clipping and disk-backed tables.

This engine is synchronous and CPU-only. Call it from the isolated worker
introduced by the controller, never from a GUI render callback.
"""

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .models import (
    INTERSECTION_DTYPE,
    INTERVAL_CELL_DTYPE,
    INTERVAL_DTYPE,
    QUERY_CHUNK_ITEMS,
    IntersectionClass as Event,
    SurfaceGeometry,
    SurfaceMappingDiagnostics,
    SurfaceMappingRequest,
    SurfaceMappingResult,
    _check_cancel,
)
from .queries import (
    barycentric,
    candidate_blocks,
    closest_points,
    empty_records,
    point_support,
    shared_location,
    validate_query,
)


class _Tables:
    """Bounded append buffers; successful results own their temporary mappings."""

    def __init__(self, parent: str | Path):
        self.buffers = {}
        self.streams = {}
        self.used = {}
        self.counts = {}
        self.maps = []
        self.closed = True
        self.directory = TemporaryDirectory(prefix="surface_mapping_", dir=parent)
        self.closed = False
        try:
            for name, dtype in (
                ("points", INTERSECTION_DTYPE),
                ("intervals", INTERVAL_DTYPE),
                ("cells", INTERVAL_CELL_DTYPE),
            ):
                self.buffers[name] = np.empty(QUERY_CHUNK_ITEMS, dtype=dtype)
                self.streams[name] = (Path(self.directory.name) / name).open("wb")
                self.used[name] = self.counts[name] = 0
        except BaseException:
            self.close()
            raise

    def append(self, name: str, record: np.void) -> None:
        self.buffers[name][self.used[name]] = record
        self.used[name] += 1
        self.counts[name] += 1
        if self.used[name] == QUERY_CHUNK_ITEMS:
            self.flush(name)

    def flush(self, name: str) -> None:
        size = self.used[name]
        if size:
            payload = self.buffers[name][:size].tobytes()
            if self.streams[name].write(payload) != len(payload):
                raise OSError("Incomplete surface mapping table write")
            self.used[name] = 0

    def finish(self, name: str, order: list[str]) -> np.ndarray:
        self.flush(name)
        self.streams[name].close()
        dtype = self.buffers[name].dtype
        if not self.counts[name]:
            result = np.empty(0, dtype=dtype)
            result.flags.writeable = False
            return result
        path = Path(self.directory.name) / name
        mapped = np.memmap(path, dtype=dtype, mode="r+", shape=(self.counts[name],))
        self.maps.append(mapped)
        mapped.sort(order=order, kind="quicksort")
        mapped.flush()
        mapped._mmap.close()
        self.maps.pop()
        result = np.memmap(path, dtype=dtype, mode="r", shape=(self.counts[name],))
        self.maps.append(result)
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with ExitStack() as cleanup:
            cleanup.callback(self.directory.cleanup)
            for stream in self.streams.values():
                cleanup.callback(stream.close)
            for mapped in self.maps:
                cleanup.callback(mapped._mmap.close)

    def __del__(self) -> None:
        self.close()


@dataclass
class _Hit:
    row: int
    start: float
    end: float
    start_weights: np.ndarray
    end_weights: np.ndarray
    overlap: bool = False


def _clip_coplanar(
    p: np.ndarray,
    q: np.ndarray,
    triangle: np.ndarray,
    normal: np.ndarray,
    tolerance: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    axis = int(np.argmax(np.abs(normal)))
    axes = [i for i in range(3) if i != axis]
    vertices = triangle[:, axes]
    basis = (vertices[1:] - vertices[0]).T
    a = np.linalg.solve(basis, p[axes] - vertices[0])
    b = np.linalg.solve(basis, q[axes] - vertices[0])
    first, last = np.r_[1 - a.sum(), a], np.r_[1 - b.sum(), b]
    lower, upper = 0.0, 1.0
    for x, y, tol in zip(first, last, tolerance):
        if max(x, y) < -tol:
            return None
        if min(x, y) >= -tol:
            continue
        slope = y - x
        boundary = -tol if max(x, y) < 0 else 0.0
        if slope > 0:
            lower = max(lower, (boundary - x) / slope)
        elif slope < 0:
            upper = min(upper, (boundary - x) / slope)
    if lower > upper:
        # A near-edge contact may be admitted, but never widened into an interval.
        contact = np.clip((lower + upper) / 2, 0, 1)
        if np.any(first + contact * (last - first) < -tolerance):
            return None
        lower = upper = contact
    lower, upper = np.clip([lower, upper], 0, 1)
    delta = last - first
    return float(lower), float(upper), first + lower * delta, first + upper * delta


def _segment_hits(
    geometry: SurfaceGeometry,
    p: np.ndarray,
    q: np.ndarray,
    cancel_check: Callable[[], bool] | None,
) -> list[_Hit]:
    epsilon = geometry.epsilon_mm
    length = float(np.linalg.norm(q - p))
    hits = []
    for rows in candidate_blocks(
        geometry, np.minimum(p, q) - epsilon, np.maximum(p, q) + epsilon, cancel_check
    ):
        triangles = geometry.points_ras[geometry.triangles[rows]]
        if length <= epsilon:
            _, weights, distances = closest_points(p, triangles)
            for i in np.flatnonzero(distances <= epsilon):
                hits.append(_Hit(int(rows[i]), 0, 0, weights[i], weights[i]))
            continue
        d0 = np.einsum("ij,ij->i", p - triangles[:, 0], geometry.normals[rows])
        d1 = np.einsum("ij,ij->i", q - triangles[:, 0], geometry.normals[rows])
        parallel = np.abs(geometry.normals[rows] @ (q - p)) <= (
            64 * np.finfo(float).eps * length
        )
        coplanar = parallel & (np.abs(d0) <= epsilon) & (np.abs(d1) <= epsilon)
        for i in np.flatnonzero(coplanar):
            row = int(rows[i])
            clipped = _clip_coplanar(
                p,
                q,
                triangles[i],
                geometry.normals[row],
                geometry.barycentric_tolerances[row],
            )
            if clipped is not None:
                start, end, a, b = clipped
                hits.append(
                    _Hit(row, start, end, a, b, (end - start) * length > epsilon)
                )
        eligible = ~coplanar & (d0 != d1)
        indices = np.flatnonzero(eligible)
        parameters = d0[indices] / (d0[indices] - d1[indices])
        keep = (parameters >= -epsilon / length) & (parameters <= 1 + epsilon / length)
        indices, parameters = indices[keep], np.clip(parameters[keep], 0, 1)
        if not len(indices):
            continue
        positions = p + parameters[:, None] * (q - p)
        weights = barycentric(positions, triangles[indices])
        inside = np.all(
            weights >= -geometry.barycentric_tolerances[rows[indices]], axis=1
        )
        for i in np.flatnonzero(inside):
            hits.append(
                _Hit(
                    int(rows[indices[i]]),
                    float(parameters[i]),
                    float(parameters[i]),
                    weights[i],
                    weights[i],
                )
            )
    return hits


@dataclass
class _Point:
    segment: int
    parameter: float
    arc: float
    position: np.ndarray
    hits: dict[int, np.ndarray] = field(default_factory=dict)
    ambiguous: bool = False


@dataclass(eq=False)
class _Interval:
    start_segment: int
    end_segment: int
    start_t: float
    end_t: float
    start_arc: float
    end_arc: float
    start: np.ndarray
    end: np.ndarray
    start_weights: np.ndarray
    end_weights: np.ndarray
    start_row: int
    end_row: int
    rows: set[int]
    ambiguous: bool = False


def _neighbors(
    line: np.ndarray, point: _Point, epsilon: float, normals: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    def neighbor(indices):
        previous = direction = None
        for i in indices:
            candidate = np.asarray(line[i], dtype=np.float64)
            delta = candidate - point.position
            length = np.linalg.norm(delta)
            if length <= epsilon:
                continue
            if direction is not None:
                if np.array_equal(candidate, previous):
                    continue
                if (
                    np.linalg.norm(np.cross(direction, delta))
                    > 64 * np.finfo(float).eps * length
                    or np.dot(candidate - previous, direction) <= 0
                ):
                    break
            direction = delta / length
            previous = candidate
            if np.any(np.abs(normals @ delta) > epsilon):
                break
        return previous

    return (
        neighbor(range(point.segment, -1, -1)),
        neighbor(range(point.segment + 1, len(line))),
    )


def _classify(geometry: SurfaceGeometry, line: np.ndarray, point: _Point) -> Event:
    if point.ambiguous:
        return Event.AMBIGUOUS
    rows = np.array(sorted(point.hits), dtype=np.int64)
    weights = np.array([point.hits[int(row)] for row in rows])
    positions = np.tile(point.position, (len(rows), 1))
    if not shared_location(geometry, rows, positions, weights):
        return Event.AMBIGUOUS
    support = point_support(geometry, int(rows[0]), weights[0])
    if len(support) < 3:
        from vtkmodules.vtkCommonCore import vtkIdList

        incident = vtkIdList()
        geometry.polydata.GetPointCells(int(support[0]), incident)
        rows = np.asarray([
            row for row in (incident.GetId(i) for i in range(incident.GetNumberOfIds()))
            if len(support) == 1 or support[1] in geometry.triangles[row]
        ], dtype=np.int64)
        edges = {}
        for row in rows:
            face = geometry.triangles[row]
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((int(face[i]), int(face[j]))))
                if set(support).issubset(edge):
                    edges.setdefault(edge, []).append(
                        (int(row), 1 if face[i] < face[j] else -1)
                    )
        counts = [len(faces) for faces in edges.values()]
        if any(count > 2 for count in counts):
            return Event.AMBIGUOUS
        if any(count != 2 for count in counts):
            return Event.TOUCH
        adjacency = {int(row): [] for row in rows}
        for faces in edges.values():
            (a, da), (b, db) = faces
            adjacency[a].append((b, -da * db))
            adjacency[b].append((a, -da * db))
        pending = [int(rows[0])]
        signs = {pending[0]: 1}
        while pending:
            a = pending.pop()
            for b, relative in adjacency[a]:
                orientation = signs[a] * relative
                if b in signs:
                    if signs[b] != orientation:
                        return Event.AMBIGUOUS
                else:
                    signs[b] = orientation
                    pending.append(b)
        if len(signs) != len(rows):
            return Event.AMBIGUOUS
        normals = geometry.normals[rows] * np.array([signs[int(row)] for row in rows])[:, None]
    else:
        normals = geometry.normals[rows]
    before, after = _neighbors(line, point, geometry.epsilon_mm, normals)
    if before is None or after is None:
        return Event.TOUCH
    sides = []
    for position in (before, after):
        distances = normals @ (position - point.position)
        positive, negative = (
            distances > geometry.epsilon_mm,
            distances < -geometry.epsilon_mm,
        )
        if np.any(positive) and np.any(negative):
            return Event.AMBIGUOUS
        sides.append(1 if np.any(positive) else -1 if np.any(negative) else 0)
    return Event.CROSSING if sides[0] * sides[1] == -1 else Event.TOUCH


class _SupportIndex:
    def __init__(self, geometry: SurfaceGeometry):
        self.geometry = geometry
        self.cache = {}
        self.built = False

    def neighbors(self, row: int) -> set[int]:
        from vtkmodules.vtkCommonCore import vtkIdList

        if row in self.cache:
            return self.cache[row]
        geometry = self.geometry
        if not self.built:
            geometry.polydata.BuildLinks()
            self.built = True
        face = geometry.triangles[row]
        result = set()
        ids = vtkIdList()
        for i, j, opposite in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            geometry.polydata.GetCellEdgeNeighbors(row, int(face[i]), int(face[j]), ids)
            if ids.GetNumberOfIds() != 1:
                continue
            other = ids.GetId(0)
            other_face = geometry.triangles[other]
            extra = np.setdiff1d(other_face, face[[i, j]])
            if len(extra) != 1 or extra[0] == face[opposite]:
                continue
            if (
                np.linalg.norm(np.cross(geometry.normals[row], geometry.normals[other]))
                > 64 * np.finfo(float).eps
            ):
                result.add(other)
                continue
            origin, end = geometry.points_ras[face[[i, j]]]
            va = geometry.points_ras[face[opposite]] - origin
            vb = geometry.points_ras[extra[0]] - origin
            if np.dot(np.cross(end - origin, va), np.cross(end - origin, vb)) < 0:
                result.add(other)
        self.cache[row] = result
        return result

    def compatible(self, first: set[int], second: set[int]) -> bool:
        if first & second:
            return True
        if len(first) > len(second):
            first, second = second, first
        return any(self.neighbors(row) & second for row in first)


def _merge_intervals(
    geometry: SurfaceGeometry,
    support: _SupportIndex,
    active: list[_Interval],
    incoming: _Interval,
) -> None:
    epsilon = geometry.epsilon_mm
    merge = []
    for existing in active:
        if (
            incoming.start_arc > existing.end_arc + epsilon
            or existing.start_arc > incoming.end_arc + epsilon
        ):
            continue
        if support.compatible(existing.rows, incoming.rows):
            merge.append(existing)
        else:
            existing.ambiguous = incoming.ambiguous = True
    for existing in merge:
        active.remove(existing)
        if (existing.start_arc, existing.start_row) < (
            incoming.start_arc,
            incoming.start_row,
        ):
            incoming.start_segment, incoming.start_t, incoming.start_arc = (
                existing.start_segment,
                existing.start_t,
                existing.start_arc,
            )
            incoming.start, incoming.start_weights = (
                existing.start,
                existing.start_weights,
            )
            incoming.start_row = existing.start_row
        if (existing.end_arc, -existing.end_row) > (
            incoming.end_arc,
            -incoming.end_row,
        ):
            incoming.end_segment, incoming.end_t, incoming.end_arc = (
                existing.end_segment,
                existing.end_t,
                existing.end_arc,
            )
            incoming.end, incoming.end_weights = existing.end, existing.end_weights
            incoming.end_row = existing.end_row
        incoming.rows.update(existing.rows)
        incoming.ambiguous |= existing.ambiguous
    active.append(incoming)


def _write_interval(
    storage: _Tables, geometry: SurfaceGeometry, streamline_id: int, interval: _Interval
) -> None:
    row = empty_records(1, INTERVAL_DTYPE)[0]
    interval_id = storage.counts["intervals"]
    row["streamline_id"], row["interval_id"] = streamline_id, interval_id
    for prefix in ("start", "end"):
        row[prefix + "_segment_id"] = getattr(interval, prefix + "_segment")
        row[prefix + "_segment_t"] = getattr(interval, prefix + "_t")
        row[prefix + "_arc_length_mm"] = getattr(interval, prefix + "_arc")
        row[prefix + "_position_ras"] = getattr(interval, prefix)
        row[prefix + "_barycentric"] = getattr(interval, prefix + "_weights")
        triangle = getattr(interval, prefix + "_row")
        row[prefix + "_triangle_id"] = geometry.triangle_ids[triangle]
        row[prefix + "_source_cell_id"] = geometry.source_cell_ids[triangle]
    row["class"] = Event.AMBIGUOUS if interval.ambiguous else Event.COPLANAR_OVERLAP
    storage.append("intervals", row)
    relation = empty_records(1, INTERVAL_CELL_DTYPE)[0]
    relation["interval_id"] = interval_id
    for triangle in sorted(interval.rows):
        relation["triangle_id"] = geometry.triangle_ids[triangle]
        relation["source_cell_id"] = geometry.source_cell_ids[triangle]
        storage.append("cells", relation)


def _write_point(
    storage: _Tables,
    geometry: SurfaceGeometry,
    line: np.ndarray,
    streamline_id: int,
    point: _Point,
    support: _SupportIndex,
) -> Event:
    if len(point.hits) > 1:
        # Prefer true intersections over tolerance-only hits on adjacent coplanar faces.
        interior = {row for row, weights in point.hits.items()
                    if weights.min() >= -64 * np.finfo(float).eps}
        if interior and len(interior) < len(point.hits):
            remaining = point.hits.keys() - interior
            pending = list(interior)
            while pending and remaining:
                row = pending.pop()
                for other in support.neighbors(row) & remaining:
                    cross = np.cross(geometry.normals[row], geometry.normals[other])
                    if np.linalg.norm(cross) <= 64 * np.finfo(float).eps:
                        remaining.remove(other)
                        pending.append(other)
                        del point.hits[other]
    row = empty_records(1, INTERSECTION_DTYPE)[0]
    triangle = min(point.hits)
    event = _classify(geometry, line, point)
    row["streamline_id"], row["segment_id"] = streamline_id, point.segment
    row["segment_t"], row["arc_length_mm"] = point.parameter, point.arc
    row["position_ras"], row["barycentric"] = point.position, point.hits[triangle]
    row["class"] = event
    row["triangle_id"], row["source_cell_id"] = (
        geometry.triangle_ids[triangle],
        geometry.source_cell_ids[triangle],
    )
    storage.append("points", row)
    return event


def _cover_point(point: _Point, active: list[_Interval], epsilon: float) -> bool:
    for interval in active:
        if interval.start_arc - epsilon <= point.arc <= interval.end_arc + epsilon:
            point.hits = {
                row: weights
                for row, weights in point.hits.items()
                if row not in interval.rows
            }
            point.ambiguous = True
    return not point.hits


def _line_events(
    geometry: SurfaceGeometry,
    support: _SupportIndex,
    line: np.ndarray,
    streamline_id: int,
    storage: _Tables,
    counts: np.ndarray,
    cancel_check: Callable[[], bool] | None,
) -> bool:
    epsilon = geometry.epsilon_mm
    active, pending = [], []
    arc, crossing = 0.0, False

    def drain(frontier: float) -> None:
        nonlocal pending, active, crossing
        retained = []
        for point in pending:
            if _cover_point(point, active, epsilon):
                continue
            if point.arc < frontier:
                event = _write_point(storage, geometry, line, streamline_id, point, support)
                counts[event] += 1
                crossing |= event == Event.CROSSING
            else:
                retained.append(point)
        pending = retained
        retained = []
        for interval in active:
            if interval.end_arc < frontier:
                _write_interval(storage, geometry, streamline_id, interval)
                counts[Event.COPLANAR_OVERLAP] += 1
                counts[Event.AMBIGUOUS] += int(interval.ambiguous)
            else:
                retained.append(interval)
        active = retained

    for start in range(0, len(line) - 1, QUERY_CHUNK_ITEMS):
        _check_cancel(cancel_check)
        block = np.asarray(
            line[start : min(start + QUERY_CHUNK_ITEMS + 1, len(line))],
            dtype=np.float64,
        )
        lengths = np.linalg.norm(np.diff(block, axis=0), axis=1)
        for offset, length in enumerate(lengths):
            _check_cancel(cancel_check)
            segment = start + offset
            p, q = block[offset : offset + 2]
            hits = _segment_hits(geometry, p, q, cancel_check)
            for hit in sorted(hits, key=lambda hit: (hit.start, hit.row)):
                drain(arc + hit.start * length - epsilon)
                if hit.overlap:
                    incoming = _Interval(
                        segment,
                        segment,
                        hit.start,
                        hit.end,
                        arc + hit.start * length,
                        arc + hit.end * length,
                        p + hit.start * (q - p),
                        p + hit.end * (q - p),
                        hit.start_weights,
                        hit.end_weights,
                        hit.row,
                        hit.row,
                        {hit.row},
                    )
                    _merge_intervals(geometry, support, active, incoming)
                else:
                    position, location = (
                        p + hit.start * (q - p),
                        arc + hit.start * length,
                    )
                    for point in pending:
                        if (
                            abs(point.arc - location) <= epsilon
                            and np.linalg.norm(point.position - position) <= epsilon
                        ):
                            point.hits[hit.row] = hit.start_weights
                            break
                    else:
                        pending.append(
                            _Point(
                                segment,
                                hit.start,
                                location,
                                position,
                                {hit.row: hit.start_weights},
                            )
                        )
            # Retain the last vertex until the following segment has contributed.
            drain(arc + length - epsilon)
            arc += float(length)
    drain(np.inf)
    return crossing


def map_intersections(
    geometry: SurfaceGeometry,
    request: SurfaceMappingRequest,
    streamlines: Sequence[np.ndarray],
    *,
    output_directory: str | Path,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> SurfaceMappingResult:
    """Return sorted events on original segments, with explicit file ownership.

    Use the result as a context manager or call close() after its last consumer.
    A failed/cancelled query removes its unpublished files. Nonfinite or malformed
    streamlines produce one INVALID row and no partial geometric events.
    """
    validate_query(geometry, request, streamlines, "surface_intersections")
    _check_cancel(cancel_check)
    storage = _Tables(output_directory)
    counts = np.zeros(len(Event), dtype=np.int64)
    support = _SupportIndex(geometry)
    valid = crossing_streamlines = 0
    try:
        for streamline_id in request.streamline_ids:
            _check_cancel(cancel_check)
            line = streamlines[int(streamline_id)]
            finite = len(line.shape) == 2 and line.shape[1] == 3 and len(line) >= 2
            if finite:
                previous = None
                for start in range(0, len(line), QUERY_CHUNK_ITEMS):
                    _check_cancel(cancel_check)
                    block = np.asarray(
                        line[start : start + QUERY_CHUNK_ITEMS], dtype=np.float64
                    )
                    with np.errstate(over="ignore", invalid="ignore"):
                        lengths = np.linalg.norm(np.diff(block, axis=0), axis=1)
                        boundary = (
                            0.0
                            if previous is None
                            else np.linalg.norm(block[0] - previous)
                        )
                    if not (
                        np.isfinite(block).all()
                        and np.isfinite(lengths).all()
                        and np.isfinite(boundary)
                    ):
                        finite = False
                        break
                    previous = block[-1]
            if not finite:
                row = empty_records(1, INTERSECTION_DTYPE)[0]
                row["streamline_id"], row["class"] = streamline_id, Event.INVALID
                storage.append("points", row)
                counts[Event.INVALID] += 1
                continue
            valid += 1
            crossing_streamlines += _line_events(
                geometry,
                support,
                line,
                int(streamline_id),
                storage,
                counts,
                cancel_check,
            )
        if progress:
            progress("aggregation")
        points = storage.finish(
            "points", ["streamline_id", "arc_length_mm", "triangle_id"]
        )
        intervals = storage.finish(
            "intervals", ["streamline_id", "start_arc_length_mm", "interval_id"]
        )
        cells = storage.finish("cells", ["interval_id", "triangle_id"])
        _check_cancel(cancel_check)
        return SurfaceMappingResult(
            request=request,
            diagnostics=SurfaceMappingDiagnostics(
                requested_streamlines=len(request.streamline_ids),
                valid_streamlines=valid,
                crossing_events=int(counts[Event.CROSSING]),
                touch_events=int(counts[Event.TOUCH]),
                coplanar_intervals=int(counts[Event.COPLANAR_OVERLAP]),
                ambiguous_events=int(counts[Event.AMBIGUOUS]),
                invalid_events=int(counts[Event.INVALID]),
                crossing_streamlines=int(crossing_streamlines),
            ),
            geometry_diagnostics=geometry.diagnostics,
            intersections=points,
            intervals=intervals,
            interval_cells=cells,
            _storage=storage,
        )
    except BaseException:
        storage.close()
        raise
