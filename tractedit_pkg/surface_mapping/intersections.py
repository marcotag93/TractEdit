# -*- coding: utf-8 -*-
"""Original-segment surface events with coplanar clipping and disk-backed tables.

This engine is synchronous and CPU-only. Call it from the isolated worker
introduced by the controller, never from a GUI render callback.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import chain
import math
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
    _plane_weights,
    candidate_blocks,
    closest_points,
    empty_records,
    flat_streamline_source,
    point_support,
    shared_location,
    validate_query,
)

_PARALLEL_TOL = 64.0 * np.finfo(float).eps


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
            pbuf = self.buffers["points"]
            self._p_sid = pbuf["streamline_id"]
            self._p_seg = pbuf["segment_id"]
            self._p_t = pbuf["segment_t"]
            self._p_arc = pbuf["arc_length_mm"]
            self._p_pos = pbuf["position_ras"]
            self._p_bary = pbuf["barycentric"]
            self._p_cls = pbuf["class"]
            self._p_tri = pbuf["triangle_id"]
            self._p_cell = pbuf["source_cell_id"]
        except BaseException:
            self.close()
            raise

    def append(self, name: str, record: np.void) -> None:
        self.buffers[name][self.used[name]] = record
        self.used[name] += 1
        self.counts[name] += 1
        if self.used[name] == QUERY_CHUNK_ITEMS:
            self.flush(name)

    def append_point(
        self,
        streamline_id: int,
        segment_id: int,
        segment_t: float,
        arc_length_mm: float,
        position_ras: tuple[float, float, float] | np.ndarray,
        barycentric: tuple[float, float, float] | np.ndarray,
        event_class: int,
        triangle_id: int,
        source_cell_id: int,
    ) -> None:
        idx = self.used["points"]
        self._p_sid[idx] = streamline_id
        self._p_seg[idx] = segment_id
        self._p_t[idx] = segment_t
        self._p_arc[idx] = arc_length_mm
        self._p_pos[idx] = position_ras
        self._p_bary[idx] = barycentric
        self._p_cls[idx] = event_class
        self._p_tri[idx] = triangle_id
        self._p_cell[idx] = source_cell_id
        idx += 1
        self.used["points"] = idx
        self.counts["points"] += 1
        if idx == QUERY_CHUNK_ITEMS:
            self.flush("points")

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


@dataclass(slots=True)
class _Hit:
    row: int
    start: float
    end: float
    start_weights: tuple[float, float, float] | np.ndarray
    end_weights: tuple[float, float, float] | np.ndarray
    overlap: bool = False
    interior_normal: list[float] | None = None


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


def _degenerate_segment_hits(
    geometry: SurfaceGeometry,
    p: np.ndarray,
    blocks: Iterator[np.ndarray],
) -> list[_Hit]:
    epsilon = geometry.epsilon_mm
    hits = []
    for rows in blocks:
        _, weights, distances = closest_points(
            p, geometry.points_ras[geometry.triangles[rows]]
        )
        for i in np.flatnonzero(distances <= epsilon):
            hits.append(_Hit(int(rows[i]), 0.0, 0.0, weights[i], weights[i]))
    return hits


def _eval_segment_batch(
    geometry: SurfaceGeometry,
    seg_targets: list[tuple[dict[int, list[_Hit]], int]],
    seg_p: list[Sequence[float]],
    seg_q: list[Sequence[float]],
    seg_len: list[float],
    seg_blocks: list[np.ndarray],
    cancel_check: Callable[[], bool] | None,
) -> None:
    _check_cancel(cancel_check)
    batch_size = len(seg_blocks)
    counts = np.fromiter(
        (len(b) for b in seg_blocks), dtype=np.int64, count=batch_size
    )
    tri_rows = seg_blocks[0] if batch_size == 1 else np.concatenate(seg_blocks)
    seg_p_arr = np.asarray(seg_p, dtype=np.float64)
    seg_q_arr = np.asarray(seg_q, dtype=np.float64)
    seg_d_arr = seg_q_arr - seg_p_arr
    seg_len_arr = np.asarray(seg_len, dtype=np.float64)
    p_pair = np.repeat(seg_p_arr, counts, axis=0)
    q_pair = np.repeat(seg_q_arr, counts, axis=0)
    d_pair = np.repeat(seg_d_arr, counts, axis=0)
    len_pair = np.repeat(seg_len_arr, counts)
    pts_ras = geometry.points_ras
    v0 = pts_ras[geometry.triangles[tri_rows, 0]]
    nr = geometry.normals[tri_rows]
    nx, ny, nz = nr[:, 0], nr[:, 1], nr[:, 2]
    d0 = (
        (p_pair[:, 0] - v0[:, 0]) * nx
        + (p_pair[:, 1] - v0[:, 1]) * ny
        + (p_pair[:, 2] - v0[:, 2]) * nz
    )
    d1 = (
        (q_pair[:, 0] - v0[:, 0]) * nx
        + (q_pair[:, 1] - v0[:, 1]) * ny
        + (q_pair[:, 2] - v0[:, 2]) * nz
    )
    nd = d_pair[:, 0] * nx + d_pair[:, 1] * ny + d_pair[:, 2] * nz
    epsilon = geometry.epsilon_mm
    parallel = np.abs(nd) <= _PARALLEL_TOL * len_pair
    coplanar = parallel & (np.abs(d0) <= epsilon) & (np.abs(d1) <= epsilon)
    seg_of_pair = None
    if np.any(coplanar):
        seg_of_pair = np.repeat(np.arange(batch_size, dtype=np.int64), counts)
        for i in np.flatnonzero(coplanar):
            s = int(seg_of_pair[i])
            row = int(tri_rows[i])
            clipped = _clip_coplanar(
                seg_p_arr[s],
                seg_q_arr[s],
                pts_ras[geometry.triangles[row]],
                geometry.normals[row],
                geometry.barycentric_tolerances[row],
            )
            if clipped is not None:
                start, end, a, b = clipped
                length_s = float(seg_len_arr[s])
                seg_hits, seg_id = seg_targets[s]
                seg_hits.setdefault(seg_id, []).append(
                    _Hit(row, start, end, a, b, (end - start) * length_s > epsilon)
                )
    eligible = ~coplanar & (d0 != d1)
    if not np.any(eligible):
        return
    idx = np.flatnonzero(eligible)
    d0_e = d0[idx]
    params = d0_e / (d0_e - d1[idx])
    tol_t = epsilon / len_pair[idx]
    keep = (params >= -tol_t) & (params <= 1.0 + tol_t)
    if not np.any(keep):
        return
    idx = idx[keep]
    params = np.clip(params[keep], 0.0, 1.0)
    rows_k = tri_rows[idx]
    positions = p_pair[idx] + params[:, None] * d_pair[idx]
    v0_k = v0[idx]
    tri_k = geometry.triangles[rows_k]
    a_k = pts_ras[tri_k[:, 1]] - v0_k
    b_k = pts_ras[tri_k[:, 2]] - v0_k
    v_w, w_w = _plane_weights(a_k, b_k, positions - v0_k)
    u_w = 1.0 - v_w - w_w
    tols = geometry.barycentric_tolerances[rows_k]
    inside = (u_w >= -tols[:, 0]) & (v_w >= -tols[:, 1]) & (w_w >= -tols[:, 2])
    hit_j = np.flatnonzero(inside)
    if not len(hit_j):
        return
    if seg_of_pair is None:
        seg_of_pair = np.repeat(np.arange(batch_size, dtype=np.int64), counts)
    hit_idx = idx[hit_j]
    u_h = u_w[hit_j]
    v_h = v_w[hit_j]
    w_h = w_w[hit_j]
    tols_h = tols[hit_j]
    strict_h = (
        (u_h > tols_h[:, 0]) & (v_h > tols_h[:, 1]) & (w_h > tols_h[:, 2])
    ).tolist()
    norms_h = nr[hit_idx].tolist()
    for s, row, t_val, wu, wv, ww, is_strict, norm in zip(
        seg_of_pair[hit_idx].tolist(),
        rows_k[hit_j].tolist(),
        params[hit_j].tolist(),
        u_h.tolist(),
        v_h.tolist(),
        w_h.tolist(),
        strict_h,
        norms_h,
    ):
        seg_hits, seg_id = seg_targets[s]
        w_tup = (wu, wv, ww)
        hit = _Hit(
            row, t_val, t_val, w_tup, w_tup, False, norm if is_strict else None
        )
        bucket = seg_hits.get(seg_id)
        if bucket is None:
            seg_hits[seg_id] = [hit]
        else:
            bucket.append(hit)


@dataclass(slots=True)
class _Point:
    segment: int
    parameter: float
    arc: float
    position: tuple[float, float, float]
    hits: dict[int, tuple[float, float, float] | np.ndarray] = field(
        default_factory=dict
    )
    ambiguous: bool = False
    interior_normal: list[float] | None = None


@dataclass(eq=False, slots=True)
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
    origin = np.asarray(point.position, dtype=np.float64)

    def neighbor(indices):
        previous = direction = None
        for i in indices:
            candidate = np.asarray(line[i], dtype=np.float64)
            delta = candidate - origin
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


def _classify(
    geometry: SurfaceGeometry,
    line: np.ndarray,
    point: _Point,
    line_pts: list[list[float]] | None = None,
) -> Event:
    if point.ambiguous:
        return Event.AMBIGUOUS
    if len(point.hits) == 1:
        normal = point.interior_normal
        if normal is None:
            row, w = next(iter(point.hits.items()))
            tol0, tol1, tol2 = geometry.barycentric_tolerances[row].tolist()
            if w[0] > tol0 and w[1] > tol1 and w[2] > tol2:
                normal = geometry.normals[row].tolist()
        if normal is not None:
            epsilon = geometry.epsilon_mm
            nx, ny, nz = normal
            px, py, pz = point.position
            seg = point.segment
            v_before = line_pts[seg] if line_pts is not None else line[seg].tolist()
            dx = v_before[0] - px
            dy = v_before[1] - py
            dz = v_before[2] - pz
            lb = math.hypot(dx, dy, dz)
            if lb <= epsilon and seg > 0:
                v_before = (
                    line_pts[seg - 1]
                    if line_pts is not None
                    else line[seg - 1].tolist()
                )
                dx = v_before[0] - px
                dy = v_before[1] - py
                dz = v_before[2] - pz
                lb = math.hypot(dx, dy, dz)
            if lb > epsilon:
                dot_b = nx * dx + ny * dy + nz * dz
                if abs(dot_b) > epsilon:
                    after_idx = seg + 1
                    n_line = len(line_pts) if line_pts is not None else len(line)
                    v_after = (
                        line_pts[after_idx]
                        if line_pts is not None
                        else line[after_idx].tolist()
                    )
                    dx2 = v_after[0] - px
                    dy2 = v_after[1] - py
                    dz2 = v_after[2] - pz
                    la = math.hypot(dx2, dy2, dz2)
                    if la <= epsilon and after_idx + 1 < n_line:
                        v_after = (
                            line_pts[after_idx + 1]
                            if line_pts is not None
                            else line[after_idx + 1].tolist()
                        )
                        dx2 = v_after[0] - px
                        dy2 = v_after[1] - py
                        dz2 = v_after[2] - pz
                        la = math.hypot(dx2, dy2, dz2)
                    if la > epsilon:
                        dot_a = nx * dx2 + ny * dy2 + nz * dz2
                        if abs(dot_a) > epsilon:
                            return (
                                Event.CROSSING
                                if (dot_b > epsilon) != (dot_a > epsilon)
                                else Event.TOUCH
                            )
    rows = np.array(sorted(point.hits), dtype=np.int64)
    weights = np.array([point.hits[int(row)] for row in rows], dtype=np.float64)
    positions = np.tile(np.asarray(point.position, dtype=np.float64), (len(rows), 1))
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
    line_pts: list[list[float]] | None = None,
) -> Event:
    if len(point.hits) > 1:
        # Prefer true intersections over tolerance-only hits on adjacent coplanar faces.
        interior = {
            row
            for row, weights in point.hits.items()
            if min(weights) >= -64 * np.finfo(float).eps
        }
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
    triangle = next(iter(point.hits)) if len(point.hits) == 1 else min(point.hits)
    event = _classify(geometry, line, point, line_pts)
    storage.append_point(
        streamline_id,
        point.segment,
        point.parameter,
        point.arc,
        point.position,
        point.hits[triangle],
        event,
        int(geometry.triangle_ids[triangle]),
        int(geometry.source_cell_ids[triangle]),
    )
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


def _drain_state(
    frontier: float,
    pending: list[_Point],
    active: list[_Interval],
    geometry: SurfaceGeometry,
    line: np.ndarray,
    streamline_id: int,
    storage: _Tables,
    counts: np.ndarray,
    support: _SupportIndex,
    line_pts: list[list[float]] | None = None,
) -> tuple[list[_Point], list[_Interval], bool]:
    epsilon = geometry.epsilon_mm
    crossing = False
    if pending:
        retained_p = []
        for point in pending:
            if active and _cover_point(point, active, epsilon):
                continue
            if point.arc < frontier:
                event = _write_point(
                    storage, geometry, line, streamline_id, point, support, line_pts
                )
                counts[event] += 1
                crossing |= event == Event.CROSSING
            else:
                retained_p.append(point)
        pending = retained_p
    if active:
        retained_a = []
        for interval in active:
            if interval.end_arc < frontier:
                _write_interval(storage, geometry, streamline_id, interval)
                counts[Event.COPLANAR_OVERLAP] += 1
                counts[Event.AMBIGUOUS] += int(interval.ambiguous)
            else:
                retained_a.append(interval)
        active = retained_a
    return pending, active, crossing


def _apply_block_hits(
    geometry: SurfaceGeometry,
    support: _SupportIndex,
    line: np.ndarray,
    block: np.ndarray,
    lengths: Sequence[float] | np.ndarray,
    start_segment: int,
    seg_hits: dict[int, list[_Hit]],
    streamline_id: int,
    storage: _Tables,
    counts: np.ndarray,
    pending: list[_Point],
    active: list[_Interval],
    arc: float,
    block_pts: list[list[float]] | None = None,
    line_pts: list[list[float]] | None = None,
) -> tuple[list[_Point], list[_Interval], float, bool]:
    epsilon = geometry.epsilon_mm
    eps2 = epsilon * epsilon
    crossing = False
    for offset in range(len(lengths)):
        length = float(lengths[offset])
        segment = start_segment + offset
        hits = seg_hits.get(segment)
        if hits:
            if len(hits) > 1:
                hits.sort(key=lambda hit: (hit.start, hit.row))
            if block_pts is not None:
                p_vec = block_pts[offset]
                q_vec = block_pts[offset + 1]
            else:
                p_vec = block[offset].tolist()
                q_vec = block[offset + 1].tolist()
            p0, p1, p2 = p_vec
            dx_pq = q_vec[0] - p0
            dy_pq = q_vec[1] - p1
            dz_pq = q_vec[2] - p2
            for hit in hits:
                hit_frontier = arc + hit.start * length - epsilon
                if active or (pending and pending[0].arc < hit_frontier):
                    pending, active, did_cross = _drain_state(
                        hit_frontier,
                        pending,
                        active,
                        geometry,
                        line,
                        streamline_id,
                        storage,
                        counts,
                        support,
                        line_pts,
                    )
                    crossing |= did_cross
                if hit.overlap:
                    p_arr = block[offset]
                    delta_arr = block[offset + 1] - p_arr
                    incoming = _Interval(
                        segment,
                        segment,
                        hit.start,
                        hit.end,
                        arc + hit.start * length,
                        arc + hit.end * length,
                        p_arr + hit.start * delta_arr,
                        p_arr + hit.end * delta_arr,
                        hit.start_weights,
                        hit.end_weights,
                        hit.row,
                        hit.row,
                        {hit.row},
                    )
                    _merge_intervals(geometry, support, active, incoming)
                else:
                    t = hit.start
                    px = p0 + t * dx_pq
                    py = p1 + t * dy_pq
                    pz = p2 + t * dz_pq
                    location = arc + t * length
                    for point in pending:
                        if abs(point.arc - location) <= epsilon:
                            ppx, ppy, ppz = point.position
                            dx = ppx - px
                            dy = ppy - py
                            dz = ppz - pz
                            if dx * dx + dy * dy + dz * dz <= eps2:
                                point.hits[hit.row] = hit.start_weights
                                point.interior_normal = (
                                    hit.interior_normal
                                    if len(point.hits) == 1
                                    else None
                                )
                                break
                    else:
                        pending.append(
                            _Point(
                                segment,
                                t,
                                location,
                                (px, py, pz),
                                {hit.row: hit.start_weights},
                                False,
                                hit.interior_normal,
                            )
                        )
        # Retain the last vertex until the following segment has contributed.
        seg_frontier = arc + length - epsilon
        if active or (pending and pending[0].arc < seg_frontier):
            pending, active, did_cross = _drain_state(
                seg_frontier,
                pending,
                active,
                geometry,
                line,
                streamline_id,
                storage,
                counts,
                support,
                line_pts,
            )
            crossing |= did_cross
        arc += length
    return pending, active, arc, crossing


def _validate_short_line(
    line: np.ndarray,
) -> tuple[np.ndarray, list[list[float]], list[float]] | None:
    block = np.asarray(line, dtype=np.float64)
    n_pts = len(block)
    if n_pts <= 8:
        pts = block.tolist()
        lengths: list[float] = []
        p0, p1, p2 = pts[0]
        if not (math.isfinite(p0) and math.isfinite(p1) and math.isfinite(p2)):
            return None
        for idx in range(1, n_pts):
            q0, q1, q2 = pts[idx]
            if not (math.isfinite(q0) and math.isfinite(q1) and math.isfinite(q2)):
                return None
            seg_len = math.hypot(q0 - p0, q1 - p1, q2 - p2)
            if not math.isfinite(seg_len):
                return None
            lengths.append(seg_len)
            p0, p1, p2 = q0, q1, q2
        return block, pts, lengths
    if not np.isfinite(block).all():
        return None
    dx = block[1:, 0] - block[:-1, 0]
    dy = block[1:, 1] - block[:-1, 1]
    dz = block[1:, 2] - block[:-1, 2]
    with np.errstate(over="ignore", invalid="ignore"):
        s = dx * dx + dy * dy + dz * dz
    if np.isfinite(s).all():
        return block, block.tolist(), np.sqrt(s).tolist()
    with np.errstate(over="ignore", invalid="ignore"):
        lengths_arr = np.linalg.norm(block[1:] - block[:-1], axis=1)
    if not np.isfinite(lengths_arr).all():
        return None
    return block, block.tolist(), lengths_arr.tolist()


def _validate_long_line(
    line: np.ndarray, cancel_check: Callable[[], bool] | None
) -> bool:
    previous = None
    for start in range(0, len(line), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel_check)
        block = np.asarray(line[start : start + QUERY_CHUNK_ITEMS], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            lengths = np.linalg.norm(np.diff(block, axis=0), axis=1)
            boundary = (
                0.0 if previous is None else np.linalg.norm(block[0] - previous)
            )
        if not (
            np.isfinite(block).all()
            and np.isfinite(lengths).all()
            and np.isfinite(boundary)
        ):
            return False
        previous = block[-1]
    return True


def _enqueue_segments(
    geometry: SurfaceGeometry,
    block: np.ndarray,
    block_pts: list[list[float]],
    lengths: list[float],
    start_segment: int,
    seg_hits: dict[int, list[_Hit]],
    locator,
    vtk_ids,
    b_targets: list[tuple[dict[int, list[_Hit]], int]],
    b_p: list[Sequence[float]],
    b_q: list[Sequence[float]],
    b_len: list[float],
    b_blocks: list[np.ndarray],
    b_pairs: int,
    flush_cb: Callable[[], None],
    cancel_check: Callable[[], bool] | None,
    check_every_64: bool = False,
) -> int:
    epsilon = geometry.epsilon_mm
    for offset in range(len(lengths)):
        if check_every_64 and (offset & 63) == 0:
            _check_cancel(cancel_check)
        length = lengths[offset]
        segment = start_segment + offset
        p_vec = block_pts[offset]
        q_vec = block_pts[offset + 1]
        px, py, pz = p_vec
        qx, qy, qz = q_vec
        blocks = iter(
            candidate_blocks(
                geometry,
                (
                    (px if px < qx else qx) - epsilon,
                    (py if py < qy else qy) - epsilon,
                    (pz if pz < qz else qz) - epsilon,
                ),
                (
                    (px if px > qx else qx) + epsilon,
                    (py if py > qy else qy) + epsilon,
                    (pz if pz > qz else qz) + epsilon,
                ),
                cancel_check,
                locator=locator,
                ids=vtk_ids,
                check_initial_cancel=False,
            )
        )
        first_block = next(blocks, None)
        if first_block is None or not len(first_block):
            continue
        second_block = next(blocks, None)
        if length <= epsilon:
            all_blocks = (
                (first_block,)
                if second_block is None
                else chain((first_block, second_block), blocks)
            )
            hits = _degenerate_segment_hits(geometry, block[offset], all_blocks)
            if hits:
                seg_hits[segment] = hits
        elif second_block is not None:
            for blk in chain((first_block, second_block), blocks):
                if len(blk):
                    _eval_segment_batch(
                        geometry,
                        [(seg_hits, segment)],
                        [p_vec],
                        [q_vec],
                        [length],
                        [blk],
                        cancel_check,
                    )
        else:
            block_len = len(first_block)
            if b_pairs + block_len > QUERY_CHUNK_ITEMS and b_blocks:
                flush_cb()
                b_pairs = 0
            b_targets.append((seg_hits, segment))
            b_p.append(p_vec)
            b_q.append(q_vec)
            b_len.append(length)
            b_blocks.append(first_block)
            b_pairs += block_len
    return b_pairs


def _stream_long_line(
    geometry: SurfaceGeometry,
    support: _SupportIndex,
    line: np.ndarray,
    streamline_id: int,
    storage: _Tables,
    counts: np.ndarray,
    locator,
    vtk_ids,
    cancel_check: Callable[[], bool] | None,
) -> bool:
    active: list[_Interval] = []
    pending: list[_Point] = []
    arc, crossing = 0.0, False
    for start in range(0, len(line) - 1, QUERY_CHUNK_ITEMS):
        _check_cancel(cancel_check)
        block = np.asarray(
            line[start : min(start + QUERY_CHUNK_ITEMS + 1, len(line))],
            dtype=np.float64,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            lengths = np.linalg.norm(np.diff(block, axis=0), axis=1).tolist()
        block_pts = block.tolist()
        seg_hits: dict[int, list[_Hit]] = {}
        b_targets: list[tuple[dict[int, list[_Hit]], int]] = []
        b_p: list[Sequence[float]] = []
        b_q: list[Sequence[float]] = []
        b_len: list[float] = []
        b_blocks: list[np.ndarray] = []

        def flush_long() -> None:
            if b_blocks:
                _eval_segment_batch(
                    geometry, b_targets, b_p, b_q, b_len, b_blocks, cancel_check
                )
                b_targets.clear()
                b_p.clear()
                b_q.clear()
                b_len.clear()
                b_blocks.clear()

        _enqueue_segments(
            geometry,
            block,
            block_pts,
            lengths,
            start,
            seg_hits,
            locator,
            vtk_ids,
            b_targets,
            b_p,
            b_q,
            b_len,
            b_blocks,
            0,
            flush_long,
            cancel_check,
            check_every_64=True,
        )
        flush_long()
        pending, active, arc, did_cross = _apply_block_hits(
            geometry,
            support,
            line,
            block,
            lengths,
            start,
            seg_hits,
            streamline_id,
            storage,
            counts,
            pending,
            active,
            arc,
            block_pts,
        )
        crossing |= did_cross
    if pending or active:
        _, _, did_cross = _drain_state(
            math.inf,
            pending,
            active,
            geometry,
            line,
            streamline_id,
            storage,
            counts,
            support,
        )
        crossing |= did_cross
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
    from vtkmodules.vtkCommonCore import vtkIdList

    storage = _Tables(output_directory)
    counts = np.zeros(len(Event), dtype=np.int64)
    support = _SupportIndex(geometry)
    locator = geometry.get_locator()
    vtk_ids = vtkIdList()
    invalid_row = empty_records(1, INTERSECTION_DTYPE)[0]
    invalid_row["class"] = Event.INVALID
    valid = crossing_streamlines = 0

    b_targets: list[tuple[dict[int, list[_Hit]], int]] = []
    b_p: list[Sequence[float]] = []
    b_q: list[Sequence[float]] = []
    b_len: list[float] = []
    b_blocks: list[np.ndarray] = []
    b_pairs = 0
    completed_lines: list[
        tuple[int, np.ndarray, list[list[float]], list[float], dict[int, list[_Hit]]]
    ] = []
    completed_points = 0

    def flush_batch() -> None:
        nonlocal b_pairs, completed_points, crossing_streamlines
        if b_blocks:
            _eval_segment_batch(
                geometry, b_targets, b_p, b_q, b_len, b_blocks, cancel_check
            )
            b_targets.clear()
            b_p.clear()
            b_q.clear()
            b_len.clear()
            b_blocks.clear()
            b_pairs = 0
        if completed_lines:
            for sid, blk, pts, lens, hits_map in completed_lines:
                if not hits_map:
                    continue
                pending, active, _, did_cross = _apply_block_hits(
                    geometry,
                    support,
                    blk,
                    blk,
                    lens,
                    0,
                    hits_map,
                    sid,
                    storage,
                    counts,
                    [],
                    [],
                    0.0,
                    pts,
                    pts,
                )
                if pending or active:
                    _, _, tail_cross = _drain_state(
                        math.inf,
                        pending,
                        active,
                        geometry,
                        blk,
                        sid,
                        storage,
                        counts,
                        support,
                        pts,
                    )
                    did_cross |= tail_cross
                crossing_streamlines += int(did_cross)
            completed_lines.clear()
            completed_points = 0

    def enqueue_short_line(
        sid: int,
        block: np.ndarray,
        line_pts: list[list[float]],
        lengths: list[float],
    ) -> None:
        nonlocal b_pairs, completed_points
        seg_hits: dict[int, list[_Hit]] = {}
        b_pairs = _enqueue_segments(
            geometry,
            block,
            line_pts,
            lengths,
            0,
            seg_hits,
            locator,
            vtk_ids,
            b_targets,
            b_p,
            b_q,
            b_len,
            b_blocks,
            b_pairs,
            flush_batch,
            cancel_check,
        )
        completed_lines.append((sid, block, line_pts, lengths, seg_hits))
        completed_points += len(line_pts)
        if completed_points >= QUERY_CHUNK_ITEMS or len(completed_lines) >= 1024:
            flush_batch()

    def record_invalid(sid: int) -> None:
        flush_batch()
        invalid_row["streamline_id"] = sid
        storage.append("points", invalid_row)
        counts[Event.INVALID] += 1

    def process_line(sid: int, line: np.ndarray) -> None:
        nonlocal valid, crossing_streamlines
        valid_shape = len(line.shape) == 2 and line.shape[1] == 3 and len(line) >= 2
        if not valid_shape:
            record_invalid(sid)
            return
        if len(line) > QUERY_CHUNK_ITEMS + 1:
            flush_batch()
            if not _validate_long_line(line, cancel_check):
                record_invalid(sid)
                return
            valid += 1
            crossing_streamlines += int(
                _stream_long_line(
                    geometry,
                    support,
                    line,
                    sid,
                    storage,
                    counts,
                    locator,
                    vtk_ids,
                    cancel_check,
                )
            )
            return
        validated = _validate_short_line(line)
        if validated is None:
            record_invalid(sid)
            return
        valid += 1
        block, line_pts, lengths = validated
        enqueue_short_line(sid, block, line_pts, lengths)

    try:
        req_ids = request.streamline_ids
        flat = flat_streamline_source(streamlines, req_ids)
        if flat is not None:
            flat_data, flat_offs, flat_lens, by_pos = flat
            n_req = len(req_ids)
            chunk_pts_cap = QUERY_CHUNK_ITEMS * 4
            pos = 0
            while pos < n_req:
                _check_cancel(cancel_check)
                stop = min(pos + 1024, n_req)
                c_sids = req_ids[pos:stop]
                if by_pos:
                    c_offs = flat_offs[pos:stop]
                    c_lens = flat_lens[pos:stop]
                else:
                    c_offs = flat_offs[c_sids].astype(np.int64, copy=False)
                    c_lens = flat_lens[c_sids].astype(np.int64, copy=False)
                total_pts = int(c_lens.sum())
                if total_pts > chunk_pts_cap and len(c_lens) > 1:
                    fit = int(
                        np.searchsorted(
                            np.cumsum(c_lens), chunk_pts_cap, side="right"
                        )
                    )
                    stop = pos + max(1, fit)
                    c_sids = req_ids[pos:stop]
                    c_offs = c_offs[: stop - pos]
                    c_lens = c_lens[: stop - pos]
                    total_pts = int(c_lens.sum())
                fast_chunk = (
                    total_pts <= chunk_pts_cap
                    and bool(
                        np.all((c_lens >= 2) & (c_lens <= QUERY_CHUNK_ITEMS + 1))
                    )
                    and (
                        len(c_offs) == 1
                        or bool(np.all(c_offs[1:] == c_offs[:-1] + c_lens[:-1]))
                    )
                )
                if fast_chunk:
                    base = int(c_offs[0])
                    chunk_arr = np.asarray(
                        flat_data[base : base + total_pts], dtype=np.float64
                    )
                    if bool(np.isfinite(chunk_arr).all()):
                        dx = chunk_arr[1:, 0] - chunk_arr[:-1, 0]
                        dy = chunk_arr[1:, 1] - chunk_arr[:-1, 1]
                        dz = chunk_arr[1:, 2] - chunk_arr[:-1, 2]
                        with np.errstate(over="ignore", invalid="ignore"):
                            s2 = dx * dx + dy * dy + dz * dz
                        if bool(np.isfinite(s2).all()):
                            seg_lens_all = np.sqrt(s2).tolist()
                            pts_all = chunk_arr.tolist()
                            rel = 0
                            for k, (sid, ln) in enumerate(
                                zip(c_sids.tolist(), c_lens.tolist())
                            ):
                                if ((pos + k) & 63) == 0:
                                    _check_cancel(cancel_check)
                                valid += 1
                                enqueue_short_line(
                                    sid,
                                    chunk_arr[rel : rel + ln],
                                    pts_all[rel : rel + ln],
                                    seg_lens_all[rel : rel + ln - 1],
                                )
                                rel += ln
                            pos = stop
                            continue
                for k, (sid, off, ln) in enumerate(
                    zip(c_sids.tolist(), c_offs.tolist(), c_lens.tolist())
                ):
                    if ((pos + k) & 63) == 0:
                        _check_cancel(cancel_check)
                    process_line(sid, flat_data[off : off + ln])
                pos = stop
        else:
            for line_index, streamline_id in enumerate(req_ids):
                if (line_index & 63) == 0:
                    _check_cancel(cancel_check)
                sid = int(streamline_id)
                process_line(sid, streamlines[sid])
        flush_batch()
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
