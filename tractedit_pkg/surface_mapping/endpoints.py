# -*- coding: utf-8 -*-
"""Nearest-triangle endpoint mapping in scanner RAS millimetres."""

from collections.abc import Callable, Iterator, Sequence
from itertools import chain
import math

import numpy as np

from .models import (
    ENDPOINT_DTYPE,
    QUERY_CHUNK_ITEMS,
    EndpointStatus,
    SurfaceGeometry,
    SurfaceMappingDiagnostics,
    SurfaceMappingRequest,
    SurfaceMappingResult,
    _check_cancel,
)
from .queries import (
    candidate_blocks,
    closest_points,
    empty_records,
    flat_streamline_source,
    shared_location,
    validate_query,
)

_NORMAL_TOL2 = (64.0 * np.finfo(float).eps) ** 2


def _terminal(
    line: np.ndarray, end: int, cancel_check: Callable[[], bool] | None
) -> tuple[tuple[float, float, float], tuple[float, float, float] | None]:
    length = len(line)
    raw_point = line[0 if end == 0 else length - 1]
    px, py, pz = float(raw_point[0]), float(raw_point[1]), float(raw_point[2])
    point = (px, py, pz)
    if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
        return point, None
    if length < 2:
        return point, None
    neighbor = line[1 if end == 0 else length - 2]
    dx = px - float(neighbor[0])
    dy = py - float(neighbor[1])
    dz = pz - float(neighbor[2])
    if dx != 0.0 or dy != 0.0 or dz != 0.0:
        norm = math.hypot(dx, dy, dz)
        if math.isfinite(norm) and norm > 0.0:
            return point, (dx / norm, dy / norm, dz / norm)
        return point, None
    point_arr = np.array(point, dtype=np.float64)
    for start in range(1, length, QUERY_CHUNK_ITEMS):
        _check_cancel(cancel_check)
        stop = min(start + QUERY_CHUNK_ITEMS, length)
        block = np.asarray(
            (
                line[start:stop]
                if end == 0
                else line[length - stop : length - start][::-1]
            ),
            dtype=np.float64,
        )
        delta = point_arr - block
        distinct = np.flatnonzero(np.any(delta != 0, axis=1))
        if len(distinct):
            tangent = delta[distinct[0]]
            tx, ty, tz = float(tangent[0]), float(tangent[1]), float(tangent[2])
            norm = math.hypot(tx, ty, tz)
            if math.isfinite(norm) and norm > 0.0:
                return point, (tx / norm, ty / norm, tz / norm)
            return point, None
    return point, None


def _chunk_tangents(
    data: np.ndarray,
    offs: np.ndarray,
    lens: np.ndarray,
    pts: np.ndarray,
    nbrs: np.ndarray,
    finite_pts: np.ndarray,
    end: int,
    cancel_check: Callable[[], bool] | None,
) -> list[tuple[float, float, float] | None]:
    n = len(lens)
    out: list[tuple[float, float, float] | None] = [None] * n
    delta = pts - nbrs
    with np.errstate(over="ignore", invalid="ignore"):
        s2 = (
            delta[:, 0] * delta[:, 0]
            + delta[:, 1] * delta[:, 1]
            + delta[:, 2] * delta[:, 2]
        )
    ok = finite_pts & (lens >= 2) & np.isfinite(s2) & (s2 > 0.0)
    if np.any(ok):
        ok_idx = np.flatnonzero(ok)
        norms = np.sqrt(s2[ok_idx])
        tangents = (delta[ok_idx] / norms[:, None]).tolist()
        for k, t_vec in zip(ok_idx.tolist(), tangents):
            out[k] = (t_vec[0], t_vec[1], t_vec[2])
    zero_mask = finite_pts & (lens > 2) & (s2 == 0.0)
    if np.any(zero_mask):
        for k in np.flatnonzero(zero_mask).tolist():
            off_k = int(offs[k])
            len_k = int(lens[k])
            _, out[k] = _terminal(data[off_k : off_k + len_k], end, cancel_check)
    overflow_mask = (
        finite_pts
        & (lens >= 2)
        & ~ok
        & np.isfinite(delta).all(axis=1)
        & np.any(delta != 0.0, axis=1)
    )
    if np.any(overflow_mask):
        for k in np.flatnonzero(overflow_mask).tolist():
            dx, dy, dz = (
                float(delta[k, 0]),
                float(delta[k, 1]),
                float(delta[k, 2]),
            )
            norm = math.hypot(dx, dy, dz)
            if math.isfinite(norm) and norm > 0.0:
                out[k] = (dx / norm, dy / norm, dz / norm)
    return out


def _resolve_ties(
    geometry: SurfaceGeometry,
    rows: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
) -> tuple[int, EndpointStatus, bool]:
    chosen = int(np.argmin(geometry.triangle_ids[rows]))
    row = int(rows[chosen])
    status = (
        EndpointStatus.ASSIGNED
        if shared_location(geometry, rows, positions, weights)
        else EndpointStatus.AMBIGUOUS
    )
    nr = geometry.normals[rows]
    n0 = geometry.normals[row]
    cx = nr[:, 1] * n0[2] - nr[:, 2] * n0[1]
    cy = nr[:, 2] * n0[0] - nr[:, 0] * n0[2]
    cz = nr[:, 0] * n0[1] - nr[:, 1] * n0[0]
    nonunique = bool(np.any(cx * cx + cy * cy + cz * cz > _NORMAL_TOL2))
    return chosen, status, nonunique


def _associate_multiblock(
    geometry: SurfaceGeometry,
    point: tuple[float, float, float] | np.ndarray,
    tangent: tuple[float, float, float] | np.ndarray | None,
    limit: float,
    record: np.void,
    blocks: Iterator[np.ndarray],
) -> None:
    point_arr = np.asarray(point, dtype=np.float64)
    epsilon = geometry.epsilon_mm
    rows_list, pos_list, w_list, d_list = [], [], [], []
    for block in blocks:
        q, bary, d = closest_points(
            point_arr, geometry.points_ras[geometry.triangles[block]]
        )
        keep = d <= limit
        if np.any(keep):
            rows_list.append(block[keep])
            pos_list.append(q[keep])
            w_list.append(bary[keep])
            d_list.append(d[keep])
    if not rows_list:
        raise RuntimeError("Surface locator candidate search lost its closest cell")
    rows = np.concatenate(rows_list)
    positions = np.concatenate(pos_list)
    weights = np.concatenate(w_list)
    distances = np.concatenate(d_list)
    minimum = float(distances.min())
    tied = distances <= minimum + epsilon
    rows = rows[tied]
    positions = positions[tied]
    weights = weights[tied]
    distances = distances[tied]
    if len(rows) == 1:
        chosen, status, nonunique = 0, EndpointStatus.ASSIGNED, False
    else:
        chosen, status, nonunique = _resolve_ties(geometry, rows, positions, weights)
    row = int(rows[chosen])
    record["status"] = status
    record["triangle_id"] = geometry.triangle_ids[row]
    record["source_cell_id"] = geometry.source_cell_ids[row]
    record["position_ras"] = positions[chosen]
    record["barycentric"] = weights[chosen]
    record["distance_mm"] = distances[chosen]
    record["normal_nonunique"] = nonunique
    if tangent is not None:
        nr = geometry.normals[row]
        dot = abs(
            float(tangent[0]) * float(nr[0])
            + float(tangent[1]) * float(nr[1])
            + float(tangent[2]) * float(nr[2])
        )
        record["angle_deg"] = math.degrees(math.acos(1.0 if dot > 1.0 else dot))


def _flush_batch(
    geometry: SurfaceGeometry,
    rec_views: tuple[np.ndarray, ...],
    rec_indices: list[int],
    ep_points: list[tuple[float, float, float]],
    ep_tangents: list[tuple[float, float, float] | None],
    ep_limits: list[float],
    ep_blocks: list[np.ndarray],
    cancel_check: Callable[[], bool] | None,
) -> None:
    _check_cancel(cancel_check)
    batch_size = len(rec_indices)
    counts = np.fromiter((len(b) for b in ep_blocks), dtype=np.int64, count=batch_size)
    offsets = np.empty(batch_size, dtype=np.int64)
    offsets[0] = 0
    if batch_size > 1:
        np.cumsum(counts[:-1], out=offsets[1:])
    tri_rows = ep_blocks[0] if batch_size == 1 else np.concatenate(ep_blocks)
    points_batch = np.repeat(np.asarray(ep_points, dtype=np.float64), counts, axis=0)
    limits = np.asarray(ep_limits, dtype=np.float64)
    # Cull locator-bin triangles outside the exact distance box before projection.
    tri = geometry.triangles[tri_rows]
    pts = geometry.points_ras
    v0 = pts[tri[:, 0]]
    v1 = pts[tri[:, 1]]
    v2 = pts[tri[:, 2]]
    lim_pad = np.repeat(limits, counts) * (1.0 + 1e-9) + 1e-12
    px_p, py_p, pz_p = points_batch[:, 0], points_batch[:, 1], points_batch[:, 2]
    d0x, d1x, d2x = v0[:, 0] - px_p, v1[:, 0] - px_p, v2[:, 0] - px_p
    d0y, d1y, d2y = v0[:, 1] - py_p, v1[:, 1] - py_p, v2[:, 1] - py_p
    d0z, d1z, d2z = v0[:, 2] - pz_p, v1[:, 2] - pz_p, v2[:, 2] - pz_p
    aabb_ok = ~(
        ((d0x > lim_pad) & (d1x > lim_pad) & (d2x > lim_pad))
        | ((d0x < -lim_pad) & (d1x < -lim_pad) & (d2x < -lim_pad))
        | ((d0y > lim_pad) & (d1y > lim_pad) & (d2y > lim_pad))
        | ((d0y < -lim_pad) & (d1y < -lim_pad) & (d2y < -lim_pad))
        | ((d0z > lim_pad) & (d1z > lim_pad) & (d2z > lim_pad))
        | ((d0z < -lim_pad) & (d1z < -lim_pad) & (d2z < -lim_pad))
    )
    keep_counts = np.add.reduceat(aabb_ok, offsets, dtype=np.int64)
    if not np.all(keep_counts > 0):
        raise RuntimeError("Surface locator candidate search lost its closest cell")
    ok_idx = np.flatnonzero(aabb_ok)
    k_offsets = np.empty(batch_size, dtype=np.int64)
    k_offsets[0] = 0
    if batch_size > 1:
        np.cumsum(keep_counts[:-1], out=k_offsets[1:])
    tri_rows_ok = tri_rows[ok_idx]
    triangles_ok = np.empty((len(ok_idx), 3, 3), dtype=np.float64)
    triangles_ok[:, 0] = v0[ok_idx]
    triangles_ok[:, 1] = v1[ok_idx]
    triangles_ok[:, 2] = v2[ok_idx]
    positions, weights, distances = closest_points(points_batch[ok_idx], triangles_ok)
    minima = np.minimum.reduceat(distances, k_offsets)
    if not np.all(minima <= limits):
        raise RuntimeError("Surface locator candidate search lost its closest cell")
    bounds = np.minimum(limits, minima + geometry.epsilon_mm)
    tied_mask = distances <= np.repeat(bounds, keep_counts)
    tied_counts = np.add.reduceat(tied_mask, k_offsets, dtype=np.int64)
    tied_positions = np.flatnonzero(tied_mask)
    tied_starts = np.empty(batch_size, dtype=np.int64)
    tied_starts[0] = 0
    if batch_size > 1:
        np.cumsum(tied_counts[:-1], out=tied_starts[1:])
    chosen_abs = tied_positions[tied_starts]
    out_status = np.full(batch_size, EndpointStatus.ASSIGNED, dtype=np.uint8)
    out_nonunique = np.zeros(batch_size, dtype=bool)
    for k in np.flatnonzero(tied_counts > 1):
        t_start = int(tied_starts[k])
        t_abs = tied_positions[t_start : t_start + int(tied_counts[k])]
        chosen, status, nonunique = _resolve_ties(
            geometry, tri_rows_ok[t_abs], positions[t_abs], weights[t_abs]
        )
        chosen_abs[k] = t_abs[chosen]
        out_status[k] = status
        out_nonunique[k] = nonunique
    out_idx = np.asarray(rec_indices, dtype=np.int64)
    chosen_rows = tri_rows_ok[chosen_abs]
    (
        rec_status,
        rec_tri,
        rec_cell,
        rec_pos,
        rec_bary,
        rec_dist,
        rec_nonuniq,
        rec_angle,
    ) = rec_views
    rec_status[out_idx] = out_status
    rec_tri[out_idx] = geometry.triangle_ids[chosen_rows]
    rec_cell[out_idx] = geometry.source_cell_ids[chosen_rows]
    rec_pos[out_idx] = positions[chosen_abs]
    rec_bary[out_idx] = weights[chosen_abs]
    rec_dist[out_idx] = distances[chosen_abs]
    rec_nonuniq[out_idx] = out_nonunique
    t_positions = [k for k, t in enumerate(ep_tangents) if t is not None]
    if t_positions:
        t_pos = np.asarray(t_positions, dtype=np.int64)
        t_mat = np.asarray([ep_tangents[k] for k in t_positions], dtype=np.float64)
        n_mat = geometry.normals[chosen_rows[t_pos]]
        dots = np.abs(
            t_mat[:, 0] * n_mat[:, 0]
            + t_mat[:, 1] * n_mat[:, 1]
            + t_mat[:, 2] * n_mat[:, 2]
        )
        rec_angle[out_idx[t_pos]] = np.degrees(np.arccos(np.clip(dots, 0.0, 1.0)))


def map_endpoints(
    geometry: SurfaceGeometry,
    request: SurfaceMappingRequest,
    streamlines: Sequence[np.ndarray],
    *,
    cancel_check: Callable[[], bool] | None = None,
    endpoint_records: np.ndarray | None = None,
    progress: Callable[[str], None] | None = None,
) -> SurfaceMappingResult:
    """Map storage endpoints without reading or promoting the full tractogram.

    Finite endpoints remain queryable when no finite distinct terminal tangent
    exists. Empty/malformed lines and nonfinite endpoints have INVALID rows.
    Only ASSIGNED rows contribute to the unsmoothed vertex map and density.
    An optional writable ENDPOINT_DTYPE table has two rows per requested ID;
    its caller owns the backing storage, allowing isolated disk-backed queries.
    """
    validate_query(geometry, request, streamlines, "endpoint_proximity")
    _check_cancel(cancel_check)
    streamline_ids = request.streamline_ids
    count = 2 * len(streamline_ids)
    if endpoint_records is None:
        records = empty_records(count, ENDPOINT_DTYPE)
        records["status"] = EndpointStatus.INVALID
        records["streamline_id"][0::2] = streamline_ids
        records["streamline_id"][1::2] = streamline_ids
        records["endpoint"][0::2] = 0
        records["endpoint"][1::2] = 1
    else:
        if (
            endpoint_records.shape != (count,)
            or endpoint_records.dtype != ENDPOINT_DTYPE
            or not endpoint_records.flags.writeable
        ):
            raise ValueError("Expected a writable endpoint table with two rows per ID")
        records = endpoint_records
        template = empty_records(min(count, QUERY_CHUNK_ITEMS), ENDPOINT_DTYPE)
        template["status"] = EndpointStatus.INVALID
        template["endpoint"][0::2] = 0
        template["endpoint"][1::2] = 1
        for start in range(0, count, QUERY_CHUNK_ITEMS):
            _check_cancel(cancel_check)
            stop = min(count, start + QUERY_CHUNK_ITEMS)
            chunk = records[start:stop]
            chunk[:] = template[: stop - start]
            ids_slice = streamline_ids[start // 2 : stop // 2]
            chunk["streamline_id"][0::2] = ids_slice
            chunk["streamline_id"][1::2] = ids_slice
    from vtkmodules.vtkCommonCore import reference, vtkIdList
    from vtkmodules.vtkCommonDataModel import vtkGenericCell

    locator = geometry.get_locator()
    cell = vtkGenericCell()
    ids = vtkIdList()
    closest_pos = [0.0, 0.0, 0.0]
    cell_id = reference(-1)
    sub_id = reference(-1)
    distance2 = reference(math.nan)
    epsilon = geometry.epsilon_mm
    maximum = request.parameters.max_distance_mm
    max_allowed = maximum + epsilon

    rec_status = records["status"]
    rec_dist = records["distance_mm"]
    rec_views = (
        rec_status,
        records["triangle_id"],
        records["source_cell_id"],
        records["position_ras"],
        records["barycentric"],
        rec_dist,
        records["normal_nonunique"],
        records["angle_deg"],
    )

    rec_indices: list[int] = []
    ep_points: list[tuple[float, float, float]] = []
    ep_tangents: list[tuple[float, float, float] | None] = []
    ep_limits: list[float] = []
    ep_blocks: list[np.ndarray] = []
    batch_pairs = 0

    def flush() -> None:
        nonlocal batch_pairs
        if rec_indices:
            _flush_batch(
                geometry,
                rec_views,
                rec_indices,
                ep_points,
                ep_tangents,
                ep_limits,
                ep_blocks,
                cancel_check,
            )
            rec_indices.clear()
            ep_points.clear()
            ep_tangents.clear()
            ep_limits.clear()
            ep_blocks.clear()
            batch_pairs = 0

    def query_endpoint(
        rec_idx: int,
        point: tuple[float, float, float],
        tangent: tuple[float, float, float] | None,
    ) -> None:
        nonlocal batch_pairs
        px, py, pz = point
        cell_id.set(-1)
        distance2.set(math.nan)
        locator.FindClosestPoint(point, closest_pos, cell, cell_id, sub_id, distance2)
        d2_val = float(distance2)
        if int(cell_id) < 0 or not math.isfinite(d2_val):
            return
        distance = math.sqrt(d2_val) if d2_val > 0.0 else 0.0
        if distance > max_allowed:
            rec_dist[rec_idx] = distance
            rec_status[rec_idx] = EndpointStatus.BEYOND_DISTANCE
            return
        radius = distance + epsilon
        blocks = iter(
            candidate_blocks(
                geometry,
                (px - radius, py - radius, pz - radius),
                (px + radius, py + radius, pz + radius),
                cancel_check,
                locator=locator,
                ids=ids,
                check_initial_cancel=False,
            )
        )
        first_block = next(blocks, None)
        if first_block is None or not len(first_block):
            raise RuntimeError("Surface locator candidate search lost its closest cell")
        second_block = next(blocks, None)
        if second_block is not None:
            _associate_multiblock(
                geometry,
                point,
                tangent,
                radius,
                records[rec_idx],
                chain((first_block, second_block), blocks),
            )
            return
        block_len = len(first_block)
        if batch_pairs + block_len > QUERY_CHUNK_ITEMS and rec_indices:
            flush()
        rec_indices.append(rec_idx)
        ep_points.append(point)
        ep_tangents.append(tangent)
        ep_limits.append(radius)
        ep_blocks.append(first_block)
        batch_pairs += block_len

    flat = flat_streamline_source(streamlines, streamline_ids)
    if flat is not None:
        flat_data, flat_offs, flat_lens, by_pos = flat
        n_req = len(streamline_ids)
        for c_start in range(0, n_req, QUERY_CHUNK_ITEMS):
            _check_cancel(cancel_check)
            c_stop = min(c_start + QUERY_CHUNK_ITEMS, n_req)
            if by_pos:
                c_offs = flat_offs[c_start:c_stop]
                c_lens = flat_lens[c_start:c_stop]
            else:
                c_sids = streamline_ids[c_start:c_stop]
                c_offs = flat_offs[c_sids].astype(np.int64, copy=False)
                c_lens = flat_lens[c_sids].astype(np.int64, copy=False)
            nonempty = c_lens > 0
            if not np.any(nonempty):
                continue
            rel_indices = np.flatnonzero(nonempty)
            v_offs = c_offs[rel_indices]
            v_lens = c_lens[rel_indices]
            p0 = np.asarray(flat_data[v_offs], dtype=np.float64)
            n0 = np.asarray(
                flat_data[v_offs + np.minimum(v_lens - 1, 1)], dtype=np.float64
            )
            p1 = np.asarray(flat_data[v_offs + v_lens - 1], dtype=np.float64)
            n1 = np.asarray(
                flat_data[v_offs + np.maximum(v_lens - 2, 0)], dtype=np.float64
            )
            fin0 = np.isfinite(p0).all(axis=1)
            fin1 = np.isfinite(p1).all(axis=1)
            t0_list = _chunk_tangents(
                flat_data, v_offs, v_lens, p0, n0, fin0, 0, cancel_check
            )
            t1_list = _chunk_tangents(
                flat_data, v_offs, v_lens, p1, n1, fin1, 1, cancel_check
            )
            p0_list = p0.tolist()
            p1_list = p1.tolist()
            fin0_list = fin0.tolist()
            fin1_list = fin1.tolist()
            for step, (rel_i, f0, f1, pt0, pt1, tg0, tg1) in enumerate(
                zip(
                    rel_indices.tolist(),
                    fin0_list,
                    fin1_list,
                    p0_list,
                    p1_list,
                    t0_list,
                    t1_list,
                )
            ):
                if (step & 127) == 0:
                    _check_cancel(cancel_check)
                base = 2 * (c_start + rel_i)
                if f0:
                    query_endpoint(base, (pt0[0], pt0[1], pt0[2]), tg0)
                if f1:
                    query_endpoint(base + 1, (pt1[0], pt1[1], pt1[2]), tg1)
    else:
        for index, streamline_id in enumerate(streamline_ids):
            if (index & 127) == 0:
                _check_cancel(cancel_check)
            line = streamlines[int(streamline_id)]
            if len(line.shape) != 2 or line.shape[1] != 3 or not len(line):
                continue
            base = 2 * index
            for end in (0, 1):
                point, tangent = _terminal(line, end, cancel_check)
                px, py, pz = point
                if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
                    continue
                query_endpoint(base + end, point, tangent)

    flush()
    if progress:
        progress("aggregation")
    contributions = np.zeros(len(geometry.points_ras), dtype=np.float64)
    statuses = np.zeros(len(EndpointStatus), dtype=np.int64)
    valid = 0
    for start in range(0, len(records), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel_check)
        block = records[start : start + QUERY_CHUNK_ITEMS]
        statuses += np.bincount(block["status"], minlength=len(EndpointStatus))
        valid += int(
            np.count_nonzero(
                np.all(
                    block["status"].reshape(-1, 2) != EndpointStatus.INVALID,
                    axis=1,
                )
            )
        )
        assigned = block[block["status"] == EndpointStatus.ASSIGNED]
        rows = np.searchsorted(geometry.triangle_ids, assigned["triangle_id"])
        np.add.at(
            contributions,
            geometry.triangles[rows].ravel(),
            assigned["barycentric"].ravel(),
        )
    density_valid = geometry.vertex_areas_mm2 > 0
    density = np.full(len(contributions), np.nan)
    np.divide(
        contributions, geometry.vertex_areas_mm2, out=density, where=density_valid
    )
    diagnostics = SurfaceMappingDiagnostics(
        requested_streamlines=len(request.streamline_ids),
        valid_streamlines=valid,
        evaluated_endpoints=len(records),
        assigned_endpoints=int(statuses[EndpointStatus.ASSIGNED]),
        beyond_distance_endpoints=int(statuses[EndpointStatus.BEYOND_DISTANCE]),
        ambiguous_endpoints=int(statuses[EndpointStatus.AMBIGUOUS]),
        invalid_endpoints=int(statuses[EndpointStatus.INVALID]),
    )
    for array in (records, contributions, density, density_valid):
        array.flags.writeable = False
    _check_cancel(cancel_check)
    return SurfaceMappingResult(
        request=request,
        diagnostics=diagnostics,
        geometry_diagnostics=geometry.diagnostics,
        endpoints=records,
        endpoint_contributions=contributions,
        endpoint_density=density,
        endpoint_density_valid=density_valid,
    )
