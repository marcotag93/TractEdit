# -*- coding: utf-8 -*-
"""Bounded geometric primitives shared by the two surface analysis engines."""

from collections.abc import Callable, Iterator, Sequence

import numpy as np

from .models import (
    QUERY_CHUNK_ITEMS,
    SurfaceGeometry,
    SurfaceMappingRequest,
    _check_cancel,
)


def validate_query(
    geometry: SurfaceGeometry,
    request: SurfaceMappingRequest,
    streamlines: Sequence[np.ndarray],
    mode: str,
) -> None:
    if request.geometry_key != geometry.key:
        raise ValueError("Request does not own this surface geometry")
    if request.parameters.mode != mode:
        raise ValueError("Request mode does not match the analysis engine")
    if request.streamline_ids[-1] >= len(streamlines):
        raise ValueError("Requested streamline ID is outside the tractogram")


def candidate_blocks(
    geometry: SurfaceGeometry,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    cancel_check: Callable[[], bool] | None,
    *,
    locator=None,
    ids=None,
    check_initial_cancel: bool = True,
) -> Iterator[np.ndarray]:
    if check_initial_cancel:
        _check_cancel(cancel_check)
    if ids is None:
        from vtkmodules.vtkCommonCore import vtkIdList

        ids = vtkIdList()
    else:
        ids.Reset()
    if locator is None:
        locator = geometry.get_locator()
    locator.FindCellsWithinBounds(
        (
            float(lower[0]),
            float(upper[0]),
            float(lower[1]),
            float(upper[1]),
            float(lower[2]),
            float(upper[2]),
        ),
        ids,
    )
    count = ids.GetNumberOfIds()
    get_id = ids.GetId
    for start in range(0, count, QUERY_CHUNK_ITEMS):
        if start:
            _check_cancel(cancel_check)
        stop = min(start + QUERY_CHUNK_ITEMS, count)
        yield np.fromiter(
            map(get_id, range(start, stop)),
            dtype=np.int64,
            count=stop - start,
        )


def _plane_weights(
    a: np.ndarray, b: np.ndarray, relative: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    denominator = nx * nx + ny * ny + nz * nz
    rx, ry, rz = relative[:, 0], relative[:, 1], relative[:, 2]
    v = (
        (ry * bz - rz * by) * nx
        + (rz * bx - rx * bz) * ny
        + (rx * by - ry * bx) * nz
    ) / denominator
    w = (
        (ay * rz - az * ry) * nx
        + (az * rx - ax * rz) * ny
        + (ax * ry - ay * rx) * nz
    ) / denominator
    return v, w


def barycentric(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Plane coordinates using translated cross products, avoiding Gram cancellation."""
    point = np.asarray(point, dtype=np.float64)
    v0 = triangles[:, 0]
    a = triangles[:, 1] - v0
    b = triangles[:, 2] - v0
    v, w = _plane_weights(a, b, point - v0)
    weights = np.empty((len(triangles), 3), dtype=np.float64)
    weights[:, 0] = 1.0 - v - w
    weights[:, 1] = v
    weights[:, 2] = w
    return weights


def closest_points(
    point: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = np.asarray(point, dtype=np.float64)
    v0 = triangles[:, 0]
    a = triangles[:, 1] - v0
    b = triangles[:, 2] - v0
    relative = point - v0
    v, w = _plane_weights(a, b, relative)
    u = 1.0 - v - w
    weights = np.empty((len(triangles), 3), dtype=np.float64)
    weights[:, 0] = u
    weights[:, 1] = v
    weights[:, 2] = w
    offset = v[:, None] * a + w[:, None] * b
    positions = v0 + offset
    diff = offset - relative
    dist2 = (
        diff[:, 0] * diff[:, 0]
        + diff[:, 1] * diff[:, 1]
        + diff[:, 2] * diff[:, 2]
    )
    outside = (u < 0.0) | (v < 0.0) | (w < 0.0)
    if np.any(outside):
        out_idx = np.flatnonzero(outside)
        p_out = point if point.ndim == 1 else point[out_idx]
        v0_o = v0[out_idx]
        v1_o = triangles[out_idx, 1]
        v2_o = triangles[out_idx, 2]
        a_o = a[out_idx]
        b_o = b[out_idx]
        rel0 = relative[out_idx]
        t0 = np.clip(
            (
                rel0[:, 0] * a_o[:, 0]
                + rel0[:, 1] * a_o[:, 1]
                + rel0[:, 2] * a_o[:, 2]
            )
            / (
                a_o[:, 0] * a_o[:, 0]
                + a_o[:, 1] * a_o[:, 1]
                + a_o[:, 2] * a_o[:, 2]
            ),
            0.0,
            1.0,
        )
        off0 = t0[:, None] * a_o
        d0 = off0 - rel0
        best_d2 = d0[:, 0] * d0[:, 0] + d0[:, 1] * d0[:, 1] + d0[:, 2] * d0[:, 2]
        best_pos = v0_o + off0
        best_w = np.zeros((len(out_idx), 3), dtype=np.float64)
        best_w[:, 0] = 1.0 - t0
        best_w[:, 1] = t0
        for i, j, start, edge in ((1, 2, v1_o, v2_o - v1_o), (2, 0, v2_o, -b_o)):
            rel_e = p_out - start
            t = np.clip(
                (
                    rel_e[:, 0] * edge[:, 0]
                    + rel_e[:, 1] * edge[:, 1]
                    + rel_e[:, 2] * edge[:, 2]
                )
                / (
                    edge[:, 0] * edge[:, 0]
                    + edge[:, 1] * edge[:, 1]
                    + edge[:, 2] * edge[:, 2]
                ),
                0.0,
                1.0,
            )
            off_e = t[:, None] * edge
            de = off_e - rel_e
            d2 = (
                de[:, 0] * de[:, 0]
                + de[:, 1] * de[:, 1]
                + de[:, 2] * de[:, 2]
            )
            choose = d2 < best_d2
            if np.any(choose):
                best_d2[choose] = d2[choose]
                best_pos[choose] = start[choose] + off_e[choose]
                best_w[choose] = 0.0
                best_w[choose, i] = 1.0 - t[choose]
                best_w[choose, j] = t[choose]
        positions[out_idx] = best_pos
        weights[out_idx] = best_w
        dist2[out_idx] = best_d2
    return positions, weights, np.sqrt(dist2)


def point_support(
    geometry: SurfaceGeometry, row: int, weights: np.ndarray
) -> tuple[int, ...]:
    ids = geometry.triangles[row]
    near = weights <= geometry.barycentric_tolerances[row]
    if not near.any():
        return tuple(sorted(ids))
    vertices = geometry.points_ras[ids]
    relative = weights[1:] @ (vertices[1:] - vertices[0])
    distances = np.linalg.norm((vertices - vertices[0]) - relative, axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] <= geometry.epsilon_mm:
        return (ids[nearest],)
    if near.sum() == 1 and np.all(weights >= 0):
        return tuple(sorted(ids[~near]))
    starts = vertices - vertices[0]
    edges = np.roll(starts, -1, axis=0) - starts
    offsets = relative - starts
    t = np.einsum("ij,ij->i", offsets, edges) / np.einsum("ij,ij->i", edges, edges)
    distances = np.linalg.norm(offsets - np.clip(t, 0, 1)[:, None] * edges, axis=1)
    edge = int(np.argmin(distances))
    if distances[edge] <= geometry.epsilon_mm:
        return tuple(sorted((ids[edge], ids[(edge + 1) % 3])))
    return tuple(sorted(ids))


def shared_location(
    geometry: SurfaceGeometry,
    rows: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
) -> bool:
    """Only a common indexed edge/vertex can collapse distinct triangle hits."""
    if len(rows) <= 1:
        return True
    if np.any(np.linalg.norm(positions - positions[0], axis=1) > geometry.epsilon_mm):
        return False
    support = point_support(geometry, int(rows[0]), weights[0])
    if len(support) == 3:
        return False
    faces = set()
    for row, weight in zip(rows, weights):
        face = tuple(sorted(geometry.triangles[row]))
        if face in faces or point_support(geometry, int(row), weight) != support:
            return False
        faces.add(face)
    return True


def empty_records(count: int, dtype: np.dtype) -> np.ndarray:
    records = np.zeros(count, dtype=dtype)
    for name in dtype.names:
        field = dtype[name]
        kind = field.base.kind
        if kind == "f":
            records[name] = np.nan
        elif kind == "i":
            records[name] = -1
    return records


def flat_streamline_source(
    streamlines: Sequence[np.ndarray],
    ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    if ids is not None:
        points = getattr(streamlines, "points", None)
        offsets = getattr(streamlines, "offsets", None)
        snap_ids = getattr(streamlines, "ids", None)
        if (
            isinstance(points, np.ndarray)
            and points.ndim == 2
            and points.shape[1] == 3
            and points.dtype.kind in ("f", "i", "u")
            and isinstance(offsets, np.ndarray)
            and offsets.ndim == 1
            and isinstance(snap_ids, np.ndarray)
            and len(offsets) == len(snap_ids) + 1
            and (snap_ids is ids or np.array_equal(snap_ids, ids))
        ):
            offs = offsets[:-1].astype(np.int64, copy=False)
            lens = offsets[1:].astype(np.int64, copy=False) - offs
            return points, offs, lens, True
    data = getattr(streamlines, "_data", None)
    src_offsets = getattr(streamlines, "_offsets", None)
    src_lengths = getattr(streamlines, "_lengths", None)
    if (
        isinstance(data, np.ndarray)
        and data.ndim == 2
        and data.shape[1] == 3
        and data.dtype.kind in ("f", "i", "u")
        and isinstance(src_offsets, np.ndarray)
        and isinstance(src_lengths, np.ndarray)
        and src_offsets.ndim == 1
        and src_lengths.ndim == 1
        and len(src_offsets) == len(streamlines)
        and len(src_lengths) == len(streamlines)
    ):
        return data, src_offsets, src_lengths, False
    return None
