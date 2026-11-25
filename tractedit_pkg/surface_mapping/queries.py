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
    lower: np.ndarray,
    upper: np.ndarray,
    cancel_check: Callable[[], bool] | None,
) -> Iterator[np.ndarray]:
    from vtkmodules.vtkCommonCore import vtkIdList

    _check_cancel(cancel_check)
    ids = vtkIdList()
    geometry.get_locator().FindCellsWithinBounds(
        np.column_stack((lower, upper)).ravel(), ids
    )
    for start in range(0, ids.GetNumberOfIds(), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel_check)
        yield np.fromiter(
            (
                ids.GetId(i)
                for i in range(
                    start, min(start + QUERY_CHUNK_ITEMS, ids.GetNumberOfIds())
                )
            ),
            dtype=np.int64,
        )


def barycentric(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Plane coordinates using translated cross products, avoiding Gram cancellation."""
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    normal = np.cross(a, b)
    denominator = np.einsum("ij,ij->i", normal, normal)
    relative = point - triangles[:, 0]
    v = np.einsum("ij,ij->i", np.cross(relative, b), normal) / denominator
    w = np.einsum("ij,ij->i", np.cross(a, relative), normal) / denominator
    return np.column_stack((1 - v - w, v, w))


def closest_points(
    point: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = barycentric(point, triangles)
    positions = triangles[:, 0] + (
        weights[:, 1, None] * (triangles[:, 1] - triangles[:, 0])
        + weights[:, 2, None] * (triangles[:, 2] - triangles[:, 0])
    )
    distances = np.linalg.norm(positions - point, axis=1)
    outside = np.any(weights < 0, axis=1)
    distances[outside] = np.inf
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, j] - triangles[:, i]
        t = np.clip(
            np.einsum("ij,ij->i", point - triangles[:, i], edge)
            / np.einsum("ij,ij->i", edge, edge),
            0,
            1,
        )
        q = triangles[:, i] + t[:, None] * edge
        d = np.linalg.norm(q - point, axis=1)
        choose = d < distances
        positions[choose], distances[choose] = q[choose], d[choose]
        weights[choose] = 0
        weights[choose, i] = 1 - t[choose]
        weights[choose, j] = t[choose]
    return positions, weights, distances


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
