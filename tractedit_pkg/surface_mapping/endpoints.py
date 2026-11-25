# -*- coding: utf-8 -*-
"""Nearest-triangle endpoint mapping in scanner RAS millimetres."""

from collections.abc import Callable, Sequence

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
    shared_location,
    validate_query,
)


def _terminal(
    line: np.ndarray, end: int, cancel_check: Callable[[], bool] | None
) -> tuple[np.ndarray, np.ndarray | None]:
    point = np.asarray(line[0 if end == 0 else len(line) - 1], dtype=np.float64)
    if not np.all(np.isfinite(point)):
        return point, None
    for start in range(1, len(line), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel_check)
        stop = min(start + QUERY_CHUNK_ITEMS, len(line))
        block = np.asarray(
            (
                line[start:stop]
                if end == 0
                else line[len(line) - stop : len(line) - start][::-1]
            ),
            dtype=np.float64,
        )
        delta = point - block
        distinct = np.flatnonzero(np.any(delta != 0, axis=1))
        if len(distinct):
            tangent = delta[distinct[0]]
            norm = np.linalg.norm(tangent)
            return point, tangent / norm if np.isfinite(norm) and norm > 0 else None
    return point, None


def _associate(
    geometry: SurfaceGeometry,
    point: np.ndarray,
    tangent: np.ndarray | None,
    maximum: float,
    record: np.void,
    cancel_check: Callable[[], bool] | None,
) -> None:
    from vtkmodules.vtkCommonCore import reference
    from vtkmodules.vtkCommonDataModel import vtkGenericCell

    position = [np.nan] * 3
    cell_id, sub_id, distance2 = reference(-1), reference(-1), reference(np.nan)
    geometry.get_locator().FindClosestPoint(
        point, position, vtkGenericCell(), cell_id, sub_id, distance2
    )
    if int(cell_id) < 0 or not np.isfinite(float(distance2)):
        record["status"] = EndpointStatus.INVALID
        return
    distance = float(np.sqrt(max(0, float(distance2))))
    epsilon = geometry.epsilon_mm
    record["distance_mm"] = distance
    if distance > maximum + epsilon:
        record["status"] = EndpointStatus.BEYOND_DISTANCE
        return
    radius = distance + epsilon
    rows, positions, weights, distances = [], [], [], []
    for block in candidate_blocks(
        geometry, point - radius, point + radius, cancel_check
    ):
        q, bary, d = closest_points(
            point, geometry.points_ras[geometry.triangles[block]]
        )
        keep = d <= distance + epsilon
        rows.append(block[keep])
        positions.append(q[keep])
        weights.append(bary[keep])
        distances.append(d[keep])
    if not rows or not any(len(block) for block in rows):
        raise RuntimeError("Surface locator candidate search lost its closest cell")
    rows = np.concatenate(rows)
    positions, weights, distances = map(np.concatenate, (positions, weights, distances))
    minimum = distances.min()
    tied = distances <= minimum + epsilon
    rows, positions, weights, distances = (
        array[tied] for array in (rows, positions, weights, distances)
    )
    chosen = int(np.argmin(geometry.triangle_ids[rows]))
    row = int(rows[chosen])
    record["status"] = (
        EndpointStatus.ASSIGNED
        if shared_location(geometry, rows, positions, weights)
        else EndpointStatus.AMBIGUOUS
    )
    record["triangle_id"] = geometry.triangle_ids[row]
    record["source_cell_id"] = geometry.source_cell_ids[row]
    record["position_ras"] = positions[chosen]
    record["barycentric"] = weights[chosen]
    record["distance_mm"] = distances[chosen]
    record["normal_nonunique"] = np.any(
        np.linalg.norm(np.cross(geometry.normals[rows], geometry.normals[row]), axis=1)
        > 64 * np.finfo(float).eps
    )
    if tangent is not None:
        record["angle_deg"] = np.degrees(
            np.arccos(np.clip(abs(np.dot(tangent, geometry.normals[row])), 0, 1))
        )


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
    count = 2 * len(request.streamline_ids)
    if endpoint_records is None:
        records = empty_records(count, ENDPOINT_DTYPE)
    else:
        if (
            endpoint_records.shape != (count,)
            or endpoint_records.dtype != ENDPOINT_DTYPE
            or not endpoint_records.flags.writeable
        ):
            raise ValueError("Expected a writable endpoint table with two rows per ID")
        records = endpoint_records
        for start in range(0, count, QUERY_CHUNK_ITEMS):
            _check_cancel(cancel_check)
            stop = min(count, start + QUERY_CHUNK_ITEMS)
            records[start:stop] = empty_records(stop - start, ENDPOINT_DTYPE)
    records["status"] = EndpointStatus.INVALID
    contributions = np.zeros(len(geometry.points_ras), dtype=np.float64)
    valid = 0
    for index, streamline_id in enumerate(request.streamline_ids):
        _check_cancel(cancel_check)
        line = streamlines[int(streamline_id)]
        pair = records[2 * index : 2 * index + 2]
        pair["streamline_id"], pair["endpoint"] = streamline_id, [0, 1]
        if len(line.shape) != 2 or line.shape[1] != 3 or not len(line):
            continue
        for end in (0, 1):
            point, tangent = _terminal(line, end, cancel_check)
            if np.all(np.isfinite(point)):
                _associate(
                    geometry,
                    point,
                    tangent,
                    request.parameters.max_distance_mm,
                    pair[end],
                    cancel_check,
                )
        valid += int(np.all(pair["status"] != EndpointStatus.INVALID))
    if progress:
        progress("aggregation")
    statuses = np.zeros(len(EndpointStatus), dtype=np.int64)
    for start in range(0, len(records), QUERY_CHUNK_ITEMS):
        _check_cancel(cancel_check)
        block = records[start : start + QUERY_CHUNK_ITEMS]
        statuses += np.bincount(block["status"], minlength=len(EndpointStatus))
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
