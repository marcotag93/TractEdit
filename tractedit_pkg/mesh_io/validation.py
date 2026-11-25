from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

from collections.abc import Callable
import numpy as np
from .. import mesh_io


_VALIDATION_BLOCK_SIZE = 16_384


class MeshLoadError(ValueError):
    """The input cannot be safely published as a surface."""


class MeshLoadCancelled(Exception):
    """Cooperative cancellation; no partial result is returned."""


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise MeshLoadCancelled("Mesh loading cancelled")


def _validate_cells(
    cells: "vtkCellArray",
    point_count: int,
    kind: str,
    cancel_check: Callable[[], bool] | None,
    *, minimum_vertices: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    indices = vtk_to_numpy(cells.GetConnectivityArray())
    if (
        offsets.ndim != 1
        or indices.ndim != 1
        or offsets.dtype.kind not in "iu"
        or indices.dtype.kind not in "iu"
        or len(offsets) == 0
        or len(offsets) != cells.GetNumberOfCells() + 1
        or offsets[0] != 0
        or offsets[-1] != len(indices)
    ):
        raise MeshLoadError(f"Invalid {kind} connectivity offsets")
    for start in range(0, len(offsets) - 1, mesh_io._VALIDATION_BLOCK_SIZE):
        _check_cancel(cancel_check)
        block = offsets[start : start + mesh_io._VALIDATION_BLOCK_SIZE + 1]
        if np.any(block[1:] < block[:-1]):
            raise MeshLoadError(f"Non-monotonic {kind} connectivity offsets")
        if np.any(block[1:] - block[:-1] < minimum_vertices):
            minimum = "three" if minimum_vertices == 3 else str(minimum_vertices)
            raise MeshLoadError(f"Degenerate {kind}: fewer than {minimum} vertices")
    for start in range(0, len(indices), mesh_io._VALIDATION_BLOCK_SIZE):
        _check_cancel(cancel_check)
        block = indices[start : start + mesh_io._VALIDATION_BLOCK_SIZE]
        if np.any(block < 0) or np.any(block >= point_count):
            raise MeshLoadError(
                f"Invalid {kind} point index; expected [0, {point_count})"
            )
    return offsets, indices


def _validate_face_points(
    vertices: np.ndarray, kind: str, *, allow_zero_area: bool = False
) -> bool:
    # Only this bounded batch is promoted; source coordinates retain their dtype.
    vertices = vertices.astype(np.float64)
    scale = np.max(np.abs(vertices), axis=(1, 2), keepdims=True)
    # Power-of-two scaling avoids rounding collinear source coordinates.
    vertices = np.ldexp(vertices, -np.frexp(scale)[1])
    if kind == "strip":
        edges_a = vertices[:, 1:-1] - vertices[:, :-2]
        edges_b = vertices[:, 2:] - vertices[:, :-2]
        area = np.cross(edges_a, edges_b)
    elif vertices.shape[1] == 3:
        edges_a = vertices[:, 1] - vertices[:, 0]
        edges_b = vertices[:, 2] - vertices[:, 0]
        area = np.cross(edges_a, edges_b)
    else:
        vertices -= vertices[:, :1].copy()
        area = np.cross(vertices, np.roll(vertices, -1, axis=1)).sum(axis=1)
    nonzero = np.any(area != 0, axis=-1)
    if not allow_zero_area and not nonzero.all():
        raise MeshLoadError(f"Degenerate {kind}: zero-area face or triangle")
    return bool(nonzero.any())


_DEFAULT_VALIDATE_FACE_POINTS = _validate_face_points


def _validate_large_face(
    points: np.ndarray,
    indices: np.ndarray,
    kind: str,
    cancel_check: Callable[[], bool] | None,
) -> None:
    """A single large polygon/strip must not bypass the temporary-memory bound."""
    if kind == "strip":
        for start in range(0, len(indices) - 2, mesh_io._VALIDATION_BLOCK_SIZE - 2):
            _check_cancel(cancel_check)
            block = indices[start : start + mesh_io._VALIDATION_BLOCK_SIZE]
            mesh_io._validate_face_points(points[block][None, ...], kind)
        return
    origin = points[indices[0]].astype(np.float64)
    scale = float(np.max(np.abs(origin)))
    for start in range(0, len(indices), mesh_io._VALIDATION_BLOCK_SIZE):
        _check_cancel(cancel_check)
        block = points[indices[start : start + mesh_io._VALIDATION_BLOCK_SIZE]]
        scale = max(scale, float(np.max(np.abs(block))))
    if scale == 0:
        raise MeshLoadError("Degenerate polygon: zero-area face")
    exponent = -np.frexp(scale)[1]
    scaled_origin = np.ldexp(origin, exponent)
    area = np.zeros(3, dtype=np.float64)
    for start in range(0, len(indices), mesh_io._VALIDATION_BLOCK_SIZE):
        _check_cancel(cancel_check)
        stop = min(start + mesh_io._VALIDATION_BLOCK_SIZE, len(indices))
        current_ids = indices[start:stop]
        following_ids = np.empty_like(current_ids)
        following_ids[:-1] = indices[start + 1 : stop]
        following_ids[-1] = indices[stop] if stop < len(indices) else indices[0]
        if np.any(current_ids == following_ids):
            raise MeshLoadError("Degenerate polygon: repeated adjacent vertex")
        current = (
            np.ldexp(points[current_ids].astype(np.float64), exponent) - scaled_origin
        )
        following = (
            np.ldexp(points[following_ids].astype(np.float64), exponent) - scaled_origin
        )
        area += np.cross(current, following).sum(axis=0)
    if np.all(area == 0):
        raise MeshLoadError("Degenerate polygon: zero-area face")


def _validate_faces(
    points: np.ndarray,
    offsets: np.ndarray,
    indices: np.ndarray,
    kind: str,
    cancel_check: Callable[[], bool] | None,
    *,
    allow_zero_area: bool = False,
) -> bool:
    has_area = False
    use_aot_tri = (
        kind == "polygon"
        and mesh_io._validate_face_points is _DEFAULT_VALIDATE_FACE_POINTS
        and points.dtype in (np.float32, np.float64)
        and points.flags.c_contiguous
        and indices.dtype == np.int64
        and indices.flags.c_contiguous
    )
    aot_tri_fn = None
    if use_aot_tri:
        from .._numba_aot._parallel_wrappers import validate_mesh_triangles_range

        aot_tri_fn = validate_mesh_triangles_range
    face_step = (
        max(mesh_io._VALIDATION_BLOCK_SIZE, 131_072)
        if use_aot_tri and mesh_io._VALIDATION_BLOCK_SIZE >= 16_384
        else mesh_io._VALIDATION_BLOCK_SIZE
    )
    for start in range(0, len(offsets) - 1, face_step):
        _check_cancel(cancel_check)
        stop = min(start + face_step, len(offsets) - 1)
        lengths = offsets[start + 1 : stop + 1] - offsets[start:stop]
        if aot_tri_fn is not None and bool(np.all(lengths == 3)):
            tri_indices = indices[int(offsets[start]) : int(offsets[stop])]
            err, chunk_has_area = aot_tri_fn(
                points, tri_indices, allow_zero_area, 0, stop - start
            )
            if err == 1:
                raise MeshLoadError("Degenerate polygon: repeated adjacent vertex")
            if err == 2:
                raise MeshLoadError("Degenerate polygon: zero-area face or triangle")
            has_area |= chunk_has_area
            continue
        for width in np.unique(lengths):
            cell_ids = np.flatnonzero(lengths == width) + start
            if width > mesh_io._VALIDATION_BLOCK_SIZE:
                for cell_id in cell_ids:
                    _validate_large_face(
                        points,
                        indices[offsets[cell_id] : offsets[cell_id + 1]],
                        kind,
                        cancel_check,
                    )
                    has_area = True
                continue
            batch_size = max(1, mesh_io._VALIDATION_BLOCK_SIZE // int(width))
            for batch_start in range(0, len(cell_ids), batch_size):
                _check_cancel(cancel_check)
                batch = cell_ids[batch_start : batch_start + batch_size]
                ids = indices[offsets[batch, None] + np.arange(width)]
                if kind == "polygon" and np.any(ids == np.roll(ids, -1, axis=1)):
                    raise MeshLoadError("Degenerate polygon: repeated adjacent vertex")
                if allow_zero_area and kind == "polygon" and width == 3:
                    has_area |= mesh_io._validate_face_points(
                        points[ids], kind, allow_zero_area=True
                    )
                else:
                    mesh_io._validate_face_points(points[ids], kind)
                    has_area = True
    return has_area


def _validate_polydata(
    polydata: "vtkPolyData", cancel_check: Callable[[], bool] | None,
    *, allow_zero_area: bool = False, allow_lines: bool = False,
    annotation: bool = False,
) -> tuple[float, float, float, float, float, float]:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    if not allow_lines and (
        polydata.GetNumberOfLines() or polydata.GetNumberOfVerts()
    ):
        raise MeshLoadError(
            "Mesh input contains line or vertex cells; export only the surface"
        )
    point_count = polydata.GetNumberOfPoints()
    if not point_count or polydata.GetPoints() is None:
        raise MeshLoadError("VTK surface read failed: surface contains no points")
    has_lines = allow_lines and bool(polydata.GetNumberOfLines())
    if not (polydata.GetNumberOfPolys() or polydata.GetNumberOfStrips() or has_lines):
        raise MeshLoadError("Surface contains no polygon faces or triangle strips")
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    if points.shape != (point_count, 3) or points.dtype.kind not in "fiu":
        raise MeshLoadError("Surface points must be a numeric (N, 3) array")
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    if points.dtype in (np.float32, np.float64) and points.flags.c_contiguous:
        from .._numba_aot._parallel_wrappers import validate_mesh_points_range

        point_step = (
            max(mesh_io._VALIDATION_BLOCK_SIZE, 262_144)
            if mesh_io._VALIDATION_BLOCK_SIZE >= 16_384
            else mesh_io._VALIDATION_BLOCK_SIZE
        )
        for start in range(0, point_count, point_step):
            _check_cancel(cancel_check)
            stop = min(start + point_step, point_count)
            ok, block_bounds = validate_mesh_points_range(points, start, stop)
            if not ok:
                raise MeshLoadError("Surface points contain non-finite coordinates")
            minimum = np.minimum(minimum, block_bounds[:3])
            maximum = np.maximum(maximum, block_bounds[3:])
    else:
        for start in range(0, point_count, mesh_io._VALIDATION_BLOCK_SIZE):
            _check_cancel(cancel_check)
            block = points[start : start + mesh_io._VALIDATION_BLOCK_SIZE]
            if not np.isfinite(block).all():
                raise MeshLoadError("Surface points contain non-finite coordinates")
            minimum = np.minimum(minimum, block.min(axis=0))
            maximum = np.maximum(maximum, block.max(axis=0))
    if allow_lines:
        _validate_cells(
            polydata.GetLines(), point_count, "line", cancel_check, minimum_vertices=2,
        )
        _validate_cells(
            polydata.GetVerts(), point_count, "vertex", cancel_check,
            minimum_vertices=1,
        )
    has_area = False
    for cells, kind in (
        (polydata.GetPolys(), "polygon"),
        (polydata.GetStrips(), "strip"),
    ):
        offsets, indices = _validate_cells(cells, point_count, kind, cancel_check)
        has_area |= mesh_io._validate_faces(
            points, offsets, indices, kind, cancel_check, allow_zero_area=allow_zero_area
        )
    if not has_area and not has_lines and not annotation:
        raise MeshLoadError("Degenerate surface: zero-area faces only")
    return tuple(float(value) for pair in zip(minimum, maximum) for value in pair)
