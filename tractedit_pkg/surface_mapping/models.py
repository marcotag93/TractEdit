# -*- coding: utf-8 -*-
"""Scientific surface-mapping contracts and private scanner-RAS geometry.

No GUI integration or association engine is installed by importing this module.
Published source buffers are immutable; intentional NumPy edits must call the
owning VTK array's Modified() before a geometry key can detect them.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from enum import IntEnum
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID, uuid4

import numpy as np

from ..data_contracts import MeshLayer

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray,
        vtkPolyData,
        vtkStaticCellLocator,
    )


ALGORITHM_VERSION = "surface-mapping-v2"
QUERY_CHUNK_ITEMS = 4096
_GEOMETRY_CHUNK = 65_536
SurfaceMappingMode = Literal["endpoint_proximity", "surface_intersections", "both"]
SurfaceMappingScope = Literal["visible", "selected_visible"]


class EndpointStatus(IntEnum):
    ASSIGNED = 0
    BEYOND_DISTANCE = 1
    AMBIGUOUS = 2
    INVALID = 3


class IntersectionClass(IntEnum):
    CROSSING = 0
    TOUCH = 1
    COPLANAR_OVERLAP = 2
    AMBIGUOUS = 3
    INVALID = 4


# Missing identifiers use -1 and missing floating values use NaN, never zero.
ENDPOINT_DTYPE = np.dtype(
    [
        ("streamline_id", "i8"),
        ("endpoint", "u1"),
        ("status", "u1"),
        ("triangle_id", "i8"),
        ("source_cell_id", "i8"),
        ("position_ras", "f8", (3,)),
        ("barycentric", "f8", (3,)),
        ("distance_mm", "f8"),
        ("angle_deg", "f8"),
        ("normal_nonunique", "?"),
    ]
)
INTERSECTION_DTYPE = np.dtype(
    [
        ("streamline_id", "i8"),
        ("segment_id", "i8"),
        ("segment_t", "f8"),
        ("arc_length_mm", "f8"),
        ("position_ras", "f8", (3,)),
        ("barycentric", "f8", (3,)),
        ("class", "u1"),
        ("triangle_id", "i8"),
        ("source_cell_id", "i8"),
    ]
)
INTERVAL_DTYPE = np.dtype(
    [
        ("streamline_id", "i8"),
        ("interval_id", "i8"),
        ("start_segment_id", "i8"),
        ("end_segment_id", "i8"),
        ("start_segment_t", "f8"),
        ("end_segment_t", "f8"),
        ("start_arc_length_mm", "f8"),
        ("end_arc_length_mm", "f8"),
        ("start_position_ras", "f8", (3,)),
        ("end_position_ras", "f8", (3,)),
        ("start_barycentric", "f8", (3,)),
        ("end_barycentric", "f8", (3,)),
        ("start_triangle_id", "i8"),
        ("end_triangle_id", "i8"),
        ("start_source_cell_id", "i8"),
        ("end_source_cell_id", "i8"),
        ("class", "u1"),
    ]
)
INTERVAL_CELL_DTYPE = np.dtype(
    [
        ("interval_id", "i8"),
        ("triangle_id", "i8"),
        ("source_cell_id", "i8"),
    ]
)


class SurfaceGeometryError(ValueError):
    """The authoritative surface cannot support a geometric query."""


class SurfaceMappingCancelled(RuntimeError):
    """Preparation was cancelled; no partial geometry may be published."""


@dataclass(frozen=True)
class SurfaceMappingParameters:
    mode: SurfaceMappingMode = "endpoint_proximity"
    scope: SurfaceMappingScope = "visible"
    max_distance_mm: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in ("endpoint_proximity", "surface_intersections", "both"):
            raise ValueError("Invalid surface mapping mode")
        if self.scope not in ("visible", "selected_visible"):
            raise ValueError("Invalid surface mapping scope")
        if (
            isinstance(self.max_distance_mm, (bool, np.bool_))
            or not isinstance(self.max_distance_mm, Real)
            or not np.isfinite(self.max_distance_mm)
            or self.max_distance_mm < 0
        ):
            raise ValueError("Maximum endpoint distance must be finite and nonnegative")

    @property
    def modes(self) -> tuple[str, ...]:
        if self.mode == "both":
            return ("endpoint_proximity", "surface_intersections")
        return (self.mode,)


@dataclass(frozen=True)
class SurfaceGeometryKey:
    mesh_id: str
    source_identity: str
    geometry_revision: tuple[tuple[str, int], ...]
    source_to_ras: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class SurfaceMappingRequest:
    mesh_id: str
    geometry_key: SurfaceGeometryKey
    tractogram_id: str
    tractogram_version: int
    visibility_version: int
    streamline_ids: np.ndarray
    parameters: SurfaceMappingParameters = field(
        default_factory=SurfaceMappingParameters
    )
    selection_version: int | None = None
    request_id: str = field(default_factory=lambda: str(uuid4()))
    algorithm_version: str = field(default=ALGORITHM_VERSION, init=False)

    def metadata(self):
        values = {
            item.name: getattr(self, item.name) for item in fields(self)
            if item.name != "streamline_ids"
        }
        values["geometry_key"] = asdict(self.geometry_key)
        values["parameters"] = asdict(self.parameters)
        return values

    def __post_init__(self) -> None:
        if self.mesh_id != self.geometry_key.mesh_id or not self.tractogram_id:
            raise ValueError("Request identities do not match their owners")
        try:
            if str(UUID(self.request_id)) != self.request_id:
                raise ValueError("Noncanonical request UUID")
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("Request ID must be a canonical UUID") from error
        versions = [self.tractogram_version, self.visibility_version]
        if self.selection_version is not None:
            versions.append(self.selection_version)
        for version in versions:
            if type(version) is not int or version < 0:
                raise ValueError(
                    "Data and scope revisions must be nonnegative integers"
                )
        if (
            self.parameters.scope == "selected_visible"
            and self.selection_version is None
        ):
            raise ValueError("Selected scope requires a selection revision")
        ids = np.asarray(self.streamline_ids)
        if (
            ids.ndim != 1
            or not len(ids)
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("Scope requires nonempty original nonnegative int64 IDs")
        owned = ids.astype(np.int64)
        if np.any(owned[1:] <= owned[:-1]):
            owned.sort()
            keep = np.empty(len(owned), dtype=bool)
            keep[0] = True
            np.not_equal(owned[1:], owned[:-1], out=keep[1:])
            owned = owned[keep]
        owned.flags.writeable = False
        object.__setattr__(self, "streamline_ids", owned)


@dataclass(frozen=True)
class SurfaceMappingDiagnostics:
    """Endpoint statuses partition evaluated_endpoints (two per requested ID).

    Point-event counts exclude intervals except ambiguous_events, which counts
    ambiguous points and intervals. coplanar_intervals includes ambiguous
    intervals; crossing_streamlines counts unique IDs with proven crossings.
    """

    requested_streamlines: int = 0
    valid_streamlines: int = 0
    evaluated_endpoints: int = 0
    assigned_endpoints: int = 0
    beyond_distance_endpoints: int = 0
    ambiguous_endpoints: int = 0
    invalid_endpoints: int = 0
    crossing_events: int = 0
    touch_events: int = 0
    coplanar_intervals: int = 0
    ambiguous_events: int = 0
    invalid_events: int = 0
    crossing_streamlines: int = 0


@dataclass(frozen=True, eq=False)
class SurfaceMappingResult:
    """Columnar tables may be memmaps; ownership is given by the frozen request."""

    request: SurfaceMappingRequest
    diagnostics: SurfaceMappingDiagnostics
    geometry_diagnostics: "SurfaceGeometryDiagnostics"
    endpoints: np.ndarray | None = None
    intersections: np.ndarray | None = None
    intervals: np.ndarray | None = None
    interval_cells: np.ndarray | None = None
    endpoint_contributions: np.ndarray | None = None
    endpoint_density: np.ndarray | None = None
    endpoint_density_valid: np.ndarray | None = None
    _storage: "_ResultStorage | None" = field(default=None, repr=False)

    def close(self) -> None:
        """Release mapped result files after all consumers have finished."""
        if self._storage is not None:
            self._storage.close()

    def __enter__(self) -> "SurfaceMappingResult":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _ResultStorage(Protocol):
    def close(self) -> None:
        """Release mapped buffers and the temporary files they own."""


@dataclass(frozen=True)
class SurfaceGeometryDiagnostics:
    source_points: int
    source_polygons: int
    source_strips: int
    ignored_vertex_cells: int
    ignored_line_cells: int
    derived_triangles: int
    valid_triangles: int
    degenerate_triangles: int
    untriangulated_cells: int
    nonplanar_polygons: int
    boundary_edges: int
    nonmanifold_edges: int
    duplicate_faces: int


@dataclass(eq=False)
class SurfaceGeometry:
    """Detached valid triangles; locator cell IDs index rows of these arrays.

    triangle_ids retains the pre-exclusion triangulation IDs. NumPy arrays are
    read-only and VTK owns separate buffers. No source attributes are carried.
    """

    key: SurfaceGeometryKey
    points_ras: np.ndarray
    triangles: np.ndarray
    triangle_ids: np.ndarray
    source_cell_ids: np.ndarray
    normals: np.ndarray
    areas_mm2: np.ndarray
    vertex_areas_mm2: np.ndarray
    barycentric_tolerances: np.ndarray
    epsilon_mm: float
    diagnostics: SurfaceGeometryDiagnostics
    polydata: "vtkPolyData" = field(repr=False)
    _locator: "vtkStaticCellLocator | None" = field(
        default=None, init=False, repr=False
    )

    @property
    def locator_built(self) -> bool:
        return self._locator is not None

    def get_locator(self) -> "vtkStaticCellLocator":
        """Build on explicit demand only; call from a single analysis worker.

        Native BuildLocator cannot be interrupted here. The future isolated
        process controller must bound cancellation latency for this phase.
        """
        if self._locator is None:
            from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

            locator = vtkStaticCellLocator()
            locator.SetDataSet(self.polydata)
            locator.BuildLocator()
            self._locator = locator
        return self._locator


def _validate_affine(matrix: np.ndarray) -> None:
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.array_equal(matrix[3], [0, 0, 0, 1])
        or np.linalg.slogdet(matrix[:3, :3])[0] == 0
    ):
        raise SurfaceGeometryError("Expected a finite nonsingular source-to-RAS affine")


@lru_cache(maxsize=16)
def _validated_affine_values(raw: bytes) -> tuple:
    matrix = np.frombuffer(raw, dtype=np.float64).reshape(4, 4)
    _validate_affine(matrix)
    return tuple(matrix.ravel())


def surface_geometry_key(layer: MeshLayer) -> SurfaceGeometryKey:
    """O(1) revision check including topology, coordinate buffers and affine."""
    source = layer["polydata"]
    points = source.GetPoints()
    arrays = [points, points.GetData() if points is not None else None]
    for cells in (
        source.GetVerts(),
        source.GetLines(),
        source.GetPolys(),
        source.GetStrips(),
    ):
        arrays.extend((cells, cells.GetOffsetsArray(), cells.GetConnectivityArray()))
    matrix = np.asarray(layer["source_to_ras"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise SurfaceGeometryError("Expected a 4x4 source-to-RAS affine")
    return SurfaceGeometryKey(
        layer["id"],
        source.GetAddressAsString(""),
        tuple(
            (
                (item.GetAddressAsString(""), item.GetMTime())
                if item is not None
                else ("none", 0)
            )
            for item in arrays
        ),
        _validated_affine_values(matrix.tobytes()),
    )


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise SurfaceMappingCancelled("Surface mapping cancelled")


def _validate_cells(cells: "vtkCellArray", point_count: int, minimum: int) -> None:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    indices = vtk_to_numpy(cells.GetConnectivityArray())
    if (
        len(offsets) != cells.GetNumberOfCells() + 1
        or offsets[0] != 0
        or offsets[-1] != len(indices)
        or np.any(np.diff(offsets) < minimum)
        or np.any(indices < 0)
        or np.any(indices >= point_count)
    ):
        raise SurfaceGeometryError("Invalid source surface connectivity")


def _nonplanar_polygons(
    points: np.ndarray,
    cells: "vtkCellArray",
    epsilon: float,
    cancel_check: Callable[[], bool] | None,
) -> int:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    indices = vtk_to_numpy(cells.GetConnectivityArray())
    total = 0
    for index in np.flatnonzero(np.diff(offsets) > 3):
        _check_cancel(cancel_check)
        vectors = (
            points[indices[offsets[index] : offsets[index + 1]]]
            - points[indices[offsets[index]]]
        )
        longest = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
        crosses = np.cross(longest, vectors)
        sizes = np.linalg.norm(crosses, axis=1)
        chosen = int(np.argmax(sizes))
        if sizes[chosen] > 0:
            distances = np.abs(vectors @ (crosses[chosen] / sizes[chosen]))
            total += int(np.any(distances > epsilon))
    return total


def _topology_counts(triangles: np.ndarray) -> tuple[int, int, int]:
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    unique_faces = np.unique(np.sort(triangles, axis=1), axis=0)
    return (
        int(np.count_nonzero(counts == 1)),
        int(np.count_nonzero(counts > 2)),
        len(triangles) - len(unique_faces),
    )


def mapping_source(layer, cancel_check):
    """Recover verified OBJ position IDs before triangulation or worker staging."""
    mesh = layer["polydata"]
    path = Path(layer.get("path", ""))
    if path.suffix.lower() != ".obj":
        return mesh

    from hashlib import sha256
    from vtkmodules.util.numpy_support import (
        numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy,
    )
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from ..mesh.obj_geometry import read_obj_faces_stream, verify_obj_faces

    def check():
        _check_cancel(cancel_check)

    digest = sha256()
    with path.open("rb") as stream:
        vertices, offsets, indices, _, _ = read_obj_faces_stream(
            stream, check, with_uv=False, update_digest=digest.update,
        )
    if digest.hexdigest() != layer.get("source_sha256"):
        raise SurfaceGeometryError("Cannot verify OBJ source identity; reload the mesh")
    verify_obj_faces(mesh, vertices, offsets, indices, check)
    if mesh.GetNumberOfCells() != mesh.GetNumberOfPolys():
        raise SurfaceGeometryError("Cannot verify non-polygon OBJ cell correspondence")
    vertices = vertices.astype(vtk_to_numpy(mesh.GetPoints().GetData()).dtype, copy=False)
    points, polygons = vtkPoints(), vtkCellArray()
    points.SetData(numpy_to_vtk(vertices))
    polygons.SetData(numpy_to_vtkIdTypeArray(offsets), numpy_to_vtkIdTypeArray(indices))
    source = vtkPolyData()
    source._surface_mapping_buffers = (vertices, offsets, indices)
    source.SetPoints(points)
    source.SetPolys(polygons)
    return source


def prepare_surface_geometry(
    layer: MeshLayer,
    *,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> SurfaceGeometry:
    """Validate and triangulate privately in RAS without welding or repair.

    Raises:
        SurfaceGeometryError: Invalid coordinates/topology or no valid faces.
        SurfaceMappingCancelled: Cancellation observed before publication.
    """
    from vtkmodules.util.numpy_support import (
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter

    _check_cancel(cancel_check)
    key = surface_geometry_key(layer)
    matrix = np.asarray(key.source_to_ras, dtype=np.float64).reshape(4, 4)
    source = mapping_source(layer, cancel_check)
    count = source.GetNumberOfPoints()
    if not count or not source.GetNumberOfPolys() + source.GetNumberOfStrips():
        raise SurfaceGeometryError("Surface requires points and polygon or strip cells")
    original = vtk_to_numpy(source.GetPoints().GetData())
    if original.shape != (count, 3):
        raise SurfaceGeometryError("Expected three coordinates per surface point")
    points = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, _GEOMETRY_CHUNK):
        _check_cancel(cancel_check)
        block = original[start : start + _GEOMETRY_CHUNK].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            points[start : start + len(block)] = (
                block @ matrix[:3, :3].T + matrix[:3, 3]
            )
        if not np.all(np.isfinite(points[start : start + len(block)])):
            raise SurfaceGeometryError(
                "Nonfinite source or transformed surface coordinates"
            )
    scale = max(
        1.0,
        float(np.linalg.norm(np.ptp(points, axis=0))),
        float(np.max(np.abs(points))),
    )
    if not np.isfinite(scale):
        raise SurfaceGeometryError("Surface coordinate range exceeds float64 geometry")
    epsilon = max(1e-6, 64 * np.finfo(np.float64).eps * scale)
    for cells, minimum in (
        (source.GetVerts(), 1),
        (source.GetLines(), 2),
        (source.GetPolys(), 3),
        (source.GetStrips(), 3),
    ):
        _check_cancel(cancel_check)
        _validate_cells(cells, count, minimum)
    nonplanar = _nonplanar_polygons(points, source.GetPolys(), epsilon, cancel_check)
    private = vtkPolyData()
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(points, deep=True))
    private.SetPoints(vertices)
    for setter, cells in (
        (private.SetPolys, source.GetPolys()),
        (private.SetStrips, source.GetStrips()),
    ):
        owned = vtkCellArray()
        owned.DeepCopy(cells)
        setter(owned)
    first = source.GetNumberOfVerts() + source.GetNumberOfLines()
    source_ids = np.arange(first, source.GetNumberOfCells(), dtype=np.int64)
    array = numpy_to_vtkIdTypeArray(source_ids, deep=True)
    array.SetName("source_cell_id")
    private.GetCellData().AddArray(array)
    triangulate = vtkTriangleFilter()
    triangulate.SetInputData(private)
    triangulate.PassVertsOff()
    triangulate.PassLinesOff()
    if cancel_check is not None:
        triangulate.AddObserver(
            "ProgressEvent",
            lambda caller, event: (
                caller.SetAbortExecute(True) if cancel_check() else None
            ),
        )
    if progress:
        progress("triangulation")
    triangulate.Update()
    _check_cancel(cancel_check)
    if progress:
        progress("geometry_validation")
    output = triangulate.GetOutput()
    cells = vtk_to_numpy(output.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    if not len(cells):
        raise SurfaceGeometryError("Triangulation produced no surface triangles")
    owners = vtk_to_numpy(output.GetCellData().GetArray("source_cell_id"))
    valid = np.empty(len(cells), dtype=bool)
    normals = np.empty((len(cells), 3), dtype=np.float64)
    areas = np.empty(len(cells), dtype=np.float64)
    tolerances = np.empty((len(cells), 3), dtype=np.float64)
    for start in range(0, len(cells), _GEOMETRY_CHUNK):
        _check_cancel(cancel_check)
        stop = start + _GEOMETRY_CHUNK
        coordinates = points[cells[start:stop]]
        edges = coordinates[:, [2, 0, 1]] - coordinates[:, [1, 2, 0]]
        lengths = np.linalg.norm(edges, axis=2)
        crosses = np.cross(edges[:, 1], edges[:, 2])
        sizes = np.linalg.norm(crosses, axis=1)
        if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(sizes)):
            raise SurfaceGeometryError(
                "Triangle numerical range exceeds float64 geometry"
            )
        maximum = lengths.max(axis=1)
        keep = sizes > epsilon * maximum
        valid[start:stop] = keep
        areas[start:stop] = sizes * 0.5
        np.divide(crosses, sizes[:, None], out=normals[start:stop], where=keep[:, None])
        np.divide(
            epsilon * lengths,
            sizes[:, None],
            out=tolerances[start:stop],
            where=keep[:, None],
        )
    if not np.any(valid):
        raise SurfaceGeometryError("Surface has no nondegenerate triangles")
    triangles = np.array(cells[valid], dtype=np.int64, order="C", copy=True)
    triangle_ids = np.flatnonzero(valid)
    cell_ids = np.array(owners[valid], dtype=np.int64, copy=True)
    normals, areas, tolerances = normals[valid], areas[valid], tolerances[valid]
    vertex_areas = np.zeros(count, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_areas, triangles[:, corner], areas / 3)
    _check_cancel(cancel_check)
    boundary, nonmanifold, duplicates = _topology_counts(triangles)
    diagnostics = SurfaceGeometryDiagnostics(
        source_points=count,
        source_polygons=source.GetNumberOfPolys(),
        source_strips=source.GetNumberOfStrips(),
        ignored_vertex_cells=source.GetNumberOfVerts(),
        ignored_line_cells=source.GetNumberOfLines(),
        derived_triangles=len(cells),
        valid_triangles=len(triangles),
        degenerate_triangles=int(np.count_nonzero(~valid)),
        untriangulated_cells=len(source_ids) - len(np.unique(owners)),
        nonplanar_polygons=nonplanar,
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        duplicate_faces=duplicates,
    )
    result = vtkPolyData()
    result.SetPoints(vertices)
    vtk_cells = vtkCellArray()
    vtk_cells.SetData(
        numpy_to_vtkIdTypeArray(
            np.arange(len(triangles) + 1, dtype=np.int64) * 3, deep=True
        ),
        numpy_to_vtkIdTypeArray(triangles.ravel(), deep=True),
    )
    result.SetPolys(vtk_cells)
    _check_cancel(cancel_check)
    if key != surface_geometry_key(layer):
        raise SurfaceGeometryError("Source geometry changed during preparation")
    for values in (
        points,
        triangles,
        triangle_ids,
        cell_ids,
        normals,
        areas,
        vertex_areas,
        tolerances,
    ):
        values.flags.writeable = False
    return SurfaceGeometry(
        key=key,
        points_ras=points,
        triangles=triangles,
        triangle_ids=triangle_ids,
        source_cell_ids=cell_ids,
        normals=normals,
        areas_mm2=areas,
        vertex_areas_mm2=vertex_areas,
        barycentric_tolerances=tolerances,
        epsilon_mm=epsilon,
        diagnostics=diagnostics,
        polydata=result,
    )
