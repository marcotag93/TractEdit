# -*- coding: utf-8 -*-
"""Private CPU geometry for optional mesh overlays; no renderer or source edits."""

from dataclasses import dataclass
from math import ceil
from collections.abc import Callable

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..data_contracts import MeshLayer
from ..mesh_io import MeshLoadCancelled


MAX_MESH_NORMAL_GLYPHS = 10_000
BOUNDARY_PROCESS_THRESHOLD = 250_000
_NORMAL_CHUNK = 65_536
MAX_INTERACTION_ELEMENTS = 20_000


def mesh_interaction_preview(source, component, limit=MAX_INTERACTION_ELEMENTS):
    """Bound only the temporary display; retain exact source coordinates."""
    from vtk.util.numpy_support import numpy_to_vtkIdTypeArray

    if source.GetNumberOfPoints() == 0:
        return None
    if limit == 0:
        return vtk.vtkPolyData()
    points = vtk_to_numpy(source.GetPoints().GetData())
    if component == "vertices":
        if len(points) <= limit:
            return None
        ids = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        coordinates = points[ids]
        offsets = np.array([0, len(ids)], dtype=np.int64)
    elif component == "boundary_edges":
        lines = source.GetLines()
        cell_offsets = vtk_to_numpy(lines.GetOffsetsArray())
        segment_offsets = np.concatenate(([0], np.cumsum(np.maximum(np.diff(cell_offsets) - 1, 0))))
        total = int(segment_offsets[-1])
        if total <= limit:
            return None
        chosen = np.linspace(0, total - 1, limit, dtype=np.int64)
        owners = np.searchsorted(segment_offsets, chosen, side="right") - 1
        selected = cell_offsets[owners] + chosen - segment_offsets[owners]
        connectivity = vtk_to_numpy(lines.GetConnectivityArray())
        ids = np.column_stack((connectivity[selected], connectivity[selected + 1])).ravel()
        coordinates = points[ids]
        offsets = np.arange(0, len(ids) + 1, 2, dtype=np.int64)
    else:
        groups = (source.GetPolys(), source.GetStrips())
        counts = [cells.GetNumberOfConnectivityIds() for cells in groups]
        total = sum(counts)
        if total <= limit:
            return None
        chosen = np.linspace(0, total - 1, limit, dtype=np.int64)
        segments = []
        base = 0
        for index, (cells, count) in enumerate(zip(groups, counts)):
            selected = chosen[(chosen >= base) & (chosen < base + count)] - base
            base += count
            if not len(selected):
                continue
            cell_offsets = vtk_to_numpy(cells.GetOffsetsArray())
            connectivity = vtk_to_numpy(cells.GetConnectivityArray())
            owners = np.searchsorted(cell_offsets, selected, side="right") - 1
            following = selected + 1
            if index == 0:
                following = np.where(following == cell_offsets[owners + 1],
                                     cell_offsets[owners], following)
            else:
                valid = following < cell_offsets[owners + 1]
                selected, following = selected[valid], following[valid]
            segments.append(np.column_stack((connectivity[selected], connectivity[following])))
        ids = np.concatenate(segments).ravel()
        coordinates = points[ids]
        offsets = np.arange(0, len(ids) + 1, 2, dtype=np.int64)
    preview = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(coordinates, deep=True))
    preview.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(ids), dtype=np.int64), deep=True))
    (preview.SetVerts if component == "vertices" else preview.SetLines)(cells)
    return preview


@dataclass(frozen=True)
class MeshOverlayData:
    polydata: vtk.vtkPolyData
    face_count: int = 0


def prepare_display_overlay(
    source: vtk.vtkPolyData, component: str, cancel_check: Callable[[], bool],
) -> MeshOverlayData:
    """Keep large native boundary filters cancellable outside the GUI process."""
    if (
        component == "boundary_edges"
        and sum(cells.GetNumberOfConnectivityIds() - 2 * cells.GetNumberOfCells()
                for cells in (source.GetPolys(), source.GetStrips()))
        >= BOUNDARY_PROCESS_THRESHOLD
    ):
        from .process import prepare_overlay_in_process

        return prepare_overlay_in_process(source, component, cancel_check)
    return prepare_mesh_overlay(source, component, cancel_check=cancel_check)


def mesh_geometry_key(layer: MeshLayer) -> tuple:
    source = layer["polydata"]
    arrays = (source.GetPoints(), source.GetPolys(), source.GetStrips())
    return (source, *(item.GetMTime() for item in arrays))


def mesh_ras_bounds(layer: MeshLayer) -> tuple[float, ...]:
    bounds = layer["source_bounds"]
    corners = np.array([[x, y, z] for x in bounds[:2]
                        for y in bounds[2:4] for z in bounds[4:6]])
    matrix = layer["source_to_ras"]
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return tuple(np.column_stack((world.min(axis=0), world.max(axis=0))).ravel())


def overlay_source_snapshot(source: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Borrow immutable buffers in a private shell; the worker deep-copies them."""
    result = vtk.vtkPolyData()
    result.SetPoints(source.GetPoints())
    result.SetPolys(source.GetPolys())
    result.SetStrips(source.GetStrips())
    return result


def normal_glyph_count(data: MeshOverlayData, density: float) -> int:
    return min(ceil(data.face_count * density / 100), MAX_MESH_NORMAL_GLYPHS)


def sampled_normal_polydata(data: MeshOverlayData, density: float) -> vtk.vtkPolyData:
    source = data.polydata
    count = normal_glyph_count(data, density)
    if count == source.GetNumberOfPoints():
        return source
    result = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    if count:
        ids = np.linspace(0, source.GetNumberOfPoints() - 1, count, dtype=np.int64)
        points.SetData(numpy_to_vtk(
            vtk_to_numpy(source.GetPoints().GetData())[ids], deep=True,
        ))
        vectors = numpy_to_vtk(
            vtk_to_numpy(source.GetPointData().GetVectors())[ids], deep=True,
        )
        vectors.SetName("face_normals")
        result.GetPointData().SetVectors(vectors)
    result.SetPoints(points)
    return result


def prepare_mesh_overlay(
    source: vtk.vtkPolyData, component: str,
    cancel_check: Callable[[], bool] = lambda: False,
) -> MeshOverlayData:
    """Compute on owned data. Polygon/strip normals refer to derived triangles."""
    def check():
        if cancel_check():
            raise MeshLoadCancelled("Mesh overlay cancelled")

    check()
    private = vtk.vtkPolyData()
    private.DeepCopy(overlay_source_snapshot(source))
    check()
    if component == "boundary_edges":
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(private)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.ManifoldEdgesOff()
        edges.PassLinesOff()
        edges.ColoringOff()
        edges.AddObserver("ProgressEvent", lambda caller, event: (
            caller.SetAbortExecute(True) if cancel_check() else None
        ))
        edges.Update()
        check()
        output = vtk.vtkPolyData()
        output.ShallowCopy(edges.GetOutput())
        return MeshOverlayData(output)
    if component != "surface_normals":
        raise ValueError("Expected boundary edges or surface normals")
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(private)
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.AddObserver("ProgressEvent", lambda caller, event: (
        caller.SetAbortExecute(True) if cancel_check() else None
    ))
    triangles.Update()
    check()
    geometry = triangles.GetOutput()
    points = vtk_to_numpy(geometry.GetPoints().GetData())
    cells = vtk_to_numpy(geometry.GetPolys().GetConnectivityArray()).reshape(-1, 3)

    def triangle_vectors(ids):
        vertices = points[ids].astype(np.float64)
        vectors = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        return vertices, vectors, np.linalg.norm(vectors, axis=1)

    counts = []
    for start in range(0, len(cells), _NORMAL_CHUNK):
        check()
        sizes = triangle_vectors(cells[start:start + _NORMAL_CHUNK])[2]
        counts.append(np.count_nonzero(np.isfinite(sizes) & (sizes > 0)))
    total = int(sum(counts))
    count = min(total, MAX_MESH_NORMAL_GLYPHS)
    chosen = np.linspace(0, total - 1, count, dtype=np.int64)
    centres = np.empty((count, 3), dtype=np.float64)
    normals = np.empty((count, 3), dtype=np.float64)
    offset = 0
    for index, valid_count in enumerate(counts):
        check()
        first, last = np.searchsorted(chosen, [offset, offset + valid_count])
        selected = chosen[first:last] - offset
        offset += valid_count
        if first == last:
            continue
        chunk = cells[index * _NORMAL_CHUNK:(index + 1) * _NORMAL_CHUNK]
        if valid_count == len(chunk):
            vertices, vectors, sizes = triangle_vectors(chunk[selected])
        else:
            vertices, vectors, sizes = triangle_vectors(chunk)
            selected = np.flatnonzero(np.isfinite(sizes) & (sizes > 0))[selected]
            vertices, vectors, sizes = vertices[selected], vectors[selected], sizes[selected]
        centres[first:last] = vertices.mean(axis=1)
        normals[first:last] = vectors / sizes[:, None]
    check()
    output = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(centres, deep=True))
    output.SetPoints(vtk_points)
    vectors = numpy_to_vtk(normals, deep=True)
    vectors.SetName("face_normals")
    output.GetPointData().SetVectors(vectors)
    return MeshOverlayData(output, total)
