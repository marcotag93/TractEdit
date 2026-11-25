"""GUI-owned presentations over immutable OBJ geometry."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..data_contracts import MeshLayer, MeshTexturePreparation
from ..mesh_texture import (
    DEFAULT_BUDGETS, MeshTextureBudgets, MeshTextureError, transform_texture_uv,
    validate_texture_source,
)
from .actors import create_mesh_actor


@dataclass
class MeshTexturePart:
    actor: vtk.vtkActor
    source_cells: np.ndarray
    textured: bool
    bridges: tuple[vtk.vtkDataArray, ...]


@dataclass
class MeshTexturePresentation:
    preparation: MeshTexturePreparation
    uniform_mapper: vtk.vtkPolyDataMapper
    mapper: vtk.vtkPolyDataMapper
    texture: vtk.vtkTexture
    diffuse_color: tuple[float, float, float]
    uniform_color: tuple[float, float, float]
    has_transparency: bool
    # Retain the Python bridges until their native consumers are detached.
    pixel_array: vtk.vtkDataArray
    uv_array: vtk.vtkDataArray
    enabled: bool = False
    parts: tuple[MeshTexturePart, ...] = ()
    source_cells: np.ndarray | None = None
    bridges: tuple[vtk.vtkDataArray, ...] = ()
    primary_has_transparency: bool | None = None
    uniform_lighting: bool = True

    def release(self, render_window) -> None:
        textures = {id(self.texture): self.texture} if self.texture is not None else {}
        mappers = [self.mapper]
        for part in self.parts:
            texture = part.actor.GetTexture()
            if texture is not None:
                textures[id(texture)] = texture
            mapper = part.actor.GetMapper()
            if mapper is not None:
                mappers.append(mapper)
            part.actor.SetTexture(None)
            part.actor.SetMapper(None)
        for texture in textures.values():
            if isinstance(render_window, vtk.vtkRenderWindow):
                texture.ReleaseGraphicsResources(render_window)
            texture.SetInputData(None)
        for mapper in mappers:
            if isinstance(render_window, vtk.vtkRenderWindow):
                mapper.ReleaseGraphicsResources(render_window)
            mapper.SetInputData(None)


def derived_uv_bytes(layer: MeshLayer, preparation: MeshTexturePreparation) -> int:
    layout = preparation.get("face_layout", {})
    if len(layout) != 1 or not next(iter(layout.values()))["complete_uv"]:
        source = layer["polydata"]
        corners = source.GetPolys().GetConnectivityArray().GetNumberOfValues()
        point_bytes = source.GetPoints().GetData().GetDataTypeSize() * 3
        normals = source.GetPointData().GetNormals()
        normal_bytes = normals.GetDataTypeSize() * 3 if normals is not None else 0
        arrays = _material_uv_arrays(source)
        uv_bytes = max((
            arrays[name].GetDataTypeSize() * 2 for name in layout if name in arrays
        ), default=0)
        return (
            corners * (point_bytes + normal_bytes + uv_bytes + 8)
            + source.GetNumberOfPolys() * 16 + (len(layout) + 2) * 8
        )
    name = next(iter(preparation["face_layout"]))
    material = preparation["materials"].get(name)
    diffuse = material["diffuse_map"] if material is not None else None
    if diffuse is None or (
        diffuse["scale"][:2] == (1, 1) and diffuse["offset"][:2] == (0, 0)
    ):
        return 0
    uv = layer["polydata"].GetPointData().GetArray(name)
    return uv.GetDataSize() * uv.GetDataTypeSize() if uv is not None else 0


def _create_single_texture_presentation(
    layer: MeshLayer,
    preparation: MeshTexturePreparation,
    uniform_mapper: vtk.vtkPolyDataMapper,
    *,
    gl_maximum: int,
    budgets: MeshTextureBudgets = DEFAULT_BUDGETS,
) -> MeshTexturePresentation:
    """Require full lexical UV coverage; mixed face groups belong to partitioning."""
    source = layer["polydata"]
    validate_texture_source(layer, preparation)
    layout = preparation.get("face_layout", {})
    if Path(layer["path"]).suffix.lower() != ".obj" or len(layout) != 1:
        raise MeshTextureError(
            "texture_partition_required", "A single textured face group is required"
        )
    name, summary = next(iter(layout.items()))
    material = preparation["materials"].get(name)
    if not name or not summary["complete_uv"] or material is None:
        raise MeshTextureError(
            "texture_partition_required", "Some faces have no UV map"
        )
    key = material["image_key"]
    if key is None or key not in preparation["images"]:
        raise MeshTextureError(
            "texture_unavailable", "The diffuse image is unavailable"
        )
    names = source.GetFieldData().GetAbstractArray("MaterialNames")
    ids = source.GetCellData().GetArray("MaterialIds")
    uv_array = source.GetPointData().GetArray(name)
    polys = source.GetPolys()
    if (
        source.GetNumberOfCells() != source.GetNumberOfPolys()
        or summary["face_count"] != source.GetNumberOfPolys()
        or summary["corner_count"] != polys.GetConnectivityArray().GetNumberOfValues()
        or names is None
        or not names.IsA("vtkStringArray")
        or ids is None
        or ids.GetNumberOfComponents() != 1
        or ids.GetNumberOfTuples() != source.GetNumberOfPolys()
        or uv_array is None
        or uv_array.GetNumberOfComponents() != 2
        or uv_array.GetNumberOfTuples() != source.GetNumberOfPoints()
    ):
        raise MeshTextureError(
            "texture_geometry", "OBJ face and UV correspondence changed"
        )
    expected_ids = tuple(
        idx for idx in range(names.GetNumberOfValues())
        if names.GetValue(idx) == name
    )
    if not expected_ids:
        raise MeshTextureError(
            "texture_geometry", "OBJ material correspondence changed"
        )
    material_ids = vtk_to_numpy(ids)
    expected_id = expected_ids[0]
    for start in range(0, len(material_ids), 16384):
        block = material_ids[start : start + 16384]
        valid = (
            bool(np.all(block == expected_id))
            if len(expected_ids) == 1
            else bool(np.isin(block, expected_ids).all())
        )
        if not valid:
            raise MeshTextureError(
                "texture_geometry", "OBJ material correspondence changed"
            )
    image = preparation["images"][key]
    _validate_texture_dimensions(image, gl_maximum, budgets)
    if (
        preparation["costs"]["resident_estimate_bytes"]
        > budgets.resident_estimate_bytes
    ):
        raise MeshTextureError(
            "texture_budget", "Texture exceeds the resident estimate"
        )
    diffuse_map = material["diffuse_map"]
    data = vtk.vtkPolyData()
    data.ShallowCopy(source)
    texture_uv = uv_array
    if derived_uv_bytes(layer, preparation):
        uv = transform_texture_uv(
            vtk_to_numpy(uv_array),
            diffuse_map,
            budgets=budgets,
            retained_bytes=preparation["costs"]["decoded_cpu_bytes"],
        )
        texture_uv = numpy_to_vtk(uv, deep=False)
        texture_uv.SetName(name)
    data.GetPointData().SetTCoords(texture_uv)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    mapper.ScalarVisibilityOff()
    texture, pixel_array = _shared_texture(image, diffuse_map["clamp"])
    return MeshTexturePresentation(
        preparation,
        uniform_mapper,
        mapper,
        texture,
        material["diffuse_color"],
        layer["color"],
        image["has_transparency"],
        pixel_array,
        texture_uv,
    )


def _material_uv_arrays(source):
    data = source.GetPointData()
    return {
        array.GetName(): array for index in range(data.GetNumberOfArrays())
        if (array := data.GetArray(index)) is not None
    }


def _face_groups(source, preparation, budgets):
    layout = preparation.get("face_layout", {})
    names = source.GetFieldData().GetAbstractArray("MaterialNames")
    ids = source.GetCellData().GetArray("MaterialIds")
    count = source.GetNumberOfPolys()
    if (
        source.GetNumberOfCells() != count or not count
        or names is None or not names.IsA("vtkStringArray") or ids is None
        or ids.GetNumberOfComponents() != 1 or ids.GetNumberOfTuples() != count
        or count * 32 > budgets.temporary_cpu_bytes
    ):
        raise MeshTextureError("texture_geometry", "Invalid OBJ face correspondence")
    material_ids = vtk_to_numpy(ids)
    offsets = vtk_to_numpy(source.GetPolys().GetOffsetsArray())
    slots = [""] + [names.GetValue(index) for index in range(names.GetNumberOfValues())]
    if len(set(slots)) != len(slots) or len(slots) > budgets.max_materials + 1:
        raise MeshTextureError("texture_geometry", "Invalid OBJ material names")
    coverage = preparation.get("face_uv")
    if coverage is None:
        if not all(item["complete_uv"] for item in layout.values()):
            raise MeshTextureError("texture_geometry", "Missing face UV coverage")
    elif coverage.dtype != np.uint8 or coverage.shape != (count,):
        raise MeshTextureError("texture_geometry", "Invalid face UV coverage")
    face_counts = np.zeros(len(slots), dtype=np.int64)
    corner_counts = np.zeros(len(slots), dtype=np.int64)
    missing = np.zeros(len(slots), dtype=np.int64)
    codes = np.empty(count, dtype=np.int32)
    textured = {}
    arrays = _material_uv_arrays(source)
    for index, name in enumerate(slots):
        material = preparation["materials"].get(name)
        if material is not None and material["image_key"] in preparation["images"]:
            uv = arrays.get(name)
            if (
                uv is None or uv.GetNumberOfComponents() != 2
                or uv.GetNumberOfTuples() != source.GetNumberOfPoints()
                or uv.GetDataType() not in (vtk.VTK_FLOAT, vtk.VTK_DOUBLE)
            ):
                raise MeshTextureError(
                    "texture_geometry", "Invalid material UV array"
                )
            textured[index] = name
    lookup = np.array([
        index if index in textured else 0 for index in range(len(slots))
    ])
    unassigned = layout.get("", {}).get("face_count", 0)
    for start in range(0, count, 16384):
        values = material_ids[start:start + 16384]
        if np.any(values < -1) or np.any(values >= len(slots) - 1):
            raise MeshTextureError(
                "texture_geometry", "OBJ material correspondence changed"
            )
        indices = values.astype(np.int64) + 1
        prefix = min(len(indices), max(0, unassigned - start))
        if prefix:
            # VTK 9.2.6 assigns material zero before the first usemtl.
            if np.any((values[:prefix] != 0) & (values[:prefix] != -1)):
                raise MeshTextureError(
                    "texture_geometry", "Unassigned OBJ material changed"
                )
            indices[:prefix] = 0
        lengths = np.diff(offsets[start:start + len(values) + 1])
        counts_block = np.bincount(indices, minlength=len(slots))
        face_counts += counts_block
        if np.all(lengths == 3):
            corner_counts += counts_block * 3
        else:
            corner_counts += np.bincount(
                indices, weights=lengths, minlength=len(slots)
            ).astype(np.int64)
        codes[start:start + len(values)] = lookup[indices]
        if coverage is not None:
            block = coverage[start:start + len(values)]
            if np.any(block > 1):
                raise MeshTextureError("texture_geometry", "Invalid face UV coverage")
            missing_mask = block == 0
            if np.any(missing_mask):
                missing += np.bincount(indices[missing_mask], minlength=len(slots))
                codes[start:start + len(values)][missing_mask] = 0
    actual = {
        name: {"face_count": int(face_counts[index]),
               "corner_count": int(corner_counts[index]),
               "complete_uv": not bool(missing[index])}
        for index, name in enumerate(slots) if face_counts[index]
    }
    if actual != layout:
        raise MeshTextureError(
            "texture_geometry", "OBJ face and UV correspondence changed"
        )
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    groups = []
    for cells in np.split(order, boundaries):
        code = int(codes[cells[0]])
        groups.append((textured.get(code), cells))
    if not any(name is not None for name, _ in groups):
        raise MeshTextureError(
            "texture_unavailable", "No faces have a usable diffuse map"
        )
    return sorted(groups, key=lambda group: group[0] is None)


def _subset_mapper(source, cells, name, diffuse, source_uv_array=None):
    source_offsets = vtk_to_numpy(source.GetPolys().GetOffsetsArray())
    source_connections = vtk_to_numpy(source.GetPolys().GetConnectivityArray())
    offsets = np.empty(len(cells) + 1, dtype=np.int64)
    offsets[0] = 0
    all_triangles = True
    for start in range(0, len(cells), 16384):
        block = cells[start:start + 16384]
        widths = source_offsets[block + 1] - source_offsets[block]
        if all_triangles and not np.all(widths == 3):
            all_triangles = False
        offsets[start + 1:start + len(block) + 1] = widths
    np.cumsum(offsets, out=offsets)
    count = int(offsets[-1])
    source_points = vtk_to_numpy(source.GetPoints().GetData())
    source_normals = source.GetPointData().GetNormals()
    if source_normals is not None and (
        source_normals.GetNumberOfComponents() != 3
        or source_normals.GetNumberOfTuples() != source.GetNumberOfPoints()
    ):
        raise MeshTextureError("texture_geometry", "Invalid OBJ normal correspondence")
    normal_values = vtk_to_numpy(source_normals) if source_normals is not None else None
    points = np.empty((count, 3), dtype=source_points.dtype)
    normals = (
        np.empty((count, 3), dtype=normal_values.dtype)
        if normal_values is not None else None
    )
    source_uv = vtk_to_numpy(source_uv_array) if source_uv_array is not None else None
    uv = np.empty((count, 2), dtype=source_uv.dtype) if source_uv is not None else None

    def copy_corners(indices, dest):
        points[dest] = source_points[indices]
        if normals is not None:
            normals[dest] = normal_values[indices]
        if uv is not None:
            target = uv[dest]
            with np.errstate(over="ignore", invalid="ignore"):
                np.multiply(source_uv[indices], diffuse["scale"][:2], out=target)
                np.add(target, diffuse["offset"][:2], out=target)
            if not np.isfinite(target).all():
                raise MeshTextureError("invalid_uv", "UV transform overflowed")

    if all_triangles:
        step = max(1, 16384 // 3)
        corner_offsets = np.arange(3, dtype=np.int64)
        for cell_start in range(0, len(cells), step):
            batch = cells[cell_start:cell_start + step]
            c_start = cell_start * 3
            c_end = c_start + len(batch) * 3
            indices = source_connections[
                (source_offsets[batch][:, None] + corner_offsets).ravel()
            ]
            copy_corners(indices, slice(c_start, c_end))
    else:
        for start in range(0, count, 16384):
            positions = np.arange(start, min(start + 16384, count))
            faces = np.searchsorted(offsets, positions, side="right") - 1
            indices = source_connections[
                source_offsets[cells[faces]] + positions - offsets[faces]
            ]
            copy_corners(indices, slice(start, start + len(positions)))
    data = vtk.vtkPolyData()
    point_array = numpy_to_vtk(points, deep=False)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(point_array)
    data.SetPoints(vtk_points)
    offset_array = numpy_to_vtk(offsets, deep=False, array_type=vtk.VTK_ID_TYPE)
    connections = numpy_to_vtk(
        np.arange(count, dtype=np.int64), deep=False, array_type=vtk.VTK_ID_TYPE,
    )
    polys = vtk.vtkCellArray()
    polys.SetData(offset_array, connections)
    data.SetPolys(polys)
    bridges = [point_array, offset_array, connections]
    if normals is not None:
        array = numpy_to_vtk(normals, deep=False)
        array.SetName(source_normals.GetName())
        data.GetPointData().SetNormals(array)
        bridges.append(array)
    uv_array = None
    if uv is not None:
        uv_array = numpy_to_vtk(uv, deep=False)
        uv_array.SetName(name)
        data.GetPointData().SetTCoords(uv_array)
        bridges.append(uv_array)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    mapper.ScalarVisibilityOff()
    return mapper, uv_array, tuple(bridges)


def _validate_texture_dimensions(image, gl_maximum, budgets):
    pixels = image["pixels"]
    if max(pixels.shape[:2]) > min(gl_maximum, budgets.max_dimension):
        raise MeshTextureError(
            "image_dimension", "Image exceeds the OpenGL texture limit"
        )


def _shared_texture(image, clamp):
    pixels = image["pixels"]
    data = vtk.vtkImageData()
    data.SetDimensions(pixels.shape[1], pixels.shape[0], 1)
    array = numpy_to_vtk(pixels.reshape(-1, pixels.shape[2]), deep=False)
    data.GetPointData().SetScalars(array)
    texture = vtk.vtkTexture()
    texture.SetInputData(data)
    texture.SetColorModeToDirectScalars()
    texture.InterpolateOn()
    texture.MipmapOff()
    texture.SetWrap(vtk.vtkTexture.ClampToEdge if clamp else vtk.vtkTexture.Repeat)
    return texture, array


def create_mesh_texture_presentation(
    layer: MeshLayer, preparation: MeshTexturePreparation,
    uniform_mapper: vtk.vtkPolyDataMapper, *, gl_maximum: int,
    budgets: MeshTextureBudgets = DEFAULT_BUDGETS,
) -> MeshTexturePresentation:
    layout = preparation.get("face_layout", {})
    if len(layout) == 1 and next(iter(layout.values()))["complete_uv"]:
        return _create_single_texture_presentation(
            layer, preparation, uniform_mapper, gl_maximum=gl_maximum, budgets=budgets,
        )
    validate_texture_source(layer, preparation)
    if Path(layer["path"]).suffix.lower() != ".obj":
        raise MeshTextureError("texture_geometry", "Only OBJ supports diffuse maps")
    payload = derived_uv_bytes(layer, preparation)
    retained = preparation["costs"]["decoded_cpu_bytes"]
    retained += preparation.get("face_uv", np.empty(0)).nbytes
    if (
        retained + payload > min(budgets.layer_cache_bytes, budgets.global_cache_bytes)
        or payload > budgets.temporary_cpu_bytes
        or preparation["costs"]["resident_estimate_bytes"]
        > budgets.resident_estimate_bytes
    ):
        raise MeshTextureError(
            "texture_budget", "Material presentation exceeds the budget"
        )
    groups = _face_groups(layer["polydata"], preparation, budgets)
    arrays = _material_uv_arrays(layer["polydata"])
    textures, parts, mappers = {}, [], []
    first = None
    try:
        for name, cells in groups:
            material = preparation["materials"][name] if name else None
            diffuse = material["diffuse_map"] if material else None
            mapper, uv_array, bridges = _subset_mapper(
                layer["polydata"], cells, name, diffuse, arrays.get(name),
            )
            mappers.append(mapper)
            texture, pixel_array, transparent = None, None, False
            color = layer["color"]
            if material is not None:
                key = (material["image_key"], diffuse["clamp"])
                image = preparation["images"][key[0]]
                if key not in textures:
                    _validate_texture_dimensions(image, gl_maximum, budgets)
                    textures[key] = _shared_texture(image, key[1])
                texture, pixel_array = textures[key]
                transparent = image["has_transparency"]
                color = material["diffuse_color"]
                bridges += (pixel_array,)
            if first is None:
                first = MeshTexturePresentation(
                    preparation, uniform_mapper, mapper, texture, color,
                    layer["color"], transparent, pixel_array, uv_array,
                    source_cells=cells, bridges=bridges,
                    primary_has_transparency=transparent,
                )
            else:
                actor = create_mesh_actor(
                    mapper.GetInput(), layer["source_to_ras"], color, layer["opacity"], mapper=mapper,
                )
                actor.SetTexture(texture)
                actor.GetProperty().SetLighting(texture is None)
                actor.SetForceTranslucent(transparent)
                actor.VisibilityOff()
                parts.append(MeshTexturePart(actor, cells, name is not None, bridges))
                first.has_transparency |= transparent
        first.parts = tuple(parts)
        return first
    except Exception:
        for part in parts:
            part.actor.SetTexture(None)
            part.actor.SetMapper(None)
        for texture, _ in textures.values():
            texture.SetInputData(None)
        for mapper in mappers:
            mapper.SetInputData(None)
        raise
