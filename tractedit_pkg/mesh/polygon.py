"""Closed territories on the original triangulated mesh."""

from hashlib import sha256
from heapq import heappop, heappush
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from .obj_geometry import read_obj_faces_stream, verify_obj_faces


def _obj(stream, check, update_digest=None):
    vertices, offsets, indices, uv, corners = read_obj_faces_stream(
        stream, check, with_uv=True, update_digest=update_digest,
    )
    if not np.all(np.diff(offsets) == 3):
        raise ValueError("Surface polygons require original triangles")
    return vertices, indices.reshape(-1, 3), uv, corners.reshape(-1, 3)


def _ply_geometry(stream, faces, check):
    from ..mesh_io import _check_ply_header

    encoding, elements = _check_ply_header(stream, None)
    endian = "<" if encoding == b"binary_little_endian" else ">"

    def skip(element):
        properties = element.properties
        if encoding != b"ascii" and all(prop.count_dtype is None for prop in properties):
            width = sum(prop.dtype.itemsize for prop in properties)
            stream.seek(element.count * width, 1)
            return
        for index in range(element.count):
            if index % 8192 == 0:
                check()
            if encoding == b"ascii":
                stream.readline()
                continue
            for prop in element.properties:
                count = 1
                if prop.count_dtype is not None:
                    dtype = prop.count_dtype.newbyteorder(endian)
                    count = int(np.frombuffer(stream.read(dtype.itemsize), dtype)[0])
                    if count < 0:
                        raise ValueError("Invalid PLY list count")
                stream.seek(count * prop.dtype.itemsize, 1)

    elements = iter(elements)
    for vertex in elements:
        if vertex.name == b"vertex":
            break
        skip(vertex)
    fields = vertex.properties
    if vertex.name != b"vertex" or any(p.count_dtype for p in fields):
        raise ValueError("Unsupported PLY vertex table")
    if encoding == b"ascii":
        columns = [
            next(i for i, p in enumerate(fields) if p.name == axis)
            for axis in (b"x", b"y", b"z")
        ]
        points = np.empty((vertex.count, 3))
        for i in range(vertex.count):
            if i % 8192 == 0:
                check()
            row = stream.readline().split()
            points[i] = [float(row[j]) for j in columns]
    else:
        dtype = [(p.name.decode(), p.dtype.newbyteorder(endian)) for p in fields]
        rows = np.fromfile(stream, dtype=dtype, count=vertex.count)
        points = np.column_stack([rows[k] for k in "xyz"])

    for face in elements:
        if face.name == b"face":
            break
        skip(face)
    if not any(prop.name == b"texcoord" for prop in face.properties):
        return points, None
    if face.count != len(faces):
        raise ValueError("PLY triangle count differs from displayed geometry")
    corner_uv = np.empty((face.count, 6), dtype=np.float64)
    for index in range(face.count):
        if index % 8192 == 0:
            check()
        if encoding == b"ascii":
            tokens = stream.readline().split()
            cursor = 0
        for prop in face.properties:
            count = 1
            if encoding == b"ascii":
                if prop.count_dtype is not None:
                    count = int(tokens[cursor])
                    cursor += 1
                values = tokens[cursor : cursor + count]
                cursor += count
            else:
                if prop.count_dtype is not None:
                    dtype = prop.count_dtype.newbyteorder(endian)
                    count = int(np.frombuffer(stream.read(dtype.itemsize), dtype)[0])
                dtype = prop.dtype.newbyteorder(endian)
                values = np.frombuffer(stream.read(count * dtype.itemsize), dtype)
            if prop.name == b"vertex_indices" and not np.array_equal(
                np.asarray(values, dtype=np.int64), faces[index]
            ):
                raise ValueError("Cannot verify PLY triangle corner order")
            if prop.name == b"texcoord":
                if count != 6:
                    raise ValueError("Expected three PLY texture coordinate pairs")
                corner_uv[index] = values
    if not np.isfinite(corner_uv).all():
        raise ValueError("Nonfinite PLY corner UVs")
    return points, corner_uv


def load_surface(layer, check):
    from vtk.util.numpy_support import vtk_to_numpy
    from ..session.io import source_stamp
    from ..session.mesh import mesh_attribute_values
    from ..session.state import _same

    path = Path(layer["path"])
    stamp = source_stamp(path)
    if layer.get("source_stamp", stamp) != stamp:
        raise ValueError("Mesh source changed since loading")
    mesh = layer["polydata"]
    with path.open("rb") as stream:
        digest = sha256()
        uv = corners = corner_uv = None
        if path.suffix.lower() == ".obj":
            vertices, faces, uv, corners = _obj(stream, check, update_digest=digest.update)
            fingerprint = digest.hexdigest()
            if layer.get("source_sha256", fingerprint) != fingerprint:
                raise ValueError("Mesh source changed since loading")
        else:
            while block := stream.read(1024 * 1024):
                check()
                digest.update(block)
            fingerprint = digest.hexdigest()
            if layer.get("source_sha256", fingerprint) != fingerprint:
                raise ValueError("Mesh source changed since loading")
            stream.seek(0)
            from ..mesh_io import load_mesh

            current = load_mesh(path, cancel_check=check)["polydata"]
            original = mesh.GetPolys()
            fresh = current.GetPolys()
            pairs = (
                (mesh.GetPoints().GetData(), current.GetPoints().GetData()),
                (original.GetConnectivityArray(), fresh.GetConnectivityArray()),
                (original.GetOffsetsArray(), fresh.GetOffsetsArray()),
            )
            if not _same(
                mesh_attribute_values(mesh), mesh_attribute_values(current), check
            ) or not all(
                np.array_equal(vtk_to_numpy(a), vtk_to_numpy(b)) for a, b in pairs
            ):
                raise ValueError("Mesh source changed since loading")
            cells = mesh.GetPolys()
            if mesh.GetNumberOfCells() != mesh.GetNumberOfPolys() or not np.all(
                np.diff(vtk_to_numpy(cells.GetOffsetsArray())) == 3
            ):
                raise ValueError("Surface polygons require original triangles")
            faces = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1, 3)
            display = vtk_to_numpy(mesh.GetPoints().GetData())
            vertices = display
            if path.suffix.lower() == ".ply":
                vertices, corner_uv = _ply_geometry(stream, faces, check)
            if len(vertices) != len(display) or not np.array_equal(
                vertices.astype(display.dtype), display
            ):
                raise ValueError("Cannot verify the original vertex order")
            texcoords = current.GetPointData().GetTCoords()
            if (
                corner_uv is None and texcoords
                and current.GetCellData().GetArray("original_corner_uv") is None
                and texcoords.GetNumberOfComponents() == 2
            ):
                uv = vtk_to_numpy(texcoords).copy()
                corners = faces
    if path.suffix.lower() == ".obj":
        verify_obj_faces(
            mesh, vertices, np.arange(len(faces) + 1) * 3, faces.ravel(), check
        )
    if len(faces) != mesh.GetNumberOfPolys() or not len(faces):
        raise ValueError("Source triangle correspondence unavailable")
    surface = Surface(vertices, faces)
    if corner_uv is not None:
        surface.face_attributes["original_corner_uv"] = corner_uv
    if path.suffix.lower() == ".vtp":
        for name, width in (("original_uv_indices", 3), ("original_corner_uv", 6)):
            field = mesh.GetCellData().GetArray(name)
            if field is not None:
                values = vtk_to_numpy(field)
                valid = (values.dtype.kind in "iu" and np.all(values >= 0)
                         if name == "original_uv_indices"
                         else values.dtype.kind in "iuf" and np.isfinite(values).all())
                if not valid or values.shape != (len(faces), width):
                    raise ValueError(f"Invalid source {name}")
                surface.face_attributes[name] = values
    reference = mesh.GetFieldData().GetAbstractArray("reference_space")
    if (
        reference
        and reference.IsA("vtkStringArray")
        and reference.GetNumberOfValues() == 1
    ):
        surface.reference_space = reference.GetValue(0)
    surface.uv, surface.corners = uv, corners
    surface.digest = fingerprint
    if source_stamp(path) != stamp:
        raise ValueError("Mesh source changed during preparation")
    check()
    return surface


class Surface:
    def __init__(self, vertices, faces):
        self.uv = self.corners = None
        self.face_attributes = {}
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces)
        if self.faces.min() < 0 or self.faces.max() >= len(vertices):
            raise ValueError("Invalid source vertex ID")
        edges = self.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        edges.sort(axis=1)
        keys = edges[:, 0].astype(np.int64) * len(vertices) + edges[:, 1]
        unique, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        self.edge_keys = unique
        self.edges = np.column_stack(np.divmod(unique, len(vertices)))
        self.face_edges = inverse.reshape(-1, 3)
        self.owners = np.argsort(inverse, kind="stable") // 3
        self.offsets = np.r_[0, np.cumsum(counts)]
        a, b = self.edges.T
        distance = np.linalg.norm(self.vertices[a] - self.vertices[b], axis=1)
        self.graph = coo_matrix(
            (np.tile(distance, 2), (np.r_[a, b], np.r_[b, a])),
            shape=(len(vertices), len(vertices)),
        ).tocsr()

    def corner_attributes(self, faces):
        attributes = {key: data[faces] for key, data in self.face_attributes.items()}
        if self.corners is not None:
            corners = self.corners[faces]
            if np.all(corners >= 0):
                attributes.setdefault("original_uv_indices", corners)
                if "original_corner_uv" not in attributes:
                    attributes["original_corner_uv"] = self.uv[corners].reshape(-1, 6)
            elif np.any(corners >= 0):
                raise ValueError("Territory contains partially textured triangles")
        return attributes

    def path(self, start, end, check=None):
        if start == end:
            raise ValueError("Choose a different vertex")
        vertices = self.vertices
        tx, ty, tz = float(vertices[end, 0]), float(vertices[end, 1]), float(vertices[end, 2])
        graph = self.graph
        indptr, indices, data = graph.indptr, graph.indices, graph.data
        queue = [(0.0, 0.0, start)]
        distance = {start: 0.0}
        parent = {}
        visited = 0
        while queue:
            _, cost, vertex = heappop(queue)
            if cost != distance[vertex]:
                continue
            if vertex == end:
                route = [end]
                while route[-1] != start:
                    route.append(parent[route[-1]])
                return route[::-1]
            if check and visited % 128 == 0:
                check()
            visited += 1
            for i in range(indptr[vertex], indptr[vertex + 1]):
                neighbor = int(indices[i])
                candidate = cost + float(data[i])
                if candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor] = candidate
                    parent[neighbor] = vertex
                    dx = float(vertices[neighbor, 0]) - tx
                    dy = float(vertices[neighbor, 1]) - ty
                    dz = float(vertices[neighbor, 2]) - tz
                    heappush(queue, (candidate + sqrt(dx * dx + dy * dy + dz * dz), candidate, neighbor))
        raise ValueError("Points are on disconnected surfaces")

    def boundary(self, ids):
        edges, counts = np.unique(self.face_edges[ids], return_counts=True)
        return self.edges[edges[counts == 1]]

    def verify_region(self, region, check):
        faces = region["faces"]
        if region["digest"] != self.digest or faces.max() >= len(self.faces):
            raise ValueError("Invalid saved polygon source or face ID")
        restored = self.region(region["loop"], faces[0], check)
        if not np.array_equal(restored, np.sort(faces)):
            raise ValueError("Saved polygon faces differ from its boundary")

    def region(self, loop, seed, check=None):
        if not 0 <= seed < len(self.faces):
            raise ValueError("Invalid interior triangle")
        if len(loop) < 4 or loop[0] != loop[-1] or len(set(loop[:-1])) != len(loop) - 1:
            raise ValueError("Boundary must be one simple closed cycle")
        border = np.sort(np.column_stack((loop[:-1], loop[1:])), axis=1)
        border = np.unique(border, axis=0)
        stride = len(self.vertices)
        keys = self.edge_keys
        barrier = np.searchsorted(keys, border[:, 0].astype(np.int64) * stride + border[:, 1])
        if np.any(barrier >= len(keys)) or not np.array_equal(
            self.edges[barrier], border
        ):
            raise ValueError("Boundary contains a non-mesh edge")
        visited_edges = np.zeros(len(keys), dtype=bool)
        visited_edges[barrier] = True
        visited_faces = np.zeros(len(self.faces), dtype=bool)
        front = np.array([seed], dtype=np.int64)
        visited_faces[front] = True
        while front.size:
            if check:
                check()
            edges = np.unique(self.face_edges[front])
            edges = edges[~visited_edges[edges]]
            visited_edges[edges] = True
            starts = self.offsets[edges]
            counts = self.offsets[edges + 1] - starts
            offsets = np.repeat(starts - np.cumsum(counts) + counts, counts)
            faces = np.unique(self.owners[offsets + np.arange(offsets.size)])
            front = faces[~visited_faces[faces]]
            visited_faces[front] = True
        ids = np.flatnonzero(visited_faces)
        if not np.array_equal(self.boundary(ids), border):
            raise ValueError("Chosen side does not have exactly the drawn boundary")
        return ids
