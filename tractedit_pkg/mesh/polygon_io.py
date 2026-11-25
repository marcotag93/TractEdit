"""VTP interchange for an original-triangle patch and its closed boundary."""

from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from ..transactional_io import staged_output_set

SUPPORTED_POLYGON_EXTENSIONS = (".vtp", ".obj")


def _array(values, name):
    result = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    result.SetName(name)
    return result


def _data(points, cells, *, line=False):
    data = vtk.vtkPolyData()
    coords = vtk.vtkPoints()
    coords.SetData(_array(points, "Points"))
    data.SetPoints(coords)
    indices = np.asarray(cells, dtype=np.int64)
    offsets = np.arange(0, indices.size + 1, indices.shape[1], dtype=np.int64)
    array = vtk.vtkCellArray()
    array.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(indices.ravel(), deep=True),
    )
    if line:
        data.SetLines(array)
    else:
        data.SetPolys(array)
    return data


def ras_points(surface, ids, layer):
    points = surface.vertices[ids].copy()
    if layer["coordinate_system"] == "LPS":
        points[:, :2] *= -1
    return points


def patch_geometry(surface, region, layer):
    vertices, local = np.unique(surface.faces[region["faces"]], return_inverse=True)
    return _data(ras_points(surface, vertices, layer), local.reshape(-1, 3)), vertices


def datasets(surface, region, layer):
    faces, loop = region["faces"], region["loop"]
    patch, vertices = patch_geometry(surface, region, layer)
    patch.GetPointData().AddArray(_array(vertices, "original_vertex_id"))
    patch.GetCellData().AddArray(_array(faces, "original_face_id"))
    for name, values in surface.corner_attributes(faces).items():
        patch.GetCellData().AddArray(_array(values, name))
    vertices, local = np.unique(loop, return_inverse=True)
    line = _data(ras_points(surface, vertices, layer), local.reshape(1, -1), line=True)
    line.GetPointData().AddArray(_array(vertices, "original_vertex_id"))
    line.GetCellData().AddArray(_array(np.array([0]), "boundary_id"))
    fields = {"SPACE": "RAS", "units": "mm", "source_sha256": surface.digest}
    reference = region.get("reference_space", getattr(surface, "reference_space", None))
    if reference:
        fields["reference_space"] = reference
    for data in (line, patch):
        for name, value in fields.items():
            field = vtk.vtkStringArray()
            field.SetName(name)
            field.InsertNextValue(value)
            data.GetFieldData().AddArray(field)
    return line, patch


def output_paths(path, extension=None):
    base = Path(path)
    ext = (extension if extension is not None else (base.suffix or ".vtp")).lower()
    if ext not in SUPPORTED_POLYGON_EXTENSIONS:
        raise ValueError("Supported surface polygon formats: VTP and OBJ")
    stem = base.stem.removesuffix("_polygon").removesuffix("_patch")
    return tuple(
        base.with_name(f"{stem}{suffix}{ext}") for suffix in ("_polygon", "_patch")
    )


def resolve_import_pair(paths):
    """Resolve one or two selected VTP/OBJ files into a (boundary, patch) pair."""
    selected = [Path(p) for p in paths]
    if len(selected) not in (1, 2):
        raise ValueError("Select a matching polygon and patch file (.vtp or .obj)")
    exts = [p.suffix.lower() for p in selected]
    if any(ext not in SUPPORTED_POLYGON_EXTENSIONS for ext in exts):
        raise ValueError("Supported surface polygon formats: VTP and OBJ")
    if len(selected) == 1:
        chosen = selected[0]
        stem, lower = chosen.stem, chosen.stem.lower()
        if lower.endswith("_polygon"):
            sibling = chosen.with_name(stem[:-8] + "_patch" + chosen.suffix)
            pair = (chosen, sibling)
        elif lower.endswith("_patch"):
            sibling = chosen.with_name(stem[:-6] + "_polygon" + chosen.suffix)
            pair = (sibling, chosen)
        else:
            raise ValueError(
                "Select both the polygon and patch files, or a *_polygon/*_patch file "
                "with its matching sibling in the same folder"
            )
        if not sibling.is_file():
            raise ValueError(f"Matching sibling file not found: {sibling.name}")
        return pair
    if exts[0] != exts[1]:
        raise ValueError("Polygon and patch files must have the same format (.vtp or .obj)")
    if selected[0].resolve() == selected[1].resolve():
        raise ValueError("Select distinct polygon and patch files")
    line, patch = sorted(selected, key=lambda p: p.stem.lower().endswith("_patch"))
    return line, patch


def _obj_header(kind, surface, region):
    lines = [
        f"# TractEdit surface polygon {kind}; SPACE=RAS; units=mm\n",
        f"# source_sha256={surface.digest}\n",
    ]
    reference = region.get("reference_space", getattr(surface, "reference_space", None))
    if reference:
        clean = reference if (reference.isascii() and "\r" not in reference and "\n" not in reference) else ascii(reference)
        lines.append(f"# reference_space={clean}\n")
    return "".join(lines)


def write_obj_pair(paths, surface, region, layer, color, check=None, *, protected_paths=()):
    from .polylines import protected_export_path, validated_color

    color = validated_color(color)
    for path in paths:
        protected_export_path(path, protected_paths)
    if check:
        check()
    faces, loop = region["faces"], region["loop"]
    attributes = surface.corner_attributes(faces)
    boundary_pts = ras_points(surface, loop[:-1], layer)
    vertex_ids, local = np.unique(surface.faces[faces], return_inverse=True)
    tris = local.reshape(-1, 3)
    patch_pts = ras_points(surface, vertex_ids, layer)
    uv_coords = uv_tris = None
    if surface.corners is not None and "original_uv_indices" in attributes:
        uv_ids, local_uv = np.unique(attributes["original_uv_indices"], return_inverse=True)
        uv_coords = surface.uv[uv_ids]
        uv_tris = local_uv.reshape(-1, 3)
    elif "original_corner_uv" in attributes:
        corner_uv = attributes["original_corner_uv"].reshape(-1, 2)
        uv_coords, local_uv = np.unique(corner_uv, axis=0, return_inverse=True)
        uv_tris = local_uv.reshape(-1, 3)
    rgb = " ".join(format(v, ".17g") for v in color)
    line_path, patch_path = paths
    with staged_output_set(paths) as staged:
        with staged[line_path].open("w", encoding="ascii", newline="\n") as stream:
            stream.write(_obj_header("boundary", surface, region))
            for index, point in enumerate(boundary_pts):
                if check and index % 4096 == 0:
                    check()
                xyz = " ".join(format(v + 0.0, ".17g") for v in point)
                stream.write(f"v {xyz} {rgb}\n")
            stream.write(
                "l " + " ".join(str(i + 1) for i in range(len(boundary_pts))) + " 1\n"
            )
        with staged[patch_path].open("w", encoding="ascii", newline="\n") as stream:
            stream.write(_obj_header("patch", surface, region))
            for index, point in enumerate(patch_pts):
                if check and index % 4096 == 0:
                    check()
                xyz = " ".join(format(v + 0.0, ".17g") for v in point)
                stream.write(f"v {xyz} {rgb}\n")
            if uv_coords is not None:
                for index, uv in enumerate(uv_coords):
                    if check and index % 4096 == 0:
                        check()
                    stream.write(f"vt {format(uv[0], '.17g')} {format(uv[1], '.17g')}\n")
                for index, (tri, uv_tri) in enumerate(zip(tris, uv_tris)):
                    if check and index % 4096 == 0:
                        check()
                    a, b, c = tri + 1
                    ta, tb, tc = uv_tri + 1
                    stream.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
            else:
                for index, tri in enumerate(tris):
                    if check and index % 4096 == 0:
                        check()
                    a, b, c = tri + 1
                    stream.write(f"f {a} {b} {c}\n")
        if check:
            check()
        for path in paths:
            protected_export_path(path, protected_paths)


def write_pair(paths, surface, region, layer, check=None, *, protected_paths=()):
    from .polylines import protected_export_path

    for path in paths:
        protected_export_path(path, protected_paths)
    if check:
        check()
    with staged_output_set(paths) as staged:
        for path, data in zip(paths, datasets(surface, region, layer)):
            if check:
                check()
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(str(staged[path]))
            writer.SetInputData(data)
            if not writer.Write() or writer.GetErrorCode():
                raise OSError(f"Could not write {path.name}")
        if check:
            check()
        for path in paths:
            protected_export_path(path, protected_paths)


def read_pair(line_path, patch_path, surface, layer, check=None, *, return_boundary=False):
    from ..mesh_io import read_annotation_vtp

    def string_field(fields, name):
        value = fields.GetAbstractArray(name)
        if value is None:
            return None
        if (not value.IsA("vtkStringArray") or value.GetNumberOfComponents() != 1
                or value.GetNumberOfValues() != 1):
            raise ValueError(f"Invalid annotation {name}")
        return value.GetValue(0)

    def read(path):
        if check:
            check()
        data = read_annotation_vtp(path, cancel_check=check)
        fields = data.GetFieldData()
        units = string_field(fields, "units")
        if units not in (None, "mm"):
            raise ValueError("Expected annotation coordinates in millimetres")
        reference = string_field(fields, "reference_space")
        if reference is not None and getattr(surface, "reference_space", None) not in (
            None, reference,
        ):
            raise ValueError("Annotation reference space differs from source mesh")
        field = string_field(fields, "source_sha256")
        if field not in (None, surface.digest):
            raise ValueError("Annotation belongs to a different mesh")
        vertices = ids(data.GetPointData(), "original_vertex_id", len(surface.vertices))
        space = string_field(fields, "SPACE")
        actual = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float64)
        if space == "LPS":
            actual[:, :2] *= -1
        elif space not in (None, "RAS"):
            raise ValueError("Unsupported annotation coordinates")
        if len(vertices) != len(actual) or not np.allclose(
            actual, ras_points(surface, vertices, layer), rtol=0, atol=1e-6
        ):
            raise ValueError("Annotation coordinates differ from the source mesh")
        return data, vertices, reference

    def ids(attributes, name, limit):
        field = attributes.GetArray(name)
        if field is None:
            raise ValueError(f"Missing {name}")
        values = vtk_to_numpy(field)
        if (
            values.ndim != 1
            or values.dtype.kind not in "iu"
            or not len(values)
            or values.min() < 0
            or values.max() >= limit
        ):
            raise ValueError(f"Invalid {name}")
        return values.astype(np.int64)

    line, vertices, line_reference = read(line_path)
    patch, patch_vertices, patch_reference = read(patch_path)
    if (line_reference is not None and patch_reference is not None
            and line_reference != patch_reference):
        raise ValueError("Boundary and patch reference spaces differ")
    if line.GetNumberOfPolys() and patch.GetNumberOfLines():
        line_path = patch_path
        line, patch = patch, line
        vertices, patch_vertices = patch_vertices, vertices
    if line.GetNumberOfCells() != 1 or line.GetNumberOfLines() != 1:
        raise ValueError("Expected one closed boundary line")
    loop = vertices[vtk_to_numpy(line.GetLines().GetConnectivityArray())]
    vertices = patch_vertices
    faces = ids(patch.GetCellData(), "original_face_id", len(surface.faces))
    polys = patch.GetPolys()
    if (
        len(faces) != patch.GetNumberOfCells()
        or len(faces) != patch.GetNumberOfPolys()
        or not np.all(np.diff(vtk_to_numpy(polys.GetOffsetsArray())) == 3)
    ):
        raise ValueError("Patch must contain original triangles")
    actual = vertices[vtk_to_numpy(polys.GetConnectivityArray())].reshape(-1, 3)
    if not np.array_equal(
        np.sort(actual, axis=1), np.sort(surface.faces[faces], axis=1)
    ):
        raise ValueError("Patch connectivity differs from original faces")
    attributes = None
    for name in ("original_uv_indices", "original_corner_uv"):
        field = patch.GetCellData().GetArray(name)
        if field is None:
            continue
        if attributes is None:
            attributes = surface.corner_attributes(faces)
        if name not in attributes:
            raise ValueError("Unexpected patch UV attributes")
        expected = attributes[name]
        values = vtk_to_numpy(field)
        if values.shape != expected.shape:
            raise ValueError("Invalid patch UV dimensions")
        order = np.argsort(actual, axis=1)
        expected_order = np.argsort(surface.faces[faces], axis=1)
        if name == "original_corner_uv":
            values = values.reshape(-1, 3, 2)
            expected = expected.reshape(-1, 3, 2)
            order = order[:, :, None]
            expected_order = expected_order[:, :, None]
        if not np.array_equal(
            np.take_along_axis(values, order, axis=1),
            np.take_along_axis(expected, expected_order, axis=1),
        ):
            raise ValueError("Patch UVs differ from the source mesh")
    if not np.array_equal(
        surface.region(loop.tolist(), faces[0], check), np.sort(faces)
    ):
        raise ValueError("Patch does not match the supplied boundary")
    region = {"loop": loop, "faces": faces, "digest": surface.digest}
    reference = patch_reference or line_reference
    if reference:
        region["reference_space"] = reference
    return (region, line_path) if return_boundary else region


def _read_polygon_obj(path, check=None):
    from .polylines import validated_color

    meta = {}
    vertices, colors, uvs, lines, faces, face_uvs = [], [], [], [], [], []
    has_uv = None
    with Path(path).open("r", encoding="ascii", errors="strict") as stream:
        for number, raw in enumerate(stream):
            if check and number % 4096 == 0:
                check()
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                for part in stripped[1:].split(";"):
                    if "=" not in part:
                        continue
                    key, value = (token.strip() for token in part.split("=", 1))
                    if key in ("SPACE", "units", "source_sha256", "reference_space"):
                        if key in meta and meta[key] != value:
                            raise ValueError(f"Conflicting annotation {key}")
                        meta[key] = value
                continue
            fields = stripped.partition("#")[0].split()
            if not fields:
                continue
            cmd, args = fields[0], fields[1:]
            if cmd == "v":
                if len(args) not in (3, 6):
                    raise ValueError("OBJ vertex must have 3 (XYZ) or 6 (XYZ RGB) values")
                vals = [float(a) for a in args]
                if not all(np.isfinite(vals)):
                    raise ValueError("Nonfinite OBJ vertex coordinate")
                vertices.append(vals[:3])
                if len(vals) == 6:
                    colors.append(validated_color(vals[3:6]))
            elif cmd == "vt":
                if len(args) != 2:
                    raise ValueError("OBJ texture coordinate must have 2 values")
                uv = [float(a) for a in args]
                if not all(np.isfinite(uv)):
                    raise ValueError("Nonfinite OBJ texture coordinate")
                uvs.append(uv)
            elif cmd == "l":
                if len(args) < 2:
                    raise ValueError("OBJ line must reference at least two vertices")
                line_ids = []
                for token in args:
                    idx = int(token.split("/")[0])
                    if idx == 0:
                        raise ValueError("Invalid OBJ line index")
                    resolved = idx - 1 if idx > 0 else len(vertices) + idx
                    if not 0 <= resolved < len(vertices):
                        raise ValueError("OBJ line index out of range")
                    line_ids.append(resolved)
                lines.append(np.asarray(line_ids, dtype=np.int64))
            elif cmd == "f":
                if len(args) != 3:
                    raise ValueError("Patch must contain original triangles")
                tri, tri_uv = [], []
                for token in args:
                    parts = token.split("/")
                    if len(parts) > 3 or not parts[0]:
                        raise ValueError("Invalid OBJ face syntax")
                    v_idx = int(parts[0])
                    if v_idx == 0:
                        raise ValueError("Invalid OBJ face vertex index")
                    v_res = v_idx - 1 if v_idx > 0 else len(vertices) + v_idx
                    if not 0 <= v_res < len(vertices):
                        raise ValueError("OBJ face vertex index out of range")
                    tri.append(v_res)
                    if len(parts) > 1 and parts[1]:
                        t_idx = int(parts[1])
                        if t_idx == 0:
                            raise ValueError("Invalid OBJ texture index")
                        t_res = t_idx - 1 if t_idx > 0 else len(uvs) + t_idx
                        if not 0 <= t_res < len(uvs):
                            raise ValueError("OBJ texture index out of range")
                        tri_uv.append(t_res)
                face_has_uv = len(tri_uv) == 3
                if len(tri_uv) not in (0, 3) or (has_uv is not None and has_uv != face_has_uv):
                    raise ValueError("Inconsistent OBJ face UV references")
                has_uv = face_has_uv
                faces.append(tri)
                if face_has_uv:
                    face_uvs.append(tri_uv)
            elif cmd not in ("o", "g", "s", "vn"):
                raise ValueError(f"Unsupported OBJ annotation record: {cmd}")
    if not vertices:
        raise ValueError("OBJ annotation contains no vertices")
    points = np.asarray(vertices, dtype=np.float64)
    color = (
        colors[0]
        if len(colors) == len(vertices)
        and np.allclose(colors, colors[0], rtol=0, atol=1e-6)
        else None
    )
    return {
        "points": points,
        "uv": np.asarray(uvs, dtype=np.float64) if uvs else None,
        "lines": lines,
        "faces": np.asarray(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64),
        "face_uvs": np.asarray(face_uvs, dtype=np.int64) if face_uvs else None,
        "meta": meta,
        "color": color,
    }


def read_obj_pair(
    line_path,
    patch_path,
    surface,
    layer,
    check=None,
    *,
    return_boundary=False,
    return_color=False,
):
    """Read and topologically verify a paired OBJ boundary and patch against surface."""
    def parse_checked(path):
        if check:
            check()
        try:
            data = _read_polygon_obj(path, check=check)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(str(error)) from error
        meta = data["meta"]
        if meta.get("units") not in (None, "mm"):
            raise ValueError("Expected annotation coordinates in millimetres")
        reference = meta.get("reference_space")
        if reference is not None and getattr(surface, "reference_space", None) not in (
            None, reference,
        ):
            raise ValueError("Annotation reference space differs from source mesh")
        digest = meta.get("source_sha256")
        if digest not in (None, surface.digest):
            raise ValueError("Annotation belongs to a different mesh")
        space = meta.get("SPACE")
        if space == "LPS":
            data["points"][:, :2] *= -1
        elif space not in (None, "RAS"):
            raise ValueError("Unsupported annotation coordinates")
        return data, reference

    line, line_ref = parse_checked(line_path)
    patch, patch_ref = parse_checked(patch_path)
    if line_ref is not None and patch_ref is not None and line_ref != patch_ref:
        raise ValueError("Boundary and patch reference spaces differ")
    if len(line["faces"]) and not line["lines"] and patch["lines"] and not len(patch["faces"]):
        line_path, patch_path = patch_path, line_path
        line, patch = patch, line
    if len(line["lines"]) != 1 or len(line["faces"]):
        raise ValueError("Expected one closed boundary line")
    if not len(patch["faces"]) or patch["lines"]:
        raise ValueError("Patch must contain original triangles")

    line_idx = line["lines"][0]
    line_pts = line["points"]
    if len(line_idx) < 4:
        raise ValueError("Expected one closed boundary line")
    if line_idx[0] == line_idx[-1] or np.allclose(
        line_pts[line_idx[0]], line_pts[line_idx[-1]], rtol=0, atol=1e-6
    ):
        boundary_pts = line_pts[line_idx[:-1]]
    else:
        raise ValueError("Expected one closed boundary line")

    surface_ras = ras_points(surface, np.arange(len(surface.vertices)), layer)
    tree = cKDTree(surface_ras)
    patch_cands = tree.query_ball_point(patch["points"], r=1e-6)
    if any(not group for group in patch_cands):
        raise ValueError("Annotation coordinates differ from the source mesh")

    cand_vids = np.unique([vid for group in patch_cands for vid in group])
    mask = np.zeros(len(surface.vertices), dtype=bool)
    mask[cand_vids] = True
    active_faces = np.flatnonzero(mask[surface.faces].all(axis=1))
    face_lookup = {}
    for fid in active_faces:
        key = tuple(sorted(int(v) for v in surface.faces[fid]))
        face_lookup.setdefault(key, []).append(int(fid))

    has_patch_uv = patch["face_uvs"] is not None

    def face_uv(fid):
        if "original_corner_uv" in surface.face_attributes:
            return surface.face_attributes["original_corner_uv"][fid].reshape(3, 2)
        if surface.corners is not None:
            corners = surface.corners[fid]
            if np.all(corners >= 0):
                return surface.uv[corners]
        return None

    def uv_matches(fid, tri_verts, tri_uv):
        src_uv = face_uv(fid)
        if src_uv is None:
            return False
        src_verts = surface.faces[fid]
        for corner, vid in enumerate(tri_verts):
            pos = int(np.nonzero(src_verts == vid)[0][0])
            if not np.allclose(tri_uv[corner], src_uv[pos], rtol=0, atol=1e-6):
                return False
        return True

    matched_faces = []
    for f_idx, tri in enumerate(patch["faces"]):
        if check and f_idx % 4096 == 0:
            check()
        tri_uv = patch["uv"][patch["face_uvs"][f_idx]] if has_patch_uv else None
        hits = []
        seen = set()
        for va, vb, vc in product(
            patch_cands[tri[0]], patch_cands[tri[1]], patch_cands[tri[2]]
        ):
            if va == vb or vb == vc or va == vc:
                continue
            for fid in face_lookup.get(tuple(sorted((va, vb, vc))), ()):
                if fid in seen:
                    continue
                if has_patch_uv and not uv_matches(fid, (va, vb, vc), tri_uv):
                    continue
                seen.add(fid)
                hits.append(fid)
        if not hits:
            raise ValueError(
                "Patch UVs differ from the source mesh"
                if has_patch_uv and any(
                    face_lookup.get(tuple(sorted((va, vb, vc))))
                    for va, vb, vc in product(
                        patch_cands[tri[0]], patch_cands[tri[1]], patch_cands[tri[2]]
                    )
                    if va != vb and vb != vc and va != vc
                )
                else "Patch connectivity differs from original faces"
            )
        if len(hits) > 1:
            raise ValueError("Patch matches ambiguous coincident faces on source mesh")
        matched_faces.append(hits[0])

    faces = np.asarray(matched_faces, dtype=np.int64)
    if len(np.unique(faces)) != len(faces):
        raise ValueError("Patch contains duplicate original triangles")

    border_edges = surface.boundary(faces)
    if not len(border_edges):
        raise ValueError("Patch does not match the supplied boundary")
    boundary_vids = np.unique(border_edges)
    adj = {int(v): set() for v in boundary_vids}
    for u, v in border_edges:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    b_tree = cKDTree(surface_ras[boundary_vids])
    loop_cands = [
        [int(boundary_vids[idx]) for idx in b_tree.query_ball_point(pt, r=1e-6)]
        for pt in boundary_pts
    ]
    if any(not group for group in loop_cands):
        raise ValueError("Annotation coordinates differ from the source mesh")
    if all(len(group) == 1 for group in loop_cands):
        resolved = [group[0] for group in loop_cands]
    else:
        resolved = None
        for start in loop_cands[0]:
            path_ids = [start]
            used = {start}
            for group in loop_cands[1:]:
                nxt = next(
                    (v for v in group if v in adj[path_ids[-1]] and v not in used),
                    None,
                )
                if nxt is None:
                    break
                path_ids.append(nxt)
                used.add(nxt)
            if len(path_ids) == len(loop_cands) and start in adj[path_ids[-1]]:
                resolved = path_ids
                break
        if resolved is None:
            raise ValueError("Patch does not match the supplied boundary")

    loop = np.asarray([*resolved, resolved[0]], dtype=np.int64)
    if not np.array_equal(
        surface.region(loop.tolist(), int(faces[0]), check), np.sort(faces)
    ):
        raise ValueError("Patch does not match the supplied boundary")

    region = {"loop": loop, "faces": faces, "digest": surface.digest}
    reference = patch_ref or line_ref
    if reference:
        region["reference_space"] = reference
    color = line["color"] if line["color"] is not None else patch["color"]
    if return_boundary and return_color:
        return region, line_path, color
    if return_boundary:
        return region, line_path
    if return_color:
        return region, color
    return region
