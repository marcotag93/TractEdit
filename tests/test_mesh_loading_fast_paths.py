"""Deterministic unit tests for Phase 1 & Phase 2 mesh/texture fast paths."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOImage import vtkPNGWriter

from tractedit_pkg import mesh_io, mesh_texture
from tractedit_pkg._numba_aot._parallel_wrappers import (
    validate_mesh_points_range,
    validate_mesh_triangles_range,
)
from tractedit_pkg.mesh import obj_geometry
from tractedit_pkg.mesh.model import create_mesh_layer
from tractedit_pkg.session.mesh_texture import (
    prepare_saved_textures,
    texture_configuration,
)
from tractedit_pkg.visualization.mesh_texture import create_mesh_texture_presentation


def _write_png(path: Path, rgb: tuple[int, int, int] = (64, 128, 192)) -> None:
    image = vtkImageData()
    image.SetDimensions(2, 2, 1)
    image.AllocateScalars(3, 3)
    scalars = image.GetPointData().GetScalars()
    for i in range(4):
        scalars.SetTuple3(i, *rgb)
    writer = vtkPNGWriter()
    writer.SetInputData(image)
    writer.SetFileName(str(path))
    writer.Write()


def test_aot_point_and_triangle_validation_matches_numpy_reference():
    rng = np.random.default_rng(7)
    for dtype in (np.float32, np.float64):
        pts = rng.uniform(-100.0, 100.0, size=(128, 3)).astype(dtype)
        ok, bounds = validate_mesh_points_range(pts, 0, len(pts))
        assert ok
        expected_min = pts.min(axis=0).astype(np.float64)
        expected_max = pts.max(axis=0).astype(np.float64)
        np.testing.assert_array_equal(bounds[:3], expected_min)
        np.testing.assert_array_equal(bounds[3:], expected_max)

        pts_nan = pts.copy()
        pts_nan[77, 1] = np.nan
        ok_nan, _ = validate_mesh_points_range(pts_nan, 0, len(pts_nan))
        assert not ok_nan

    # Extreme scales test exact frexp/ldexp power-of-two scaling parity.
    for scale in (1e-30, 1.0, 1e30):
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        ) * scale
        valid_idx = np.array([0, 1, 2], dtype=np.int64)
        collinear_idx = np.array([0, 1, 3], dtype=np.int64)
        repeated_idx = np.array([0, 1, 1], dtype=np.int64)

        assert validate_mesh_triangles_range(pts, valid_idx, False, 0, 1) == (0, True)
        assert validate_mesh_triangles_range(pts, collinear_idx, False, 0, 1) == (2, False)
        assert validate_mesh_triangles_range(pts, collinear_idx, True, 0, 1) == (0, False)
        assert validate_mesh_triangles_range(pts, repeated_idx, True, 0, 1) == (1, False)


def test_obj_stream_parser_bit_exact_against_python_reference(tmp_path):
    lines = [
        b"# header comment\n",
        b"mtllib test.mtl\n",
        b"o brain\n",
        b"v 0 0 0\n",
        b"v -0.0 1.250000000000001 -3.5\n",
        b"v 1.5e-4 2.1234567890123456789 4.0 0.5 0.5 0.5\n",
        b"v .25 5. -123456789.125 # trailing comment\n",
        b"v 10 20 30\n",
        b"vt 0.0 0.0\n",
        b"vt 1.0 0.123456789012345\n",
        b"vt 0.5 1.0\n",
        b"vt 1e-3 0.75\n",
        b"vn 0 0 1\n",
        b"usemtl mat1\n",
        b"f 1/1/1 2/2/1 3/3/1\n",
        b"f 1/1 3/3 4/4\n",
        b"f 1//1 2//1 4//1\n",
        b"f 1 2 3\n",
        b"f -4/-4 -3/-3 \\\n",
        b"  -2/-2 -1/-1\n",
        b"f 2/2 3/3 4/4 5/1 # quad\n",
    ]
    payload = b"".join(lines)
    path = tmp_path / "mixed.obj"
    path.write_bytes(payload)

    ref = obj_geometry.read_obj_faces(iter(lines), lambda: None, with_uv=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        fast = obj_geometry.read_obj_faces_stream(
            stream, lambda: None, with_uv=True, update_digest=digest.update,
        )
    assert digest.hexdigest() == hashlib.sha256(payload).hexdigest()
    for a, b in zip(ref, fast):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
        if a.dtype == np.float64:
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


def test_obj_large_block_preflight_and_texture_index_reuse(tmp_path):
    _write_png(tmp_path / "tex.png")
    (tmp_path / "surf.mtl").write_text(
        "newmtl cortex\nmap_Kd tex.png\n", encoding="ascii"
    )
    # Exceed _OBJ_BLOCK_BYTES // 2 (512 KiB) so the AOT block scanner executes.
    n_tris = 12_000
    body = ["mtllib surf.mtl\n", "usemtl cortex\n"]
    body.extend(
        f"v {i * 0.001:.6f} {(i % 2) * 0.5:.6f} 0.0\n"
        for i in range(n_tris + 2)
    )
    body.extend("vt 0.25 0.75\n" for _ in range(n_tris + 2))
    body.extend("vn 0.0 0.0 1.0\n" for _ in range(n_tris + 2))
    body.extend(
        f"f {i}/{i}/{i} {i + 1}/{i + 1}/{i + 1} {i + 2}/{i + 2}/{i + 2}\n"
        for i in range(1, n_tris + 1)
    )
    obj = tmp_path / "surf.obj"
    obj.write_bytes("".join(body).encode("ascii"))
    assert obj.stat().st_size > mesh_io._OBJ_BLOCK_BYTES // 2

    loaded = mesh_io.load_mesh(obj)
    assert loaded["polygon_count"] == n_tris
    assert "obj_texture_index" in loaded
    idx = loaded["obj_texture_index"]
    assert idx["face_layout"] == {
        "cortex": {"face_count": n_tris, "corner_count": 3 * n_tris, "complete_uv": True}
    }
    assert idx["face_uv_bytes"] is None

    packed = {
        "path": loaded["path"],
        "stamp": loaded["source_stamp"],
        "sha256": loaded["source_sha256"],
        "size": loaded["source_stamp"][2],
        "index": idx,
    }
    direct = mesh_texture.prepare_obj_textures(obj, gl_maximum=16384)
    indexed = mesh_texture.prepare_obj_textures(
        obj, gl_maximum=16384, _obj_index=packed
    )
    assert indexed["source"] == direct["source"]
    assert indexed["face_layout"] == direct["face_layout"]
    assert indexed["materials"] == direct["materials"]
    assert indexed["costs"] == direct["costs"]

    config, assets = texture_configuration(indexed, True)
    restored = prepare_saved_textures(
        obj, config, assets, gl_maximum=16384,
        budgets=mesh_texture.DEFAULT_BUDGETS, _obj_index=packed,
    )
    assert restored["face_layout"] == direct["face_layout"]

    # Mutating the OBJ on disk must invalidate the stale index.
    modified = obj.read_bytes().replace(b"usemtl cortex\n", b"usemtl OTHER!\n", 1)
    obj.write_bytes(modified)
    stale_check = mesh_texture._validate_obj_index(obj.resolve(), packed, None)
    assert stale_check is None


def test_binary_ply_multi_property_faces_and_single_pass_sha256(tmp_path):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment SPACE=RAS\n"
        "element vertex 4\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "element face 3\n"
        "property list uchar int vertex_indices\n"
        "property list uchar float texcoord\n"
        "property uchar red\n"
        "end_header\n"
    ).encode("ascii")
    verts = b"".join(
        struct.pack("<ddd", *xyz)
        for xyz in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0))
    )
    # First 2 faces are homogeneous triangles, 3rd is a quad to exercise fallback.
    f0 = struct.pack("<BiiiBffffffB", 3, 0, 1, 2, 6, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 200)
    f1 = struct.pack("<BiiiBffffffB", 3, 1, 3, 2, 6, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 201)
    f2 = struct.pack(
        "<BiiiiBffffffffB", 4, 0, 1, 3, 2, 8, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 202
    )
    trailing = b"\n"
    payload = header + verts + f0 + f1 + f2 + trailing
    path = tmp_path / "textured_double.ply"
    path.write_bytes(payload)

    result = mesh_io.load_mesh(path)
    assert result["polygon_count"] == 3
    assert result["source_sha256"] == hashlib.sha256(payload).hexdigest()


def test_texture_presentation_single_and_multimaterial_triangles_and_quads(tmp_path):
    import vtk

    _write_png(tmp_path / "a.png", (255, 0, 0))
    _write_png(tmp_path / "b.png", (0, 255, 0))
    (tmp_path / "single.mtl").write_text(
        "newmtl matA\nmap_Kd a.png\n", encoding="ascii"
    )
    single_obj = tmp_path / "single.obj"
    single_obj.write_text(
        "mtllib single.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "usemtl matA\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 3/3 4/4\n",
        encoding="ascii",
    )
    s_loaded = mesh_io.load_mesh(single_obj)
    s_layer = create_mesh_layer(s_loaded, "RAS")
    s_prep = mesh_texture.prepare_obj_textures(single_obj, gl_maximum=16384)
    s_mapper = vtk.vtkPolyDataMapper()
    s_mapper.SetInputData(s_layer["polydata"])
    s_pres = create_mesh_texture_presentation(
        s_layer, s_prep, s_mapper, gl_maximum=16384
    )
    assert s_pres.parts == ()
    s_pres.release(None)

    (tmp_path / "multi.mtl").write_text(
        "newmtl matA\nmap_Kd a.png\nnewmtl matB\nmap_Kd b.png\nnewmtl matC\nKd 0.5 0.5 0.5\n",
        encoding="ascii",
    )
    obj = tmp_path / "multi.obj"
    obj.write_text(
        "mtllib multi.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "usemtl matA\n"
        "f 1/1 2/2 3/3\n"
        "usemtl matB\n"
        "f 1/1 2/2 3/3 4/4\n"
        "usemtl matC\n"
        "f 1/1 3/3 4/4\n",
        encoding="ascii",
    )
    loaded = mesh_io.load_mesh(obj)
    layer = create_mesh_layer(loaded, "RAS")
    prep = mesh_texture.prepare_obj_textures(obj, gl_maximum=16384)

    uniform_mapper = vtk.vtkPolyDataMapper()
    uniform_mapper.SetInputData(layer["polydata"])
    presentation = create_mesh_texture_presentation(
        layer, prep, uniform_mapper, gl_maximum=16384
    )
    # 3 parts: matA (tri), matB (quad), untextured fallback matC (tri)
    assert len(presentation.parts) == 2
    assert presentation.source_cells.tolist() == [0]
    part_cells = [part.source_cells.tolist() for part in presentation.parts]
    assert part_cells == [[1], [2]]
    presentation.release(None)
