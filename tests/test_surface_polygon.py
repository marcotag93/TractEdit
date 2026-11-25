import numpy as np
import pytest

from tractedit_pkg.mesh.polygon import Surface
from tractedit_pkg.mesh.polygon_io import (
    output_paths,
    read_obj_pair,
    read_pair,
    resolve_import_pair,
    write_obj_pair,
    write_pair,
)


def tetrahedron():
    surface = Surface(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
    )
    surface.digest = "a" * 64
    surface.corners = None
    return surface


def test_surface_polygon_side_and_invalid_boundary():
    surface = tetrahedron()
    loop = [0, 1, 2, 0]
    np.testing.assert_array_equal(surface.region(loop, 0), [0])
    np.testing.assert_array_equal(surface.region(loop, 2), [1, 2, 3])
    assert surface.path(0, 2) == [0, 2]
    with pytest.raises(ValueError):
        surface.region([0, 1, 0, 2, 0], 0)


def test_surface_polygon_pair(tmp_path):
    surface = tetrahedron()
    region = dict(
        faces=np.array([0]), loop=np.array([0, 1, 2, 0]), digest=surface.digest
    )
    paths = output_paths(tmp_path / "example.vtp")
    layer = {"coordinate_system": "LPS"}
    write_pair(paths, surface, region, layer)
    result = read_pair(*paths, surface, layer)
    for name in ("faces", "loop"):
        np.testing.assert_array_equal(result[name], region[name])


def test_original_zero_area_triangle_round_trip(tmp_path):
    from tractedit_pkg.mesh_io import load_mesh

    surface = Surface(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]],
        [[0, 1, 2], [2, 1, 3]],
    )
    surface.digest = "b" * 64
    loop = np.array([0, 1, 3, 2, 0])
    region = dict(loop=loop, faces=surface.region(loop, 1), digest=surface.digest)
    paths = output_paths(tmp_path / "mixed_area")
    layer = {"coordinate_system": "RAS"}
    write_pair(paths, surface, region, layer)
    with pytest.raises(ValueError, match="zero-area"):
        load_mesh(paths[1])
    restored = read_pair(*paths, surface, layer)
    np.testing.assert_array_equal(restored["faces"], [0, 1])


def test_pair_requires_consistent_reference_space(tmp_path):
    surface = tetrahedron()
    layer = {"coordinate_system": "RAS"}
    region = dict(
        loop=np.array([0, 1, 2, 0]), faces=np.array([0]), digest=surface.digest
    )
    first, second = (output_paths(tmp_path / name) for name in ("first", "second"))
    write_pair(first, surface, dict(region, reference_space="A"), layer)
    write_pair(second, surface, dict(region, reference_space="B"), layer)
    with pytest.raises(ValueError, match="reference spaces differ"):
        read_pair(first[0], second[1], surface, layer)
    assert read_pair(*second, surface, layer)["reference_space"] == "B"


def test_surface_polygon_obj_pair(tmp_path):
    from tractedit_pkg.mesh_io import load_mesh

    surface = tetrahedron()
    surface.uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    surface.corners = surface.faces.copy()
    region = dict(
        faces=np.array([0]), loop=np.array([0, 1, 2, 0]), digest=surface.digest
    )
    paths = output_paths(tmp_path / "example.obj")
    assert [p.name for p in paths] == ["example_polygon.obj", "example_patch.obj"]
    with pytest.raises(ValueError, match="VTP and OBJ"):
        output_paths(tmp_path / "example.stl")
    layer = {"coordinate_system": "LPS"}
    write_obj_pair(paths, surface, region, layer, (1.0, 0.25, 0.0))
    boundary_text = paths[0].read_text(encoding="ascii")
    assert "SPACE=RAS; units=mm" in boundary_text
    assert "\nl 1 2 3 1\n" in boundary_text
    patch_text = paths[1].read_text(encoding="ascii")
    assert "\nvt " in patch_text and "\nf 1/1 2/2 3/3\n" in patch_text
    loaded = load_mesh(paths[1])
    assert loaded["polygon_count"] == 1 and loaded["point_count"] == 3
    resolved = resolve_import_pair([paths[0]])
    assert resolved == paths
    restored, color = read_obj_pair(*resolved, surface, layer, return_color=True)
    for name in ("faces", "loop"):
        np.testing.assert_array_equal(restored[name], region[name])
    assert np.allclose(color, (1.0, 0.25, 0.0))
    for bad in ([tmp_path / "trace.csv"], [paths[0], tmp_path / "example_patch.vtp"]):
        with pytest.raises(ValueError):
            resolve_import_pair(bad)
