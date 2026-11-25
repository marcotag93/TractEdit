"""Core polyline geometry and file output contracts."""

import numpy as np
import pytest
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

from tractedit_pkg.mesh.polylines import polyline_polydata, save_polyline


POINTS = np.array([[1.125, -2.5, 3.0], [4.0, 5.25, 6.0], [7.0, 8.0, 9.5]])


@pytest.mark.parametrize(
    "extension, reader_type",
    [
        ("vtp", vtkXMLPolyDataReader),
        ("vtk", vtkPolyDataReader),
    ],
)
def test_native_roundtrip_preserves_order_coordinates_and_color(
    tmp_path,
    extension,
    reader_type,
):
    path = tmp_path / ("line." + extension)
    save_polyline(path, POINTS, (1.0, 0.0, 0.0))
    reader = reader_type()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    np.testing.assert_array_equal(vtk_to_numpy(data.GetPoints().GetData()), POINTS)
    assert data.GetNumberOfLines() == 1
    assert data.GetNumberOfPolys() == 0
    np.testing.assert_array_equal(
        vtk_to_numpy(data.GetLines().GetConnectivityArray()),
        [0, 1, 2],
    )
    np.testing.assert_array_equal(
        vtk_to_numpy(data.GetPointData().GetScalars()),
        [[255, 0, 0]] * 3,
    )
    assert data.GetFieldData().GetAbstractArray("SPACE").GetValue(0) == "RAS"


@pytest.mark.parametrize(
    "points",
    [
        [],
        [[1, 2, 3]],
        [[1, 2]],
        [[0, 0, 0], [np.nan, 1, 2]],
        [[0, 0, 0], [0, 0, 0]],
    ],
)
def test_invalid_geometry_is_rejected_before_vtk(points):
    with pytest.raises(ValueError):
        polyline_polydata(points, (1, 0, 0))


def test_export_refuses_to_overwrite_a_source(tmp_path):
    path = tmp_path / "source.vtp"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="source"):
        save_polyline(path, POINTS, (1, 0, 0), protected_paths=[path])
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize("profile", ["xyz", "tractedit"])
def test_csv_roundtrip_preserves_float64_vertices(tmp_path, profile):
    from tractedit_pkg.mesh.polyline_csv import load_polyline_csv

    path = tmp_path / "trace.csv"
    points = np.array([[1.0000000000000002, -2, 3], [4, 5, np.nextafter(6, 7)]])
    save_polyline(path, points, (1, 0, 0), csv_profile=profile, name="Trace")
    result = load_polyline_csv(path, space="RAS")
    np.testing.assert_array_equal(result["points"], points)
    assert result["show_points"] is True


def test_display_markers_preserve_export_topology():
    from tractedit_pkg.visualization.actors import create_mesh_polyline_actor

    line = create_mesh_polyline_actor(
        POINTS, (1, 0, 0), line_width=4.0, point_size=9.0, show_points=True,
    )
    markers = line.polyline_markers
    assert markers.GetVisibility() and not markers.GetPickable()
    assert line.GetProperty().GetLineWidth() == 4.0
    assert markers.GetProperty().GetPointSize() == 9.0
    assert line.GetMapper().GetInput().GetPoints() is (
        markers.GetMapper().GetInput().GetPoints()
    )
    assert polyline_polydata(POINTS, (1, 0, 0)).GetNumberOfVerts() == 0


def test_indexed_csv_and_complete_metadata(tmp_path):
    from tractedit_pkg.mesh.polyline_csv import load_polyline_csv

    path = tmp_path / "indexed.csv"
    path.write_text("index,x,y,z\n42,1,2,3\n43,4,5,6\n", encoding="utf-8")
    np.testing.assert_array_equal(
        load_polyline_csv(path, space="LPS")["points"],
        [[-1, -2, 3], [-4, -5, 6]],
    )
    save_polyline(
        path,
        POINTS,
        (0.1, 0.2, 0.3),
        csv_profile="tractedit",
        name="Trace",
        line_width=4.5,
        point_size=12,
        show_points=False,
        visible=False,
    )
    loaded = load_polyline_csv(path)
    assert loaded["name"] == "Trace" and loaded["color"] == (0.1, 0.2, 0.3)
    assert loaded["line_width"] == 4.5 and loaded["point_size"] == 12
    assert not loaded["show_points"] and not loaded["visible"]


@pytest.mark.parametrize("body", ["0,0,0\n1,,2\n", "0,0,0\nnan,1,2\n"])
def test_csv_invalid_rows_preserve_existing_source(tmp_path, body):
    from tractedit_pkg.mesh.polyline_csv import load_polyline_csv

    path = tmp_path / "source.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="Line 2"):
        load_polyline_csv(path, space="RAS")
    with pytest.raises(ValueError, match="source"):
        save_polyline(path, POINTS, (1, 0, 0), protected_paths=[path])
    assert path.read_text(encoding="utf-8") == body


def test_csv_atomic_write_failure_preserves_destination(tmp_path, monkeypatch):
    from tractedit_pkg import transactional_io

    path = tmp_path / "existing.csv"
    path.write_bytes(b"original")

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(transactional_io.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        save_polyline(path, POINTS, (1, 0, 0))
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("row", ["\t1\t2\t3", "1\t2\t3\t"])
def test_tsv_keeps_empty_outer_columns(tmp_path, row):
    from tractedit_pkg.mesh.polyline_csv import load_polyline_csv

    path = tmp_path / "trace.tsv"
    path.write_text("0\t0\t0\n" + row + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="Line 2"):
        load_polyline_csv(path, delimiter="\t")
