"""Annotation groups in the compact scene tree."""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QTreeWidget

from tractedit_pkg.logic.mesh_polyline_manager import MeshPolylineManager


@pytest.mark.parametrize(
    "owner,label", [(None, "Standalone polylines"), ("mesh", "Polylines")]
)
def test_polyline_group_appears_only_for_populated_owner(owner, label):
    app = QApplication.instance() or QApplication([])
    tree = QTreeWidget()
    records = []
    manager = SimpleNamespace(container=lambda mesh_id: records)
    MeshPolylineManager.build_tree(manager, tree, owner)
    assert tree.topLevelItemCount() == 0

    records.append({
        "id": "trace", "name": "Sulcus", "color": (1, 0, 0),
        "visible": True, "points": [(0, 0, 0), (1, 0, 0)],
    })
    MeshPolylineManager.build_tree(manager, tree, owner)
    group = tree.topLevelItem(0)
    assert group.text(0) == f"{label} (1)"
    assert not group.isExpanded()
    trace = group.child(0)
    assert trace.text(0) == "Sulcus"
    assert trace.checkState(0) == Qt.CheckState.Checked
    assert not trace.icon(0).isNull()
    tree.close()
    assert app is QApplication.instance()


def test_mesh_tree_separates_polylines_and_surface_polygons():
    app = QApplication.instance() or QApplication([])
    tree = QTreeWidget()
    records = [
        {
            "id": "trace", "name": "Sulcus", "color": (1, 0, 0),
            "visible": True, "points": [(0, 0, 0), (1, 0, 0)],
        },
        {
            "id": "poly", "name": "Cortex", "color": (0, 1, 0),
            "visible": True, "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)],
            "region": {"faces": [0], "loop": [0, 1, 2, 0], "digest": "a" * 64},
        },
    ]
    manager = SimpleNamespace(container=lambda mesh_id: records)
    MeshPolylineManager.build_tree(manager, tree, "mesh")
    assert tree.topLevelItemCount() == 2
    lines_group, polys_group = tree.topLevelItem(0), tree.topLevelItem(1)
    assert lines_group.text(0) == "Polylines (1)"
    assert lines_group.data(0, Qt.ItemDataRole.UserRole)["type"] == "mesh_polylines"
    assert lines_group.child(0).data(0, Qt.ItemDataRole.UserRole)["type"] == "mesh_polyline"
    assert polys_group.text(0) == "Surface polygons (1)"
    assert polys_group.data(0, Qt.ItemDataRole.UserRole)["type"] == "mesh_polygons"
    poly_item = polys_group.child(0)
    assert poly_item.text(0) == "Cortex"
    assert poly_item.data(0, Qt.ItemDataRole.UserRole) == {
        "type": "mesh_polygon", "id": "mesh", "component": "poly",
    }
    assert not poly_item.icon(0).isNull()

    tree.clear()
    records.pop(0)
    MeshPolylineManager.build_tree(manager, tree, "mesh")
    assert tree.topLevelItemCount() == 1
    assert tree.topLevelItem(0).text(0) == "Surface polygons (1)"
    tree.close()
    assert app is QApplication.instance()
