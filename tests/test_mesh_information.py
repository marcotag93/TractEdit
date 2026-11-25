# -*- coding: utf-8 -*-
"""Mesh information rows tolerate removal before the tree is rebuilt."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidgetItem

from tractedit_pkg.ui.data_panel import DataPanelManager


def information_panel(layers):
    panel = DataPanelManager.__new__(DataPanelManager)
    panel.mw = SimpleNamespace(
        mesh_layers=layers, mesh_manager=Mock(), vtk_panel=Mock()
    )
    panel.mw.mesh_manager.texture_control_state.return_value = (
        False,
        False,
        "Texture",
        "details",
    )
    item = QTreeWidgetItem(["Texture: previous"])
    item.setData(
        0,
        Qt.ItemDataRole.UserRole,
        {
            "type": "mesh_information",
            "id": "mesh",
            "component": "information_texture",
        },
    )
    return panel, item


def test_removed_mesh_information_skips_texture_queries():
    panel, item = information_panel({})
    panel._sync_mesh_information(item)
    assert item.text(0) == "Texture: previous"
    panel.mw.mesh_manager.texture_control_state.assert_not_called()
    panel.mw.mesh_manager.texture_status.assert_not_called()


@pytest.mark.parametrize(
    "path,status,expected",
    [
        ("surface.vtp", "unloaded", "Unavailable"),
        ("surface.obj", "unloaded", "OFF (not examined)"),
        ("surface.obj", "loading", "Loading..."),
        ("surface.obj", "unavailable", "Unavailable"),
    ],
)
def test_live_mesh_information_still_updates(path, status, expected):
    panel, item = information_panel({"mesh": {"path": path}})
    panel.mw.mesh_manager.texture_status.return_value = status
    panel._sync_mesh_information(item)
    assert item.text(0) == "Texture: " + expected
    assert item.toolTip(0) == "details"
