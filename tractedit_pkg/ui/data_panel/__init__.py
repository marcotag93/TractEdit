# -*- coding: utf-8 -*-

"""
Data panel package for TractEdit UI.

Re-exports DataPanelManager, widget helpers, and Qt menu/dialog symbols for
callers and monkeypatch-based tests.
"""

from __future__ import annotations

from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QInputDialog, QMenu, QMessageBox

from .controller import DataPanelManager, _DataTreeWidget, logger
from .mesh_inspector import _MeshTargetLabel
from .tree import classify_region_hemisphere, image_volume_label

__all__ = [
    "DataPanelManager",
    "QAction",
    "QActionGroup",
    "QInputDialog",
    "QMenu",
    "QMessageBox",
    "_DataTreeWidget",
    "_MeshTargetLabel",
    "classify_region_hemisphere",
    "image_volume_label",
    "logger",
]
