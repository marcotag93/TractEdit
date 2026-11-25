# -*- coding: utf-8 -*-
"""Asynchronous surface loading and Qt-thread publication of mesh layers."""

import logging
import os
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Unpack

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog, QDialog, QFileDialog, QInputDialog, QMessageBox, QProgressDialog,
)

from ...workers import CancellableWorkerThread
from ...render_batch import request_render
from ...file_dialogs import choose_input
from ... import file_io
from ...data_contracts import (
    MeshCoordinateChoice, MeshLayer, MeshLoadResult, MeshTexturePreparation,
    MeshComponent, MeshComponentStyle,
)
from ...mesh_io import MeshLoadCancelled, load_mesh
from ...mesh.model import (
    MAX_MESH_LAYERS,
    create_mesh_layer, find_mesh_layer, mesh_subtree_postorder,
    validate_mesh_forest, validate_mesh_parent,
)
from ...mesh.presentation import (
    validate_mesh_boolean,
    validate_mesh_component,
    validate_mesh_component_style,
)
from ...ui.mesh_import_dialog import MeshImportDialog
from ...utils import iter_tree_items as _tree_items, signals_blocked, ui_tree_transaction

if TYPE_CHECKING:
    from ...main_window import MainWindow
    from ...mesh_texture import MeshTextureBudgets, MeshTextureCache

logger = logging.getLogger(__name__)


from .overlay_controller import MeshLoaderThread, MeshOverlayLoaderThread
from .texture_controller import MeshTextureLoaderThread
from .manager import MeshManager, _tree_item_key
