# -*- coding: utf-8 -*-
"""Surface mouse tracing and mesh-associated or standalone annotation lifecycle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import vtk
from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, QThread, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QTreeWidgetItem,
)

from ... import file_io
from ...render_batch import request_render
from ...file_dialogs import choose_input
from ..mesh_manager import MeshLoaderThread
from ...mesh.polylines import (
    DEFAULT_POLYLINE_COLOR,
    MAX_POLYLINES,
    MAX_POLYLINE_POINTS,
    POLYLINE_STYLE_FIELDS,
    load_polyline_vtp,
    save_polyline,
    validated_color,
    validated_name,
    validated_style,
    validate_polylines,
)
from ...visualization.actors import create_mesh_polyline_actor, polyline_display_data
from ...utils import MAX_STACK_LEVELS, ROI_COLORS, ui_tree_transaction
from ..state_manager import ActionType

if TYPE_CHECKING:
    from ...data_contracts import MeshLayer, MeshPolyline
    from ...main_window import MainWindow
    from ...mesh.polyline_csv import PolylineCSVData

logger = logging.getLogger(__name__)


from .io_worker import PolylineCSVThread
from .manager import MeshPolylineManager
