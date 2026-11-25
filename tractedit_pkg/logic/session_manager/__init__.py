# -*- coding: utf-8 -*-
"""Session orchestration with isolated scene preparation and atomic publication."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, NotRequired, TypedDict

import numpy as np
from PyQt6.QtCore import QByteArray, QEventLoop, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QSplitter,
)

from ...workers import CancellableWorkerThread, create_progress_dialog, run_modal_worker
from ...file_dialogs import choose_input
from ...render_batch import batch_render, defer_render, request_render
from ..connectivity import region_contour_bounds, region_contour_input
from ...mesh.presentation import MESH_TREE_COMPONENTS
from ...session.io import (
    SessionCancelled,
    SessionError,
    checkpoint,
    checked_fingerprint,
    source_stamp,
    resolve_session_sources,
    SourceResolutionError,
    read_session,
    timestamped_path,
    write_session,
    validate_sources,
)
from ...session.state import (
    SOURCE_FIELDS,
    STATE_FIELDS,
    associate_sources,
    capture_state,
    dispose_sources,
    load_source,
    validate_state,
)
from ...session.mesh import (
    MAX_MESH_LAYERS,
    _valid_flag_tuple,
    restored_mesh_layer,
    validate_mesh_document,
)
from ...session.mesh_texture import texture_asset_role, texture_references

logger = logging.getLogger(__name__)

from .views import (
    CAMERA_PROPERTIES,
    SCENES,
    SLICE_ACTORS,
    SessionViews,
    _presentation,
    _tree_items,
    capture_views,
    restore_cameras,
    restore_mesh_tree,
    validate_views,
)
from .restore import (
    _build_region_contour,
    _update_odf_tunnel_actor,
    deferred_odf_owner,
    materialize_session_odf,
    materialize_session_regions,
    populate_window,
    prepare_session,
)
from .manager import (
    OPEN_FILE_FILTER,
    SAVE_FILE_FILTER,
    SessionManager,
    SessionWorker,
)
