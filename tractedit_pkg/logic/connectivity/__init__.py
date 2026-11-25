# -*- coding: utf-8 -*-

"""
Connectivity Manager for TractEdit application.

Handles structural connectivity matrix computation from streamlines
and FreeSurfer parcellation/segmentation volumes.

Features:
- Load FreeSurfer parcellation files (aparc+aseg, etc.)
- Compute endpoint-based connectivity matrices with AOT-compiled optimization
- Export matrices as CSV (with region names) or NPY format + PNG matrix image
- Parse FreeSurfer color LUT for region names
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Optional, Dict, Tuple, List, Any, Iterable

from ...render_batch import batch_render, request_render
from ...file_dialogs import choose_input
from ...utils import signals_blocked
from ...transactional_io import staged_output_set, transactional_save

import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)
from PyQt6.QtCore import Qt
import nibabel as nib

from ...reference_grid import validate_volume_geometry

if TYPE_CHECKING:
    from contextlib import ExitStack

    from ...main_window import MainWindow

logger = logging.getLogger(__name__)


from tractedit_pkg._numba_aot._parallel_wrappers import (
    compute_endpoint_labels as _compute_endpoint_labels,
)
from .freesurfer_lut import (
    FREESURFER_LUT_PATHS, FREESURFER_BUILTIN_LABELS, parse_freesurfer_lut,
)
from .export import _create_connectivity_visualization
from .overlay import region_contour_bounds, region_contour_input, _hue_to_rgb
from .manager import (
    ConnectivityManager, _has_valid_endpoints, _ATLAS_READ_BLOCK_BYTES,
    _read_atlas_c_int32,
)
