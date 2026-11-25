# -*- coding: utf-8 -*-

"""
Drawing package for 2D/3D ROI drawing operations.

Re-exports DrawingManager, rasterization helpers, and module symbols used by
callers and tests.
"""

from __future__ import annotations

import numpy as np
import vtk

from .manager import (
    _PREVIEW_DEPTH_MM,
    _PREVIEW_VIEW_AXES,
    DrawingManager,
    _remove_actor_from_2d_scenes,
    logger,
)
from .rasterizer import (
    DrawingRasterizerMixin,
    _clear_roi_for_shape,
    _display_points_to_voxel,
    _fill_polygon,
    _rasterize_world_sphere,
    _rectangle_grid_geometry,
)

__all__ = [
    "DrawingManager",
    "DrawingRasterizerMixin",
    "_PREVIEW_DEPTH_MM",
    "_PREVIEW_VIEW_AXES",
    "_clear_roi_for_shape",
    "_display_points_to_voxel",
    "_fill_polygon",
    "_rasterize_world_sphere",
    "_rectangle_grid_geometry",
    "_remove_actor_from_2d_scenes",
    "logger",
    "np",
    "vtk",
]
