# -*- coding: utf-8 -*-

"""
Coordinate transformation utilities for TractEdit visualization.

Provides helper functions for converting between voxel indices and
world (RASmm) coordinates using affine transformation matrices.
"""

# ============================================================================
# Imports
# ============================================================================

from typing import List, Optional
import numpy as np


# ============================================================================
# Coordinate Transformation Functions
# ============================================================================


def voxel_to_world(vox_coord: List[float], affine: np.ndarray) -> np.ndarray:
    """
    Converts a voxel index [i, j, k] to world RASmm coordinates [x, y, z].

    Args:
        vox_coord: Voxel coordinates as [i, j, k].
        affine: 4x4 affine transformation matrix.

    Returns:
        World coordinates as numpy array [x, y, z].
    """
    homog_vox = np.array([vox_coord[0], vox_coord[1], vox_coord[2], 1.0])
    world_coord = np.dot(affine, homog_vox)
    return world_coord[:3]


def world_to_voxel(world_coord: List[float], inv_affine: np.ndarray) -> np.ndarray:
    """
    Converts world RASmm coordinates [x, y, z] to voxel indices [i, j, k].

    Args:
        world_coord: World coordinates as [x, y, z].
        inv_affine: Inverse of the 4x4 affine transformation matrix.

    Returns:
        Voxel coordinates as numpy array [i, j, k] (float values).
    """
    homog_world = np.array([world_coord[0], world_coord[1], world_coord[2], 1.0])
    vox_coord = np.dot(inv_affine, homog_world)
    return vox_coord[:3]  # Return float voxel coords
