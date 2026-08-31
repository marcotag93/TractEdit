# -*- coding: utf-8 -*-

"""
Coordinate transformation utilities for TractEdit visualization.

Provides helper functions for converting between voxel indices and
world (RASmm) coordinates using affine transformation matrices.
"""

# ============================================================================
# Imports
# ============================================================================

from typing import List, Optional, Tuple
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


def camera_frame_from_affine(
    affine: np.ndarray,
    slice_axis: int,
    view_sign: int,
    view_up_axis: int,
    radiological: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return an upright camera direction and view-up vector for a voxel plane."""
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("Camera affine must be a finite 4x4 matrix.")
    if slice_axis not in (0, 1, 2) or view_up_axis not in (0, 1, 2):
        raise ValueError("Camera axes must be voxel axes 0, 1, or 2.")
    if slice_axis == view_up_axis or view_sign not in (-1, 1):
        raise ValueError("Camera frame axes or view sign are invalid.")

    displayed_affine = affine.copy()
    if radiological:
        displayed_affine = np.diag([-1.0, 1.0, 1.0, 1.0]) @ displayed_affine

    linear = displayed_affine[:3, :3]
    if np.linalg.matrix_rank(linear) < 3:
        raise ValueError("Camera affine must have three independent spatial axes.")

    in_plane_axes = [axis for axis in range(3) if axis != slice_axis]
    direction = np.cross(
        linear[:, in_plane_axes[0]], linear[:, in_plane_axes[1]]
    )
    preferred_direction = view_sign * linear[:, slice_axis]
    if np.dot(direction, preferred_direction) < 0:
        direction *= -1.0
    direction /= np.linalg.norm(direction)

    view_up = linear[:, view_up_axis]
    view_up = view_up - direction * np.dot(view_up, direction)
    view_up_norm = np.linalg.norm(view_up)
    if view_up_norm <= np.finfo(np.float64).eps:
        raise ValueError("Camera view-up axis is degenerate for this affine.")
    view_up /= view_up_norm
    return direction, view_up


def translate_camera_to_plane(
    position: np.ndarray,
    focal_point: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate a camera onto a slice plane while preserving pan and distance."""
    position = np.asarray(position, dtype=np.float64)
    focal_point = np.asarray(focal_point, dtype=np.float64)
    plane_point = np.asarray(plane_point, dtype=np.float64)
    plane_normal = np.asarray(plane_normal, dtype=np.float64)
    normal_length = np.linalg.norm(plane_normal)
    if normal_length <= np.finfo(np.float64).eps:
        raise ValueError("Slice plane normal must be non-zero.")

    plane_normal = plane_normal / normal_length
    offset = np.dot(plane_point - focal_point, plane_normal) * plane_normal
    return position + offset, focal_point + offset
