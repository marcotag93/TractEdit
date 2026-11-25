# -*- coding: utf-8 -*-

"""
Pure NumPy rasterization helpers and mixin for 2D/3D ROI drawing.

Keeps voxel rasterization, coordinate projection, and polygon fill logic
isolated from VTK scene management and Qt cursor handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


def _clear_roi_for_shape(roi_data: np.ndarray) -> None:
    """Clear the old positive ROI, preserving legacy behavior for other dtypes."""
    if roi_data.dtype == np.uint8:
        # All nonzero uint8 values are positive. The old bounding box therefore
        # contained every nonzero voxel, so clearing the array is equivalent.
        roi_data.fill(0)
        return

    old_nonzero = np.argwhere(roi_data > 0)
    if len(old_nonzero):
        old_min = old_nonzero.min(axis=0)
        old_max = old_nonzero.max(axis=0)
        roi_data[
            old_min[0] : old_max[0] + 1,
            old_min[1] : old_max[1] + 1,
            old_min[2] : old_max[2] + 1,
        ] = 0


def _rasterize_world_sphere(
    roi_data: np.ndarray,
    center_voxel: np.ndarray,
    radius_world: float,
    affine: np.ndarray,
    shape: Tuple[int, ...],
    fill_value: int,
) -> bool:
    """Rasterize a world-space sphere into an arbitrary voxel grid."""
    linear = affine[:3, :3]
    inverse_linear = np.linalg.inv(linear)
    half_extents = radius_world * np.linalg.norm(inverse_linear, axis=1)
    minimum = np.floor(center_voxel - half_extents).astype(int)
    maximum = np.ceil(center_voxel + half_extents).astype(int)
    minimum = np.maximum(minimum, 0)
    maximum = np.minimum(maximum, np.asarray(shape[:3]) - 1)
    if np.any(minimum > maximum):
        return False

    dx = (
        np.arange(minimum[0], maximum[0] + 1, dtype=float)[:, None, None]
        - center_voxel[0]
    )
    dy = (
        np.arange(minimum[1], maximum[1] + 1, dtype=float)[None, :, None]
        - center_voxel[1]
    )
    dz = (
        np.arange(minimum[2], maximum[2] + 1, dtype=float)[None, None, :]
        - center_voxel[2]
    )
    distance_squared = np.zeros(tuple(maximum - minimum + 1), dtype=float)
    for row in linear:
        world_component = row[0] * dx + row[1] * dy + row[2] * dz
        distance_squared += world_component * world_component

    roi_patch = roi_data[
        minimum[0] : maximum[0] + 1,
        minimum[1] : maximum[1] + 1,
        minimum[2] : maximum[2] + 1,
    ]
    sphere_mask = distance_squared <= radius_world**2
    if not np.any(sphere_mask):
        return False
    roi_patch[sphere_mask] = fill_value
    return True


def _rectangle_grid_geometry(
    voxel_points: np.ndarray,
    affine: np.ndarray,
    shape: Tuple[int, ...],
    view_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return inclusive voxel bounds and world corners for a drawn rectangle."""
    if len(voxel_points) < 2:
        return None

    axis = {"sagittal": 0, "coronal": 1, "axial": 2}.get(view_type)
    if axis is None:
        return None

    first = np.rint(voxel_points[0]).astype(np.int64)
    second = np.rint(voxel_points[1]).astype(np.int64)
    plane_index = int(np.rint((voxel_points[0, axis] + voxel_points[1, axis]) / 2))
    first[axis] = plane_index
    second[axis] = plane_index
    minimum = np.maximum(np.minimum(first, second), 0)
    maximum = np.minimum(np.maximum(first, second), np.asarray(shape[:3]) - 1)
    if np.any(minimum > maximum):
        return None
    plane_index = int(minimum[axis])

    if axis == 2:
        corners_voxel = np.array(
            [
                [minimum[0], minimum[1], plane_index],
                [maximum[0], minimum[1], plane_index],
                [maximum[0], maximum[1], plane_index],
                [minimum[0], maximum[1], plane_index],
            ],
            dtype=float,
        )
    elif axis == 1:
        corners_voxel = np.array(
            [
                [minimum[0], plane_index, minimum[2]],
                [maximum[0], plane_index, minimum[2]],
                [maximum[0], plane_index, maximum[2]],
                [minimum[0], plane_index, maximum[2]],
            ],
            dtype=float,
        )
    else:
        corners_voxel = np.array(
            [
                [plane_index, minimum[1], minimum[2]],
                [plane_index, maximum[1], minimum[2]],
                [plane_index, maximum[1], maximum[2]],
                [plane_index, minimum[1], maximum[2]],
            ],
            dtype=float,
        )

    homogeneous = np.column_stack((corners_voxel, np.ones(4)))
    corners_world = (affine @ homogeneous.T).T[:, :3]
    return minimum, maximum, corners_world


def _display_points_to_voxel(
    points: np.ndarray,
    roi_inv_affine: np.ndarray,
    actor_obj: Any = None,
) -> np.ndarray:
    """Convert 2D-displayed world point(s) to continuous voxel coordinates."""
    points_arr = np.asarray(points, dtype=float)
    single_point = points_arr.ndim == 1
    pts_2d = points_arr.reshape(1, 3) if single_point else points_arr
    homog_points = np.hstack([pts_2d, np.ones((len(pts_2d), 1))])

    if actor_obj is None:
        # Fallback to affine (RAS+ data — inv_affine gives correct voxel indices)
        vox_points = np.dot(roi_inv_affine, homog_points.T).T[:, :3]
    else:
        import vtk

        matrix = actor_obj.GetMatrix()
        inv_matrix = vtk.vtkMatrix4x4()
        inv_matrix.DeepCopy(matrix)
        inv_matrix.Invert()

        mat_np = np.zeros((4, 4))
        for r in range(4):
            for c in range(4):
                mat_np[r, c] = inv_matrix.GetElement(r, c)

        model_points = np.dot(mat_np, homog_points.T).T[:, :3]

        # Get image data for spacing/origin
        mapper = actor_obj.GetMapper()
        image_data = mapper.GetInput()

        if not image_data or not isinstance(image_data, vtk.vtkImageData):
            try:
                conn = mapper.GetInputConnection(0, 0)
                if conn:
                    producer = conn.GetProducer()
                    if hasattr(producer, "GetInputConnection"):
                        input_conn = producer.GetInputConnection(0, 0)
                        if input_conn:
                            image_data = input_conn.GetProducer().GetOutput()
            except (AttributeError, RuntimeError):
                logger.debug("Failed to traverse VTK pipeline for image data.")

        if image_data and isinstance(image_data, vtk.vtkImageData):
            spacing = image_data.GetSpacing()
            origin = image_data.GetOrigin()
            vox_points = (model_points - np.array(origin)) / np.array(spacing)
        else:
            vox_points = model_points

    return vox_points[0] if single_point else vox_points


def _fill_polygon(
    roi_data: np.ndarray,
    vox_points_float: np.ndarray,
    shape: Tuple[int, ...],
    view_type: str,
) -> bool:
    """Fills the interior of the drawn polygon using ray casting."""
    if len(vox_points_float) < 3:
        return False

    # Determine indices (columns) for 2D plane
    const_axis = 2  # Default axial
    plane_axes = (0, 1)

    if view_type == "axial":
        const_axis = 2
        plane_axes = (0, 1)
    elif view_type == "coronal":
        const_axis = 1
        plane_axes = (0, 2)
    elif view_type == "sagittal":
        const_axis = 0
        plane_axes = (1, 2)

    # Get the slice index (average of all points in const axis)
    const_val = int(np.round(np.mean(vox_points_float[:, const_axis])))
    if const_val < 0 or const_val >= shape[const_axis]:
        return False

    # Get 2D polygon vertices
    poly = vox_points_float[:, plane_axes]

    # Bounding Box
    min_x = int(np.floor(np.min(poly[:, 0])))
    max_x = int(np.ceil(np.max(poly[:, 0])))
    min_y = int(np.floor(np.min(poly[:, 1])))
    max_y = int(np.ceil(np.max(poly[:, 1])))

    min_x = max(0, min_x)
    max_x = min(shape[plane_axes[0]] - 1, max_x)
    min_y = max(0, min_y)
    max_y = min(shape[plane_axes[1]] - 1, max_y)

    # Create grid points
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)

    if len(x_range) == 0 or len(y_range) == 0:
        return False

    xx, yy = np.meshgrid(x_range, y_range)
    points = np.vstack((xx.flatten(), yy.flatten())).T

    if len(points) == 0:
        return False

    # Ray Casting Algorithm (Vectorized)
    path = poly
    n_vertices = len(path)
    j = n_vertices - 1

    x = points[:, 0]
    y = points[:, 1]

    mask = np.zeros(len(points), dtype=bool)

    for i in range(n_vertices):
        xi, yi = path[i]
        xj, yj = path[j]

        # Intersection condition
        # 1. Edge must cross the horizontal ray line (yi > y) != (yj > y)
        # 2. Intersection x-coordinate must be less than point x (ray to the right)
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi
        )

        mask ^= intersect
        j = i

    # Reshape mask back to grid shape (rows=y, cols=x)
    mask_grid = mask.reshape(len(y_range), len(x_range))

    # Transpose to match (X, Y) indexing for ROI data assignment
    mask_transposed = mask_grid.T

    # Build slicing tuple
    slices = [slice(None)] * 3
    slices[const_axis] = const_val
    slices[plane_axes[0]] = slice(min_x, max_x + 1)
    slices[plane_axes[1]] = slice(min_y, max_y + 1)

    # Fill
    roi_patch = roi_data[tuple(slices)]
    roi_patch[mask_transposed] = 1

    return True


class DrawingRasterizerMixin:
    """Mixin implementing voxel rasterization methods for DrawingManager."""

    panel: "VTKPanel"

    def _world_to_voxel_points(
        self,
        roi_name: str,
        roi_inv_affine: np.ndarray,
        shape: Tuple[int, ...],
        view_type: str,
    ) -> np.ndarray:
        """Convert world points to voxel coordinates."""
        actor_key = f"{view_type}_2d"
        roi_actors = self.panel.roi_slice_actors.get(roi_name)
        actor_obj = roi_actors[actor_key] if roi_actors and actor_key in roi_actors else None
        points_world = np.array(self.panel.drawing_preview_points)
        return _display_points_to_voxel(points_world, roi_inv_affine, actor_obj)

    def _rasterize_sphere(
        self,
        roi_name: str,
        roi_data: np.ndarray,
        vox_points_float: np.ndarray,
        shape: Tuple[int, ...],
        view_type: str,
    ) -> bool:
        """Rasterize a sphere ROI."""
        if len(vox_points_float) < 2:
            return False

        center_vox = vox_points_float[0]
        radius_world = np.linalg.norm(
            self.panel.drawing_preview_points[0] - self.panel.drawing_preview_points[1]
        )

        # Remove any existing 3D rectangle actor
        if roi_name in self.panel.roi_slice_actors:
            old_rect = self.panel.roi_slice_actors[roi_name].get("rectangle_3d")
            if old_rect:
                self.panel.scene.rm(old_rect)
                self.panel.roi_slice_actors[roi_name]["rectangle_3d"] = None
            old_pts = self.panel.roi_slice_actors[roi_name].get("rectangle_points")
            if old_pts is not None:
                self.panel.roi_slice_actors[roi_name]["rectangle_points"] = None

        # Clear rectangle params
        self.panel.rectangle_params_per_roi.pop(roi_name, None)

        _clear_roi_for_shape(roi_data)

        roi_affine = self.panel.main_window.roi_layers[roi_name]["affine"]
        _rasterize_world_sphere(
            roi_data,
            center_vox,
            radius_world,
            roi_affine,
            shape,
            1,
        )

        center_to_store = self.panel.drawing_preview_points[0].copy()
        if view_type in ["axial", "coronal"]:
            center_to_store[0] = -center_to_store[0]

        self.panel.sphere_params_per_roi[roi_name] = {
            "center": center_to_store,
            "radius": radius_world,
            "roi_name": roi_name,
            "view_type": view_type,
        }

        return True

    def _rasterize_rectangle(
        self,
        roi_name: str,
        roi_data: np.ndarray,
        vox_points_float: np.ndarray,
        shape: Tuple[int, ...],
        view_type: str,
    ) -> bool:
        """Rasterize a rectangle ROI."""
        # Remove any existing 3D sphere actor and source
        if roi_name in self.panel.roi_slice_actors:
            old_sphere = self.panel.roi_slice_actors[roi_name].get("sphere_3d")
            if old_sphere:
                self.panel.scene.rm(old_sphere)
                self.panel.roi_slice_actors[roi_name]["sphere_3d"] = None
            old_src = self.panel.roi_slice_actors[roi_name].get("sphere_source")
            if old_src is not None:
                self.panel.roi_slice_actors[roi_name]["sphere_source"] = None

        # Clear sphere params
        self.panel.sphere_params_per_roi.pop(roi_name, None)

        if len(vox_points_float) < 2:
            return False

        roi_affine = self.panel.main_window.roi_layers[roi_name]["affine"]
        geometry = _rectangle_grid_geometry(
            vox_points_float, roi_affine, shape, view_type
        )
        if geometry is None:
            return False
        min_v, max_v, corners_world = geometry

        _clear_roi_for_shape(roi_data)

        roi_data[
            min_v[0] : max_v[0] + 1,
            min_v[1] : max_v[1] + 1,
            min_v[2] : max_v[2] + 1,
        ] = 1

        # Store params with 3D-corrected coordinates
        start_to_store = self.panel.drawing_preview_points[0].copy()
        end_to_store = self.panel.drawing_preview_points[1].copy()
        if view_type in ["axial", "coronal"]:
            start_to_store[0] = -start_to_store[0]
            end_to_store[0] = -end_to_store[0]

        self.panel.rectangle_params_per_roi[roi_name] = {
            "start": start_to_store,
            "end": end_to_store,
            "view_type": view_type,
            "voxel_min": min_v.copy(),
            "voxel_max": max_v.copy(),
            "corners": corners_world,
        }

        return True

    def _rasterize_freehand(
        self,
        roi_data: np.ndarray,
        vox_points_float: np.ndarray,
        shape: Tuple[int, ...],
        view_type: str,
    ) -> bool:
        """Rasterize freehand drawing (pencil/eraser)."""
        all_indices = []

        if len(vox_points_float) == 1:
            all_indices.append(vox_points_float)
        else:
            for i in range(len(vox_points_float) - 1):
                p0 = vox_points_float[i]
                p1 = vox_points_float[i + 1]
                dist = np.linalg.norm(p1 - p0)
                n_steps = int(np.ceil(dist * 10))

                MAX_STEPS = 100000
                if n_steps > MAX_STEPS:
                    logger.debug(
                        f"Ignoring draw stroke with excessive distance: {dist}"
                    )
                    continue

                if n_steps > 0:
                    try:
                        segment_points = np.linspace(p0, p1, num=n_steps + 1)
                        all_indices.append(segment_points)
                    except (ValueError, MemoryError) as e:
                        logger.debug(f"Skipping invalid segment: {e}")
                        continue

        if not all_indices:
            return False

        full_path = np.vstack(all_indices)
        indices = np.round(full_path).astype(int)

        valid_mask = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < shape[1])
            & (indices[:, 2] >= 0)
            & (indices[:, 2] < shape[2])
        )
        valid_indices = indices[valid_mask]

        if len(valid_indices) == 0:
            return False

        # Apply brush size
        brush_size = getattr(self.panel.main_window, "draw_brush_size", 1)
        if brush_size > 1:
            valid_indices = self._expand_brush(
                valid_indices, shape, view_type, brush_size
            )

        if len(valid_indices) == 0:
            return False

        # Get current values
        current_values = roi_data[
            valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
        ]

        is_eraser = getattr(self.panel, "is_eraser_mode", False)

        if is_eraser:
            if np.any(current_values == 1):
                roi_data[
                    valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
                ] = 0
                return True
        else:
            if np.any(current_values == 0):
                roi_data[
                    valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
                ] = 1
                return True

        return False

    def _expand_brush(
        self,
        valid_indices: np.ndarray,
        shape: Tuple[int, ...],
        view_type: str,
        brush_size: int,
    ) -> np.ndarray:
        """Expand indices based on brush size."""
        lower = -((brush_size - 1) // 2)
        upper = (brush_size // 2) + 1
        r_full = range(lower, upper)
        r_flat = range(0, 1)

        rx, ry, rz = r_full, r_full, r_full

        if view_type == "axial":
            rz = r_flat
        elif view_type == "coronal":
            ry = r_flat
        elif view_type == "sagittal":
            rx = r_flat

        offsets = []
        for dx in rx:
            for dy in ry:
                for dz in rz:
                    offsets.append([dx, dy, dz])
        offsets = np.array(offsets)

        expanded_indices = valid_indices[:, np.newaxis, :] + offsets[np.newaxis, :, :]
        valid_indices = expanded_indices.reshape(-1, 3)

        valid_mask = (
            (valid_indices[:, 0] >= 0)
            & (valid_indices[:, 0] < shape[0])
            & (valid_indices[:, 1] >= 0)
            & (valid_indices[:, 1] < shape[1])
            & (valid_indices[:, 2] >= 0)
            & (valid_indices[:, 2] < shape[2])
        )
        return valid_indices[valid_mask]

    def _rasterize_sphere_at_position(
        self,
        roi_name: str,
        roi_data: np.ndarray,
        center_w: np.ndarray,
        radius_w: float,
        shape: Tuple[int, ...],
        view_type: str,
        fill_val: int,
    ) -> None:
        """Rasterize a sphere at the given world position."""
        roi_layer = self.panel.main_window.roi_layers[roi_name]
        roi_inv_affine = roi_layer["inv_affine"]

        actor_key = f"{view_type}_2d"
        roi_actors = self.panel.roi_slice_actors.get(roi_name)
        actor_obj = roi_actors[actor_key] if roi_actors and actor_key in roi_actors else None

        center_vox = _display_points_to_voxel(center_w, roi_inv_affine, actor_obj)

        _rasterize_world_sphere(
            roi_data,
            center_vox,
            radius_w,
            roi_layer["affine"],
            shape,
            fill_val,
        )

    def _fill_polygon(
        self,
        roi_name: str,
        roi_data: np.ndarray,
        vox_points_float: np.ndarray,
        shape: Tuple[int, ...],
        view_type: str,
    ) -> bool:
        """Fills the interior of the drawn polygon using ray casting."""
        return _fill_polygon(roi_data, vox_points_float, shape, view_type)
