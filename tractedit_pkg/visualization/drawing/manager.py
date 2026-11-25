# -*- coding: utf-8 -*-

"""
Drawing manager for ROI drawing operations on 2D views.

Handles pencil, eraser, sphere, and rectangle drawing modes,
preview visualization, and 3D shape actors.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import vtk
from fury import window
from PyQt6.QtCore import Qt

from ...render_batch import batch_render, request_render
from ...utils import signals_blocked
from .. import coordinates
from .rasterizer import DrawingRasterizerMixin

if TYPE_CHECKING:
    from ..vtk_panel import VTKPanel

logger = logging.getLogger(__name__)

_PREVIEW_VIEW_AXES = {
    "axial": (2, 1, 1, True),
    "coronal": (1, -1, 2, True),
    "sagittal": (0, 1, 2, False),
}
_PREVIEW_DEPTH_MM = 0.25


def _remove_actor_from_2d_scenes(panel: "VTKPanel", actor_obj: Any) -> None:
    """Remove an actor from all active 2D slice scenes."""
    if actor_obj is None:
        return
    for scene in (panel.axial_scene, panel.coronal_scene, panel.sagittal_scene):
        if scene:
            try:
                scene.rm(actor_obj)
            except (ValueError, RuntimeError):
                logger.debug("Failed to remove preview actor from scene.")


# ============================================================================
# Drawing Manager Class
# ============================================================================


class DrawingManager(DrawingRasterizerMixin):
    """
    Manages ROI drawing operations on 2D views.

    Handles drawing modes (pencil, eraser, sphere, rectangle),
    preview visualization, and rasterization to volume data.
    """

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        """
        Initialize the DrawingManager.

        Args:
            vtk_panel: Reference to the parent VTKPanel instance.
        """
        self.panel = vtk_panel
        # Counter for throttling sphere/rectangle intersection updates
        self._sphere_update_counter: int = 0
        self._rectangle_update_counter: int = 0

    def set_drawing_mode(
        self,
        enabled: bool,
        is_eraser: bool = False,
        is_sphere: bool = False,
        is_rectangle: bool = False,
    ) -> None:
        """Enables or disables drawing mode for 2D panels."""
        self.panel.is_drawing_mode = enabled
        self.panel.is_eraser_mode = is_eraser
        self.panel.is_sphere_mode = is_sphere
        self.panel.is_rectangle_mode = is_rectangle

        # Update cursor style for 2D views (visual feedback)
        if enabled:
            cursor = Qt.CursorShape.CrossCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor

        if self.panel.axial_vtk_widget:
            self.panel.axial_vtk_widget.setCursor(cursor)
        if self.panel.coronal_vtk_widget:
            self.panel.coronal_vtk_widget.setCursor(cursor)
        if self.panel.sagittal_vtk_widget:
            self.panel.sagittal_vtk_widget.setCursor(cursor)

    def handle_draw_on_2d(self, interactor: vtk.vtkRenderWindowInteractor) -> None:
        """
        Handles drawing on 2D views. Collects points for the preview path.
        Actual rasterization happens in finish_drawing.
        """
        if not self.panel.main_window or not self.panel.main_window.current_drawing_roi:
            return

        roi_name = self.panel.main_window.current_drawing_roi
        if roi_name not in self.panel.main_window.roi_layers:
            return

        try:
            # Determine which 2D view was clicked
            active_scene = None
            view_type = ""

            if interactor == self.panel.axial_interactor:
                view_type = "axial"
                active_scene = self.panel.axial_scene
            elif interactor == self.panel.coronal_interactor:
                view_type = "coronal"
                active_scene = self.panel.coronal_scene
            elif interactor == self.panel.sagittal_interactor:
                view_type = "sagittal"
                active_scene = self.panel.sagittal_scene
            else:
                return

            # Store the view type for use in finish_drawing
            self.panel.current_drawing_view_type = view_type

            # Get click position and pick world coordinates
            display_pos = interactor.GetEventPosition()
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(display_pos[0], display_pos[1], 0, active_scene)
            world_pos = np.array(picker.GetPickPosition())

            # Ensure ROI actors exist
            if roi_name not in self.panel.roi_slice_actors:
                roi_data = self.panel.main_window.roi_layers[roi_name]["data"]
                roi_affine = self.panel.main_window.roi_layers[roi_name]["affine"]
                self.panel.add_roi_layer(roi_name, roi_data, roi_affine)
                self.panel.main_window.roi_visibility[roi_name] = True

            # Handle different drawing modes
            if getattr(self.panel, "is_sphere_mode", False):
                self._handle_sphere_mode(interactor, roi_name, world_pos, view_type)
            elif getattr(self.panel, "is_rectangle_mode", False):
                self._handle_rectangle_mode(interactor, roi_name, world_pos, view_type)
            else:
                # Normal drawing: append points
                self.panel.drawing_preview_points.append(world_pos)

            sphere_preview = (
                getattr(self.panel, "is_sphere_mode", False)
                and len(self.panel.drawing_preview_points) >= 2
            )
            if not (sphere_preview and getattr(self.panel, "is_moving_sphere", False)):
                self._update_drawing_preview(active_scene)

            if sphere_preview:
                self._update_sphere_realtime(roi_name, view_type)

        except (ValueError, IndexError, RuntimeError, AttributeError) as e:
            logger.error(f"Error during drawing: {e}", exc_info=True)

    def _handle_sphere_mode(
        self,
        interactor: vtk.vtkRenderWindowInteractor,
        roi_name: str,
        world_pos: np.ndarray,
        view_type: str,
    ) -> None:
        """Handle sphere drawing/moving mode."""
        is_ctrl = interactor.GetControlKey()
        roi_params = self.panel.sphere_params_per_roi.get(roi_name)

        if is_ctrl and roi_params:
            # Check if interactor has a preview radius from Ctrl+Scroll
            interactor_style = interactor.GetInteractorStyle()
            if (
                hasattr(interactor_style, "_is_adjusting_radius")
                and interactor_style._is_adjusting_radius
                and interactor_style._preview_radius > 0
            ):
                radius = interactor_style._preview_radius
            else:
                radius = roi_params["radius"]

            self.panel.is_moving_sphere = True

            if not self.panel.drawing_preview_points:
                self.panel.drawing_preview_points.append(world_pos)
                edge_pos = world_pos + np.array([radius, 0, 0])
                self.panel.drawing_preview_points.append(edge_pos)
            else:
                self.panel.drawing_preview_points[0] = world_pos
                edge_pos = world_pos + np.array([radius, 0, 0])
                if len(self.panel.drawing_preview_points) > 1:
                    self.panel.drawing_preview_points[1] = edge_pos
                else:
                    self.panel.drawing_preview_points.append(edge_pos)
        else:
            # Normal Create Mode
            self.panel.is_moving_sphere = False
            if not self.panel.drawing_preview_points:
                self.panel.drawing_preview_points.append(world_pos)
            else:
                if len(self.panel.drawing_preview_points) > 1:
                    self.panel.drawing_preview_points[1] = world_pos
                else:
                    self.panel.drawing_preview_points.append(world_pos)

    def _handle_rectangle_mode(
        self,
        interactor: vtk.vtkRenderWindowInteractor,
        roi_name: str,
        world_pos: np.ndarray,
        view_type: str,
    ) -> None:
        """Handle rectangle drawing/moving mode."""
        # Remove any existing 3D sphere actor for this ROI
        if roi_name in self.panel.roi_slice_actors:
            old_sphere = self.panel.roi_slice_actors[roi_name].get("sphere_3d")
            if old_sphere:
                self.panel.scene.rm(old_sphere)
                self.panel.roi_slice_actors[roi_name]["sphere_3d"] = None
                request_render(self.panel, self.panel.render_window)

        is_ctrl = interactor.GetControlKey()
        rect_params = self.panel.rectangle_params_per_roi.get(roi_name)
        stored_view_type = (
            rect_params.get("view_type", "sagittal") if rect_params else "sagittal"
        )

        if is_ctrl and rect_params:
            # Move Mode - only show yellow preview, apply on Ctrl release
            self.panel.is_moving_rectangle = True

            start_stored = np.array(rect_params["start"]).copy()
            end_stored = np.array(rect_params["end"]).copy()

            # The dimensions are the same regardless of X correction
            dimensions = end_stored - start_stored

            # Calculate new position based on 2D click
            new_center = world_pos

            if view_type in ["axial", "coronal"]:
                # The stored X is negated, so negate the X dimension
                dimensions[0] = -dimensions[0]

            new_start = new_center - (dimensions / 2.0)
            new_end = new_center + (dimensions / 2.0)

            if not self.panel.drawing_preview_points:
                self.panel.drawing_preview_points.append(new_start)
                self.panel.drawing_preview_points.append(new_end)
            else:
                self.panel.drawing_preview_points[0] = new_start
                if len(self.panel.drawing_preview_points) > 1:
                    self.panel.drawing_preview_points[1] = new_end
                else:
                    self.panel.drawing_preview_points.append(new_end)

        else:
            # Normal Create Mode
            self.panel.is_moving_rectangle = False
            if not self.panel.drawing_preview_points:
                self.panel.drawing_preview_points.append(world_pos)
            else:
                if len(self.panel.drawing_preview_points) > 1:
                    self.panel.drawing_preview_points[1] = world_pos
                else:
                    self.panel.drawing_preview_points.append(world_pos)

    def _update_sphere_realtime(self, roi_name: str, view_type: str) -> None:
        """
        Update sphere visuals in real-time during drag.

        When Ctrl is held (move/resize mode), only yellow circle preview is shown.
        When creating new sphere (no Ctrl), 3D sphere preview is shown.
        Intersection calculation is throttled for smooth performance.
        """
        center = self.panel.drawing_preview_points[0]
        edge = self.panel.drawing_preview_points[1]
        radius = np.linalg.norm(center - edge)

        # Apply X-axis correction for axial/coronal views
        center_corrected = center.copy()
        if view_type in ["axial", "coronal"]:
            center_corrected[0] = -center_corrected[0]

        # Check if we're in Ctrl mode (move/resize preview)
        is_ctrl_mode = getattr(self.panel, "is_moving_sphere", False)

        if is_ctrl_mode:
            # Use yellow circle preview only
            # Get the appropriate 2D scene
            if view_type == "axial":
                scene = self.panel.axial_scene
            elif view_type == "coronal":
                scene = self.panel.coronal_scene
            else:
                scene = self.panel.sagittal_scene

            if scene:
                self._show_radius_preview(center, radius, view_type, scene)

            # Don't do intersection updates during preview
            return
        else:
            # Normal create mode
            self.update_3d_sphere_visuals(roi_name, center_corrected, radius)

        # For large tractograms, defer intersection calculation to mouse release
        LARGE_TRACTOGRAM_THRESHOLD = 100000
        mw = self.panel.main_window
        if (
            mw
            and mw.tractogram_data
            and len(mw.tractogram_data) > LARGE_TRACTOGRAM_THRESHOLD
        ):
            return

        # Throttled intersection update
        # ##TODO to refactor
        self._sphere_update_counter += 1
        if self._sphere_update_counter >= 15:
            self._sphere_update_counter = 0
            roi_mgr = getattr(mw, "roi_manager", None) if mw else None
            if roi_mgr is not None:
                roi_mgr.update_sphere_roi_intersection(
                    roi_name, center_corrected, radius
                )
            elif mw and hasattr(mw, "update_sphere_roi_intersection"):
                mw.update_sphere_roi_intersection(roi_name, center_corrected, radius)

    def _update_drawing_preview(self, scene: window.Scene) -> None:
        """Updates the preview line/shape on the active 2D scene."""
        if len(self.panel.drawing_preview_points) < 2:
            return

        try:
            is_sphere = getattr(self.panel, "is_sphere_mode", False)
            is_rectangle = getattr(self.panel, "is_rectangle_mode", False)

            if is_sphere:
                polydata = self._create_circle_preview()
            elif is_rectangle:
                polydata = self._create_rectangle_preview()
            else:
                polydata = self._create_line_preview()

            # Remove old preview from ALL scenes
            if self.panel.preview_line_actor:
                _remove_actor_from_2d_scenes(self.panel, self.panel.preview_line_actor)
                self.panel.preview_line_actor = None

            # Create new preview actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            self.panel.preview_line_actor = vtk.vtkActor()
            self.panel.preview_line_actor.SetMapper(mapper)

            # Style: Yellow, thick, always on top
            prop = self.panel.preview_line_actor.GetProperty()
            prop.SetColor(1.0, 1.0, 0.0)

            brush_size = getattr(self.panel.main_window, "draw_brush_size", 1)
            line_width = max(3, brush_size * 2)
            prop.SetLineWidth(line_width)
            prop.SetLighting(False)
            prop.SetOpacity(1.0)

            if scene:
                scene.add(self.panel.preview_line_actor)
                scene.GetRenderWindow().Render()

        except (ValueError, IndexError, RuntimeError, AttributeError) as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)

    def _create_circle_preview(self) -> vtk.vtkPolyData:
        """Create a circle preview for sphere mode."""
        center = self.panel.drawing_preview_points[0]
        edge = self.panel.drawing_preview_points[1]
        radius = np.linalg.norm(center - edge)

        view_type = getattr(self.panel, "current_drawing_view_type", "axial")
        return self._circle_preview_polydata(center, radius, view_type)

    def _preview_slice_frame(
        self, view_type: str, reference_point: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Return the displayed voxel grid, camera direction and active plane."""
        axis, sign, up, radiological = _PREVIEW_VIEW_AXES[view_type]
        affine = np.asarray(self.panel.main_window.anatomical_image_affine, dtype=float)
        displayed_affine = affine.copy()
        if radiological:
            displayed_affine = np.diag([-1.0, 1.0, 1.0, 1.0]) @ displayed_affine
        direction, _ = coordinates.camera_frame_from_affine(
            affine, axis, sign, up, radiological=radiological
        )
        indices = getattr(self.panel, "current_slice_indices", {})
        slice_index = indices.get("xyz"[axis]) if indices else None
        if slice_index is None:
            voxel_point = np.linalg.inv(displayed_affine) @ np.append(
                reference_point, 1.0
            )
            slice_index = voxel_point[axis]
        plane_voxel = np.zeros(4, dtype=float)
        plane_voxel[axis] = slice_index
        plane_voxel[3] = 1.0
        plane_point = (displayed_affine @ plane_voxel)[:3]
        return displayed_affine, direction, plane_point, axis

    def _circle_preview_polydata(
        self, center: np.ndarray, radius: float, view_type: str
    ) -> vtk.vtkPolyData:
        """Place a world-mm circle parallel to and in front of the 2D slice."""
        circle = vtk.vtkRegularPolygonSource()
        self._set_preview_circle(circle, center, radius, view_type)
        circle.Update()
        return circle.GetOutput()

    def _set_preview_circle(self, circle, center, radius, view_type):
        _, direction, plane_point, _ = self._preview_slice_frame(view_type, center)
        center = np.asarray(center, dtype=float)
        center = center - np.dot(center - plane_point, direction) * direction
        center += _PREVIEW_DEPTH_MM * direction

        circle.SetCenter(*center)
        circle.SetRadius(float(radius))
        circle.SetNumberOfSides(50)
        circle.GeneratePolygonOff()
        circle.SetNormal(*direction)

    def _show_radius_preview(
        self,
        center: np.ndarray,
        radius: float,
        view_type: str,
        scene: window.Scene,
    ) -> None:
        """
        Shows a yellow circle preview at the given center with the given radius.

        Used by the spinbox radius control to provide visual feedback.

        Args:
            center: Center point in world coordinates (display adjusted).
            radius: Radius in mm.
            view_type: View type ('axial', 'coronal', 'sagittal').
            scene: The 2D scene to add the preview to.
        """
        try:
            preview = self.panel.preview_line_actor
            if preview is not None and scene.HasViewProp(preview):
                mapper = preview.GetMapper()
                circle = mapper.GetInputAlgorithm()
                if not isinstance(circle, vtk.vtkRegularPolygonSource):
                    circle = vtk.vtkRegularPolygonSource()
                    mapper.SetInputConnection(circle.GetOutputPort())
                self._set_preview_circle(circle, center, radius, view_type)
                preview.GetProperty().SetLineWidth(3)
                scene.GetRenderWindow().Render()
                return

            # Remove old preview from ALL scenes
            if self.panel.preview_line_actor:
                _remove_actor_from_2d_scenes(self.panel, self.panel.preview_line_actor)
                self.panel.preview_line_actor = None

            # Create new preview actor
            mapper = vtk.vtkPolyDataMapper()
            circle = vtk.vtkRegularPolygonSource()
            self._set_preview_circle(circle, center, radius, view_type)
            mapper.SetInputConnection(circle.GetOutputPort())

            self.panel.preview_line_actor = vtk.vtkActor()
            self.panel.preview_line_actor.SetMapper(mapper)

            # UX design: Yellow, thick, always on top
            prop = self.panel.preview_line_actor.GetProperty()
            prop.SetColor(1.0, 1.0, 0.0)
            prop.SetLineWidth(3)
            prop.SetLighting(False)
            prop.SetOpacity(1.0)

            scene.add(self.panel.preview_line_actor)
            scene.GetRenderWindow().Render()

        except (ValueError, IndexError, RuntimeError, AttributeError) as e:
            logger.error(f"Error showing radius preview: {e}", exc_info=True)

    def _create_rectangle_preview(self) -> vtk.vtkPolyData:
        """Create a rectangle preview for rectangle mode."""
        p1 = self.panel.drawing_preview_points[0]
        p2 = self.panel.drawing_preview_points[1]
        view_type = getattr(self.panel, "current_drawing_view_type", "")

        displayed_affine, direction, _, axis = self._preview_slice_frame(view_type, p1)
        inverse = np.linalg.inv(displayed_affine)
        first = (inverse @ np.append(p1, 1.0))[:3]
        second = (inverse @ np.append(p2, 1.0))[:3]
        slice_indices = getattr(self.panel, "current_slice_indices", {})
        plane_index = slice_indices.get("xyz"[axis]) if slice_indices else None
        if plane_index is None:
            plane_index = first[axis]
        in_plane_axes = [index for index in range(3) if index != axis]

        points = vtk.vtkPoints()
        for first_axis, second_axis in ((0, 0), (1, 0), (1, 1), (0, 1)):
            voxel = first.copy()
            voxel[axis] = plane_index
            voxel[in_plane_axes[0]] = (first, second)[first_axis][in_plane_axes[0]]
            voxel[in_plane_axes[1]] = (first, second)[second_axis][in_plane_axes[1]]
            world = (displayed_affine @ np.append(voxel, 1.0))[:3]
            points.InsertNextPoint(*(world + _PREVIEW_DEPTH_MM * direction))

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(5)
        for i in range(4):
            polyline.GetPointIds().SetId(i, i)
        polyline.GetPointIds().SetId(4, 0)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        return polydata

    def _create_line_preview(self) -> vtk.vtkPolyData:
        """Create a line preview for pencil/eraser mode."""
        points = vtk.vtkPoints()
        for p in self.panel.drawing_preview_points:
            points.InsertNextPoint(p[0], p[1], p[2])

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(self.panel.drawing_preview_points))
        for i in range(len(self.panel.drawing_preview_points)):
            polyline.GetPointIds().SetId(i, i)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        return polydata

    @batch_render(lambda self: self.panel)
    def finish_drawing(self) -> None:
        """
        Called when mouse is released.
        Rasterizes the preview path into the ROI volume.
        """
        if not self.panel.main_window or not self.panel.main_window.current_drawing_roi:
            self._cleanup_preview()
            return

        roi_name = self.panel.main_window.current_drawing_roi
        if roi_name not in self.panel.main_window.roi_layers:
            self._cleanup_preview()
            return

        if not self.panel.drawing_preview_points:
            self._cleanup_preview()
            return

        try:
            roi_layer = self.panel.main_window.roi_layers[roi_name]
            roi_data = roi_layer["data"]
            roi_affine = roi_layer["affine"]
            roi_inv_affine = roi_layer["inv_affine"]
            shape = roi_data.shape

            view_type = getattr(self.panel, "current_drawing_view_type", None)
            if not view_type:
                return

            # Convert world points to voxel coordinates
            vox_points_float = self._world_to_voxel_points(
                roi_name, roi_inv_affine, shape, view_type
            )

            # Rasterize based on mode
            is_sphere = getattr(self.panel, "is_sphere_mode", False)
            is_rectangle = getattr(self.panel, "is_rectangle_mode", False)
            changed = False
            state_manager = getattr(self.panel.main_window, "state_manager", None)
            history = (
                state_manager.roi_modification(roi_name)
                if state_manager is not None
                else nullcontext()
            )
            with history:
                if is_sphere:
                    changed = self._rasterize_sphere(
                        roi_name, roi_data, vox_points_float, shape, view_type
                    )
                elif is_rectangle:
                    changed = self._rasterize_rectangle(
                        roi_name, roi_data, vox_points_float, shape, view_type
                    )
                else:
                    auto_fill = getattr(
                        self.panel.main_window, "auto_fill_voxels", False
                    )
                    is_eraser = getattr(self.panel, "is_eraser_mode", False)
                    changed_stroke = self._rasterize_freehand(
                        roi_data, vox_points_float, shape, view_type
                    )
                    changed_fill = False
                    if auto_fill and not is_eraser:
                        changed_fill = self._fill_polygon(
                            roi_name, roi_data, vox_points_float, shape, view_type
                        )
                    changed = changed_stroke or changed_fill

                if changed and not is_sphere and not is_rectangle:
                    self.panel.sphere_params_per_roi.pop(roi_name, None)
                    self.panel.rectangle_params_per_roi.pop(roi_name, None)

            if changed:
                # The final render runs after the yellow preview is removed.
                self.panel.update_roi_layer(roi_name, roi_data, roi_affine, render=False)
                status_msg = (
                    "ROI erased."
                    if getattr(self.panel, "is_eraser_mode", False)
                    else "ROI updated."
                )
                self.panel.update_status(status_msg)
                self._update_intersection_on_finish(
                    roi_name, is_sphere, is_rectangle, view_type
                )

        except Exception as e:
            logger.error(f"Error finishing drawing: {e}", exc_info=True)
        finally:
            self._cleanup_preview()
            self.panel._render_all()

    def _update_intersection_on_finish(
        self,
        roi_name: str,
        is_sphere: bool,
        is_rectangle: bool,
        view_type: str,
    ) -> None:
        """Force ROI-streamline intersection update after rasterization."""
        points = self.panel.drawing_preview_points
        if len(points) < 2:
            return

        mw = self.panel.main_window
        if not mw:
            return

        roi_mgr = getattr(mw, "roi_manager", None)

        if is_sphere:
            center = points[0]
            edge = points[1]
            radius = np.linalg.norm(center - edge)

            center_corrected = center.copy()
            if view_type in ["axial", "coronal"]:
                center_corrected[0] = -center_corrected[0]

            if roi_mgr is not None:
                roi_mgr.update_sphere_roi_intersection(
                    roi_name, center_corrected, radius
                )
            elif hasattr(mw, "update_sphere_roi_intersection"):
                mw.update_sphere_roi_intersection(roi_name, center_corrected, radius)

        elif is_rectangle:
            if roi_mgr is not None:
                roi_mgr.update_rectangle_roi_intersection(roi_name)
            elif hasattr(mw, "update_rectangle_roi_intersection"):
                mw.update_rectangle_roi_intersection(roi_name)

    def _cleanup_preview(self) -> None:
        """Clean up preview actor and reset state."""
        if self.panel.preview_line_actor:
            _remove_actor_from_2d_scenes(self.panel, self.panel.preview_line_actor)
            self.panel.preview_line_actor = None

        self.panel.drawing_preview_points = []
        self.panel.current_drawing_view_type = None

    @batch_render(lambda self: self.panel)
    def adjust_sphere_radius(self, delta: float, view_type: str = "axial") -> None:
        """Adjusts the radius of the last created sphere."""
        if not self.panel.main_window or not self.panel.main_window.current_drawing_roi:
            return

        roi_name = self.panel.main_window.current_drawing_roi
        roi_params = self.panel.sphere_params_per_roi.get(roi_name)

        if not roi_params or roi_name not in self.panel.main_window.roi_layers:
            return

        center_world = roi_params["center"].copy()
        current_radius = roi_params["radius"]
        stored_view_type = roi_params.get("view_type", "axial")

        # Restore to world coordinates for rasterization
        if stored_view_type in ["axial", "coronal"]:
            center_world[0] = -center_world[0]

        new_radius = max(0.5, current_radius + delta)
        if new_radius == current_radius:
            return

        mw = self.panel.main_window
        roi_layer = mw.roi_layers[roi_name]
        roi_data = roi_layer["data"]
        shape = roi_data.shape

        # Check max value before clearing
        current_max = roi_data.max()
        fill_val = current_max if current_max > 0 else 1

        state_manager = getattr(mw, "state_manager", None)
        history = (
            state_manager.roi_modification(roi_name)
            if state_manager is not None
            else nullcontext()
        )
        with history:
            roi_data.fill(0)
            self._rasterize_sphere_at_position(
                roi_name,
                roi_data,
                center_world,
                new_radius,
                shape,
                stored_view_type,
                fill_val,
            )
            self.panel.sphere_params_per_roi[roi_name]["radius"] = new_radius

        # Sync the radius spinbox if it exists
        if hasattr(mw, "sphere_radius_spinbox"):
            with signals_blocked(mw.sphere_radius_spinbox):
                mw.sphere_radius_spinbox.setValue(new_radius)

        # Remove any existing 3D preview sphere
        if roi_name in self.panel.roi_slice_actors:
            old_sphere = self.panel.roi_slice_actors[roi_name].get("sphere_3d")
            if old_sphere:
                try:
                    self.panel.scene.rm(old_sphere)
                except (ValueError, RuntimeError):
                    logger.debug("Failed to remove old sphere actor from scene.")
                self.panel.roi_slice_actors[roi_name]["sphere_3d"] = None

        # Refresh ROI visualization properly with color
        roi_affine = mw.roi_layers[roi_name]["affine"]
        self.panel.update_roi_layer(roi_name, roi_data, roi_affine)

        # Update ROI intersection for include/exclude filters
        center_for_3d = center_world.copy()
        if stored_view_type in ["axial", "coronal"]:
            center_for_3d[0] = -center_for_3d[0]
        roi_mgr = getattr(mw, "roi_manager", None) if mw else None
        if roi_mgr is not None:
            roi_mgr.update_sphere_roi_intersection(roi_name, center_for_3d, new_radius)
        elif mw and hasattr(mw, "update_sphere_roi_intersection"):
            mw.update_sphere_roi_intersection(roi_name, center_for_3d, new_radius)

        self.panel.update_status(f"Sphere radius adjusted to {new_radius:.2f}")
        self.panel._render_all()

    def update_3d_sphere_visuals(
        self,
        roi_name: str,
        center: np.ndarray,
        radius: float,
        render: bool = True,
    ) -> None:
        """
        Updates the 3D sphere actor for an ROI in real-time.

        Optimized to reuse existing VTK sphere source instead of recreating
        the actor on every update, which significantly improves drag performance.
        """
        if roi_name not in self.panel.roi_slice_actors:
            return

        # Get assigned color
        assigned_color = (1.0, 0.0, 0.0)
        if self.panel.main_window and roi_name in self.panel.main_window.roi_layers:
            assigned_color = self.panel.main_window.roi_layers[roi_name].get(
                "color", (1.0, 0.0, 0.0)
            )

        existing_sphere = self.panel.roi_slice_actors[roi_name].get("sphere_3d")
        sphere_source = self.panel.roi_slice_actors[roi_name].get("sphere_source")

        # Check if we can reuse the existing sphere
        if existing_sphere is not None and sphere_source is not None:
            # Update existing sphere source in-place (fast path)
            sphere_source.SetCenter(center[0], center[1], center[2])
            sphere_source.SetRadius(radius)
            sphere_source.Update()

            # Update color if changed
            existing_sphere.GetProperty().SetColor(*assigned_color)

            # Mark actor as modified and render
            existing_sphere.Modified()
            if render:
                request_render(self.panel, self.panel.render_window)
            return

        # Need to create a new sphere actor
        if existing_sphere is not None:
            try:
                self.panel.scene.rm(existing_sphere)
            except (ValueError, RuntimeError):
                logger.debug("Failed to remove existing sphere actor from scene.")
            self.panel.roi_slice_actors[roi_name]["sphere_3d"] = None
            self.panel.roi_slice_actors[roi_name]["sphere_source"] = None

        # Create new VTK native sphere (supports in-place updates)
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(center[0], center[1], center[2])
        sphere_source.SetRadius(radius)
        sphere_source.SetPhiResolution(32)
        sphere_source.SetThetaResolution(32)
        sphere_source.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(mapper)
        sphere_actor.GetProperty().SetColor(*assigned_color)
        sphere_actor.GetProperty().SetOpacity(0.5)

        self.panel.scene.add(sphere_actor)
        self.panel.roi_slice_actors[roi_name]["sphere_3d"] = sphere_actor
        self.panel.roi_slice_actors[roi_name]["sphere_source"] = sphere_source
        if render:
            request_render(self.panel, self.panel.render_window)

    def update_3d_rectangle_visuals(
        self,
        roi_name: str,
        start: np.ndarray,
        end: np.ndarray,
        view_type: str,
        render: bool = True,
        corners: Optional[np.ndarray] = None,
    ) -> None:
        """
        Updates the 3D rectangle actor for an ROI in real-time.

        Optimized to reuse existing VTK points instead of recreating
        the actor on every update, which improves drag performance.
        """
        if roi_name not in self.panel.roi_slice_actors:
            return

        # Get assigned color
        assigned_color = (1.0, 0.0, 0.0)
        if self.panel.main_window and roi_name in self.panel.main_window.roi_layers:
            assigned_color = self.panel.main_window.roi_layers[roi_name].get(
                "color", (1.0, 0.0, 0.0)
            )

        if corners is None:
            min_pt = np.minimum(start, end)
            max_pt = np.maximum(start, end)
            if view_type == "axial":
                z_pos = (min_pt[2] + max_pt[2]) / 2.0
                corners = np.array(
                    [
                        [min_pt[0], min_pt[1], z_pos],
                        [max_pt[0], min_pt[1], z_pos],
                        [max_pt[0], max_pt[1], z_pos],
                        [min_pt[0], max_pt[1], z_pos],
                    ]
                )
            elif view_type == "coronal":
                y_pos = (min_pt[1] + max_pt[1]) / 2.0
                corners = np.array(
                    [
                        [min_pt[0], y_pos, min_pt[2]],
                        [max_pt[0], y_pos, min_pt[2]],
                        [max_pt[0], y_pos, max_pt[2]],
                        [min_pt[0], y_pos, max_pt[2]],
                    ]
                )
            else:
                x_pos = (min_pt[0] + max_pt[0]) / 2.0
                corners = np.array(
                    [
                        [x_pos, min_pt[1], min_pt[2]],
                        [x_pos, max_pt[1], min_pt[2]],
                        [x_pos, max_pt[1], max_pt[2]],
                        [x_pos, min_pt[1], max_pt[2]],
                    ]
                )
        else:
            corners = np.asarray(corners, dtype=float)
            if corners.shape != (4, 3):
                raise ValueError("Rectangle corners must have shape (4, 3).")

        existing_rect = self.panel.roi_slice_actors[roi_name].get("rectangle_3d")
        rect_points = self.panel.roi_slice_actors[roi_name].get("rectangle_points")

        # Check if we can reuse the existing rectangle
        if existing_rect is not None and rect_points is not None:
            # Update existing points in-place (fast path)
            for i, corner in enumerate(corners):
                rect_points.SetPoint(i, corner[0], corner[1], corner[2])
            rect_points.Modified()

            # Update color if changed and ensure two-sided rendering
            prop = existing_rect.GetProperty()
            prop.SetColor(*assigned_color)
            prop.BackfaceCullingOff()
            prop.FrontfaceCullingOff()
            existing_rect.Modified()

            if render:
                request_render(self.panel, self.panel.render_window)
            return

        # Need to create a new rectangle actor (first time)
        if existing_rect is not None:
            try:
                self.panel.scene.rm(existing_rect)
            except (ValueError, RuntimeError):
                logger.debug("Failed to remove existing rectangle actor from scene.")
            self.panel.roi_slice_actors[roi_name]["rectangle_3d"] = None
            self.panel.roi_slice_actors[roi_name]["rectangle_points"] = None

        # Create new VTK rectangle
        points = vtk.vtkPoints()
        for corner in corners:
            points.InsertNextPoint(corner)

        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)
        for i in range(4):
            polygon.GetPointIds().SetId(i, i)

        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polygons)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        rectangle_actor = vtk.vtkActor()
        rectangle_actor.SetMapper(mapper)

        prop = rectangle_actor.GetProperty()
        prop.SetColor(assigned_color[0], assigned_color[1], assigned_color[2])
        prop.SetOpacity(0.4)
        prop.SetRepresentationToSurface()
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(assigned_color[0], assigned_color[1], assigned_color[2])
        prop.SetLineWidth(2.0)

        # Enable two-sided rendering so rectangle is visible from all camera angles
        prop.BackfaceCullingOff()
        prop.FrontfaceCullingOff()

        self.panel.scene.add(rectangle_actor)
        self.panel.roi_slice_actors[roi_name]["rectangle_3d"] = rectangle_actor
        self.panel.roi_slice_actors[roi_name]["rectangle_points"] = points

        if render:
            request_render(self.panel, self.panel.render_window)
