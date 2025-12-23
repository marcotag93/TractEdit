# -*- coding: utf-8 -*-

"""
Drawing manager for ROI drawing operations on 2D views.

Handles pencil, eraser, sphere, and rectangle drawing modes,
including preview visualization and rasterization to 3D volumes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import vtk
from fury import actor, window
from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


class DrawingManager:
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
                self.panel.set_roi_layer_color(
                    roi_name, (0.0, 188.0 / 255.0, 212.0 / 255.0)
                )
                self.panel.main_window.roi_visibility[roi_name] = True

            # Handle different drawing modes
            if getattr(self.panel, "is_sphere_mode", False):
                self._handle_sphere_mode(interactor, roi_name, world_pos, view_type)
            elif getattr(self.panel, "is_rectangle_mode", False):
                self._handle_rectangle_mode(interactor, roi_name, world_pos, view_type)
            else:
                # Normal drawing: append points
                self.panel.drawing_preview_points.append(world_pos)

            # Update the preview line
            self._update_drawing_preview(active_scene)

            # Real-time ROI effect update for sphere mode
            if (
                getattr(self.panel, "is_sphere_mode", False)
                and len(self.panel.drawing_preview_points) >= 2
            ):
                self._update_sphere_realtime(roi_name, view_type)

        except Exception as e:
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
                self.panel.render_window.Render()

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
        if (
            self.panel.main_window
            and self.panel.main_window.tractogram_data
            and len(self.panel.main_window.tractogram_data) > LARGE_TRACTOGRAM_THRESHOLD
        ):
            return

        # Throttled intersection update
        # ##TODO this probably can be handled better, will be refactored later
        self._sphere_update_counter += 1
        if self._sphere_update_counter >= 15:
            self._sphere_update_counter = 0
            if self.panel.main_window and hasattr(
                self.panel.main_window, "update_sphere_roi_intersection"
            ):
                self.panel.main_window.update_sphere_roi_intersection(
                    roi_name, center_corrected, radius
                )

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
                for s in [
                    self.panel.axial_scene,
                    self.panel.coronal_scene,
                    self.panel.sagittal_scene,
                ]:
                    if s:
                        try:
                            s.rm(self.panel.preview_line_actor)
                        except Exception:
                            pass
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

        except Exception as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)

    def _create_circle_preview(self) -> vtk.vtkPolyData:
        """Create a circle preview for sphere mode."""
        center = self.panel.drawing_preview_points[0]
        edge = self.panel.drawing_preview_points[1]
        radius = np.linalg.norm(center - edge)

        circle = vtk.vtkRegularPolygonSource()
        circle.SetCenter(center)
        circle.SetRadius(radius)
        circle.SetNumberOfSides(50)
        circle.GeneratePolygonOff()

        view_type = getattr(self.panel, "current_drawing_view_type", "")
        if view_type == "axial":
            circle.SetNormal(0, 0, 1)
        elif view_type == "coronal":
            circle.SetNormal(0, 1, 0)
        elif view_type == "sagittal":
            circle.SetNormal(1, 0, 0)

        circle.Update()
        return circle.GetOutput()

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
            # Create circle polydata
            circle = vtk.vtkRegularPolygonSource()
            circle.SetCenter(center[0], center[1], center[2])
            circle.SetRadius(radius)
            circle.SetNumberOfSides(50)
            circle.GeneratePolygonOff()

            if view_type == "axial":
                circle.SetNormal(0, 0, 1)
            elif view_type == "coronal":
                circle.SetNormal(0, 1, 0)
            elif view_type == "sagittal":
                circle.SetNormal(1, 0, 0)

            circle.Update()

            # Remove old preview from ALL scenes
            if self.panel.preview_line_actor:
                for s in [
                    self.panel.axial_scene,
                    self.panel.coronal_scene,
                    self.panel.sagittal_scene,
                ]:
                    if s:
                        try:
                            s.rm(self.panel.preview_line_actor)
                        except Exception:
                            pass
                self.panel.preview_line_actor = None

            # Create new preview actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(circle.GetOutput())

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

        except Exception as e:
            logger.error(f"Error showing radius preview: {e}", exc_info=True)

    def _create_rectangle_preview(self) -> vtk.vtkPolyData:
        """Create a rectangle preview for rectangle mode."""
        p1 = self.panel.drawing_preview_points[0]
        p2 = self.panel.drawing_preview_points[1]
        view_type = getattr(self.panel, "current_drawing_view_type", "")

        points = vtk.vtkPoints()

        if view_type == "axial":
            z = p1[2]
            points.InsertNextPoint(p1[0], p1[1], z)
            points.InsertNextPoint(p2[0], p1[1], z)
            points.InsertNextPoint(p2[0], p2[1], z)
            points.InsertNextPoint(p1[0], p2[1], z)
        elif view_type == "coronal":
            y = p1[1]
            points.InsertNextPoint(p1[0], y, p1[2])
            points.InsertNextPoint(p2[0], y, p1[2])
            points.InsertNextPoint(p2[0], y, p2[2])
            points.InsertNextPoint(p1[0], y, p2[2])
        elif view_type == "sagittal":
            x = p1[0]
            points.InsertNextPoint(x, p1[1], p1[2])
            points.InsertNextPoint(x, p2[1], p1[2])
            points.InsertNextPoint(x, p2[1], p2[2])
            points.InsertNextPoint(x, p1[1], p2[2])
        else:
            points.InsertNextPoint(p1[0], p1[1], p1[2])
            points.InsertNextPoint(p2[0], p2[1], p2[2])

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

    def finish_drawing(self) -> None:
        """
        Called when mouse is released.
        Rasterizes the preview path into the ROI volume.
        """
        if not self.panel.main_window or not self.panel.main_window.current_drawing_roi:
            self.panel.drawing_preview_points = []
            return

        roi_name = self.panel.main_window.current_drawing_roi
        if roi_name not in self.panel.main_window.roi_layers:
            self.panel.drawing_preview_points = []
            return

        if not self.panel.drawing_preview_points:
            return

        try:
            roi_layer = self.panel.main_window.roi_layers[roi_name]
            roi_data = roi_layer["data"]
            roi_affine = roi_layer["affine"]
            roi_inv_affine = roi_layer["inv_affine"]
            shape = roi_data.shape

            # Save state for undo
            if self.panel.main_window and hasattr(
                self.panel.main_window, "_save_roi_state_for_undo"
            ):
                self.panel.main_window._save_roi_state_for_undo(roi_name)

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

            if is_sphere:
                changed = self._rasterize_sphere(
                    roi_name, roi_data, vox_points_float, shape, view_type
                )
            elif is_rectangle:
                changed = self._rasterize_rectangle(
                    roi_name, roi_data, vox_points_float, shape, view_type
                )
            else:
                changed = self._rasterize_freehand(
                    roi_data, vox_points_float, shape, view_type
                )

            # Update actor if changed
            if changed:
                self.panel.update_roi_layer(roi_name, roi_data, roi_affine)
                status_msg = (
                    "ROI erased."
                    if getattr(self.panel, "is_eraser_mode", False)
                    else "ROI updated."
                )
                self.panel.update_status(status_msg)

                # Force intersection update on finish for sphere/rectangle
                if is_sphere and len(self.panel.drawing_preview_points) >= 2:
                    center = self.panel.drawing_preview_points[0]
                    edge = self.panel.drawing_preview_points[1]
                    radius = np.linalg.norm(center - edge)

                    # Apply X-axis correction for axial/coronal views
                    center_corrected = center.copy()
                    if view_type in ["axial", "coronal"]:
                        center_corrected[0] = -center_corrected[0]

                    if self.panel.main_window and hasattr(
                        self.panel.main_window, "update_sphere_roi_intersection"
                    ):
                        self.panel.main_window.update_sphere_roi_intersection(
                            roi_name, center_corrected, radius
                        )
                elif is_rectangle and len(self.panel.drawing_preview_points) >= 2:
                    start = self.panel.drawing_preview_points[0].copy()
                    end = self.panel.drawing_preview_points[1].copy()

                    # Apply X-axis correction for axial/coronal views
                    if view_type in ["axial", "coronal"]:
                        start[0] = -start[0]
                        end[0] = -end[0]

                    if self.panel.main_window and hasattr(
                        self.panel.main_window, "update_rectangle_roi_intersection"
                    ):
                        min_v = np.minimum(start, end)
                        max_v = np.maximum(start, end)
                        epsilon = 0.5
                        if view_type == "axial":
                            min_v[2] -= epsilon
                            max_v[2] += epsilon
                        elif view_type == "coronal":
                            min_v[1] -= epsilon
                            max_v[1] += epsilon
                        elif view_type == "sagittal":
                            min_v[0] -= epsilon
                            max_v[0] += epsilon
                        self.panel.main_window.update_rectangle_roi_intersection(
                            roi_name, min_v, max_v
                        )

            # Clean up preview
            self._cleanup_preview()
            self.panel._render_all()

        except Exception as e:
            logger.error(f"Error finishing drawing: {e}", exc_info=True)
            self.panel.drawing_preview_points = []

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

        points_world = np.array(self.panel.drawing_preview_points)
        homog_points = np.hstack([points_world, np.ones((len(points_world), 1))])

        if not roi_actors or actor_key not in roi_actors:
            # Fallback to affine
            vox_points_float = np.dot(roi_inv_affine, homog_points.T).T[:, :3]
            if view_type in ["axial", "coronal"]:
                vox_points_float[:, 0] = (shape[0] - 1) - vox_points_float[:, 0]
            elif view_type == "sagittal":
                vox_points_float[:, 0] = (shape[0] - 1) - vox_points_float[:, 0] - 1
        else:
            actor_obj = roi_actors[actor_key]
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
                except Exception:
                    pass

            if image_data and isinstance(image_data, vtk.vtkImageData):
                spacing = image_data.GetSpacing()
                origin = image_data.GetOrigin()
                vox_points_float = (model_points - np.array(origin)) / np.array(spacing)
            else:
                vox_points_float = model_points

            # Apply flips
            if view_type in ["axial", "coronal"]:
                vox_points_float[:, 0] = (shape[0] - 1) - vox_points_float[:, 0]
            elif view_type == "sagittal":
                vox_points_float[:, 0] = (shape[0] - 1) - vox_points_float[:, 0] - 1

        return vox_points_float

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
        edge_vox = vox_points_float[1]
        radius_vox = np.linalg.norm(center_vox - edge_vox)

        # Clear rectangle params
        self.panel.rectangle_params_per_roi.pop(roi_name, None)

        # Define bounding box for the NEW sphere
        min_vox = np.floor(center_vox - radius_vox).astype(int)
        max_vox = np.ceil(center_vox + radius_vox).astype(int)
        min_vox = np.maximum(min_vox, 0)
        max_vox = np.minimum(max_vox, np.array(shape) - 1)

        # Clear only what's needed instead of entire volume
        old_nonzero = np.argwhere(roi_data > 0)
        if len(old_nonzero) > 0:
            old_min = old_nonzero.min(axis=0)
            old_max = old_nonzero.max(axis=0)
            roi_data[
                old_min[0] : old_max[0] + 1,
                old_min[1] : old_max[1] + 1,
                old_min[2] : old_max[2] + 1,
            ] = 0

        x_range = np.arange(min_vox[0], max_vox[0] + 1)
        y_range = np.arange(min_vox[1], max_vox[1] + 1)
        z_range = np.arange(min_vox[2], max_vox[2] + 1)

        if len(x_range) > 0 and len(y_range) > 0 and len(z_range) > 0:
            xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")
            dist_sq = (
                (xx - center_vox[0]) ** 2
                + (yy - center_vox[1]) ** 2
                + (zz - center_vox[2]) ** 2
            )
            mask = dist_sq <= radius_vox**2

            roi_slice = roi_data[
                min_vox[0] : max_vox[0] + 1,
                min_vox[1] : max_vox[1] + 1,
                min_vox[2] : max_vox[2] + 1,
            ]
            roi_slice[mask] = 1

            # Store params
            if len(self.panel.drawing_preview_points) >= 2:
                center_to_store = self.panel.drawing_preview_points[0].copy()
                if view_type in ["axial", "coronal"]:
                    center_to_store[0] = -center_to_store[0]

                self.panel.sphere_params_per_roi[roi_name] = {
                    "center": center_to_store,
                    "radius": np.linalg.norm(
                        self.panel.drawing_preview_points[0]
                        - self.panel.drawing_preview_points[1]
                    ),
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
        # Remove any existing 3D sphere actor
        if roi_name in self.panel.roi_slice_actors:
            old_sphere = self.panel.roi_slice_actors[roi_name].get("sphere_3d")
            if old_sphere:
                self.panel.scene.rm(old_sphere)
                self.panel.roi_slice_actors[roi_name]["sphere_3d"] = None

        # Clear sphere params
        self.panel.sphere_params_per_roi.pop(roi_name, None)

        if len(vox_points_float) < 2:
            return False

        p1_vox = vox_points_float[0]
        p2_vox = vox_points_float[1]

        min_v = np.minimum(p1_vox, p2_vox).astype(int)
        max_v = np.maximum(p1_vox, p2_vox).astype(int)
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, np.array(shape) - 1)

        # Clear only existing ROI data instead of entire volume
        old_nonzero = np.argwhere(roi_data > 0)
        if len(old_nonzero) > 0:
            old_min = old_nonzero.min(axis=0)
            old_max = old_nonzero.max(axis=0)
            roi_data[
                old_min[0] : old_max[0] + 1,
                old_min[1] : old_max[1] + 1,
                old_min[2] : old_max[2] + 1,
            ] = 0

        # Ensure thickness for 2D drawing
        if view_type == "axial" and min_v[2] == max_v[2]:
            max_v[2] += 1
        elif view_type == "coronal" and min_v[1] == max_v[1]:
            max_v[1] += 1
        elif view_type == "sagittal" and min_v[0] == max_v[0]:
            max_v[0] += 1

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

    def _cleanup_preview(self) -> None:
        """Clean up preview actor and reset state."""
        if self.panel.preview_line_actor:
            for s in [
                self.panel.axial_scene,
                self.panel.coronal_scene,
                self.panel.sagittal_scene,
            ]:
                try:
                    s.rm(self.panel.preview_line_actor)
                except Exception:
                    pass
            self.panel.preview_line_actor = None

        self.panel.drawing_preview_points = []
        self.panel.current_drawing_view_type = None

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

        # Un-flip X for Axial/Coronal
        if stored_view_type in ["axial", "coronal"]:
            center_world[0] = -center_world[0]

        new_radius = max(0.5, current_radius + delta)
        if new_radius == current_radius:
            return

        # Save state for undo
        if self.panel.main_window and hasattr(
            self.panel.main_window, "_save_roi_state_for_undo"
        ):
            self.panel.main_window._save_roi_state_for_undo(roi_name)

        roi_layer = self.panel.main_window.roi_layers[roi_name]
        roi_data = roi_layer["data"]
        shape = roi_data.shape

        # Check max value before clearing
        current_max = roi_data.max()
        fill_val = current_max if current_max > 0 else 1

        roi_data.fill(0)

        # Rasterize new sphere
        self._rasterize_sphere_at_position(
            roi_name,
            roi_data,
            center_world,
            new_radius,
            shape,
            stored_view_type,
            fill_val,
        )

        # Update params
        self.panel.sphere_params_per_roi[roi_name]["radius"] = new_radius

        # Sync the radius spinbox if it exists
        mw = self.panel.main_window
        if hasattr(mw, "sphere_radius_spinbox"):
            mw.sphere_radius_spinbox.blockSignals(True)
            mw.sphere_radius_spinbox.setValue(new_radius)
            mw.sphere_radius_spinbox.blockSignals(False)

        # Remove any existing 3D preview sphere
        if roi_name in self.panel.roi_slice_actors:
            old_sphere = self.panel.roi_slice_actors[roi_name].get("sphere_3d")
            if old_sphere:
                try:
                    self.panel.scene.rm(old_sphere)
                except Exception:
                    pass
                self.panel.roi_slice_actors[roi_name]["sphere_3d"] = None

        # Refresh ROI visualization properly with color
        roi_affine = mw.roi_layers[roi_name]["affine"]
        self.panel.update_roi_layer(roi_name, roi_data, roi_affine)

        # Update ROI intersection for include/exclude filters
        center_for_3d = center_world.copy()
        if stored_view_type in ["axial", "coronal"]:
            center_for_3d[0] = -center_for_3d[0]
        if mw and hasattr(mw, "update_sphere_roi_intersection"):
            mw.update_sphere_roi_intersection(roi_name, center_for_3d, new_radius)

        self.panel.update_status(f"Sphere radius adjusted to {new_radius:.2f}")
        self.panel._render_all()

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

        center_vox = None
        edge_vox = None

        if roi_actors and actor_key in roi_actors:
            actor_obj = roi_actors[actor_key]
            matrix = actor_obj.GetMatrix()

            inv_matrix = vtk.vtkMatrix4x4()
            inv_matrix.DeepCopy(matrix)
            inv_matrix.Invert()

            mat_np = np.zeros((4, 4))
            for r in range(4):
                for c in range(4):
                    mat_np[r, c] = inv_matrix.GetElement(r, c)

            p_h = np.append(center_w, 1.0)
            model_point = np.dot(mat_np, p_h)[:3]

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
                except Exception:
                    pass

            if image_data and isinstance(image_data, vtk.vtkImageData):
                spacing = image_data.GetSpacing()
                origin = image_data.GetOrigin()
                center_vox = (model_point - np.array(origin)) / np.array(spacing)
            else:
                center_vox = model_point

            edge_w = center_w + np.array([radius_w, 0, 0])
            p_h_e = np.append(edge_w, 1.0)
            model_point_e = np.dot(mat_np, p_h_e)[:3]

            if image_data and isinstance(image_data, vtk.vtkImageData):
                edge_vox = (model_point_e - np.array(origin)) / np.array(spacing)
            else:
                edge_vox = model_point_e
        else:
            p_h = np.append(center_w, 1.0)
            center_vox = np.dot(roi_inv_affine, p_h)[:3]

            edge_w = center_w + np.array([radius_w, 0, 0])
            p_h_e = np.append(edge_w, 1.0)
            edge_vox = np.dot(roi_inv_affine, p_h_e)[:3]

        # Apply flips
        if view_type in ["axial", "coronal"]:
            center_vox[0] = (shape[0] - 1) - center_vox[0]
            edge_vox[0] = (shape[0] - 1) - edge_vox[0]
        elif view_type == "sagittal":
            center_vox[0] = (shape[0] - 1) - center_vox[0] - 1
            edge_vox[0] = (shape[0] - 1) - edge_vox[0] - 1

        radius_v = np.linalg.norm(center_vox - edge_vox)

        min_v = np.floor(center_vox - radius_v).astype(int)
        max_v = np.ceil(center_vox + radius_v).astype(int)
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, np.array(shape) - 1)

        x_r = np.arange(min_v[0], max_v[0] + 1)
        y_r = np.arange(min_v[1], max_v[1] + 1)
        z_r = np.arange(min_v[2], max_v[2] + 1)

        if len(x_r) > 0 and len(y_r) > 0 and len(z_r) > 0:
            xx, yy, zz = np.meshgrid(x_r, y_r, z_r, indexing="ij")
            dist_sq = (
                (xx - center_vox[0]) ** 2
                + (yy - center_vox[1]) ** 2
                + (zz - center_vox[2]) ** 2
            )
            mask = dist_sq <= radius_v**2

            roi_slice = roi_data[
                min_v[0] : max_v[0] + 1,
                min_v[1] : max_v[1] + 1,
                min_v[2] : max_v[2] + 1,
            ]
            roi_slice[mask] = fill_val

    def update_3d_sphere_visuals(
        self, roi_name: str, center: np.ndarray, radius: float
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
            self.panel.render_window.Render()
            return

        # Need to create a new sphere actor
        if existing_sphere is not None:
            try:
                self.panel.scene.rm(existing_sphere)
            except Exception:
                pass
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
        self.panel.render_window.Render()

    def update_3d_rectangle_visuals(
        self,
        roi_name: str,
        start: np.ndarray,
        end: np.ndarray,
        view_type: str,
        render: bool = True,
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

        # Calculate corners
        min_pt = np.minimum(start, end)
        max_pt = np.maximum(start, end)

        if view_type == "axial":
            z_pos = (min_pt[2] + max_pt[2]) / 2.0
            corners = [
                [min_pt[0], min_pt[1], z_pos],
                [max_pt[0], min_pt[1], z_pos],
                [max_pt[0], max_pt[1], z_pos],
                [min_pt[0], max_pt[1], z_pos],
            ]
        elif view_type == "coronal":
            y_pos = (min_pt[1] + max_pt[1]) / 2.0
            corners = [
                [min_pt[0], y_pos, min_pt[2]],
                [max_pt[0], y_pos, min_pt[2]],
                [max_pt[0], y_pos, max_pt[2]],
                [min_pt[0], y_pos, max_pt[2]],
            ]
        else:  # sagittal
            x_pos = (min_pt[0] + max_pt[0]) / 2.0
            corners = [
                [x_pos, min_pt[1], min_pt[2]],
                [x_pos, max_pt[1], min_pt[2]],
                [x_pos, max_pt[1], max_pt[2]],
                [x_pos, min_pt[1], max_pt[2]],
            ]

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
                self.panel.render_window.Render()
            return

        # Need to create a new rectangle actor (first time)
        if existing_rect is not None:
            try:
                self.panel.scene.rm(existing_rect)
            except Exception:
                pass
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
            self.panel.render_window.Render()
