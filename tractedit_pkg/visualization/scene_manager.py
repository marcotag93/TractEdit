# -*- coding: utf-8 -*-

"""
Scene manager for TractEdit visualization.

Manages the 4-view camera states, crosshair updates, and scene
initialization for the axial, coronal, sagittal, and 3D views.
"""

import logging
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any

import numpy as np
import vtk
from fury import window, actor

from . import actors
from . import coordinates

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


class SceneManager:
    """
    Manages camera states, crosshair updates, and scene initialization.

    Handles the 4-view layout (1x 3D, 3x 2D orthogonal views) camera
    configuration and coordinate-synchronized crosshair visualization.
    """

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        """
        Initialize the scene manager.

        Args:
            vtk_panel: Reference to the parent VTKPanel instance.
        """
        self.vtk_panel = vtk_panel

    # =========================================================================
    # Camera Management
    # =========================================================================

    def setup_ortho_cameras(self) -> None:
        """Sets the cameras for the 2D views to be orthogonal."""
        vp = self.vtk_panel
        if not vp.axial_scene or not vp.coronal_scene or not vp.sagittal_scene:
            return

        try:
            # Call with reset=True for initial setup
            self.update_axial_camera(reset_zoom_pan=True)
            self.update_coronal_camera(reset_zoom_pan=True)
            self.update_sagittal_camera(reset_zoom_pan=True)
        except Exception as e:
            logger.error(f"Error setting up ortho cameras: {e}")

    def update_axial_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the axial 2D camera to follow its slice."""
        vp = self.vtk_panel
        if not vp.axial_scene:
            return
        try:
            cam = vp.axial_scene.GetActiveCamera()
            if not cam.GetParallelProjection():
                cam.SetParallelProjection(1)

            if reset_zoom_pan or not vp.axial_slice_actor:
                # Full reset (initial setup or fallback)
                vp.axial_scene.reset_camera()
                if not cam.GetParallelProjection():
                    cam.SetParallelProjection(1)
                fp = cam.GetFocalPoint()
                dist = cam.GetDistance()
                cam.SetPosition(fp[0], fp[1], fp[2] + dist)  # View from +Z
                cam.SetFocalPoint(fp[0], fp[1], fp[2])
                cam.SetViewUp(0, 1, 0)
            else:
                # Just follow the slice in Z, keeping X,Y pan and zoom
                actor_center = vp.axial_slice_actor.GetCenter()

                current_fp = cam.GetFocalPoint()
                current_pos = cam.GetPosition()

                new_focal_z = actor_center[2]
                z_delta = new_focal_z - current_fp[2]

                # Only update if the slice has actually moved in Z
                if abs(z_delta) > 1e-6:
                    cam.SetFocalPoint(current_fp[0], current_fp[1], new_focal_z)
                    cam.SetPosition(
                        current_pos[0], current_pos[1], current_pos[2] + z_delta
                    )

            vp.axial_scene.reset_clipping_range()
        except Exception as e:
            logger.error(f"Error updating axial camera: {e}")

    def update_coronal_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the coronal 2D camera to follow its slice."""
        vp = self.vtk_panel
        if not vp.coronal_scene:
            return
        try:
            cam = vp.coronal_scene.GetActiveCamera()
            if not cam.GetParallelProjection():
                cam.SetParallelProjection(1)

            if reset_zoom_pan or not vp.coronal_slice_actor:
                # Full reset
                vp.coronal_scene.reset_camera()
                if not cam.GetParallelProjection():
                    cam.SetParallelProjection(1)
                fp = cam.GetFocalPoint()
                dist = cam.GetDistance()
                cam.SetPosition(fp[0], fp[1] - dist, fp[2])  # View from -Y
                cam.SetFocalPoint(fp[0], fp[1], fp[2])
                cam.SetViewUp(0, 0, 1)  # Up is Z
            else:
                # Just follow the slice in Y
                actor_center = vp.coronal_slice_actor.GetCenter()

                current_fp = cam.GetFocalPoint()
                current_pos = cam.GetPosition()

                new_focal_y = actor_center[1]
                y_delta = new_focal_y - current_fp[1]

                if abs(y_delta) > 1e-6:
                    cam.SetFocalPoint(current_fp[0], new_focal_y, current_fp[2])
                    cam.SetPosition(
                        current_pos[0], current_pos[1] + y_delta, current_pos[2]
                    )

            vp.coronal_scene.reset_clipping_range()
        except Exception as e:
            logger.error(f"Error updating coronal camera: {e}")

    def update_sagittal_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the sagittal 2D camera to follow its slice."""
        vp = self.vtk_panel
        if not vp.sagittal_scene:
            return
        try:
            cam = vp.sagittal_scene.GetActiveCamera()
            if not cam.GetParallelProjection():
                cam.SetParallelProjection(1)

            if reset_zoom_pan or not vp.sagittal_slice_actor:
                vp.sagittal_scene.reset_camera()
                if not cam.GetParallelProjection():
                    cam.SetParallelProjection(1)
                fp = cam.GetFocalPoint()
                dist = cam.GetDistance()
                cam.SetPosition(fp[0] + dist, fp[1], fp[2])  # Was fp[0] - dist
                cam.SetFocalPoint(fp[0], fp[1], fp[2])
                cam.SetViewUp(0, 0, 1)  # Up is Z
            else:
                actor_center = vp.sagittal_slice_actor.GetCenter()

                current_fp = cam.GetFocalPoint()
                current_pos = cam.GetPosition()

                new_focal_x = actor_center[0]
                x_delta = new_focal_x - current_fp[0]

                if abs(x_delta) > 1e-6:
                    cam.SetFocalPoint(new_focal_x, current_fp[1], current_fp[2])
                    cam.SetPosition(
                        current_pos[0] + x_delta, current_pos[1], current_pos[2]
                    )

            vp.sagittal_scene.reset_clipping_range()

            # Expand the clipping range to ensure the crosshair in the overlay is never clipped
            near, far = cam.GetClippingRange()
            cam.SetClippingRange(near * 0.01, far * 100)

            # Don't reset clipping range on overlay - it shares the camera
            if (
                hasattr(vp, "sagittal_overlay_renderer")
                and vp.sagittal_overlay_renderer
            ):
                vp.sagittal_overlay_renderer.SetActiveCamera(cam)
        except Exception as e:
            logger.error(f"Error updating sagittal camera: {e}")

    def reset_2d_view(self, view_type: str) -> None:
        """
        Resets the camera for a specific 2D view and re-renders it.

        Args:
            view_type: One of "axial", "coronal", or "sagittal".
        """
        vp = self.vtk_panel
        try:
            if view_type == "axial":
                self.update_axial_camera(reset_zoom_pan=True)
                if vp.axial_render_window:
                    vp.axial_render_window.Render()

            elif view_type == "coronal":
                self.update_coronal_camera(reset_zoom_pan=True)
                if vp.coronal_render_window:
                    vp.coronal_render_window.Render()

            elif view_type == "sagittal":
                self.update_sagittal_camera(reset_zoom_pan=True)
                if vp.sagittal_render_window:
                    vp.sagittal_render_window.Render()

        except Exception as e:
            logger.error(f"Error resetting 2D view ({view_type}): {e}")

    # =========================================================================
    # Crosshair Management
    # =========================================================================

    def clear_crosshairs(self) -> None:
        """Removes the 2D crosshair actors from their scenes."""
        vp = self.vtk_panel

        # Remove from main scenes
        actor_scene_pairs = [
            (vp.axial_crosshair_actor, vp.axial_scene),
            (vp.coronal_crosshair_actor, vp.coronal_scene),
        ]
        for act, scn in actor_scene_pairs:
            if scn and act:
                try:
                    scn.rm(act)
                except (ValueError, AttributeError):
                    pass
                except Exception as e:
                    logger.error(f"Error removing crosshair actor: {e}")

        # Remove sagittal crosshair from overlay renderer
        if (
            hasattr(vp, "sagittal_overlay_renderer")
            and vp.sagittal_overlay_renderer
            and vp.sagittal_crosshair_actor
        ):
            try:
                vp.sagittal_overlay_renderer.RemoveActor(vp.sagittal_crosshair_actor)
            except Exception as e:
                logger.error(f"Error removing sagittal crosshair actor: {e}")

        vp.axial_crosshair_actor = None
        vp.coronal_crosshair_actor = None
        vp.sagittal_crosshair_actor = None
        vp.crosshair_lines = {}
        vp.crosshair_appenders = {}

    def create_or_update_crosshairs(self) -> None:
        """Creates or updates the 2D crosshair actors based on current slice indices."""
        vp = self.vtk_panel

        if (
            vp.main_window is None
            or vp.main_window.anatomical_image_data is None
            or not all(vp.current_slice_indices.values())
        ):
            self.clear_crosshairs()
            return

        try:
            # Get current state
            x, y, z = (
                vp.current_slice_indices["x"],
                vp.current_slice_indices["y"],
                vp.current_slice_indices["z"],
            )
            x_min, x_max = vp.image_extents["x"]
            y_min, y_max = vp.image_extents["y"]
            z_min, z_max = vp.image_extents["z"]

            # Define line endpoints in WORLD coordinates
            main_affine = vp.main_window.anatomical_image_affine

            # Axial View (X-Y plane):
            ax_line_x_p1 = coordinates.voxel_to_world([x, y_min, z], main_affine)
            ax_line_x_p2 = coordinates.voxel_to_world([x, y_max, z], main_affine)
            ax_line_y_p1 = coordinates.voxel_to_world([x_min, y, z], main_affine)
            ax_line_y_p2 = coordinates.voxel_to_world([x_max, y, z], main_affine)

            # Coronal View (X-Z plane):
            co_line_x_p1 = coordinates.voxel_to_world([x, y, z_min], main_affine)
            co_line_x_p2 = coordinates.voxel_to_world([x, y, z_max], main_affine)
            co_line_z_p1 = coordinates.voxel_to_world([x_min, y, z], main_affine)
            co_line_z_p2 = coordinates.voxel_to_world([x_max, y, z], main_affine)

            # Sagittal View (Y-Z plane):
            sa_line_y_p1 = coordinates.voxel_to_world([x, y_min, z], main_affine)
            sa_line_y_p2 = coordinates.voxel_to_world([x, y_max, z], main_affine)
            sa_line_z_p1 = coordinates.voxel_to_world([x, y, z_min], main_affine)
            sa_line_z_p2 = coordinates.voxel_to_world([x, y, z_max], main_affine)

            # Check if actors need to be created
            if vp.axial_crosshair_actor is None:
                self._create_crosshair_actors()

            # Update the line source endpoints
            # Convert numpy arrays to tuples for VTK compatibility
            vp.crosshair_lines["ax_x"].SetPoint1(tuple(ax_line_x_p1))
            vp.crosshair_lines["ax_x"].SetPoint2(tuple(ax_line_x_p2))
            vp.crosshair_lines["ax_y"].SetPoint1(tuple(ax_line_y_p1))
            vp.crosshair_lines["ax_y"].SetPoint2(tuple(ax_line_y_p2))

            vp.crosshair_lines["co_x"].SetPoint1(tuple(co_line_x_p1))
            vp.crosshair_lines["co_x"].SetPoint2(tuple(co_line_x_p2))
            vp.crosshair_lines["co_z"].SetPoint1(tuple(co_line_z_p1))
            vp.crosshair_lines["co_z"].SetPoint2(tuple(co_line_z_p2))

            vp.crosshair_lines["sa_y"].SetPoint1(tuple(sa_line_y_p1))
            vp.crosshair_lines["sa_y"].SetPoint2(tuple(sa_line_y_p2))
            vp.crosshair_lines["sa_z"].SetPoint1(tuple(sa_line_z_p1))
            vp.crosshair_lines["sa_z"].SetPoint2(tuple(sa_line_z_p2))

            # Force pipeline update and mark actors as modified
            for key in ["ax_x", "ax_y", "co_x", "co_z", "sa_y", "sa_z"]:
                vp.crosshair_lines[key].Update()
            for key in ["axial", "coronal", "sagittal"]:
                vp.crosshair_appenders[key].Update()

            if vp.axial_crosshair_actor:
                vp.axial_crosshair_actor.Modified()
            if vp.coronal_crosshair_actor:
                vp.coronal_crosshair_actor.Modified()
            if vp.sagittal_crosshair_actor:
                vp.sagittal_crosshair_actor.Modified()

        except Exception as e:
            logger.error(f"Error creating/updating crosshairs:", exc_info=True)
            self.clear_crosshairs()

    def _create_crosshair_actors(self) -> None:
        """Creates the crosshair actors for all three views."""
        vp = self.vtk_panel

        # --- Axial ---
        vp.crosshair_lines["ax_x"] = vtk.vtkLineSource()
        vp.crosshair_lines["ax_y"] = vtk.vtkLineSource()
        vp.crosshair_appenders["axial"] = vtk.vtkAppendPolyData()
        vp.crosshair_appenders["axial"].AddInputConnection(
            vp.crosshair_lines["ax_x"].GetOutputPort()
        )
        vp.crosshair_appenders["axial"].AddInputConnection(
            vp.crosshair_lines["ax_y"].GetOutputPort()
        )
        ax_mapper = vtk.vtkPolyDataMapper()
        ax_mapper.SetInputConnection(vp.crosshair_appenders["axial"].GetOutputPort())
        ax_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        ax_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
        vp.axial_crosshair_actor = vtk.vtkActor()
        vp.axial_crosshair_actor.SetMapper(ax_mapper)
        vp.axial_crosshair_actor.GetProperty().SetColor(1, 1, 0)  # Yellow
        vp.axial_crosshair_actor.GetProperty().SetLineWidth(1.0)
        vp.axial_crosshair_actor.GetProperty().SetOpacity(0.8)
        vp.axial_scene.add(vp.axial_crosshair_actor)

        # --- Coronal ---
        vp.crosshair_lines["co_x"] = vtk.vtkLineSource()
        vp.crosshair_lines["co_z"] = vtk.vtkLineSource()
        vp.crosshair_appenders["coronal"] = vtk.vtkAppendPolyData()
        vp.crosshair_appenders["coronal"].AddInputConnection(
            vp.crosshair_lines["co_x"].GetOutputPort()
        )
        vp.crosshair_appenders["coronal"].AddInputConnection(
            vp.crosshair_lines["co_z"].GetOutputPort()
        )
        co_mapper = vtk.vtkPolyDataMapper()
        co_mapper.SetInputConnection(vp.crosshair_appenders["coronal"].GetOutputPort())
        co_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        co_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
        vp.coronal_crosshair_actor = vtk.vtkActor()
        vp.coronal_crosshair_actor.SetMapper(co_mapper)
        vp.coronal_crosshair_actor.GetProperty().SetColor(1, 1, 0)
        vp.coronal_crosshair_actor.GetProperty().SetLineWidth(1.0)
        vp.coronal_crosshair_actor.GetProperty().SetOpacity(0.8)
        vp.coronal_scene.add(vp.coronal_crosshair_actor)

        # --- Sagittal ---
        vp.crosshair_lines["sa_y"] = vtk.vtkLineSource()
        vp.crosshair_lines["sa_z"] = vtk.vtkLineSource()
        vp.crosshair_appenders["sagittal"] = vtk.vtkAppendPolyData()
        vp.crosshair_appenders["sagittal"].AddInputConnection(
            vp.crosshair_lines["sa_y"].GetOutputPort()
        )
        vp.crosshair_appenders["sagittal"].AddInputConnection(
            vp.crosshair_lines["sa_z"].GetOutputPort()
        )
        sa_mapper = vtk.vtkPolyDataMapper()
        sa_mapper.SetInputConnection(vp.crosshair_appenders["sagittal"].GetOutputPort())
        vp.sagittal_crosshair_actor = vtk.vtkActor()
        vp.sagittal_crosshair_actor.SetMapper(sa_mapper)
        vp.sagittal_crosshair_actor.GetProperty().SetColor(1, 1, 0)
        vp.sagittal_crosshair_actor.GetProperty().SetLineWidth(1.0)
        vp.sagittal_crosshair_actor.GetProperty().SetOpacity(0.8)
        # Disable depth testing so crosshair always renders on top
        vp.sagittal_crosshair_actor.GetProperty().SetLighting(False)
        vp.sagittal_overlay_renderer.AddActor(vp.sagittal_crosshair_actor)

    # =========================================================================
    # Scene UI Creation
    # =========================================================================

    def create_scene_ui(self) -> None:
        """Creates the initial UI elements (Axes, Text) within the VTK scene."""
        vp = self.vtk_panel

        # Status Text Actor
        vp.status_text_actor = actors.create_status_text_actor()
        vp.scene.add(vp.status_text_actor)

        # Instruction Text Actor
        vp.instruction_text_actor = actors.create_instruction_text_actor()
        vp.scene.add(vp.instruction_text_actor)

        # Axes Actor
        vp.axes_actor = actors.create_axes_actor()
        vp.scene.add(vp.axes_actor)

        # 3D Orientation Cube
        try:
            cube_actor = actors.create_orientation_cube()
            vp.orientation_widget = actors.create_orientation_widget(
                cube_actor, vp.interactor
            )
        except Exception as e:
            logger.warning(f"Could not create 3D orientation cube: {e}")

        # 2D Orientation Labels
        self.create_2d_orientation_labels()

    def create_2d_orientation_labels(self) -> None:
        """Creates the A/P/L/R/S/I labels for the 2D ortho views."""
        vp = self.vtk_panel

        # Clear any existing labels
        for label_actor in vp.orientation_labels:
            if vp.axial_scene:
                try:
                    vp.axial_scene.rm(label_actor)
                except Exception:
                    pass
            if vp.coronal_scene:
                try:
                    vp.coronal_scene.rm(label_actor)
                except Exception:
                    pass
            if vp.sagittal_scene:
                try:
                    vp.sagittal_scene.rm(label_actor)
                except Exception:
                    pass
        vp.orientation_labels = []

        # Axial View (A/P, R/L)
        self._add_2d_label("A", vp.axial_scene, (0.5, 0.98), v_align="Top")
        self._add_2d_label("P", vp.axial_scene, (0.5, 0.02), v_align="Bottom")
        self._add_2d_label("L", vp.axial_scene, (0.98, 0.5), h_align="Right")
        self._add_2d_label("R", vp.axial_scene, (0.02, 0.5), h_align="Left")

        # Coronal View (S/I, R/L)
        self._add_2d_label("S", vp.coronal_scene, (0.5, 0.98), v_align="Top")
        self._add_2d_label("I", vp.coronal_scene, (0.5, 0.02), v_align="Bottom")
        self._add_2d_label("L", vp.coronal_scene, (0.98, 0.5), h_align="Right")
        self._add_2d_label("R", vp.coronal_scene, (0.02, 0.5), h_align="Left")

        # Sagittal View (S/I, A/P)
        self._add_2d_label("S", vp.sagittal_scene, (0.5, 0.98), v_align="Top")
        self._add_2d_label("I", vp.sagittal_scene, (0.5, 0.02), v_align="Bottom")
        self._add_2d_label("A", vp.sagittal_scene, (0.98, 0.5), h_align="Right")
        self._add_2d_label("P", vp.sagittal_scene, (0.02, 0.5), h_align="Left")

    def _add_2d_label(
        self,
        text: str,
        scene: window.Scene,
        pos: Tuple[float, float],
        h_align: str = "Center",
        v_align: str = "Center",
    ) -> None:
        """
        Helper function to create and add a single 2D text label.

        Args:
            text: Label text.
            scene: FURY scene to add the label to.
            pos: Normalized display position.
            h_align: Horizontal alignment.
            v_align: Vertical alignment.
        """
        text_actor = actors.create_2d_label_actor(
            text, pos, h_align=h_align, v_align=v_align
        )
        scene.add(text_actor)
        self.vtk_panel.orientation_labels.append(text_actor)
