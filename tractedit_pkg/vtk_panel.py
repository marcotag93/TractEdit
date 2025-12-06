# -*- coding: utf-8 -*-

"""
Manages the VTK scene, actors, and interactions for TractEdit.
Handles display of streamlines and anatomical image slices using a custom interactor style.
"""

import os
import numpy as np
import vtk
import logging
from PyQt6.QtWidgets import (
    QVBoxLayout, QMessageBox, QApplication, QFileDialog, 
    QWidget, QHBoxLayout, QSplitter, QMenu
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QAction
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from fury import window, actor, colormap
from .utils import ColorMode
from typing import Optional, List, Dict, Any, Tuple, Set
import nibabel as nib
from functools import partial

logger = logging.getLogger(__name__)

class CustomInteractorStyle2D(vtk.vtkInteractorStyleImage):
    """
    Custom interactor style for the 2D views.
    
    Overrides default vtkInteractorStyleImage behavior:
    - Left-click-drag: Navigates slices.
    - Right-click-drag: Zooms (default).
    - Middle-click-drag: Pans (default).
    """
    
    def __init__(self, vtk_panel_ref: 'VTKPanel') -> None:
        """
        Args:
            vtk_panel_ref: A reference to the main VTKPanel instance.
        """
        super().__init__()
        self.vtk_panel: 'VTKPanel' = vtk_panel_ref
        
        # Add event listeners for default VTK events
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.OnLeftButtonDown)
        self.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, self.OnLeftButtonUp)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.OnMouseMove)

    def OnLeftButtonDown(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles left mouse button press."""
        # Call the VTKPanel's navigation logic
        if self.vtk_panel:
            self.vtk_panel.is_navigating_2d = True
            self.vtk_panel._navigate_2d_view(self.GetInteractor(), event_id)       

    def OnLeftButtonUp(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles left mouse button release."""
        if self.vtk_panel:
            self.vtk_panel.is_navigating_2d = False
            self.vtk_panel._update_slow_slice_components()
            
    def OnMouseMove(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles mouse move."""
        if self.vtk_panel and self.vtk_panel.is_navigating_2d:
            # If we are in navigation mode, call the logic
            self.vtk_panel._navigate_2d_view(self.GetInteractor(), event_id)
        else:
            super().OnMouseMove()
            
class VTKPanel:
    """
    Manages the VTK rendering window, scene, actors (streamlines, image slices),
    and interactions.
    """
    def __init__(self, parent_widget: QWidget, main_window_ref: Any) -> None:
        """
        Initializes the VTK panel.

        Args:
            parent_widget: The PyQt widget this panel will reside in.
            main_window_ref: A reference to the main MainWindow instance.
        """
        self.main_window: Any = main_window_ref

        # 1. Create the main vertical splitter (3D view on top, 2D views on bottom)
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # 2. Create the main 3D VTK widget
        self.vtk_widget: QVTKRenderWindowInteractor = QVTKRenderWindowInteractor()
        main_splitter.addWidget(self.vtk_widget) # Add to top

        # 3. Create the bottom container for 2D views
        ortho_container = QWidget()
        ortho_layout = QHBoxLayout(ortho_container)
        ortho_layout.setContentsMargins(0, 0, 0, 0)
        
        # 4. Create the horizontal splitter for the three 2D views
        ortho_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.axial_vtk_widget: QVTKRenderWindowInteractor = QVTKRenderWindowInteractor()
        self.coronal_vtk_widget: QVTKRenderWindowInteractor = QVTKRenderWindowInteractor()
        self.sagittal_vtk_widget: QVTKRenderWindowInteractor = QVTKRenderWindowInteractor()
        
        self.axial_vtk_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.coronal_vtk_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sagittal_vtk_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.axial_vtk_widget.customContextMenuRequested.connect(
            partial(self._show_2d_context_menu, widget=self.axial_vtk_widget, view_type='axial')
        )
        self.coronal_vtk_widget.customContextMenuRequested.connect(
            partial(self._show_2d_context_menu, widget=self.coronal_vtk_widget, view_type='coronal')
        )
        self.sagittal_vtk_widget.customContextMenuRequested.connect(
            partial(self._show_2d_context_menu, widget=self.sagittal_vtk_widget, view_type='sagittal')
        )

        ortho_splitter.addWidget(self.axial_vtk_widget)
        ortho_splitter.addWidget(self.coronal_vtk_widget)
        ortho_splitter.addWidget(self.sagittal_vtk_widget)
        
        ortho_layout.addWidget(ortho_splitter)
        ortho_container.setLayout(ortho_layout)
        
        # 5. Add the 2D view container to the bottom 
        main_splitter.addWidget(ortho_container)
        
        # 6. Set initial size ratio (e.g., 70% 3D, 30% 2D)
        main_splitter.setSizes([700, 300])

        # 7. Add the main splitter to the parent widget's layout
        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter)
        parent_widget.setLayout(layout)

        # --- FURY/VTK Scene Setup ---
        # Main 3D Scene
        self.scene: window.Scene = window.Scene()
        self.scene.background((0.1, 0.1, 0.1))
        self.render_window: vtk.vtkRenderWindow = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.scene)
        self.interactor: vtk.vtkRenderWindowInteractor = self.render_window.GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Axial 2D Scene
        self.axial_scene: window.Scene = window.Scene()
        self.axial_scene.background((0.0, 0.0, 0.0))
        self.axial_render_window: vtk.vtkRenderWindow = self.axial_vtk_widget.GetRenderWindow()
        self.axial_render_window.AddRenderer(self.axial_scene)
        self.axial_interactor: vtk.vtkRenderWindowInteractor = self.axial_render_window.GetInteractor()
        style_2d_axial = CustomInteractorStyle2D(self)
        self.axial_interactor.SetInteractorStyle(style_2d_axial)
        
        # Coronal 2D Scene
        self.coronal_scene: window.Scene = window.Scene()
        self.coronal_scene.background((0.0, 0.0, 0.0))
        self.coronal_render_window: vtk.vtkRenderWindow = self.coronal_vtk_widget.GetRenderWindow()
        self.coronal_render_window.AddRenderer(self.coronal_scene)
        self.coronal_interactor: vtk.vtkRenderWindowInteractor = self.coronal_render_window.GetInteractor()
        style_2d_coronal = CustomInteractorStyle2D(self)
        self.coronal_interactor.SetInteractorStyle(style_2d_coronal)

        # Sagittal 2D Scene
        self.sagittal_scene: window.Scene = window.Scene()
        self.sagittal_scene.background((0.0, 0.0, 0.0))
        self.sagittal_render_window: vtk.vtkRenderWindow = self.sagittal_vtk_widget.GetRenderWindow()
        self.sagittal_render_window.SetNumberOfLayers(2)  # Enable multiple layers
        self.sagittal_render_window.AddRenderer(self.sagittal_scene)
        self.sagittal_scene.SetLayer(0)  # Base layer for slice
        
        # Create overlay renderer for sagittal crosshair (renders on top)
        self.sagittal_overlay_renderer = vtk.vtkRenderer()
        self.sagittal_overlay_renderer.SetLayer(1)  # Overlay layer
        self.sagittal_overlay_renderer.InteractiveOff()  # Don't intercept mouse events
        self.sagittal_overlay_renderer.PreserveColorBufferOn()  # Keep background from layer 0
        self.sagittal_render_window.AddRenderer(self.sagittal_overlay_renderer)
        
        self.sagittal_interactor: vtk.vtkRenderWindowInteractor = self.sagittal_render_window.GetInteractor()
        style_2d_sagittal = CustomInteractorStyle2D(self)
        self.sagittal_interactor.SetInteractorStyle(style_2d_sagittal)

        # --- Actor Placeholders ---
        self.streamlines_actor: Optional[vtk.vtkActor] = None
        self.highlight_actor: Optional[vtk.vtkActor] = None
        self.radius_actor: Optional[vtk.vtkActor] = None
        self.current_radius_actor_radius: Optional[float] = None
        self.axial_slice_actor: Optional[vtk.vtkActor] = None
        self.coronal_slice_actor: Optional[vtk.vtkActor] = None
        self.sagittal_slice_actor: Optional[vtk.vtkActor] = None
        self.status_text_actor: Optional[vtk.vtkTextActor] = None
        self.instruction_text_actor: Optional[vtk.vtkTextActor] = None
        self.axes_actor: Optional[vtk.vtkActor] = None
        self.orientation_widget: Optional[vtk.vtkOrientationMarkerWidget] = None
        self.orientation_labels: List[vtk.vtkTextActor] = [] 
        
        # --- Crosshair Actors ---
        self.axial_crosshair_actor: Optional[vtk.vtkActor] = None
        self.coronal_crosshair_actor: Optional[vtk.vtkActor] = None
        self.sagittal_crosshair_actor: Optional[vtk.vtkActor] = None
        self.crosshair_lines: Dict[str, vtk.vtkLineSource] = {} # Stores vtkLineSource objects for updates
        self.crosshair_appenders: Dict[str, vtk.vtkAppendPolyData] = {} # Stores vtkAppendPolyData 
        
        self.is_navigating_2d: bool = False
        
        self.roi_slice_actors: Dict[str, Dict[str, vtk.vtkActor]] = {} # Key: path, Val: {'axial':, 'coronal':, 'sagittal':}
        
        self.roi_highlight_actor: Optional[vtk.vtkActor] = None 
        
        # --- Slice Navigation State ---
        self.current_slice_indices: Dict[str, Optional[int]] = {'x': None, 'y': None, 'z': None}
        self.image_shape_vox: Optional[Tuple[int, ...]] = None
        self.image_extents: Dict[str, Optional[Tuple[int, int]]] = {'x': None, 'y': None, 'z': None}

        self._create_scene_ui()

        # --- Setup Interaction Callbacks ---
        # KeyPress for all windows
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0)
        self.axial_interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0)
        self.coronal_interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0)
        self.sagittal_interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0)

        # --- Initialize VTK Widget ---
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        
        self.axial_vtk_widget.Initialize()
        self.axial_vtk_widget.Start()
        
        self.coronal_vtk_widget.Initialize()
        self.coronal_vtk_widget.Start()
        
        self.sagittal_vtk_widget.Initialize()
        self.sagittal_vtk_widget.Start()
        
        self.scene.reset_camera()
        
        self._render_all() 
        
    def set_anatomical_slice_visibility(self, visible: bool) -> None:
        """Sets visibility for anatomical slices and associated crosshairs."""
        vis_flag = 1 if visible else 0
        
        # 1. Update Slice Actors
        for act in [self.axial_slice_actor, self.coronal_slice_actor, self.sagittal_slice_actor]:
            if act: act.SetVisibility(vis_flag)
            
        # 2. Update Crosshair Actors
        for act in [self.axial_crosshair_actor, self.coronal_crosshair_actor, self.sagittal_crosshair_actor]:
            if act: act.SetVisibility(vis_flag)
            
        self._render_all()

    def update_roi_highlight_actor(self) -> None:
        """
        Updates the Red highlight actor based on ROI 'Select' indices.
        """
        if not self.scene: return

        # Remove existing
        if self.roi_highlight_actor:
            try: self.scene.rm(self.roi_highlight_actor)
            except: pass
            self.roi_highlight_actor = None
            
        if not self.main_window or not self.main_window.tractogram_data: return
        
        indices = getattr(self.main_window, 'roi_highlight_indices', set())
        if not indices: return
        
        # Get actual streamline data
        tractogram = self.main_window.tractogram_data
        streamlines_list = []
        for idx in indices:
            try:
                if idx < len(tractogram):
                    sl = tractogram[idx]
                    if len(sl) > 0:
                        streamlines_list.append(sl)
            except: pass
            
        if not streamlines_list: return
        
        try:
            # Create Red Actor (1, 0, 0)
            self.roi_highlight_actor = actor.line(
                streamlines_list,
                colors=(1, 0, 0),
                linewidth=4,
                opacity=1.0,
                depth_cue=False # Make it pop out
            )
            self.scene.add(self.roi_highlight_actor)
            
            # Use mapper offset to ensure it draws ON TOP of the grey/yellow bundles
            mapper = self.roi_highlight_actor.GetMapper()
            if mapper:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -20) # Stronger offset than yellow
                
            self.render_window.Render()
            
        except Exception as e:
            logger.error(f"Error creating ROI highlight actor: {e}")

    def _render_all(self) -> None:
        """Renders all four VTK windows if they are initialized."""
        try:
            if self.render_window and self.render_window.GetInteractor().GetInitialized():
                self.render_window.Render()
            if self.axial_render_window and self.axial_render_window.GetInteractor().GetInitialized():
                self.axial_render_window.Render()
            if self.coronal_render_window and self.coronal_render_window.GetInteractor().GetInitialized():
                self.coronal_render_window.Render()
            if self.sagittal_render_window and self.sagittal_render_window.GetInteractor().GetInitialized():
                self.sagittal_render_window.Render()
        except Exception as e:
            logger.warning(f"Warning: Error during _render_all: {e}")
            
    def _show_2d_context_menu(self, position: QPoint, widget: QVTKRenderWindowInteractor, view_type: str) -> None:
        """
        Shows a context menu for the 2D view that was right-clicked.
        
        Args:
            position: The QPoint where the click occurred (local to the widget).
            widget: The QVTKRenderWindowInteractor that was clicked.
            view_type: 'axial', 'coronal', or 'sagittal'.
        """
        # Only show the menu if an anatomical image is loaded
        if not self.main_window or self.main_window.anatomical_image_data is None:
            return
            
        menu = QMenu(widget)
        
        # Create the "Reset View" action
        reset_action = QAction("Reset View", widget)
        
        # Connect the action to our new reset helper function
        reset_action.triggered.connect(partial(self._reset_2d_view, view_type=view_type))
        
        menu.addAction(reset_action)
        
        # Show the menu at the global position of the click
        menu.exec(widget.mapToGlobal(position))

    def _reset_2d_view(self, view_type: str) -> None:
        """Resets the camera for a specific 2D view and re-renders it."""
        try:
            if view_type == 'axial':
                self._update_axial_camera(reset_zoom_pan=True)
                if self.axial_render_window: self.axial_render_window.Render()
                    
            elif view_type == 'coronal':
                self._update_coronal_camera(reset_zoom_pan=True)
                if self.coronal_render_window: self.coronal_render_window.Render()
                    
            elif view_type == 'sagittal':
                self._update_sagittal_camera(reset_zoom_pan=True)
                if self.sagittal_render_window: self.sagittal_render_window.Render()
                
        except Exception as e:
            logger.error(f"Error resetting 2D view ({view_type}): {e}")
                        
    def _setup_ortho_cameras(self) -> None:
        """Sets the cameras for the 2D views to be orthogonal."""
        if not self.axial_scene or not self.coronal_scene or not self.sagittal_scene:
            return
            
        try:
            # Call with reset=True for initial setup
            self._update_axial_camera(reset_zoom_pan=True)
            self._update_coronal_camera(reset_zoom_pan=True)
            self._update_sagittal_camera(reset_zoom_pan=True)
        except Exception as e:
            logger.error(f"Error setting up ortho cameras: {e}")
            
    def _update_axial_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the axial 2D camera to follow its slice."""
        if not self.axial_scene: return
        try:
            cam = self.axial_scene.GetActiveCamera()
            if not cam.GetParallelProjection(): cam.SetParallelProjection(1)

            if reset_zoom_pan or not self.axial_slice_actor:
                # Full reset (initial setup or fallback)
                self.axial_scene.reset_camera()
                if not cam.GetParallelProjection(): cam.SetParallelProjection(1)
                fp = cam.GetFocalPoint()
                dist = cam.GetDistance()
                cam.SetPosition(fp[0], fp[1], fp[2] + dist) # View from +Z
                cam.SetFocalPoint(fp[0], fp[1], fp[2])
                cam.SetViewUp(0, 1, 0)
            else:
                # Just follow the slice in Z, keeping X,Y pan and zoom
                actor_center = self.axial_slice_actor.GetCenter()
                
                current_fp = cam.GetFocalPoint()
                current_pos = cam.GetPosition()
                
                new_focal_z = actor_center[2]
                z_delta = new_focal_z - current_fp[2]

                # Only update if the slice has actually moved in Z
                if abs(z_delta) > 1e-6:
                    cam.SetFocalPoint(current_fp[0], current_fp[1], new_focal_z)
                    cam.SetPosition(current_pos[0], current_pos[1], current_pos[2] + z_delta)
            
            self.axial_scene.reset_clipping_range()
        except Exception as e:
            logger.error(f"Error updating axial camera: {e}")

    def _update_coronal_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the coronal 2D camera to follow its slice."""
        if not self.coronal_scene: return
        try:
            cam = self.coronal_scene.GetActiveCamera()
            if not cam.GetParallelProjection(): cam.SetParallelProjection(1)

            if reset_zoom_pan or not self.coronal_slice_actor:
                # Full reset
                self.coronal_scene.reset_camera()
                if not cam.GetParallelProjection(): cam.SetParallelProjection(1)
                fp = cam.GetFocalPoint()
                dist = cam.GetDistance()
                cam.SetPosition(fp[0], fp[1] - dist, fp[2]) # View from -Y
                cam.SetFocalPoint(fp[0], fp[1], fp[2])
                cam.SetViewUp(0, 0, 1) # Up is Z
            else:
                # Just follow the slice in Y
                actor_center = self.coronal_slice_actor.GetCenter()
                
                current_fp = cam.GetFocalPoint()
                current_pos = cam.GetPosition()
                
                new_focal_y = actor_center[1]
                y_delta = new_focal_y - current_fp[1]

                if abs(y_delta) > 1e-6:
                    cam.SetFocalPoint(current_fp[0], new_focal_y, current_fp[2])
                    cam.SetPosition(current_pos[0], current_pos[1] + y_delta, current_pos[2])
            
            self.coronal_scene.reset_clipping_range()
        except Exception as e:
            logger.error(f"Error updating coronal camera: {e}")

    def _update_sagittal_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the sagittal 2D camera to follow its slice."""
        if not self.sagittal_scene: return
        try:
            cam = self.sagittal_scene.GetActiveCamera()
            if not cam.GetParallelProjection(): cam.SetParallelProjection(1)
            
            if reset_zoom_pan or not self.sagittal_slice_actor:
                self.sagittal_scene.reset_camera()
                if not cam.GetParallelProjection(): cam.SetParallelProjection(1)
                fp = cam.GetFocalPoint()
                dist = cam.GetDistance()
                cam.SetPosition(fp[0] + dist, fp[1], fp[2]) # Was fp[0] - dist
                cam.SetFocalPoint(fp[0], fp[1], fp[2])
                cam.SetViewUp(0, 0, 1) # Up is Z
            else:
                actor_center = self.sagittal_slice_actor.GetCenter()
                
                current_fp = cam.GetFocalPoint()
                current_pos = cam.GetPosition()
                
                new_focal_x = actor_center[0]
                x_delta = new_focal_x - current_fp[0]

                if abs(x_delta) > 1e-6:
                    cam.SetFocalPoint(new_focal_x, current_fp[1], current_fp[2])
                    cam.SetPosition(current_pos[0] + x_delta, current_pos[1], current_pos[2])
            
            self.sagittal_scene.reset_clipping_range()
            
            # Expand the clipping range to ensure the crosshair in the overlay is never clipped
            # The crosshair sits at the same depth as the slice and can get clipped at edges
            near, far = cam.GetClippingRange()
            cam.SetClippingRange(near * 0.01, far * 100) 
            
            # Don't reset clipping range on overlay - it shares the camera and would override main scene's clipping
            if hasattr(self, 'sagittal_overlay_renderer') and self.sagittal_overlay_renderer:
                self.sagittal_overlay_renderer.SetActiveCamera(cam)
        except Exception as e:
            logger.error(f"Error updating sagittal camera: {e}")
            
    def _create_2d_label_actor(self, text: str, scene: window.Scene, pos: Tuple[float, float], h_align: str = "Center", v_align: str = "Center") -> None:
        """Helper function to create a single 2D text label."""
        text_prop = vtk.vtkTextProperty()
        text_prop.SetFontSize(14)
        text_prop.SetColor(0.8, 0.8, 0.8)
        text_prop.SetFontFamilyToArial()
        text_prop.BoldOff()
        text_prop.ShadowOff()

        # Set Horizontal Alignment
        if h_align == "Left": text_prop.SetJustificationToLeft()
        elif h_align == "Center": text_prop.SetJustificationToCentered()
        elif h_align == "Right": text_prop.SetJustificationToRight()

        # Set Vertical Alignment
        if v_align == "Top": text_prop.SetVerticalJustificationToTop()
        elif v_align == "Center": text_prop.SetVerticalJustificationToCentered()
        elif v_align == "Bottom": text_prop.SetVerticalJustificationToBottom()

        text_actor = vtk.vtkTextActor()
        text_actor.SetTextProperty(text_prop)
        text_actor.SetInput(text)
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        text_actor.GetPositionCoordinate().SetValue(pos[0], pos[1])
        
        scene.add(text_actor)
        self.orientation_labels.append(text_actor) 
        
    def _create_2d_orientation_labels(self) -> None:
        """Creates the A/P/L/R/S/I labels for the 2D ortho views."""
        
        # Clear any existing labels
        for actor in self.orientation_labels:
            if self.axial_scene: self.axial_scene.rm(actor)
            if self.coronal_scene: self.coronal_scene.rm(actor)
            if self.sagittal_scene: self.sagittal_scene.rm(actor)
        self.orientation_labels = []
        
    # --- Axial View (A/P, R/L) ---
        self._create_2d_label_actor("A", self.axial_scene, (0.5, 0.98), v_align="Top")
        self._create_2d_label_actor("P", self.axial_scene, (0.5, 0.02), v_align="Bottom")
        self._create_2d_label_actor("L", self.axial_scene, (0.98, 0.5), h_align="Right") 
        self._create_2d_label_actor("R", self.axial_scene, (0.02, 0.5), h_align="Left") 

        # --- Coronal View (S/I, R/L) ---
        self._create_2d_label_actor("S", self.coronal_scene, (0.5, 0.98), v_align="Top")
        self._create_2d_label_actor("I", self.coronal_scene, (0.5, 0.02), v_align="Bottom")
        self._create_2d_label_actor("L", self.coronal_scene, (0.98, 0.5), h_align="Right") 
        self._create_2d_label_actor("R", self.coronal_scene, (0.02, 0.5), h_align="Left")  
     
        # --- Sagittal View (S/I, A/P) ---
        self._create_2d_label_actor("S", self.sagittal_scene, (0.5, 0.98), v_align="Top")
        self._create_2d_label_actor("I", self.sagittal_scene, (0.5, 0.02), v_align="Bottom")
        self._create_2d_label_actor("A", self.sagittal_scene, (0.98, 0.5), h_align="Right")
        self._create_2d_label_actor("P", self.sagittal_scene, (0.02, 0.5), h_align="Left")

    def _create_scene_ui(self) -> None:
        """Creates the initial UI elements (Axes, Text) within the VTK scene."""
        status_prop = vtk.vtkTextProperty()
        status_prop.SetFontSize(14)
        status_prop.SetColor(0.95, 0.95, 0.95)
        status_prop.SetFontFamilyToArial()
        status_prop.SetJustificationToLeft()
        status_prop.SetVerticalJustificationToBottom()

        self.status_text_actor = vtk.vtkTextActor()
        self.status_text_actor.SetTextProperty(status_prop)
        self.status_text_actor.SetInput("Status: Initializing...")
        self.status_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self.status_text_actor.GetPositionCoordinate().SetValue(0.01, 0.01)
        self.scene.add(self.status_text_actor)

        instr_prop = vtk.vtkTextProperty()
        instr_prop.SetFontSize(12)
        instr_prop.SetColor(0.8, 0.8, 0.8)
        instr_prop.SetFontFamilyToArial()
        instr_prop.SetJustificationToLeft()
        instr_prop.SetVerticalJustificationToBottom()
        
        self.instruction_text_actor = vtk.vtkTextActor()
        self.instruction_text_actor.SetTextProperty(instr_prop)
        instruction_text = (
            "Selection: [S] Select | [D] Del | [C] Clear | [+/-] Radius | [Esc] Hide Sphere\n"
            "File/Edit: [Ctrl+S] Save | [Ctrl+Z] Undo | [Ctrl+Y] Redo"
        )
        
        self.instruction_text_actor.SetInput(instruction_text)
        self.instruction_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self.instruction_text_actor.GetPositionCoordinate().SetValue(0.01, 0.05)
        self.scene.add(self.instruction_text_actor) 
        
        self.axes_actor = actor.axes(scale=(25, 25, 25))
        self.scene.add(self.axes_actor)

        # --- Add 3D Orientation Cube ---
        try:
            cube_actor = vtk.vtkAnnotatedCubeActor()
            cube_actor.SetXPlusFaceText("R")  # Right
            cube_actor.SetXMinusFaceText("L") # Left
            cube_actor.SetYPlusFaceText("A")  # Anterior
            cube_actor.SetYMinusFaceText("P") # Posterior
            cube_actor.SetZPlusFaceText("S")  # Superior
            cube_actor.SetZMinusFaceText("I") # Inferior
            
            # Configure text properties
            prop = cube_actor.GetTextEdgesProperty()
            prop.SetColor(0.9, 0.9, 0.9)
            prop.SetLineWidth(1.0)
            
            # Configure cube face properties
            cube_actor.GetCubeProperty().SetColor(0.3, 0.3, 0.3)
            
            # Configure face text properties
            face_props = [
                cube_actor.GetXPlusFaceProperty(), cube_actor.GetXMinusFaceProperty(),
                cube_actor.GetYPlusFaceProperty(), cube_actor.GetYMinusFaceProperty(),
                cube_actor.GetZPlusFaceProperty(), cube_actor.GetZMinusFaceProperty()
            ]
            for face_prop in face_props:
                face_prop.SetColor(0.9, 0.9, 0.9)
                face_prop.SetInterpolationToFlat()
            
            self.orientation_widget = vtk.vtkOrientationMarkerWidget()
            self.orientation_widget.SetOrientationMarker(cube_actor)
            self.orientation_widget.SetInteractor(self.interactor)
            self.orientation_widget.SetViewport(0.85, 0.0, 1.0, 0.15)
            self.orientation_widget.SetEnabled(1)
            self.orientation_widget.InteractiveOff() 

        except Exception as e:
            logger.warning(f"Could not create 3D orientation cube: {e}")

        # --- Add 2D Orientation Labels ---
        self._create_2d_orientation_labels()

    def update_status(self, message: str) -> None:
        """Updates the status text displayed in the VTK window."""
        if self.status_text_actor is None:
            return

        try:
            undo_possible = False
            redo_possible = False
            data_loaded = False
            current_radius = self.main_window.selection_radius_3d if self.main_window else 5.0

            if self.main_window:
                undo_possible = bool(self.main_window.undo_stack)
                redo_possible = bool(self.main_window.redo_stack)
                data_loaded = bool(self.main_window.tractogram_data)

            status_suffix = ""
            if "Deleted" in message and undo_possible: status_suffix = " (Ctrl+Z to Undo)"
            elif "Undo successful" in message:
                status_suffix = f" ({len(self.main_window.undo_stack)} undo remaining"
                status_suffix += ", Ctrl+Y to Redo)" if redo_possible else ")"
            elif "Redo successful" in message:
                status_suffix = f" ({len(self.main_window.redo_stack)} redo remaining"
                status_suffix += ", Ctrl+Z to Undo)" if undo_possible else ")"

            prefix = f"[Radius: {current_radius:.1f}mm] " if data_loaded else ""
            
            # Avoid prefixing slice navigation messages
            if message.startswith("Slice View:"):
                prefix = ""
                status_suffix = "" 
                
            full_message = f"Status: {prefix}{message}{status_suffix}"
            self.status_text_actor.SetInput(str(full_message))

            if self.render_window and self.render_window.GetInteractor().GetInitialized():
                self.render_window.Render()
        except Exception as e:
            logger.error(f"Error updating status text actor:", exc_info=e)

    # --- Selection Sphere Actor Management ---
    def _ensure_radius_actor_exists(self, radius: float, center_point: Tuple[float, float, float]) -> None:
        """Creates the VTK sphere actor if it doesn't exist."""
        if self.radius_actor is not None: 
            return

        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(radius)
        sphere_source.SetCenter(0, 0, 0)
        sphere_source.SetPhiResolution(16)
        sphere_source.SetThetaResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())
        self.radius_actor = vtk.vtkActor()
        self.radius_actor.SetMapper(mapper)
        self.radius_actor.SetPosition(center_point)
        prop = self.radius_actor.GetProperty()
        prop.SetColor(0.2, 0.5, 1.0)
        prop.SetOpacity(0.3)
        prop.SetRepresentationToWireframe()
        prop.SetLineWidth(1.0)
        self.scene.add(self.radius_actor)
        self.current_radius_actor_radius = radius
        self.radius_actor.SetVisibility(0)

    def _update_existing_radius_actor(self, center_point: Optional[Tuple[float, float, float]], radius: Optional[float], visible: bool) -> bool:
        """Updates properties of the existing VTK sphere actor."""
        if self.radius_actor is None: 
            return False

        needs_update = False
        current_visibility = self.radius_actor.GetVisibility()
        current_position = np.array(self.radius_actor.GetPosition())
        current_radius_val = self.current_radius_actor_radius

        if visible and not current_visibility:
            self.radius_actor.SetVisibility(1)
            needs_update = True
        elif not visible and current_visibility:
            self.radius_actor.SetVisibility(0)
            needs_update = True

        if visible:
            if center_point is not None:
                new_position = np.array(center_point)
                if not np.array_equal(current_position, new_position):
                    self.radius_actor.SetPosition(new_position)
                    needs_update = True
            if radius is not None and radius != current_radius_val:
                mapper = self.radius_actor.GetMapper()
                if mapper and mapper.GetInputConnection(0, 0):
                    source = mapper.GetInputConnection(0, 0).GetProducer()
                    if isinstance(source, vtk.vtkSphereSource):
                        source.SetRadius(radius)
                        self.current_radius_actor_radius = radius
                        needs_update = True

        return needs_update

    def update_radius_actor(self, center_point: Optional[Tuple[float, float, float]] = None, radius: Optional[float] = None, visible: bool = False) -> None:
        """Creates or updates the selection sphere actor (standard VTK)."""
        if not self.scene: 
            return
        if radius is None and self.main_window: radius = self.main_window.selection_radius_3d

        needs_render = False
        if visible and center_point is not None and radius is not None:
            if self.radius_actor is None:
                 self._ensure_radius_actor_exists(radius, center_point)
                 if self.radius_actor: 
                     self.radius_actor.SetVisibility(1 if visible else 0)
                     needs_render = True
            else:
                 needs_render = self._update_existing_radius_actor(center_point, radius, visible)
        elif self.radius_actor is not None:
             needs_render = self._update_existing_radius_actor(None, None, visible)

        if needs_render and self.render_window and self.render_window.GetInteractor().GetInitialized():
            self.render_window.Render()

    # --- Anatomical Slice Actor Management ---
    def update_anatomical_slices(self) -> None:
        """Creates or updates the FURY slicer actors for anatomical image."""
        if not self.scene: 
            return 
        if not self.main_window or self.main_window.anatomical_image_data is None:
            self.clear_anatomical_slices() 
            return

        image_data = self.main_window.anatomical_image_data
        affine = self.main_window.anatomical_image_affine

        # Basic checks
        if image_data is None or affine is None or image_data.ndim < 3:
            logger.error("Invalid image data or affine provided for slices.")
            self.clear_anatomical_slices()
            return

        # Ensure data is contiguous float32
        if not image_data.flags['C_CONTIGUOUS']:
            image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        else:
            image_data = image_data.astype(np.float32, copy=False)

        # Get dimensions and extents
        shape = image_data.shape[:3]
        
        self.image_shape_vox = shape
        self.image_extents['x'] = (0, shape[0] - 1)
        self.image_extents['y'] = (0, shape[1] - 1)
        self.image_extents['z'] = (0, shape[2] - 1)

        # Initialize or keep current slice indices
        if self.current_slice_indices['x'] is None or self.current_slice_indices['x'] > self.image_extents['x'][1]:
            self.current_slice_indices['x'] = shape[0] // 2
        if self.current_slice_indices['y'] is None or self.current_slice_indices['y'] > self.image_extents['y'][1]:
            self.current_slice_indices['y'] = shape[1] // 2
        if self.current_slice_indices['z'] is None or self.current_slice_indices['z'] > self.image_extents['z'][1]:
            self.current_slice_indices['z'] = shape[2] // 2
        
        current_x = self.current_slice_indices['x']
        current_y = self.current_slice_indices['y']
        current_z = self.current_slice_indices['z']
        x_extent = self.image_extents['x']
        y_extent = self.image_extents['y']
        z_extent = self.image_extents['z']

        # --- Clear existing slicer actors ---
        self.clear_anatomical_slices(reset_state=False) 

        try:
            # --- Determine Value Range ---
            img_min = np.min(image_data)
            img_max = np.max(image_data)

            if not np.isfinite(img_min) or not np.isfinite(img_max):
                logger.warning("Image contains non-finite values. Clamping range.")
                finite_data = image_data[np.isfinite(image_data)]
                if finite_data.size > 0: img_min, img_max = np.min(finite_data), np.max(finite_data)
                else: img_min, img_max = 0.0, 1.0

            # Ensure range has distinct min/max
            if img_max <= img_min: value_range = (img_min - 1.0, img_max + 1.0)
            else: value_range = (float(img_min), float(img_max))

            # Slicer parameters
            slicer_opacity = 1.0 
            interpolation_mode = 'nearest' 

            # --- Create Slicer Actors using FURY's default gray LUT ---
            # 1. Axial Slice (Z plane)
            self.axial_slice_actor = actor.slicer(
                image_data, affine=affine,
                value_range=value_range,
                opacity=slicer_opacity, interpolation=interpolation_mode
            )
            self.axial_slice_actor.display_extent(x_extent[0], x_extent[1], y_extent[0], y_extent[1], current_z, current_z)
            self.scene.add(self.axial_slice_actor)
            if self.axial_scene: self.axial_scene.add(self.axial_slice_actor)

            # 2. Coronal Slice (Y plane)
            self.coronal_slice_actor = actor.slicer(
                image_data, affine=affine,
                value_range=value_range,
                opacity=slicer_opacity, interpolation=interpolation_mode
            )
            self.coronal_slice_actor.display_extent(x_extent[0], x_extent[1], current_y, current_y, z_extent[0], z_extent[1])
            self.scene.add(self.coronal_slice_actor)
            if self.coronal_scene: self.coronal_scene.add(self.coronal_slice_actor)

            # 3. Sagittal Slice (X plane)
            self.sagittal_slice_actor = actor.slicer(
                image_data, affine=affine,
                value_range=value_range,
                opacity=slicer_opacity, interpolation=interpolation_mode
            )
            self.sagittal_slice_actor.display_extent(current_x, current_x, y_extent[0], y_extent[1], z_extent[0], z_extent[1])
            self.scene.add(self.sagittal_slice_actor)
            if self.sagittal_scene: self.sagittal_scene.add(self.sagittal_slice_actor)
            
            # --- Create crosshair actors ---
            self._create_or_update_crosshairs()

        except TypeError as te:
             error_msg = f"TypeError during slice actor creation/display: {te}"
             logger.error(error_msg, exc_info=True)
             QMessageBox.critical(self.main_window, "Slice Actor TypeError", error_msg)
             self.clear_anatomical_slices()
        except Exception as e:
            error_msg = f"Error during anatomical slice actor creation/addition: {e}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self.main_window, "Slice Actor Error", error_msg)
            self.clear_anatomical_slices()
            
        self._setup_ortho_cameras()

        # Final render
        self._render_all()
        self.update_status(f"Slice View: X={current_x} Y={current_y} Z={current_z}")

    def clear_anatomical_slices(self, reset_state: bool = True) -> None:
        """
        Removes anatomical slice actors from all active scenes and resets slice state.
        """
        self._clear_crosshairs()

        # Create lists of valid objects only (filter out None) to reduce nesting
        actors_to_remove = [
            a for a in [self.axial_slice_actor, self.coronal_slice_actor, self.sagittal_slice_actor] 
            if a is not None
        ]
        
        scenes_to_check = [
            s for s in [self.scene, self.axial_scene, self.coronal_scene, self.sagittal_scene] 
            if s is not None
        ]

        scene_changed = False

        for actor in actors_to_remove:
            for scene in scenes_to_check:
                try:
                    # Attempt to remove actor from scene
                    scene.rm(actor)
                    scene_changed = True
                except (ValueError, AttributeError):
                    pass
                except Exception as e:
                    logger.error(f"Error removing slice actor: {e}", exc_info=True)

        # Reset actor references
        self.axial_slice_actor = None
        self.coronal_slice_actor = None
        self.sagittal_slice_actor = None

        if reset_state:
            self.current_slice_indices = {'x': None, 'y': None, 'z': None}
            self.image_shape_vox = None
            self.image_extents = {'x': None, 'y': None, 'z': None}

        if scene_changed:
            self._render_all()
            
    def _debug_roi_alignment(self, key: str, main_vox: np.ndarray, roi_vox: np.ndarray) -> None: 
        """Debug method to check ROI alignment issues."""
        main_world = self._voxel_to_world(main_vox, self.main_window.anatomical_image_affine)
        roi_world = self._voxel_to_world(roi_vox, self.main_window.roi_layers[key]['affine'])
        
        logger.info(f"  Main voxel: {main_vox} -> World: {main_world}")
        logger.info(f"  ROI voxel: {roi_vox} -> World: {roi_world}")
        logger.info(f"  World difference: {np.linalg.norm(main_world - roi_world)}")
        
        # Check if orientations match
        main_orient = nib.aff2axcodes(self.main_window.anatomical_image_affine)
        roi_orient = nib.aff2axcodes(self.main_window.roi_layers[key]['affine'])
        logger.info(f"  Main orientation: {main_orient}")
        logger.info(f"  ROI orientation: {roi_orient}")

    # --- Clear Crosshairs ---
    def _clear_crosshairs(self) -> None:
        """Removes the 2D crosshair actors from their scenes."""
        # Remove axial and coronal from their scenes
        for scn, act in [(self.axial_scene, self.axial_crosshair_actor),
                         (self.coronal_scene, self.coronal_crosshair_actor)]:
            if scn and act:
                try:
                    scn.rm(act)
                except (ValueError, AttributeError):
                    pass
                except Exception as e:
                    logger.error(f"Error removing crosshair actor: {e}")
        
        # Remove sagittal crosshair from overlay renderer
        if hasattr(self, 'sagittal_overlay_renderer') and self.sagittal_overlay_renderer and self.sagittal_crosshair_actor:
            try:
                self.sagittal_overlay_renderer.RemoveActor(self.sagittal_crosshair_actor)
            except Exception as e:
                logger.error(f"Error removing sagittal crosshair actor: {e}")

        self.axial_crosshair_actor = None
        self.coronal_crosshair_actor = None
        self.sagittal_crosshair_actor = None
        self.crosshair_lines = {}
        self.crosshair_appenders = {}
    
    # --- Voxel <-> World Coordinate Helpers ---
    def _voxel_to_world(self, vox_coord: List[float], affine: Optional[np.ndarray] = None) -> np.ndarray:
        """Converts a voxel index [i, j, k] to world RASmm coordinates [x, y, z]."""
        if affine is None:
            if self.main_window is None or self.main_window.anatomical_image_affine is None:
                return np.array(vox_coord) # Fallback
            affine = self.main_window.anatomical_image_affine
        
        homog_vox = np.array([vox_coord[0], vox_coord[1], vox_coord[2], 1.0])
        world_coord = np.dot(affine, homog_vox)
        return world_coord[:3]

    def _world_to_voxel(self, world_coord: List[float], inv_affine: Optional[np.ndarray] = None) -> np.ndarray:
        """Converts world RASmm coordinates [x, y, z] to voxel indices [i, j, k]."""
        if inv_affine is None:
            if self.main_window is None or self.main_window.anatomical_image_affine is None:
                return np.array(world_coord) # Fallback
            affine = self.main_window.anatomical_image_affine
            inv_affine = np.linalg.inv(affine)
            
        homog_world = np.array([world_coord[0], world_coord[1], world_coord[2], 1.0])
        vox_coord = np.dot(inv_affine, homog_world)
        return vox_coord[:3] # Return float voxel coords

    # --- Create/Update Crosshairs ---
    def _create_or_update_crosshairs(self) -> None:
        """Creates or updates the 2D crosshair actors based on current slice indices."""
        if (self.main_window is None or 
            self.main_window.anatomical_image_data is None or
            not all(self.current_slice_indices.values())):
            self._clear_crosshairs()
            return
            
        try:
            # Get current state
            x, y, z = self.current_slice_indices['x'], self.current_slice_indices['y'], self.current_slice_indices['z']
            x_min, x_max = self.image_extents['x']
            y_min, y_max = self.image_extents['y']
            z_min, z_max = self.image_extents['z']

            # --- 1. Define line endpoints in WORLD coordinates ---
            main_affine = self.main_window.anatomical_image_affine
            # Axial View (X-Y plane):
            ax_line_x_p1 = self._voxel_to_world([x, y_min, z], main_affine)
            ax_line_x_p2 = self._voxel_to_world([x, y_max, z], main_affine)
            ax_line_y_p1 = self._voxel_to_world([x_min, y, z], main_affine)
            ax_line_y_p2 = self._voxel_to_world([x_max, y, z], main_affine)
            
            # Coronal View (X-Z plane):
            co_line_x_p1 = self._voxel_to_world([x, y, z_min], main_affine)
            co_line_x_p2 = self._voxel_to_world([x, y, z_max], main_affine)
            co_line_z_p1 = self._voxel_to_world([x_min, y, z], main_affine)
            co_line_z_p2 = self._voxel_to_world([x_max, y, z], main_affine)
            
            # Sagittal View (Y-Z plane):
            sa_line_y_p1 = self._voxel_to_world([x, y_min, z], main_affine)
            sa_line_y_p2 = self._voxel_to_world([x, y_max, z], main_affine)
            sa_line_z_p1 = self._voxel_to_world([x, y, z_min], main_affine)
            sa_line_z_p2 = self._voxel_to_world([x, y, z_max], main_affine)
            
            # --- 2. Check if actors need to be created ---
            if self.axial_crosshair_actor is None:
                
                # --- Axial ---
                self.crosshair_lines['ax_x'] = vtk.vtkLineSource()
                self.crosshair_lines['ax_y'] = vtk.vtkLineSource()
                self.crosshair_appenders['axial'] = vtk.vtkAppendPolyData()
                self.crosshair_appenders['axial'].AddInputConnection(self.crosshair_lines['ax_x'].GetOutputPort())
                self.crosshair_appenders['axial'].AddInputConnection(self.crosshair_lines['ax_y'].GetOutputPort())
                ax_mapper = vtk.vtkPolyDataMapper()
                ax_mapper.SetInputConnection(self.crosshair_appenders['axial'].GetOutputPort())
                ax_mapper.SetResolveCoincidentTopologyToPolygonOffset()
                ax_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
                self.axial_crosshair_actor = vtk.vtkActor()
                self.axial_crosshair_actor.SetMapper(ax_mapper)
                self.axial_crosshair_actor.GetProperty().SetColor(1, 1, 0) # Yellow
                self.axial_crosshair_actor.GetProperty().SetLineWidth(1.0)
                self.axial_crosshair_actor.GetProperty().SetOpacity(0.8)
                self.axial_scene.add(self.axial_crosshair_actor)

                # --- Coronal ---
                self.crosshair_lines['co_x'] = vtk.vtkLineSource()
                self.crosshair_lines['co_z'] = vtk.vtkLineSource()
                self.crosshair_appenders['coronal'] = vtk.vtkAppendPolyData()
                self.crosshair_appenders['coronal'].AddInputConnection(self.crosshair_lines['co_x'].GetOutputPort())
                self.crosshair_appenders['coronal'].AddInputConnection(self.crosshair_lines['co_z'].GetOutputPort())
                co_mapper = vtk.vtkPolyDataMapper()
                co_mapper.SetInputConnection(self.crosshair_appenders['coronal'].GetOutputPort())
                co_mapper.SetResolveCoincidentTopologyToPolygonOffset()
                co_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
                self.coronal_crosshair_actor = vtk.vtkActor()
                self.coronal_crosshair_actor.SetMapper(co_mapper)
                self.coronal_crosshair_actor.GetProperty().SetColor(1, 1, 0)
                self.coronal_crosshair_actor.GetProperty().SetLineWidth(1.0)
                self.coronal_crosshair_actor.GetProperty().SetOpacity(0.8)
                self.coronal_scene.add(self.coronal_crosshair_actor)
                
                # --- Sagittal ---
                self.crosshair_lines['sa_y'] = vtk.vtkLineSource()
                self.crosshair_lines['sa_z'] = vtk.vtkLineSource()
                self.crosshair_appenders['sagittal'] = vtk.vtkAppendPolyData()
                self.crosshair_appenders['sagittal'].AddInputConnection(self.crosshair_lines['sa_y'].GetOutputPort())
                self.crosshair_appenders['sagittal'].AddInputConnection(self.crosshair_lines['sa_z'].GetOutputPort())
                sa_mapper = vtk.vtkPolyDataMapper()
                sa_mapper.SetInputConnection(self.crosshair_appenders['sagittal'].GetOutputPort())
                self.sagittal_crosshair_actor = vtk.vtkActor()
                self.sagittal_crosshair_actor.SetMapper(sa_mapper)
                self.sagittal_crosshair_actor.GetProperty().SetColor(1, 1, 0)
                self.sagittal_crosshair_actor.GetProperty().SetLineWidth(1.0)
                self.sagittal_crosshair_actor.GetProperty().SetOpacity(0.8)
                # Disable depth testing so crosshair always renders on top
                self.sagittal_crosshair_actor.GetProperty().SetLighting(False)
                self.sagittal_overlay_renderer.AddActor(self.sagittal_crosshair_actor)

            # --- 3. Update the line source endpoints ---
            # Convert numpy arrays to tuples for VTK compatibility
            self.crosshair_lines['ax_x'].SetPoint1(tuple(ax_line_x_p1))
            self.crosshair_lines['ax_x'].SetPoint2(tuple(ax_line_x_p2))
            self.crosshair_lines['ax_y'].SetPoint1(tuple(ax_line_y_p1))
            self.crosshair_lines['ax_y'].SetPoint2(tuple(ax_line_y_p2))
            
            self.crosshair_lines['co_x'].SetPoint1(tuple(co_line_x_p1))
            self.crosshair_lines['co_x'].SetPoint2(tuple(co_line_x_p2))
            self.crosshair_lines['co_z'].SetPoint1(tuple(co_line_z_p1))
            self.crosshair_lines['co_z'].SetPoint2(tuple(co_line_z_p2))

            self.crosshair_lines['sa_y'].SetPoint1(tuple(sa_line_y_p1))
            self.crosshair_lines['sa_y'].SetPoint2(tuple(sa_line_y_p2))
            self.crosshair_lines['sa_z'].SetPoint1(tuple(sa_line_z_p1))
            self.crosshair_lines['sa_z'].SetPoint2(tuple(sa_line_z_p2))
            
            # --- 4. Force pipeline update and mark actors as modified ---
            for key in ['ax_x', 'ax_y', 'co_x', 'co_z', 'sa_y', 'sa_z']:
                self.crosshair_lines[key].Update()
            for key in ['axial', 'coronal', 'sagittal']:
                self.crosshair_appenders[key].Update()
                
            if self.axial_crosshair_actor: self.axial_crosshair_actor.Modified()
            if self.coronal_crosshair_actor: self.coronal_crosshair_actor.Modified()
            if self.sagittal_crosshair_actor: self.sagittal_crosshair_actor.Modified()

        except Exception as e:
            logger.error(f"Error creating/updating crosshairs:", exc_info=True)
            self._clear_crosshairs()

    # --- Move Slice ---
    def move_slice(self, axis: str, direction: int) -> None:
        """Moves the specified slice ('x', 'y', 'z') by 'direction' (+1 or -1)."""
        if (not self.image_shape_vox or 
            axis not in self.current_slice_indices or 
            self.current_slice_indices[axis] is None):
            return

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_index = axis_map[axis]
        
        current_index = self.current_slice_indices[axis]
        max_index = self.image_shape_vox[axis_index] - 1
        
        new_index = current_index + direction
        new_index = max(0, min(new_index, max_index)) # Clamp

        if new_index == current_index:
            self.update_status(f"Slice View: Reached {axis.upper()} limit ({new_index}).")
            return
            
        # --- Delegate to the master update function ---
        kwargs = {axis: new_index}
        self.set_slice_indices(**kwargs)
 
        self._update_slow_slice_components() 
 
    # --- Master Slice Update Function ---
    def set_slice_indices(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        """
        Sets the slice indices, updates anatomical slices and crosshairs.
        This is called rapidly during a mouse drag.
        """
        if (not self.image_shape_vox or
            not self.axial_slice_actor or
            not self.coronal_slice_actor or
            not self.sagittal_slice_actor):
            return 

        # --- 1. Determine new indices and clamp them ---
        current = self.current_slice_indices

        new_x = x if x is not None else current['x']
        new_y = y if y is not None else current['y']
        new_z = z if z is not None else current['z']

        new_x = max(self.image_extents['x'][0], min(new_x, self.image_extents['x'][1]))
        new_y = max(self.image_extents['y'][0], min(new_y, self.image_extents['y'][1]))
        new_z = max(self.image_extents['z'][0], min(new_z, self.image_extents['z'][1]))

        # --- 2. Check if anything changed ---
        if (new_x == current['x'] and new_y == current['y'] and new_z == current['z']):
            return

        # --- 3. Update state ---
        self.current_slice_indices = {'x': new_x, 'y': new_y, 'z': new_z}

        try:
            # --- 4. Update all 3D slice actors (FAST) ---
            x_ext = self.image_extents['x']
            y_ext = self.image_extents['y']
            z_ext = self.image_extents['z']

            # Track which slice actually moved
            slice_moved = {'x': False, 'y': False, 'z': False}

            if new_x != current['x']:
                self.sagittal_slice_actor.display_extent(new_x, new_x, y_ext[0], y_ext[1], z_ext[0], z_ext[1])
                slice_moved['x'] = True

            if new_y != current['y']:
                self.coronal_slice_actor.display_extent(x_ext[0], x_ext[1], new_y, new_y, z_ext[0], z_ext[1])
                slice_moved['y'] = True

            if new_z != current['z']:
                self.axial_slice_actor.display_extent(x_ext[0], x_ext[1], y_ext[0], y_ext[1], new_z, new_z)
                slice_moved['z'] = True
                
            # --- 5. Update all ROI Slicers ---
            if self.main_window and self.roi_slice_actors and self.image_extents['x']:
                c_x = (self.image_extents['x'][0] + self.image_extents['x'][1]) / 2.0
                c_y = (self.image_extents['y'][0] + self.image_extents['y'][1]) / 2.0
                c_z = (self.image_extents['z'][0] + self.image_extents['z'][1]) / 2.0

                sag_plane_vox_center = np.array([new_x, c_y, c_z, 1.0])
                cor_plane_vox_center = np.array([c_x, new_y, c_z, 1.0])
                ax_plane_vox_center = np.array([c_x, c_y, new_z, 1.0])

                for key, actor_dict in self.roi_slice_actors.items():
                    roi_info = self.main_window.roi_layers.get(key)
                    if not roi_info or 'T_main_to_roi' not in roi_info: continue
                    
                    T_main_to_roi = roi_info['T_main_to_roi']
                    roi_shape = roi_info['data'].shape
                    roi_x_ext = (0, roi_shape[0] - 1)
                    roi_y_ext = (0, roi_shape[1] - 1)
                    roi_z_ext = (0, roi_shape[2] - 1)

                    try:
                        if slice_moved['x']:
                            roi_vox_c = T_main_to_roi.dot(sag_plane_vox_center)
                            new_roi_x = max(roi_x_ext[0], min(int(round(roi_vox_c[0])), roi_x_ext[1]))
                            # Update both 3D and 2D actors
                            if actor_dict.get('sagittal_3d'):
                                actor_dict['sagittal_3d'].display_extent(new_roi_x, new_roi_x, roi_y_ext[0], roi_y_ext[1], roi_z_ext[0], roi_z_ext[1])
                            if actor_dict.get('sagittal_2d'):
                                actor_dict['sagittal_2d'].display_extent(new_roi_x, new_roi_x, roi_y_ext[0], roi_y_ext[1], roi_z_ext[0], roi_z_ext[1])

                        if slice_moved['y']:
                            roi_vox_c = T_main_to_roi.dot(cor_plane_vox_center)
                            new_roi_y = max(roi_y_ext[0], min(int(round(roi_vox_c[1])), roi_y_ext[1]))
                            if actor_dict.get('coronal_3d'):
                                actor_dict['coronal_3d'].display_extent(roi_x_ext[0], roi_x_ext[1], new_roi_y, new_roi_y, roi_z_ext[0], roi_z_ext[1])
                            if actor_dict.get('coronal_2d'):
                                actor_dict['coronal_2d'].display_extent(roi_x_ext[0], roi_x_ext[1], new_roi_y, new_roi_y, roi_z_ext[0], roi_z_ext[1])

                        if slice_moved['z']:
                            roi_vox_c = T_main_to_roi.dot(ax_plane_vox_center)
                            new_roi_z = max(roi_z_ext[0], min(int(round(roi_vox_c[2])), roi_z_ext[1]))
                            if actor_dict.get('axial_3d'):
                                actor_dict['axial_3d'].display_extent(roi_x_ext[0], roi_x_ext[1], roi_y_ext[0], roi_y_ext[1], new_roi_z, new_roi_z)
                            if actor_dict.get('axial_2d'):
                                actor_dict['axial_2d'].display_extent(roi_x_ext[0], roi_x_ext[1], roi_y_ext[0], roi_y_ext[1], new_roi_z, new_roi_z)

                    except Exception as e:
                        logger.error(f"Error updating ROI layer {key} slice: {e}")

            self._create_or_update_crosshairs()

            # --- 6. Update 2D cameras for moved slices ---
            if slice_moved['x']:
                self._update_sagittal_camera()
            if slice_moved['y']:
                self._update_coronal_camera()
            if slice_moved['z']:
                self._update_axial_camera()

            # --- 7. Update status and render ---
            self._render_all()

        except Exception as e:
            error_msg = f"Error in set_slice_indices: {e}"
            logger.error(error_msg, exc_info=True)
    
    def _update_slow_slice_components(self) -> None:
            """
            Updates the "slow" components of the slice view (status text, RAS coords)
            after a drag-navigate is finished.
            """
            if (self.main_window is None or 
                not all(self.current_slice_indices.values()) or
                not self.image_extents['x']):
                
                # Still try to clear the coordinate display if no image
                if self.main_window:
                    self.main_window.update_ras_coordinate_display(None)
                return
            
            c = self.current_slice_indices
            
            # --- 1. Update status text ---
            status_msg = f"Slice View: X={c['x']}/{self.image_extents['x'][1]}  Y={c['y']}/{self.image_extents['y'][1]}  Z={c['z']}/{self.image_extents['z'][1]}"
            self.update_status(status_msg)

            # --- 2. Update RAS Coordinate Display ---
            try:
                main_affine = self.main_window.anatomical_image_affine
                if main_affine is not None:
                    # Get world coordinates
                    world_coord = self._voxel_to_world([c['x'], c['y'], c['z']], main_affine)
                    
                    # Update the main window's new display
                    self.main_window.update_ras_coordinate_display(world_coord)
                    
                else:
                    self.main_window.update_ras_coordinate_display(None)
            except Exception as e:
                logger.error(f"Error updating RAS coordinate display: {e}")
                self.main_window.update_ras_coordinate_display(None) # Pass None to signal error/clear
                
    def set_slices_from_ras(self, ras_coords: np.ndarray) -> None:
        """
        Moves the slice crosshairs to the specified RAS coordinate.
        Converts RAS -> Voxel, then calls set_slice_indices.
        
        Args:
            ras_coords: A (3,) numpy array of [X, Y, Z] world coordinates.
        """
        # 1. Check if we have the necessary info
        if (self.main_window is None or
            self.main_window.anatomical_image_affine is None or
            not all(self.image_extents.values()) or
            self.image_extents['x'] is None): # Check one specifically
            
            self.update_status("Error: Cannot set slices from RAS, no anatomical image loaded.")
            return

        try:
            # 2. Convert RAS world coordinate to (float) voxel coordinate
            inv_affine = np.linalg.inv(self.main_window.anatomical_image_affine)
            voxel_pos_float = self._world_to_voxel(ras_coords, inv_affine)

            # 3. Round to nearest integer voxel index
            new_x = int(round(voxel_pos_float[0]))
            new_y = int(round(voxel_pos_float[1]))
            new_z = int(round(voxel_pos_float[2]))
            
            # 4. Clamp to valid voxel range
            new_x = max(self.image_extents['x'][0], min(new_x, self.image_extents['x'][1]))
            new_y = max(self.image_extents['y'][0], min(new_y, self.image_extents['y'][1]))
            new_z = max(self.image_extents['z'][0], min(new_z, self.image_extents['z'][1]))
            
            # 5. Call the master update function
            self.set_slice_indices(x=new_x, y=new_y, z=new_z)
            
            # 6. Update the status bar and text input
            self._update_slow_slice_components()

        except Exception as e:
            self.update_status(f"Error setting slice from RAS: {e}")
            logger.error(f"Error setting slice from RAS:", exc_info=True)
         
    def _navigate_2d_view(self, obj: vtk.vtkObject, event_id: str) -> None:
        """
        Picks the 3D coordinate from a 2D view and updates the slice indices.
        """

        # 1. Check if anatomical image is loaded
        if (self.main_window is None or
            self.main_window.anatomical_image_data is None or
            None in self.current_slice_indices.values()):
            self.is_navigating_2d = False # avoid freeze
            return

        # 2. Get interactor and click position (display coords)
        interactor = obj
        if not interactor: return
        display_pos = interactor.GetEventPosition()

        # 3. Pick the 3D world coordinate at that pixel
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
        if not renderer: return

        picker.Pick(display_pos[0], display_pos[1], 0, renderer)

        # Check if the picker missed 
        if picker.GetCellId() < 0:
            # self.is_navigating_2d = False
            return
            
        # Get the 3D position of the pick
        world_pos = picker.GetPickPosition()

        try:
            # 4. Convert this world coordinate back to a (float) voxel coordinate
            inv_affine = np.linalg.inv(self.main_window.anatomical_image_affine)
            voxel_pos_float = self._world_to_voxel(world_pos, inv_affine)

            # 5. Round to nearest integer voxel index
            new_x = int(round(voxel_pos_float[0]))
            new_y = int(round(voxel_pos_float[1]))
            new_z = int(round(voxel_pos_float[2]))
            
            # --- Clamp to valid voxel range ---
            new_x = max(self.image_extents['x'][0], min(new_x, self.image_extents['x'][1]))
            new_y = max(self.image_extents['y'][0], min(new_y, self.image_extents['y'][1]))
            new_z = max(self.image_extents['z'][0], min(new_z, self.image_extents['z'][1]))
            
            # Abort if the pick produced an out-of-range index
            if (new_x < self.image_extents['x'][0] or new_x > self.image_extents['x'][1] or
                new_y < self.image_extents['y'][0] or new_y > self.image_extents['y'][1] or
                new_z < self.image_extents['z'][0] or new_z > self.image_extents['z'][1]):
                self.is_navigating_2d = False   
                return

            # 6. Call the master update function
            if interactor == self.axial_interactor:
                # Axial view (X-Y plane), so keep current Z
                current_z = self.current_slice_indices['z']
                self.set_slice_indices(x=new_x, y=new_y, z=current_z)

            elif interactor == self.coronal_interactor:
                # Coronal view (X-Z plane), so keep current Y
                current_y = self.current_slice_indices['y']
                self.set_slice_indices(x=new_x, y=current_y, z=new_z)

            elif interactor == self.sagittal_interactor:
                # Sagittal view (Y-Z plane), so keep current X
                current_x = self.current_slice_indices['x']
                self.set_slice_indices(x=current_x, y=new_y, z=new_z)

            else:
                self.set_slice_indices(x=new_x, y=new_y, z=new_z)

        except Exception as e:
            logger.error(f"Error during 2D window click navigation:", exc_info=True)
            self.update_status("Error navigating with click.")
            

    # --- Streamline Actor Management ---
    def update_highlight(self) -> None:
        """Updates the actor for highlighted/selected streamlines."""
        if not self.scene: return

        # Safely Remove Existing Highlight Actor
        actor_removed = False
        if self.highlight_actor is not None:
            try:
                self.scene.rm(self.highlight_actor)
                actor_removed = True
            except (ValueError, AttributeError): actor_removed = True 
            except Exception as e: 
                logger.error(f"  Error removing highlight actor: {e}. Proceeding cautiously.")
            finally: self.highlight_actor = None

        # Check prerequisites
        if not self.main_window or \
           not hasattr(self.main_window, 'selected_streamline_indices') or \
           not self.main_window.tractogram_data:
            if self.main_window: self.main_window._update_action_states()
            return

        selected_indices: Set[int] = self.main_window.selected_streamline_indices
        tractogram: 'nib.streamlines.ArraySequence' = self.main_window.tractogram_data

        # Create new actor only if there's a valid selection
        if selected_indices:
            valid_indices = {idx for idx in selected_indices if 0 <= idx < len(tractogram)}
            
            # Create a concrete list of *non-empty* selected streamlines
            selected_sl_data_list = []
            for idx in valid_indices:
                try:
                    sl = tractogram[idx]
                    if sl is not None and len(sl) > 0:
                        selected_sl_data_list.append(sl)
                except Exception:
                    pass # Ignore if index fails
            
            if selected_sl_data_list: # Check if the list is not empty
                try:
                    highlight_linewidth = 6 
                    self.highlight_actor = actor.line(
                        selected_sl_data_list,     # Pass the list
                        colors=(1, 1, 0),          # Bright Yellow
                        linewidth=highlight_linewidth,
                        opacity=1.0                # Fully opaque
                    )
                    self.scene.add(self.highlight_actor)
                    
                    if self.main_window and not self.main_window.bundle_is_visible:
                        self.highlight_actor.SetVisibility(0)
                    
                    if self.highlight_actor:
                        try:
                            mapper = self.highlight_actor.GetMapper()
                            if mapper:
                                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -15)
                        except Exception as e:
                            logger.warning(f"Warning: Could not apply mapper offset to highlight_actor: {e}")
                    
                except Exception as e:
                     logger.error(f"Error creating highlight actor:", exc_info=True)
                     self.highlight_actor = None

        # Update UI action states
        if self.main_window: self.main_window._update_action_states()

    # --- Function signature updated ---
    def _calculate_scalar_colors(self, streamlines_gen: 'generator', scalar_gen: 'generator', vmin: float, vmax: float) -> Optional[Dict[str, Any]]:
        """
        Calculates vertex colors based on a list of scalar arrays per streamline,
        using the provided vmin and vmax for the colormap range.
        Returns a single concatenated (TotalPoints, 3) numpy array for FURY.
        """
        
        # --- 1. Create the LUT ---
        try:
            lut = vtk.vtkLookupTable()
            table_min = vmin - 0.5 if vmin == vmax else vmin
            table_max = vmax + 0.5 if vmin == vmax else vmax
            
            lut.SetTableRange(table_min, table_max)
            lut.SetHueRange(0.667, 0.0) # Blue to Red (standard)
            lut.Build()
        except Exception as e:
            logger.error(f"Error creating scalar LUT: {e}. Defaulting to grey.")
            return None # Fallback to grey
        
        # --- 3. Build the list of color arrays (one per *non-empty* streamline) ---
        default_color_rgb = np.array([128, 128, 128], dtype=np.uint8)
        vertex_colors_list: List[np.ndarray] = [] # List to hold individual np.arrays
        rgb_output: List[float] = [0.0, 0.0, 0.0]

        for sl, sl_scalars in zip(streamlines_gen, scalar_gen):
            num_points = len(sl) if sl is not None else 0
            if num_points == 0: # Skip empty streamlines
                continue
                
            sl_colors_rgb = np.empty((num_points, 3), dtype=np.uint8)
            
            # Check if this (non-empty) streamline has valid scalar data
            has_valid_scalar_for_this_sl = False
            if sl_scalars is not None and hasattr(sl_scalars, 'size') and len(sl_scalars) == num_points:
                try:
                    for j in range(num_points):
                        lut.GetColor(sl_scalars[j], rgb_output)
                        sl_colors_rgb[j] = [int(c * 255) for c in rgb_output]
                    has_valid_scalar_for_this_sl = True
                except Exception:
                    has_valid_scalar_for_this_sl = False # e.g., non-numeric data 

            if not has_valid_scalar_for_this_sl:
                sl_colors_rgb[:] = default_color_rgb # Fill with default color
            
            vertex_colors_list.append(sl_colors_rgb)

        if not vertex_colors_list: 
            return None 

        # --- 4. Concatenate all color arrays into one big array ---
        try:
            concatenated_colors = np.concatenate(vertex_colors_list, axis=0)
        except ValueError as ve:
             logger.error(f"Failed to concatenate color arrays: {ve}")
             return None # Fallback to grey

        return {'colors': concatenated_colors, 'opacity': 0.8, 'linewidth': 3}

    def _get_streamline_actor_params(self) -> Dict[str, Any]:
        """Determines parameters for the main streamlines actor."""
        params: Dict[str, Any] = {'colors': (0.8, 0.8, 0.8), 'opacity': 0.5, 'linewidth': 2}
        if not self.main_window or not self.main_window.tractogram_data: 
            return params

        tractogram: 'nib.streamlines.ArraySequence' = self.main_window.tractogram_data
        indices_to_draw: Set[int] = self.main_window.visible_indices
        
        # Create a list of *only the visible, non-empty* streamlines.
        visible_streamlines_list = []
        visible_indices_list = [] # Keep track of indices for scalars
        for i in indices_to_draw:
             try:
                 sl = tractogram[i]
                 if sl is not None and len(sl) > 0:
                     visible_streamlines_list.append(sl)
                     visible_indices_list.append(i)
             except Exception:
                 pass # Ignore if an index fails
        
        # If no non-empty streamlines are visible, return default grey
        if not visible_streamlines_list:
             params['streamlines_list'] = [] # Pass empty list
             return params

        current_mode: ColorMode = self.main_window.current_color_mode

        if current_mode == ColorMode.ORIENTATION:
            try:
                # Pass the concrete list
                params['colors'] = colormap.line_colors(visible_streamlines_list)
                params['opacity'] = 0.8
            except Exception as e: 
                logger.warning(f"Error calculating orientation colors: {e}. Using default.")
        
        elif current_mode == ColorMode.SCALAR:
            scalar_data = self.main_window.scalar_data_per_point
            active_scalar = self.main_window.active_scalar_name
            if scalar_data and active_scalar and active_scalar in scalar_data:
                scalar_sequence = scalar_data.get(active_scalar)
                if scalar_sequence:
                    vmin = self.main_window.scalar_min_val
                    vmax = self.main_window.scalar_max_val
                    
                    # We need to get the scalars *corresponding to the non-empty streamlines*
                    visible_scalars_list = [
                        scalar_sequence[i] for i in visible_indices_list
                    ]
                    
                    scalar_params = self._calculate_scalar_colors(
                        iter(visible_streamlines_list), 
                        iter(visible_scalars_list), 
                        vmin, vmax
                    )
                    # if scalar_params is None, we just keep the default grey
                    if scalar_params: 
                        params = scalar_params

        params['streamlines_list'] = visible_streamlines_list
        return params
                     
    def update_main_streamlines_actor(self) -> None:
        """Recreates the main streamlines actor based on current data and color mode."""
        if not self.scene: 
            return

        actor_removed = False
        if self.streamlines_actor is not None:
            try:
                self.scene.rm(self.streamlines_actor)
                actor_removed = True
            except ValueError:
                actor_removed = True
            except Exception as e: 
                logger.warning(f"  Error removing streamline actor: {e}. Proceeding cautiously.")
            finally: 
                self.streamlines_actor = None

        if not self.main_window or not self.main_window.tractogram_data:
            self.update_highlight()
            return

        # Get original data
        tractogram: 'nib.streamlines.ArraySequence' = self.main_window.tractogram_data
        indices_to_draw: Set[int] = self.main_window.visible_indices
        
        if not indices_to_draw: 
             self.update_highlight()
             return
             
        # Get parameters
        actor_params = self._get_streamline_actor_params()
        
        # Pop the streamlines list from the params
        streamlines_to_draw_list = actor_params.pop('streamlines_list', None)
        
        # If the list is empty or None there's nothing to draw.
        if not streamlines_to_draw_list:
             self.update_highlight()
             return

        try:
            self.streamlines_actor = actor.line(streamlines_to_draw_list, **actor_params) 
            if self.main_window and not self.main_window.bundle_is_visible:
                self.streamlines_actor.SetVisibility(0)
            self.scene.add(self.streamlines_actor)
            
            if self.streamlines_actor:
                try:
                    mapper = self.streamlines_actor.GetMapper()
                    if mapper:
                        mapper.SetResolveCoincidentTopologyToPolygonOffset()
                        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -15)
                except Exception as e:
                    logger.warning(f"Warning: Could not apply mapper offset to streamlines_actor: {e}")
                    
        except Exception as e:
            logger.error(f"Error creating main streamlines actor:", exc_info=True)
            QMessageBox.warning(self.main_window, "Actor Error", f"Could not display streamlines: {e}")
            try:
                if self.streamlines_actor:
                    try: self.scene.rm(self.streamlines_actor)
                    except: 
                        pass
                
                if streamlines_to_draw_list:
                    self.streamlines_actor = actor.line(streamlines_to_draw_list, colors=(0.8, 0.8, 0.8), opacity=0.5, linewidth=2)
                    self.scene.add(self.streamlines_actor)
                    
                    if self.streamlines_actor:
                        try:
                            mapper = self.streamlines_actor.GetMapper()
                            if mapper:
                                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -15)
                        except Exception as e:
                            logger.warning(f"Warning: Could not apply mapper offset to fallback_actor: {e}")   
                
                if self.main_window:
                     self.main_window.current_color_mode = ColorMode.DEFAULT
                     self.main_window.color_default_action.setChecked(True)
            except Exception as fallback_e:
                logger.error(f"Error creating fallback streamlines actor: {fallback_e}")
                self.streamlines_actor = None

        self.update_highlight()
        if self.render_window and self.render_window.GetInteractor().GetInitialized():
            self.render_window.Render()

    # --- Interaction Callbacks ---
    def _handle_save_shortcut(self) -> None:
        """Handles the Ctrl+S save shortcut logic (saves streamlines)."""
        if not self.main_window: 
            return
        if not self.main_window.tractogram_data:
             self.update_status("Save shortcut (Ctrl+S): No streamlines loaded to save.")
             return
        self.update_radius_actor(visible=False)
        try: self.main_window._trigger_save_streamlines()
        except AttributeError: self.update_status("Error: Save function not found.")
        except Exception as e: self.update_status(f"Error during save: {e}")

    def _handle_radius_change(self, increase: bool = True) -> None:
        """Handles increasing or decreasing the selection radius via main window."""
        if not self.main_window: 
            return
        if not self.main_window.tractogram_data:
             self.update_status("Radius change (+/-): No streamlines loaded.")
             return
        if increase: self.main_window._increase_radius()
        else: self.main_window._decrease_radius()

    def _find_streamlines_in_radius(self, center_point: np.ndarray, radius: float) -> Set[int]:
        """
        Finds indices of streamlines intersecting a sphere.
        Uses Bounding Box check (speed) and Segment Distance (precision). 
        """
        if not self.main_window or not self.main_window.tractogram_data: 
            return set()
            
        tractogram = self.main_window.tractogram_data
        indices_to_search = self.main_window.visible_indices
        
        indices_in_radius: Set[int] = set()
        radius_sq = radius * radius
        
        # Pre-calculate bounds for the sphere for fast rejection
        sphere_min = center_point - radius
        sphere_max = center_point + radius
        
        for idx in indices_to_search:
            if not isinstance(tractogram, nib.streamlines.ArraySequence) or idx >= len(tractogram):
                continue 
            
            try:
                sl = tractogram[idx]
                if not isinstance(sl, np.ndarray) or sl.ndim != 2 or sl.shape[1] != 3 or sl.size == 0: 
                    continue
            
                # --- Bounding Box Check ---
                # If the streamline's bounding box doesn't overlap the sphere's box, skip it.
                sl_min = np.min(sl, axis=0)
                sl_max = np.max(sl, axis=0)
                
                if np.any(sl_max < sphere_min) or np.any(sl_min > sphere_max):
                    continue

                # --- Segment Distance ---
                # Check distance to points first (fastest)
                diff_sq = np.sum((sl - center_point)**2, axis=1)
                if np.min(diff_sq) < radius_sq:
                    indices_in_radius.add(idx)
                    continue
                
                # If points didn't trigger, check the segments between points (slowest).
                # This catches the case where the sphere is *between* two points. We vectorize the calculation for the whole streamline.
                p1 = sl[:-1]
                p2 = sl[1:]
                
                segment_vec = p2 - p1 # Vector from p1 to p2
                point_vec = center_point - p1 # Vector from p1 to center

                # Project point_vec onto segment_vec (dot product)
                seg_len_sq = np.sum(segment_vec**2, axis=1)
                
                # Avoid division by zero
                seg_len_sq[seg_len_sq == 0] = 1.0
                
                t = np.sum(point_vec * segment_vec, axis=1) / seg_len_sq
                
                # Clamp t to segment [0, 1]
                t = np.clip(t, 0, 1)
                
                # Find closest point on segment
                closest_points = p1 + segment_vec * t[:, np.newaxis]
                
                # Check distances
                seg_dists_sq = np.sum((closest_points - center_point)**2, axis=1)
                
                if np.min(seg_dists_sq) < radius_sq:
                    indices_in_radius.add(idx)

            except Exception as e: 
                logger.warning(f"Warning: Error processing streamline {idx} for selection: {e}")
                
        return indices_in_radius

    def _toggle_selection(self, indices_to_toggle: Set[int]) -> None:
        """Toggles the selection state for given indices and updates status/highlight."""
        if not self.main_window or not hasattr(self.main_window, 'selected_streamline_indices'): 
            return
        current_selection: Set[int] = self.main_window.selected_streamline_indices
        if current_selection is None: self.main_window.selected_streamline_indices = current_selection = set()

        added_count, removed_count = 0, 0
        for idx in indices_to_toggle:
            if idx in current_selection: current_selection.remove(idx); removed_count += 1
            else: current_selection.add(idx); added_count += 1

        if added_count > 0 or removed_count > 0:
            total_selected = len(current_selection)
            status_msg = (f"Radius Sel: Found {len(indices_to_toggle)}. "
                          f"Added {added_count}, Removed {removed_count}. Total Sel: {total_selected}")
            self.update_status(status_msg)
            self.update_highlight()
        elif indices_to_toggle:
             self.update_status(f"Radius Sel: Found {len(indices_to_toggle)}. Selection unchanged.")
             
    def _handle_streamline_selection(self) -> None:
        """Handles the logic for selecting streamlines triggered by the 's' key."""
        if not self.scene or not self.main_window or not self.main_window.tractogram_data:
            self.update_status("Select ('s'): No streamlines loaded to select from.")
            self.update_radius_actor(visible=False)
            return

        display_pos = self.interactor.GetEventPosition()
        
        picker = vtk.vtkCellPicker() 
        picker.SetTolerance(0.005) 
        picker.Pick(display_pos[0], display_pos[1], 0, self.render_window.GetRenderers().GetFirstRenderer())

        picked_actor = picker.GetActor()
        click_pos_world = picker.GetPickPosition()

        if not picked_actor or not click_pos_world or len(click_pos_world) != 3 or picker.GetCellId() < 0: 
            self.update_status("Select ('s'): Please click directly on visible streamlines.")
            self.update_radius_actor(visible=False)
            return

        p_center_arr = np.array(click_pos_world)
        radius = self.main_window.selection_radius_3d
        self.update_radius_actor(center_point=p_center_arr, radius=radius, visible=True)
        indices_in_radius = self._find_streamlines_in_radius(p_center_arr, radius)

        if not indices_in_radius:
             self.update_status("Radius Sel: No streamlines found within radius at click position.")
             self._toggle_selection(set())
        else:
             self._toggle_selection(indices_in_radius)
        
        # --- Hide selection sphere after selection is applied ---
        self.update_radius_actor(visible=False)

    def key_press_callback(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles key press events forwarded from the VTK interactor."""
        if not self.scene or not self.main_window: 
            return

        interactor = obj
        if not interactor: 
             return
        key_sym = interactor.GetKeySym()
        key = key_sym.lower() if key_sym and isinstance(key_sym, str) else ""
        ctrl = interactor.GetControlKey() == 1
        shift = interactor.GetShiftKey() == 1

        handler_key = (key, ctrl, shift)

        # --- Handle non-data-dependent keys first ---
        non_data_handlers = {
            ('z', True, False): self.main_window._perform_undo,
            ('y', True, False): self.main_window._perform_redo,
            ('z', True, True): self.main_window._perform_redo, # Ctrl+Shift+Z
            ('escape', False, False): self.main_window._hide_sphere,
            ('s', True, False): self._handle_save_shortcut,
        }
        if handler_key in non_data_handlers:
            non_data_handlers[handler_key]()
            return

        # --- Handle Slice Navigation Keys (require anatomical image) ---
        anatomical_loaded = self.main_window.anatomical_image_data is not None
        if anatomical_loaded:
            slice_nav_handlers = {
                # (key, ctrl, shift) : (axis, direction)
                ('up', False, False): ('z', 1),   # Axial Up
                ('down', False, False): ('z', -1), # Axial Down
                ('right', False, False): ('x', 1),  # Sagittal Right
                ('left', False, False): ('x', -1), # Sagittal Left
                ('up', True, False): ('y', 1),    # Coronal Up
                ('down', True, False): ('y', -1),  # Coronal Down
            }
            if handler_key in slice_nav_handlers:
                axis, direction = slice_nav_handlers[handler_key]
                self.move_slice(axis, direction)
                return 

        # --- Handle Streamline-dependent keys ---
        streamline_data_handlers = {
            's': self._handle_streamline_selection,
            'plus': lambda: self._handle_radius_change(increase=True),
            'equal': lambda: self._handle_radius_change(increase=True),
            'minus': lambda: self._handle_radius_change(increase=False),
            'd': self.main_window._perform_delete_selection,
            'c': self.main_window._perform_clear_selection,
        }
        streamline_keys_for_status = {'s', 'plus', 'equal', 'minus', 'd', 'c'}

        streamlines_loaded = bool(self.main_window.tractogram_data)
        if not streamlines_loaded:
            if key in streamline_keys_for_status:
                self.update_status(f"Action ('{key}') requires streamlines. Load a trk/tck file first.")
            return

        if key in streamline_data_handlers and not ctrl and not shift:
             streamline_data_handlers[key]()
             
    # --- ROI Layer Actor Management ---
    def add_roi_layer(self, key: str, data: np.ndarray, affine: np.ndarray) -> None:
        """Creates and adds slicer actors for a new ROI layer (separating 2D/3D actors)."""
        if not self.scene or not self.main_window or self.main_window.anatomical_image_affine is None:
            logger.error("Cannot add ROI layer: Main scene or image not initialized.")
            return

        # 1. Determine value range
        data_min = np.min(data)
        data_max = np.max(data)
        value_range = (data_min, data_max)
        
        if data_max <= data_min:
            value_range = (data_min - 0.5, data_max + 0.5)
        elif data_min == 0 and data_max > 0:
            value_range = (0.1, data_max) 
        
        slicer_opacity = 0.5 
        interpolation_mode = 'nearest'

        # 2. Get current slice indices
        c_idx = self.current_slice_indices
        if c_idx['x'] is None:
            self.update_status("Error: Slices not initialized.")
            return

        # 3. Calculate initial ROI slice indices
        main_img_affine = self.main_window.anatomical_image_affine
        roi_info = self.main_window.roi_layers.get(key)
        if not roi_info or 'inv_affine' not in roi_info:
            return
            
        inv_roi_affine = roi_info['inv_affine']
        c_x, c_y, c_z = c_idx['x'], c_idx['y'], c_idx['z']
        world_c = self._voxel_to_world([c_x, c_y, c_z], main_img_affine)
        roi_vox_c = self._world_to_voxel(world_c, inv_roi_affine)
        roi_x, roi_y, roi_z = int(round(roi_vox_c[0])), int(round(roi_vox_c[1])), int(round(roi_vox_c[2]))

        # 4. Get ROI extents
        roi_shape = data.shape
        roi_x_ext = (0, roi_shape[0] - 1)
        roi_y_ext = (0, roi_shape[1] - 1)
        roi_z_ext = (0, roi_shape[2] - 1)

        try:
            slicer_params = {
                'affine': affine,
                'value_range': value_range,
                'opacity': slicer_opacity,
                'interpolation': interpolation_mode
            }
            
            # --- Create TWO sets of actors ---
            # Set 1: For the main 3D Scene (No artificial flipping)
            ax_3d = actor.slicer(data, **slicer_params)
            cor_3d = actor.slicer(data, **slicer_params)
            sag_3d = actor.slicer(data, **slicer_params)
            
            # Set 2: For the 2D Ortho Views (Will get display correction)
            ax_2d = actor.slicer(data, **slicer_params)
            cor_2d = actor.slicer(data, **slicer_params)
            sag_2d = actor.slicer(data, **slicer_params)
            
            is_visible = True
            if self.main_window:
                is_visible = self.main_window.roi_visibility.get(key, True)
            vis_flag = 1 if is_visible else 0
            
            for act in [ax_3d, cor_3d, sag_3d, ax_2d, cor_2d, sag_2d]:
                act.SetVisibility(vis_flag)
            
            # Set initial display extents for ALL actors
            for ax in [ax_3d, ax_2d]:
                ax.display_extent(roi_x_ext[0], roi_x_ext[1], roi_y_ext[0], roi_y_ext[1], roi_z, roi_z)
            for cor in [cor_3d, cor_2d]:
                cor.display_extent(roi_x_ext[0], roi_x_ext[1], roi_y, roi_y, roi_z_ext[0], roi_z_ext[1])
            for sag in [sag_3d, sag_2d]:
                sag.display_extent(roi_x, roi_x, roi_y_ext[0], roi_y_ext[1], roi_z_ext[0], roi_z_ext[1])
            
            # Store in expanded dictionary
            self.roi_slice_actors[key] = {
                'axial_3d': ax_3d, 'coronal_3d': cor_3d, 'sagittal_3d': sag_3d,
                'axial_2d': ax_2d, 'coronal_2d': cor_2d, 'sagittal_2d': sag_2d
            }
            
            # 6. Add to respective scenes
            self.scene.add(ax_3d); self.scene.add(cor_3d); self.scene.add(sag_3d)
            self.axial_scene.add(ax_2d)
            self.coronal_scene.add(cor_2d)
            self.sagittal_scene.add(sag_2d)
            
            # --- Apply transformation ONLY to 2D actors ---
            self._apply_display_correction(key)
            
            # ---Set default color to Dark Grey (0.25, 0.25, 0.25) --
            self.set_roi_layer_color(key, (1.0, 0.0, 0.0))
            
            self.update_status(f"Added ROI layer: {os.path.basename(key)}")
            self._render_all()

        except Exception as e:
            logger.error(f"Error creating ROI slicer actors:", exc_info=True)
            QMessageBox.critical(self.main_window, "ROI Actor Error", f"Could not create slicer actors for ROI:\n{e}")
            if key in self.main_window.roi_layers: del self.main_window.roi_layers[key]
            if key in self.roi_slice_actors: del self.roi_slice_actors[key]

    def _apply_display_correction(self, roi_key: str) -> None:
        """Applies display correction only to 2D ROI slices."""
        if roi_key not in self.roi_slice_actors:
            return
            
        roi_actors = self.roi_slice_actors[roi_key]
        
        # Only correct the 2D actors
        targets = ['axial_2d', 'coronal_2d']
        
        for actor_type in targets:
            actor_obj = roi_actors.get(actor_type)
            if actor_obj:
                current_pos = actor_obj.GetPosition()
                # Flip in X direction
                actor_obj.SetScale(-1, 1, 1)
                
                # Heuristic adjustment for position
                bounds = actor_obj.GetBounds()
                if bounds[0] != 1.0 and bounds[1] != 1.0:
                    actor_obj.SetPosition(-current_pos[0], current_pos[1], current_pos[2])
                        
    def set_roi_layer_visibility(self, key: str, visible: bool) -> None:
        """Sets visibility for all 2D and 3D actors of an ROI."""
        if key not in self.roi_slice_actors: return
        
        vis_flag = 1 if visible else 0
        changed = False
        
        # Iterate over all actors stored for this key
        for act in self.roi_slice_actors[key].values():
            if act and act.GetVisibility() != vis_flag:
                act.SetVisibility(vis_flag)
                changed = True
        
        if changed: self._render_all()

    def remove_roi_layer(self, key: str) -> None:
        """Removes a specific ROI layer's actors from all scenes."""
        if key not in self.roi_slice_actors: return

        scenes_to_check = [self.scene, self.axial_scene, self.coronal_scene, self.sagittal_scene]
        
        for act in self.roi_slice_actors[key].values():
            if act is not None:
                for scn in scenes_to_check:
                    try: scn.rm(act)
                    except: pass

        del self.roi_slice_actors[key]
        self._render_all()


    def clear_all_roi_layers(self) -> None:
        """Removes all ROI slice actors from all scenes."""
        if not self.roi_slice_actors: return
            
        scenes_to_check = [self.scene, self.axial_scene, self.coronal_scene, self.sagittal_scene]
        
        for actor_dict in self.roi_slice_actors.values():
            for act in actor_dict.values():
                if act:
                    for scn in scenes_to_check:
                        try: scn.rm(act)
                        except: pass

        self.roi_slice_actors.clear()
        self._render_all()

    def set_roi_layer_color(self, key: str, color: Tuple[float, float, float]) -> None:
        """Sets the color (tint) for a specific ROI layer."""
        if key not in self.roi_slice_actors: return
        
        r, g, b = color
        for act in self.roi_slice_actors[key].values():
            if act:
                prop = act.GetProperty()
                lut = vtk.vtkLookupTable()
                lut.SetNumberOfTableValues(256)
                w = prop.GetColorWindow()
                l = prop.GetColorLevel()
                lut.SetTableRange(l - w/2, l + w/2)
                lut.Build()
                for i in range(256):
                    alpha = 0.0 if i == 0 else 1.0
                    lut.SetTableValue(i, r * i/255, g * i/255, b * i/255, alpha)
                prop.SetLookupTable(lut)
        
        self._render_all()
            
    def take_screenshot(self) -> None:
        """Saves a screenshot of the VTK view with an opaque black background, hiding UI overlays."""
        if not self.render_window or not self.scene:
            QMessageBox.warning(self.main_window, "Screenshot Error", "Render window or scene not available.")
            return
        if not (self.main_window.tractogram_data or self.main_window.anatomical_image_data):
            QMessageBox.warning(self.main_window, "Screenshot Error", "No data loaded to take screenshot of.")
            return

        default_filename = "tractedit_screenshot.png"
        base_name = "tractedit_view"
        if self.main_window.original_trk_path: base_name = os.path.splitext(os.path.basename(self.main_window.original_trk_path))[0]
        elif self.main_window.anatomical_image_path: base_name = os.path.splitext(os.path.basename(self.main_window.anatomical_image_path))[0]
        
        # Add slice info to filename if slices are visible
        slice_info = ""
        if self.axial_slice_actor and self.axial_slice_actor.GetVisibility() and self.current_slice_indices['z'] is not None:
            slice_info = f"_x{self.current_slice_indices['x']}_y{self.current_slice_indices['y']}_z{self.current_slice_indices['z']}"

        default_filename = f"{base_name}{slice_info}_screenshot.png"

        output_path, _ = QFileDialog.getSaveFileName(
            self.main_window,"Save Screenshot", default_filename,
            "PNG Image Files (*.png);;JPEG Image Files (*.jpg *.jpeg);;TIFF Image Files (*.tif *.tiff);;All Files (*.*)")
        if not output_path: self.update_status("Screenshot cancelled."); return

        self.update_status(f"Preparing screenshot: {os.path.basename(output_path)}...")
        QApplication.processEvents()

        original_background = self.scene.GetBackground()
        original_status_visibility = self.status_text_actor.GetVisibility() if self.status_text_actor else 0
        original_instruction_visibility = self.instruction_text_actor.GetVisibility() if self.instruction_text_actor else 0
        original_axes_visibility = self.axes_actor.GetVisibility() if self.axes_actor else 0
        original_radius_actor_visibility = self.radius_actor.GetVisibility() if self.radius_actor else 0
        original_slice_vis = {}
        slice_actors = {'axial': self.axial_slice_actor, 'coronal': self.coronal_slice_actor, 'sagittal': self.sagittal_slice_actor}
        for name, act in slice_actors.items(): original_slice_vis[name] = act.GetVisibility() if act else 0

        try:
            if self.status_text_actor: self.status_text_actor.SetVisibility(0)
            if self.instruction_text_actor: self.instruction_text_actor.SetVisibility(0)
            if self.axes_actor: self.axes_actor.SetVisibility(0)
            if self.radius_actor: self.radius_actor.SetVisibility(0)

            self.scene.background((0.0, 0.0, 0.0))
            self.render_window.Render()

            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(self.render_window)
            window_to_image_filter.SetInputBufferTypeToRGB()
            window_to_image_filter.ReadFrontBufferOff()
            window_to_image_filter.ShouldRerenderOff()
            window_to_image_filter.Update()

            _, output_ext = os.path.splitext(output_path); output_ext = output_ext.lower()
            writer: Optional[vtk.vtkImageWriter] = None
            if output_ext == '.png': writer = vtk.vtkPNGWriter()
            elif output_ext in ['.jpg', '.jpeg']: 
                writer = vtk.vtkJPEGWriter()
                writer.SetQuality(95)
                writer.ProgressiveOn()
            elif output_ext in ['.tif', '.tiff']: 
                writer = vtk.vtkTIFFWriter()
                writer.SetCompressionToPackBits()
            else:
                output_path = os.path.splitext(output_path)[0] + ".png"
                writer = vtk.vtkPNGWriter()
                QMessageBox.warning(self.main_window, "Screenshot Info", f"Unsupported extension '{output_ext}'. Saving as PNG.")

            writer.SetFileName(output_path)
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()
            self.update_status(f"Screenshot saved: {os.path.basename(output_path)}")

        except Exception as e:
            error_msg = f"Error taking screenshot:"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self.main_window, "Screenshot Error", error_msg)
            self.update_status(f"Error saving screenshot.")
        finally:
            if original_background: self.scene.background(original_background)
            if self.status_text_actor: self.status_text_actor.SetVisibility(original_status_visibility)
            if self.instruction_text_actor: self.instruction_text_actor.SetVisibility(original_instruction_visibility)
            if self.axes_actor: self.axes_actor.SetVisibility(original_axes_visibility)
            if self.radius_actor: self.radius_actor.SetVisibility(original_radius_actor_visibility)

            if self.render_window and self.render_window.GetInteractor().GetInitialized():
                self.render_window.Render()