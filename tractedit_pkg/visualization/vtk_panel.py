# -*- coding: utf-8 -*-

"""
Main VTK panel module for TractEdit visualization.

Manages the VTK rendering window, scene, actors (streamlines, image slices),
and interactions. Uses helper modules for interactor styles, actor creation,
scene management, and coordinate transformations.
"""

# ============================================================================
# Imports
# ============================================================================

import os
import numpy as np
import vtk
import logging
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QMessageBox,
    QApplication,
    QFileDialog,
    QWidget,
    QHBoxLayout,
    QSplitter,
    QMenu,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QAction
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from fury import window, actor, colormap
from ..utils import ColorMode
from typing import Optional, List, Dict, Any, Tuple, Set
import nibabel as nib
from functools import partial
from vtk.util import numpy_support
from .interactor import CustomInteractorStyle2D
from .scene_manager import SceneManager
from .drawing import DrawingManager
from .selection import SelectionManager
from .streamlines import StreamlinesManager
from .scale_bar import ScaleBarManager
from . import coordinates

logger = logging.getLogger(__name__)


# ============================================================================
# VTK Panel Class
# ============================================================================


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

        # Create the main vertical splitter (3D view on top, 2D views on bottom)
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Create the main 3D VTK widget
        self.vtk_widget: QVTKRenderWindowInteractor = QVTKRenderWindowInteractor()
        main_splitter.addWidget(self.vtk_widget)  # Add to top

        # Create the bottom container for 2D views
        ortho_container = QWidget()
        ortho_layout = QHBoxLayout(ortho_container)
        ortho_layout.setContentsMargins(0, 0, 0, 0)

        # Create the horizontal splitter for the three 2D views
        ortho_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.axial_vtk_widget: QVTKRenderWindowInteractor = QVTKRenderWindowInteractor()
        self.coronal_vtk_widget: QVTKRenderWindowInteractor = (
            QVTKRenderWindowInteractor()
        )
        self.sagittal_vtk_widget: QVTKRenderWindowInteractor = (
            QVTKRenderWindowInteractor()
        )

        self.axial_vtk_widget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.coronal_vtk_widget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.sagittal_vtk_widget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )

        self.axial_vtk_widget.customContextMenuRequested.connect(
            partial(
                self._show_2d_context_menu,
                widget=self.axial_vtk_widget,
                view_type="axial",
            )
        )
        self.coronal_vtk_widget.customContextMenuRequested.connect(
            partial(
                self._show_2d_context_menu,
                widget=self.coronal_vtk_widget,
                view_type="coronal",
            )
        )
        self.sagittal_vtk_widget.customContextMenuRequested.connect(
            partial(
                self._show_2d_context_menu,
                widget=self.sagittal_vtk_widget,
                view_type="sagittal",
            )
        )

        ortho_splitter.addWidget(self.axial_vtk_widget)
        ortho_splitter.addWidget(self.coronal_vtk_widget)
        ortho_splitter.addWidget(self.sagittal_vtk_widget)

        ortho_layout.addWidget(ortho_splitter)
        ortho_container.setLayout(ortho_layout)

        # Add the 2D view container to the bottom
        main_splitter.addWidget(ortho_container)

        # Set initial size ratio (e.g., 70% 3D, 30% 2D)
        main_splitter.setSizes([700, 300])

        # Add the main splitter to the parent widget's layout
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

        # Reduce anti-aliasing for better performance (4 samples vs default 8)
        # Trade-off: Slightly jagged edges, but smoother interaction
        ##TODO - add option menu in Settings to allow user to disable it or change it
        self.render_window.SetMultiSamples(4)

        self.interactor: vtk.vtkRenderWindowInteractor = (
            self.render_window.GetInteractor()
        )
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Axial 2D Scene
        self.axial_scene: window.Scene = window.Scene()
        self.axial_scene.background((0.0, 0.0, 0.0))
        self.axial_render_window: vtk.vtkRenderWindow = (
            self.axial_vtk_widget.GetRenderWindow()
        )
        self.axial_render_window.SetNumberOfLayers(2)  # Enable multiple layers
        self.axial_render_window.AddRenderer(self.axial_scene)
        self.axial_scene.SetLayer(0)  # Base layer for slice

        # Create overlay renderer for axial scale bar (renders on top)
        self.axial_overlay_renderer = vtk.vtkRenderer()
        self.axial_overlay_renderer.SetLayer(1)  # Overlay layer
        self.axial_overlay_renderer.InteractiveOff()
        self.axial_overlay_renderer.PreserveColorBufferOn()
        self.axial_overlay_renderer.SetActiveCamera(self.axial_scene.GetActiveCamera())
        self.axial_render_window.AddRenderer(self.axial_overlay_renderer)

        self.axial_interactor: vtk.vtkRenderWindowInteractor = (
            self.axial_render_window.GetInteractor()
        )
        style_2d_axial = CustomInteractorStyle2D(self)
        self.axial_interactor.SetInteractorStyle(style_2d_axial)

        # Coronal 2D Scene
        self.coronal_scene: window.Scene = window.Scene()
        self.coronal_scene.background((0.0, 0.0, 0.0))
        self.coronal_render_window: vtk.vtkRenderWindow = (
            self.coronal_vtk_widget.GetRenderWindow()
        )
        self.coronal_render_window.SetNumberOfLayers(2)  # Enable multiple layers
        self.coronal_render_window.AddRenderer(self.coronal_scene)
        self.coronal_scene.SetLayer(0)  # Base layer for slice

        # Create overlay renderer for coronal scale bar (renders on top)
        self.coronal_overlay_renderer = vtk.vtkRenderer()
        self.coronal_overlay_renderer.SetLayer(1)  # Overlay layer
        self.coronal_overlay_renderer.InteractiveOff()
        self.coronal_overlay_renderer.PreserveColorBufferOn()
        self.coronal_overlay_renderer.SetActiveCamera(
            self.coronal_scene.GetActiveCamera()
        )
        self.coronal_render_window.AddRenderer(self.coronal_overlay_renderer)

        self.coronal_interactor: vtk.vtkRenderWindowInteractor = (
            self.coronal_render_window.GetInteractor()
        )
        style_2d_coronal = CustomInteractorStyle2D(self)
        self.coronal_interactor.SetInteractorStyle(style_2d_coronal)

        # Sagittal 2D Scene
        self.sagittal_scene: window.Scene = window.Scene()
        self.sagittal_scene.background((0.0, 0.0, 0.0))
        self.sagittal_render_window: vtk.vtkRenderWindow = (
            self.sagittal_vtk_widget.GetRenderWindow()
        )
        self.sagittal_render_window.SetNumberOfLayers(2)  # Enable multiple layers
        self.sagittal_render_window.AddRenderer(self.sagittal_scene)
        self.sagittal_scene.SetLayer(0)  # Base layer for slice

        # Create overlay renderer for sagittal crosshair and scale bar (renders on top)
        self.sagittal_overlay_renderer = vtk.vtkRenderer()
        self.sagittal_overlay_renderer.SetLayer(1)  # Overlay layer
        self.sagittal_overlay_renderer.InteractiveOff()  # Don't intercept mouse events
        self.sagittal_overlay_renderer.PreserveColorBufferOn()  # Keep background from layer 0
        self.sagittal_overlay_renderer.SetActiveCamera(
            self.sagittal_scene.GetActiveCamera()
        )
        self.sagittal_render_window.AddRenderer(self.sagittal_overlay_renderer)

        self.sagittal_interactor: vtk.vtkRenderWindowInteractor = (
            self.sagittal_render_window.GetInteractor()
        )
        style_2d_sagittal = CustomInteractorStyle2D(self)
        self.sagittal_interactor.SetInteractorStyle(style_2d_sagittal)

        # --- Actor Placeholders ---
        self.streamlines_actor: Optional[vtk.vtkActor] = None
        self.odf_actor: Optional[vtk.vtkActor] = None
        self.highlight_actor: Optional[vtk.vtkActor] = None
        self.radius_actor: Optional[vtk.vtkActor] = None
        self.current_radius_actor_radius: Optional[float] = None
        self.axial_slice_actor: Optional[vtk.vtkActor] = None
        self.coronal_slice_actor: Optional[vtk.vtkActor] = None
        self.sagittal_slice_actor: Optional[vtk.vtkActor] = None
        self.axial_slice_actor_2d: Optional[vtk.vtkActor] = None
        self.coronal_slice_actor_2d: Optional[vtk.vtkActor] = None
        self.sagittal_slice_actor_2d: Optional[vtk.vtkActor] = None
        self.status_text_actor: Optional[vtk.vtkTextActor] = None
        self.instruction_text_actor: Optional[vtk.vtkTextActor] = None
        self.axes_actor: Optional[vtk.vtkActor] = None
        self.orientation_widget: Optional[vtk.vtkOrientationMarkerWidget] = None
        self.orientation_labels: List[vtk.vtkTextActor] = []

        # --- Crosshair Actors ---
        self.axial_crosshair_actor: Optional[vtk.vtkActor] = None
        self.coronal_crosshair_actor: Optional[vtk.vtkActor] = None
        self.sagittal_crosshair_actor: Optional[vtk.vtkActor] = None
        self.crosshair_lines: Dict[str, vtk.vtkLineSource] = (
            {}
        )  # Stores vtkLineSource objects for updates
        self.crosshair_appenders: Dict[str, vtk.vtkAppendPolyData] = (
            {}
        )  # Stores vtkAppendPolyData

        self.is_navigating_2d: bool = False

        # Drawing Mode State
        self.is_drawing_mode: bool = False
        self.is_drawing_active: bool = (
            False  # Tracks if mouse button is down while drawing
        )
        self.brush_size: int = 1
        self.draw_stroke_count: int = 0
        self.update_interval: int = 15

        # Preview-based drawing: collect points, show preview, convert on release
        self.drawing_preview_points: list = (
            []
        )  # List of (world_x, world_y, world_z, view_type)
        self.preview_line_actor: Optional[vtk.vtkActor] = None

        self.sphere_params_per_roi: Dict[str, Dict[str, Any]] = (
            {}
        )  # Key: roi_name, Val: {center, radius, view_type}
        self.rectangle_params_per_roi: Dict[str, Dict[str, Any]] = (
            {}
        )  # Key: roi_name, Val: {start, end, view_type}

        self.roi_slice_actors: Dict[str, Dict[str, vtk.vtkActor]] = (
            {}
        )  # Key: path, Val: {'axial':, 'coronal':, 'sagittal':}

        self.roi_highlight_actor: Optional[vtk.vtkActor] = None

        self.scalar_bar_actor: Optional[vtk.vtkScalarBarActor] = None

        # --- Scene Manager (handles camera, crosshairs, UI labels) ---
        self.scene_manager: SceneManager = SceneManager(self)

        # --- Drawing Manager (handles ROI drawing operations) ---
        self.drawing_manager: DrawingManager = DrawingManager(self)

        # --- Selection Manager (handles streamline selection) ---
        self.selection_manager: SelectionManager = SelectionManager(self)

        # --- Streamlines Manager (handles streamline visualization) ---
        self.streamlines_manager: StreamlinesManager = StreamlinesManager(self)

        # --- Scale Bar Manager (handles 2D view scale bars) ---
        self.scale_bar_manager: ScaleBarManager = ScaleBarManager()

        # --- Fast Render Mode State ---
        # Reduces rendering quality during mouse interactions for smoother UX
        # VTK DesiredUpdateRate: HIGHER = faster rendering (lower quality)
        #                        LOWER = slower rendering (higher quality)
        self._is_interacting: bool = False
        self._original_desired_update_rate: float = 0.0001  # Low = high quality at rest
        self._fast_update_rate: float = (
            30.0  # High = fast/low quality during interaction
        )

        # --- Overlay Progress Bar ---
        self.overlay_progress_bar = QProgressBar(self.vtk_widget)
        self.overlay_progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #333;
                color: white;
                text-align: center;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
            }
        """
        )
        self.overlay_progress_bar.hide()

        # --- Slice Navigation State ---
        self.current_slice_indices: Dict[str, Optional[int]] = {
            "x": None,
            "y": None,
            "z": None,
        }
        self.image_shape_vox: Optional[Tuple[int, ...]] = None
        self.image_extents: Dict[str, Optional[Tuple[int, int]]] = {
            "x": None,
            "y": None,
            "z": None,
        }
        # Full-resolution shape from mmap image (used for 2D slice navigation)
        self.mmap_image_shape: Optional[Tuple[int, int, int]] = None
        self.mmap_image_extents: Dict[str, Optional[Tuple[int, int]]] = {
            "x": None,
            "y": None,
            "z": None,
        }

        self._create_scene_ui()

        # --- Setup Interaction Callbacks ---
        self.interactor.AddObserver(
            vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0
        )
        self.axial_interactor.AddObserver(
            vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0
        )
        self.coronal_interactor.AddObserver(
            vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0
        )
        self.sagittal_interactor.AddObserver(
            vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0
        )

        # --- Fast Render Mode Observers (3D view only) ---
        # VTK events fire opposite to expected: StartInteraction = mouse released,
        # EndInteraction = mouse pressed. So we swap the handlers.
        self.interactor.AddObserver(
            vtk.vtkCommand.StartInteractionEvent,
            lambda obj, event: self._disable_fast_render(),
        )
        self.interactor.AddObserver(
            vtk.vtkCommand.EndInteractionEvent,
            lambda obj, event: self._enable_fast_render(),
        )

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

        # Update Slice Actors
        for act in [
            self.axial_slice_actor,
            self.coronal_slice_actor,
            self.sagittal_slice_actor,
        ]:
            if act:
                act.SetVisibility(vis_flag)

        # Update 2D Slice Actors as well
        for act in [
            self.axial_slice_actor_2d,
            self.coronal_slice_actor_2d,
            self.sagittal_slice_actor_2d,
        ]:
            if act:
                act.SetVisibility(vis_flag)

        # Update Crosshair Actors
        for act in [
            self.axial_crosshair_actor,
            self.coronal_crosshair_actor,
            self.sagittal_crosshair_actor,
        ]:
            if act:
                act.SetVisibility(vis_flag)

        self._render_all()

    def update_roi_highlight_actor(self) -> None:
        """
        Updates the Red highlight actor based on ROI 'Select' indices.
        Delegates to StreamlinesManager.
        """
        self.streamlines_manager.update_roi_highlight_actor()

    def _render_all(self) -> None:
        """Renders all four VTK windows if they are initialized."""
        try:
            if (
                self.render_window
                and self.render_window.GetInteractor().GetInitialized()
            ):
                self.render_window.Render()
            if (
                self.axial_render_window
                and self.axial_render_window.GetInteractor().GetInitialized()
            ):
                self.axial_render_window.Render()
            if (
                self.coronal_render_window
                and self.coronal_render_window.GetInteractor().GetInitialized()
            ):
                self.coronal_render_window.Render()
            if (
                self.sagittal_render_window
                and self.sagittal_render_window.GetInteractor().GetInitialized()
            ):
                self.sagittal_render_window.Render()
        except Exception as e:
            logger.warning(f"Warning: Error during _render_all: {e}")

    # --- Fast Render Mode Methods ---
    def _on_interaction_start(self, obj: vtk.vtkObject, event: str) -> None:
        """Called when user starts rotating/panning/zooming the 3D view."""
        self._enable_fast_render()

    def _on_interaction_end(self, obj: vtk.vtkObject, event: str) -> None:
        """Called when user stops interacting with the 3D view."""
        self._disable_fast_render()

    def _enable_fast_render(self) -> None:
        """
        Enables fast render mode for smoother interaction.

        Reduces rendering quality for speed during mouse interaction.
        Also performs a GPU "pre-warm" to eliminate initial stuttering
        when the GPU has been idle.
        """
        if self._is_interacting:
            return
        self._is_interacting = True

        # High update rate = VTK tries to hit framerate by reducing quality
        if self.render_window:
            self.render_window.SetDesiredUpdateRate(self._fast_update_rate)

        # Flat shading = faster, no lighting calculations per vertex
        if self.streamlines_actor:
            prop = self.streamlines_actor.GetProperty()
            if prop:
                prop.SetInterpolationToFlat()

        ## TODO - this might not work well on some configurations (integrated GPUs), or cause visual artifacts.
        # will be moved to Settings, allowing user to disable it if needed ##
        # Disable antialiasing
        if self.render_window:
            self.render_window.SetMultiSamples(0)

            # In theory, VTK should be able to scale depending on the GPU capabilities (integrated and dedicated), but seems odd.
            ## TODO - this might not work well on some configurations (integrated GPUs), or cause visual artifacts.
            # GPU Pre-Warm: Force an immediate render to wake up the GPU
            # This eliminates the 1-2 second stutter when starting interaction after the GPU has been idle
            try:
                self.render_window.Render()
            except Exception:
                pass  # Ignore any errors during pre-warm

    def _disable_fast_render(self) -> None:
        """
        Restores full quality rendering after interaction stops.

        Re-enables smooth shading and high quality for static viewing.
        """
        if not self._is_interacting:
            return
        self._is_interacting = False

        # Low update rate = VTK can take time for quality
        if self.render_window:
            self.render_window.SetDesiredUpdateRate(self._original_desired_update_rate)

        # Gouraud shading = smooth, calculates lighting per vertex
        if self.streamlines_actor:
            prop = self.streamlines_actor.GetProperty()
            if prop:
                prop.SetInterpolationToGouraud()

        # Enable antialiasing
        if self.render_window:
            self.render_window.SetMultiSamples(4)

        # Force immediate high-quality re-render
        if self.render_window and self.render_window.GetInteractor().GetInitialized():
            self.render_window.Render()

    def update_odf_actor(
        self,
        odf_amplitudes: np.ndarray,
        sphere: Any,
        affine: np.ndarray,
        extent: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """
        Creates/Updates the ODF glyph actor.

        Args:
            odf_amplitudes: 4D array (x, y, z, N_vertices) of SF amplitudes.
            sphere: The SimpleSphere object used for reconstruction.
            affine: The affine of the ODF volume.
            extent: (min_x, max_x, min_y, max_y, min_z, max_z) in voxel coordinates.
        """
        if not self.scene:
            return

        # Remove existing
        if self.odf_actor:
            self.scene.rm(self.odf_actor)
            self.odf_actor = None

        if odf_amplitudes is None:
            self.render_window.Render()
            return

        try:
            # Create ODF Slicer Actor
            self.odf_actor = actor.odf_slicer(
                odf_amplitudes,
                sphere=sphere,
                affine=affine,
                scale=0.5,
                norm=True,
                global_cm=False,
            )
            # Set the display extent
            if extent is not None:
                self.odf_actor.display_extent(*extent)

            self.scene.add(self.odf_actor)

            # Optional: Add to 2D scenes if we want them visible there too (heavy on performance and bit messy)
            # if self.axial_scene: self.axial_scene.add(self.odf_actor)
            # if self.coronal_scene: self.coronal_scene.add(self.odf_actor)
            # if self.sagittal_scene: self.sagittal_scene.add(self.odf_actor)

            self.render_window.Render()
            self.update_status("ODF Glyphs updated.")

        except Exception as e:
            logger.error(f"Error creating ODF actor: {e}", exc_info=True)
            self.update_status("Error creating ODF actor.")

    def remove_odf_actor(self) -> None:
        """Removes the ODF actor from the scene."""
        if self.odf_actor and self.scene:
            self.scene.rm(self.odf_actor)
            self.odf_actor = None
            self.render_window.Render()

    def set_drawing_mode(
        self,
        enabled: bool,
        is_eraser: bool = False,
        is_sphere: bool = False,
        is_rectangle: bool = False,
    ) -> None:
        """Enables or disables drawing mode for 2D panels."""
        self.drawing_manager.set_drawing_mode(
            enabled, is_eraser, is_sphere, is_rectangle
        )

    def _handle_draw_on_2d(self, interactor: vtk.vtkRenderWindowInteractor) -> None:
        """Handles drawing on 2D views. Delegates to DrawingManager."""
        self.drawing_manager.handle_draw_on_2d(interactor)

    def _update_roi_data_fast(self, roi_name: str, data: np.ndarray) -> None:
        """Fast in-place update of ROI data without recreating actors (no flicker)."""
        if roi_name not in self.roi_slice_actors:
            return

        try:
            # Just trigger a render - data is already updated in memory
            # The VTK actors will use the updated numpy array reference
            self._render_all()
        except Exception as e:
            logger.error(f"Error in fast ROI update: {e}", exc_info=True)

    def _update_drawing_preview(self, scene: window.Scene) -> None:
        """Updates the preview line on the active 2D scene. Delegates to DrawingManager."""
        self.drawing_manager._update_drawing_preview(scene)

    def _finish_drawing(self) -> None:
        """Rasterizes the preview path into the ROI volume. Delegates to DrawingManager."""
        self.drawing_manager.finish_drawing()

    def _remove_actor_set(self, actor_dict: Dict[str, vtk.vtkActor]) -> None:
        """Removes a set of actors from all scenes."""
        scenes_to_check = [
            self.scene,
            self.axial_scene,
            self.coronal_scene,
            self.sagittal_scene,
        ]
        for act in actor_dict.values():
            if act is not None:
                for scn in scenes_to_check:
                    try:
                        scn.rm(act)
                    except Exception:
                        pass

    def _adjust_sphere_radius(self, delta: float, view_type: str = "axial") -> None:
        """Adjusts the radius of the last created sphere. Delegates to DrawingManager."""
        self.drawing_manager.adjust_sphere_radius(delta, view_type)

    def _update_roi_data_in_place(self, key: str, data: np.ndarray) -> None:
        """
        Updates the underlying VTK image data for an ROI without recreating actors.
        This prevents flickering.
        """
        if key not in self.roi_slice_actors:
            return

        for actor_key, actor_obj in self.roi_slice_actors[key].items():
            if actor_obj is None:
                continue

            # Skip non-actor entries (e.g., 'sphere_source' is a vtkSphereSource)
            if not hasattr(actor_obj, "GetMapper"):
                continue

            # Retrieve the vtkImageData
            mapper = actor_obj.GetMapper()
            if not mapper:
                continue

            # The mapper input might be the image data directly
            vtk_input = mapper.GetInput()

            if isinstance(vtk_input, vtk.vtkImageData):
                self._update_vtk_image_scalars(vtk_input, data)
            else:
                # Fallback: try to find input connection (e.g. if Reslice filter is used)
                try:
                    conn = mapper.GetInputConnection(0, 0)
                    if conn:
                        producer = conn.GetProducer()
                        # If producer is a filter (like vtkImageReslice), get its input
                        if hasattr(producer, "GetInputConnection"):
                            input_conn = producer.GetInputConnection(0, 0)
                            if input_conn:
                                real_input = input_conn.GetProducer().GetOutput()
                                if isinstance(real_input, vtk.vtkImageData):
                                    self._update_vtk_image_scalars(real_input, data)
                except Exception:
                    pass

    def _update_vtk_image_scalars(
        self, image_data: vtk.vtkImageData, numpy_data: np.ndarray
    ) -> None:
        """
        Efficiently updates the scalars of a vtkImageData object from a numpy array.
        """
        from vtk.util import numpy_support

        # Ensure data is contiguous and of the right type
        if not numpy_data.flags.c_contiguous:
            numpy_data = np.ascontiguousarray(numpy_data)

        # Create VTK array
        # Note: assuming the numpy array shape matches the image dimensions
        vtk_array = numpy_support.numpy_to_vtk(
            num_array=numpy_data.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )

        # Update point data
        image_data.GetPointData().SetScalars(vtk_array)
        image_data.Modified()

    def update_roi_layer(self, key: str, data: np.ndarray, affine: np.ndarray) -> None:
        """
        Updates an existing ROI layer with new data using a 'Swap' strategy.
        Adds new actors BEFORE removing old ones to prevent flickering.
        """
        # Keep reference to old actors
        old_actors = self.roi_slice_actors.get(key)

        # Add new actors (render=False to avoid intermediate frame)
        self.add_roi_layer(key, data, affine, render=False)

        # Remove old actors
        if old_actors:
            self._remove_actor_set(old_actors)

        # Restore assigned color for manual ROIs
        if "manual_roi" in key:
            assigned_color = (1.0, 0.0, 0.0)
            if self.main_window and key in self.main_window.roi_layers:
                assigned_color = self.main_window.roi_layers[key].get(
                    "color", (1.0, 0.0, 0.0)
                )
            self.set_roi_layer_color(key, assigned_color)

        # Final Render
        self._render_all()

        # Explicitly render sagittal window
        if (
            self.sagittal_render_window
            and self.sagittal_render_window.GetInteractor().GetInitialized()
        ):
            self.sagittal_render_window.Render()

    def _show_2d_context_menu(
        self, position: QPoint, widget: QVTKRenderWindowInteractor, view_type: str
    ) -> None:
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
        reset_action.triggered.connect(
            partial(self._reset_2d_view, view_type=view_type)
        )

        menu.addAction(reset_action)

        # Show the menu at the global position of the click
        menu.exec(widget.mapToGlobal(position))

    def _reset_2d_view(self, view_type: str) -> None:
        """Resets the camera for a specific 2D view and re-renders it."""
        self.scene_manager.reset_2d_view(view_type)

    def _setup_ortho_cameras(self) -> None:
        """Sets the cameras for the 2D views to be orthogonal."""
        self.scene_manager.setup_ortho_cameras()

    def _update_axial_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the axial 2D camera to follow its slice."""
        self.scene_manager.update_axial_camera(reset_zoom_pan)

    def _update_coronal_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the coronal 2D camera to follow its slice."""
        self.scene_manager.update_coronal_camera(reset_zoom_pan)

    def _update_sagittal_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the sagittal 2D camera to follow its slice."""
        self.scene_manager.update_sagittal_camera(reset_zoom_pan)

    def _create_scene_ui(self) -> None:
        """Creates the initial UI elements (Axes, Text) within the VTK scene."""
        # Delegate to SceneManager
        self.scene_manager.create_scene_ui()

    def update_status(self, message: str) -> None:
        """Updates the status text displayed in the VTK window."""
        if self.status_text_actor is None:
            return

        try:
            undo_possible = False
            redo_possible = False
            data_loaded = False
            current_radius = (
                self.main_window.selection_radius_3d if self.main_window else 5.0
            )

            if self.main_window:
                undo_possible = bool(self.main_window.unified_undo_stack)
                redo_possible = bool(self.main_window.unified_redo_stack)
                data_loaded = bool(self.main_window.tractogram_data)

            status_suffix = ""
            if "Deleted" in message and undo_possible:
                status_suffix = " (Ctrl+Z to Undo)"
            elif "Undone" in message or "undone" in message:
                status_suffix = (
                    f" ({len(self.main_window.unified_undo_stack)} undo remaining"
                )
                status_suffix += ", Ctrl+Y to Redo)" if redo_possible else ")"
            elif "Redone" in message or "redone" in message:
                status_suffix = (
                    f" ({len(self.main_window.unified_redo_stack)} redo remaining"
                )
                status_suffix += ", Ctrl+Z to Undo)" if undo_possible else ")"

            prefix = f"[Radius: {current_radius:.1f}mm] " if data_loaded else ""

            # Avoid prefixing slice navigation messages
            if message.startswith("Slice View:"):
                prefix = ""
                status_suffix = ""

            full_message = f"Status: {prefix}{message}{status_suffix}"
            self.status_text_actor.SetInput(str(full_message))

            if (
                self.render_window
                and self.render_window.GetInteractor().GetInitialized()
            ):
                self.render_window.Render()
        except Exception as e:
            logger.error(f"Error updating status text actor:", exc_info=e)

    def update_progress_bar(
        self, value: int, maximum: int, visible: bool = True
    ) -> None:
        """
        Updates the overlay progress bar positioned at the bottom of the VTK window.
        """
        if not visible:
            self.overlay_progress_bar.hide()
            return

        # Position it dynamically at the bottom, to the right of the status text
        w = self.vtk_widget.width()
        h = self.vtk_widget.height()

        # Size
        bar_w = 200
        bar_h = 15

        # Position
        x_pos = 500
        y_pos = h - bar_h - 9  # 9px margin from bottom

        if x_pos + bar_w > w:
            x_pos = max(10, w - bar_w - 5)

        self.overlay_progress_bar.setGeometry(x_pos, y_pos, bar_w, bar_h)

        self.overlay_progress_bar.setRange(0, maximum)
        self.overlay_progress_bar.setValue(value)
        self.overlay_progress_bar.show()

    # Selection Sphere Actor Management
    def _ensure_radius_actor_exists(
        self, radius: float, center_point: Tuple[float, float, float]
    ) -> None:
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
        self.radius_actor.SetPickable(
            0
        )  # Exclude from picking to avoid sphere moving towards camera on repeated 's' presses
        prop = self.radius_actor.GetProperty()
        prop.SetColor(0.2, 0.5, 1.0)
        prop.SetOpacity(0.3)
        prop.SetRepresentationToWireframe()
        prop.SetLineWidth(1.0)
        self.scene.add(self.radius_actor)
        self.current_radius_actor_radius = radius
        self.radius_actor.SetVisibility(0)

    def _update_existing_radius_actor(
        self,
        center_point: Optional[Tuple[float, float, float]],
        radius: Optional[float],
        visible: bool,
    ) -> bool:
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

    def update_radius_actor(
        self,
        center_point: Optional[Tuple[float, float, float]] = None,
        radius: Optional[float] = None,
        visible: bool = False,
    ) -> None:
        """Creates or updates the selection sphere actor (standard VTK)."""
        if not self.scene:
            return
        if radius is None and self.main_window:
            radius = self.main_window.selection_radius_3d

        needs_render = False
        if visible and center_point is not None and radius is not None:
            if self.radius_actor is None:
                self._ensure_radius_actor_exists(radius, center_point)
                if self.radius_actor:
                    self.radius_actor.SetVisibility(1 if visible else 0)
                    needs_render = True
            else:
                needs_render = self._update_existing_radius_actor(
                    center_point, radius, visible
                )
        elif self.radius_actor is not None:
            needs_render = self._update_existing_radius_actor(None, None, visible)

        if (
            needs_render
            and self.render_window
            and self.render_window.GetInteractor().GetInitialized()
        ):
            self.render_window.Render()

    # Anatomical Slice Actor Management
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
        if not image_data.flags["C_CONTIGUOUS"]:
            image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        else:
            image_data = image_data.astype(np.float32, copy=False)

        # Get dimensions and extents
        shape = image_data.shape[:3]

        self.image_shape_vox = shape
        self.image_extents["x"] = (0, shape[0] - 1)
        self.image_extents["y"] = (0, shape[1] - 1)
        self.image_extents["z"] = (0, shape[2] - 1)

        # Store full-resolution shape from mmap image if available
        mmap_img = getattr(self.main_window, "anatomical_mmap_image", None)
        if mmap_img is not None:
            mmap_shape = mmap_img.shape
            self.mmap_image_shape = mmap_shape
            self.mmap_image_extents["x"] = (0, mmap_shape[0] - 1)
            self.mmap_image_extents["y"] = (0, mmap_shape[1] - 1)
            self.mmap_image_extents["z"] = (0, mmap_shape[2] - 1)
            logger.info(
                f"Mmap image available: full-res shape {mmap_shape}, "
                f"display shape {shape}"
            )
        else:
            self.mmap_image_shape = None
            self.mmap_image_extents = {"x": None, "y": None, "z": None}

        # Initialize or keep current slice indices
        if (
            self.current_slice_indices["x"] is None
            or self.current_slice_indices["x"] > self.image_extents["x"][1]
        ):
            self.current_slice_indices["x"] = shape[0] // 2
        if (
            self.current_slice_indices["y"] is None
            or self.current_slice_indices["y"] > self.image_extents["y"][1]
        ):
            self.current_slice_indices["y"] = shape[1] // 2
        if (
            self.current_slice_indices["z"] is None
            or self.current_slice_indices["z"] > self.image_extents["z"][1]
        ):
            self.current_slice_indices["z"] = shape[2] // 2

        current_x = self.current_slice_indices["x"]
        current_y = self.current_slice_indices["y"]
        current_z = self.current_slice_indices["z"]
        x_extent = self.image_extents["x"]
        y_extent = self.image_extents["y"]
        z_extent = self.image_extents["z"]

        # Clear existing slicer actors
        self.clear_anatomical_slices(reset_state=False)

        try:
            # Determine Value Range
            img_min = np.min(image_data)
            img_max = np.max(image_data)

            if not np.isfinite(img_min) or not np.isfinite(img_max):
                logger.warning("Image contains non-finite values. Clamping range.")
                finite_data = image_data[np.isfinite(image_data)]
                if finite_data.size > 0:
                    img_min, img_max = np.min(finite_data), np.max(finite_data)
                else:
                    img_min, img_max = 0.0, 1.0

            # Ensure range has distinct min/max
            if img_max <= img_min:
                value_range = (img_min - 1.0, img_max + 1.0)
            else:
                value_range = (float(img_min), float(img_max))

            # Get opacity from MainWindow state
            slicer_opacity = getattr(self.main_window, "image_opacity", 1.0)
            interpolation_mode = "nearest"

            # Create Slicer Actors using FURY's default gray LUT
            # Axial Slice (Z plane)
            self.axial_slice_actor = actor.slicer(
                image_data,
                affine=affine,
                value_range=value_range,
                opacity=slicer_opacity,
                interpolation=interpolation_mode,
            )
            self.axial_slice_actor.display_extent(
                x_extent[0], x_extent[1], y_extent[0], y_extent[1], current_z, current_z
            )
            self.scene.add(self.axial_slice_actor)

            # 2D Actor (Uses FIXED opacity of 1.0)
            self.axial_slice_actor_2d = actor.slicer(
                image_data,
                affine=affine,
                value_range=value_range,
                opacity=1.0,
                interpolation=interpolation_mode,
            )
            self.axial_slice_actor_2d.display_extent(
                x_extent[0], x_extent[1], y_extent[0], y_extent[1], current_z, current_z
            )
            if self.axial_scene:
                self.axial_scene.add(self.axial_slice_actor_2d)
                # X-flip: Convert FURY's neurological output to radiological display convention
                self.axial_slice_actor_2d.SetScale(-1, 1, 1)

            # Coronal Slice (Y plane)
            self.coronal_slice_actor = actor.slicer(
                image_data,
                affine=affine,
                value_range=value_range,
                opacity=slicer_opacity,
                interpolation=interpolation_mode,
            )
            self.coronal_slice_actor.display_extent(
                x_extent[0], x_extent[1], current_y, current_y, z_extent[0], z_extent[1]
            )
            self.scene.add(self.coronal_slice_actor)

            # 2D Actor
            self.coronal_slice_actor_2d = actor.slicer(
                image_data,
                affine=affine,
                value_range=value_range,
                opacity=1.0,
                interpolation=interpolation_mode,
            )
            self.coronal_slice_actor_2d.display_extent(
                x_extent[0], x_extent[1], current_y, current_y, z_extent[0], z_extent[1]
            )
            if self.coronal_scene:
                self.coronal_scene.add(self.coronal_slice_actor_2d)
                # X-flip: Convert FURY's neurological output to radiological display convention
                self.coronal_slice_actor_2d.SetScale(-1, 1, 1)

            # Sagittal Slice (X plane)
            self.sagittal_slice_actor = actor.slicer(
                image_data,
                affine=affine,
                value_range=value_range,
                opacity=slicer_opacity,
                interpolation=interpolation_mode,
            )
            self.sagittal_slice_actor.display_extent(
                current_x, current_x, y_extent[0], y_extent[1], z_extent[0], z_extent[1]
            )
            self.scene.add(self.sagittal_slice_actor)

            # 2D Actor
            self.sagittal_slice_actor_2d = actor.slicer(
                image_data,
                affine=affine,
                value_range=value_range,
                opacity=1.0,
                interpolation=interpolation_mode,
            )
            self.sagittal_slice_actor_2d.display_extent(
                current_x, current_x, y_extent[0], y_extent[1], z_extent[0], z_extent[1]
            )
            if self.sagittal_scene:
                self.sagittal_scene.add(self.sagittal_slice_actor_2d)

            # Create crosshair actors
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

        # Initialize scale bars for 2D views
        try:
            self.scale_bar_manager.initialize(
                axial_renderer=self.axial_overlay_renderer,
                coronal_renderer=self.coronal_overlay_renderer,
                sagittal_renderer=self.sagittal_overlay_renderer,
                axial_camera=self.axial_scene.GetActiveCamera(),
                coronal_camera=self.coronal_scene.GetActiveCamera(),
                sagittal_camera=self.sagittal_scene.GetActiveCamera(),
            )
            self.scale_bar_manager.update_all()
        except Exception as e:
            logger.warning(f"Failed to initialize scale bars: {e}")

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
            a
            for a in [
                self.axial_slice_actor,
                self.coronal_slice_actor,
                self.sagittal_slice_actor,
                self.axial_slice_actor_2d,
                self.coronal_slice_actor_2d,
                self.sagittal_slice_actor_2d,
            ]
            if a is not None
        ]

        scenes_to_check = [
            s
            for s in [
                self.scene,
                self.axial_scene,
                self.coronal_scene,
                self.sagittal_scene,
            ]
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
        self.axial_slice_actor_2d = None
        self.coronal_slice_actor_2d = None
        self.sagittal_slice_actor_2d = None

        # Clear scale bars
        self.scale_bar_manager.clear()

        if reset_state:
            self.current_slice_indices = {"x": None, "y": None, "z": None}
            self.image_shape_vox = None
            self.image_extents = {"x": None, "y": None, "z": None}
            self.mmap_image_shape = None
            self.mmap_image_extents = {"x": None, "y": None, "z": None}

        if scene_changed:
            self._render_all()

    def _debug_roi_alignment(
        self, key: str, main_vox: np.ndarray, roi_vox: np.ndarray
    ) -> None:
        """Debug method to check ROI alignment issues."""
        main_world = self._voxel_to_world(
            main_vox, self.main_window.anatomical_image_affine
        )
        roi_world = self._voxel_to_world(
            roi_vox, self.main_window.roi_layers[key]["affine"]
        )

        logger.info(f"  Main voxel: {main_vox} -> World: {main_world}")
        logger.info(f"  ROI voxel: {roi_vox} -> World: {roi_world}")
        logger.info(f"  World difference: {np.linalg.norm(main_world - roi_world)}")

        # Check if orientations match
        main_orient = nib.aff2axcodes(self.main_window.anatomical_image_affine)
        roi_orient = nib.aff2axcodes(self.main_window.roi_layers[key]["affine"])
        logger.info(f"  Main orientation: {main_orient}")
        logger.info(f"  ROI orientation: {roi_orient}")

    def set_streamlines_opacity(self, value: float) -> None:
        """Sets the opacity of the main streamlines actor."""
        if self.streamlines_actor:
            self.streamlines_actor.GetProperty().SetOpacity(value)
            self.render_window.Render()

    def set_anatomical_opacity(self, value: float) -> None:
        """Sets the opacity of the anatomical slices."""
        for act in [
            self.axial_slice_actor,
            self.coronal_slice_actor,
            self.sagittal_slice_actor,
        ]:
            if act:
                act.GetProperty().SetOpacity(value)
        self.render_window.Render()

    def set_roi_opacity(self, key: str, value: float) -> None:
        """Sets the opacity for a specific ROI layer."""
        if key in self.roi_slice_actors:
            for act in self.roi_slice_actors[
                key
            ].values():  # Update the opacity for all the actors
                if act:
                    act.GetProperty().SetOpacity(value)

            self._render_all()

    # Clear Crosshairs
    def _clear_crosshairs(self) -> None:
        """Removes the 2D crosshair actors from their scenes."""
        self.scene_manager.clear_crosshairs()

    # Voxel <-> World Coordinate Helpers
    def _voxel_to_world(
        self, vox_coord: List[float], affine: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Converts a voxel index [i, j, k] to world RASmm coordinates [x, y, z]."""
        if affine is None:
            if (
                self.main_window is None
                or self.main_window.anatomical_image_affine is None
            ):
                return np.array(vox_coord)  # Fallback
            affine = self.main_window.anatomical_image_affine

        homog_vox = np.array([vox_coord[0], vox_coord[1], vox_coord[2], 1.0])
        world_coord = np.dot(affine, homog_vox)
        return world_coord[:3]

    def _world_to_voxel(
        self, world_coord: List[float], inv_affine: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Converts world RASmm coordinates [x, y, z] to voxel indices [i, j, k]."""
        if inv_affine is None:
            if (
                self.main_window is None
                or self.main_window.anatomical_image_affine is None
            ):
                return np.array(world_coord)  # Fallback
            affine = self.main_window.anatomical_image_affine
            inv_affine = np.linalg.inv(affine)

        homog_world = np.array([world_coord[0], world_coord[1], world_coord[2], 1.0])
        vox_coord = np.dot(inv_affine, homog_world)
        return vox_coord[:3]  # Return float voxel coords

    # Create/Update Crosshairs
    def _create_or_update_crosshairs(self) -> None:
        """Creates or updates the 2D crosshair actors based on current slice indices."""
        self.scene_manager.create_or_update_crosshairs()

    # Move Slice
    def move_slice(self, axis: str, direction: int) -> None:
        """Moves the specified slice ('x', 'y', 'z') by 'direction' (+1 or -1)."""
        if (
            not self.image_shape_vox
            or axis not in self.current_slice_indices
            or self.current_slice_indices[axis] is None
        ):
            return

        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_index = axis_map[axis]

        current_index = self.current_slice_indices[axis]
        max_index = self.image_shape_vox[axis_index] - 1

        new_index = current_index + direction
        new_index = max(0, min(new_index, max_index))  # Clamp

        if new_index == current_index:
            self.update_status(
                f"Slice View: Reached {axis.upper()} limit ({new_index})."
            )
            return

        # Delegate to the master update function
        kwargs = {axis: new_index}
        self.set_slice_indices(**kwargs)
        self._update_slow_slice_components()

    # Master Slice Update Function
    def set_slice_indices(
        self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None
    ) -> None:
        """
        Sets the slice indices, updates anatomical slices and crosshairs.
        This is called rapidly during a mouse drag.
        """
        if (
            not self.image_shape_vox
            or not self.axial_slice_actor
            or not self.coronal_slice_actor
            or not self.sagittal_slice_actor
        ):
            return

        # Determine new indices and clamp them
        current = self.current_slice_indices

        new_x = x if x is not None else current["x"]
        new_y = y if y is not None else current["y"]
        new_z = z if z is not None else current["z"]

        new_x = max(self.image_extents["x"][0], min(new_x, self.image_extents["x"][1]))
        new_y = max(self.image_extents["y"][0], min(new_y, self.image_extents["y"][1]))
        new_z = max(self.image_extents["z"][0], min(new_z, self.image_extents["z"][1]))

        # Check if anything changed
        if new_x == current["x"] and new_y == current["y"] and new_z == current["z"]:
            return

        # Update state
        self.current_slice_indices = {"x": new_x, "y": new_y, "z": new_z}

        try:
            # Update all 3D slice actors (FAST)
            x_ext = self.image_extents["x"]
            y_ext = self.image_extents["y"]
            z_ext = self.image_extents["z"]

            # Track which slice actually moved
            slice_moved = {"x": False, "y": False, "z": False}

            if new_x != current["x"]:
                self.sagittal_slice_actor.display_extent(
                    new_x, new_x, y_ext[0], y_ext[1], z_ext[0], z_ext[1]
                )
                if self.sagittal_slice_actor_2d:
                    self.sagittal_slice_actor_2d.display_extent(
                        new_x, new_x, y_ext[0], y_ext[1], z_ext[0], z_ext[1]
                    )
                slice_moved["x"] = True

            if new_y != current["y"]:
                self.coronal_slice_actor.display_extent(
                    x_ext[0], x_ext[1], new_y, new_y, z_ext[0], z_ext[1]
                )
                if self.coronal_slice_actor_2d:
                    self.coronal_slice_actor_2d.display_extent(
                        x_ext[0], x_ext[1], new_y, new_y, z_ext[0], z_ext[1]
                    )
                slice_moved["y"] = True

            if new_z != current["z"]:
                self.axial_slice_actor.display_extent(
                    x_ext[0], x_ext[1], y_ext[0], y_ext[1], new_z, new_z
                )
                if self.axial_slice_actor_2d:
                    self.axial_slice_actor_2d.display_extent(
                        x_ext[0], x_ext[1], y_ext[0], y_ext[1], new_z, new_z
                    )
                slice_moved["z"] = True

            # Update all ROI Slicers
            if self.main_window and self.roi_slice_actors and self.image_extents["x"]:
                c_x = (self.image_extents["x"][0] + self.image_extents["x"][1]) / 2.0
                c_y = (self.image_extents["y"][0] + self.image_extents["y"][1]) / 2.0
                c_z = (self.image_extents["z"][0] + self.image_extents["z"][1]) / 2.0

                sag_plane_vox_center = np.array([new_x, c_y, c_z, 1.0])
                cor_plane_vox_center = np.array([c_x, new_y, c_z, 1.0])
                ax_plane_vox_center = np.array([c_x, c_y, new_z, 1.0])

                for key, actor_dict in self.roi_slice_actors.items():
                    roi_info = self.main_window.roi_layers.get(key)
                    if not roi_info or "T_main_to_roi" not in roi_info:
                        continue

                    T_main_to_roi = roi_info["T_main_to_roi"]
                    roi_shape = roi_info["data"].shape
                    roi_x_ext = (0, roi_shape[0] - 1)
                    roi_y_ext = (0, roi_shape[1] - 1)
                    roi_z_ext = (0, roi_shape[2] - 1)

                    try:
                        if slice_moved["x"]:
                            roi_vox_c = T_main_to_roi.dot(sag_plane_vox_center)
                            # Apply +1 offset for sagittal to match drawing coordinate transform
                            new_roi_x = max(
                                roi_x_ext[0],
                                min(int(round(roi_vox_c[0])) + 1, roi_x_ext[1]),
                            )
                            # Update both 3D and 2D actors
                            if actor_dict.get("sagittal_3d"):
                                actor_dict["sagittal_3d"].display_extent(
                                    new_roi_x,
                                    new_roi_x,
                                    roi_y_ext[0],
                                    roi_y_ext[1],
                                    roi_z_ext[0],
                                    roi_z_ext[1],
                                )
                            if actor_dict.get("sagittal_2d"):
                                actor_dict["sagittal_2d"].display_extent(
                                    new_roi_x,
                                    new_roi_x,
                                    roi_y_ext[0],
                                    roi_y_ext[1],
                                    roi_z_ext[0],
                                    roi_z_ext[1],
                                )

                        if slice_moved["y"]:
                            roi_vox_c = T_main_to_roi.dot(cor_plane_vox_center)
                            new_roi_y = max(
                                roi_y_ext[0],
                                min(int(round(roi_vox_c[1])), roi_y_ext[1]),
                            )
                            if actor_dict.get("coronal_3d"):
                                actor_dict["coronal_3d"].display_extent(
                                    roi_x_ext[0],
                                    roi_x_ext[1],
                                    new_roi_y,
                                    new_roi_y,
                                    roi_z_ext[0],
                                    roi_z_ext[1],
                                )
                            if actor_dict.get("coronal_2d"):
                                actor_dict["coronal_2d"].display_extent(
                                    roi_x_ext[0],
                                    roi_x_ext[1],
                                    new_roi_y,
                                    new_roi_y,
                                    roi_z_ext[0],
                                    roi_z_ext[1],
                                )

                        if slice_moved["z"]:
                            roi_vox_c = T_main_to_roi.dot(ax_plane_vox_center)
                            new_roi_z = max(
                                roi_z_ext[0],
                                min(int(round(roi_vox_c[2])), roi_z_ext[1]),
                            )
                            if actor_dict.get("axial_3d"):
                                actor_dict["axial_3d"].display_extent(
                                    roi_x_ext[0],
                                    roi_x_ext[1],
                                    roi_y_ext[0],
                                    roi_y_ext[1],
                                    new_roi_z,
                                    new_roi_z,
                                )
                            if actor_dict.get("axial_2d"):
                                actor_dict["axial_2d"].display_extent(
                                    roi_x_ext[0],
                                    roi_x_ext[1],
                                    roi_y_ext[0],
                                    roi_y_ext[1],
                                    new_roi_z,
                                    new_roi_z,
                                )

                    except Exception as e:
                        logger.error(f"Error updating ROI layer {key} slice: {e}")

            self._create_or_update_crosshairs()

            # Update 2D cameras for moved slices
            if slice_moved["x"]:
                self._update_sagittal_camera()
            if slice_moved["y"]:
                self._update_coronal_camera()
            if slice_moved["z"]:
                self._update_axial_camera()

            # Update status and render
            self._render_all()

        except Exception as e:
            error_msg = f"Error in set_slice_indices: {e}"
            logger.error(error_msg, exc_info=True)

    def _update_slow_slice_components(self) -> None:
        """
        Updates the "slow" components of the slice view (status text, RAS coords)
        after a drag-navigate is finished.
        """
        if (
            self.main_window is None
            or not all(self.current_slice_indices.values())
            or not self.image_extents["x"]
        ):

            # Still try to clear the coordinate display if no image
            if self.main_window:
                self.main_window.update_ras_coordinate_display(None)
            return

        c = self.current_slice_indices

        # Update status text
        status_msg = f"Slice View: X={c['x']}/{self.image_extents['x'][1]}  Y={c['y']}/{self.image_extents['y'][1]}  Z={c['z']}/{self.image_extents['z'][1]}"
        self.update_status(status_msg)

        # Update RAS Coordinate Display
        try:
            main_affine = self.main_window.anatomical_image_affine
            if main_affine is not None:
                # Get world coordinates
                world_coord = self._voxel_to_world(
                    [c["x"], c["y"], c["z"]], main_affine
                )

                # Update the main window's new display
                self.main_window.update_ras_coordinate_display(world_coord)

            else:
                self.main_window.update_ras_coordinate_display(None)
        except Exception as e:
            logger.error(f"Error updating RAS coordinate display: {e}")
            self.main_window.update_ras_coordinate_display(None)

    def set_slices_from_ras(self, ras_coords: np.ndarray) -> None:
        """
        Moves the slice crosshairs to the specified RAS coordinate.
        Converts RAS -> Voxel, then calls set_slice_indices.

        Args:
            ras_coords: A (3,) numpy array of [X, Y, Z] world coordinates.
        """
        # Check if we have the necessary info
        if (
            self.main_window is None
            or self.main_window.anatomical_image_affine is None
            or not all(self.image_extents.values())
            or self.image_extents["x"] is None
        ):

            self.update_status(
                "Error: Cannot set slices from RAS, no anatomical image loaded."
            )
            return

        try:
            # Convert RAS world coordinate to (float) voxel coordinate
            inv_affine = np.linalg.inv(self.main_window.anatomical_image_affine)
            voxel_pos_float = self._world_to_voxel(ras_coords, inv_affine)

            # Round to nearest integer voxel index
            new_x = int(round(voxel_pos_float[0]))
            new_y = int(round(voxel_pos_float[1]))
            new_z = int(round(voxel_pos_float[2]))

            # Clamp to valid voxel range
            new_x = max(
                self.image_extents["x"][0], min(new_x, self.image_extents["x"][1])
            )
            new_y = max(
                self.image_extents["y"][0], min(new_y, self.image_extents["y"][1])
            )
            new_z = max(
                self.image_extents["z"][0], min(new_z, self.image_extents["z"][1])
            )

            # Call the master update function
            self.set_slice_indices(x=new_x, y=new_y, z=new_z)

            # Update the status bar and text input
            self._update_slow_slice_components()

        except Exception as e:
            self.update_status(f"Error setting slice from RAS: {e}")
            logger.error(f"Error setting slice from RAS:", exc_info=True)

    def _navigate_2d_view(self, obj: vtk.vtkObject, event_id: str) -> None:
        """
        Picks the 3D coordinate from a 2D view and updates the slice indices.
        """

        # Check if anatomical image is loaded
        if (
            self.main_window is None
            or self.main_window.anatomical_image_data is None
            or None in self.current_slice_indices.values()
        ):
            self.is_navigating_2d = False  # avoid freeze
            return

        # Get interactor and click position (display coords)
        interactor = obj
        if not interactor:
            return
        display_pos = interactor.GetEventPosition()

        # Pick the 3D world coordinate at that pixel
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
        if not renderer:
            return

        picker.Pick(display_pos[0], display_pos[1], 0, renderer)

        # Check if the picker missed
        if picker.GetCellId() < 0:
            # self.is_navigating_2d = False
            return

        # Get the 3D position of the pick
        world_pos = picker.GetPickPosition()

        try:
            # Convert this world coordinate back to a (float) voxel coordinate
            inv_affine = np.linalg.inv(self.main_window.anatomical_image_affine)
            voxel_pos_float = self._world_to_voxel(world_pos, inv_affine)

            # Round to nearest integer voxel index
            new_x = int(round(voxel_pos_float[0]))
            new_y = int(round(voxel_pos_float[1]))
            new_z = int(round(voxel_pos_float[2]))

            # Clamp to valid voxel range
            new_x = max(
                self.image_extents["x"][0], min(new_x, self.image_extents["x"][1])
            )
            new_y = max(
                self.image_extents["y"][0], min(new_y, self.image_extents["y"][1])
            )
            new_z = max(
                self.image_extents["z"][0], min(new_z, self.image_extents["z"][1])
            )

            # Abort if the pick produced an out-of-range index
            if (
                new_x < self.image_extents["x"][0]
                or new_x > self.image_extents["x"][1]
                or new_y < self.image_extents["y"][0]
                or new_y > self.image_extents["y"][1]
                or new_z < self.image_extents["z"][0]
                or new_z > self.image_extents["z"][1]
            ):
                self.is_navigating_2d = False
                return

            # Call the master update function
            if interactor == self.axial_interactor:
                # Axial view (X-Y plane), so keep current Z
                current_z = self.current_slice_indices["z"]
                self.set_slice_indices(x=new_x, y=new_y, z=current_z)

            elif interactor == self.coronal_interactor:
                # Coronal view (X-Z plane), so keep current Y
                current_y = self.current_slice_indices["y"]
                self.set_slice_indices(x=new_x, y=current_y, z=new_z)

            elif interactor == self.sagittal_interactor:
                # Sagittal view (Y-Z plane), so keep current X
                current_x = self.current_slice_indices["x"]
                self.set_slice_indices(x=current_x, y=new_y, z=new_z)

            else:
                self.set_slice_indices(x=new_x, y=new_y, z=new_z)

        except Exception as e:
            logger.error(f"Error during 2D window click navigation:", exc_info=True)
            self.update_status("Error navigating with click.")

    # Streamline Actor Management
    def update_highlight(self) -> None:
        """Updates the actor for highlighted/selected streamlines. Delegates to StreamlinesManager."""
        self.streamlines_manager.update_highlight()

    def _calculate_scalar_colors(
        self,
        streamlines_gen,
        scalar_gen,
        vmin: float,
        vmax: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates vertex colors based on scalar arrays.
        Delegates to StreamlinesManager.
        """
        return self.streamlines_manager.calculate_scalar_colors(
            streamlines_gen, scalar_gen, vmin, vmax
        )

    def _update_scalar_bar(self, lut: vtk.vtkLookupTable, title: str) -> None:
        """Creates or updates the scalar bar actor. Delegates to StreamlinesManager."""
        self.streamlines_manager.update_scalar_bar(lut, title)

    def _get_streamline_actor_params(self) -> Dict[str, Any]:
        """
        Determines parameters for the main streamlines actor using STRIDE.
        Delegates to StreamlinesManager.
        """
        return self.streamlines_manager.get_streamline_actor_params()

    def update_main_streamlines_actor(self) -> None:
        """
        Recreates the main actor using the skipped/strided dataset.
        Delegates to StreamlinesManager.
        """
        self.streamlines_manager.update_main_streamlines_actor()

    # Interaction Callbacks
    def _handle_save_shortcut(self) -> None:
        """Handles the Ctrl+S save shortcut logic (saves streamlines)."""
        if not self.main_window:
            return
        if not self.main_window.tractogram_data:
            self.update_status("Save shortcut (Ctrl+S): No streamlines loaded to save.")
            return
        self.update_radius_actor(visible=False)
        try:
            self.main_window._trigger_save_streamlines()
        except AttributeError:
            self.update_status("Error: Save function not found.")
        except Exception as e:
            self.update_status(f"Error during save: {e}")

    def _handle_radius_change(self, increase: bool = True) -> None:
        """Handles increasing or decreasing the selection radius via main window."""
        if not self.main_window:
            return
        if not self.main_window.tractogram_data:
            self.update_status("Radius change (+/-): No streamlines loaded.")
            return
        if increase:
            self.main_window._increase_radius()
        else:
            self.main_window._decrease_radius()

    def _find_streamlines_in_radius(
        self, center_point: np.ndarray, radius: float, check_all: bool = False
    ) -> Set[int]:
        """
        Finds streamlines within a sphere. Delegates to SelectionManager.
        """
        return self.selection_manager.find_streamlines_in_radius(
            center_point, radius, check_all
        )

    def _find_streamlines_in_box(
        self, min_point: np.ndarray, max_point: np.ndarray, check_all: bool = False
    ) -> Set[int]:
        """
        Finds streamlines within a box. Delegates to SelectionManager.
        """
        return self.selection_manager.find_streamlines_in_box(
            min_point, max_point, check_all
        )

    def _toggle_selection(self, indices_to_toggle: Set[int]) -> None:
        """Toggles the selection state for given indices. Delegates to SelectionManager."""
        self.selection_manager.toggle_selection(indices_to_toggle)

    def _handle_streamline_selection(self) -> None:
        """Handles the logic for selecting streamlines. Delegates to SelectionManager."""
        self.selection_manager.handle_streamline_selection()

    def _handle_inverse_selection(self) -> None:
        """Handles the logic for invert selection. Delegates to SelectionManager."""
        self.selection_manager.invert_selection()

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

        # Block VTK's default number key behaviors (e.g., '3' toggles stereo mode)
        vtk_blocked_keys = {"1", "2", "3", "4"}
        if key in vtk_blocked_keys and not ctrl and not shift:
            interactor.SetKeyCode(chr(0))  # Neutralize the key to prevent VTK handling
            return

        # Handle non-data-dependent keys first
        non_data_handlers = {
            ("z", True, False): self.main_window._perform_undo,
            ("y", True, False): self.main_window._perform_redo,
            ("z", True, True): self.main_window._perform_redo,  # Ctrl+Shift+Z
            ("escape", False, False): self.main_window._hide_sphere,
            ("s", True, False): self._handle_save_shortcut,
        }
        if handler_key in non_data_handlers:
            non_data_handlers[handler_key]()
            return

        # Handle Slice Navigation Keys
        anatomical_loaded = self.main_window.anatomical_image_data is not None
        if anatomical_loaded:
            slice_nav_handlers = {
                # (key, ctrl, shift) : (axis, direction)
                ("up", False, False): ("z", 1),  # Axial Up
                ("down", False, False): ("z", -1),  # Axial Down
                ("right", False, False): ("x", 1),  # Sagittal Right
                ("left", False, False): ("x", -1),  # Sagittal Left
                ("up", True, False): ("y", 1),  # Coronal Up
                ("down", True, False): ("y", -1),  # Coronal Down
            }
            if handler_key in slice_nav_handlers:
                axis, direction = slice_nav_handlers[handler_key]
                self.move_slice(axis, direction)
                return

        # Handle Streamline-dependent keys
        streamline_data_handlers = {
            "s": self._handle_streamline_selection,
            "i": self._handle_inverse_selection,
            "plus": lambda: self._handle_radius_change(increase=True),
            "equal": lambda: self._handle_radius_change(increase=True),
            "minus": lambda: self._handle_radius_change(increase=False),
            "d": self.main_window._perform_delete_selection,
            "c": self.main_window._perform_clear_selection,
        }
        streamline_keys_for_status = {"s", "i", "plus", "equal", "minus", "d", "c"}

        streamlines_loaded = bool(self.main_window.tractogram_data)
        if not streamlines_loaded:
            if key in streamline_keys_for_status:
                self.update_status(
                    f"Action ('{key}') requires streamlines. Load a trk/tck file first."
                )
            return

        if key in streamline_data_handlers and not ctrl and not shift:
            streamline_data_handlers[key]()

    # ROI Layer Actor Management
    def add_roi_layer(
        self, key: str, data: np.ndarray, affine: np.ndarray, render: bool = True
    ) -> None:
        """Creates and adds slicer actors for a new ROI layer (separating 2D/3D actors)."""
        if (
            not self.scene
            or not self.main_window
            or self.main_window.anatomical_image_affine is None
        ):
            logger.error("Cannot add ROI layer: Main scene or image not initialized.")
            return

        data_min = np.min(data)
        data_max = np.max(data)

        # If data is uniform (e.g., all zeros), set range to (0, 1).
        # This ensures 0 maps to the start of the LUT (Transparent) and 1 maps to the end (Color).
        if data_max <= data_min:
            value_range = (0, 1)
        else:
            value_range = (data_min, data_max)

        # Get opacity from MainWindow state
        slicer_opacity = 0.5  # Default
        if (
            hasattr(self.main_window, "roi_opacities")
            and key in self.main_window.roi_opacities
        ):
            slicer_opacity = self.main_window.roi_opacities[key]

        interpolation_mode = "nearest"

        # Get current slice indices
        c_idx = self.current_slice_indices
        if c_idx["x"] is None:
            self.update_status("Error: Slices not initialized.")
            return

        # Calculate initial ROI slice indices
        main_img_affine = self.main_window.anatomical_image_affine
        roi_info = self.main_window.roi_layers.get(key)
        if not roi_info or "inv_affine" not in roi_info:
            return

        inv_roi_affine = roi_info["inv_affine"]
        c_x, c_y, c_z = c_idx["x"], c_idx["y"], c_idx["z"]
        world_c = self._voxel_to_world([c_x, c_y, c_z], main_img_affine)
        roi_vox_c = self._world_to_voxel(world_c, inv_roi_affine)
        roi_x, roi_y, roi_z = (
            int(round(roi_vox_c[0])),
            int(round(roi_vox_c[1])),
            int(round(roi_vox_c[2])),
        )

        # Get ROI extents
        roi_shape = data.shape
        roi_x_ext = (0, roi_shape[0] - 1)
        roi_y_ext = (0, roi_shape[1] - 1)
        roi_z_ext = (0, roi_shape[2] - 1)

        try:
            slicer_params = {
                "affine": affine,
                "value_range": value_range,
                "opacity": slicer_opacity,
                "interpolation": interpolation_mode,
            }

            # Create TWO sets of actors
            # Set 1: For the main 3D Scene
            ax_3d = actor.slicer(data, **slicer_params)
            cor_3d = actor.slicer(data, **slicer_params)
            sag_3d = actor.slicer(data, **slicer_params)

            # Set 2: For the 2D Ortho Views
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
                ax.display_extent(
                    roi_x_ext[0], roi_x_ext[1], roi_y_ext[0], roi_y_ext[1], roi_z, roi_z
                )
            for cor in [cor_3d, cor_2d]:
                cor.display_extent(
                    roi_x_ext[0], roi_x_ext[1], roi_y, roi_y, roi_z_ext[0], roi_z_ext[1]
                )
            # Apply +1 offset for sagittal to match slice navigation behavior
            roi_x_adjusted = roi_x + 1
            for sag in [sag_3d, sag_2d]:
                sag.display_extent(
                    roi_x_adjusted,
                    roi_x_adjusted,
                    roi_y_ext[0],
                    roi_y_ext[1],
                    roi_z_ext[0],
                    roi_z_ext[1],
                )

            # Store in expanded dictionary
            self.roi_slice_actors[key] = {
                "axial_3d": ax_3d,
                "coronal_3d": cor_3d,
                "sagittal_3d": sag_3d,
                "axial_2d": ax_2d,
                "coronal_2d": cor_2d,
                "sagittal_2d": sag_2d,
            }

            # Add to respective scenes
            self.axial_scene.add(ax_2d)
            self.coronal_scene.add(cor_2d)
            self.sagittal_scene.add(sag_2d)

            # X-flip for axial/coronal ROI actors to match anatomical actors
            # Anatomical actors have SetScale(-1, 1, 1) for radiological convention
            ax_2d.SetScale(-1, 1, 1)
            cor_2d.SetScale(-1, 1, 1)

            # Get assigned color from MainWindow (or default to Red)
            assigned_color = (1.0, 0.0, 0.0)
            if self.main_window and key in self.main_window.roi_layers:
                assigned_color = self.main_window.roi_layers[key].get(
                    "color", (1.0, 0.0, 0.0)
                )

            # 3D Sphere Visualization
            # If this ROI is a sphere, create a 3D sphere actor for the main window
            if (
                hasattr(self, "sphere_params_per_roi")
                and key in self.sphere_params_per_roi
            ):
                params = self.sphere_params_per_roi[key]
                center = params["center"]
                radius = params["radius"]

                # Create sphere actor
                sphere_actor = actor.sphere(
                    centers=np.array([center]),
                    colors=np.array([assigned_color]),  # Use assigned color
                    radii=radius,
                    opacity=0.5,  # Semi-transparent to see streamlines inside
                )

                self.scene.add(sphere_actor)
                sphere_actor.SetVisibility(vis_flag)

                # Add to actor dict so it gets managed (removed/hidden) automatically
                self.roi_slice_actors[key]["sphere_3d"] = sphere_actor

            # Rectangle Visualization
            # If this ROI is a rectangle, create a 2D rectangle actor for the main window
            if (
                hasattr(self, "rectangle_params_per_roi")
                and key in self.rectangle_params_per_roi
            ):
                params = self.rectangle_params_per_roi[key]
                start = np.array(params["start"])
                end = np.array(params["end"])
                view_type = params.get("view_type", "axial")

                # Use the helper method which creates the visualization
                self._update_3d_rectangle_visuals(
                    key, start, end, view_type, render=render
                )

                # Set visibility
                rect_actor = self.roi_slice_actors[key].get("rectangle_3d")
                if rect_actor:
                    rect_actor.SetVisibility(vis_flag)

            # For loaded ROIs that are not spheres or rectangles, create a 3D contour actor
            is_sphere = (
                hasattr(self, "sphere_params_per_roi")
                and key in self.sphere_params_per_roi
            )
            is_rectangle = (
                hasattr(self, "rectangle_params_per_roi")
                and key in self.rectangle_params_per_roi
            )

            if not is_sphere and not is_rectangle:
                # Create a 3D contour actor for this ROI
                try:
                    contour_actor = actor.contour_from_roi(
                        data,
                        affine=affine,
                        color=assigned_color,
                        opacity=0.5,
                    )
                    contour_actor.SetVisibility(vis_flag)
                    self.scene.add(contour_actor)
                    self.roi_slice_actors[key]["contour_3d"] = contour_actor
                except Exception as e:
                    logger.warning(f"Could not create 3D contour for ROI {key}: {e}")

            # Apply transformation ONLY to 2D actors
            self._apply_display_correction(key)

            # Set color
            self.set_roi_layer_color(key, assigned_color)

            # Explicitly force sagittal actor to update (fixes display issue)
            if sag_2d:
                mapper = sag_2d.GetMapper()
                if mapper:
                    mapper.Modified()
                sag_2d.Modified()

            if render:
                self._render_all()

        except Exception as e:
            logger.error(f"Error creating ROI slicer actors:", exc_info=True)
            QMessageBox.critical(
                self.main_window,
                "ROI Actor Error",
                f"Could not create slicer actors for ROI:\n{e}",
            )
            if key in self.main_window.roi_layers:
                del self.main_window.roi_layers[key]
            if key in self.roi_slice_actors:
                del self.roi_slice_actors[key]

    def _apply_display_correction(self, roi_key: str) -> None:
        """Applies display correction only to 2D ROI slices."""
        if roi_key not in self.roi_slice_actors:
            return

        roi_actors = self.roi_slice_actors[roi_key]

        # Only correct the 2D actors
        targets = ["axial_2d", "coronal_2d"]

        for actor_type in targets:
            actor_obj = roi_actors.get(actor_type)
            if actor_obj:
                current_pos = actor_obj.GetPosition()
                # X-flip: Convert FURY's neurological output to radiological display convention
                actor_obj.SetScale(-1, 1, 1)

                # Adjustment for position
                bounds = actor_obj.GetBounds()
                if bounds[0] != 1.0 and bounds[1] != 1.0:
                    actor_obj.SetPosition(
                        -current_pos[0], current_pos[1], current_pos[2]
                    )

    def _update_3d_sphere_visuals(
        self, roi_name: str, center: np.ndarray, radius: float
    ) -> None:
        """Updates the 3D sphere actor for an ROI. Delegates to DrawingManager."""
        self.drawing_manager.update_3d_sphere_visuals(roi_name, center, radius)

    def _update_3d_rectangle_visuals(
        self,
        roi_name: str,
        start: np.ndarray,
        end: np.ndarray,
        view_type: str,
        render: bool = True,
    ) -> None:
        """Updates the 3D rectangle actor for an ROI. Delegates to DrawingManager."""
        self.drawing_manager.update_3d_rectangle_visuals(
            roi_name, start, end, view_type, render
        )

    def set_roi_layer_visibility(self, key: str, visible: bool) -> None:
        """Sets visibility for all 2D and 3D actors of an ROI."""
        if key not in self.roi_slice_actors:
            return

        vis_flag = 1 if visible else 0
        changed = False

        # Iterate over all actors stored for this key
        for act in self.roi_slice_actors[key].values():
            if act and act.GetVisibility() != vis_flag:
                act.SetVisibility(vis_flag)
                changed = True

        if changed:
            self._render_all()

    def remove_roi_layer(self, key: str) -> None:
        """Removes a specific ROI layer's actors from all scenes."""
        if key not in self.roi_slice_actors:
            return

        scenes_to_check = [
            self.scene,
            self.axial_scene,
            self.coronal_scene,
            self.sagittal_scene,
        ]

        for act in self.roi_slice_actors[key].values():
            if act is not None:
                for scn in scenes_to_check:
                    try:
                        scn.rm(act)
                    except Exception:
                        pass

        del self.roi_slice_actors[key]
        self._render_all()

    def rename_roi_layer(self, old_key: str, new_key: str) -> None:
        """Renames an ROI layer key."""
        if old_key in self.roi_slice_actors:
            self.roi_slice_actors[new_key] = self.roi_slice_actors.pop(old_key)

    def clear_all_roi_layers(self) -> None:
        """Removes all ROI slice actors from all scenes."""
        if not self.roi_slice_actors:
            return

        scenes_to_check = [
            self.scene,
            self.axial_scene,
            self.coronal_scene,
            self.sagittal_scene,
        ]

        for actor_dict in self.roi_slice_actors.values():
            for act in actor_dict.values():
                if act:
                    for scn in scenes_to_check:
                        try:
                            scn.rm(act)
                        except Exception:
                            pass

        self.roi_slice_actors.clear()
        self._render_all()

    def set_roi_layer_color(self, key: str, color: Tuple[float, float, float]) -> None:
        """Sets the color (tint) for a specific ROI layer."""
        if key not in self.roi_slice_actors:
            return

        r, g, b = color
        for act_key, act in self.roi_slice_actors[key].items():
            if not act:
                continue

            # Skip non-actor entries (e.g., 'sphere_source', 'rectangle_points')
            if act_key in ("sphere_source", "rectangle_points"):
                continue

            # Handle Sphere Actor (SetColor directly)
            if act_key == "sphere_3d":
                act.GetProperty().SetColor(r, g, b)
                continue

            # Handle Rectangle Actor (SetColor directly with edge color)
            if act_key == "rectangle_3d":
                prop = act.GetProperty()
                prop.SetColor(r, g, b)
                prop.SetEdgeColor(r, g, b)
                continue

            # Handle Contour Actor (SetColor directly)
            if act_key == "contour_3d":
                act.GetProperty().SetColor(r, g, b)
                continue

            # Handle Slicer Actors (Use LUT)
            prop = act.GetProperty()
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(256)
            w = prop.GetColorWindow()
            l = prop.GetColorLevel()
            lut.SetTableRange(l - w / 2, l + w / 2)
            lut.Build()
            for i in range(256):
                alpha = 0.0 if i == 0 else 1.0
                lut.SetTableValue(i, r * i / 255, g * i / 255, b * i / 255, alpha)
            prop.SetLookupTable(lut)

        self._render_all()

    def take_screenshot(self) -> None:
        """Saves a screenshot of the VTK view with an opaque black background, hiding UI overlays."""
        if not self.render_window or not self.scene:
            QMessageBox.warning(
                self.main_window,
                "Screenshot Error",
                "Render window or scene not available.",
            )
            return
        if not (
            self.main_window.tractogram_data or self.main_window.anatomical_image_data
        ):
            QMessageBox.warning(
                self.main_window,
                "Screenshot Error",
                "No data loaded to take screenshot of.",
            )
            return

        default_filename = "tractedit_screenshot.png"
        base_name = "tractedit_view"
        if self.main_window.original_trk_path:
            base_name = os.path.splitext(
                os.path.basename(self.main_window.original_trk_path)
            )[0]
        elif self.main_window.anatomical_image_path:
            base_name = os.path.splitext(
                os.path.basename(self.main_window.anatomical_image_path)
            )[0]

        slice_info = ""
        if (
            self.axial_slice_actor
            and self.axial_slice_actor.GetVisibility()
            and self.current_slice_indices["z"] is not None
        ):
            slice_info = f"_x{self.current_slice_indices['x']}_y{self.current_slice_indices['y']}_z{self.current_slice_indices['z']}"

        default_filename = f"{base_name}{slice_info}_screenshot.png"

        output_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save Screenshot",
            default_filename,
            "PNG Image Files (*.png);;JPEG Image Files (*.jpg *.jpeg);;TIFF Image Files (*.tif *.tiff);;All Files (*.*)",
        )
        if not output_path:
            self.update_status("Screenshot cancelled.")
            return

        self.update_status(f"Preparing screenshot: {os.path.basename(output_path)}...")
        QApplication.processEvents()

        original_background = self.scene.GetBackground()
        original_status_visibility = (
            self.status_text_actor.GetVisibility() if self.status_text_actor else 0
        )
        original_instruction_visibility = (
            self.instruction_text_actor.GetVisibility()
            if self.instruction_text_actor
            else 0
        )
        original_axes_visibility = (
            self.axes_actor.GetVisibility() if self.axes_actor else 0
        )
        original_radius_actor_visibility = (
            self.radius_actor.GetVisibility() if self.radius_actor else 0
        )
        original_slice_vis = {}
        slice_actors = {
            "axial": self.axial_slice_actor,
            "coronal": self.coronal_slice_actor,
            "sagittal": self.sagittal_slice_actor,
        }
        for name, act in slice_actors.items():
            original_slice_vis[name] = act.GetVisibility() if act else 0

        try:
            if self.status_text_actor:
                self.status_text_actor.SetVisibility(0)
            if self.instruction_text_actor:
                self.instruction_text_actor.SetVisibility(0)
            if self.axes_actor:
                self.axes_actor.SetVisibility(0)
            if self.radius_actor:
                self.radius_actor.SetVisibility(0)

            self.scene.background((0.0, 0.0, 0.0))
            self.render_window.Render()

            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(self.render_window)
            window_to_image_filter.SetInputBufferTypeToRGB()
            window_to_image_filter.ReadFrontBufferOff()
            window_to_image_filter.ShouldRerenderOff()
            window_to_image_filter.Update()

            _, output_ext = os.path.splitext(output_path)
            output_ext = output_ext.lower()
            writer: Optional[vtk.vtkImageWriter] = None
            if output_ext == ".png":
                writer = vtk.vtkPNGWriter()
            elif output_ext in [".jpg", ".jpeg"]:
                writer = vtk.vtkJPEGWriter()
                writer.SetQuality(95)
                writer.ProgressiveOn()
            elif output_ext in [".tif", ".tiff"]:
                writer = vtk.vtkTIFFWriter()
                writer.SetCompressionToPackBits()
            else:
                output_path = os.path.splitext(output_path)[0] + ".png"
                writer = vtk.vtkPNGWriter()
                QMessageBox.warning(
                    self.main_window,
                    "Screenshot Info",
                    f"Unsupported extension '{output_ext}'. Saving as PNG.",
                )

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
            if original_background:
                self.scene.background(original_background)
            if self.status_text_actor:
                self.status_text_actor.SetVisibility(original_status_visibility)
            if self.instruction_text_actor:
                self.instruction_text_actor.SetVisibility(
                    original_instruction_visibility
                )
            if self.axes_actor:
                self.axes_actor.SetVisibility(original_axes_visibility)
            if self.radius_actor:
                self.radius_actor.SetVisibility(original_radius_actor_visibility)

            if (
                self.render_window
                and self.render_window.GetInteractor().GetInitialized()
            ):
                self.render_window.Render()
