# -*- coding: utf-8 -*-

"""
Main VTK panel module for TractEdit visualization.

Manages the VTK rendering window, scene, actors (streamlines, image slices),
and interactions. Uses helper modules for interactor styles, actor creation,
scene management, mesh/ROI/slice sub-managers, and coordinate transformations.
"""

# ============================================================================
# Imports
# ============================================================================

import os
import numpy as np
import vtk
import logging
from contextlib import ExitStack
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
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtGui import QAction
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from fury import window, actor
from ..render_batch import request_render
from ..data_contracts import (
    MeshComponent,
    MeshComponentStyle,
    MeshLayer,
    MeshTexturePreparation,
)
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple, Set
from functools import partial
from ..ui.theme_manager import ThemeManager
from .interactor import CustomInteractorStyle2D
from .scene_manager import SceneManager
from .drawing import DrawingManager
from .selection import SelectionManager
from .streamlines import StreamlinesManager
from .scale_bar import ScaleBarManager
from .mesh_scene import MeshSceneManager
from .roi_actors import (
    ROISceneManager,
    _ROI_SLICE_ACTOR_KEYS,
    _crop_roi_for_contour,
)
from .slice_manager import ORTHO_VIEW_SPECS, SliceManager
from .actors import (
    apply_slicer_affine,
    create_slicer_actor,
    create_mesh_actor,
    create_mesh_component_actor,
    create_mesh_advanced_actor,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..mesh_texture import MeshTextureBudgets
    from .mesh_texture import MeshTexturePresentation


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

        for spec in ORTHO_VIEW_SPECS:
            widget = QVTKRenderWindowInteractor()
            widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            widget.customContextMenuRequested.connect(
                partial(
                    self._show_2d_context_menu,
                    widget=widget,
                    view_type=spec.name,
                )
            )
            ortho_splitter.addWidget(widget)
            setattr(self, f"{spec.name}_vtk_widget", widget)

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
        self.render_window.SetAlphaBitPlanes(1)

        # Reduce anti-aliasing for better performance (4 samples vs default 8)
        self.render_window.SetMultiSamples(4)

        self.interactor: vtk.vtkRenderWindowInteractor = (
            self.render_window.GetInteractor()
        )
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Orthogonal 2D Scenes (axial, coronal, sagittal)
        for spec in ORTHO_VIEW_SPECS:
            widget = getattr(self, f"{spec.name}_vtk_widget")
            scene_2d: window.Scene = window.Scene()
            scene_2d.background((0.0, 0.0, 0.0))
            rw_2d: vtk.vtkRenderWindow = widget.GetRenderWindow()
            rw_2d.SetNumberOfLayers(2)
            rw_2d.AddRenderer(scene_2d)
            scene_2d.SetLayer(0)

            overlay_renderer = vtk.vtkRenderer()
            overlay_renderer.SetLayer(1)
            overlay_renderer.InteractiveOff()
            overlay_renderer.PreserveColorBufferOn()
            overlay_renderer.SetActiveCamera(scene_2d.GetActiveCamera())
            rw_2d.AddRenderer(overlay_renderer)

            interactor_2d: vtk.vtkRenderWindowInteractor = rw_2d.GetInteractor()
            interactor_2d.SetInteractorStyle(CustomInteractorStyle2D(self))

            setattr(self, f"{spec.name}_scene", scene_2d)
            setattr(self, f"{spec.name}_render_window", rw_2d)
            setattr(self, f"{spec.name}_overlay_renderer", overlay_renderer)
            setattr(self, f"{spec.name}_interactor", interactor_2d)

        # --- Actor Placeholders ---
        self.mesh_actors: Dict[str, vtk.vtkActor] = {}
        self._mesh_component_actors: Dict[str, Dict[str, vtk.vtkActor]] = {}
        self._mesh_overlay_data = {}
        self._mesh_camera_observers = ()
        self._mesh_texture_presentations: Dict[str, "MeshTexturePresentation"] = {}
        self._mesh_render_settings: Optional[Dict[str, Any]] = None
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
        self.axes_widget: Optional[vtk.vtkOrientationMarkerWidget] = None
        self.orientation_widget: Optional[vtk.vtkOrientationMarkerWidget] = None
        self.orientation_labels: List[vtk.vtkTextActor] = []

        # Crosshair Actors
        self.axial_crosshair_actor: Optional[vtk.vtkActor] = None
        self.coronal_crosshair_actor: Optional[vtk.vtkActor] = None
        self.sagittal_crosshair_actor: Optional[vtk.vtkActor] = None
        self.crosshair_lines: Dict[str, vtk.vtkLineSource] = {}
        self.crosshair_appenders: Dict[str, vtk.vtkAppendPolyData] = {}

        self.is_navigating_2d: bool = False

        # Drawing Mode State
        self.is_drawing_mode: bool = False
        self.is_drawing_active: bool = False
        self.brush_size: int = 1
        self.draw_stroke_count: int = 0
        self.update_interval: int = 15

        # Preview-based drawing: collect points, show preview, convert on release
        self.drawing_preview_points: list = []
        self.preview_line_actor: Optional[vtk.vtkActor] = None

        self.sphere_params_per_roi: Dict[str, Dict[str, Any]] = {}
        self.rectangle_params_per_roi: Dict[str, Dict[str, Any]] = {}
        self.roi_slice_actors: Dict[str, Dict[str, vtk.vtkActor]] = {}

        self.roi_highlight_actor: Optional[vtk.vtkActor] = None
        # Cyan contour rendered on the keeper set during inversion mode.
        # Color: #05B8CC -> VTK (0.020, 0.722, 0.800)
        self.invert_contour_actor: Optional[vtk.vtkActor] = None
        self.scalar_bar_actor: Optional[vtk.vtkScalarBarActor] = None

        # Sub-managers
        self.scene_manager: SceneManager = SceneManager(self)
        self.drawing_manager: DrawingManager = DrawingManager(self)
        self.selection_manager: SelectionManager = SelectionManager(self)
        self.streamlines_manager: StreamlinesManager = StreamlinesManager(self)
        self.scale_bar_manager: ScaleBarManager = ScaleBarManager()
        self.mesh_scene_manager: MeshSceneManager = MeshSceneManager(self)
        self.roi_scene_manager: ROISceneManager = ROISceneManager(self)
        self.slice_manager: SliceManager = SliceManager(self)

        # Fast Render Mode State
        # VTK DesiredUpdateRate: HIGHER = faster rendering (lower quality)
        #                        LOWER = slower rendering (higher quality)
        self._is_interacting: bool = False
        self._interaction_streamline_property = None
        self._original_desired_update_rate: float = 0.0001
        self._fast_update_rate: float = 30.0

        # Overlay Progress Bar
        self.overlay_progress_bar = QProgressBar(self.vtk_widget)
        self.overlay_progress_bar.setStyleSheet(ThemeManager.bar_style(True))
        self.overlay_progress_bar.hide()

        # Slice Navigation State
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

        self.scene_manager.create_scene_ui()

        # Setup Interaction Callbacks
        self.interactor.AddObserver(
            vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0
        )
        for spec in ORTHO_VIEW_SPECS:
            getattr(self, f"{spec.name}_interactor").AddObserver(
                vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0
            )

        # Fast Render Mode Observers (3D view only)
        self.interactor.AddObserver(
            vtk.vtkCommand.StartInteractionEvent,
            self._on_interaction_start,
        )
        self.interactor.AddObserver(
            vtk.vtkCommand.EndInteractionEvent,
            self._on_interaction_end,
        )

        # Initialize VTK Widgets
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        for spec in ORTHO_VIEW_SPECS:
            widget = getattr(self, f"{spec.name}_vtk_widget")
            widget.Initialize()
            widget.Start()

        self.scene.reset_camera()
        self._render_all()

    @property
    def mesh_scene_manager(self) -> MeshSceneManager:
        manager = self.__dict__.get("_mesh_scene_manager")
        if manager is None:
            manager = MeshSceneManager(self)
            self._mesh_scene_manager = manager
        return manager

    @mesh_scene_manager.setter
    def mesh_scene_manager(self, value: MeshSceneManager) -> None:
        self._mesh_scene_manager = value

    @property
    def roi_scene_manager(self) -> ROISceneManager:
        manager = self.__dict__.get("_roi_scene_manager")
        if manager is None:
            manager = ROISceneManager(self)
            self._roi_scene_manager = manager
        return manager

    @roi_scene_manager.setter
    def roi_scene_manager(self, value: ROISceneManager) -> None:
        self._roi_scene_manager = value

    @property
    def slice_manager(self) -> SliceManager:
        manager = self.__dict__.get("_slice_manager")
        if manager is None:
            manager = SliceManager(self)
            self._slice_manager = manager
        return manager

    @slice_manager.setter
    def slice_manager(self, value: SliceManager) -> None:
        self._slice_manager = value

    def set_anatomical_slice_visibility(self, visible: bool) -> None:
        """Sets visibility for anatomical slices and associated crosshairs."""
        self.slice_manager.set_anatomical_slice_visibility(visible)

    def clear_invert_contour(self) -> None:
        """Remove the cyan keeper-contour actor from the scene.

        Safe to call when no contour actor exists.
        """
        if self.invert_contour_actor is not None:
            try:
                self.scene.rm(self.invert_contour_actor)
            except (ValueError, AttributeError):
                pass
            self.invert_contour_actor = None

    def clear_highlight(self) -> None:
        """Remove the yellow selection-highlight actor from the scene.

        Unlike ``update_highlight()``, this method does *not* rebuild the
        actor — it only removes it.  This is required in the large-inversion
        code path: the secondary guard inside ``update_highlight()`` returns
        early (to avoid RAM exhaustion) without removing any pre-existing
        yellow actor, so a stale highlight would otherwise remain visible.
        """
        if self.highlight_actor is not None:
            try:
                self.scene.rm(self.highlight_actor)
            except (ValueError, AttributeError):
                pass
            self.highlight_actor = None

    @staticmethod
    def _is_render_window_ready(render_window) -> bool:
        """Check whether a VTK render window is initialised and safe to render.

        Guards against ``GetInteractor()`` returning ``None``, which would
        cause an ``AttributeError`` on the chained ``.GetInitialized()`` call.
        """
        if render_window is None:
            return False
        interactor = render_window.GetInteractor()
        return interactor is not None and interactor.GetInitialized()

    def _render_all(self) -> None:
        """Renders all four VTK windows if they are initialized."""
        try:
            windows = (
                self.render_window,
                *(
                    getattr(self, f"{spec.name}_render_window")
                    for spec in ORTHO_VIEW_SPECS
                ),
            )
            for rw in windows:
                if self._is_render_window_ready(rw):
                    request_render(self, rw)
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Warning: Error during _render_all: {e}")

    # Fast Render Mode Methods
    def _on_interaction_start(self, obj: vtk.vtkObject, event: str) -> None:
        """Called when user starts rotating/panning/zooming the 3D view."""
        self._mesh_interaction_state = self.interactor.GetInteractorStyle().GetState()
        timer = getattr(self, "_mesh_interaction_timer", None)
        if timer is not None:
            timer.stop()
        self._enable_fast_render()

    def _on_interaction_end(self, obj: vtk.vtkObject, event: str) -> None:
        """Called when user stops interacting with the 3D view."""
        preview = self._set_mesh_interaction_preview(True)
        surface = any(
            mesh_actor.GetVisibility() and mesh_actor.GetProperty().GetOpacity() > 0.0
            for mesh_actor in self.mesh_actors.values()
        )
        state = getattr(self, "_mesh_interaction_state", None)
        if preview or (surface and state == vtk.VTKIS_DOLLY):
            timer = getattr(self, "_mesh_interaction_timer", None)
            if timer is None:
                timer = self._mesh_interaction_timer = QTimer(self.main_window)
                timer.setSingleShot(True)
                timer.timeout.connect(self._disable_fast_render)
            # Keep wheel notches in one interaction instead of rendering full
            # detail during the short pauses between them.
            timer.start(350)
        else:
            # StopState renders through the interactor after this callback returns.
            native_render = (
                state in (vtk.VTKIS_ROTATE, vtk.VTKIS_PAN, vtk.VTKIS_SPIN, vtk.VTKIS_DOLLY)
                and obj is self.interactor
                and obj.GetEnabled() and obj.GetEnableRender()
            )
            self._disable_fast_render(render=not native_render)

    def _set_mesh_interaction_preview(self, enabled: bool) -> bool:
        active = False
        for components in getattr(self, "_mesh_component_actors", {}).values():
            for overlay in components.values():
                preview = getattr(overlay, "_mesh_interaction_mapper", None)
                if preview is not None:
                    use_preview = enabled and bool(overlay.GetVisibility())
                    overlay.SetMapper(preview if use_preview else overlay._mesh_full_mapper)
                    active |= use_preview
        return active

    def _sync_mesh_interaction_budget(self) -> None:
        """Prepare bounded display levels only when the overlay composition changes."""
        from ..mesh.overlays import (
            MAX_INTERACTION_ELEMENTS,
            mesh_interaction_preview,
        )

        entries = []
        for components in getattr(self, "_mesh_component_actors", {}).values():
            for component, overlay in components.items():
                base = getattr(overlay, "_mesh_preview_base_mapper", None)
                if base is None:
                    continue
                source = base.GetInput()
                kind = component
                if (
                    component == "edges"
                    and source.GetNumberOfLines()
                    and not (source.GetNumberOfPolys() or source.GetNumberOfStrips())
                ):
                    kind = "boundary_edges"
                count = source.GetNumberOfPoints()
                if kind == "boundary_edges":
                    count = (
                        source.GetLines().GetNumberOfConnectivityIds()
                        - source.GetNumberOfLines()
                    )
                elif kind == "edges":
                    count = sum(
                        c.GetNumberOfConnectivityIds()
                        for c in (source.GetPolys(), source.GetStrips())
                    )
                entries.append((overlay, base, source, kind, count))
        divisor = 1
        while (
            sum(
                min(count, MAX_INTERACTION_ELEMENTS // divisor)
                for mesh_actor, _, _, _, count in entries
                if mesh_actor.GetVisibility()
            )
            > MAX_INTERACTION_ELEMENTS
        ):
            divisor *= 2
        prepared = []
        for overlay, base, source, kind, count in entries:
            limit = MAX_INTERACTION_ELEMENTS // divisor
            old_levels = overlay._mesh_preview_levels
            same_input = getattr(overlay, "_mesh_preview_input", None) is source
            levels = dict(old_levels) if same_input else {}
            mapper = base
            if overlay.GetVisibility() and count > limit:
                mapper = levels.get(limit)
                if mapper is None:
                    data = mesh_interaction_preview(source, kind, limit)
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(data)
                    mapper.ScalarVisibilityOff()
                    levels[limit] = mapper
            prepared.append((overlay, mapper, source, levels, same_input))
        for overlay, mapper, source, levels, same_input in prepared:
            old_levels = overlay._mesh_preview_levels
            overlay._mesh_preview_levels = levels
            overlay._mesh_preview_input = source
            full = overlay._mesh_full_mapper
            if mapper is full:
                if hasattr(overlay, "_mesh_interaction_mapper"):
                    del overlay._mesh_interaction_mapper
            else:
                overlay._mesh_interaction_mapper = mapper
            overlay.SetMapper(
                mapper
                if getattr(self, "_is_interacting", False) and overlay.GetVisibility()
                else full
            )
            if not same_input:
                for previous in old_levels.values():
                    if isinstance(self.render_window, vtk.vtkWindow):
                        previous.ReleaseGraphicsResources(self.render_window)
                    previous.RemoveAllInputs()

    def _enable_fast_render(self) -> None:
        """Enables fast render mode for smoother interaction."""
        if self._is_interacting:
            return
        self._is_interacting = True
        self._set_mesh_interaction_preview(True)

        if self.render_window:
            self.render_window.SetDesiredUpdateRate(self._fast_update_rate)

        if self.streamlines_actor:
            prop = self.streamlines_actor.GetProperty()
            if prop:
                self._interaction_streamline_property = (prop, prop.GetInterpolation())
                prop.SetInterpolationToFlat()

        if self.render_window:
            self._set_interaction_multisamples(0)
            try:
                self.render_window.Render()
            except RuntimeError:
                logger.debug("VTK render failed during GPU pre-warm.")

    def _disable_fast_render(self, *, render: bool = True) -> None:
        """Restores full quality rendering after interaction stops."""
        timer = getattr(self, "_mesh_interaction_timer", None)
        if timer is not None:
            timer.stop()
        if not self._is_interacting:
            return
        self._is_interacting = False
        self._set_mesh_interaction_preview(False)

        if self.render_window:
            self.render_window.SetDesiredUpdateRate(self._original_desired_update_rate)

        if self._interaction_streamline_property is not None:
            prop, interpolation = self._interaction_streamline_property
            prop.SetInterpolation(interpolation)
            self._interaction_streamline_property = None

        if self.render_window:
            self._set_interaction_multisamples(4)

        if render and self._is_render_window_ready(self.render_window):
            self.render_window.Render()

    def update_odf_actor(
        self,
        odf_amplitudes: np.ndarray,
        sphere: Any,
        affine: np.ndarray,
        extent: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Creates/Updates the ODF glyph actor."""
        if not self.scene:
            return

        if self.odf_actor:
            self.scene.rm(self.odf_actor)
            self.odf_actor = None

        if odf_amplitudes is None:
            self.render_window.Render()
            return

        try:
            self.odf_actor = actor.odf_slicer(
                odf_amplitudes,
                sphere=sphere,
                affine=affine,
                scale=0.5,
                norm=True,
                global_cm=False,
            )
            if extent is not None:
                self.odf_actor.display_extent(*extent)

            # Mark geometry as static so VTK can reuse GPU buffers.
            self.odf_actor.GetMapper().StaticOn()
            self.scene.add(self.odf_actor)

            self.update_status("ODF Glyphs updated.", render=False)
            request_render(self, self.render_window)

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error(f"Error creating ODF actor: {e}", exc_info=True)
            self.update_status("Error creating ODF actor.")

    def remove_odf_actor(self) -> None:
        """Removes the ODF actor from the scene."""
        if self.main_window is not None:
            self.main_window._session_deferred_odf = False
        if self.odf_actor and self.scene:
            visible = self.odf_actor.GetVisibility()
            self.scene.rm(self.odf_actor)
            self.odf_actor = None
            if visible:
                request_render(self, self.render_window)

    def set_drawing_mode(
        self,
        enabled: bool,
        is_eraser: bool = False,
        is_sphere: bool = False,
        is_rectangle: bool = False,
    ) -> None:
        """Enables or disables drawing mode for 2D panels."""
        if enabled:
            self.main_window.mesh_polyline_manager.cancel()
        self.drawing_manager.set_drawing_mode(
            enabled, is_eraser, is_sphere, is_rectangle
        )

    def _update_drawing_preview(self, scene: window.Scene) -> None:
        """Updates the preview line on the active 2D scene. Delegates to DrawingManager."""
        self.drawing_manager._update_drawing_preview(scene)

    def _add_slice_to_stack(
        self, view: str, slice_actor: vtk.vtkImageSlice, layer: int
    ) -> None:
        """Composite 2D ROIs above anatomy without changing their world geometry."""
        if not hasattr(self, "_slice_stacks_2d"):
            self._slice_stacks_2d = {}
        stack = self._slice_stacks_2d.get(view)
        if stack is None:
            stack = vtk.vtkImageStack()
            # Navigation picks the anatomical plane, independently of ROI depth.
            stack.SetActiveLayer(0)
            self._slice_stacks_2d[view] = stack
            getattr(self, f"{view}_scene").add(stack)
        slice_actor.GetProperty().SetLayerNumber(layer)
        slice_actor.ForceOpaqueOn()
        stack.AddImage(slice_actor)
        self._update_stack_active_layer(stack)

    @staticmethod
    def _update_stack_active_layer(stack: vtk.vtkImageStack) -> None:
        """Pick anatomy when visible, otherwise a visible ROI as before."""
        layers = [
            image.GetProperty().GetLayerNumber()
            for image in stack.GetImages()
            if image.GetVisibility()
        ]
        stack.SetActiveLayer(0 if 0 in layers else max(layers, default=0))

    def _remove_slice_from_stacks(self, slice_actor: vtk.vtkProp) -> None:
        """Release replaced/removed slices as well as their renderer membership."""
        if isinstance(slice_actor, vtk.vtkImageSlice):
            for stack in getattr(self, "_slice_stacks_2d", {}).values():
                if stack.HasImage(slice_actor):
                    stack.RemoveImage(slice_actor)
                    self._update_stack_active_layer(stack)

    def _remove_actor_set(self, actor_dict: Dict[str, vtk.vtkActor]) -> None:
        """Removes a set of actors from all scenes."""
        scenes_to_check = [
            self.scene,
            *(getattr(self, f"{spec.name}_scene", None) for spec in ORTHO_VIEW_SPECS),
        ]
        for act in actor_dict.values():
            if act is None or not isinstance(act, vtk.vtkProp):
                continue
            self._remove_slice_from_stacks(act)
            for scn in scenes_to_check:
                if scn is None:
                    continue
                try:
                    scn.rm(act)
                except (ValueError, RuntimeError):
                    logger.debug("Failed to remove actor from scene.")

    def _adjust_sphere_radius(self, delta: float, view_type: str = "axial") -> None:
        """Adjusts the radius of the last created sphere. Delegates to DrawingManager."""
        self.drawing_manager.adjust_sphere_radius(delta, view_type)

    def _update_roi_data_in_place(self, key: str, data: np.ndarray) -> bool:
        return self.roi_scene_manager.update_roi_data_in_place(key, data)

    def update_roi_layer(
        self, key: str, data: np.ndarray, affine: np.ndarray, render: bool = True
    ) -> None:
        """Update an ROI layer, optionally leaving the final render to the caller."""
        self.roi_scene_manager.update_roi_layer(key, data, affine, render=render)

    def _show_2d_context_menu(
        self, position: QPoint, widget: QVTKRenderWindowInteractor, view_type: str
    ) -> None:
        """Shows a context menu for the 2D view that was right-clicked."""
        if not self.main_window or self.main_window.anatomical_image_data is None:
            return

        menu = QMenu(widget)
        reset_action = QAction("Reset View", widget)
        reset_action.triggered.connect(
            partial(self.scene_manager.reset_2d_view, view_type=view_type)
        )
        menu.addAction(reset_action)
        menu.exec(widget.mapToGlobal(position))

    def _setup_ortho_cameras(self) -> None:
        """Sets the cameras for the 2D views to be orthogonal."""
        self.scene_manager.setup_ortho_cameras()

    def _initialize_3d_camera(self) -> None:
        """Position the 3D camera to frame the anatomical image properly."""
        self.slice_manager.initialize_3d_camera()

    def _update_axial_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the axial 2D camera to follow its slice."""
        self.scene_manager.update_axial_camera(reset_zoom_pan)

    def _update_coronal_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the coronal 2D camera to follow its slice."""
        self.scene_manager.update_coronal_camera(reset_zoom_pan)

    def _update_sagittal_camera(self, reset_zoom_pan: bool = False) -> None:
        """Updates the sagittal 2D camera to follow its slice."""
        self.scene_manager.update_sagittal_camera(reset_zoom_pan)

    def update_status(self, message: str, *, render: bool = True) -> None:
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

            if message.startswith("Slice View:"):
                prefix = ""
                status_suffix = ""

            full_message = f"Status: {prefix}{message}{status_suffix}"
            self.status_text_actor.SetInput(str(full_message))

            if render and self._is_render_window_ready(self.render_window):
                request_render(self, self.render_window)
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Error updating status text actor:", exc_info=e)

    def update_progress_status(self, message: str) -> None:
        self.update_status(message, render=False)
        self.main_window.statusBar().showMessage(message, 3000)

    def update_progress_bar(
        self, value: int, maximum: int, visible: bool = True
    ) -> None:
        """Updates the overlay progress bar positioned at the bottom of the VTK window."""
        if not visible:
            self.overlay_progress_bar.hide()
            return

        w = self.vtk_widget.width()
        h = self.vtk_widget.height()

        bar_w = 200
        bar_h = 15

        x_pos = 500
        y_pos = h - bar_h - 9

        if x_pos + bar_w > w:
            x_pos = max(10, w - bar_w - 5)

        updates_enabled = self.overlay_progress_bar.updatesEnabled()
        self.overlay_progress_bar.setUpdatesEnabled(False)
        try:
            self.overlay_progress_bar.setGeometry(x_pos, y_pos, bar_w, bar_h)
            self.overlay_progress_bar.setRange(0, maximum)
            self.overlay_progress_bar.setValue(value)
            self.overlay_progress_bar.show()
        finally:
            self.overlay_progress_bar.setUpdatesEnabled(updates_enabled)

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
        self.radius_actor.SetPickable(0)
        prop = self.radius_actor.GetProperty()
        prop.SetColor(0.2, 0.5, 1.0)
        prop.SetOpacity(0.3)
        prop.SetRepresentationToSurface()
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

        if needs_render and self._is_render_window_ready(self.render_window):
            request_render(self, self.render_window)

    # Anatomical Slice Delegates
    def update_anatomical_slices(self, preserve_view: bool = False) -> None:
        """Creates or updates the FURY slicer actors for anatomical image."""
        self.slice_manager.update_anatomical_slices(preserve_view=preserve_view)

    def clear_anatomical_slices(self, reset_state: bool = True) -> None:
        """Removes anatomical slice actors from all active scenes and resets slice state."""
        self.slice_manager.clear_anatomical_slices(reset_state=reset_state)

    # Mesh Scene Delegates
    def _set_interaction_multisamples(self, samples: int) -> None:
        self.mesh_scene_manager.set_interaction_multisamples(samples)

    def _sync_mesh_transparency(self) -> None:
        self.mesh_scene_manager.sync_mesh_transparency()

    def add_mesh_layer(self, layer: MeshLayer, *, render: bool = True) -> vtk.vtkActor:
        return self.mesh_scene_manager.add_mesh_layer(layer, render=render)

    def remove_mesh_layer(self, key: str, *, render: bool = True) -> None:
        self.mesh_scene_manager.remove_mesh_layer(key, render=render)

    def _release_mesh_component(self, mesh_actor: vtk.vtkActor) -> None:
        self.mesh_scene_manager.release_mesh_component(mesh_actor)

    def _discard_mesh_components(self, key: str) -> None:
        self.mesh_scene_manager.discard_mesh_components(key)

    def _sync_mesh_camera_observers(self) -> None:
        self.mesh_scene_manager.sync_mesh_camera_observers()

    def _mesh_scene_bounds(self, *, clipping: bool) -> tuple[float, ...] | None:
        return self.mesh_scene_manager.mesh_scene_bounds(clipping=clipping)

    def _reset_mesh_camera(self, caller, event) -> None:
        self.mesh_scene_manager.reset_mesh_camera(caller, event)

    def _reset_mesh_clipping(self, caller, event) -> None:
        self.mesh_scene_manager.reset_mesh_clipping(caller, event)

    def mesh_overlay_data(self, layer: MeshLayer, component: str):
        return self.mesh_scene_manager.mesh_overlay_data(layer, component)

    def publish_mesh_overlay(
        self, layer: MeshLayer, component: str, data, *, render: bool = True,
    ) -> None:
        self.mesh_scene_manager.publish_mesh_overlay(
            layer, component, data, render=render
        )

    def set_mesh_layer_visibility(
        self, key: str, visible: bool, *, layer: MeshLayer | None = None,
        render: bool = True,
    ) -> bool:
        return self.mesh_scene_manager.set_mesh_layer_visibility(
            key, visible, layer=layer, render=render
        )

    def set_mesh_component_style(
        self, key: str, component: MeshComponent, style: MeshComponentStyle,
        *, render: bool = True,
    ) -> None:
        self.mesh_scene_manager.set_mesh_component_style(
            key, component, style, render=render
        )

    def set_mesh_opacity(self, key: str, value: float, *, render: bool = True) -> None:
        self.mesh_scene_manager.set_mesh_opacity(key, value, render=render)

    def set_mesh_color(
        self, key: str, color: tuple[float, float, float], *, render: bool = True
    ) -> None:
        self.mesh_scene_manager.set_mesh_color(key, color, render=render)

    def maximum_texture_size(self) -> int:
        return self.mesh_scene_manager.maximum_texture_size()

    def mesh_texture_enabled(self, key: str) -> bool:
        return self.mesh_scene_manager.mesh_texture_enabled(key)

    def set_mesh_texture(
        self, layer: MeshLayer, preparation: MeshTexturePreparation | None,
        enabled: bool, *, gl_maximum: int | None = None,
        budgets: "MeshTextureBudgets | None" = None, render: bool = True,
    ) -> None:
        self.mesh_scene_manager.set_mesh_texture(
            layer,
            preparation,
            enabled,
            gl_maximum=gl_maximum,
            budgets=budgets,
            render=render,
        )

    def discard_mesh_texture(self, key: str) -> None:
        self.mesh_scene_manager.discard_mesh_texture(key)

    def fit_mesh_in_view(self, key: str, *, render: bool = True) -> None:
        self.mesh_scene_manager.fit_mesh_in_view(key, render=render)

    def set_streamlines_opacity(self, value: float) -> None:
        """Sets the opacity of the main streamlines actor."""
        if self.streamlines_actor:
            self.streamlines_actor.GetProperty().SetOpacity(value)
            self.render_window.Render()

    def set_anatomical_opacity(self, value: float) -> None:
        """Sets the opacity of the anatomical slices."""
        self.slice_manager.set_anatomical_opacity(value)

    def set_roi_opacity(self, key: str, value: float) -> None:
        """Sets the opacity for a specific ROI layer."""
        self.roi_scene_manager.set_roi_opacity(key, value)

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
                return np.array(vox_coord)
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
                return np.array(world_coord)
            affine = self.main_window.anatomical_image_affine
            inv_affine = np.linalg.inv(affine)

        homog_world = np.array([world_coord[0], world_coord[1], world_coord[2], 1.0])
        vox_coord = np.dot(inv_affine, homog_world)
        return vox_coord[:3]

    # Create/Update Crosshairs
    def _create_or_update_crosshairs(self) -> None:
        """Creates or updates the 2D crosshair actors based on current slice indices."""
        self.scene_manager.create_or_update_crosshairs()

    # Slice Navigation Delegates
    def move_slice(self, axis: str, direction: int) -> None:
        self.slice_manager.move_slice(axis, direction)

    def set_slice_indices(
        self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None
    ) -> None:
        self.slice_manager.set_slice_indices(x=x, y=y, z=z)

    def _update_slow_slice_components(self, *, render: bool = True) -> None:
        self.slice_manager.update_slow_slice_components(render=render)

    def set_slices_from_ras(self, ras_coords: np.ndarray) -> None:
        self.slice_manager.set_slices_from_ras(ras_coords)

    def _navigate_2d_view(self, obj: vtk.vtkObject, event_id: str) -> None:
        self.slice_manager.navigate_2d_view(obj, event_id)

    # Streamline Actor Management
    def update_highlight(self) -> None:
        """Updates the actor for highlighted/selected streamlines. Delegates to StreamlinesManager."""
        self.streamlines_manager.update_highlight()

    def stage_bundle_scene(self, rollback: ExitStack) -> None:
        """Keep the resident actors and camera available until the load commits."""
        camera = self.scene.GetActiveCamera()
        previous_camera = vtk.vtkCamera()
        previous_camera.DeepCopy(camera)
        rollback.callback(camera.DeepCopy, previous_camera)
        if self.status_text_actor is not None:
            rollback.callback(
                self.status_text_actor.SetInput, self.status_text_actor.GetInput()
            )

        presentation = self.main_window.surface_mapping_manager.presentation
        if presentation is not None:
            rollback.callback(presentation.sync)
            rollback.callback(setattr, presentation, "visible", presentation.visible)
            presentation.visible = False
            presentation.sync()

        def restore_actor(name, previous, attached):
            current = getattr(self, name)
            if current is not None:
                self.scene.rm(current)
            setattr(self, name, previous)
            if attached:
                self.scene.add(previous)

        for name in (
            "streamlines_actor",
            "highlight_actor",
            "invert_contour_actor",
            "roi_highlight_actor",
        ):
            previous = getattr(self, name)
            attached = previous is not None and self.scene.HasViewProp(previous)
            rollback.callback(restore_actor, name, previous, attached)
            if previous is not None:
                if attached:
                    self.scene.rm(previous)
                setattr(self, name, None)

        for scene_actor in (self.scalar_bar_actor, self.odf_actor):
            if scene_actor is not None and scene_actor.GetVisibility():
                rollback.callback(scene_actor.SetVisibility, True)
                scene_actor.SetVisibility(False)

    def update_main_streamlines_actor(
        self, force: bool = False, *, loading: bool = False
    ) -> None:
        """Recreates the main actor using the skipped/strided dataset."""
        self.streamlines_manager.update_main_streamlines_actor(
            force=force, loading=loading
        )

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
        except (OSError, RuntimeError, ValueError) as e:
            self.update_status(f"Error during save: {e}")

    def _handle_radius_change(self, increase: bool = True) -> None:
        """Handles increasing or decreasing the selection radius via main window."""
        if not self.main_window:
            return
        if not self.main_window.tractogram_data:
            self.update_status("Radius change (+/-): No streamlines loaded.")
            return
        if increase:
            self.main_window.state_manager.increase_radius()
        else:
            self.main_window.state_manager.decrease_radius()

    def _find_streamlines_in_radius(
        self, center_point: np.ndarray, radius: float, check_all: bool = False
    ) -> Set[int]:
        """Finds streamlines within a sphere. Delegates to SelectionManager."""
        return self.selection_manager.find_streamlines_in_radius(
            center_point, radius, check_all
        )

    def _find_streamlines_in_oriented_box(
        self,
        voxel_to_world: np.ndarray,
        voxel_min: np.ndarray,
        voxel_max: np.ndarray,
        check_all: bool = False,
    ) -> Set[int]:
        """Find streamlines intersecting an affine-oriented voxel box."""
        return self.selection_manager.find_streamlines_in_oriented_box(
            voxel_to_world, voxel_min, voxel_max, check_all
        )

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
            interactor.SetKeyCode(chr(0))
            return

        # Handle non-data-dependent keys first
        non_data_handlers = {
            ("z", True, False): self.main_window._perform_undo,
            ("y", True, False): self.main_window._perform_redo,
            ("z", True, True): self.main_window._perform_redo,
            ("escape", False, False): self.main_window.state_manager.hide_sphere,
            ("s", True, False): self._handle_save_shortcut,
        }
        if handler_key in non_data_handlers:
            non_data_handlers[handler_key]()
            return

        # Handle Slice Navigation Keys
        anatomical_loaded = self.main_window.anatomical_image_data is not None
        if anatomical_loaded:
            slice_nav_handlers = {
                ("up", False, False): ("z", 1),
                ("down", False, False): ("z", -1),
                ("up", True, False): ("y", 1),
                ("down", True, False): ("y", -1),
                # Radiological convention: screen-right = anatomical-left = decreasing voxel X
                ("right", False, False): ("x", -1),
                ("left", False, False): ("x", 1),
            }
            if handler_key in slice_nav_handlers:
                axis, direction = slice_nav_handlers[handler_key]
                self.move_slice(axis, direction)
                return

        streamline_data_handlers = {
            "i": self.selection_manager.invert_selection,
            "plus": lambda: self._handle_radius_change(increase=True),
            "equal": lambda: self._handle_radius_change(increase=True),
            "minus": lambda: self._handle_radius_change(increase=False),
            "d": self.main_window.state_manager.perform_delete_selection,
            "c": self.main_window.state_manager.perform_clear_selection,
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
            interactor.SetKeyCode(chr(0))
            return

        if key == "s" and not ctrl:
            self.selection_manager.handle_streamline_selection(deselect=shift)
            interactor.SetKeyCode(chr(0))
            return

    # ROI Layer Actor Delegates
    def _remove_roi_actor_entries(self, key: str, *actor_keys: str) -> None:
        self.roi_scene_manager.remove_roi_actor_entries(key, *actor_keys)

    def _sync_roi_3d_actor(
        self,
        key: str,
        data: np.ndarray,
        affine: np.ndarray,
        render: bool,
    ) -> None:
        self.roi_scene_manager.sync_roi_3d_actor(key, data, affine, render)

    def add_roi_layer(
        self, key: str, data: np.ndarray, affine: np.ndarray, render: bool = True
    ) -> None:
        self.roi_scene_manager.add_roi_layer(key, data, affine, render=render)

    def _apply_display_correction(self, roi_key: str) -> None:
        """Applies display correction only to 2D ROI slices."""
        self.roi_scene_manager.apply_display_correction(roi_key)

    def set_roi_layer_visibility(self, key: str, visible: bool) -> None:
        self.roi_scene_manager.set_roi_layer_visibility(key, visible)

    def remove_roi_layer(self, key: str) -> None:
        self.roi_scene_manager.remove_roi_layer(key)

    def clear_all_roi_layers(self) -> None:
        self.roi_scene_manager.clear_all_roi_layers()

    def set_roi_layer_color(
        self,
        key: str,
        color: Tuple[float, float, float],
        render: bool = True,
    ) -> None:
        self.roi_scene_manager.set_roi_layer_color(key, color, render=render)

    def take_screenshot(self) -> None:
        """Saves a screenshot of the VTK view with an opaque black background, hiding UI overlays."""
        self._disable_fast_render()
        if not self.render_window or not self.scene:
            QMessageBox.warning(
                self.main_window,
                "Screenshot Error",
                "Render window or scene not available.",
            )
            return
        if (
            self.main_window.tractogram_data is None
            and self.main_window.anatomical_image_data is None
            and not self.main_window.mesh_layers
            and not self.main_window.standalone_polylines
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
            slice_info = (
                f"_x{self.current_slice_indices['x']}"
                f"_y{self.current_slice_indices['y']}"
                f"_z{self.current_slice_indices['z']}"
            )

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
            self.update_status(
                f"Screenshot saved: {os.path.basename(output_path)}", render=False
            )

        except (RuntimeError, OSError, ValueError, AttributeError):
            error_msg = "Error taking screenshot:"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self.main_window, "Screenshot Error", error_msg)
            self.update_status("Error saving screenshot.", render=False)
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

            if self._is_render_window_ready(self.render_window):
                request_render(self, self.render_window)

    def cleanup(self) -> None:
        """Safely cleans up VTK resources for all 4 views."""
        timer = getattr(self, "_mesh_interaction_timer", None)
        if timer is not None:
            timer.stop()
        for mesh_id in tuple(self.mesh_actors):
            self.discard_mesh_texture(mesh_id)
            self._discard_mesh_components(mesh_id)
        self.mesh_actors.clear()

        for widget_attr in ("axes_widget", "orientation_widget"):
            widget = getattr(self, widget_attr, None)
            if widget:
                try:
                    widget.SetEnabled(0)
                    widget.SetInteractor(None)
                    setattr(self, widget_attr, None)
                except RuntimeError:
                    logger.debug("Failed to release %s.", widget_attr)

        view_prefixes = ("", *(f"{spec.name}_" for spec in ORTHO_VIEW_SPECS))

        for prefix in view_prefixes:
            scene = getattr(self, f"{prefix}scene", None)
            if scene:
                try:
                    scene.clear()
                except RuntimeError:
                    logger.debug("Failed to clear VTK scene during cleanup.")

        for prefix in view_prefixes:
            interactor = getattr(self, f"{prefix}interactor", None)
            if interactor:
                try:
                    style = interactor.GetInteractorStyle()
                    if style is not None:
                        style.RemoveAllObservers()
                        if hasattr(style, "vtk_panel"):
                            style.vtk_panel = None
                        interactor.SetInteractorStyle(None)
                    interactor.RemoveAllObservers()
                    if interactor.GetInitialized():
                        interactor.TerminateApp()
                except RuntimeError:
                    logger.debug("Failed to clean up VTK interactor.")

        for prefix in view_prefixes:
            widget = getattr(self, f"{prefix}vtk_widget", None)
            if widget:
                try:
                    rw = widget.GetRenderWindow()
                    if rw:
                        rw.Finalize()
                    widget.close()
                except RuntimeError:
                    logger.debug("Failed to finalize VTK render window.")

        for attr in ("scene", "interactor", "render_window", "vtk_widget"):
            for prefix in view_prefixes:
                setattr(self, f"{prefix}{attr}", None)
