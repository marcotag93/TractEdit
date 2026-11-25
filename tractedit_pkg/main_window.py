# -*- coding: utf-8 -*-

"""
Contains the MainWindow class for the tractedit GUI application.

Handles the main application window, menus, actions, status bar,
and coordinates interactions between UI elements, data state,
file I/O, and the VTK panel.
"""

# ============================================================================
# Imports
# ============================================================================

import os
import numpy as np
from typing import TYPE_CHECKING, Optional, List, Set, Dict, Any, Tuple
import nibabel as nib
import logging

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QLabel,
    QStatusBar,
    QApplication,
    QToolBar,
    QDoubleSpinBox,
    QSpinBox,
    QSlider,
    QDockWidget,
    QTreeWidget,
    QStyle,
    QLineEdit,
    QCheckBox,
    QToolButton,
    QProgressDialog,
)
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QIcon,
    QCloseEvent,
    QPixmap,
)
from PyQt6.QtCore import QEvent, Qt, pyqtSlot, QTimer, QSettings

from . import file_io
from . import SESSION_EXTENSION
from . import odf_utils
from .file_dialogs import choose_input
from .render_batch import batch_render, request_render
from .reference_grid import ReferenceGrid
from .file_io.density_map import density_grid_from_points, density_image
from .transactional_io import transactional_save
from .utils import (
    ColorMode,
    get_formatted_datetime,
    get_asset_path,
    format_tuple,
    signals_blocked,
    DEFAULT_SELECTION_RADIUS,
    TARGET_RENDER_COUNT,
)
from .visualization import VTKPanel

if TYPE_CHECKING:
    from .data_contracts import (
        AnatomicalImageLoadResult, MeshLayer, MeshPolyline, RoiLayer,
    )
from .visualization.drawing import (
    _rasterize_world_sphere,
    _remove_actor_from_2d_scenes,
)
from .ui import (
    ActionsManager,
    ToolbarsManager,
    DataPanelManager,
    DrawingModesManager,
    ThemeManager,
    ThemeMode,
)
from .ui.data_panel import classify_region_hemisphere, image_volume_label
from .logic import (
    ROIManager,
    StateManager,
    ScalarManager,
    ConnectivityManager,
    ImageLoaderManager,
    ODFManager,
)
from .logic.state_manager import retain_polyline_history
from .logic.mesh_manager import MeshManager
from .logic.mesh_polyline_manager import MeshPolylineManager
from .logic.surface_mapping_manager import SurfaceMappingManager
from .ui.update_notifier import UpdateNotifier
from .ui.appimage_integration import AppImageIntegrationController

logger = logging.getLogger(__name__)


# ============================================================================
# Main Window Class
# ============================================================================


class MainWindow(QMainWindow):
    """
    Main application window for TractEdit.
    Sets up the UI, manages application state (streamlines, selection, undo/redo),
    and delegates rendering/interaction to VTKPanel and file I/O to file_io.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._tree_icons = {}
        super().__init__(parent)

        # Initialize Streamline Data Variables
        self.tractogram_data: Optional["nib.streamlines.ArraySequence"] = None
        self._tractogram_data_version: int = 0
        self.streamline_bboxes: Optional[np.ndarray] = None
        self.visible_indices: Set[int] = set()
        self.original_trk_header: Optional[Dict[str, Any]] = (
            None  # Header dict from loaded file
        )
        self.original_trk_affine: Optional[np.ndarray] = (
            None  # Affine matrix (affine_to_rasmm)
        )
        self.tractogram_reference_grid: Optional[ReferenceGrid] = None
        self.original_trk_path: Optional[str] = None  # Full path
        self.original_file_extension: Optional[str] = (
            None  # '.trk', '.tck', '.trx', or None
        )
        self.trx_file_reference: Optional[Any] = None  # TRX memmap object reference
        self.scalar_data_per_point: Optional[
            Dict[str, "nib.streamlines.ArraySequence"]
        ] = None  # Dictionary: {scalar_name: [scalar_array_sl0, ...]}
        self.data_per_streamline: Optional[Dict[str, np.ndarray]] = None
        self.scalar_display: Optional[dict] = None
        self.active_scalar_name: Optional[str] = (
            None  # Key for the currently active scalar
        )
        self.selected_streamline_indices: Set[int] = (
            set()
        )  # Indices of selected streamlines

        # Inversion-mode state — reset whenever the selection is cleared,
        # deleted, or a new sphere selection starts.
        self._inversion_active: bool = False
        self._inversion_keeper_indices: Set[int] = set()

        self.selection_radius_3d: float = (
            DEFAULT_SELECTION_RADIUS  # Radius for sphere selection
        )
        self.render_stride: int = 1  # 1 = Show all, 100 = Show 1%
        self._skip_user_disabled: bool = False  # User explicitly turned off skip
        self.bundle_opacity: float = 1.0  # Default to 1.0
        self.image_opacity: float = 1.0

        # Dirty tracking for actor rebuild optimization
        self._visibility_version: int = 0
        self._last_visibility_version: int = -1
        self._last_render_stride: int = 1
        self._last_color_mode: Optional[ColorMode] = None
        self._last_active_scalar: Optional[str] = None
        self._last_tube_mode: bool = False
        self._last_bundle_opacity: float = 1.0
        self.roi_opacities: Dict[str, float] = {}

        # Background thread references (for cleanup)
        self._loader_thread: Optional[Any] = None
        self._image_loader_thread: Optional[Any] = None
        self._medoid_thread: Optional[Any] = None
        self._mesh_loader_thread: Optional[Any] = None
        self._mesh_texture_thread: Optional[Any] = None
        self._mesh_overlay_thread: Optional[Any] = None
        self._background_workers: List[Any] = []
        self._deferred_trx_owners: List[Any] = []
        self._bundle_load_generation = 0
        self._image_load_generation = 0
        self._medoid_generation = 0
        self._mesh_load_generation = 0
        self._polyline_io_thread = None
        self._polyline_io_generation = 0
        self._polyline_preview_thread = None
        self._polyline_preview_generation = 0
        self._mesh_texture_generation = 0
        self._mesh_overlay_generation = 0
        self._mesh_commit_active = False
        self._shutdown_requested = False
        self.session_path: Optional[str] = None
        self._session_busy = False
        self._session_source_associations: Dict[str, Any] = {}
        self._session_deferred_odf = False
        self._session_deferred_odf_owner = None
        self._session_deferred_regions: Dict[int, Dict[str, Any]] = {}

        # Initialize Anatomical Image Data Variables
        self.anatomical_image_path: Optional[str] = None
        self.anatomical_image_data: Optional[np.ndarray] = None  # Numpy array
        self.anatomical_image_affine: Optional[np.ndarray] = None  # 4x4 numpy array
        self.anatomical_mmap_image: Optional["file_io.MemoryMappedImage"] = None
        self.anatomical_reference_grid: Optional[ReferenceGrid] = None

        # Unified Undo/Redo Stacks (all operations - streamlines and ROI)
        self.unified_undo_stack: List[Dict[str, Any]] = []
        self.unified_redo_stack: List[Dict[str, Any]] = []

        # View State
        self.current_color_mode: ColorMode = ColorMode.DEFAULT
        self.bundle_is_visible: bool = True
        self.image_is_visible: bool = True
        self.roi_visibility: Dict[str, bool] = {}
        self.render_as_tubes: bool = False  # False = Lines, True = Tubes

        # Scalar Range Variables
        self.scalar_min_val: float = 0.0  # Current min value for the colormap
        self.scalar_max_val: float = 1.0  # Current max value for the colormap
        self.scalar_data_min: float = 0.0  # Actual min value in the loaded data
        self.scalar_data_max: float = 1.0  # Actual max value in the loaded data
        self.scalar_range_initialized: bool = (
            False  # Flag to check if range has been calculated
        )
        self.scalar_toolbar: Optional[QToolBar] = None
        self.scalar_min_spinbox: Optional[QDoubleSpinBox] = None
        self.scalar_max_spinbox: Optional[QDoubleSpinBox] = None
        self.scalar_min_slider: Optional[QSlider] = None
        self.scalar_max_slider: Optional[QSlider] = None

        # ODF / Glyphs Data
        self.odf_data: Optional[np.ndarray] = None
        self.odf_affine: Optional[np.ndarray] = None
        self.odf_path: Optional[str] = None
        self.odf_sh_order: int = 0
        self.odf_sphere = None
        self.odf_basis_matrix = None
        self.odf_tunnel_sphere = None  # Lower-res sphere for tunnel display
        self.odf_tunnel_basis = None  # SH basis for tunnel display sphere
        self.odf_tunnel_is_visible: bool = False
        self.MAX_ODF_STREAMLINES = 26000  # Safety limit for Tunnel View

        # Parcellation / Connectivity Data
        self.parcellation_data: Optional[np.ndarray] = None
        self._parcellation_label_count: Optional[int] = None
        self._parcellation_data_version: int = 0
        self.parcellation_affine: Optional[np.ndarray] = None
        self.parcellation_path: Optional[str] = None
        self.parcellation_labels: Dict[int, str] = {}  # Label ID -> Region name

        # Data Panel / Dock Widget
        self.data_dock_widget: Optional[QDockWidget] = None
        self.data_tree_widget: Optional[QTreeWidget] = None

        # Debounce timer for data panel updates (prevents expensive rebuilds)
        self._data_panel_debounce_timer: Optional[QTimer] = None
        self._data_panel_update_pending: bool = False

        # ROI Layer Data Variables
        self.roi_layers: Dict[str, "RoiLayer"] = (
            {}
        )  # Key: path, Val: {'data':, 'affine':, 'inv_affine':}
        self.mesh_layers: Dict[str, "MeshLayer"] = {}
        self.standalone_polylines: list["MeshPolyline"] = []
        self.mesh_polyline_mesh_id: Optional[str] = None
        self.mesh_polyline_points: list[np.ndarray] = []

        # Status Bar Widgets
        self.permanent_status_widget: Optional[QWidget] = None
        self.data_info_label: Optional[QLabel] = None
        self.ras_coordinate_label: Optional[QLabel] = None

        # ROI Logic State
        self.roi_states: Dict[str, Dict[str, bool]] = {}
        self.roi_intersection_cache: Dict[str, Set[int]] = {}
        self.roi_highlight_indices: Set[int] = set()
        self.manual_visible_indices: Set[int] = (
            set()
        )  # Tracks manual deletions separate from filters

        # Manual ROI Drawing State
        self.is_drawing_mode: bool = False
        self.is_eraser_mode: bool = False  # Eraser mode for ROI
        self.current_drawing_roi: Optional[str] = None
        self.manual_roi_counter: int = 0
        self.draw_brush_size: int = 1  # Number of voxels (1 = single voxel)

        # Drawing Mode State (sphere / rectangle)
        self.is_sphere_mode: bool = False
        self.is_rectangle_mode: bool = False

        # Parcellation Overlay & Region State
        self._parcellation_overlay_visible: bool = False
        self._parcellation_overlay_cached: bool = False
        self._parcellation_overlay_cache_key: Optional[Tuple[Any, ...]] = None
        self.parcellation_overlay_actor: Optional[Any] = None
        self.parcellation_connected_labels: Set[int] = set()
        self.parcellation_region_visibility: Dict[int, bool] = {}
        self.parcellation_main_labels: Set[int] = set()
        self.parcellation_label_colors: Dict[int, Tuple[float, ...]] = {}
        self.parcellation_region_actors: Dict[int, Any] = {}
        self.parcellation_region_states: Dict[int, Dict[str, bool]] = {}

        # Parcellation filter data (also reset in _clear_parcellation)
        self.parcellation_region_intersection_cache: Dict[int, Set[int]] = {}
        self.parcellation_start_labels: Optional[np.ndarray] = None
        self.parcellation_end_labels: Optional[np.ndarray] = None
        self.parcellation_visible_indices: Optional[np.ndarray] = None
        self._parcellation_endpoint_labels_computed: Optional[np.ndarray] = None
        self._parcellation_endpoint_cache_key: Optional[
            Tuple[int, int, int, bytes, int, int, int]
        ] = None

        # Pending / Cache
        self._pending_expanded_items: Set[str] = set()

        # Actions — File menu
        self.load_file_action: Optional[QAction] = None
        self.load_bg_image_action: Optional[QAction] = None
        self.load_odf_action: Optional[QAction] = None
        self.view_odf_tunnel_action: Optional[QAction] = None
        self.load_parcellation_action: Optional[QAction] = None
        self.view_parcellation_action: Optional[QAction] = None
        self.close_bundle_action: Optional[QAction] = None
        self.clear_bg_image_action: Optional[QAction] = None
        self.load_roi_action: Optional[QAction] = None
        self.new_roi_action: Optional[QAction] = None
        self.draw_mode_action: Optional[QAction] = None
        self.erase_mode_action: Optional[QAction] = None
        self.sphere_mode_action: Optional[QAction] = None
        self.rectangle_mode_action: Optional[QAction] = None
        self.clear_all_rois_action: Optional[QAction] = None
        self.clear_all_data_action: Optional[QAction] = None
        self.save_file_action: Optional[QAction] = None
        self.screenshot_action: Optional[QAction] = None
        self.export_html_action: Optional[QAction] = None
        self.save_density_map_action: Optional[QAction] = None
        self.exit_action: Optional[QAction] = None

        # Actions — Edit menu
        self.undo_action: Optional[QAction] = None
        self.redo_action: Optional[QAction] = None

        # Actions — View / Geometry
        self.geo_lines_action: Optional[QAction] = None
        self.geo_tubes_action: Optional[QAction] = None
        self.geometry_action_group: Optional[QActionGroup] = None

        # Actions — View / Coloring
        self.color_default_action: Optional[QAction] = None
        self.color_orientation_action: Optional[QAction] = None
        self.color_scalar_action: Optional[QAction] = None
        self.coloring_action_group: Optional[QActionGroup] = None

        # Actions — Commands menu
        self.calc_centroid_action: Optional[QAction] = None
        self.calc_medoid_action: Optional[QAction] = None
        self.compute_connectivity_action: Optional[QAction] = None
        self.clear_select_action: Optional[QAction] = None
        self.delete_select_action: Optional[QAction] = None
        self.reset_camera_action: Optional[QAction] = None
        self.increase_radius_action: Optional[QAction] = None
        self.decrease_radius_action: Optional[QAction] = None
        self.hide_sphere_action: Optional[QAction] = None

        # Actions — Settings / Theme
        self.theme_light_action: Optional[QAction] = None
        self.theme_dark_action: Optional[QAction] = None
        self.theme_system_action: Optional[QAction] = None
        self.theme_action_group: Optional[QActionGroup] = None
        self.auto_fill_action: Optional[QAction] = None

        # Actions — Help / Data panel
        self.about_action: Optional[QAction] = None
        self.toggle_data_panel_action: Optional[QAction] = None

        # Toolbar widgets
        self.main_toolbar: Optional[QToolBar] = None
        self.skip_checkbox: Optional[QCheckBox] = None
        self.skip_spinbox: Optional[QSpinBox] = None
        self.opacity_label: Optional[QLabel] = None
        self.opacity_slider: Optional[QSlider] = None
        self.brush_label: Optional[QLabel] = None
        self.brush_size_label: Optional[QLabel] = None
        self.brush_size_slider: Optional[QSlider] = None
        self.draw_mode_button: Optional[QToolButton] = None
        self.erase_mode_button: Optional[QToolButton] = None
        self.sphere_mode_button: Optional[QToolButton] = None
        self.rectangle_mode_button: Optional[QToolButton] = None
        self.sphere_radius_container: Optional[QWidget] = None
        self.sphere_radius_spinbox: Optional[QDoubleSpinBox] = None
        self.scalar_reset_button: Optional[QAction] = None

        # Status bar widgets
        self.status_bar: Optional[QStatusBar] = None
        self.ras_label: Optional[QLabel] = None
        self.ras_coordinate_input: Optional[QLineEdit] = None

        # Central widget / VTK panel
        self.central_widget: Optional[QWidget] = None
        self.vtk_panel: Optional[Any] = None  # VTKPanel (imported lazily)

        # Window Properties
        self.setWindowTitle("TractEdit GUI - Interactive Editor")
        self.setMinimumSize(800, 600)

        # Settings
        self.auto_fill_voxels: bool = False
        self.settings = QSettings("TractEdit", "TractEdit")
        self.update_notifier = UpdateNotifier(self)
        self.appimage_integration = AppImageIntegrationController(self)

        # UI Managers
        self.actions_manager = ActionsManager(self)
        self.toolbars_manager = ToolbarsManager(self)
        self.data_panel_manager = DataPanelManager(self)
        self.drawing_modes_manager = DrawingModesManager(self)
        self.theme_manager = ThemeManager(self)

        # Logic Managers (must be created before ActionsManager/ToolbarsManager wire direct signals)
        self.roi_manager = ROIManager(self)
        self.state_manager = StateManager(self)
        self.scalar_manager = ScalarManager(self)
        self.connectivity_manager = ConnectivityManager(self)
        self.image_loader_manager = ImageLoaderManager(self)
        self.odf_manager = ODFManager(self)
        self.mesh_manager = MeshManager(self)
        self.mesh_polyline_manager = MeshPolylineManager(self)
        self.surface_mapping_manager = SurfaceMappingManager(self)

        # Setup UI Components
        self.actions_manager.create_actions()
        self.data_panel_manager.create_data_panel()  # Must be before create_menus for dock toggle
        self.actions_manager.create_menus()
        self.toolbars_manager.create_main_toolbar()
        self.toolbars_manager.create_scalar_toolbar()
        self.toolbars_manager.setup_status_bar()
        self.toolbars_manager.setup_central_widget()  # This creates the VTKPanel
        self.vtk_panel.vtk_widget.installEventFilter(self.mesh_polyline_manager)

        # Initialize theme after all UI components are created
        self.theme_manager.initialize_theme()

        # Initial Status Update
        self._update_initial_status()
        self._update_action_states()
        self._update_bundle_info_display()

        # Load persisted settings
        self._load_settings()

    @pyqtSlot(int)
    def _on_brush_size_changed(self, value: int) -> None:
        """Updates the brush size when the slider value changes."""
        self.draw_brush_size = value
        self.brush_size_label.setText(str(value))
        if self.vtk_panel:
            self.vtk_panel.update_progress_status(
                f"Brush size set to {value} voxel{'s' if value > 1 else ''}"
            )

    @pyqtSlot(float)
    def _on_sphere_radius_preview(self, value: float) -> None:
        """
        Shows a yellow circle preview when the radius spinbox value changes.

        The preview appears on the 2D view where the sphere was created,
        showing the new radius at the sphere's center location.
        """
        if not self.vtk_panel:
            return

        roi_name = self.current_drawing_roi
        if not roi_name:
            return

        # Check if sphere params exist for this ROI
        if not hasattr(self.vtk_panel, "sphere_params_per_roi"):
            return
        if roi_name not in self.vtk_panel.sphere_params_per_roi:
            return

        roi_params = self.vtk_panel.sphere_params_per_roi[roi_name]
        center_3d = roi_params["center"].copy()
        stored_view_type = roi_params.get("view_type", "axial")

        # Undo radiological X-flip for 2D preview display (stored center is in 3D world coords)
        center_display = center_3d.copy()
        if stored_view_type in ["axial", "coronal"]:
            center_display[0] = -center_display[0]

        # Get the appropriate 2D scene
        if stored_view_type == "axial":
            scene = self.vtk_panel.axial_scene
        elif stored_view_type == "coronal":
            scene = self.vtk_panel.coronal_scene
        else:
            scene = self.vtk_panel.sagittal_scene

        if not scene:
            return

        # Create yellow circle preview
        self.vtk_panel.drawing_manager._show_radius_preview(
            center_display, value, stored_view_type, scene
        )

    @pyqtSlot()
    @batch_render(lambda self: self.vtk_panel)
    def _on_sphere_radius_changed(self) -> None:
        """
        Updates the sphere radius when editing is finished.

        Directly modifies the ROI voxel data and refreshes visualization,
        bypassing the preview system for immediate update with correct color.
        """
        if not self.vtk_panel:
            return

        # Get value from spinbox since editingFinished doesn't pass it
        if self.sphere_radius_spinbox is None:
            return
        value = self.sphere_radius_spinbox.value()

        roi_name = self.current_drawing_roi
        if not roi_name:
            return

        # Check if sphere params exist for this ROI
        if not hasattr(self.vtk_panel, "sphere_params_per_roi"):
            return
        if roi_name not in self.vtk_panel.sphere_params_per_roi:
            return
        if roi_name not in self.roi_layers:
            return

        roi_params = self.vtk_panel.sphere_params_per_roi[roi_name]
        current_radius = roi_params.get("radius", 0)

        # Only update if radius actually changed
        if abs(current_radius - value) < 0.01:
            return

        # Get stored center (already in 3D-corrected format)
        center_3d = roi_params["center"].copy()
        stored_view_type = roi_params.get("view_type", "axial")

        # Get ROI layer data
        roi_layer = self.roi_layers[roi_name]
        roi_data = roi_layer["data"]
        roi_affine = roi_layer["affine"]
        shape = roi_data.shape

        with self.state_manager.roi_modification(roi_name):
            roi_data.fill(0)
            center_world = center_3d.copy()
            if stored_view_type in ["axial", "coronal"]:
                center_world[0] = -center_world[0]

            self.vtk_panel.drawing_manager._rasterize_sphere_at_position(
                roi_name,
                roi_data,
                center_world,
                value,
                shape,
                stored_view_type,
                1,
            )
            self.vtk_panel.sphere_params_per_roi[roi_name]["radius"] = value

        # Remove any existing 3D preview sphere
        if roi_name in self.vtk_panel.roi_slice_actors:
            old_sphere = self.vtk_panel.roi_slice_actors[roi_name].get("sphere_3d")
            if old_sphere:
                try:
                    self.vtk_panel.scene.rm(old_sphere)
                except (ValueError, RuntimeError):
                    logger.debug("Failed to remove old sphere actor from scene.")
                self.vtk_panel.roi_slice_actors[roi_name]["sphere_3d"] = None

        # Remove yellow circle preview
        if self.vtk_panel.preview_line_actor:
            _remove_actor_from_2d_scenes(self.vtk_panel, self.vtk_panel.preview_line_actor)
            self.vtk_panel.preview_line_actor = None

        # Refresh ROI visualization (2D slices and 3D)
        self.vtk_panel.update_roi_layer(roi_name, roi_data, roi_affine)

        # Update ROI intersection for filters
        self.roi_manager.update_sphere_roi_intersection(roi_name, center_3d, value)

        self.vtk_panel.update_status(f"Sphere radius set to {value:.1f} mm")
        self.vtk_panel._render_all()

    @pyqtSlot(bool)
    def _on_skip_toggled(self, checked: bool) -> None:
        """Enables/Disables the skip feature and resets view if turned off."""
        self.skip_spinbox.setEnabled(checked)

        if checked:
            # User re-enabled skip — clear the override so auto-calc works
            self._skip_user_disabled = False
            self._on_skip_changed()
        else:
            # User explicitly disabled skip — remember the override
            self._skip_user_disabled = True
            self.render_stride = 1
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()

    @pyqtSlot()
    def _on_skip_changed(self) -> None:
        """Calculates stride from skip percentage and updates VTK."""
        # Safety: Do not update if the feature is toggled off
        if not self.skip_checkbox.isChecked():
            return

        value = self.skip_spinbox.value()

        percent_shown = 100 - value
        self.render_stride = max(1, int(100 / percent_shown))

        # Update VTK
        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()

    @batch_render(lambda self: self.vtk_panel)
    def _auto_calculate_skip_level(
        self, *, update_actor: bool = True, notify: bool = False,
    ) -> None:
        """
        Automatically sets the skip percentage based on visible streamlines.
        Target: Render approximately 20,000 streamlines for optimal performance.
        Uses visible_indices count so the stride adapts after deletions,
        undo/redo, and ROI filter changes.

        Respects user override: if the user explicitly unchecked the skip
        checkbox, auto-calculation will not re-enable it until new data
        is loaded or the user manually re-enables skip.
        """
        if not self.tractogram_data:
            return

        # Respect user's explicit decision to disable skip
        if self._skip_user_disabled:
            return

        visible_count = len(self.visible_indices)

        # Block signals to prevent VTK updates while adjusting widgets
        with signals_blocked(self.skip_checkbox, self.skip_spinbox):
            if visible_count > TARGET_RENDER_COUNT:
                # Calculate how many we want to KEEP (ratio)
                keep_ratio = TARGET_RENDER_COUNT / visible_count

                # Convert to percentage to SKIP
                # Example: 100k fibers. Target 20k. Keep 0.2. Skip 0.8 (80%)
                skip_percent = int((1.0 - keep_ratio) * 100)

                # Clamp between 0 and 99
                skip_percent = max(0, min(99, skip_percent))

                self.skip_checkbox.setChecked(True)
                self.skip_spinbox.setEnabled(True)
                self.skip_spinbox.setValue(skip_percent)

                # Update internal stride variable manually
                self.render_stride = max(1, int(100 / (100 - skip_percent)))

                if (update_actor or notify) and self.vtk_panel:
                    self.vtk_panel.update_status(
                        f"Auto-Skip: {skip_percent}% skipped for performance."
                    )
            else:
                # Bundle is small enough, show all
                self.skip_checkbox.setChecked(False)
                self.skip_spinbox.setEnabled(False)
                self.skip_spinbox.setValue(0)
                self.render_stride = 1

        # Trigger Visual Update
        if update_actor and self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()

    @pyqtSlot(int)
    def _on_opacity_slider_changed(self, value: int) -> None:
        """Updates opacity. Delegates to DataPanelManager."""
        self.data_panel_manager.on_opacity_slider_changed(value)

    def _update_initial_status(self) -> None:
        """Sets the initial status message in the VTK panel."""
        date_str = get_formatted_datetime()
        self.vtk_panel.update_status(f"Ready ({date_str}). Load data.")

    def _update_action_states(self) -> None:
        """Updates action states. Delegates to ActionsManager."""
        self.actions_manager.update_action_states()
        mapping = getattr(self, "surface_mapping_manager", None)
        if mapping is not None:
            mapping.refresh_panel()

    @pyqtSlot()
    def _trigger_calculate_centroid(self) -> None:
        """Wrapper to calculate and save centroid."""
        file_io.calculate_and_save_statistic(self, "centroid")

    @pyqtSlot()
    def _trigger_calculate_medoid(self) -> None:
        """Wrapper to calculate and save medoid."""
        file_io.calculate_and_save_statistic(self, "medoid")

    @batch_render(lambda self: self.vtk_panel)
    def _set_geometry_mode(self, as_tubes: bool) -> None:
        """Switches between Line and Tube rendering."""
        if self.render_as_tubes == as_tubes:
            return

        self.render_as_tubes = as_tubes

        if self.vtk_panel:
            self.vtk_panel.update_status(
                f"Rendering geometry set to: {'Tubes' if as_tubes else 'Lines'}"
            )
            self.vtk_panel.update_main_streamlines_actor()

    @pyqtSlot()
    def _trigger_new_roi(self) -> None:
        """Creates a new ROI. Delegates to DrawingModesManager."""
        self.drawing_modes_manager.trigger_new_roi()

    def _reset_all_drawing_modes(self) -> None:
        """Resets all drawing modes. Delegates to DrawingModesManager."""
        self.drawing_modes_manager.reset_all_drawing_modes()

    @pyqtSlot()
    def _trigger_load_odf(self) -> None:
        """Loads a NIfTI file as ODF coefficients. Delegates to ODFManager."""
        (getattr(self, "odf_manager", None) or ODFManager(self)).trigger_load_odf()

    @pyqtSlot(bool)
    def _toggle_parcellation_overlay(self, checked: bool) -> None:
        """Toggles the visibility of the 3D parcellation overlay."""
        if not checked:
            # Hide the overlay
            self.connectivity_manager.remove_parcellation_overlay()
            self._sync_parcellation_toggle_state(False)
            return

        if self.parcellation_region_actors or getattr(
            self, "_session_deferred_regions", {}
        ):
            cache_current = (
                self.connectivity_manager.is_parcellation_overlay_cache_current()
            )
            if cache_current:
                shown = self.connectivity_manager._show_parcellation_actors()
                self._sync_parcellation_toggle_state(shown is not False)
            elif self.connectivity_manager.create_parcellation_overlay():
                self._sync_parcellation_toggle_state(True)
            else:
                self._sync_parcellation_toggle_state(False)
        else:
            self._sync_parcellation_toggle_state(False)
            if self.parcellation_data is not None:
                self.vtk_panel.update_status(
                    "Use 'Calculate Intersection' to create the overlay"
                )

    def _sync_parcellation_toggle_state(self, checked: bool) -> None:
        """Syncs the parcellation toggle state between menu and data panel."""
        # Update menu action
        if self.view_parcellation_action is not None:
            with signals_blocked(self.view_parcellation_action):
                self.view_parcellation_action.setChecked(checked)

        # Update data panel checkbox (parcellation is now nested under header) - ##TODO - to refactor
        if self.data_tree_widget:
            with signals_blocked(self.data_tree_widget):
                # Search through top-level headers and their children
                for i in range(self.data_tree_widget.topLevelItemCount()):
                    header = self.data_tree_widget.topLevelItem(i)
                    # Check if this is the FreeSurfer Parcellation header
                    if header.text(0) == "FreeSurfer Parcellation":
                        # Look for the actual file item
                        for j in range(header.childCount()):
                            child = header.child(j)
                            item_data = child.data(0, Qt.ItemDataRole.UserRole)
                            if (
                                item_data
                                and isinstance(item_data, dict)
                                and item_data.get("type") == "parcellation"
                            ):
                                child.setCheckState(
                                    0,
                                    (
                                        Qt.CheckState.Checked
                                        if checked
                                        else Qt.CheckState.Unchecked
                                    ),
                                )
                                break
                        break

    def _toggle_parcellation_region(
        self, label: int, visible: bool, batch_mode: bool = False
    ) -> None:
        """Toggles visibility of an individual parcellation region."""
        # Update visibility state
        self.parcellation_region_visibility[label] = visible

        # Delegate to connectivity manager
        self.connectivity_manager.toggle_region_visibility(label, visible, batch_mode)

    def _classify_region_hemisphere(self, label_name: str) -> str:
        """Classifies a FreeSurfer region name into hemisphere."""
        return classify_region_hemisphere(label_name)

    def _clear_parcellation(self) -> None:
        """Clears all parcellation data and overlay."""
        self.connectivity_manager._reset_parcellation_state()
        self._update_action_states()
        self._update_data_panel_display()

    @pyqtSlot(bool)
    def _toggle_odf_tunnel(self, checked: bool) -> None:
        """Computes the mask and updates the VTK actor. Delegates to ODFManager."""
        (getattr(self, "odf_manager", None) or ODFManager(self)).toggle_odf_tunnel(
            checked
        )

    def _toggle_odf_tunnel_visibility(self, visible: bool) -> None:
        """Toggles ODF tunnel actor visibility. Delegates to ODFManager."""
        (
            getattr(self, "odf_manager", None) or ODFManager(self)
        ).toggle_odf_tunnel_visibility(visible)

    @pyqtSlot()
    def _remove_odf_data(self) -> None:
        """Removes ODF data and tunnel actor. Delegates to ODFManager."""
        (getattr(self, "odf_manager", None) or ODFManager(self)).remove_odf_data()

    def _update_bundle_info_display(self) -> None:
        """Update status-bar information and the visible streamline count."""
        manager = getattr(self, "data_panel_manager", None)
        if manager is not None:
            manager.tractogram_data.update_visible_count()
        if not self.data_info_label:  # Check if label exists
            return
        bundle_text = "Bundle: None"
        image_text = "Image: None"

        # Streamline Info
        if self.tractogram_data is not None:
            count = len(self.visible_indices)
            filename = (
                os.path.basename(self.original_trk_path)
                if self.original_trk_path
                else "Unknown"
            )
            file_type_info = (
                f" ({self.original_file_extension.upper()})"
                if self.original_file_extension
                else ""
            )
            scalar_info = (
                f" | Scalar: {self.active_scalar_name}"
                if self.active_scalar_name
                else ""
            )
            header = (
                self.original_trk_header if self.original_trk_header is not None else {}
            )

            dims_str, vox_str, order = "N/A", "N/A", "N/A"
            if self.tractogram_reference_grid is not None:
                dims_str = format_tuple(
                    self.tractogram_reference_grid.shape,
                    precision=0,
                )
                vox_str = format_tuple(
                    self.tractogram_reference_grid.voxel_sizes,
                    precision=2,
                )
                order = self.tractogram_reference_grid.voxel_order
            elif "dimensions" in header:
                dims_val = header["dimensions"]
                if (
                    isinstance(dims_val, (tuple, list, np.ndarray))
                    and len(dims_val) == 3
                ):
                    dims_str = format_tuple(dims_val, precision=0)
            if self.tractogram_reference_grid is None and "voxel_sizes" in header:
                vox_val = header["voxel_sizes"]
                if isinstance(vox_val, (tuple, list, np.ndarray)) and len(vox_val) == 3:
                    vox_str = format_tuple(vox_val, precision=2)
            if (
                self.tractogram_reference_grid is None
                and "voxel_order" in header
                and isinstance(header["voxel_order"], str)
            ):
                order = header["voxel_order"]

            bundle_text = (
                f"Bundle: {filename}{file_type_info} | #: {count} | Dim={dims_str} | "
                f"VoxSize={vox_str} | Order={order}{scalar_info}"
            )

        # Anatomical Image Info
        if self.anatomical_image_data is not None:
            filename = (
                os.path.basename(self.anatomical_image_path)
                if self.anatomical_image_path
                else "Unknown"
            )
            image_shape = (
                self.anatomical_reference_grid.shape
                if self.anatomical_reference_grid is not None
                else self.anatomical_image_data.shape
            )
            shape_str = format_tuple(image_shape, precision=0)
            image_text = f"Image: {filename} | Shape={shape_str}"
            image_text += self._image_volume_label()

        # ROI Info
        roi_text = ""
        if self.roi_layers:
            roi_text = f" | ROIs: {len(self.roi_layers)}"

        # Combine and Set
        separator = (
            " || "
            if self.tractogram_data is not None
            and self.anatomical_image_data is not None
            else " | "
        )
        if self.tractogram_data is None and self.anatomical_image_data is None:
            final_text = " No data loaded "
        elif self.tractogram_data is not None and self.anatomical_image_data is None:
            final_text = f" {bundle_text} "
        elif self.tractogram_data is None and self.anatomical_image_data is not None:
            final_text = f" {image_text} "
        else:
            final_text = f" {bundle_text}{separator}{image_text}{roi_text} "

        self.data_info_label.setText(final_text)

    def _update_data_panel_display(self) -> None:
        """Schedules a debounced data panel rebuild. Delegates to DataPanelManager."""
        self.data_panel_manager.update_data_panel_display()

    @pyqtSlot()
    def _perform_data_panel_update(self, *, refresh_selection: bool = False) -> None:
        """Performs the debounced data panel rebuild. Delegates to DataPanelManager."""
        self.data_panel_manager.perform_data_panel_update(
            refresh_selection=refresh_selection
        )

    def changeEvent(self, event: QEvent) -> None:
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            self._tree_icons.clear()
        super().changeEvent(event)

    def _tree_icon(self, pixmap: QStyle.StandardPixmap) -> QIcon:
        if pixmap not in self._tree_icons:
            self._tree_icons[pixmap] = self.style().standardIcon(pixmap)
        return self._tree_icons[pixmap]

    def _build_mesh_tree_items(self) -> None:
        """Build surface items using UUID identity. Delegates to DataPanelManager."""
        self.data_panel_manager.build_mesh_tree_items()

    def _image_volume_label(self) -> str:
        """Return the 4D volume/RGB suffix for the loaded anatomical image."""
        return image_volume_label(self)

    # Undo/Redo Core Logic
    @pyqtSlot()
    def _perform_undo(self) -> None:
        """Performs undo. Delegates to StateManager."""
        self.state_manager.perform_undo()

    @pyqtSlot()
    def _perform_redo(self) -> None:
        """Performs redo. Delegates to StateManager."""
        self.state_manager.perform_redo()

    # Command Actions Logic
    @pyqtSlot()
    def _perform_reset_camera(self) -> None:
        """Resets camera. Delegates to StateManager."""
        self.state_manager.perform_reset_camera()

    # View Action Logic
    @pyqtSlot(object)
    def _set_color_mode(self, mode: ColorMode) -> None:
        """Sets color mode. Delegates to StateManager."""
        self.state_manager.set_color_mode(mode)

    # GUI Action Methods
    @batch_render(lambda self: self.vtk_panel)
    def _close_bundle(self, keep_image: bool = False) -> None:
        """
        Closes the current streamline bundle.
        Args:
            keep_image: If True, the anatomical image is NOT removed.
        """
        try:
            if not self.tractogram_data:
                if self.vtk_panel:
                    self.vtk_panel.update_status("No bundle open to close.")
                return

            self.surface_mapping_manager.invalidate()
            if self.vtk_panel:
                msg = (
                    "Closing bundle..."
                    if keep_image
                    else "Closing bundle (also clears image)..."
                )
                self.vtk_panel.update_progress_status(msg)
                QApplication.processEvents()

                # Remove/hide streamline-related actors
                self.vtk_panel.update_radius_actor(visible=False)
                self.selected_streamline_indices = set()
                self.vtk_panel.update_highlight()
                self.vtk_panel.remove_odf_actor()

                # Clear inversion mode so it does not bleed into the next session
                if self._inversion_active:
                    self._inversion_active = False
                    self._inversion_keeper_indices = set()
                    self.vtk_panel.clear_invert_contour()

                # Clear anatomical slices if present AND NOT keep_image
                if not keep_image and self.anatomical_image_data is not None:
                    self._trigger_clear_anatomical_image(notify=False)

            # Reset streamline data state
            self.tractogram_data = None
            self._tractogram_data_version += 1
            self.streamline_bboxes = None
            self.manual_visible_indices = set()
            self.visible_indices = set()
            self._visibility_version += 1
            self.original_trk_header = None
            self.original_trk_affine = None
            self.tractogram_reference_grid = None
            self.original_trk_path = None
            self.original_file_extension = None

            # Close TRX memmap file reference (releases temp directory)
            self._close_trx_file()

            self.scalar_data_per_point = None
            self.data_per_streamline = None
            self.active_scalar_name = None
            self.scalar_display = None
            (getattr(self, "odf_manager", None) or ODFManager(self)).reset_odf_state()
            retain_polyline_history(self)
            self.current_color_mode = ColorMode.ORIENTATION
            self.color_default_action.setChecked(True)
            self.scalar_range_initialized = False
            if self.scalar_toolbar:
                self.scalar_toolbar.setVisible(False)

            self.connectivity_manager.invalidate_parcellation_cache(
                clear_region_states=True,
                render=False,
            )

            # Update VTK
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()  # Should remove streamline actor
                status_msg = (
                    "Bundle closed."
                    if keep_image
                    else "Bundle closed (Image also cleared)."
                )
                self.vtk_panel.update_status(status_msg)

            # Reset Geometry to Lines default
            self.render_as_tubes = False
            self.geo_lines_action.setChecked(True)

            # Update UI
            self._update_bundle_info_display()
            self._update_action_states()
            self._update_data_panel_display()  # Refresh data panel tree widget

        except Exception as e:
            logger.error(f"Error in _close_bundle: {e}", exc_info=True)
            # Ensure critical state is cleared even if error occurs
            self.tractogram_data = None
            self._update_action_states()

    def load_initial_files(
        self,
        bundle_path: Optional[str] = None,
        anat_path: Optional[str] = None,
        roi_paths: Optional[List[str]] = None,
        roi_in: Optional[List[List[float]]] = None,
        radius: Optional[List[float]] = None,
    ) -> None:
        """Loads files specified via command line arguments."""
        try:
            if bundle_path and str(bundle_path).lower().endswith(SESSION_EXTENSION):
                self._open_session(bundle_path)
                return
            if anat_path:
                if os.path.exists(anat_path):
                    logger.info(f"Loading initial anatomical image: {anat_path}")
                    # Clear existing ROIs/Image if any
                    self._trigger_clear_anatomical_image(notify=False)

                    self._load_initial_anat_threaded(
                        anat_path, bundle_path, roi_paths, roi_in, radius
                    )
                    return  # Remaining loads are chained via on_finished
                else:
                    logger.error(f"Anatomical image path not found: {anat_path}")

            if bundle_path and str(bundle_path).lower().endswith(".obj"):
                self.mesh_manager.load(bundle_path)
            elif bundle_path:
                if os.path.exists(bundle_path):
                    logger.info(f"Loading initial bundle: {bundle_path}")
                    self.scalar_range_initialized = False
                    if self.scalar_toolbar:
                        self.scalar_toolbar.setVisible(False)
                    self.bundle_is_visible = True

                    file_io.load_streamlines_file(self, file_path=bundle_path)
                else:
                    logger.error(f"Bundle path not found: {bundle_path}")

            if roi_paths:
                (
                    getattr(self, "image_loader_manager", None)
                    or ImageLoaderManager(self)
                ).load_roi_files(roi_paths)

            # Handle --roi (Create Sphere ROI)
            if roi_in and self.anatomical_image_data is not None:
                logger.info(f"Creating {len(roi_in)} Sphere ROIs from CLI arguments.")

                # Ensure radius is a list and matches length
                if radius is None:
                    radius_list = [5.0] * len(roi_in)
                else:
                    radius_list = radius
                    if len(radius_list) < len(roi_in):
                        last_r = radius_list[-1] if radius_list else 5.0
                        radius_list.extend([last_r] * (len(roi_in) - len(radius_list)))

                for i, coords in enumerate(roi_in):
                    r_val = radius_list[i]

                    # Validate coordinates and radius
                    center_world = np.array(coords)
                    if not np.all(np.isfinite(center_world)):
                        logger.error(
                            f"Invalid ROI coordinates (non-finite values): {coords}"
                        )
                        continue
                    if not np.isfinite(r_val) or r_val <= 0:
                        logger.error(f"Invalid ROI radius: {r_val}")
                        continue

                    logger.info(
                        f"Creating Sphere ROI {i+1}/{len(roi_in)} at {coords} with radius {r_val}"
                    )

                    # Create New ROI
                    self._trigger_new_roi()

                    # Get the newly created ROI name
                    roi_name = self.current_drawing_roi
                    if roi_name:
                        # We need to manually trigger the sphere drawing logic in VTKPanel
                        roi_layer = self.roi_layers[roi_name]
                        roi_data = roi_layer["data"]
                        roi_affine = roi_layer["affine"]
                        roi_inv_affine = roi_layer["inv_affine"]
                        shape = roi_data.shape

                        # Transform center to voxel space
                        p_h = np.append(center_world, 1.0)
                        center_vox = np.dot(roi_inv_affine, p_h)[:3]

                        _rasterize_world_sphere(
                            roi_data,
                            center_vox,
                            r_val,
                            roi_affine,
                            shape,
                            1,
                        )

                        if self.vtk_panel:
                            # Continuous geometry remains valid even when no voxel
                            # center falls inside the sphere. Actors need it first.
                            self.vtk_panel.sphere_params_per_roi[roi_name] = {
                                "center": center_world,
                                "radius": r_val,
                                "view_type": "axial",
                            }
                            self.vtk_panel.add_roi_layer(roi_name, roi_data, roi_affine)
                            self.vtk_panel.update_status(
                                f"Created Sphere ROI at {coords} (r={r_val}mm)"
                            )

            self._update_action_states()
            self._update_data_panel_display()
        except Exception as e:
            logger.error(f"Error in load_initial_files: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Startup Error", f"Error loading initial files:\n{e}"
            )

    def _load_initial_anat_threaded(
        self,
        anat_path: str,
        bundle_path: Optional[str] = None,
        roi_paths: Optional[List[str]] = None,
        roi_in: Optional[List[List[float]]] = None,
        radius: Optional[List[float]] = None,
        *,
        anatomical_display: Optional[dict] = None,
    ) -> None:
        """Loads an anatomical image on a background thread. Delegates to ImageLoaderManager."""
        (
            getattr(self, "image_loader_manager", None) or ImageLoaderManager(self)
        ).load_anatomical_file(
            anat_path,
            initial_load=True,
            bundle_path=bundle_path,
            roi_paths=roi_paths,
            roi_in=roi_in,
            radius=radius,
            anatomical_display=anatomical_display,
        )

    # Action Trigger Wrappers
    @pyqtSlot()
    def _trigger_clear_anatomical_image(self, notify: bool = True) -> None:
        """Clears the currently loaded anatomical image. Delegates to ImageLoaderManager."""
        (
            getattr(self, "image_loader_manager", None) or ImageLoaderManager(self)
        ).clear_anatomical_image(notify=notify)

    @pyqtSlot()
    def _open_session(self, path: Optional[str] = None) -> None:
        from .logic.session_manager import SessionManager

        SessionManager(self).open(path)

    def _save_session(self) -> None:
        from .logic.session_manager import SessionManager

        SessionManager(self).save()

    def _trigger_load_mesh(self) -> None:
        parent_id = None
        items = self.data_tree_widget.selectedItems()
        if len(items) == 1:
            data = items[0].data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, dict) and data.get("type") == "mesh":
                parent_id = data.get("id")
        self.mesh_manager.trigger_load(parent_id=parent_id)

    def _trigger_load_streamlines(self) -> None:
        """Wrapper to call the streamline load function from file_io."""
        file_io.load_streamlines_file(self)

    @pyqtSlot()
    def _trigger_save_streamlines(self) -> None:
        """Wrapper to call the streamline save function from file_io."""
        file_io.save_streamlines_file(self)

    @pyqtSlot()
    def _trigger_save_density_map(self) -> None:
        """
        Calculates and saves a Track Density Imaging (TDI) map of the currently
        visible streamlines. Uses the anatomical image grid if available for
        maximum accuracy/alignment, otherwise derives a grid from the tractogram.
        """
        if not self.tractogram_data:
            return

        # Determine Target Grid
        affine = None
        shape = None
        reference_grid = self.anatomical_reference_grid
        if reference_grid is None:
            reference_grid = self.tractogram_reference_grid

        # Priority A: Loaded Anatomical Image
        if reference_grid is not None:
            affine = reference_grid.affine
            shape = reference_grid.shape

        elif self.anatomical_image_data is not None:
            reference_grid = ReferenceGrid(
                affine=self.anatomical_image_affine,
                shape=self.anatomical_image_data.shape[:3],
                provenance="anatomical-preview-fallback",
            )
            affine = reference_grid.affine
            shape = reference_grid.shape

        # Priority B: Original Header Info (if compatible/available)
        elif self.original_trk_header:
            reference_grid = ReferenceGrid.from_header(
                self.original_trk_header,
                provenance="tractogram-header",
            )
            if reference_grid is not None:
                affine = reference_grid.affine
                shape = reference_grid.shape

        def visible_points():
            streamlines = [
                sl for i in self.visible_indices
                if (sl := self.tractogram_data[i]) is not None and len(sl)
            ]
            return np.concatenate(streamlines, axis=0) if streamlines else None

        points = None
        # Priority C: Compute Bounding Box (Fallback)
        # If no reference is found, we create a 1mm isotropic grid around the bundle
        if affine is None or shape is None:
            self.vtk_panel.update_status("Calculating density grid from bounds...")
            QApplication.processEvents()

            try:
                points = visible_points()
                if points is None:
                    QMessageBox.warning(self, "Error", "No visible streamlines to map.")
                    return
                reference_grid = density_grid_from_points(
                    points, provenance="synthetic:tdi-bounds"
                )

            except (ValueError, IndexError, TypeError) as e:
                logger.error(f"Error computing bounds: {e}")
                self.vtk_panel.update_status("Error computing density bounds.")
                return

        default_filename = "density_map.nii.gz"
        if self.original_trk_path:
            base_name = os.path.splitext(os.path.basename(self.original_trk_path))[0]
            default_filename = f"{base_name}_density_map.nii.gz"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Density Map", default_filename, "NIfTI Files (*.nii.gz *.nii)"
        )
        if not file_path:
            return

        self.vtk_panel.update_status("Computing density map...")
        self.vtk_panel.update_progress_bar(0, 0, visible=True)
        QApplication.processEvents()

        try:
            # Compute Density
            # Retrieve only visible streamlines
            if points is None:
                points = visible_points()
                if points is None:
                    raise ValueError("No valid streamlines found in current view.")

            nifti_img, _ = density_image(points, reference_grid)
            transactional_save(file_path, lambda path: nib.save(nifti_img, path))

            self.vtk_panel.update_status(
                f"Saved density map: {os.path.basename(file_path)}"
            )

        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Error saving density map: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Could not save density map:\n{e}")
            self.vtk_panel.update_status("Error saving density map.")
        finally:
            self.vtk_panel.update_progress_bar(0, 0, visible=False)

    @pyqtSlot()
    def _trigger_screenshot(self) -> None:
        """Wrapper to call the screenshot function in vtk_panel."""
        if (
            self.tractogram_data is None
            and self.anatomical_image_data is None
            and not self.mesh_layers
            and not self.standalone_polylines
        ):
            QMessageBox.warning(
                self, "Screenshot Error", "No data loaded to take a screenshot of."
            )
            return
        if self.vtk_panel:
            try:
                self.vtk_panel.take_screenshot()
            except AttributeError:
                QMessageBox.warning(
                    self, "Error", "Screenshot function not available in VTK panel."
                )
            except (RuntimeError, ValueError, OSError) as e:
                QMessageBox.critical(
                    self, "Screenshot Error", f"Could not take screenshot:\n{e}"
                )
        else:
            QMessageBox.warning(self, "Screenshot Error", "VTK panel not initialized.")

    @pyqtSlot()
    def _trigger_export_html(self) -> None:
        """Exports the current visualization to an interactive HTML file."""
        if self.tractogram_data is None and self.anatomical_image_data is None:
            message = "No data loaded to export."
            if self.standalone_polylines:
                message += " Polyline annotations are excluded from HTML export."
            QMessageBox.warning(self, "Export Error", message)
            return

        # Determine default filename
        default_name = "visualization.html"
        if self.original_trk_path:
            base_name = os.path.splitext(os.path.basename(self.original_trk_path))[0]
            default_name = f"{base_name}_viewer.html"

        # Get save path from user
        title = "Export to HTML"
        if any(layer["visible"] for layer in self.mesh_layers.values()):
            title += " — visible surface meshes will be excluded"
        if self.standalone_polylines or any(
            layer.get("polylines") for layer in self.mesh_layers.values()
        ):
            title += " — polyline annotations will be excluded"
        file_path, _ = QFileDialog.getSaveFileName(
            self, title, default_name, "HTML Files (*.html)"
        )

        if not file_path:
            if self.vtk_panel:
                self.vtk_panel.update_status("HTML export cancelled.")
            return

        # Ensure .html extension
        if not file_path.lower().endswith(".html"):
            file_path += ".html"

        if self.vtk_panel:
            self.vtk_panel.update_status("Exporting to HTML...")
        QApplication.processEvents()

        try:
            from .visualization.html_export import export_to_html

            success = export_to_html(self, file_path)

            if success:
                if self.vtk_panel:
                    self.vtk_panel.update_status(
                        f"Exported: {os.path.basename(file_path)}"
                    )
            else:
                if self.vtk_panel:
                    self.vtk_panel.update_status("HTML export failed.")
                QMessageBox.warning(
                    self,
                    "Export Error",
                    "Failed to export HTML. Check console for details.",
                )

        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"HTML export error: {e}", exc_info=True)
            if self.vtk_panel:
                self.vtk_panel.update_status("HTML export error.")
            QMessageBox.critical(
                self, "Export Error", f"Could not export to HTML:\n{e}"
            )

    # Background Image Methods
    @pyqtSlot()
    def _trigger_load_anatomical_image(self) -> None:
        """Triggers loading of an anatomical image. Delegates to ImageLoaderManager."""
        (
            getattr(self, "image_loader_manager", None) or ImageLoaderManager(self)
        ).trigger_load_anatomical_image()

    # ROI Image Methods
    @pyqtSlot()
    def _trigger_load_roi(self) -> None:
        """Triggers loading of ROI image layer(s). Delegates to ImageLoaderManager."""
        (
            getattr(self, "image_loader_manager", None) or ImageLoaderManager(self)
        ).trigger_load_roi()

    @pyqtSlot()
    def _trigger_clear_all_rois(self, notify: bool = True) -> None:
        """Clears all loaded ROI image layers. Delegates to ImageLoaderManager."""
        (
            getattr(self, "image_loader_manager", None) or ImageLoaderManager(self)
        ).clear_all_rois(notify=notify)

    @pyqtSlot()
    @batch_render(lambda self: self.vtk_panel)
    def _trigger_clear_all_data(self) -> None:
        """Clears all loaded data (streamlines, anatomical image, ROIs, parcellation) without confirmation."""
        if self._session_busy or self._mesh_commit_active:
            return
        self.mesh_polyline_manager.cancel_io(all_targets=True)
        self.unified_undo_stack.clear()
        self.unified_redo_stack.clear()
        has_data = (
            self.tractogram_data is not None
            or self.anatomical_image_data is not None
            or bool(self.roi_layers)
            or self.parcellation_data is not None
            or bool(self.mesh_layers)
            or bool(self.standalone_polylines)
            or self._mesh_loader_thread is not None
            or self._polyline_io_thread is not None
        )

        if not has_data:
            self._update_action_states()
            return

        if self.mesh_layers or self._mesh_loader_thread is not None:
            self.mesh_manager.clear()
        if self.standalone_polylines:
            self.mesh_polyline_manager.clear_standalone()

        if self.tractogram_data is not None:
            self._close_bundle()

        # Clear ROIs
        if self.roi_layers:
            self._trigger_clear_all_rois(notify=False)

        # Clear Parcellation overlay and data
        if self.parcellation_data is not None:
            self._clear_parcellation()

        # Clear Image (if not already cleared by _close_bundle or if no bundle was loaded)
        if self.anatomical_image_data is not None:
            self._trigger_clear_anatomical_image()

        # Reset all drawing modes to prevent stuck state after clearing
        self._reset_all_drawing_modes()

        # Final UI update
        self._update_data_panel_display()
        self.vtk_panel.update_status("All data cleared.")
        self._update_action_states()

    def _set_roi_logic_mode(self, roi_path: str, mode: str) -> None:
        """Sets ROI logic mode. Delegates to ROIManager."""
        self.roi_manager.set_roi_logic_mode(roi_path, mode)

    @pyqtSlot(bool)
    @batch_render(lambda self: self.vtk_panel)
    def _toggle_image_visibility(self, visible: bool) -> None:
        """Toggles the visibility of the anatomical image slices."""
        if self.image_is_visible == visible:
            return

        self.image_is_visible = visible
        if self.vtk_panel:
            self.vtk_panel.set_anatomical_slice_visibility(visible)
            self.vtk_panel.update_status(f"Image visibility set to {visible}")

    def _rename_roi_action(self, old_path: str) -> None:
        """Renames ROI. Delegates to ROIManager."""
        self.roi_manager.rename_roi_action(old_path)

    def _remove_roi_layer_action(self, path: str) -> None:
        """Removes ROI layer. Delegates to ROIManager."""
        self.roi_manager.remove_roi_layer_action(path)

    @pyqtSlot(bool)
    def _toggle_bundle_visibility(self, visible: bool) -> None:
        """Toggles the visibility of the streamline bundle actors."""
        if self.bundle_is_visible == visible:
            return

        self.bundle_is_visible = visible

        if self.vtk_panel:
            visibility_flag = 1 if visible else 0

            # Toggle main actor
            if self.vtk_panel.streamlines_actor:
                self.vtk_panel.streamlines_actor.SetVisibility(visibility_flag)

            # Toggle highlight actor
            if self.vtk_panel.highlight_actor:
                self.vtk_panel.highlight_actor.SetVisibility(visibility_flag)

            # Also hide radius sphere if bundle is hidden
            if self.vtk_panel.radius_actor and not visible:
                self.vtk_panel.radius_actor.SetVisibility(0)

            self.vtk_panel.update_status(f"Bundle visibility set to {visible}", render=False)
            if self.vtk_panel.render_window:
                request_render(self.vtk_panel, self.vtk_panel.render_window)

    @pyqtSlot(str, bool)
    @batch_render(lambda self: self.vtk_panel)
    def _toggle_roi_visibility(self, path: str, visible: bool) -> None:
        """Toggles the visibility of a specific ROI layer."""
        if self.roi_visibility.get(path, True) == visible:
            return  # No change

        self.roi_visibility[path] = visible

        if self.vtk_panel:
            self.vtk_panel.set_roi_layer_visibility(path, visible)
            self.vtk_panel.update_status(
                f"ROI '{os.path.basename(path)}' visibility set to {visible}"
            )

        self._update_bundle_info_display()
        self._update_action_states()

    @pyqtSlot()
    def _reset_scalar_range(self) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.reset_scalar_range()

    @pyqtSlot()
    def _on_ras_coordinate_entered(self) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.on_ras_coordinate_entered()

    # Window Close Event
    def closeEvent(self, event: QCloseEvent) -> None:
        """Handles the main window close event, prompting if data is loaded."""
        if getattr(self, "_session_busy", False):
            event.ignore()
            return
        logger.info("Close event received.")
        # Explicitly check for None to avoid ValueError with numpy arrays
        data_loaded = (self.tractogram_data is not None) or (
            self.anatomical_image_data is not None
        ) or bool(self.mesh_layers) or bool(self.standalone_polylines)
        prompt_message = "Data is currently loaded.\nAre you sure you want to quit?"

        should_exit = False

        if self._shutdown_requested:
            should_exit = True
        elif data_loaded:
            # Custom message box to enforce "Yes" on the Left and "No" on the Right
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Confirm Quit")
            msg_box.setText(prompt_message)
            msg_box.setIcon(QMessageBox.Icon.Question)

            # Use ActionRole to prevent platform-specific reordering
            yes_btn = msg_box.addButton("Yes", QMessageBox.ButtonRole.ActionRole)
            no_btn = msg_box.addButton("No", QMessageBox.ButtonRole.ActionRole)

            msg_box.setDefaultButton(no_btn)
            msg_box.exec()

            if msg_box.clickedButton() == yes_btn:
                logger.info("User confirmed quit. Cleaning up...")
                should_exit = True
            else:
                logger.info("User cancelled quit.")
                event.ignore()
        else:
            logger.info("No data loaded. Cleaning up...")
            should_exit = True

        if should_exit:
            self._shutdown_requested = True
            self.update_notifier.close()
            self.appimage_integration.close()
            workers_stopped = self._cancel_background_workers(wait_ms=0)
            if not workers_stopped or self.appimage_integration.working:
                event.ignore()
                if self.vtk_panel:
                    self.vtk_panel.update_status(
                        "Waiting for background operations to stop..."
                    )
                QTimer.singleShot(50, self.close)
                return
            self._cleanup_resources()
            self._cleanup_vtk()
            event.accept()

        # On Linux and macOS, force immediate exit to prevent VTK/Qt cleanup
        # conflicts that cause segmentation faults during Py_FinalizeEx.
        if should_exit:
            import sys

            if sys.platform.startswith("linux") or sys.platform == "darwin":
                import os

                logger.info("Exiting application (Unix workaround)...")
                os._exit(0)

    def _close_trx_file(self) -> None:
        """Close the TRX memory-mapped file reference and release resources."""
        if self.trx_file_reference is not None:
            file_io._retire_trx_owner(self, self.trx_file_reference)
            self.trx_file_reference = None

    def _cleanup_resources(self) -> None:
        """Cleans up non-VTK resources before application exit."""
        self.update_notifier.close()
        self.appimage_integration.close()
        self._cancel_background_workers(wait_ms=0)
        self.mesh_polyline_manager.close()
        self.surface_mapping_manager.close()
        self.mesh_manager.close_textures()
        self.mesh_manager.cancel_overlays()
        if self._mesh_overlay_thread is not None:
            self._mesh_overlay_thread.wait()
            self._mesh_overlay_thread.discard_pending_result()
            self._mesh_overlay_thread = None

        self._mesh_load_generation += 1
        if self._mesh_loader_thread is not None:
            self._mesh_loader_thread.cancel()
            self._mesh_loader_thread.wait()
            self._mesh_loader_thread.discard_pending_result()
            self._mesh_loader_thread = None

        # Clear memory-mapped image cache
        if self.anatomical_mmap_image is not None:
            try:
                self.anatomical_mmap_image.clear_cache()
            except (AttributeError, OSError):
                logger.debug("Failed to clear memory-mapped image cache.")
            self.anatomical_mmap_image = None

        # Close TRX memmap file reference (releases temp directory)
        self._close_trx_file()
        file_io._release_deferred_trx_owners(self)
        self._background_workers.clear()
        self._loader_thread = None
        self._image_loader_thread = None
        self._medoid_thread = None

        # Clear large data arrays to help garbage collection
        self.tractogram_data = None
        self.streamline_bboxes = None
        self.anatomical_image_data = None
        self.odf_data = None
        self.parcellation_data = None
        self._parcellation_label_count = None
        self.mesh_layers.clear()

    def _cancel_background_workers(self, wait_ms: int = 0) -> bool:
        """Request cancellation and report whether every worker has stopped."""
        workers = list(getattr(self, "_background_workers", ()))
        for attr in (
            "_loader_thread",
            "_image_loader_thread",
            "_medoid_thread",
            "_mesh_loader_thread",
            "_mesh_texture_thread",
            "_mesh_overlay_thread",
            "_polyline_io_thread",
            "_polyline_preview_thread",
        ):
            worker = getattr(self, attr, None)
            if worker is not None and worker not in workers:
                workers.append(worker)

        all_stopped = True
        for worker in workers:
            try:
                cancel = getattr(worker, "cancel", None)
                if callable(cancel):
                    cancel()
                if worker.isRunning():
                    worker.wait(wait_ms)
                if worker.isRunning() or getattr(worker, "is_consuming_result", False):
                    all_stopped = False
                elif self._shutdown_requested:
                    discard = getattr(worker, "discard_pending_result", None)
                    if callable(discard):
                        discard()
            except (RuntimeError, AttributeError) as exc:
                logger.warning("Error stopping background worker: %s", exc)
        return all_stopped

    def _cleanup_vtk(self) -> None:
        """Safely cleans up VTK resources for all 4 views."""
        if self.vtk_panel:
            self.vtk_panel.cleanup()

    # Theme switching methods
    @pyqtSlot()
    def _set_theme_light(self) -> None:
        """Sets the application to light theme."""
        self.theme_manager.set_theme(ThemeMode.LIGHT)
        if self.vtk_panel:
            self.vtk_panel.update_status("Theme changed to Light")

    @pyqtSlot()
    def _set_theme_dark(self) -> None:
        """Sets the application to dark theme."""
        self.theme_manager.set_theme(ThemeMode.DARK)
        if self.vtk_panel:
            self.vtk_panel.update_status("Theme changed to Dark")

    @pyqtSlot()
    def _set_theme_system(self) -> None:
        """Sets the application to follow system theme."""
        self.theme_manager.set_theme(ThemeMode.SYSTEM)
        if self.vtk_panel:
            self.vtk_panel.update_status("Theme changed to System")

    @pyqtSlot(bool)
    def _toggle_auto_fill(self, checked: bool) -> None:
        """Toggles the ROI auto-fill setting."""
        self.auto_fill_voxels = checked
        self._save_settings()
        status = "Enabled" if checked else "Disabled"
        if self.vtk_panel:
            self.vtk_panel.update_status(f"ROI Auto-fill {status}")

    def _load_settings(self) -> None:
        """Loads persistent application settings."""
        # Load Auto-fill setting (default False)
        self.auto_fill_voxels = self.settings.value(
            "drawing/auto_fill", False, type=bool
        )

        # Sync UI action if it exists
        if self.auto_fill_action is not None:
            self.auto_fill_action.setChecked(self.auto_fill_voxels)

    def _save_settings(self) -> None:
        """Saves persistent application settings."""
        self.settings.setValue("drawing/auto_fill", self.auto_fill_voxels)

    # Help-About dialog
    @pyqtSlot()
    def _show_about_dialog(self) -> None:
        """Displays the About TractEdit information box with the application logo."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About TractEdit")

        # Load and Set Logo
        try:
            logo_path = get_asset_path("logo.png")
            pixmap = QPixmap(logo_path)

            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    150,
                    150,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                msg_box.setIconPixmap(scaled_pixmap)
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Could not load logo for About dialog: {e}")

        from tractedit_pkg import __version__ as _app_version

        about_text = f"""<b>TractEdit version {_app_version}</b><br><br>
        Author: Marco Tagliaferri, PhD Candidate in Neuroscience<br>
        Center for Mind/Brain Sciences (CIMeC)
        University of Trento, Italy
        <br><br>
        Contacts:<br>
        marco.tagliaferri@unitn.it<br>
        marco.tagliaferri93@gmail.com
        <br><br>
        <b>Research Use Only.</b> TractEdit is not a medical device and has not
        received regulatory clearance (FDA / CE). It must not be used for
        diagnosis, treatment planning, or clinical decision-making.
        <br><br>
        Licensed under the MIT License.
        """
        msg_box.setText(about_text)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
