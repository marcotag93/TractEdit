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
from typing import Optional, List, Set, Dict, Any, Tuple
import nibabel as nib
import logging

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QMenuBar,
    QFileDialog,
    QMessageBox,
    QLabel,
    QStatusBar,
    QApplication,
    QToolBar,
    QDoubleSpinBox,
    QSpinBox,
    QSlider,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QDockWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QStyle,
    QLineEdit,
    QMenu,
    QColorDialog,
    QCheckBox,
    QInputDialog,
    QToolButton,
    QProgressDialog,
)
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
    QActionGroup,
    QIcon,
    QCloseEvent,
    QPixmap,
    QPainter,
    QBrush,
    QColor,
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QSettings

from . import file_io
from . import odf_utils
from .utils import (
    ColorMode,
    get_formatted_datetime,
    get_asset_path,
    format_tuple,
    signals_blocked,
    MAX_STACK_LEVELS,
    DEFAULT_SELECTION_RADIUS,
    MIN_SELECTION_RADIUS,
    RADIUS_INCREMENT,
    SLIDER_PRECISION,
    ROI_COLORS,
    TARGET_RENDER_COUNT,
)
from .visualization import VTKPanel
from .ui import (
    ActionsManager,
    ToolbarsManager,
    DataPanelManager,
    DrawingModesManager,
    ThemeManager,
    ThemeMode,
)
from .logic import ROIManager, StateManager, ScalarManager, ConnectivityManager
from nibabel.processing import resample_from_to
from nibabel.orientations import ornt_transform, apply_orientation, io_orientation

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
        super().__init__(parent)

        # Initialize Streamline Data Variables
        self.tractogram_data: Optional["nib.streamlines.ArraySequence"] = None
        self.streamline_bboxes: Optional[np.ndarray] = None
        self.visible_indices: Set[int] = set()
        self.original_trk_header: Optional[Dict[str, Any]] = (
            None  # Header dict from loaded file
        )
        self.original_trk_affine: Optional[np.ndarray] = (
            None  # Affine matrix (affine_to_rasmm)
        )
        self.original_trk_path: Optional[str] = None  # Full path
        self.original_file_extension: Optional[str] = (
            None  # '.trk', '.tck', '.trx', or None
        )
        self.trx_file_reference: Optional[Any] = None  # TRX memmap object reference
        self.scalar_data_per_point: Optional[
            Dict[str, "nib.streamlines.ArraySequence"]
        ] = None  # Dictionary: {scalar_name: [scalar_array_sl0, ...]}
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

        # Initialize Anatomical Image Data Variables
        self.anatomical_image_path: Optional[str] = None
        self.anatomical_image_data: Optional[np.ndarray] = None  # Numpy array
        self.anatomical_image_affine: Optional[np.ndarray] = None  # 4x4 numpy array
        self.anatomical_mmap_image: Optional["file_io.MemoryMappedImage"] = None

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
        self.odf_tunnel_sphere = None       # Lower-res sphere for tunnel display
        self.odf_tunnel_basis = None         # SH basis for tunnel display sphere
        self.odf_tunnel_is_visible: bool = False
        self.MAX_ODF_STREAMLINES = 26000  # Safety limit for Tunnel View

        # Parcellation / Connectivity Data
        self.parcellation_data: Optional[np.ndarray] = None
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
        self.roi_layers: Dict[str, Dict[str, Any]] = (
            {}
        )  # Key: path, Val: {'data':, 'affine':, 'inv_affine':}

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
        self.parcellation_overlay_actor: Optional[Any] = None
        self.parcellation_connected_labels: Set[int] = set()
        self.parcellation_region_visibility: Dict[int, bool] = {}
        self.parcellation_main_labels: Set[int] = set()
        self.parcellation_label_colors: Dict[int, Tuple[float, ...]] = {}
        self.parcellation_region_actors: Dict[int, Any] = {}
        self.parcellation_region_states: Dict[int, Dict[str, bool]] = {}

        # Parcellation filter data (also reset in _clear_parcellation)
        self.parcellation_region_intersection_cache: Dict[str, Set[int]] = {}
        self.parcellation_start_labels: Optional[np.ndarray] = None
        self.parcellation_end_labels: Optional[np.ndarray] = None
        self.parcellation_visible_indices: Optional[Set[int]] = None

        # Pending / Cache
        self._pending_expanded_items: Set[str] = set()

        # Actions — File menu
        self.load_file_action: Optional[QAction] = None
        self.replace_bundle_action: Optional[QAction] = None
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

        # UI Managers
        self.actions_manager = ActionsManager(self)
        self.toolbars_manager = ToolbarsManager(self)
        self.data_panel_manager = DataPanelManager(self)
        self.drawing_modes_manager = DrawingModesManager(self)
        self.theme_manager = ThemeManager(self)

        # Logic Managers
        self.roi_manager = ROIManager(self)
        self.state_manager = StateManager(self)
        self.scalar_manager = ScalarManager(self)
        self.connectivity_manager = ConnectivityManager(self)

        # Setup UI Components
        self.actions_manager.create_actions()
        self.data_panel_manager.create_data_panel()  # Must be before create_menus for dock toggle
        self.actions_manager.create_menus()
        self.toolbars_manager.create_main_toolbar()
        self.toolbars_manager.create_scalar_toolbar()
        self.toolbars_manager.setup_status_bar()
        self.toolbars_manager.setup_central_widget()  # This creates the VTKPanel

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
            self.vtk_panel.update_status(
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

        # Save undo state
        self._save_roi_state_for_undo(roi_name)

        # Get ROI layer data
        roi_layer = self.roi_layers[roi_name]
        roi_data = roi_layer["data"]
        roi_affine = roi_layer["affine"]
        roi_inv_affine = roi_layer["inv_affine"]
        shape = roi_data.shape

        # Clear existing ROI data
        roi_data.fill(0)

        # Convert center from world to voxel coordinates
        center_world = center_3d.copy()
        if stored_view_type in ["axial", "coronal"]:
            center_world[0] = -center_world[0]

        p_h = np.append(center_world, 1.0)
        center_vox = np.dot(roi_inv_affine, p_h)[:3]

        # Calculate radius in voxels (approximate using affine scaling)
        voxel_sizes = np.abs(np.diag(roi_affine)[:3])
        radius_vox = value / np.mean(voxel_sizes)

        # Rasterize sphere
        min_vox = np.floor(center_vox - radius_vox).astype(int)
        max_vox = np.ceil(center_vox + radius_vox).astype(int)
        min_vox = np.maximum(min_vox, 0)
        max_vox = np.minimum(max_vox, np.array(shape) - 1)

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
            roi_data[
                min_vox[0] : max_vox[0] + 1,
                min_vox[1] : max_vox[1] + 1,
                min_vox[2] : max_vox[2] + 1,
            ][mask] = 1

        # Update stored radius
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
            for scene in [
                self.vtk_panel.axial_scene,
                self.vtk_panel.coronal_scene,
                self.vtk_panel.sagittal_scene,
            ]:
                if scene:
                    try:
                        scene.rm(self.vtk_panel.preview_line_actor)
                    except (ValueError, RuntimeError):
                        logger.debug("Failed to remove preview actor from scene.")
            self.vtk_panel.preview_line_actor = None

        # Refresh ROI visualization (2D slices and 3D)
        self.vtk_panel.update_roi_layer(roi_name, roi_data, roi_affine)

        # Update ROI intersection for filters
        self.update_sphere_roi_intersection(roi_name, center_3d, value)

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

    def _auto_calculate_skip_level(self) -> None:
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

                if self.vtk_panel:
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
        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()

    @pyqtSlot()
    def _on_data_item_selected(self) -> None:
        """Updates opacity slider. Delegates to DataPanelManager."""
        self.data_panel_manager.on_data_item_selected()

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

    @pyqtSlot()
    def _trigger_calculate_centroid(self) -> None:
        """Wrapper to calculate and save centroid."""
        file_io.calculate_and_save_statistic(self, "centroid")

    @pyqtSlot()
    def _trigger_calculate_medoid(self) -> None:
        """Wrapper to calculate and save medoid."""
        file_io.calculate_and_save_statistic(self, "medoid")

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

    @pyqtSlot(bool)
    def _toggle_draw_mode(self, checked: bool) -> None:
        """Toggles draw mode. Delegates to DrawingModesManager."""
        self.drawing_modes_manager.toggle_draw_mode(checked)

    @pyqtSlot(bool)
    def _toggle_erase_mode(self, checked: bool) -> None:
        """Toggles erase mode. Delegates to DrawingModesManager."""
        self.drawing_modes_manager.toggle_erase_mode(checked)

    @pyqtSlot(bool)
    def _toggle_sphere_mode(self, checked: bool) -> None:
        """Toggles sphere mode. Delegates to DrawingModesManager."""
        self.drawing_modes_manager.toggle_sphere_mode(checked)

    @pyqtSlot(bool)
    def _toggle_rectangle_mode(self, checked: bool) -> None:
        """Toggles rectangle mode. Delegates to DrawingModesManager."""
        self.drawing_modes_manager.toggle_rectangle_mode(checked)

    def _reset_all_drawing_modes(self) -> None:
        """Resets all drawing modes. Delegates to DrawingModesManager."""
        self.drawing_modes_manager.reset_all_drawing_modes()

    @pyqtSlot()
    def _trigger_load_odf(self) -> None:
        """Loads a NIfTI file as ODF coefficients."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load ODF (SH Coefficients)", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if not file_path:
            return

        try:
            self.vtk_panel.update_status("Loading ODF data...")
            QApplication.processEvents()

            img = nib.load(file_path)
            data = img.get_fdata()
            affine = img.affine

            # Validate Shape
            if data.ndim != 4:
                QMessageBox.warning(
                    self, "ODF Error", "File must be a 4D volume (SH coefficients)."
                )
                return

            n_coeffs = data.shape[-1]
            try:
                sh_order = odf_utils.calculate_sh_order(n_coeffs)
            except ValueError as e:
                QMessageBox.warning(self, "ODF Error", str(e))
                return

            self.odf_data = data
            self.odf_affine = affine
            self.odf_path = file_path
            self.odf_sh_order = sh_order

            # Pre-compute Sphere and Basis (Tournier07) ##TODO - handle other basis types
            self.odf_sphere = odf_utils.generate_symmetric_sphere(
                radius=1.0, subdivisions=3
            )
            self.vtk_panel.update_status("Computing SH Basis...")
            QApplication.processEvents()

            self.odf_basis_matrix = odf_utils.compute_sh_basis(
                self.odf_sphere.vertices, sh_order, basis_type="tournier07"
            )

            # Lower-resolution sphere for lightweight tunnel rendering
            # (subdivisions=2 → 162 vertices vs 642, ~4x fewer polygons)
            self.odf_tunnel_sphere = odf_utils.generate_symmetric_sphere(
                radius=1.0, subdivisions=2
            )
            self.odf_tunnel_basis = odf_utils.compute_sh_basis(
                self.odf_tunnel_sphere.vertices, sh_order, basis_type="tournier07"
            )

            self.vtk_panel.update_status(
                f"ODF Loaded (Order {sh_order}). Ready for Tunnel View."
            )
            self._update_action_states()
            self._update_data_panel_display()

        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Error loading ODF: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"Could not load ODF file:\n{e}")

    @pyqtSlot()
    def _trigger_load_parcellation(self) -> None:
        """Loads a FreeSurfer parcellation file. Delegates to ConnectivityManager."""
        self.connectivity_manager.load_parcellation()

    @pyqtSlot()
    def _trigger_compute_connectivity(self) -> None:
        """Computes and exports connectivity matrix. Delegates to ConnectivityManager."""
        self.connectivity_manager.compute_and_export()

    @pyqtSlot(bool)
    def _toggle_parcellation_overlay(self, checked: bool) -> None:
        """Toggles the visibility of the 3D parcellation overlay."""
        if not checked:
            # Hide the overlay
            self.connectivity_manager.remove_parcellation_overlay()
            self._sync_parcellation_toggle_state(False)
            return

        # Show the overlay (just visibility, no creation/computation)
        # If actors exist, show them; otherwise do nothing
        if self.parcellation_region_actors:
            self.connectivity_manager._show_parcellation_actors()
            self._sync_parcellation_toggle_state(True)
        else:
            # No actors to show - need to create overlay first via recalculate
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
                            item_data = child.data(
                                0, Qt.ItemDataRole.UserRole
                            )
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
        """
        Classifies a FreeSurfer region name into hemisphere.

        Args:
            label_name: The region name from FreeSurfer parcellation.

        Returns:
            'left', 'right', or 'other'
        """
        name_lower = label_name.lower()

        # Left hemisphere indicators
        if any(x in name_lower for x in ["left-", "ctx-lh-", "-lh-", "lh-", "lh_"]):
            return "left"

        # Right hemisphere indicators
        if any(x in name_lower for x in ["right-", "ctx-rh-", "-rh-", "rh-", "rh_"]):
            return "right"

        # Bilateral/midline/other structures
        return "other"

    def _clear_parcellation(self) -> None:
        """Clears all parcellation data and overlay."""
        # Remove overlay from scene
        self.connectivity_manager.remove_parcellation_overlay()

        # Clear parcellation data
        self.parcellation_data = None
        self.parcellation_affine = None
        self.parcellation_path = None
        self.parcellation_labels = {}
        self.parcellation_connected_labels = set()
        self.parcellation_region_visibility = {}
        self.parcellation_main_labels = set()
        self.parcellation_label_colors = {}

        # Clear region filter data
        self.parcellation_region_states = {}
        self.parcellation_region_intersection_cache = {}
        self.parcellation_start_labels = None
        self.parcellation_end_labels = None
        self.parcellation_visible_indices = None

        # Update menu action state
        if self.view_parcellation_action is not None:
            self.view_parcellation_action.setChecked(False)

        # Update UI
        self._update_action_states()
        self._update_data_panel_display()

    @pyqtSlot(bool)
    def _toggle_odf_tunnel(self, checked: bool) -> None:
        """Computes the mask and updates the VTK actor with progress indication."""
        if not checked:
            self.odf_tunnel_is_visible = False
            if self.vtk_panel:
                self.vtk_panel.remove_odf_actor()
            self._update_data_panel_display()
            return

        if self.odf_data is None or self.tractogram_data is None:
            return

        # Apply Stride (Skip) to match visual representation
        sorted_indices = sorted(list(self.visible_indices))

        # Apply the current render stride
        stride = self.render_stride
        strided_indices = sorted_indices[::stride]

        # Retrieve only the subset of streamlines
        current_streamlines = [self.tractogram_data[i] for i in strided_indices]

        # Check Limit against the STRIDED count
        if len(current_streamlines) > self.MAX_ODF_STREAMLINES:
            QMessageBox.warning(
                self,
                "Performance Warning",
                f"Too many streamlines selected ({len(current_streamlines)}).\n"
                f"Limit is {self.MAX_ODF_STREAMLINES}. \n\n"
                f"Tip: Increase the 'Skip %' or use ROIs to reduce the count.",
            )
            self.view_odf_tunnel_action.setChecked(False)
            return

        self.vtk_panel.update_status("Computing Tunnel View...")

        # Initialize Progress Bar (4 steps total)
        # 1. Mask Creation, 2. Mask Application, 3. SH Projection, 4. Rendering
        TOTAL_STEPS = 4
        self.vtk_panel.update_progress_bar(0, TOTAL_STEPS, visible=True)
        QApplication.processEvents()

        try:
            # Create Mask (Heavy operation)
            mask = odf_utils.create_tunnel_mask(
                current_streamlines,
                self.odf_affine,
                self.odf_data.shape,
                dilation_iter=1,
            )

            # Update Progress -> 25%
            self.vtk_panel.update_progress_bar(1, TOTAL_STEPS, visible=True)
            QApplication.processEvents()

            # Apply Mask to ODF Data
            masked_coeffs = self.odf_data * mask[..., np.newaxis]

            # Update Progress -> 50%
            self.vtk_panel.update_progress_bar(2, TOTAL_STEPS, visible=True)
            QApplication.processEvents()

            # Project to Amplitudes (SF) using lightweight tunnel sphere
            tunnel_sphere = self.odf_tunnel_sphere
            tunnel_basis = self.odf_tunnel_basis
            amplitudes_shape = self.odf_data.shape[:3] + (
                tunnel_sphere.vertices.shape[0],
            )
            odf_amplitudes = np.zeros(amplitudes_shape, dtype=np.float32)

            # Flatten mask to find indices
            mask_indices = np.where(mask)

            extent = None

            if len(mask_indices[0]) > 0:
                valid_coeffs = masked_coeffs[mask_indices]
                valid_amps = np.dot(valid_coeffs, tunnel_basis.T)
                odf_amplitudes[mask_indices] = valid_amps

                # Determine min/max for x, y, z to constrain the actor
                min_x, max_x = np.min(mask_indices[0]), np.max(mask_indices[0])
                min_y, max_y = np.min(mask_indices[1]), np.max(mask_indices[1])
                min_z, max_z = np.min(mask_indices[2]), np.max(mask_indices[2])

                extent = (min_x, max_x, min_y, max_y, min_z, max_z)

            # Update Progress -> 75%
            self.vtk_panel.update_progress_bar(3, TOTAL_STEPS, visible=True)
            QApplication.processEvents()

            # Update VTK with Extent
            self.vtk_panel.update_odf_actor(
                odf_amplitudes, tunnel_sphere, self.odf_affine, extent=extent
            )

            # Update Progress -> 100% and Hide
            self.vtk_panel.update_progress_bar(TOTAL_STEPS, TOTAL_STEPS, visible=True)
            QApplication.processEvents()

            # Mark tunnel as visible and update data panel
            self.odf_tunnel_is_visible = True
            self._update_data_panel_display()

        except (RuntimeError, ValueError, IndexError, TypeError) as e:
            logger.error(f"Error computing Tunnel View: {e}", exc_info=True)
            self.vtk_panel.update_status("Error generating Tunnel View.")
            self.view_odf_tunnel_action.setChecked(False)
            self.odf_tunnel_is_visible = False
        finally:
            self.vtk_panel.update_progress_bar(0, 0, visible=False)

    def _toggle_odf_tunnel_visibility(self, visible: bool) -> None:
        """
        Toggles the visibility of the ODF tunnel actor without recomputing.

        Args:
            visible: True to show, False to hide the ODF tunnel.
        """
        if not self.vtk_panel or not self.vtk_panel.odf_actor:
            return

        try:
            self.vtk_panel.odf_actor.SetVisibility(visible)
            self.odf_tunnel_is_visible = visible

            # Sync the menu action state
            with signals_blocked(self.view_odf_tunnel_action):
                self.view_odf_tunnel_action.setChecked(visible)

            if self.vtk_panel.render_window:
                self.vtk_panel.render_window.Render()

            status = "shown" if visible else "hidden"
            self.vtk_panel.update_status(f"ODF Tunnel {status}")

        except (ValueError, RuntimeError) as e:
            logger.error(f"Error toggling ODF tunnel visibility: {e}", exc_info=True)

    @pyqtSlot()
    def _remove_odf_data(self) -> None:
        """
        Removes the ODF data and tunnel actor from the scene.

        Clears all ODF-related data and updates the UI accordingly.
        """
        try:
            # Remove the ODF actor from the scene
            if self.vtk_panel:
                self.vtk_panel.remove_odf_actor()

            # Clear ODF data
            self.odf_data = None
            self.odf_affine = None
            self.odf_path = None
            self.odf_sh_order = 0
            self.odf_sphere = None
            self.odf_basis_matrix = None
            self.odf_tunnel_sphere = None
            self.odf_tunnel_basis = None
            self.odf_tunnel_is_visible = False

            # Update the menu action state
            with signals_blocked(self.view_odf_tunnel_action):
                self.view_odf_tunnel_action.setChecked(False)
                self.view_odf_tunnel_action.setEnabled(False)

            # Update the data panel
            self._update_data_panel_display()
            self._update_action_states()

            if self.vtk_panel:
                self.vtk_panel.update_status("ODF data removed")

        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Error removing ODF data: {e}", exc_info=True)

    def _update_bundle_info_display(self) -> None:
        """Updates the data information QLabel in the status bar for both streamlines and image."""
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
            if "dimensions" in header:
                dims_val = header["dimensions"]
                if (
                    isinstance(dims_val, (tuple, list, np.ndarray))
                    and len(dims_val) == 3
                ):
                    dims_str = format_tuple(dims_val, precision=0)
            if "voxel_sizes" in header:
                vox_val = header["voxel_sizes"]
                if (
                    isinstance(vox_val, (tuple, list, np.ndarray))
                    and len(dims_val) == 3
                ):
                    vox_str = format_tuple(vox_val, precision=2)
            if "voxel_order" in header and isinstance(header["voxel_order"], str):
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
            shape_str = format_tuple(self.anatomical_image_data.shape, precision=0)
            image_text = f"Image: {filename} | Shape={shape_str}"

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
        """
        Updates the QTreeWidget in the data panel dock.

        Uses debouncing to prevent expensive rebuilds during rapid-fire calls.
        Multiple calls within 100ms are coalesced into a single update.
        """
        if not self.data_tree_widget:
            return

        # Initialize debounce timer on first use
        if self._data_panel_debounce_timer is None:
            self._data_panel_debounce_timer = QTimer(self)
            self._data_panel_debounce_timer.setSingleShot(True)
            self._data_panel_debounce_timer.timeout.connect(
                self._perform_data_panel_update
            )

        # Mark that an update is pending and restart the timer
        self._data_panel_update_pending = True
        self._data_panel_debounce_timer.start(100)  # 100ms debounce delay

    @pyqtSlot()
    def _perform_data_panel_update(self) -> None:
        """
        Performs the actual data panel update (debounced).

        This is called by the debounce timer after the delay expires.
        Blocks signals during rebuild to prevent cascading itemChanged events.
        Delegates tree construction to focused sub-methods for maintainability.
        """
        if not self._data_panel_update_pending:
            return

        self._data_panel_update_pending = False

        if not self.data_tree_widget:
            return

        # Block signals during rebuild to prevent cascading callbacks
        self.data_tree_widget.blockSignals(True)
        try:
            # Save expansion state before clearing
            expanded_items = set()
            for i in range(self.data_tree_widget.topLevelItemCount()):
                item = self.data_tree_widget.topLevelItem(i)
                self._collect_expanded_items(item, "", expanded_items)

            self.data_tree_widget.clear()
            self._pending_expanded_items = expanded_items

            # Build each section of the data panel tree
            self._build_tractogram_tree_item()
            self._build_image_tree_item()
            self._build_roi_tree_items()
            self._build_odf_tree_item()
            self._build_parcellation_tree_item()

            self.data_tree_widget.resizeColumnToContents(0)

            # Restore expansion state
            if self._pending_expanded_items:
                for i in range(self.data_tree_widget.topLevelItemCount()):
                    item = self.data_tree_widget.topLevelItem(i)
                    self._restore_expanded_items(
                        item, "", self._pending_expanded_items
                    )
                self._pending_expanded_items = set()
        finally:
            self.data_tree_widget.blockSignals(False)

        # Force visual update to ensure checkbox states are immediately reflected
        if self.data_tree_widget.viewport():
            self.data_tree_widget.viewport().update()

    def _build_tractogram_tree_item(self) -> None:
        """Builds the Tractogram section of the data panel tree."""
        if self.tractogram_data is None:
            return

        tractogram_header = QTreeWidgetItem(self.data_tree_widget, ["Tractogram"])
        tractogram_header.setIcon(
            0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )

        bundle_name = (
            os.path.basename(self.original_trk_path)
            if self.original_trk_path
            else "Loaded Bundle"
        )

        bundle_item = QTreeWidgetItem(tractogram_header, [bundle_name])
        bundle_item.setFlags(bundle_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        bundle_state = (
            Qt.CheckState.Checked
            if self.bundle_is_visible
            else Qt.CheckState.Unchecked
        )
        bundle_item.setCheckState(0, bundle_state)
        bundle_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "bundle"})

        count = len(self.visible_indices)
        ext = (
            self.original_file_extension.upper()
            if self.original_file_extension
            else "TRK"
        )
        dims = "N/A"
        if self.original_trk_header and "dimensions" in self.original_trk_header:
            dims = format_tuple(self.original_trk_header["dimensions"], precision=0)

        tooltip_text = f"Type: {ext}\nCount: {count}\nDimensions: {dims}"
        bundle_item.setToolTip(0, tooltip_text)

        if self.scalar_data_per_point:
            scalars_root = QTreeWidgetItem(bundle_item, ["Scalars"])
            scalars_root.setIcon(
                0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            )
            for scalar_name in self.scalar_data_per_point.keys():
                scalar_item = QTreeWidgetItem(scalars_root, [scalar_name])
                if scalar_name == self.active_scalar_name:
                    font = scalar_item.font(0)
                    font.setBold(True)
                    scalar_item.setFont(0, font)

        bundle_item.setExpanded(True)
        tractogram_header.setExpanded(True)

    def _build_image_tree_item(self) -> None:
        """Builds the Anatomical Image section of the data panel tree."""
        if self.anatomical_image_data is None:
            return

        image_header = QTreeWidgetItem(self.data_tree_widget, ["Anatomical Image"])
        image_header.setIcon(
            0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )

        image_name = (
            os.path.basename(self.anatomical_image_path)
            if self.anatomical_image_path
            else "Loaded Image"
        )

        image_item = QTreeWidgetItem(image_header, [image_name])
        image_item.setFlags(image_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        image_state = (
            Qt.CheckState.Checked
            if self.image_is_visible
            else Qt.CheckState.Unchecked
        )
        image_item.setCheckState(0, image_state)
        image_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "image"})
        shape_str = format_tuple(self.anatomical_image_data.shape, precision=0)
        image_item.setToolTip(
            0, f"Path: {self.anatomical_image_path}\nShape: {shape_str}"
        )
        image_header.setExpanded(True)

    def _build_roi_tree_items(self) -> None:
        """Builds the ROI Layers section of the data panel tree."""
        if not self.roi_layers:
            return

        roi_root_item = QTreeWidgetItem(self.data_tree_widget, ["ROI Layers"])
        roi_root_item.setIcon(
            0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )

        for path, roi_info in self.roi_layers.items():
            roi_name = roi_info.get("display_name", os.path.basename(path))

            state_str = ""
            if path in self.roi_states:
                if self.roi_states[path].get("select"):
                    state_str = " [SELECT]"
                elif self.roi_states[path].get("include"):
                    state_str = " [INCLUDE]"
                elif self.roi_states[path].get("exclude"):
                    state_str = " [EXCLUDE]"

            display_text = f"{roi_name}{state_str}"
            roi_item = QTreeWidgetItem(roi_root_item, [display_text])

            # Color indicator
            roi_color = roi_info.get("color", (1.0, 0.0, 0.0))
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            c_r = int(roi_color[0] * 255)
            c_g = int(roi_color[1] * 255)
            c_b = int(roi_color[2] * 255)
            color = QColor(c_r, c_g, c_b)

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(2, 2, 12, 12)
            painter.end()

            roi_item.setIcon(0, QIcon(pixmap))

            roi_item.setFlags(roi_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            is_visible = self.roi_visibility.get(path, True)
            roi_state = (
                Qt.CheckState.Checked if is_visible else Qt.CheckState.Unchecked
            )
            roi_item.setCheckState(0, roi_state)
            roi_item.setData(
                0, Qt.ItemDataRole.UserRole, {"type": "roi", "path": path}
            )

            shape_str = format_tuple(roi_info["data"].shape, precision=0)
            roi_item.setToolTip(0, f"Path: {path}\nShape: {shape_str}")

        roi_root_item.setExpanded(True)

    def _build_odf_tree_item(self) -> None:
        """Builds the ODF Data section of the data panel tree."""
        if self.odf_data is None:
            return

        odf_header = QTreeWidgetItem(self.data_tree_widget, ["ODF Data"])
        odf_header.setIcon(
            0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )

        odf_name = (
            os.path.basename(self.odf_path) if self.odf_path else "Loaded ODF"
        )

        odf_file_item = QTreeWidgetItem(odf_header, [odf_name])
        odf_file_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "odf_data"})
        shape_str = format_tuple(self.odf_data.shape, precision=0)
        odf_file_item.setToolTip(
            0,
            f"Path: {self.odf_path}\n"
            f"Shape: {shape_str}\n"
            f"SH Order: {self.odf_sh_order}",
        )

        # ODF Tunnel View item (checkable for visibility toggle)
        if self.vtk_panel and self.vtk_panel.odf_actor is not None:
            tunnel_item = QTreeWidgetItem(odf_header, ["ODF Tunnel View"])
            tunnel_item.setFlags(
                tunnel_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            )
            tunnel_state = (
                Qt.CheckState.Checked
                if self.odf_tunnel_is_visible
                else Qt.CheckState.Unchecked
            )
            tunnel_item.setCheckState(0, tunnel_state)
            tunnel_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "odf_tunnel"})
            tunnel_item.setToolTip(
                0,
                "Toggle visibility of the ODF Tunnel View.\n"
                "Right-click to remove.",
            )

        odf_header.setExpanded(True)

    def _build_parcellation_tree_item(self) -> None:
        """Builds the FreeSurfer Parcellation section of the data panel tree."""
        if self.parcellation_data is None:
            return

        parcellation_header = QTreeWidgetItem(
            self.data_tree_widget, ["FreeSurfer Parcellation"]
        )
        parcellation_header.setIcon(
            0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )

        parc_name = (
            os.path.basename(self.parcellation_path)
            if self.parcellation_path
            else "Loaded Parcellation"
        )

        parc_item = QTreeWidgetItem(parcellation_header, [parc_name])
        parc_item.setFlags(parc_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

        parc_visible = self._parcellation_overlay_visible
        parc_state = (
            Qt.CheckState.Checked if parc_visible else Qt.CheckState.Unchecked
        )
        parc_item.setCheckState(0, parc_state)
        parc_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "parcellation"})

        shape_str = format_tuple(self.parcellation_data.shape, precision=0)
        n_labels = len(np.unique(self.parcellation_data)) - 1  # Exclude 0
        parc_item.setToolTip(
            0,
            f"Path: {self.parcellation_path}\n"
            f"Shape: {shape_str}\n"
            f"Total Labels: {n_labels}",
        )

        # Connected Regions submenu — organized by hemisphere
        connected_labels = self.parcellation_connected_labels
        if connected_labels:
            self._build_connected_regions_tree(parc_item, connected_labels)

        parc_item.setExpanded(True)
        parcellation_header.setExpanded(True)

    def _build_connected_regions_tree(
        self,
        parent_item: QTreeWidgetItem,
        connected_labels: Set[int],
    ) -> None:
        """Builds the Connected Regions sub-tree under a parcellation item."""
        connected_item = QTreeWidgetItem(
            parent_item, [f"Connected Regions ({len(connected_labels)})"]
        )
        connected_item.setIcon(
            0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        )

        main_labels = self.parcellation_main_labels

        # Classify regions by hemisphere
        left_regions: List[Tuple[int, str]] = []
        right_regions: List[Tuple[int, str]] = []
        other_regions: List[Tuple[int, str]] = []

        for label in connected_labels:
            label_name = self.parcellation_labels.get(int(label), f"Region_{label}")
            hemisphere = self._classify_region_hemisphere(label_name)
            if hemisphere == "left":
                left_regions.append((label, label_name))
            elif hemisphere == "right":
                right_regions.append((label, label_name))
            else:
                other_regions.append((label, label_name))

        left_regions.sort(key=lambda x: x[1].lower())
        right_regions.sort(key=lambda x: x[1].lower())
        other_regions.sort(key=lambda x: x[1].lower())

        # Build hemisphere folders
        hemisphere_data = [
            ("Left Hemisphere", left_regions),
            ("Right Hemisphere", right_regions),
            ("Bilateral/Other", other_regions),
        ]
        for folder_name, regions in hemisphere_data:
            if not regions:
                continue
            folder = QTreeWidgetItem(
                connected_item, [f"{folder_name} ({len(regions)})"]
            )
            folder.setIcon(
                0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            )
            self._add_region_items(folder, regions, main_labels)
            folder.setExpanded(False)

        connected_item.setExpanded(False)

    def _add_region_items(
        self,
        parent_item: QTreeWidgetItem,
        regions: List[Tuple[int, str]],
        main_labels: Set[int],
        limit: int = 100,
    ) -> None:
        """Adds individual region items to a hemisphere folder."""
        parc_states = self.parcellation_region_states

        for label, label_name in regions[:limit]:
            display_name = label_name
            state = parc_states.get(int(label), {})
            if state.get("include"):
                display_name = f"{label_name} [INC]"
            elif state.get("exclude"):
                display_name = f"{label_name} [EXC]"

            region_item = QTreeWidgetItem(parent_item, [display_name])
            region_item.setFlags(
                region_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            )

            has_actor = int(label) in main_labels
            region_visible = self.parcellation_region_visibility.get(
                int(label), has_actor
            )
            region_state = (
                Qt.CheckState.Checked
                if region_visible
                else Qt.CheckState.Unchecked
            )
            region_item.setCheckState(0, region_state)

            region_item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {"type": "parcellation_region", "label": int(label)},
            )

            actor_status = "Active (has actor)" if has_actor else "On-demand"
            region_item.setToolTip(
                0,
                f"Label ID: {label}\nStatus: {actor_status}\nToggle to show/hide",
            )

        if len(regions) > limit:
            more_item = QTreeWidgetItem(
                parent_item,
                [f"... and {len(regions) - limit} more regions"],
            )
            more_item.setDisabled(True)

    def _collect_expanded_items(
        self, item: QTreeWidgetItem, path: str, expanded_set: set
    ) -> None:
        """Recursively collects paths of expanded items."""
        # Build path using first part of item text (before any brackets/tags)
        item_text = item.text(0).split(" [")[0].split(" (")[0]
        current_path = f"{path}/{item_text}" if path else item_text

        if item.isExpanded():
            expanded_set.add(current_path)

        for i in range(item.childCount()):
            self._collect_expanded_items(item.child(i), current_path, expanded_set)

    def _restore_expanded_items(
        self, item: QTreeWidgetItem, path: str, expanded_set: set
    ) -> None:
        """Recursively restores expansion state of items."""
        # Build path using first part of item text (before any brackets/tags)
        item_text = item.text(0).split(" [")[0].split(" (")[0]
        current_path = f"{path}/{item_text}" if path else item_text

        if current_path in expanded_set:
            item.setExpanded(True)

        for i in range(item.childCount()):
            self._restore_expanded_items(item.child(i), current_path, expanded_set)

    # Undo/Redo Core Logic
    @pyqtSlot()
    def _perform_undo(self) -> None:
        """Performs undo. Delegates to StateManager."""
        self.state_manager.perform_undo()

    @pyqtSlot()
    def _perform_redo(self) -> None:
        """Performs redo. Delegates to StateManager."""
        self.state_manager.perform_redo()

    def _save_roi_state_for_undo(self, roi_name: str) -> None:
        """Saves ROI state for undo. Delegates to StateManager."""
        self.state_manager.save_roi_state_for_undo(roi_name)

    # Command Actions Logic
    @pyqtSlot()
    def _perform_clear_selection(self) -> None:
        """Clears selection. Delegates to StateManager."""
        self.state_manager.perform_clear_selection()

    @pyqtSlot()
    def _perform_reset_camera(self) -> None:
        """Resets camera. Delegates to StateManager."""
        self.state_manager.perform_reset_camera()

    @pyqtSlot()
    def _perform_delete_selection(self) -> None:
        """Deletes selection. Delegates to StateManager."""
        self.state_manager.perform_delete_selection()

    @pyqtSlot()
    def _increase_radius(self) -> None:
        """Increases radius. Delegates to StateManager."""
        self.state_manager.increase_radius()

    @pyqtSlot()
    def _decrease_radius(self) -> None:
        """Decreases radius. Delegates to StateManager."""
        self.state_manager.decrease_radius()

    @pyqtSlot()
    def _hide_sphere(self) -> None:
        """Hides sphere. Delegates to StateManager."""
        self.state_manager.hide_sphere()

    # View Action Logic
    @pyqtSlot(object)
    def _set_color_mode(self, mode: ColorMode) -> None:
        """Sets color mode. Delegates to StateManager."""
        self.state_manager.set_color_mode(mode)

    # GUI Action Methods
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

            if self.vtk_panel:
                msg = (
                    "Closing bundle..."
                    if keep_image
                    else "Closing bundle (also clears image)..."
                )
                self.vtk_panel.update_status(msg)
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
                    self.anatomical_image_path = None
                    self.anatomical_image_data = None
                    self.anatomical_image_affine = None
                    self.vtk_panel.clear_anatomical_slices()

            # Reset streamline data state
            self.tractogram_data = None
            self.streamline_bboxes = None
            self.visible_indices = set()
            self._visibility_version += 1
            self.original_trk_header = None
            self.original_trk_affine = None
            self.original_trk_path = None
            self.original_file_extension = None

            # Close TRX memmap file reference (releases temp directory)
            self._close_trx_file()

            self.scalar_data_per_point = None
            self.active_scalar_name = None
            self.odf_data = None
            self.odf_affine = None
            self.odf_path = None
            self.odf_sh_order = 0
            self.odf_sphere = None
            self.odf_basis_matrix = None
            self.odf_tunnel_sphere = None
            self.odf_tunnel_basis = None
            self.odf_tunnel_is_visible = False
            with signals_blocked(self.view_odf_tunnel_action):
                self.view_odf_tunnel_action.setChecked(False)
                self.view_odf_tunnel_action.setEnabled(False)
            self.unified_undo_stack = []
            self.unified_redo_stack = []
            self.current_color_mode = ColorMode.ORIENTATION
            self.color_default_action.setChecked(True)
            self.scalar_range_initialized = False
            if self.scalar_toolbar:
                self.scalar_toolbar.setVisible(False)

            # Invalidate parcellation intersection cache (tied to old streamlines)
            self.parcellation_region_intersection_cache = {}
            self.parcellation_region_states = {}
            self.parcellation_start_labels = None
            self.parcellation_end_labels = None
            self.parcellation_visible_indices = None

            # Update VTK
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()  # Should remove streamline actor
                status_msg = (
                    "Bundle closed."
                    if keep_image
                    else "Bundle closed (Image also cleared)."
                )
                self.vtk_panel.update_status(status_msg)
                interactor = (
                    self.vtk_panel.render_window.GetInteractor()
                    if self.vtk_panel.render_window
                    else None
                )
                if (
                    self.vtk_panel.render_window
                    and interactor is not None
                    and interactor.GetInitialized()
                ):
                    self.vtk_panel.render_window.Render()

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

            if bundle_path:
                if os.path.exists(bundle_path):
                    logger.info(f"Loading initial bundle: {bundle_path}")
                    self.scalar_range_initialized = False
                    if self.scalar_toolbar:
                        self.scalar_toolbar.setVisible(False)
                    self.bundle_is_visible = True

                    # Pass keep_image=True if we just loaded an image
                    keep_image = self.anatomical_image_data is not None
                    file_io.load_streamlines_file(
                        self, keep_image=keep_image, file_path=bundle_path
                    )
                else:
                    logger.error(f"Bundle path not found: {bundle_path}")

            if roi_paths:
                valid_rois = [p for p in roi_paths if os.path.exists(p)]
                if valid_rois:
                    logger.info(f"Loading initial ROIs: {valid_rois}")
                    loaded_rois = file_io.load_roi_images(self, file_paths=valid_rois)

                    # Iterate through every loaded ROI
                    for _, _, roi_path in loaded_rois:
                        if roi_path in self.roi_layers:
                            logger.warning(
                                f"ROI '{os.path.basename(roi_path)}' is already loaded."
                            )
                            continue

                        self.roi_visibility[roi_path] = True
                        self.roi_opacities[roi_path] = 0.5

                        try:
                            # Load the main anatomical image object from its stored path
                            if not self.anatomical_image_path:
                                logger.warning(
                                    f"Skipping ROI {roi_path}: No anatomical image loaded for reorientation."
                                )
                                continue

                            from .file_io import _align_if_oblique

                            anatomical_img = nib.load(self.anatomical_image_path)
                            anatomical_img = _align_if_oblique(
                                anatomical_img, self.anatomical_image_path, None
                            )

                            roi_img = nib.load(roi_path)
                            roi_img = _align_if_oblique(roi_img, roi_path, None)
                            # Ensure proper coordinate system alignment
                            current_ornt = nib.io_orientation(roi_img.affine)
                            target_ornt = nib.io_orientation(anatomical_img.affine)

                            if not np.array_equal(current_ornt, target_ornt):
                                if self.vtk_panel:
                                    current_axcodes = "".join(
                                        nib.aff2axcodes(roi_img.affine)
                                    )
                                    target_axcodes = "".join(
                                        nib.aff2axcodes(anatomical_img.affine)
                                    )
                                    self.vtk_panel.update_status(
                                        f"Reorienting {os.path.basename(roi_path)} ({current_axcodes} -> {target_axcodes})..."
                                    )
                                transform = ornt_transform(current_ornt, target_ornt)
                                reoriented_roi_img = roi_img.as_reoriented(transform)
                                roi_data = reoriented_roi_img.get_fdata()
                                roi_affine = reoriented_roi_img.affine
                            else:
                                roi_data = roi_img.get_fdata()
                                roi_affine = roi_img.affine

                            # Store the data
                            inv_affine = np.linalg.inv(roi_affine)
                            main_affine = self.anatomical_image_affine
                            T_main_to_roi = np.dot(inv_affine, main_affine)

                            self.roi_layers[roi_path] = {
                                "data": roi_data,
                                "affine": roi_affine,
                                "path": roi_path,
                                "inv_affine": inv_affine,
                                "T_main_to_roi": T_main_to_roi,
                            }

                            # Tell VTK panel to create and add the new actors
                            if self.vtk_panel:
                                self.vtk_panel.add_roi_layer(
                                    roi_path, roi_data, roi_affine
                                )
                                self.vtk_panel.update_status(
                                    f"Aligned and added {os.path.basename(roi_path)}."
                                )

                        except (OSError, ValueError, TypeError) as e:
                            logger.error(f"Error processing ROI {roi_path}: {e}")
                            QMessageBox.warning(
                                self,
                                "ROI Load Error",
                                f"Failed to load ROI: {os.path.basename(roi_path)}\n{e}",
                            )
                            continue

                    self._update_action_states()
                else:
                    logger.warning("No valid ROI paths found in arguments.")

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

                        # Calculate radius in voxels
                        edge_world = center_world + np.array([r_val, 0, 0])
                        p_h_e = np.append(edge_world, 1.0)
                        edge_vox = np.dot(roi_inv_affine, p_h_e)[:3]

                        radius_vox = np.linalg.norm(center_vox - edge_vox)

                        # Bounding box
                        min_v = np.floor(center_vox - radius_vox).astype(int)
                        max_v = np.ceil(center_vox + radius_vox).astype(int)
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
                            mask = dist_sq <= radius_vox**2

                            roi_slice = roi_data[
                                min_v[0] : max_v[0] + 1,
                                min_v[1] : max_v[1] + 1,
                                min_v[2] : max_v[2] + 1,
                            ]
                            roi_slice[mask] = 1

                            # Update VTK
                            if self.vtk_panel:
                                # We need to add the actor first since _trigger_new_roi doesn't add it until drawn
                                self.vtk_panel.add_roi_layer(
                                    roi_name, roi_data, roi_affine
                                )

                                # Store sphere params for future interaction
                                self.vtk_panel.sphere_params_per_roi[roi_name] = {
                                    "center": center_world,
                                    "radius": r_val,
                                    "view_type": "axial",  # Default
                                }

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
    ) -> None:
        """Loads an anatomical image on a background thread with a progress dialog.

        Once the image finishes loading, any remaining CLI arguments (bundle,
        ROIs, sphere ROIs) are loaded via ``load_initial_files``.

        Args:
            anat_path: Path to the anatomical NIfTI image.
            bundle_path: Optional path to a bundle file to load afterwards.
            roi_paths: Optional ROI paths to load afterwards.
            roi_in: Optional sphere ROI coordinates to create afterwards.
            radius: Optional radii for sphere ROIs.
        """
        # Setup progress dialog
        progress = QProgressDialog("Initializing...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Loading Image")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(350)
        progress.setStyleSheet(self.theme_manager.get_progress_dialog_style())
        progress.setValue(0)
        progress.show()

        # Create and configure the loader thread
        loader_thread = file_io.AnatomicalImageLoaderThread(anat_path)

        def on_progress(val, msg):
            progress.setValue(val)
            progress.setLabelText(msg)

        def on_error(msg):
            progress.cancel()
            logger.error(f"Error loading initial anatomical image: {msg}")
            QMessageBox.critical(
                self, "Load Error", f"Error loading image:\n{msg}"
            )
            if self.vtk_panel:
                self.vtk_panel.update_status("Error loading image.")
            # Continue loading remaining files without anatomical image
            self.load_initial_files(
                bundle_path=bundle_path,
                roi_paths=roi_paths,
                roi_in=roi_in,
                radius=radius,
            )

        def on_finished(data):
            try:
                progress.setLabelText("Creating slicer actors...")
                progress.setValue(95)
                QApplication.processEvents()

                self.anatomical_image_data = data["data"]
                self.anatomical_image_affine = data["affine"]
                self.anatomical_image_path = data["path"]
                self.anatomical_mmap_image = data.get("mmap_image")
                self.image_is_visible = True

                if self.vtk_panel:
                    self.vtk_panel.update_anatomical_slices()
                    QApplication.processEvents()
                    if self.vtk_panel.scene:
                        self.vtk_panel.scene.reset_camera()
                        self.vtk_panel.scene.reset_clipping_range()
                    if self.vtk_panel.render_window:
                        self.vtk_panel.render_window.Render()
                    self.vtk_panel.update_status(
                        f"Loaded: {os.path.basename(data['path'])}"
                    )

                self._update_bundle_info_display()
                self._update_action_states()

                progress.close()

            except (RuntimeError, ValueError, AttributeError, KeyError) as e:
                logger.error(
                    f"Error finalizing initial image load: {e}", exc_info=True
                )
                progress.close()
                QMessageBox.critical(
                    self, "Load Error", f"Error finalizing load:\n{e}"
                )

            # Continue loading remaining CLI arguments
            self.load_initial_files(
                bundle_path=bundle_path,
                roi_paths=roi_paths,
                roi_in=roi_in,
                radius=radius,
            )

        # Connect signals (QueuedConnection ensures VTK calls run on main thread)
        loader_thread.progress.connect(on_progress)
        loader_thread.error.connect(on_error)
        loader_thread.finished.connect(
            on_finished, type=Qt.ConnectionType.QueuedConnection
        )
        progress.canceled.connect(loader_thread.terminate)

        # Keep reference to prevent garbage collection
        self._image_loader_thread = loader_thread

        loader_thread.start()

    # Action Trigger Wrappers
    @pyqtSlot()
    def _trigger_clear_anatomical_image(self, notify: bool = True) -> None:
        """Clears the currently loaded anatomical image."""
        if self.anatomical_image_data is None:
            return

        self.anatomical_image_data = None
        self.anatomical_image_affine = None
        self.anatomical_image_path = None
        if self.anatomical_mmap_image:
            self.anatomical_mmap_image.clear_cache()
        self.anatomical_mmap_image = None
        self.image_is_visible = True  # Reset visibility flag

        if self.vtk_panel:
            self.vtk_panel.clear_anatomical_slices()
            if notify:
                self.vtk_panel.update_status("Anatomical image cleared.")

            # If no bundle is loaded, reset camera
            if not self.tractogram_data and self.vtk_panel.scene:
                self.vtk_panel.scene.reset_camera()

        self._update_bundle_info_display()
        self._update_action_states()

    @pyqtSlot()
    def _trigger_load_streamlines(self) -> None:
        """Wrapper to call the streamline load function from file_io."""
        self.scalar_range_initialized = False
        if self.scalar_toolbar:
            self.scalar_toolbar.setVisible(False)
        self.bundle_is_visible = True

        file_io.load_streamlines_file(self)
        # All post-load state setup (visible indices, ROI caches, skip level,
        # etc.) is handled inside the on_finished callback in file_io.py.

        # Update scalar range if scalar mode is already active
        if self.current_color_mode == ColorMode.SCALAR and self.active_scalar_name:
            self._update_scalar_data_range()
            self.scalar_range_initialized = True
            if self.scalar_toolbar:
                self.scalar_toolbar.setVisible(True)

    @pyqtSlot()
    def _trigger_replace_bundle(self) -> None:
        """Wrapper to call load_streamlines_file with keep_image=True."""
        self.scalar_range_initialized = False
        if self.scalar_toolbar:
            self.scalar_toolbar.setVisible(False)
        self.bundle_is_visible = True

        file_io.load_streamlines_file(self, keep_image=True)

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

        # Priority A: Loaded Anatomical Image
        if self.anatomical_image_data is not None:
            affine = self.anatomical_image_affine
            shape = self.anatomical_image_data.shape[:3]

        # Priority B: Original Header Info (if compatible/available)
        elif self.original_trk_header:
            try:
                # Check for standard TRK header fields
                if (
                    "dimensions" in self.original_trk_header
                    and "voxel_to_rasmm" in self.original_trk_header
                ):
                    shape = tuple(
                        int(d) for d in self.original_trk_header["dimensions"][:3]
                    )
                    affine = self.original_trk_header["voxel_to_rasmm"]
            except (KeyError, ValueError, TypeError):
                logger.debug("Failed to parse TRK header for density map grid.")

        # Priority C: Compute Bounding Box (Fallback)
        # If no reference is found, we create a 1mm isotropic grid around the bundle
        if affine is None or shape is None:
            self.vtk_panel.update_status("Calculating density grid from bounds...")
            QApplication.processEvents()

            visible_streamlines = [
                self.tractogram_data[i]
                for i in self.visible_indices
                if self.tractogram_data[i] is not None
            ]

            if not visible_streamlines:
                QMessageBox.warning(self, "Error", "No visible streamlines to map.")
                return

            try:
                # Concatenate to find global bounds
                all_points = np.concatenate(visible_streamlines, axis=0)
                min_coord = np.min(all_points, axis=0)
                max_coord = np.max(all_points, axis=0)

                # Use 1mm isotropic resolution
                voxel_size = np.array([1.0, 1.0, 1.0])

                # Add padding (5mm)
                padding = 5.0
                min_coord -= padding
                max_coord += padding

                # Calculate shape
                dims = np.ceil((max_coord - min_coord) / voxel_size).astype(int)
                shape = tuple(dims)

                # Construct Affine (Translation + Scale)
                # Maps Voxel(0,0,0) -> World(min_coord)
                affine = np.eye(4)
                affine[:3, :3] = np.diag(voxel_size)
                affine[:3, 3] = min_coord

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
            visible_streamlines = [
                self.tractogram_data[i]
                for i in self.visible_indices
                if self.tractogram_data[i] is not None
                and len(self.tractogram_data[i]) > 0
            ]

            if not visible_streamlines:
                raise ValueError("No valid streamlines found in current view.")

            # Flatten to a single array of points (N, 3)
            points = np.concatenate(visible_streamlines, axis=0)

            # Transform World (RASmm) -> Voxel Coordinates
            inv_affine = np.linalg.inv(affine)
            vox_coords = nib.affines.apply_affine(inv_affine, points)

            # Round to nearest integer voxel index
            vox_indices = np.rint(vox_coords).astype(int)

            # Filter points outside the defined grid dimensions
            valid_mask = (
                (vox_indices[:, 0] >= 0)
                & (vox_indices[:, 0] < shape[0])
                & (vox_indices[:, 1] >= 0)
                & (vox_indices[:, 1] < shape[1])
                & (vox_indices[:, 2] >= 0)
                & (vox_indices[:, 2] < shape[2])
            )
            valid_voxels = vox_indices[valid_mask]

            # Binning (Histogram)
            density_data = np.zeros(shape, dtype=np.int32)

            # Fast unbuffered summation at coordinates
            np.add.at(
                density_data,
                (valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]),
                1,
            )

            # Save to Disk
            nifti_img = nib.Nifti1Image(density_data.astype(np.float32), affine)

            # Copy header info if possible (e.g. from anatomy) to preserve orientations
            if self.anatomical_image_path and self.anatomical_image_data is not None:
                try:
                    ref_img = nib.load(self.anatomical_image_path)
                    nifti_img.header.set_zooms(ref_img.header.get_zooms()[:3])
                    nifti_img.header.set_xyzt_units(*ref_img.header.get_xyzt_units())
                except (OSError, ValueError, KeyError):
                    logger.debug("Failed to copy header info from anatomical image.")

            nib.save(nifti_img, file_path)

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
        if not (self.tractogram_data or self.anatomical_image_data):
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
        if not (self.tractogram_data or self.anatomical_image_data):
            QMessageBox.warning(self, "Export Error", "No data loaded to export.")
            return

        # Determine default filename
        default_name = "visualization.html"
        if self.original_trk_path:
            base_name = os.path.splitext(os.path.basename(self.original_trk_path))[0]
            default_name = f"{base_name}_viewer.html"

        # Get save path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to HTML", default_name, "HTML Files (*.html)"
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
        """Triggers loading of an anatomical image using a background thread."""
        # Save path before clearing for file dialog start directory
        saved_image_path = self.anatomical_image_path
        saved_trk_path = self.original_trk_path

        if self.anatomical_image_data is not None:
            # Custom message box to enforce "Yes" on the Left and "No" on the Right
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Replace Image?")
            msg_box.setText("An anatomical image is already loaded.\nReplace it?")
            msg_box.setIcon(QMessageBox.Icon.Question)

            # Use ActionRole to prevent platform-specific reordering
            yes_btn = msg_box.addButton("Yes", QMessageBox.ButtonRole.ActionRole)
            no_btn = msg_box.addButton("No", QMessageBox.ButtonRole.ActionRole)

            msg_box.setDefaultButton(no_btn)
            msg_box.exec()

            if msg_box.clickedButton() != yes_btn:
                return
            else:
                self._trigger_clear_anatomical_image()  # Clear before loading new one

        # Clear ROIs if present
        if self.roi_layers:
            self._trigger_clear_all_rois(notify=False)

        # Reset all drawing modes to prevent stuck state after clearing ROIs
        self._reset_all_drawing_modes()

        # Get file path first (this is fast)
        file_filter = "NIfTI Image Files (*.nii *.nii.gz);;All Files (*.*)"
        start_dir = ""
        if saved_image_path:
            start_dir = os.path.dirname(saved_image_path)
        elif saved_trk_path:
            start_dir = os.path.dirname(saved_trk_path)

        input_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Anatomical Image File", start_dir, file_filter
        )

        if not input_path:
            if self.vtk_panel:
                self.vtk_panel.update_status("Anatomical image load cancelled.")
            return

        # Setup Progress Dialog (Modal)
        progress = QProgressDialog("Initializing...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Loading Image")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(350)

        # Apply theme-aware style
        progress.setStyleSheet(self.theme_manager.get_progress_dialog_style())

        progress.setValue(0)
        progress.show()

        # Create and configure thread
        loader_thread = file_io.AnatomicalImageLoaderThread(input_path)

        def on_progress(val, msg):
            progress.setValue(val)
            progress.setLabelText(msg)

        def on_error(msg):
            progress.cancel()
            QMessageBox.critical(self, "Load Error", f"Error loading image:\n{msg}")
            if self.vtk_panel:
                self.vtk_panel.update_status("Error loading image.")

        def on_finished(data):
            try:
                # Update progress to show we're creating VTK actors (this is the heavy part)
                progress.setLabelText("Creating slicer actors...")
                progress.setValue(95)
                QApplication.processEvents()  # Keep UI responsive

                self.anatomical_image_data = data["data"]
                self.anatomical_image_affine = data["affine"]
                self.anatomical_image_path = data["path"]
                self.anatomical_mmap_image = data.get("mmap_image")
                self.image_is_visible = True

                if self.vtk_panel:
                    self.vtk_panel.update_anatomical_slices()
                    QApplication.processEvents()  # Allow UI to update after heavy work
                    if self.vtk_panel.scene:
                        self.vtk_panel.scene.reset_camera()
                        self.vtk_panel.scene.reset_clipping_range()
                    if self.vtk_panel.render_window:
                        self.vtk_panel.render_window.Render()
                    self.vtk_panel.update_status(
                        f"Loaded: {os.path.basename(data['path'])}"
                    )

                self._update_bundle_info_display()
                self._update_action_states()
                self._update_data_panel_display()

                # Close progress dialog after ALL work is complete
                progress.close()

            except (RuntimeError, ValueError, AttributeError, KeyError) as e:
                logger.error(f"Error in on_finished: {e}", exc_info=True)
                progress.close()  # Ensure progress is closed on error
                QMessageBox.critical(self, "Load Error", f"Error finalizing load:\n{e}")

        # Connect Signals (QueuedConnection ensures VTK calls run on main thread)
        loader_thread.progress.connect(on_progress)
        loader_thread.error.connect(on_error)
        loader_thread.finished.connect(
            on_finished, type=Qt.ConnectionType.QueuedConnection
        )
        progress.canceled.connect(loader_thread.terminate)

        # Keep reference to prevent garbage collection
        self._image_loader_thread = loader_thread

        # Start
        loader_thread.start()

    # ROI Image Methods
    @pyqtSlot()
    def _trigger_load_roi(self) -> None:
        """
        Triggers loading of ROI image layer(s),
        by reorienting both the data and the affine.
        """
        if self.anatomical_image_data is None:
            QMessageBox.warning(
                self,
                "Load Error",
                "Please load a main anatomical image before adding an ROI layer.",
            )
            return

        if not self.anatomical_image_path:
            QMessageBox.warning(
                self,
                "Load Error",
                "Cannot find the path for the loaded anatomical image. Cannot re-orient.",
            )
            return

        # Call the plural function handling multiple files
        loaded_rois = file_io.load_roi_images(self)

        if not loaded_rois:
            return  # User cancelled or all failed

        # Iterate through every loaded ROI
        for _, _, roi_path in loaded_rois:

            if roi_path in self.roi_layers:
                QMessageBox.warning(
                    self,
                    "ROI Already Loaded",
                    f"The ROI from '{os.path.basename(roi_path)}' is already loaded.",
                )
                continue

            self.roi_visibility[roi_path] = True
            self.roi_opacities[roi_path] = 0.5  # Default ROI opacity

            if self.vtk_panel:
                self.vtk_panel.update_status(
                    f"Processing {os.path.basename(roi_path)}..."
                )
                QApplication.processEvents()

            try:
                # Load the main anatomical image object from its stored path
                from .file_io import _align_if_oblique

                anatomical_img = nib.load(self.anatomical_image_path)
                anatomical_img = _align_if_oblique(
                    anatomical_img, self.anatomical_image_path, None
                )

                # Load the ROI image object again to perform reorientation operations
                roi_img = nib.load(roi_path)
                roi_img = _align_if_oblique(roi_img, roi_path, None)

                # Ensure proper coordinate system alignment
                current_ornt = nib.io_orientation(roi_img.affine)
                target_ornt = nib.io_orientation(anatomical_img.affine)

                # Check if they match
                if not np.array_equal(current_ornt, target_ornt):
                    if self.vtk_panel:
                        current_axcodes = "".join(nib.aff2axcodes(roi_img.affine))
                        target_axcodes = "".join(nib.aff2axcodes(anatomical_img.affine))
                        self.vtk_panel.update_status(
                            f"Reorienting {os.path.basename(roi_path)} ({current_axcodes} -> {target_axcodes})..."
                        )

                    # Get the transform
                    transform = ornt_transform(current_ornt, target_ornt)

                    # Use as_reoriented to transform *both* data and affine
                    reoriented_roi_img = roi_img.as_reoriented(transform)

                    # Get the *new* data and *new* affine
                    roi_data = reoriented_roi_img.get_fdata()
                    roi_affine = reoriented_roi_img.affine

                else:
                    # Orientations match, just get the original data and affine
                    roi_data = roi_img.get_fdata()
                    roi_affine = roi_img.affine

                # Store the data
                inv_affine = np.linalg.inv(roi_affine)
                main_affine = self.anatomical_image_affine
                T_main_to_roi = np.dot(inv_affine, main_affine)

                self.roi_layers[roi_path] = {
                    "data": roi_data,
                    "affine": roi_affine,
                    "path": roi_path,
                    "inv_affine": inv_affine,
                    "T_main_to_roi": T_main_to_roi,
                }

                self.roi_visibility[roi_path] = True

                # Tell VTK panel to create and add the new actors
                if self.vtk_panel:
                    self.vtk_panel.add_roi_layer(roi_path, roi_data, roi_affine)
                    self.vtk_panel.update_status(
                        f"Aligned and added {os.path.basename(roi_path)}."
                    )

            except FileNotFoundError as e:
                logger.error(f"File not found during processing: {e}")
                continue
            except np.linalg.LinAlgError:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Could not invert affine matrix for {os.path.basename(roi_path)}.",
                )
                if roi_path in self.roi_visibility:
                    del self.roi_visibility[roi_path]
                continue

        # Final UI Updates after loop
        self._update_bundle_info_display()
        self._update_action_states()
        self._update_data_panel_display()
        if self.vtk_panel:
            self.vtk_panel.update_status("ROI loading complete.")

    @pyqtSlot()
    def _trigger_clear_all_rois(self, notify: bool = True) -> None:
        """Clears all loaded ROI image layers and resets logic filters."""
        if not self.roi_layers:
            return

        self._reset_all_drawing_modes()

        self.roi_layers.clear()
        self.roi_visibility.clear()

        self.roi_states.clear()
        self.roi_intersection_cache.clear()
        self.roi_highlight_indices.clear()

        self._update_roi_visual_selection()
        self.roi_manager._apply_filters_with_skip_protection()

        if self.vtk_panel:
            self.vtk_panel.clear_all_roi_layers()
            if notify:
                self.vtk_panel.update_status("All ROI layers and filters cleared.")

        self._update_data_panel_display()
        self._update_bundle_info_display()
        self._update_action_states()

    @pyqtSlot()
    def _trigger_clear_all_data(self) -> None:
        """Clears all loaded data (streamlines, anatomical image, ROIs, parcellation) without confirmation."""
        has_data = (
            self.tractogram_data is not None
            or self.anatomical_image_data is not None
            or bool(self.roi_layers)
            or self.parcellation_data is not None
        )

        if not has_data:
            return

        # Clear ROIs first
        if self.roi_layers:
            self._trigger_clear_all_rois(notify=False)

        # Clear Parcellation overlay and data
        if self.parcellation_data is not None:
            self._clear_parcellation()

        # Clear Streamlines
        if self.tractogram_data is not None:
            self._close_bundle()

        # Clear Image (if not already cleared by _close_bundle or if no bundle was loaded)
        if self.anatomical_image_data is not None:
            self._trigger_clear_anatomical_image()

        # Reset all drawing modes to prevent stuck state after clearing
        self._reset_all_drawing_modes()

        # Final UI update
        self._update_data_panel_display()
        self.vtk_panel.update_status("All data cleared.")

    @pyqtSlot(QTreeWidgetItem, int)
    def _on_data_panel_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Handles item changes. Delegates to DataPanelManager."""
        self.data_panel_manager.on_data_panel_item_changed(item, column)

    @pyqtSlot("QPoint")
    def _on_data_panel_context_menu(self, position) -> None:
        """Shows context menu. Delegates to DataPanelManager."""
        self.data_panel_manager.on_data_panel_context_menu(position)

    def _set_roi_logic_mode(self, roi_path: str, mode: str) -> None:
        """Sets ROI logic mode. Delegates to ROIManager."""
        self.roi_manager.set_roi_logic_mode(roi_path, mode)

    @pyqtSlot(bool)
    def _toggle_image_visibility(self, visible: bool) -> None:
        """Toggles the visibility of the anatomical image slices."""
        if self.image_is_visible == visible:
            return

        self.image_is_visible = visible
        if self.vtk_panel:
            self.vtk_panel.set_anatomical_slice_visibility(visible)
            self.vtk_panel.update_status(f"Image visibility set to {visible}")

    def _compute_roi_intersection(self, roi_path: str) -> bool:
        """Computes ROI intersection. Delegates to ROIManager."""
        return self.roi_manager.compute_roi_intersection(roi_path)

    def update_sphere_roi_intersection(
        self, roi_name: str, center: np.ndarray, radius: float
    ) -> None:
        """Updates sphere ROI intersection. Delegates to ROIManager."""
        self.roi_manager.update_sphere_roi_intersection(roi_name, center, radius)

    def update_rectangle_roi_intersection(
        self, roi_name: str, min_point: np.ndarray, max_point: np.ndarray
    ) -> None:
        """Updates rectangle ROI intersection. Delegates to ROIManager."""
        self.roi_manager.update_rectangle_roi_intersection(
            roi_name, min_point, max_point
        )

    def _update_roi_visual_selection(self) -> None:
        """Updates ROI visual selection. Delegates to ROIManager."""
        self.roi_manager.update_roi_visual_selection()

    def _apply_logic_filters(self) -> None:
        """Applies logic filters. Delegates to ROIManager."""
        self.roi_manager.apply_logic_filters()

    def _change_roi_color_action(self, path: str) -> None:
        """Changes ROI color. Delegates to ROIManager."""
        self.roi_manager.change_roi_color_action(path)

    def _rename_roi_action(self, old_path: str) -> None:
        """Renames ROI. Delegates to ROIManager."""
        self.roi_manager.rename_roi_action(old_path)

    def _save_roi_action(self, roi_path: str) -> None:
        """Saves ROI. Delegates to ROIManager."""
        self.roi_manager.save_roi_action(roi_path)

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

            if self.vtk_panel.render_window:
                self.vtk_panel.render_window.Render()

        self.vtk_panel.update_status(f"Bundle visibility set to {visible}")

    @pyqtSlot(str, bool)
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

            if self.vtk_panel.render_window:
                self.vtk_panel.render_window.Render()

        self._update_bundle_info_display()
        self._update_action_states()

    # Helper functions for float <-> int mapping
    def _float_to_int_slider(self, float_val: float) -> int:
        """Delegates to ScalarManager."""
        return self.scalar_manager.float_to_int_slider(float_val)

    def _int_slider_to_float(self, slider_val: int) -> float:
        """Delegates to ScalarManager."""
        return self.scalar_manager.int_slider_to_float(slider_val)

    def _update_scalar_data_range(self) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.update_scalar_data_range()

    def _update_scalar_range_widgets(self) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.update_scalar_range_widgets()

    @pyqtSlot(int)
    def _slider_value_changed(self, slider_val: int) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.slider_value_changed(slider_val)

    @pyqtSlot()
    def _spinbox_value_changed(self) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.spinbox_value_changed()

    @pyqtSlot()
    def _reset_scalar_range(self) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.reset_scalar_range()

    @pyqtSlot()
    def _trigger_vtk_update(self) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.trigger_vtk_update()

    # Slot for VTK Panel
    def update_ras_coordinate_display(self, ras_coords: Optional[np.ndarray]) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.update_ras_coordinate_display(ras_coords)

    @pyqtSlot()
    def _on_ras_coordinate_entered(self) -> None:
        """Delegates to ScalarManager."""
        self.scalar_manager.on_ras_coordinate_entered()

    # Window Close Event
    def closeEvent(self, event: QCloseEvent) -> None:
        """Handles the main window close event, prompting if data is loaded."""
        logger.info("Close event received.")
        # Explicitly check for None to avoid ValueError with numpy arrays
        data_loaded = (self.tractogram_data is not None) or (
            self.anatomical_image_data is not None
        )
        prompt_message = "Data (streamlines and/or image) is currently loaded.\nAre you sure you want to quit?"

        should_exit = False

        if data_loaded:
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
                self._cleanup_resources()
                self._cleanup_vtk()
                event.accept()
                should_exit = True
            else:
                logger.info("User cancelled quit.")
                event.ignore()
        else:
            logger.info("No data loaded. Cleaning up...")
            self._cleanup_resources()
            self._cleanup_vtk()
            event.accept()
            should_exit = True

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
            try:
                if hasattr(self.trx_file_reference, "close"):
                    self.trx_file_reference.close()
            except OSError:
                logger.debug("Failed to close TRX file reference.")
            self.trx_file_reference = None

    def _cleanup_resources(self) -> None:
        """Cleans up non-VTK resources before application exit."""
        # Terminate any running background threads
        for thread_attr in ("_loader_thread", "_image_loader_thread"):
            thread = getattr(self, thread_attr, None)
            if thread is not None:
                try:
                    if thread.isRunning():
                        logger.info(f"Terminating {thread_attr}...")
                        thread.terminate()
                        thread.wait(1000)  # Wait up to 1 second
                except (RuntimeError, AttributeError) as e:
                    logger.warning(f"Error terminating {thread_attr}: {e}")

        # Clear memory-mapped image cache
        if self.anatomical_mmap_image is not None:
            try:
                self.anatomical_mmap_image.clear_cache()
            except (AttributeError, OSError):
                logger.debug("Failed to clear memory-mapped image cache.")
            self.anatomical_mmap_image = None

        # Close TRX memmap file reference (releases temp directory)
        self._close_trx_file()

        # Clear large data arrays to help garbage collection
        self.tractogram_data = None
        self.streamline_bboxes = None
        self.anatomical_image_data = None
        self.odf_data = None
        self.parcellation_data = None

    def _cleanup_vtk(self) -> None:
        """Safely cleans up VTK resources for all 4 views."""
        if not self.vtk_panel:
            return

        panel = self.vtk_panel

        # Disable and release orientation widget
        if hasattr(panel, "orientation_widget") and panel.orientation_widget:
            try:
                panel.orientation_widget.SetEnabled(0)
                panel.orientation_widget.SetInteractor(None)
                panel.orientation_widget = None
            except RuntimeError:
                logger.debug("Failed to release orientation widget.")

        # Clear all scenes
        scenes_to_clear = [
            panel.scene,
            getattr(panel, "axial_scene", None),
            getattr(panel, "coronal_scene", None),
            getattr(panel, "sagittal_scene", None),
        ]
        for scene in scenes_to_clear:
            if scene:
                try:
                    scene.clear()
                except RuntimeError:
                    logger.debug("Failed to clear VTK scene during cleanup.")

        # Remove all observers and terminate all interactors
        interactors = [
            getattr(panel, "interactor", None),
            getattr(panel, "axial_interactor", None),
            getattr(panel, "coronal_interactor", None),
            getattr(panel, "sagittal_interactor", None),
        ]
        for interactor in interactors:
            if interactor:
                try:
                    interactor.RemoveAllObservers()
                    if interactor.GetInitialized():
                        interactor.TerminateApp()
                except RuntimeError:
                    logger.debug("Failed to clean up VTK interactor.")

        # Close QVTKRenderWindowInteractor widgets
        qt_vtk_widgets = [
            getattr(panel, "vtk_widget", None),
            getattr(panel, "axial_vtk_widget", None),
            getattr(panel, "coronal_vtk_widget", None),
            getattr(panel, "sagittal_vtk_widget", None),
        ]
        for widget in qt_vtk_widgets:
            if widget:
                try:
                    rw = widget.GetRenderWindow()
                    if rw:
                        rw.Finalize()
                    widget.close()
                except RuntimeError:
                    logger.debug("Failed to finalize VTK render window.")

        # Set references to None to help garbage collection
        panel.scene = None
        panel.axial_scene = None
        panel.coronal_scene = None
        panel.sagittal_scene = None
        panel.interactor = None
        panel.axial_interactor = None
        panel.coronal_interactor = None
        panel.sagittal_interactor = None
        panel.render_window = None
        panel.axial_render_window = None
        panel.coronal_render_window = None
        panel.sagittal_render_window = None
        panel.vtk_widget = None
        panel.axial_vtk_widget = None
        panel.coronal_vtk_widget = None
        panel.sagittal_vtk_widget = None

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
