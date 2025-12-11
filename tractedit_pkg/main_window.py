# -*- coding: utf-8 -*-

"""
Contains the MainWindow class for the tractedit GUI application.

Handles the main application window, menus, actions, status bar,
and coordinates interactions between UI elements, data state,
file I/O, and the VTK panel.
"""

import os
import numpy as np
from typing import Optional, List, Set, Dict, Any
import nibabel as nib
import logging 

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QMenuBar, QFileDialog,
    QMessageBox, QLabel, QStatusBar, QApplication,
    QToolBar, QDoubleSpinBox, QSpinBox,
    QSlider, 
    QHBoxLayout, QSpacerItem, QSizePolicy,
    QDockWidget, QTreeWidget, QTreeWidgetItem, QStyle, 
    QLineEdit, QMenu, QColorDialog, QCheckBox
)
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup, QIcon, QCloseEvent, QPixmap
from PyQt6.QtCore import Qt, pyqtSlot

from . import file_io 
from . import odf_utils 
from .utils import (
    ColorMode, get_formatted_datetime, get_asset_path, format_tuple,
    MAX_STACK_LEVELS, DEFAULT_SELECTION_RADIUS, MIN_SELECTION_RADIUS,
    RADIUS_INCREMENT
)
from .vtk_panel import VTKPanel
from nibabel.processing import resample_from_to
from nibabel.orientations import ornt_transform, apply_orientation, io_orientation

logger = logging.getLogger(__name__)

# Constant for slider precision 
SLIDER_PRECISION = 1000 # Use 1000 steps for the slider

# Main GUI class
class MainWindow(QMainWindow):
    """
    Main application window for TractEdit.
    Sets up the UI, manages application state (streamlines, selection, undo/redo),
    and delegates rendering/interaction to VTKPanel and file I/O to file_io.
    """
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Initialize Streamline Data Variables
        self.tractogram_data: Optional['nib.streamlines.ArraySequence'] = None
        self.streamline_bboxes: Optional[np.ndarray] = None 
        self.visible_indices: Set[int] = set()
        self.original_trk_header: Optional[Dict[str, Any]] = None # Header dict from loaded file
        self.original_trk_affine: Optional[np.ndarray] = None # Affine matrix (affine_to_rasmm)
        self.original_trk_path: Optional[str] = None # Full path
        self.original_file_extension: Optional[str] = None # '.trk', '.tck', '.trx', or None
        self.scalar_data_per_point: Optional[Dict[str, 'nib.streamlines.ArraySequence']] = None # Dictionary: {scalar_name: [scalar_array_sl0, ...]}
        self.active_scalar_name: Optional[str] = None # Key for the currently active scalar
        self.selected_streamline_indices: Set[int] = set() # Indices of selected streamlines
        self.selection_radius_3d: float = DEFAULT_SELECTION_RADIUS # Radius for sphere selection
        self.render_stride: int = 1  # 1 = Show all, 100 = Show 1%
        self.bundle_opacity: float = 1.0  # Default to 1.0
        self.image_opacity: float = 1.0
        self.roi_opacities: Dict[str, float] = {}

        # Initialize Anatomical Image Data Variables
        self.anatomical_image_path: Optional[str] = None
        self.anatomical_image_data: Optional[np.ndarray] = None # Numpy array
        self.anatomical_image_affine: Optional[np.ndarray] = None # 4x4 numpy array

        # Undo/Redo Stacks
        self.undo_stack: List[Set[int]] = []
        self.redo_stack: List[Set[int]] = []

        # View State
        self.current_color_mode: ColorMode = ColorMode.DEFAULT
        self.bundle_is_visible: bool = True         
        self.image_is_visible: bool = True       
        self.roi_visibility: Dict[str, bool] = {} 
        self.render_as_tubes: bool = False  # False = Lines, True = Tubes     

        # Scalar Range Variables 
        self.scalar_min_val: float = 0.0            # Current min value for the colormap
        self.scalar_max_val: float = 1.0            # Current max value for the colormap
        self.scalar_data_min: float = 0.0           # Actual min value in the loaded data
        self.scalar_data_max: float = 1.0           # Actual max value in the loaded data
        self.scalar_range_initialized: bool = False # Flag to check if range has been calculated
        self.scalar_toolbar: Optional[QToolBar] = None          
        self.scalar_min_spinbox: Optional[QDoubleSpinBox] = None
        self.scalar_max_spinbox: Optional[QDoubleSpinBox] = None
        self.scalar_min_slider: Optional[QSlider] = None
        self.scalar_max_slider: Optional[QSlider] = None
        
        # ODF / Glyphs Data 
        self.odf_data: Optional[np.ndarray] = None
        self.odf_affine: Optional[np.ndarray] = None
        self.odf_sh_order: int = 0
        self.odf_sphere = None
        self.odf_basis_matrix = None
        self.MAX_ODF_STREAMLINES = 26000 # Safety limit for Tunnel View

        # Data Panel / Dock Widget 
        self.data_dock_widget: Optional[QDockWidget] = None
        self.data_tree_widget: Optional[QTreeWidget] = None
        
        # ROI Layer Data Variables
        self.roi_layers: Dict[str, Dict[str, Any]] = {} # Key: path, Val: {'data':, 'affine':, 'inv_affine':}
        
        # Status Bar Widgets 
        self.permanent_status_widget: Optional[QWidget] = None
        self.data_info_label: Optional[QLabel] = None
        self.ras_coordinate_label: Optional[QLabel] = None
        
        # ROI Logic State
        self.roi_states: Dict[str, Dict[str, bool]] = {} 
        self.roi_intersection_cache: Dict[str, Set[int]] = {}
        self.roi_highlight_indices: Set[int] = set()
        self.manual_visible_indices: Set[int] = set() # Tracks manual deletions separate from filters
        
        # Caching intersections to avoid re-calculating on every click
        self.roi_intersection_cache: Dict[str, Set[int]] = {}
        
        # Set of indices specifically highlighted in RED (ROI Selection)
        self.roi_highlight_indices: Set[int] = set()

        # Window Properties 
        self.setWindowTitle("TractEdit GUI (PyQt6) - Interactive trk/tck/trx Editor")
        self.setGeometry(100, 100, 1100, 850)

        # Setup UI Components   
        self._create_actions()
        self._create_menus()
        self._create_main_toolbar() 
        self._create_scalar_toolbar() 
        self._create_data_panel()  
        self._setup_status_bar()
        self._setup_central_widget() # This creates the VTKPanel

        # Initial Status Update
        self._update_initial_status()
        self._update_action_states()
        self._update_bundle_info_display()

    def _create_actions(self) -> None:
        """Creates QAction objects used in menus and potentially toolbars."""

        # File Actions 
        self.load_file_action: QAction = QAction("&Load trk/tck/trx...", self)
        self.load_file_action.setStatusTip("Load a trk, tck or trx streamline file")
        self.load_file_action.triggered.connect(self._trigger_load_streamlines)

        # Load Anatomical Image Action 
        self.load_bg_image_action: QAction = QAction("Load &Image...", self)
        self.load_bg_image_action.setStatusTip("Load a NIfTI image (.nii, .nii.gz) as background")
        self.load_bg_image_action.triggered.connect(self._trigger_load_anatomical_image)
        self.load_bg_image_action.setEnabled(False) 
        
        # ODF Actions
        self.load_odf_action: QAction = QAction("Load &ODF (SH)...", self)
        self.load_odf_action.setStatusTip("Load a NIfTI file containing Spherical Harmonics coefficients")
        self.load_odf_action.triggered.connect(self._trigger_load_odf)
        self.view_odf_tunnel_action: QAction = QAction("Show ODF &Tunnel", self)
        self.view_odf_tunnel_action.setStatusTip(f"Show ODF glyphs masked by current bundle (< {self.MAX_ODF_STREAMLINES} fibers)")
        self.view_odf_tunnel_action.setCheckable(True)
        self.view_odf_tunnel_action.triggered.connect(self._toggle_odf_tunnel)
        self.view_odf_tunnel_action.setEnabled(False)

        # Close Bundle Action
        self.close_bundle_action: QAction = QAction("&Close Bundle", self)
        self.close_bundle_action.setStatusTip("Close the current streamline bundle") # Updated tip
        self.close_bundle_action.triggered.connect(self._close_bundle)
        self.close_bundle_action.setEnabled(False)

        # Clear Anatomical Image Action
        self.clear_bg_image_action: QAction = QAction("Clear Anatomical Image", self)
        self.clear_bg_image_action.setStatusTip("Remove the background anatomical image")
        self.clear_bg_image_action.triggered.connect(self._trigger_clear_anatomical_image)
        self.clear_bg_image_action.setEnabled(False) # Enabled only when image loaded
        
        # Load ROI Action 
        self.load_roi_action: QAction = QAction("Load &ROI...", self)
        self.load_roi_action.setStatusTip("Load a NIfTI image (.nii, .nii.gz) as an ROI overlay")
        self.load_roi_action.triggered.connect(self._trigger_load_roi)
        self.load_roi_action.setEnabled(False) # Will be enabled when a main image is loaded

        # Clear All ROIs Action
        self.clear_all_rois_action: QAction = QAction("Clear All ROIs", self)
        self.clear_all_rois_action.setStatusTip("Remove all loaded ROI overlays")
        self.clear_all_rois_action.triggered.connect(self._trigger_clear_all_rois)
        self.clear_all_rois_action.setEnabled(False) # Will be enabled when ROIs are loaded
        
        # Clear All Data Action
        self.clear_all_data_action: QAction = QAction("Clear &All", self)
        self.clear_all_data_action.setStatusTip("Clear all loaded data (Streamlines, Image, ROIs)")
        self.clear_all_data_action.triggered.connect(self._trigger_clear_all_data)
        self.clear_all_data_action.setEnabled(False)

        # Save Streamlines Action
        self.save_file_action: QAction = QAction("&Save As...", self)
        self.save_file_action.setStatusTip("Save the modified streamlines to a trk, tck or trx file")
        self.save_file_action.triggered.connect(self._trigger_save_streamlines)
        self.save_file_action.setEnabled(False)

        # Screenshot Action
        self.screenshot_action: QAction = QAction("Save &Screenshot", self)
        self.screenshot_action.setStatusTip("Save a screenshot of the current view (bundle and image)")
        self.screenshot_action.setShortcut("Ctrl+P")
        self.screenshot_action.triggered.connect(self._trigger_screenshot)
        self.screenshot_action.setEnabled(False)
        
        # Calculate Centroid Action
        self.calc_centroid_action: QAction = QAction("Calculate &Centroid", self)
        self.calc_centroid_action.setStatusTip("Calculate and save the centroid (mean) of the current bundle")
        self.calc_centroid_action.triggered.connect(self._trigger_calculate_centroid)
        self.calc_centroid_action.setEnabled(False)

        # Calculate Medoid Action
        self.calc_medoid_action: QAction = QAction("Calculate &Medoid", self)
        self.calc_medoid_action.setStatusTip("Calculate and save the medoid (geometric median) of the current bundle")
        self.calc_medoid_action.triggered.connect(self._trigger_calculate_medoid)
        self.calc_medoid_action.setEnabled(False)

        # Exit Action
        self.exit_action: QAction = QAction("&Exit", self)
        self.exit_action.setStatusTip("Exit the application")
        self.exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self.exit_action.triggered.connect(self.close)
        
        # Save Density Map Action
        self.save_density_map_action: QAction = QAction("Save &Density Map...", self)
        self.save_density_map_action.setStatusTip("Generate and save a density map (TDI) of the current visible bundle")
        self.save_density_map_action.triggered.connect(self._trigger_save_density_map)
        self.save_density_map_action.setEnabled(False)
        
        # Geometry Lines/Tubes
        self.geometry_action_group: QActionGroup = QActionGroup(self)
        self.geometry_action_group.setExclusive(True)

        self.geo_lines_action: QAction = QAction("Render as &Lines", self)
        self.geo_lines_action.setStatusTip("Render streamlines as simple lines (Faster)")
        self.geo_lines_action.setCheckable(True)
        self.geo_lines_action.setChecked(True)
        self.geo_lines_action.triggered.connect(lambda: self._set_geometry_mode(as_tubes=False))
        self.geometry_action_group.addAction(self.geo_lines_action)
        self.geo_lines_action.setEnabled(False) # Enabled when data loads

        self.geo_tubes_action: QAction = QAction("Render as &Tubes", self)
        self.geo_tubes_action.setStatusTip("Render streamlines as 3D tubes (Slower, High Quality)")
        self.geo_tubes_action.setCheckable(True)
        self.geo_tubes_action.triggered.connect(lambda: self._set_geometry_mode(as_tubes=True))
        self.geometry_action_group.addAction(self.geo_tubes_action)
        self.geo_tubes_action.setEnabled(False) # Enabled when data loads

        # Edit Actions 
        self.undo_action: QAction = QAction("&Undo", self)
        self.undo_action.setStatusTip("Undo the last deletion")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo) # Ctrl+Z
        self.undo_action.triggered.connect(self._perform_undo)
        self.undo_action.setEnabled(False)

        self.redo_action: QAction = QAction("&Redo", self)
        self.redo_action.setStatusTip("Redo the last undone deletion")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo) # Ctrl+Y / Ctrl+Shift+Z
        self.redo_action.triggered.connect(self._perform_redo)
        self.redo_action.setEnabled(False)

        # View Actions (Coloring)
        self.coloring_action_group: QActionGroup = QActionGroup(self)
        self.coloring_action_group.setExclusive(True)

        self.color_default_action: QAction = QAction("&Greyscale Color", self)
        self.color_default_action.setStatusTip("Color streamlines with greyscale")
        self.color_default_action.setCheckable(True)
        self.color_default_action.setChecked(False) 
        self.color_default_action.triggered.connect(lambda: self._set_color_mode(ColorMode.DEFAULT))
        self.coloring_action_group.addAction(self.color_default_action)
        self.color_default_action.setEnabled(False)

        self.color_orientation_action: QAction = QAction("Color by Orientation", self)
        self.color_orientation_action.setStatusTip("Color streamlines by local orientation (RGB)")
        self.color_orientation_action.setCheckable(True)
        self.color_orientation_action.setChecked(True) # Default selection
        self.color_orientation_action.triggered.connect(lambda: self._set_color_mode(ColorMode.ORIENTATION))
        self.coloring_action_group.addAction(self.color_orientation_action)
        self.color_orientation_action.setEnabled(False)

        self.color_scalar_action: QAction = QAction("Color by Scalar", self)
        self.color_scalar_action.setStatusTip("Color streamlines by the first loaded scalar value per point")
        self.color_scalar_action.setCheckable(True)
        self.color_scalar_action.triggered.connect(lambda: self._set_color_mode(ColorMode.SCALAR))
        self.coloring_action_group.addAction(self.color_scalar_action)
        self.color_scalar_action.setEnabled(False)

        # Command Actions
        self.clear_select_action: QAction = QAction("&Clear Selection", self)
        self.clear_select_action.setStatusTip("Clear the current streamline selection (C)")
        self.clear_select_action.setShortcut("C")
        self.clear_select_action.triggered.connect(self._perform_clear_selection)
        self.clear_select_action.setEnabled(False)

        # Delete Selection Action
        self.delete_select_action: QAction = QAction("&Delete Selection", self)
        self.delete_select_action.setStatusTip("Delete the selected streamlines (D)")
        self.delete_select_action.setShortcut("D")
        self.delete_select_action.triggered.connect(self._perform_delete_selection)
        self.delete_select_action.setEnabled(False)
        
        # Reset Camera Action
        self.reset_camera_action: QAction = QAction("&Reset Camera", self)
        self.reset_camera_action.setStatusTip("Reset the 3D camera view")
        self.reset_camera_action.triggered.connect(self._perform_reset_camera)
        self.reset_camera_action.setEnabled(True)

        # Increase/Decrease Radius Actions
        self.increase_radius_action: QAction = QAction("&Increase Radius", self)
        self.increase_radius_action.setStatusTip(f"Increase the selection sphere radius (+{RADIUS_INCREMENT}mm)")
        self.increase_radius_action.setShortcut("+")
        self.increase_radius_action.triggered.connect(self._increase_radius)
        self.increase_radius_action.setEnabled(False)

        self.decrease_radius_action: QAction = QAction("&Decrease Radius", self)
        self.decrease_radius_action.setStatusTip(f"Decrease the selection sphere radius (-{RADIUS_INCREMENT}mm)")
        self.decrease_radius_action.setShortcut("-")
        self.decrease_radius_action.triggered.connect(self._decrease_radius)
        self.decrease_radius_action.setEnabled(False)

        self.hide_sphere_action: QAction = QAction("&Hide Selection Sphere", self)
        self.hide_sphere_action.setStatusTip("Hide the blue selection sphere (Esc)")
        self.hide_sphere_action.setShortcut("Esc")
        self.hide_sphere_action.triggered.connect(self._hide_sphere)
        self.hide_sphere_action.setEnabled(True)

        # Help Menu
        self.about_action: QAction = QAction("&About TractEdit...", self)
        self.about_action.setStatusTip("Show information about TractEdit")
        self.about_action.triggered.connect(self._show_about_dialog)

    def _create_menus(self) -> None:
        """Creates the main menu bar and populates it with actions."""
        main_bar: QMenuBar = self.menuBar()

        # File Menu 
        file_menu = main_bar.addMenu("&File")
        file_menu.addAction(self.load_file_action)      # Load streamlines
        file_menu.addAction(self.load_bg_image_action)  # Load image
        file_menu.addAction(self.load_roi_action)       # Load ROI
        file_menu.addAction(self.load_odf_action)       # Load ODF
        
        file_menu.addSeparator()
        file_menu.addAction(self.calc_centroid_action)
        file_menu.addAction(self.calc_medoid_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_bundle_action)   # Close streamlines
        file_menu.addAction(self.clear_bg_image_action) # Clear image 
        file_menu.addAction(self.clear_all_rois_action)
        file_menu.addAction(self.clear_all_data_action) 
        file_menu.addSeparator()
        file_menu.addAction(self.save_file_action)        # Save streamlines
        file_menu.addAction(self.save_density_map_action) # Save density map
        file_menu.addAction(self.screenshot_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        # Edit Menu
        edit_menu = main_bar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        # View Menu
        view_menu = main_bar.addMenu("&View")
        
        # Color Sub-menu
        color_menu = view_menu.addMenu("Streamline Color")
        color_menu.addAction(self.color_default_action)
        color_menu.addAction(self.color_orientation_action)
        color_menu.addAction(self.color_scalar_action)
        
        # Geometry Sub-menu
        geo_menu = view_menu.addMenu("Streamline &Geometry")
        geo_menu.addAction(self.geo_lines_action)
        geo_menu.addAction(self.geo_tubes_action)

        # ODF Tunnel View 
        view_menu.addSeparator()
        view_menu.addAction(self.view_odf_tunnel_action)

        # Dock Panel Toggle
        if self.data_dock_widget:
            self.toggle_data_panel_action = self.data_dock_widget.toggleViewAction()
            self.toggle_data_panel_action.setText("Data Panel")
            self.toggle_data_panel_action.setStatusTip("Show/Hide the Data Panel")
            view_menu.addSeparator()
            view_menu.addAction(self.toggle_data_panel_action)

        # Commands Menu
        commands_menu = main_bar.addMenu("&Commands")
        commands_menu.addAction(self.reset_camera_action) 
        commands_menu.addAction(self.screenshot_action)   
        commands_menu.addSeparator()
        commands_menu.addAction(self.clear_select_action)
        commands_menu.addAction(self.delete_select_action)
        commands_menu.addSeparator()
        commands_menu.addAction(self.increase_radius_action)
        commands_menu.addAction(self.decrease_radius_action)
        commands_menu.addSeparator()
        commands_menu.addAction(self.hide_sphere_action)

        # Help Menu
        help_menu = main_bar.addMenu("&Help")
        help_menu.addAction(self.about_action)
        
        # Shortcuts Submenu List 
        shortcuts_menu = help_menu.addMenu("Keyboard &Shortcuts")
        
        # Helper to add static text items
        def add_shortcut_item(text):
            act = QAction(text, self)
            act.setEnabled(False) # Disabled so it acts as a static label
            shortcuts_menu.addAction(act)

        # Selection Group
        shortcuts_menu.addSection("Selection Tools")
        add_shortcut_item("s  :  Select/Deselect at cursor")
        add_shortcut_item("d  :  Delete selection")
        add_shortcut_item("c  :  Clear selection")
        add_shortcut_item("+ / =  :  Increase sphere radius")
        add_shortcut_item("-  :  Decrease sphereradius")

        # Navigation Group
        shortcuts_menu.addSection("Slice Navigation")
        add_shortcut_item("↑ / ↓  :  Axial (Z-axis)")
        add_shortcut_item("← / →  :  Sagittal (X-axis)")
        add_shortcut_item("Ctrl + ↑ / ↓  :  Coronal (Y-axis)")

        # General Group
        shortcuts_menu.addSection("General")
        add_shortcut_item("Ctrl + S  :  Save As")
        add_shortcut_item("Ctrl + Z  :  Undo")
        add_shortcut_item("Ctrl + Y  :  Redo")
        add_shortcut_item("Ctrl + P  :  Screenshot")
        add_shortcut_item("Ctrl + Q  :  Quit")

        help_menu.addSeparator()
        help_menu.addAction(self.about_action)
        
        
    # Main Toolbar for Skip 
    def _create_main_toolbar(self) -> None:
        self.main_toolbar = QToolBar("Main Tools", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.main_toolbar)
        
        # Container for Toolbar Widgets 
        container = QWidget()
        l = QHBoxLayout(container)
        l.setContentsMargins(0, 0, 0, 0)
        
        # Skip / Density Control
        self.skip_checkbox = QCheckBox()
        self.skip_checkbox.setChecked(False) 
        self.skip_checkbox.toggled.connect(self._on_skip_toggled)
        l.addWidget(self.skip_checkbox)
        
        l.addWidget(QLabel("Skip %: "))
        self.skip_spinbox = QSpinBox()
        self.skip_spinbox.setRange(0, 99) 
        self.skip_spinbox.setValue(0)
        self.skip_spinbox.setToolTip("Percentage of streamlines to skip for rendering (0 = Show All, 99 = Show 1%)")
        self.skip_spinbox.setEnabled(False) 
        self.skip_spinbox.editingFinished.connect(self._on_skip_changed)
        l.addWidget(self.skip_spinbox)
        
        # Spacer
        l.addSpacing(20)
        l.addWidget(QLabel("|"))
        l.addSpacing(20)

        # Opacity Control
        self.opacity_label = QLabel("Opacity: ")
        l.addWidget(self.opacity_label)
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100) # 0 to 1.0
        self.opacity_slider.setValue(100)
        self.opacity_slider.setFixedWidth(120) # Fixed width for toolbar layout
        self.opacity_slider.setEnabled(False) # Disabled until selection
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        l.addWidget(self.opacity_slider)
        
        # Add stretch to push items to the left
        l.addStretch()
        
        # Add container to toolbar
        self.main_toolbar.addWidget(container)
        
    def _on_skip_toggled(self, checked: bool) -> None:
        """Enables/Disables the skip feature and resets view if turned off."""
        self.skip_spinbox.setEnabled(checked)
        
        if checked:
            # If enabled, apply whatever value is currently in the spinbox
            self._on_skip_changed()
        else:
            # If disabled, force reset to 0% skip (Show All)
            self.render_stride = 1
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
        
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
        Automatically sets the skip percentage based on the number of streamlines.
        Target: Render approximately 20,000 streamlines for optimal performance.
        """
        if not self.tractogram_data:
            return

        total_fibers = len(self.tractogram_data)
        TARGET_RENDER_COUNT = 20000 # Target render

        # Block signals to prevent VTK updates while adjusting widgets
        self.skip_checkbox.blockSignals(True)
        self.skip_spinbox.blockSignals(True)

        if total_fibers > TARGET_RENDER_COUNT:
            # Calculate how many we want to KEEP (ratio)
            keep_ratio = TARGET_RENDER_COUNT / total_fibers
            
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
                self.vtk_panel.update_status(f"Auto-Skip: {skip_percent}% skipped for performance.")
        else:
            # Bundle is small enough, show all
            self.skip_checkbox.setChecked(False)
            self.skip_spinbox.setEnabled(False)
            self.skip_spinbox.setValue(0)
            self.render_stride = 1

        # Unblock signals
        self.skip_checkbox.blockSignals(False)
        self.skip_spinbox.blockSignals(False)

        # Trigger Visual Update
        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()

    # Scalar Toolbar 
    def _create_scalar_toolbar(self) -> None:
        """Creates the toolbar for scalar range adjustment with sliders."""
        self.scalar_toolbar = QToolBar("Scalar Range", self)
        self.scalar_toolbar.setObjectName("ScalarToolbar") 

        # Spinboxes for precise input/display
        self.scalar_min_spinbox = QDoubleSpinBox(self)
        self.scalar_min_spinbox.setDecimals(3)
        self.scalar_min_spinbox.setSingleStep(0.1)
        self.scalar_min_spinbox.setRange(-1e9, 1e9)
        self.scalar_min_spinbox.setToolTip("Min scalar value")

        self.scalar_max_spinbox = QDoubleSpinBox(self)
        self.scalar_max_spinbox.setDecimals(3)
        self.scalar_max_spinbox.setSingleStep(0.1)
        self.scalar_max_spinbox.setRange(-1e9, 1e9)
        self.scalar_max_spinbox.setToolTip("Max scalar value")
        
        # Sliders for interactive dragging 
        self.scalar_min_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.scalar_min_slider.setRange(0, SLIDER_PRECISION)
        self.scalar_min_slider.setToolTip("Drag to adjust min scalar value")
        
        self.scalar_max_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.scalar_max_slider.setRange(0, SLIDER_PRECISION)
        self.scalar_max_slider.setValue(SLIDER_PRECISION)
        self.scalar_max_slider.setToolTip("Drag to adjust max scalar value")
        
        # Reset Button 
        self.scalar_reset_button: QAction = QAction("Reset", self)
        self.scalar_reset_button.setStatusTip("Reset scalar range to data min/max")
        
        # Layout 
        toolbar_widget = QWidget(self)
        layout = QHBoxLayout(toolbar_widget)
        layout.setContentsMargins(5, 0, 5, 0)
        
        layout.addWidget(QLabel(" Min: "))
        layout.addWidget(self.scalar_min_spinbox, 1)
        layout.addWidget(self.scalar_min_slider, 3)
        
        layout.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Ignored))
        
        layout.addWidget(QLabel(" Max: "))
        layout.addWidget(self.scalar_max_spinbox, 1)
        layout.addWidget(self.scalar_max_slider, 3)

        self.scalar_toolbar.addWidget(toolbar_widget)
        self.scalar_toolbar.addSeparator()
        self.scalar_toolbar.addAction(self.scalar_reset_button)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.scalar_toolbar)
        self.scalar_toolbar.setVisible(False) # Hide by default

        # Connect Signals 
        # Sliders update spinbox on valueChanged (fast, no VTK)
        self.scalar_min_slider.valueChanged.connect(self._slider_value_changed)
        self.scalar_max_slider.valueChanged.connect(self._slider_value_changed)
        
        # Sliders update VTK on sliderReleased (slow)
        self.scalar_min_slider.sliderReleased.connect(self._trigger_vtk_update)
        self.scalar_max_slider.sliderReleased.connect(self._trigger_vtk_update)
        
        # Spinboxes update slider and VTK on editingFinished (Enter pressed)
        self.scalar_min_spinbox.editingFinished.connect(self._spinbox_value_changed)
        self.scalar_max_spinbox.editingFinished.connect(self._spinbox_value_changed)
        
        # Reset button
        self.scalar_reset_button.triggered.connect(self._reset_scalar_range)

    def _create_data_panel(self) -> None:  
        """Creates the dockable panel for listing loaded data."""
        self.data_dock_widget = QDockWidget("Data Panel", self)
        self.data_dock_widget.setObjectName("DataPanelDock")
        self.data_dock_widget.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Container Widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tree Widget
        self.data_tree_widget = QTreeWidget(self)
        self.data_tree_widget.setHeaderLabels(["Loaded Data"])
        self.data_tree_widget.setMinimumWidth(200)
        self.data_tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.data_tree_widget.customContextMenuRequested.connect(self._on_data_panel_context_menu)
        self.data_tree_widget.itemChanged.connect(self._on_data_panel_item_changed)
        self.data_tree_widget.itemSelectionChanged.connect(self._on_data_item_selected)        

        # Highlighting Style Sheet
        self.data_tree_widget.setStyleSheet("""
            QTreeWidget {
                outline: 0; /* Removes the dotted focus line */
            }
            QTreeWidget::item {
                padding: 4px;
                border-radius: 4px; /* Subtle rounded corners */
            }
            
            /* SELECTION STATES */
            QTreeWidget::item:selected {
                background-color: #4a6984; /* Desaturated Steel Blue */
                color: white;
                border: none;
            }
            QTreeWidget::item:selected:!active {
                background-color: #5d7e9b; /* Slightly lighter when window loses focus */
                color: white;
                border: none;
            }

            /* HOVER STATES */
            QTreeWidget::item:hover {
                /* Ultra-subtle tint. No white/bright flash. */
                background-color: rgba(0, 0, 0, 0.03); 
                border: none;
            }
            QTreeWidget::item:selected:hover {
                background-color: #557ba0; /* Slight feedback on the blue selection itself */
                color: white;
            }
        """)
        
        layout.addWidget(self.data_tree_widget)        
        self.data_dock_widget.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.data_dock_widget)
        
    def _on_data_item_selected(self) -> None:
        """Updates the opacity slider based on the selected item."""
        items = self.data_tree_widget.selectedItems()
        if not items:
            self.opacity_slider.setEnabled(False)
            return
            
        item = items[0]
        
        parent = item.parent()
        if parent and parent.text(0) == "Scalars":
            scalar_name = item.text(0)
            
            if self.tractogram_data and self.scalar_data_per_point and scalar_name in self.scalar_data_per_point:
                self.active_scalar_name = scalar_name
                
                for i in range(parent.childCount()):
                    child = parent.child(i)
                    font = child.font(0)
                    font.setBold(child.text(0) == scalar_name)
                    child.setFont(0, font)

                self._update_scalar_data_range()
                self.scalar_range_initialized = True
                
                if self.current_color_mode != ColorMode.SCALAR:
                    self.color_scalar_action.setChecked(True)
                    self._set_color_mode(ColorMode.SCALAR)
                else:
                    if self.vtk_panel:
                        self.vtk_panel.update_main_streamlines_actor()
            
            self.opacity_slider.setEnabled(False)
            return
        
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        
        if not item_data or not isinstance(item_data, dict):
            self.opacity_slider.setEnabled(False)
            return
            
        itype = item_data.get('type')
        val = 1.0
        
        self.opacity_slider.blockSignals(True) # Prevent feedback
        
        if itype == 'bundle':
            val = self.bundle_opacity
            self.opacity_slider.setEnabled(True)
        elif itype == 'image':
            val = self.image_opacity
            self.opacity_slider.setEnabled(True)
        elif itype == 'roi':
            path = item_data.get('path')
            val = self.roi_opacities.get(path, 0.5)
            self.opacity_slider.setEnabled(True)
        else:
            self.opacity_slider.setEnabled(False)
            
        self.opacity_slider.setValue(int(val * 100))
        self.opacity_slider.blockSignals(False)
        
    def _on_opacity_slider_changed(self, value: int) -> None:
        """Updates the opacity of the selected item."""
        float_val = value / 100.0
        
        items = self.data_tree_widget.selectedItems()
        if not items: return
        item = items[0]
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data: return
        
        itype = item_data.get('type')
        
        if itype == 'bundle':
            self.bundle_opacity = float_val
            if self.vtk_panel:
                self.vtk_panel.set_streamlines_opacity(float_val)
                
        elif itype == 'image':
            self.image_opacity = float_val
            if self.vtk_panel:
                self.vtk_panel.set_anatomical_opacity(float_val)
                
        elif itype == 'roi':
            path = item_data.get('path')
            if path:
                self.roi_opacities[path] = float_val
                if self.vtk_panel:
                    self.vtk_panel.set_roi_opacity(path, float_val)

    def _setup_status_bar(self) -> None:
        """
        Creates and configures the status bar with permanent widgets for
        bundle/image info and interactive RAS coordinate display.
        """
        # Create Container Widget 
        self.permanent_status_widget = QWidget(self)
        layout = QHBoxLayout(self.permanent_status_widget)
        layout.setContentsMargins(0, 0, 5, 0) 
        layout.setSpacing(10)

        # Data Info Label
        self.data_info_label = QLabel(" No data loaded ")
        self.data_info_label.setStyleSheet("border: 1px solid grey; padding: 2px;")
        layout.addWidget(self.data_info_label, 1) # Give it stretch factor 1

        # RAS Coordinate Display 
        self.ras_label = QLabel("RAS: ", self)
        self.ras_label.setToolTip(
            "Current RAS coordinates. Enter values (e.g., '10.5, -5, 20') and press Enter."
        )
        layout.addWidget(self.ras_label, 0)
        
        self.ras_coordinate_input = QLineEdit("--, --, --", self) 
        self.ras_coordinate_input.setToolTip(
            "Current RAS coordinates. Enter values (e.g., '10.5, -5, 20') and press Enter."
        ) 
        self.ras_coordinate_input.setMinimumWidth(150) 
        self.ras_coordinate_input.setMaximumWidth(180)
        self.ras_coordinate_input.setStyleSheet("border: 1px solid grey; padding: 2px;")
        layout.addWidget(self.ras_coordinate_input, 0) 
        
        # Add Container to Status Bar 
        self.status_bar: QStatusBar = self.statusBar()
        self.status_bar.addPermanentWidget(self.permanent_status_widget)

        # Connect Signal for Manual Entry 
        self.ras_coordinate_input.returnPressed.connect(self._on_ras_coordinate_entered)

    def _setup_central_widget(self) -> None:
        """Sets up the main central widget which will contain the VTK panel."""
        self.central_widget: QWidget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.vtk_panel: VTKPanel = VTKPanel(parent_widget=self.central_widget, main_window_ref=self)

    def _update_initial_status(self) -> None:
        """Sets the initial status message in the VTK panel."""
        date_str = get_formatted_datetime()
        self.vtk_panel.update_status(f"Ready ({date_str}). Load data.")

    def _update_action_states(self) -> None:
            """Enables/disables actions based on current application state."""
            has_streamlines = self.tractogram_data is not None
            has_odf = self.odf_data is not None
            has_selection = bool(self.selected_streamline_indices)
            has_scalars = bool(self.scalar_data_per_point)
            has_image = self.anatomical_image_data is not None
            has_any_data = has_streamlines or has_image

            # File Menu
            self.load_bg_image_action.setEnabled(has_streamlines)
            self.view_odf_tunnel_action.setEnabled(has_odf and has_streamlines)
            self.load_roi_action.setEnabled(has_image)
            self.close_bundle_action.setEnabled(has_streamlines)
            self.clear_bg_image_action.setEnabled(has_image)
            self.clear_all_rois_action.setEnabled(bool(self.roi_layers))
            self.clear_all_data_action.setEnabled(has_any_data) 
            self.calc_centroid_action.setEnabled(has_streamlines)
            self.calc_medoid_action.setEnabled(has_streamlines)
            self.save_file_action.setEnabled(has_streamlines)
            self.save_density_map_action.setEnabled(has_streamlines)
            self.screenshot_action.setEnabled(has_any_data)

            # Edit Menu
            self.undo_action.setEnabled(bool(self.undo_stack))
            self.redo_action.setEnabled(bool(self.redo_stack))

            # View Menu - Streamline Colors
            self.color_default_action.setEnabled(has_streamlines)
            self.color_orientation_action.setEnabled(has_streamlines)
            self.color_scalar_action.setEnabled(has_streamlines and has_scalars)

            # Commands Menu
            self.clear_select_action.setEnabled(has_selection)
            self.delete_select_action.setEnabled(has_selection)
            self.increase_radius_action.setEnabled(has_streamlines)
            self.decrease_radius_action.setEnabled(has_streamlines)
            
            # Geometry Menu
            self.geo_lines_action.setEnabled(has_streamlines)
            self.geo_tubes_action.setEnabled(has_streamlines)
            
    def _trigger_calculate_centroid(self) -> None:
        """Wrapper to calculate and save centroid."""
        file_io.calculate_and_save_statistic(self, 'centroid')

    def _trigger_calculate_medoid(self) -> None:
        """Wrapper to calculate and save medoid."""
        file_io.calculate_and_save_statistic(self, 'medoid')
        
    def _set_geometry_mode(self, as_tubes: bool) -> None:
        """Switches between Line and Tube rendering."""
        if self.render_as_tubes == as_tubes:
            return
            
        self.render_as_tubes = as_tubes
        
        if self.vtk_panel:
            self.vtk_panel.update_status(f"Rendering geometry set to: {'Tubes' if as_tubes else 'Lines'}")
            self.vtk_panel.update_main_streamlines_actor()
            
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
                QMessageBox.warning(self, "ODF Error", "File must be a 4D volume (SH coefficients).")
                return
                
            n_coeffs = data.shape[-1]
            try:
                sh_order = odf_utils.calculate_sh_order(n_coeffs)
            except ValueError as e:
                QMessageBox.warning(self, "ODF Error", str(e))
                return
                
            self.odf_data = data
            self.odf_affine = affine
            self.odf_sh_order = sh_order
            
            # Pre-compute Sphere and Basis (Tournier07)
            self.odf_sphere = odf_utils.generate_symmetric_sphere(radius=1.0, subdivisions=3)
            self.vtk_panel.update_status("Computing SH Basis...")
            QApplication.processEvents()
            
            self.odf_basis_matrix = odf_utils.compute_sh_basis(
                self.odf_sphere.vertices, sh_order, basis_type='tournier07'
            )
            
            self.vtk_panel.update_status(f"ODF Loaded (Order {sh_order}). Ready for Tunnel View.")
            self._update_action_states()
            
        except Exception as e:
            logger.error(f"Error loading ODF: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"Could not load ODF file:\n{e}")
            
            
    def _toggle_odf_tunnel(self, checked: bool) -> None:
        """Computes the mask and updates the VTK actor with progress indication."""
        if not checked:
            if self.vtk_panel:
                self.vtk_panel.remove_odf_actor()
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
            QMessageBox.warning(self, "Performance Warning", 
                                f"Too many streamlines selected ({len(current_streamlines)}).\n"
                                f"Limit is {self.MAX_ODF_STREAMLINES}. \n\n"
                                f"Tip: Increase the 'Skip %' or use ROIs to reduce the count.")
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
                dilation_iter=1
            )
            
            # Update Progress -> 25%
            self.vtk_panel.update_progress_bar(1, TOTAL_STEPS, visible=True)
            QApplication.processEvents()
            
            # Apply Mask to ODF Data
            masked_coeffs = self.odf_data * mask[..., np.newaxis]
            
            # Update Progress -> 50%
            self.vtk_panel.update_progress_bar(2, TOTAL_STEPS, visible=True)
            QApplication.processEvents()
            
            # Project to Amplitudes (SF)
            amplitudes_shape = self.odf_data.shape[:3] + (self.odf_sphere.vertices.shape[0],)
            odf_amplitudes = np.zeros(amplitudes_shape, dtype=np.float32)
            
            # Flatten mask to find indices
            mask_indices = np.where(mask)
            
            extent = None
            
            if len(mask_indices[0]) > 0:
                valid_coeffs = masked_coeffs[mask_indices]
                valid_amps = np.dot(valid_coeffs, self.odf_basis_matrix.T)
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
                odf_amplitudes, 
                self.odf_sphere, 
                self.odf_affine,
                extent=extent 
            )
            
            # Update Progress -> 100% and Hide
            self.vtk_panel.update_progress_bar(TOTAL_STEPS, TOTAL_STEPS, visible=True)
            QApplication.processEvents()
            
        except Exception as e:
            logger.error(f"Error computing Tunnel View: {e}", exc_info=True)
            self.vtk_panel.update_status("Error generating Tunnel View.")
            self.view_odf_tunnel_action.setChecked(False)
        finally:
            self.vtk_panel.update_progress_bar(0, 0, visible=False)
            
            
    def _update_bundle_info_display(self) -> None:
        """Updates the data information QLabel in the status bar for both streamlines and image."""
        if not self.data_info_label: # Check if label exists
            return
        bundle_text = "Bundle: None"
        image_text = "Image: None"

        # Streamline Info
        if self.tractogram_data is not None:
            count = len(self.visible_indices)
            filename = os.path.basename(self.original_trk_path) if self.original_trk_path else "Unknown"
            file_type_info = f" ({self.original_file_extension.upper()})" if self.original_file_extension else ""
            scalar_info = f" | Scalar: {self.active_scalar_name}" if self.active_scalar_name else ""
            header = self.original_trk_header if self.original_trk_header is not None else {}

            dims_str, vox_str, order = "N/A", "N/A", "N/A"
            if 'dimensions' in header:
                 dims_val = header['dimensions']
                 if isinstance(dims_val, (tuple, list, np.ndarray)) and len(dims_val) == 3:
                     dims_str = format_tuple(dims_val, precision=0)
            if 'voxel_sizes' in header:
                 vox_val = header['voxel_sizes']
                 if isinstance(vox_val, (tuple, list, np.ndarray)) and len(dims_val) == 3:
                     vox_str = format_tuple(vox_val, precision=2)
            if 'voxel_order' in header and isinstance(header['voxel_order'], str):
                 order = header['voxel_order']

            bundle_text = (f"Bundle: {filename}{file_type_info} | #: {count} | Dim={dims_str} | "
                           f"VoxSize={vox_str} | Order={order}{scalar_info}")

        # Anatomical Image Info
        if self.anatomical_image_data is not None:
            filename = os.path.basename(self.anatomical_image_path) if self.anatomical_image_path else "Unknown"
            shape_str = format_tuple(self.anatomical_image_data.shape, precision=0)
            image_text = f"Image: {filename} | Shape={shape_str}"

        # ROI Info
        roi_text = ""
        if self.roi_layers:
            roi_text = f" | ROIs: {len(self.roi_layers)}"
            
        # Combine and Set
        separator = " || " if self.tractogram_data is not None and self.anatomical_image_data is not None else " | "
        if self.tractogram_data is None and not self.anatomical_image_data:
            final_text = " No data loaded "
        elif self.tractogram_data is not None and self.anatomical_image_data is None:
            final_text = f" {bundle_text} "
        elif self.tractogram_data is None and self.anatomical_image_data is not None:
             final_text = f" {image_text} "
        else:
             final_text = f" {bundle_text}{separator}{image_text}{roi_text} "
        
        self.data_info_label.setText(final_text)
        self._update_data_panel_display()


    def _update_data_panel_display(self) -> None:
        """
        Updates the QTreeWidget in the data panel dock.
        """
        if not self.data_tree_widget:
            return

        self.data_tree_widget.clear()
        
        # Streamlines
        if self.tractogram_data is not None:
            bundle_name = (os.path.basename(self.original_trk_path) 
                           if self.original_trk_path else "Loaded Bundle")
            
            bundle_item = QTreeWidgetItem(self.data_tree_widget, [bundle_name])            
            bundle_item.setFlags(bundle_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            bundle_state = Qt.CheckState.Checked if self.bundle_is_visible else Qt.CheckState.Unchecked
            bundle_item.setCheckState(0, bundle_state)
            bundle_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'bundle'})
            
            count = len(self.visible_indices)
            ext = self.original_file_extension.upper() if self.original_file_extension else "TRK"
            dims = "N/A"
            if self.original_trk_header and 'dimensions' in self.original_trk_header:
                 dims = format_tuple(self.original_trk_header['dimensions'], precision=0)
            
            tooltip_text = f"Type: {ext}\nCount: {count}\nDimensions: {dims}"
            bundle_item.setToolTip(0, tooltip_text)

            if self.scalar_data_per_point:
                scalars_root = QTreeWidgetItem(bundle_item, ["Scalars"])
                scalars_root.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
                for scalar_name in self.scalar_data_per_point.keys():
                    scalar_item = QTreeWidgetItem(scalars_root, [scalar_name])
                    if scalar_name == self.active_scalar_name:
                        font = scalar_item.font(0)
                        font.setBold(True)
                        scalar_item.setFont(0, font)
            
            bundle_item.setExpanded(True)
        else:
            item = QTreeWidgetItem(self.data_tree_widget, ["No streamlines loaded"])
            item.setDisabled(True)

        # Anatomical Image 
        if self.anatomical_image_data is not None:
            image_name = (os.path.basename(self.anatomical_image_path) 
                          if self.anatomical_image_path else "Loaded Image")
            
            image_item = QTreeWidgetItem(self.data_tree_widget, [image_name])
            image_item.setFlags(image_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            image_state = Qt.CheckState.Checked if self.image_is_visible else Qt.CheckState.Unchecked
            image_item.setCheckState(0, image_state)
            image_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'image'})
            shape_str = format_tuple(self.anatomical_image_data.shape, precision=0)
            image_item.setToolTip(0, f"Path: {self.anatomical_image_path}\nShape: {shape_str}")
        else:
            item = QTreeWidgetItem(self.data_tree_widget, ["No anatomical image"])
            item.setDisabled(True)
            
        # ROI Layers 
        if self.roi_layers:
            roi_root_item = QTreeWidgetItem(self.data_tree_widget, ["ROI Layers"])
            roi_root_item.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
            
            for path, roi_info in self.roi_layers.items():
                roi_name = os.path.basename(path)
                
                state_str = ""
                if path in self.roi_states:
                    if self.roi_states[path].get('select'): state_str = " [SELECT]"
                    elif self.roi_states[path].get('include'): state_str = " [INCLUDE]"
                    elif self.roi_states[path].get('exclude'): state_str = " [EXCLUDE]"
                
                display_text = f"{roi_name}{state_str}"
                roi_item = QTreeWidgetItem(roi_root_item, [display_text])                
                roi_item.setFlags(roi_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                is_visible = self.roi_visibility.get(path, True)
                roi_state = Qt.CheckState.Checked if is_visible else Qt.CheckState.Unchecked
                roi_item.setCheckState(0, roi_state)
                roi_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'roi', 'path': path})
                
                shape_str = format_tuple(roi_info['data'].shape, precision=0)
                roi_item.setToolTip(0, f"Path: {path}\nShape: {shape_str}")
                
            roi_root_item.setExpanded(True)

        self.data_tree_widget.resizeColumnToContents(0)

    # Undo/Redo Core Logic           
    def _perform_undo(self) -> None:
        if not self.undo_stack: return
        restored = self.undo_stack.pop()
        self.redo_stack.append(restored)
        self.manual_visible_indices.update(restored) # Update manual
        self._apply_logic_filters() # Re-apply filters
        self._update_action_states()
        
    def _perform_redo(self) -> None:
        if not self.redo_stack: return
        re_deleted = self.redo_stack.pop()
        self.undo_stack.append(re_deleted)
        self.manual_visible_indices.difference_update(re_deleted) # Update manual
        self._apply_logic_filters() # Re-apply filters 
        self._update_action_states()

    # Command Actions Logic
    def _perform_clear_selection(self) -> None:
        """Clears the current streamline selection."""
        if self.vtk_panel: self.vtk_panel.update_radius_actor(visible=False)

        if self.selected_streamline_indices:
            self.selected_streamline_indices = set()
            if self.vtk_panel:
                self.vtk_panel.update_highlight()
                self.vtk_panel.update_status("Selection cleared.")
        elif self.vtk_panel:
            self.vtk_panel.update_status("Clear: No active selection.")
        self._update_action_states()
        
    def _perform_reset_camera(self) -> None:
        """
        Resets the 3D camera view to a Front Coronal orientation.
        Centers the view and aligns it with the Y-axis (Anterior-Posterior).
        """
        if not self.vtk_panel or not self.vtk_panel.scene:
            return

        # Standard view reset
        self.vtk_panel.scene.reset_camera()
        
        # Get the camera and current parameters calculated by reset_camera()
        cam = self.vtk_panel.scene.GetActiveCamera()
        fp = cam.GetFocalPoint()
        dist = cam.GetDistance()
        
        # Re-orient to Front Coronal (Anterior View)
        cam.SetPosition(fp[0], fp[1] + dist, fp[2])
        cam.SetFocalPoint(fp[0], fp[1], fp[2])
        cam.SetViewUp(0, 0, 1) # Up is Superior (+Z)
        
        # Finalize update
        self.vtk_panel.scene.reset_clipping_range()
        if self.vtk_panel.render_window:
            self.vtk_panel.render_window.Render()
            
        self.vtk_panel.update_status("Camera reset (Front Coronal).")

    def _perform_delete_selection(self) -> None:
        if not self.selected_streamline_indices: return
        
        # Standard Undo Logic
        to_delete = self.selected_streamline_indices.copy()
        self.undo_stack.append(to_delete)
        self.redo_stack = []
        
        # Update MANUAL state, not just visible
        self.manual_visible_indices.difference_update(to_delete)
        
        self.selected_streamline_indices = set()
        self._apply_logic_filters() # Re-apply filters on new manual state
        self._update_action_states()
        
        # Hide selection sphere after deletion 
        if self.vtk_panel:
            self.vtk_panel.update_radius_actor(visible=False)
            
    def _increase_radius(self) -> None:
        """Increases the selection radius."""
        if not self.tractogram_data: 
            return
        self.selection_radius_3d += RADIUS_INCREMENT
        if self.vtk_panel:
            self.vtk_panel.update_status(f"Selection radius increased to {self.selection_radius_3d:.1f}mm.")
            if self.vtk_panel.radius_actor and self.vtk_panel.radius_actor.GetVisibility():
                center = self.vtk_panel.radius_actor.GetCenter()
                self.vtk_panel.update_radius_actor(center_point=center, radius=self.selection_radius_3d, visible=True)

    def _decrease_radius(self) -> None:
        """Decreases the selection radius."""
        if not self.tractogram_data: 
            return
        new_radius = self.selection_radius_3d - RADIUS_INCREMENT
        self.selection_radius_3d = max(MIN_SELECTION_RADIUS, new_radius)
        if self.vtk_panel:
            self.vtk_panel.update_status(f"Selection radius decreased to {self.selection_radius_3d:.1f}mm.")
            if self.vtk_panel.radius_actor and self.vtk_panel.radius_actor.GetVisibility():
                center = self.vtk_panel.radius_actor.GetCenter()
                self.vtk_panel.update_radius_actor(center_point=center, radius=self.selection_radius_3d, visible=True)


    def _hide_sphere(self) -> None:
        """Hides the selection sphere."""
        if self.vtk_panel:
            self.vtk_panel.update_radius_actor(visible=False)
            self.vtk_panel.update_status("Selection sphere hidden.")


    # View Action Logic 
    @pyqtSlot(object)
    def _set_color_mode(self, mode: ColorMode) -> None:
        """Sets the streamline coloring mode and triggers VTK update."""
        if not isinstance(mode, ColorMode): 
            return
        if not self.tractogram_data:
             self.color_default_action.setChecked(True)
             return

        # Handle scalar toolbar visibility 
        if self.current_color_mode != mode:
            if mode == ColorMode.SCALAR:
                if not self.active_scalar_name:
                    QMessageBox.warning(self, "Coloring Error", "No active scalar data loaded for streamlines.")
                    if self.current_color_mode == ColorMode.DEFAULT: self.color_default_action.setChecked(True)
                    elif self.current_color_mode == ColorMode.ORIENTATION: self.color_orientation_action.setChecked(True)
                    return
                
                # Calculate range in scalar mode
                if not self.scalar_range_initialized:
                    self._update_scalar_data_range() 
                    self.scalar_range_initialized = True
                
                if self.scalar_toolbar: self.scalar_toolbar.setVisible(True)

            elif mode == ColorMode.DEFAULT or mode == ColorMode.ORIENTATION:
                if self.scalar_toolbar: self.scalar_toolbar.setVisible(False)
        
                self.bundle_is_visible = True 

            self.current_color_mode = mode
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
                self.vtk_panel.update_status(f"Streamline color mode changed to {mode.name}.")
        
        # Ensure toolbar visibility
        if self.scalar_toolbar:
             is_scalar = (mode == ColorMode.SCALAR and bool(self.active_scalar_name))
             self.scalar_toolbar.setVisible(is_scalar)


    # GUI Action Methods 
    def _close_bundle(self) -> None:
        """
        Closes the current streamline bundle.
        """
        if not self.tractogram_data:
            if self.vtk_panel:
                self.vtk_panel.update_status("No bundle open to close.")
            return

        if self.vtk_panel:
            self.vtk_panel.update_status("Closing bundle (also clears image)...") 
            QApplication.processEvents()

            # Remove/hide streamline-related actors
            self.vtk_panel.update_radius_actor(visible=False)
            self.selected_streamline_indices = set()
            self.vtk_panel.update_highlight()
            self.vtk_panel.remove_odf_actor()

            # Clear anatomical slices if present
            if self.anatomical_image_data is not None:
                 self.anatomical_image_path = None
                 self.anatomical_image_data = None
                 self.anatomical_image_affine = None
                 self.vtk_panel.clear_anatomical_slices()

        # Reset streamline data state
        self.tractogram_data = None
        self.streamline_bboxes = None
        self.visible_indices = set()
        self.original_trk_header = None
        self.original_trk_affine = None
        self.original_trk_path = None
        self.original_file_extension = None
        self.scalar_data_per_point = None
        self.active_scalar_name = None
        self.odf_data = None
        self.odf_affine = None
        self.odf_sh_order = 0
        self.view_odf_tunnel_action.blockSignals(True) # Prevent triggering logic
        self.view_odf_tunnel_action.setChecked(False)
        self.view_odf_tunnel_action.setEnabled(False)
        self.view_odf_tunnel_action.blockSignals(False)
        self.undo_stack = []
        self.redo_stack = []
        self.current_color_mode = ColorMode.ORIENTATION
        self.color_default_action.setChecked(True)
        self.scalar_range_initialized = False
        if self.scalar_toolbar: self.scalar_toolbar.setVisible(False)

        # Update VTK
        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor() # Should remove streamline actor
            self.vtk_panel.update_status("Bundle closed (Image also cleared).") # Keep status clear
            if self.vtk_panel.render_window and self.vtk_panel.render_window.GetInteractor().GetInitialized():
                self.vtk_panel.render_window.Render()
                
        # Reset Geometry to Lines default
        self.render_as_tubes = False
        self.geo_lines_action.setChecked(True)

        # Update UI
        self._update_bundle_info_display()
        self._update_action_states()

    # Action Trigger Wrappers 
    def _trigger_load_streamlines(self) -> None:
        """Wrapper to call the streamline load function from file_io."""
        self.scalar_range_initialized = False
        if self.scalar_toolbar: self.scalar_toolbar.setVisible(False)
        self.bundle_is_visible = True
        
        file_io.load_streamlines_file(self)
        if self.tractogram_data:
            self._auto_calculate_skip_level() # Automatic skip level based on count
            self.manual_visible_indices = set(range(len(self.tractogram_data)))
            # Clear caches on new load
            self.roi_states = {}
            self.roi_intersection_cache = {}
            self.roi_highlight_indices = set()
        
        # Update scalar range if scalar mode is already active
        if self.current_color_mode == ColorMode.SCALAR and self.active_scalar_name:
             self._update_scalar_data_range()
             self.scalar_range_initialized = True
             if self.scalar_toolbar: self.scalar_toolbar.setVisible(True)


    def _trigger_save_streamlines(self) -> None:
        """Wrapper to call the streamline save function from file_io."""
        file_io.save_streamlines_file(self)
        
        
    def _trigger_save_density_map(self) -> None:
        """
        Calculates and saves a Track Density Imaging (TDI) map of the currently 
        visible streamlines. Uses the anatomical image grid if available for 
        maximum accuracy/alignment, otherwise derives a grid from the tractogram.
        """
        if not self.tractogram_data:
            return

        # Determine Target Grid (Reference) 
        affine = None
        shape = None
        
        # Priority A: Loaded Anatomical Image (Best for alignment)
        if self.anatomical_image_data is not None:
            affine = self.anatomical_image_affine
            shape = self.anatomical_image_data.shape[:3]
            
        # Priority B: Original Header Info (if compatible/available)
        elif self.original_trk_header:
            try:
                # Check for standard TRK header fields
                if 'dimensions' in self.original_trk_header and 'voxel_to_rasmm' in self.original_trk_header:
                    shape = tuple(int(d) for d in self.original_trk_header['dimensions'][:3])
                    affine = self.original_trk_header['voxel_to_rasmm']
            except Exception:
                pass
        
        # Priority C: Compute Bounding Box (Fallback)
        # If no reference is found, we create a 1mm isotropic grid around the bundle
        if affine is None or shape is None:
            self.vtk_panel.update_status("Calculating density grid from bounds...")
            QApplication.processEvents()
            
            visible_streamlines = [self.tractogram_data[i] for i in self.visible_indices 
                                  if self.tractogram_data[i] is not None]
            
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
                
            except Exception as e:
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
            visible_streamlines = [self.tractogram_data[i] for i in self.visible_indices 
                                  if self.tractogram_data[i] is not None and len(self.tractogram_data[i]) > 0]
            
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
                (vox_indices[:, 0] >= 0) & (vox_indices[:, 0] < shape[0]) &
                (vox_indices[:, 1] >= 0) & (vox_indices[:, 1] < shape[1]) &
                (vox_indices[:, 2] >= 0) & (vox_indices[:, 2] < shape[2])
            )
            valid_voxels = vox_indices[valid_mask]
            
            # Binning (Histogram)
            density_data = np.zeros(shape, dtype=np.int32)
            
            # Fast unbuffered summation at coordinates
            np.add.at(density_data, (valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]), 1)
            
            # Save to Disk 
            nifti_img = nib.Nifti1Image(density_data.astype(np.float32), affine)
            
            # Copy header info if possible (e.g. from anatomy) to preserve orientations
            if self.anatomical_image_path and self.anatomical_image_data is not None:
                 try:
                     ref_img = nib.load(self.anatomical_image_path)
                     nifti_img.header.set_zooms(ref_img.header.get_zooms()[:3])
                     nifti_img.header.set_xyzt_units(*ref_img.header.get_xyzt_units())
                 except: pass

            nib.save(nifti_img, file_path)
            
            self.vtk_panel.update_status(f"Saved density map: {os.path.basename(file_path)}")

        except Exception as e:
            logger.error(f"Error saving density map: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Could not save density map:\n{e}")
            self.vtk_panel.update_status("Error saving density map.")
        finally:
             self.vtk_panel.update_progress_bar(0, 0, visible=False)


    def _trigger_screenshot(self) -> None:
        """Wrapper to call the screenshot function in vtk_panel."""
        if not (self.tractogram_data or self.anatomical_image_data):
             QMessageBox.warning(self, "Screenshot Error", "No data loaded to take a screenshot of.")
             return
        if self.vtk_panel:
            try:
                self.vtk_panel.take_screenshot()
            except AttributeError:
                 QMessageBox.warning(self, "Error", "Screenshot function not available in VTK panel.")
            except Exception as e:
                 QMessageBox.critical(self, "Screenshot Error", f"Could not take screenshot:\n{e}")
        else:
            QMessageBox.warning(self, "Screenshot Error", "VTK panel not initialized.")


    # Background Image Methods 
    def _trigger_load_anatomical_image(self) -> None:
        """Triggers loading of an anatomical image."""
        if self.anatomical_image_data is not None:
            reply = QMessageBox.question(self, 'Replace Image?',
                                        "An anatomical image is already loaded.\nReplace it?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
            else:
                self._trigger_clear_anatomical_image() # Clear before loading new one
                
        # Clear ROIs if present
        if self.roi_layers:
            self._trigger_clear_all_rois(notify=False)

        img_data, img_affine, img_path = file_io.load_anatomical_image(self)

        if img_data is not None and img_affine is not None and img_path:
            self.anatomical_image_data = img_data
            self.anatomical_image_affine = img_affine
            self.anatomical_image_path = img_path
            self.image_is_visible = True

            if self.vtk_panel:
                self.vtk_panel.update_anatomical_slices()
                if self.vtk_panel.scene:
                    self.vtk_panel.scene.reset_camera()
                    self.vtk_panel.scene.reset_clipping_range()
                if self.vtk_panel.render_window:
                    self.vtk_panel.render_window.Render()

            self._update_bundle_info_display()
            self._update_action_states()
            
            
    # ROI Image Methods 
    def _trigger_load_roi(self) -> None:
        """
        Triggers loading of ROI image layer(s),
        by reorienting both the data and the affine.
        """
        if self.anatomical_image_data is None:
            QMessageBox.warning(self, "Load Error", "Please load a main anatomical image before adding an ROI layer.")
            return
        
        if not self.anatomical_image_path:
            QMessageBox.warning(self, "Load Error", "Cannot find the path for the loaded anatomical image. Cannot re-orient.")
            return

        # Call the plural function handling multiple files
        loaded_rois = file_io.load_roi_images(self) 

        if not loaded_rois:
            return  # User cancelled or all failed
        
        # Iterate through every loaded ROI
        for _, _, roi_path in loaded_rois:
            
            if roi_path in self.roi_layers:
                QMessageBox.warning(self, "ROI Already Loaded", 
                                    f"The ROI from '{os.path.basename(roi_path)}' is already loaded.")
                continue
            
            self.roi_visibility[roi_path] = True
            self.roi_opacities[roi_path] = 0.5 # Default ROI opacity
                
            if self.vtk_panel:
                self.vtk_panel.update_status(f"Processing {os.path.basename(roi_path)}...")
                QApplication.processEvents() 

            try:
                # Load the main anatomical image object from its stored path
                anatomical_img = nib.load(self.anatomical_image_path)
                
                # Load the ROI image object again to perform reorientation operations
                roi_img = nib.load(roi_path)
                
                # Ensure proper coordinate system alignment 
                current_ornt = nib.io_orientation(roi_img.affine)
                target_ornt = nib.io_orientation(anatomical_img.affine)
                
                # Check if they match
                if not np.array_equal(current_ornt, target_ornt):
                    if self.vtk_panel:
                        current_axcodes = ''.join(nib.aff2axcodes(roi_img.affine))
                        target_axcodes = ''.join(nib.aff2axcodes(anatomical_img.affine))
                        self.vtk_panel.update_status(f"Reorienting {os.path.basename(roi_path)} ({current_axcodes} -> {target_axcodes})...")
                    
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
                    'data': roi_data, 
                    'affine': roi_affine, 
                    'path': roi_path, 
                    'inv_affine': inv_affine,
                    'T_main_to_roi': T_main_to_roi
                }
                
                self.roi_visibility[roi_path] = True

                # Tell VTK panel to create and add the new actors
                if self.vtk_panel:
                    self.vtk_panel.add_roi_layer(roi_path, roi_data, roi_affine)
                    self.vtk_panel.update_status(f"Aligned and added {os.path.basename(roi_path)}.")

            except FileNotFoundError as e:
                logger.error(f"File not found during processing: {e}")
                continue
            except np.linalg.LinAlgError:
                QMessageBox.critical(self, "Load Error", f"Could not invert affine matrix for {os.path.basename(roi_path)}.")
                if roi_path in self.roi_visibility:
                        del self.roi_visibility[roi_path]
                continue

        # Final UI Updates after loop
        self._update_bundle_info_display()
        self._update_action_states()
        if self.vtk_panel:
            self.vtk_panel.update_status("ROI loading complete.")


    def _trigger_clear_all_rois(self, notify: bool = True) -> None:
        """Clears all loaded ROI image layers and resets logic filters."""
        if not self.roi_layers:
            return

        # Clear Data Containers
        self.roi_layers.clear()
        self.roi_visibility.clear()

        # Clear Logic States and Caches
        self.roi_states.clear()
        self.roi_intersection_cache.clear()
        self.roi_highlight_indices.clear()

        # Re-calculate Visuals
        self._update_roi_visual_selection() 
        self._apply_logic_filters()

        if self.vtk_panel:
            self.vtk_panel.clear_all_roi_layers()
            if notify:
                self.vtk_panel.update_status("All ROI layers and filters cleared.")
        
        self._update_bundle_info_display()
        self._update_action_states()
        
        
    def _trigger_clear_all_data(self) -> None:
        """Clears all loaded data (streamlines, anatomical image, ROIs) without confirmation."""
        has_data = (self.tractogram_data is not None or 
                    self.anatomical_image_data is not None or 
                    bool(self.roi_layers))
        
        if not has_data:
            return

        # Clear ROIs first
        if self.roi_layers:
            self._trigger_clear_all_rois(notify=False)
        
        # Clear Streamlines 
        if self.tractogram_data is not None:
            self._close_bundle() 
        
        # Clear Image (if not already cleared by _close_bundle or if no bundle was loaded)
        if self.anatomical_image_data is not None:
            self._trigger_clear_anatomical_image()
        
        # Final status update
        self.vtk_panel.update_status("All data cleared.")
        
    @pyqtSlot(QTreeWidgetItem, int)
    def _on_data_panel_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """
        Slot triggered when an item in the data panel (QTreeWidget) is changed,
        e.g., a checkbox is toggled.
        """
        if column != 0:
            return

        # Get the data we stored to identify the item
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data or not isinstance(item_data, dict):
            return 

        # Get the new check state
        is_checked = (item.checkState(0) == Qt.CheckState.Checked)
        item_type = item_data.get('type')

        # Block signals on the tree to prevent recursion while we process
        if self.data_tree_widget:
            self.data_tree_widget.blockSignals(True)
        
        try:
            if item_type == 'bundle':
                self._toggle_bundle_visibility(is_checked)
                
            elif item_type == 'image': 
                self._toggle_image_visibility(is_checked)
            
            elif item_type == 'roi':
                path = item_data.get('path')
                if path:
                    self._toggle_roi_visibility(path, is_checked)

        except Exception as e:
            logger.error(f"Error handling item visibility change: {e}", exc_info=True)
        finally:
            # Unblock signals
            if self.data_tree_widget:
                self.data_tree_widget.blockSignals(False)
                
                
    # Data Panel Context Menu Logic 
    def _on_data_panel_context_menu(self, position) -> None:
        """
        Grouped Logic Modes with Radio Buttons.
        """
        item = self.data_tree_widget.itemAt(position)
        if not item: return

        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        
        if item_data and isinstance(item_data, dict) and item_data.get('type') == 'roi':
            roi_path = item_data.get('path')
            
            # Ensure state dict exists
            if roi_path not in self.roi_states:
                self.roi_states[roi_path] = {'select': False, 'include': False, 'exclude': False}
            
            current_state = self.roi_states[roi_path]
            
            menu = QMenu()
            
            # Logic Mode Group (Mutually Exclusive) 
            logic_group = QActionGroup(self)
            logic_group.setExclusive(True)
            
            # Visual Only (None)
            act_none = QAction("Visual Only", self)
            act_none.setCheckable(True)
            is_none = not (current_state['select'] or current_state['include'] or current_state['exclude'])
            act_none.setChecked(is_none)
            act_none.triggered.connect(lambda: self._set_roi_logic_mode(roi_path, 'none'))
            logic_group.addAction(act_none)
            menu.addAction(act_none)
            
            # Select
            act_select = QAction("Select (Highlight)", self)
            act_select.setCheckable(True)
            act_select.setChecked(current_state['select'])
            act_select.triggered.connect(lambda: self._set_roi_logic_mode(roi_path, 'select'))
            logic_group.addAction(act_select)
            menu.addAction(act_select)

            # Include
            act_include = QAction("Include (AND)", self)
            act_include.setCheckable(True)
            act_include.setChecked(current_state['include'])
            act_include.triggered.connect(lambda: self._set_roi_logic_mode(roi_path, 'include'))
            logic_group.addAction(act_include)
            menu.addAction(act_include)
            
            # Exclude
            act_exclude = QAction("Exclude (NOT)", self)
            act_exclude.setCheckable(True)
            act_exclude.setChecked(current_state['exclude'])
            act_exclude.triggered.connect(lambda: self._set_roi_logic_mode(roi_path, 'exclude'))
            logic_group.addAction(act_exclude)
            menu.addAction(act_exclude)

            menu.addSeparator()
    
            # Standard Actions 
            change_color_action = QAction("Change Color...", self)
            change_color_action.triggered.connect(lambda: self._change_roi_color_action(roi_path))
            menu.addAction(change_color_action)

            remove_action = QAction("Remove ROI Layer", self)
            remove_action.triggered.connect(lambda: self._remove_roi_layer_action(roi_path))
            menu.addAction(remove_action)
            
            menu.exec(self.data_tree_widget.mapToGlobal(position))
            
            
    def _set_roi_logic_mode(self, roi_path: str, mode: str) -> None:
        """
        Sets the logic mode for an ROI, ensuring mutual exclusivity.
        Modes: 'none', 'select', 'include', 'exclude'
        """
        if roi_path not in self.roi_states:
            return

        # Reset all flags
        for f in ['select', 'include', 'exclude']:
            self.roi_states[roi_path][f] = False

        # Set new flag (unless mode is 'none')
        if mode != 'none':
            self.roi_states[roi_path][mode] = True
            
            # Compute intersection if needed
            if roi_path not in self.roi_intersection_cache:
                success = self._compute_roi_intersection(roi_path)
                if not success:
                    self.roi_states[roi_path][mode] = False # Revert on failure

        # Refresh Visuals
        self._update_roi_visual_selection()  # Updates Red Highlight
        self._apply_logic_filters()          # Updates Streamline Visibility
        
        # Refresh Panel Text (to show [TAG])
        self._update_data_panel_display()
        
    def _toggle_image_visibility(self, visible: bool) -> None:
        """Toggles the visibility of the anatomical image slices."""
        if self.image_is_visible == visible:
            return
            
        self.image_is_visible = visible
        if self.vtk_panel:
            self.vtk_panel.set_anatomical_slice_visibility(visible)
            self.vtk_panel.update_status(f"Image visibility set to {visible}")
        
        
    def _compute_roi_intersection(self, roi_path: str) -> bool:
        """
        Computes intersections using a Broad Phase (Bounding Box) filter 
        followed by a Narrow Phase (Voxel Grid) check.
        """
        if not self.tractogram_data or roi_path not in self.roi_layers: 
            return False
        
        # Ensure we have bounding boxes (calculated on load usually)
        if self.streamline_bboxes is None:
            # Fallback calculation if missing
            self.streamline_bboxes = np.array([
                [np.min(sl, axis=0), np.max(sl, axis=0)] 
                for sl in self.tractogram_data
            ])

        total_fibers = len(self.tractogram_data)
        self.vtk_panel.update_status(f"Computing intersection: {os.path.basename(roi_path)}...")
        self.vtk_panel.update_progress_bar(0, total_fibers, visible=True)
        QApplication.processEvents()
                
        try:
            roi_data = self.roi_layers[roi_path]['data']
            roi_affine = self.roi_layers[roi_path]['affine']
            inv_affine = self.roi_layers[roi_path]['inv_affine']
            dims = roi_data.shape
            
            # BROAD PHASE: Bounding Box Filter 
            # Find the bounding box of the ROI mask in Voxel Space
            roi_indices = np.argwhere(roi_data > 0)
            
            if roi_indices.size == 0:
                self.roi_intersection_cache[roi_path] = set()
                self.vtk_panel.update_status(f"ROI is empty. Found 0.")
                return True

            v_min = np.min(roi_indices, axis=0)
            v_max = np.max(roi_indices, axis=0) + 1 # +1 for exclusive max

            # Create the 8 corners of the ROI BBox in Voxel Space
            corners_vox = np.array([
                [v_min[0], v_min[1], v_min[2]],
                [v_min[0], v_min[1], v_max[2]],
                [v_min[0], v_max[1], v_min[2]],
                [v_min[0], v_max[1], v_max[2]],
                [v_max[0], v_min[1], v_min[2]],
                [v_max[0], v_min[1], v_max[2]],
                [v_max[0], v_max[1], v_min[2]],
                [v_max[0], v_max[1], v_max[2]],
            ])

            # Transform ROI Voxel Corners -> World Space to get World AABB
            corners_world = nib.affines.apply_affine(roi_affine, corners_vox)
            roi_world_min = np.min(corners_world, axis=0)
            roi_world_max = np.max(corners_world, axis=0)

            
            # Add small padding/tolerance to ROI bounds (e.g. 1 voxel size approx) to be safe
            tolerance = 2.0 
            roi_world_min -= tolerance
            roi_world_max += tolerance

            overlap_mask = np.all(self.streamline_bboxes[:, 1] >= roi_world_min, axis=1) & \
                           np.all(self.streamline_bboxes[:, 0] <= roi_world_max, axis=1)
            
            candidate_indices = np.where(overlap_mask)[0]
            
            # Voxel Grid Check 
            intersecting = set()
            
            # Optimization: Pre-fetch affine components for faster dot product inside loop
            T = inv_affine[:3, 3]
            R = inv_affine[:3, :3]

            # Only loop through candidates (usually < 5% of total fibers)
            for i, idx in enumerate(candidate_indices):
                # Update UI less frequently
                if i % 500 == 0:
                    self.vtk_panel.update_progress_bar(i, len(candidate_indices), visible=True)
                    QApplication.processEvents()

                sl = self.tractogram_data[idx]
                
                # Fast manual apply_affine: vox = sl @ R.T + T
                vox_float = np.dot(sl, R.T) + T
                ivox = np.rint(vox_float).astype(int)
                
                # Bounds check (Vectorized for the whole streamline)
                valid_mask = (
                    (ivox[:, 0] >= 0) & (ivox[:, 0] < dims[0]) &
                    (ivox[:, 1] >= 0) & (ivox[:, 1] < dims[1]) &
                    (ivox[:, 2] >= 0) & (ivox[:, 2] < dims[2])
                )
                
                if np.any(valid_mask):
                    valid_points = ivox[valid_mask]
                    if np.any(roi_data[valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]] > 0):
                        intersecting.add(idx)

            self.roi_intersection_cache[roi_path] = intersecting
            self.vtk_panel.update_status(f"Intersection done. Found {len(intersecting)} (Candidates: {len(candidate_indices)}).")
            return True

        except Exception as e:
            logger.warning(f"Intersection Error: {e}")
            self.vtk_panel.update_status("Intersection failed.")
            return False
            
        finally:
            self.vtk_panel.update_progress_bar(0, 0, visible=False)
        
        
    def _update_roi_visual_selection(self) -> None:
        active_selects = [p for p, s in self.roi_states.items() if s['select']]
        combined = set()
        for p in active_selects:
            combined.update(self.roi_intersection_cache.get(p, set()))
        self.roi_highlight_indices = combined
        if self.vtk_panel:
            self.vtk_panel.update_roi_highlight_actor()
            
            
    def _apply_logic_filters(self) -> None:
        if not hasattr(self, 'manual_visible_indices'):
            self.manual_visible_indices = set(range(len(self.tractogram_data))) if self.tractogram_data else set()

        # Start with manual state
        final_indices = self.manual_visible_indices.copy()
        
        # Apply Includes
        active_includes = [p for p, s in self.roi_states.items() if s['include']]
        if active_includes:
            union_includes = set()
            for p in active_includes:
                union_includes.update(self.roi_intersection_cache.get(p, set()))
            # Strict AND: Streamline must be in Manual AND (Include_A OR Include_B)
            final_indices.intersection_update(union_includes)
            
        # Apply Excludes
        active_excludes = [p for p, s in self.roi_states.items() if s['exclude']]
        for p in active_excludes:
            excl = self.roi_intersection_cache.get(p, set())
            final_indices.difference_update(excl)
            
        self.visible_indices = final_indices
        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()
        self._update_bundle_info_display()


    def _change_roi_color_action(self, path: str) -> None:
        """Opens a color picker and updates the ROI layer color."""
        color = QColorDialog.getColor()
        
        if color.isValid():
            # Convert QColor (0-255) to VTK/Normalized format (0.0-1.0)
            rgb_normalized = (color.redF(), color.greenF(), color.blueF())
            
            if self.vtk_panel:
                self.vtk_panel.set_roi_layer_color(path, rgb_normalized)
                self.vtk_panel.update_status(f"Updated color for {os.path.basename(path)}")


    def _remove_roi_layer_action(self, path: str) -> None:
        """Removes a specific ROI layer."""
        if path in self.roi_layers:
            del self.roi_layers[path]
            if path in self.roi_visibility:
                del self.roi_visibility[path]
            
            # Remove from Logic State and Cache 
            if path in self.roi_states:
                del self.roi_states[path]
            
            if path in self.roi_intersection_cache:
                del self.roi_intersection_cache[path]

            # Re-calculate Logic and Visuals 
            self._update_roi_visual_selection() 
            self._apply_logic_filters()
            
            # Update VTK Panel
            if self.vtk_panel:
                self.vtk_panel.remove_roi_layer(path)
                self.vtk_panel.update_status(f"Removed ROI: {os.path.basename(path)}")
            
            # Refresh UI
            self._update_bundle_info_display()
            self._update_action_states()


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


    def _toggle_roi_visibility(self, path: str, visible: bool) -> None:
        """Toggles the visibility of a specific ROI layer."""
        if self.roi_visibility.get(path, True) == visible:
            return # No change
            
        self.roi_visibility[path] = visible
        
        if self.vtk_panel:
            self.vtk_panel.set_roi_layer_visibility(path, visible)
            self.vtk_panel.update_status(
                f"ROI '{os.path.basename(path)}' visibility set to {visible}"
            )


    def _trigger_clear_anatomical_image(self) -> None:
        """Clears the currently loaded anatomical image."""
        if self.anatomical_image_data is None:
            if self.vtk_panel:
                self.vtk_panel.update_status("No anatomical image loaded to clear.")
            return

        if self.vtk_panel:
            self.vtk_panel.update_status("Clearing anatomical image...")

        self.anatomical_image_path = None
        self.anatomical_image_data = None
        self.anatomical_image_affine = None

        # Clear RAS coordinate display 
        if self.ras_coordinate_label: 
            self.ras_coordinate_label.setText(" RAS: --, --, -- ") 

        if self.vtk_panel:
            self.vtk_panel.clear_anatomical_slices()
            self.vtk_panel.update_status("Anatomical image cleared.")
            if not self.tractogram_data and self.vtk_panel.scene:
                 self.vtk_panel.scene.reset_camera()
                 self.vtk_panel.scene.reset_clipping_range()
            if self.vtk_panel.render_window:
                 self.vtk_panel.render_window.Render()

        self._update_bundle_info_display()
        self._update_action_states()


    # Helper functions for float <-> int mapping 
    def _float_to_int_slider(self, float_val: float) -> int:
        """Maps a float value from the data range to the slider's integer range."""
        data_min = self.scalar_data_min
        data_max = self.scalar_data_max
        
        if (data_max - data_min) == 0:
            return 0 
            
        # Clamp value to be within the data range
        float_val = max(data_min, min(data_max, float_val))
            
        percent = (float_val - data_min) / (data_max - data_min)
        return int(round(percent * SLIDER_PRECISION))
        
        
    def _int_slider_to_float(self, slider_val: int) -> float:
        """Maps an integer slider value back to the float data range."""
        data_min = self.scalar_data_min
        data_max = self.scalar_data_max
        
        if (data_max - data_min) == 0:
            return data_min
            
        percent = float(slider_val) / SLIDER_PRECISION
        return data_min + percent * (data_max - data_min)


    def _update_scalar_data_range(self) -> None:
        """Calculates the min/max range from the active scalar data."""
        if not self.active_scalar_name or not self.scalar_data_per_point:
            logger.info("Scalar range: No active scalar data to calculate range from.")
            return

        scalar_sequence = self.scalar_data_per_point.get(self.active_scalar_name)
        if not scalar_sequence:
             logger.info("Scalar range: Active scalar list is empty.")
             return
             
        try:
            valid_scalars = (s for s in scalar_sequence if s is not None and s.size > 0)

            all_scalars_flat = np.concatenate(list(valid_scalars))
            if all_scalars_flat.size == 0:
                logger.info("Scalar range: Concatenated scalar data is empty.")
                return

            data_min_val = np.min(all_scalars_flat)
            data_max_val = np.max(all_scalars_flat)

            # Handle edge case where all data is the same value
            if data_min_val == data_max_val:
                self.scalar_data_min = data_min_val - 0.5
                self.scalar_data_max = data_max_val + 0.5
            else:
                self.scalar_data_min = data_min_val
                self.scalar_data_max = data_max_val

            self.scalar_min_val = self.scalar_data_min
            self.scalar_max_val = self.scalar_data_max

            self._update_scalar_range_widgets() 

        except Exception as e:
            logger.warning(f"Error calculating scalar data range: {e}")
            self.scalar_data_min = 0.0
            self.scalar_data_max = 1.0
            self.scalar_min_val = 0.0
            self.scalar_max_val = 1.0
            self._update_scalar_range_widgets()


    def _update_scalar_range_widgets(self) -> None:
        """Updates the spinbox and slider widgets with current range and values."""
        if not self.scalar_min_spinbox or not self.scalar_max_spinbox:
            return
            
        # Block signals to prevent feedback loops
        self.scalar_min_spinbox.blockSignals(True)
        self.scalar_max_spinbox.blockSignals(True)
        self.scalar_min_slider.blockSignals(True)
        self.scalar_max_slider.blockSignals(True)

        # Set the allowed range for the spinboxes
        self.scalar_min_spinbox.setRange(self.scalar_data_min, self.scalar_data_max)
        self.scalar_max_spinbox.setRange(self.scalar_data_min, self.scalar_data_max)

        # Set the current values
        self.scalar_min_spinbox.setValue(self.scalar_min_val)
        self.scalar_max_spinbox.setValue(self.scalar_max_val)
        
        # Set the slider values
        self.scalar_min_slider.setValue(self._float_to_int_slider(self.scalar_min_val))
        self.scalar_max_slider.setValue(self._float_to_int_slider(self.scalar_max_val))

        # Unblock signals
        self.scalar_min_spinbox.blockSignals(False)
        self.scalar_max_spinbox.blockSignals(False)
        self.scalar_min_slider.blockSignals(False)
        self.scalar_max_slider.blockSignals(False)


    def _slider_value_changed(self, slider_val: int) -> None:
        """
        Slot for when slider value changes.
        Updates the corresponding spinbox, but does NOT trigger VTK update.
        """
        float_val = self._int_slider_to_float(slider_val)
        
        if self.sender() == self.scalar_min_slider:
            self.scalar_min_val = float_val
            self.scalar_min_spinbox.blockSignals(True)
            self.scalar_min_spinbox.setValue(float_val)
            self.scalar_min_spinbox.blockSignals(False)
            # Ensure min slider doesn't cross max slider
            if slider_val > self.scalar_max_slider.value():
                self.scalar_max_slider.blockSignals(True)
                self.scalar_max_slider.setValue(slider_val)
                self.scalar_max_slider.blockSignals(False)
                
        elif self.sender() == self.scalar_max_slider:
            self.scalar_max_val = float_val
            self.scalar_max_spinbox.blockSignals(True)
            self.scalar_max_spinbox.setValue(float_val)
            self.scalar_max_spinbox.blockSignals(False)
            # Ensure max slider doesn't cross min slider
            if slider_val < self.scalar_min_slider.value():
                self.scalar_min_slider.blockSignals(True)
                self.scalar_min_slider.setValue(slider_val)
                self.scalar_min_slider.blockSignals(False)


    def _spinbox_value_changed(self) -> None:
        """
        Slot for when spinbox editing is finished.
        Updates sliders and triggers VTK update.
        """
        min_val = self.scalar_min_spinbox.value()
        max_val = self.scalar_max_spinbox.value()

        # Ensure min <= max
        if min_val > max_val:
            if self.sender() == self.scalar_min_spinbox: max_val = min_val
            else: min_val = max_val
            
        self.scalar_min_val = min_val
        self.scalar_max_val = max_val
        
        self._update_scalar_range_widgets() # Resyncs both sliders and spinboxes
        self._trigger_vtk_update() # Trigger the slow update
        

    def _reset_scalar_range(self) -> None:
        """Slot to reset the scalar range to the data's full range."""
        self.scalar_min_val = self.scalar_data_min
        self.scalar_max_val = self.scalar_data_max
        self._update_scalar_range_widgets()
        self._trigger_vtk_update() # Manually trigger the changed signal to force a redraw
        
        
    def _trigger_vtk_update(self) -> None:
        """
        Validates range and triggers the (slow) VTK actor update.
        Called on slider release or spinbox edit finished.
        """
        # Validation 
        min_val = self.scalar_min_val
        max_val = self.scalar_max_val
        
        if min_val > max_val:
            self.scalar_min_val = max_val
            min_val = max_val
        
        # Update widgets one last time to be sure they are synced
        self._update_scalar_range_widgets()
        
        # Trigger Update
        if self.vtk_panel and self.current_color_mode == ColorMode.SCALAR:
            self.vtk_panel.update_main_streamlines_actor()
            self.vtk_panel.update_status(f"Scalar range set to: [{min_val:.3f}, {max_val:.3f}]")
            
            
    # Slot for VTK Panel
    def update_ras_coordinate_display(self, ras_coords: Optional[np.ndarray]) -> None:
        """
        Updates the RAS coordinate QLineEdit from the VTK panel.
        Called by vtk_panel._update_slow_slice_components.
        
        """
        if not self.ras_coordinate_input:
            return

        # Block signals to prevent _on_ras_coordinate_entered from firing
        self.ras_coordinate_input.blockSignals(True)

        if ras_coords is not None and len(ras_coords) == 3:
            display_x = -ras_coords[0]
            
            coord_str = f"{display_x:.2f}, {ras_coords[1]:.2f}, {ras_coords[2]:.2f}"
            self.ras_coordinate_input.setText(coord_str)
        else:
            # Update placeholder text 
            self.ras_coordinate_input.setText("--, --, --")

        # Unblock signals
        self.ras_coordinate_input.blockSignals(False)


    @pyqtSlot()
    def _on_ras_coordinate_entered(self) -> None:
        """
        Parses the RAS coordinate QLineEdit and tells VTKPanel to move the slices.
        """
        if not self.ras_coordinate_input or not self.vtk_panel:
            return

        # Check if an image is loaded (for the affine)
        if self.anatomical_image_data is None:
            self.vtk_panel.update_status("Error: Cannot set RAS, no anatomical image loaded.")
            # Revert text to current (None)
            self.update_ras_coordinate_display(None)
            return

        text_value = self.ras_coordinate_input.text()

        # Parse the text
        try:
            # Split by comma or space
            parts = text_value.replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError(f"Expected 3 coordinates, got {len(parts)}")

            ras_x_input = float(parts[0]) # This is neurological (+X=Right)
            ras_y = float(parts[1])
            ras_z = float(parts[2])

            # Negate the X-coordinate 
            # Convert from user's neurological input (+X=Right)
            # to the application's internal radiological convention (-X=Right)
            internal_ras_x = -ras_x_input
            ras_coords = np.array([internal_ras_x, ras_y, ras_z])

            # Send to VTKPanel
            self.vtk_panel.set_slices_from_ras(ras_coords)

        except (ValueError, TypeError) as e:
            self.vtk_panel.update_status(f"Error: Invalid RAS format. Use 'x, y, z'. ({e})")
            
            # Revert text to whatever the vtk_panel currently thinks is the coordinate
            current_ras = None
            if self.vtk_panel.current_slice_indices['x'] is not None:
                c = self.vtk_panel.current_slice_indices
                main_affine = self.anatomical_image_affine
                if main_affine is not None:
                        current_ras = self.vtk_panel._voxel_to_world([c['x'], c['y'], c['z']], main_affine)
            
            self.update_ras_coordinate_display(current_ras) # Revert to last known good value
    
    
    # Window Close Event 
    def closeEvent(self, event: QCloseEvent) -> None:
        """Handles the main window close event, prompting if data is loaded."""
        data_loaded = bool(self.tractogram_data or self.anatomical_image_data)
        prompt_message = "Data (streamlines and/or image) is currently loaded.\nAre you sure you want to quit?"

        if data_loaded:
            reply = QMessageBox.question(self, 'Confirm Quit', prompt_message,
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self._cleanup_vtk()
                event.accept()
            else:
                event.ignore()
        else:
            self._cleanup_vtk()
            event.accept()


    def _cleanup_vtk(self) -> None:
        """Safely cleans up VTK resources."""
        if hasattr(self, 'vtk_panel') and self.vtk_panel:
            if hasattr(self.vtk_panel, 'scene') and self.vtk_panel.scene:
                 try:
                     self.vtk_panel.scene.clear()
                 except Exception as e:
                     logger.error(f"Error clearing FURY scene: {e}")

            if hasattr(self.vtk_panel, 'interactor') and self.vtk_panel.interactor:
                try:
                    if self.vtk_panel.interactor.GetInitialized():
                        self.vtk_panel.interactor.TerminateApp()
                    self.vtk_panel.interactor.RemoveAllObservers()
                except Exception as e:
                    logger.error(f"Error terminating/cleaning VTK interactor: {e}")

            if hasattr(self.vtk_panel, 'render_window') and self.vtk_panel.render_window:
                try:
                    self.vtk_panel.render_window.Finalize()
                except Exception as e:
                    logger.error(f"Error finalizing VTK render window: {e}")


    # Help-About dialog 
    def _show_about_dialog(self) -> None:
        """Displays the About tractedit information box with the application logo."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About tractedit")
        
        # Load and Set Logo 
        try:
            logo_path = get_asset_path("logo.png")
            pixmap = QPixmap(logo_path)
            
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    150, 150, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                msg_box.setIconPixmap(scaled_pixmap)
        except Exception as e:
            logger.warning(f"Could not load logo for About dialog: {e}")

        about_text = """<b>TractEdit version 2.1.5</b><br><br>
        Author: Marco Tagliaferri, PhD Candidate in Neuroscience<br>
        Center for Mind/Brain Sciences (CIMeC)
        University of Trento, Italy
        <br><br>
        Contacts:<br>
        marco.tagliaferri@unitn.it<br>
        marco.tagliaferri93@gmail.com
        """
        msg_box.setText(about_text)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()