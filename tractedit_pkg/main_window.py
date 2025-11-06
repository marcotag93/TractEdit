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

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QMenuBar, QFileDialog,
    QMessageBox, QLabel, QStatusBar, QApplication,
    QToolBar, QDoubleSpinBox, 
    QSlider, 
    QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup, QIcon, QCloseEvent
from PyQt6.QtCore import Qt, pyqtSlot

from . import file_io 
from .utils import (
    ColorMode, get_formatted_datetime, get_asset_path, format_tuple,
    MAX_STACK_LEVELS, DEFAULT_SELECTION_RADIUS, MIN_SELECTION_RADIUS,
    RADIUS_INCREMENT
)
from .vtk_panel import VTKPanel

# --- Constant for slider precision ---
SLIDER_PRECISION = 1000 # Use 1000 steps for the slider

# --- Main GUI class ---
class MainWindow(QMainWindow):
    """
    Main application window for TractEdit.
    Sets up the UI, manages application state (streamlines, selection, undo/redo),
    and delegates rendering/interaction to VTKPanel and file I/O to file_io.
    """
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # --- Initialize Streamline Data Variables ---
        self.tractogram_data: Optional['nib.streamlines.ArraySequence'] = None
        self.visible_indices: Set[int] = set()
        self.original_trk_header: Optional[Dict[str, Any]] = None # Header dict from loaded file
        self.original_trk_affine: Optional[np.ndarray] = None # Affine matrix (affine_to_rasmm)
        self.original_trk_path: Optional[str] = None # Full path
        self.original_file_extension: Optional[str] = None # '.trk', '.tck', '.trx', or None
        self.scalar_data_per_point: Optional[Dict[str, 'nib.streamlines.ArraySequence']] = None # Dictionary: {scalar_name: [scalar_array_sl0, ...]}
        self.active_scalar_name: Optional[str] = None # Key for the currently active scalar
        self.selected_streamline_indices: Set[int] = set() # Indices of selected streamlines
        self.selection_radius_3d: float = DEFAULT_SELECTION_RADIUS # Radius for sphere selection

        # --- Initialize Anatomical Image Data Variables ---
        self.anatomical_image_path: Optional[str] = None
        self.anatomical_image_data: Optional[np.ndarray] = None # Numpy array
        self.anatomical_image_affine: Optional[np.ndarray] = None # 4x4 numpy array

        # Undo/Redo Stacks
        self.undo_stack: List[Set[int]] = []
        self.redo_stack: List[Set[int]] = []

        # View State
        self.current_color_mode: ColorMode = ColorMode.DEFAULT

        # --- Scalar Range Variables ---
        self.scalar_min_val: float = 0.0           # Current min value for the colormap
        self.scalar_max_val: float = 1.0           # Current max value for the colormap
        self.scalar_data_min: float = 0.0          # Actual min value in the loaded data
        self.scalar_data_max: float = 1.0          # Actual max value in the loaded data
        self.scalar_range_initialized: bool = False # Flag to check if range has been calculated
        self.scalar_toolbar: Optional[QToolBar] = None          
        self.scalar_min_spinbox: Optional[QDoubleSpinBox] = None
        self.scalar_max_spinbox: Optional[QDoubleSpinBox] = None
        self.scalar_min_slider: Optional[QSlider] = None
        self.scalar_max_slider: Optional[QSlider] = None

        # --- Window Properties ---
        self.setWindowTitle("TractEdit GUI (PyQt6) - Interactive trk/tck/trx Editor")
        self.setGeometry(100, 100, 1100, 850)

        # --- Setup UI Components ---
        self._create_actions()
        self._create_menus()
        self._create_scalar_toolbar() 
        self._setup_status_bar()
        self._setup_central_widget() # This creates the VTKPanel

        # --- Initial Status Update ---
        self._update_initial_status()
        self._update_action_states()
        self._update_bundle_info_display()

    def _create_actions(self) -> None:
        """Creates QAction objects used in menus and potentially toolbars."""

        # --- File Actions ---
        self.load_file_action: QAction = QAction("&Load trk/tck/trx...", self)
        self.load_file_action.setStatusTip("Load a trk, tck or trx streamline file")
        self.load_file_action.triggered.connect(self._trigger_load_streamlines)

        # --- Load Anatomical Image Action ---
        self.load_bg_image_action: QAction = QAction("Load &Image...", self)
        self.load_bg_image_action.setStatusTip("Load a NIfTI image (.nii, .nii.gz) as background")
        self.load_bg_image_action.triggered.connect(self._trigger_load_anatomical_image)
        self.load_bg_image_action.setEnabled(False) 

        self.close_bundle_action: QAction = QAction("&Close Bundle", self)
        self.close_bundle_action.setStatusTip("Close the current streamline bundle") # Updated tip
        self.close_bundle_action.triggered.connect(self._close_bundle)
        self.close_bundle_action.setEnabled(False)

        # --- Clear Anatomical Image Action ---
        self.clear_bg_image_action: QAction = QAction("Clear Anatomical Image", self)
        self.clear_bg_image_action.setStatusTip("Remove the background anatomical image")
        self.clear_bg_image_action.triggered.connect(self._trigger_clear_anatomical_image)
        self.clear_bg_image_action.setEnabled(False) # Enabled only when image loaded


        self.save_file_action: QAction = QAction("&Save As...", self)
        self.save_file_action.setStatusTip("Save the modified streamlines to a trk, tck or trx file")
        self.save_file_action.triggered.connect(self._trigger_save_streamlines)
        self.save_file_action.setEnabled(False)

        self.screenshot_action: QAction = QAction("Save &Screenshot", self)
        self.screenshot_action.setStatusTip("Save a screenshot of the current view (bundle and image)")
        self.screenshot_action.setShortcut("Ctrl+P")
        self.screenshot_action.triggered.connect(self._trigger_screenshot)
        self.screenshot_action.setEnabled(False)

        self.exit_action: QAction = QAction("&Exit", self)
        self.exit_action.setStatusTip("Exit the application")
        self.exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self.exit_action.triggered.connect(self.close)

        # --- Edit Actions ---
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

        # --- View Actions (Coloring) ---
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

        # --- Command Actions ---
        self.clear_select_action: QAction = QAction("&Clear Selection", self)
        self.clear_select_action.setStatusTip("Clear the current streamline selection (C)")
        self.clear_select_action.setShortcut("C")
        self.clear_select_action.triggered.connect(self._perform_clear_selection)
        self.clear_select_action.setEnabled(False)

        self.delete_select_action: QAction = QAction("&Delete Selection", self)
        self.delete_select_action.setStatusTip("Delete the selected streamlines (D)")
        self.delete_select_action.setShortcut("D")
        self.delete_select_action.triggered.connect(self._perform_delete_selection)
        self.delete_select_action.setEnabled(False)

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

        # --- Help Menu ---
        self.about_action: QAction = QAction("&About TractEdit...", self)
        self.about_action.setStatusTip("Show information about TractEdit")
        self.about_action.triggered.connect(self._show_about_dialog)

    def _create_menus(self) -> None:
        """Creates the main menu bar and populates it with actions."""
        main_bar: QMenuBar = self.menuBar()

        # --- File Menu --- 
        file_menu = main_bar.addMenu("&File")
        file_menu.addAction(self.load_file_action)      # Load streamlines
        file_menu.addAction(self.load_bg_image_action)  # Load image
        file_menu.addSeparator()
        file_menu.addAction(self.close_bundle_action)   # Close streamlines (also clears image)
        file_menu.addAction(self.clear_bg_image_action) # Clear image 
        file_menu.addSeparator()
        file_menu.addAction(self.save_file_action)      # Save streamlines
        file_menu.addSeparator()
        file_menu.addAction(self.screenshot_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        # --- Edit Menu ---
        edit_menu = main_bar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        # --- View Menu ---
        view_menu = main_bar.addMenu("&View")
        color_menu = view_menu.addMenu("Streamline Color")
        color_menu.addAction(self.color_default_action)
        color_menu.addAction(self.color_orientation_action)
        color_menu.addAction(self.color_scalar_action)

        # --- Commands Menu ---
        commands_menu = main_bar.addMenu("&Commands")
        commands_menu.addAction(self.clear_select_action)
        commands_menu.addAction(self.delete_select_action)
        commands_menu.addSeparator()
        commands_menu.addAction(self.increase_radius_action)
        commands_menu.addAction(self.decrease_radius_action)
        commands_menu.addSeparator()
        commands_menu.addAction(self.hide_sphere_action)

        # --- Help Menu ---
        help_menu = main_bar.addMenu("&Help")
        help_menu.addAction(self.about_action)

    # --- Scalar Toolbar ---
    def _create_scalar_toolbar(self) -> None:
        """Creates the toolbar for scalar range adjustment with sliders."""
        self.scalar_toolbar = QToolBar("Scalar Range", self)
        self.scalar_toolbar.setObjectName("ScalarToolbar") # For identification

        # --- Spinboxes for precise input/display ---
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
        
        # --- Sliders for interactive dragging ---
        self.scalar_min_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.scalar_min_slider.setRange(0, SLIDER_PRECISION)
        self.scalar_min_slider.setToolTip("Drag to adjust min scalar value")
        
        self.scalar_max_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.scalar_max_slider.setRange(0, SLIDER_PRECISION)
        self.scalar_max_slider.setValue(SLIDER_PRECISION)
        self.scalar_max_slider.setToolTip("Drag to adjust max scalar value")
        
        # --- Reset Button ---
        self.scalar_reset_button: QAction = QAction("Reset", self)
        self.scalar_reset_button.setStatusTip("Reset scalar range to data min/max")
        
        # --- Layout ---
        toolbar_widget = QWidget(self)
        layout = QHBoxLayout(toolbar_widget)
        layout.setContentsMargins(5, 0, 5, 0) # Tweak spacing
        
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

        # --- Connect Signals ---
        # Sliders update spinbox on valueChanged (fast, no VTK)
        self.scalar_min_slider.valueChanged.connect(self._slider_value_changed)
        self.scalar_max_slider.valueChanged.connect(self._slider_value_changed)
        
        # Sliders update VTK on sliderReleased (slow, final update)
        self.scalar_min_slider.sliderReleased.connect(self._trigger_vtk_update)
        self.scalar_max_slider.sliderReleased.connect(self._trigger_vtk_update)
        
        # Spinboxes update slider and VTK on editingFinished (Enter pressed)
        self.scalar_min_spinbox.editingFinished.connect(self._spinbox_value_changed)
        self.scalar_max_spinbox.editingFinished.connect(self._spinbox_value_changed)
        
        # Reset button
        self.scalar_reset_button.triggered.connect(self._reset_scalar_range)

    def _setup_status_bar(self) -> None:
        """Creates and configures the status bar with a permanent widget for bundle/image info."""
        self.status_bar: QStatusBar = self.statusBar()
        self.data_info_label: QLabel = QLabel(" No data loaded ")
        self.data_info_label.setStyleSheet("border: 1px solid grey; padding: 2px;")
        self.status_bar.addPermanentWidget(self.data_info_label)

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
            has_selection = bool(self.selected_streamline_indices)
            has_scalars = bool(self.scalar_data_per_point)
            has_image = self.anatomical_image_data is not None
            has_any_data = has_streamlines or has_image

            # File Menu
            self.load_bg_image_action.setEnabled(has_streamlines)
            self.close_bundle_action.setEnabled(has_streamlines)
            self.clear_bg_image_action.setEnabled(has_image)
            self.save_file_action.setEnabled(has_streamlines)
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

    def _update_bundle_info_display(self) -> None:
        """Updates the data information QLabel in the status bar for both streamlines and image."""
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

        # Combine and Set
        separator = " || " if self.tractogram_data is not None and self.anatomical_image_data is not None else " | "
        if self.tractogram_data is None and not self.anatomical_image_data:
            final_text = " No data loaded "
        elif self.tractogram_data is not None and self.anatomical_image_data is None:
            final_text = f" {bundle_text} "
        elif self.tractogram_data is None and self.anatomical_image_data is not None:
             final_text = f" {image_text} "
        else:
             final_text = f" {bundle_text}{separator}{image_text} "
        self.data_info_label.setText(final_text)

    # --- Undo/Redo Core Logic ---             
    def _perform_undo(self) -> None:
        """
        Performs the Undo operation (re-inserts deleted streamlines).
        """
        if not self.undo_stack:
            if self.vtk_panel:
                self.vtk_panel.update_status("Nothing to undo.")
            return

        # Pop the set of indices that were deleted
        indices_to_restore: Set[int] = self.undo_stack.pop()
        
        # Add them back to the visible set
        self.visible_indices.update(indices_to_restore)

        # Push the command to the redo stack
        self.redo_stack.append(indices_to_restore)
        if len(self.redo_stack) > MAX_STACK_LEVELS: self.redo_stack.pop(0)
        
        self.selected_streamline_indices = set()

        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()
            self.vtk_panel.update_status(f"Undo successful. Streamlines: {len(self.visible_indices)}")

        self._update_bundle_info_display()
        self._update_action_states()
            
    def _perform_redo(self) -> None:
        """
        Performs the Redo operation (re-deletes streamlines).
        """
        if not self.redo_stack:
            if self.vtk_panel:
                self.vtk_panel.update_status("Nothing to redo.")
            return
            
        # Pop the command
        indices_to_delete_again: Set[int] = self.redo_stack.pop()
        
        # Remove them from the visible set
        self.visible_indices.difference_update(indices_to_delete_again)

        # Push the command back to the undo stack
        self.undo_stack.append(indices_to_delete_again)
        if len(self.undo_stack) > MAX_STACK_LEVELS: self.undo_stack.pop(0)

        self.selected_streamline_indices = set()

        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()
            self.vtk_panel.update_status(f"Redo successful. Streamlines: {len(self.visible_indices)}")

        self._update_bundle_info_display()
        self._update_action_states()

    # --- Command Actions Logic ---
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

    def _perform_delete_selection(self) -> None:
        """
        Deletes the selected streamlines.
        Uses the memory-efficient Command Pattern.
        """
        if not self.selected_streamline_indices:
            if self.vtk_panel: self.vtk_panel.update_status("Delete: No streamlines selected.")
            return
        if self.vtk_panel: self.vtk_panel.update_radius_actor(visible=False)

        num_to_delete = len(self.selected_streamline_indices)
        
        # --- New Undo/Redo Command Logic ---
        indices_to_delete: Set[int] = self.selected_streamline_indices.copy()
        
        # --- Create the command object ---
        self.undo_stack.append(indices_to_delete)
        if len(self.undo_stack) > MAX_STACK_LEVELS: self.undo_stack.pop(0)
        self.redo_stack = [] # Clear redo stack on a new action

        # --- Update state ---
        self.visible_indices.difference_update(indices_to_delete)
        
        self.selected_streamline_indices = set()

        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()
            self.vtk_panel.update_status(f"Deleted {num_to_delete} streamlines. Remaining: {len(self.visible_indices)}.")
        self._update_bundle_info_display()
        self._update_action_states()

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

    # --- View Action Logic ---
    @pyqtSlot(object)
    def _set_color_mode(self, mode: ColorMode) -> None:
        """Sets the streamline coloring mode and triggers VTK update."""
        if not isinstance(mode, ColorMode): 
            return
        if not self.tractogram_data:
             self.color_default_action.setChecked(True)
             return

        # --- Handle scalar toolbar visibility ---
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

            self.current_color_mode = mode
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
                self.vtk_panel.update_status(f"Streamline color mode changed to {mode.name}.")
        
        # Ensure toolbar visibility
        if self.scalar_toolbar:
             is_scalar = (mode == ColorMode.SCALAR and bool(self.active_scalar_name))
             self.scalar_toolbar.setVisible(is_scalar)


    # --- GUI Action Methods ---
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

            # Clear anatomical slices if present
            if self.anatomical_image_data is not None:
                 self.anatomical_image_path = None
                 self.anatomical_image_data = None
                 self.anatomical_image_affine = None
                 self.vtk_panel.clear_anatomical_slices()

        # Reset streamline data state
        self.tractogram_data = None
        self.visible_indices = set()
        self.original_trk_header = None
        self.original_trk_affine = None
        self.original_trk_path = None
        self.original_file_extension = None
        self.scalar_data_per_point = None
        self.active_scalar_name = None
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

        # Update UI
        self._update_bundle_info_display()
        self._update_action_states()

    # --- Action Trigger Wrappers ---
    def _trigger_load_streamlines(self) -> None:
        """Wrapper to call the streamline load function from file_io."""
        self.scalar_range_initialized = False
        if self.scalar_toolbar: self.scalar_toolbar.setVisible(False)
        
        file_io.load_streamlines_file(self)
        
        # --- Update scalar range if scalar mode is already active
        if self.current_color_mode == ColorMode.SCALAR and self.active_scalar_name:
             self._update_scalar_data_range()
             self.scalar_range_initialized = True
             if self.scalar_toolbar: self.scalar_toolbar.setVisible(True)

    def _trigger_save_streamlines(self) -> None:
        """Wrapper to call the streamline save function from file_io."""
        file_io.save_streamlines_file(self)

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

    # --- Background Image Methods ---
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

        img_data, img_affine, img_path = file_io.load_anatomical_image(self)

        if img_data is not None and img_affine is not None and img_path:
            self.anatomical_image_data = img_data
            self.anatomical_image_affine = img_affine
            self.anatomical_image_path = img_path

            if self.vtk_panel:
                self.vtk_panel.update_anatomical_slices()
                if self.vtk_panel.scene:
                    self.vtk_panel.scene.reset_camera()
                    self.vtk_panel.scene.reset_clipping_range()
                if self.vtk_panel.render_window:
                    self.vtk_panel.render_window.Render()

            self._update_bundle_info_display()
            self._update_action_states()

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

    # --- Helper functions for float <-> int mapping ---
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
            print("Scalar range: No active scalar data to calculate range from.")
            return

        scalar_sequence = self.scalar_data_per_point.get(self.active_scalar_name)
        if not scalar_sequence:
             print("Scalar range: Active scalar list is empty.")
             return
             
        try:
            valid_scalars = (s for s in scalar_sequence if s is not None and s.size > 0)

            all_scalars_flat = np.concatenate(list(valid_scalars))
            if all_scalars_flat.size == 0:
                print("Scalar range: Concatenated scalar data is empty.")
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

            self._update_scalar_range_widgets() # This will update spinboxes and sliders

        except Exception as e:
            print(f"Error calculating scalar data range: {e}")
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
        # --- Validation ---
        min_val = self.scalar_min_val
        max_val = self.scalar_max_val
        
        if min_val > max_val:
            self.scalar_min_val = max_val
            min_val = max_val
        
        # Update widgets one last time to be sure they are synced
        self._update_scalar_range_widgets()
        
        # --- Trigger Update ---
        if self.vtk_panel and self.current_color_mode == ColorMode.SCALAR:
            self.vtk_panel.update_main_streamlines_actor()
            self.vtk_panel.update_status(f"Scalar range set to: [{min_val:.3f}, {max_val:.3f}]")

    # --- Window Close Event ---
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
                     print(f"Error clearing FURY scene: {e}")

            if hasattr(self.vtk_panel, 'interactor') and self.vtk_panel.interactor:
                try:
                    if self.vtk_panel.interactor.GetInitialized():
                        self.vtk_panel.interactor.TerminateApp()
                    self.vtk_panel.interactor.RemoveAllObservers()
                except Exception as e:
                    print(f"Error terminating/cleaning VTK interactor: {e}")

            if hasattr(self.vtk_panel, 'render_window') and self.vtk_panel.render_window:
                try:
                    self.vtk_panel.render_window.Finalize()
                except Exception as e:
                    print(f"Error finalizing VTK render window: {e}")

    # --- Help-About dialog ---
    def _show_about_dialog(self) -> None:
        """Displays the About tractedit information box."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About tractedit")
        about_text = """<b>TractEdit version 1.3.0</b><br><br>
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