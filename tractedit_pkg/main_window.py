# -*- coding: utf-8 -*-

"""
Contains the MainWindow class for the tractedit GUI application.

Handles the main application window, menus, actions, status bar,
and coordinates interactions between UI elements, data state,
file I/O, and the VTK panel.
"""

import os
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QMenuBar, QFileDialog,
    QMessageBox, QLabel, QStatusBar, QApplication
)
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup, QIcon
from PyQt6.QtCore import Qt, pyqtSlot

# --- Local Imports ---
from . import file_io # For loading/saving functions
from .utils import (
    ColorMode, get_formatted_datetime, get_asset_path, format_tuple,
    MAX_STACK_LEVELS, DEFAULT_SELECTION_RADIUS, MIN_SELECTION_RADIUS,
    RADIUS_INCREMENT
)
from .vtk_panel import VTKPanel

# --- Main GUI class ---
class MainWindow(QMainWindow):
    """
    Main application window for TractEdit.
    Sets up the UI, manages application state (streamlines, selection, undo/redo),
    and delegates rendering/interaction to VTKPanel and file I/O to file_io.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Initialize Streamline Data Variables ---
        self.streamlines_list = []
        self.original_trk_header = None # Header dict from loaded file
        self.original_trk_affine = None # Affine matrix (affine_to_rasmm)
        self.original_trk_path = None # Full path
        self.original_file_extension = None # '.trk', '.tck', '.trx', or None
        self.scalar_data_per_point = None # Dictionary: {scalar_name: [scalar_array_sl0, ...]}
        self.active_scalar_name = None # Key for the currently active scalar
        self.selected_streamline_indices = set() # Indices of selected streamlines
        self.selection_radius_3d = DEFAULT_SELECTION_RADIUS # Radius for sphere selection

        # --- Initialize Anatomical Image Data Variables ---
        self.anatomical_image_path = None
        self.anatomical_image_data = None # Numpy array
        self.anatomical_image_affine = None # 4x4 numpy array

        # Undo/Redo Stacks (only for streamline deletions)
        self.undo_stack = []
        self.redo_stack = []

        # View State
        self.current_color_mode = ColorMode.DEFAULT

        # --- Window Properties ---
        self.setWindowTitle("Tractedit GUI (PyQt6) - Interactive trk/tck Editor")
        self.setGeometry(100, 100, 1100, 850)

        # --- Setup UI Components ---
        self._create_actions()
        self._create_menus()
        self._setup_status_bar()
        self._setup_central_widget() # This creates the VTKPanel

        # --- Initial Status Update ---
        self._update_initial_status()
        self._update_action_states()
        self._update_bundle_info_display()

    def _create_actions(self):
        """Creates QAction objects used in menus and potentially toolbars."""

        # --- File Actions ---
        self.load_file_action = QAction("&Load trk/tck/trx...", self)
        self.load_file_action.setStatusTip("Load a trk, tck or trx streamline file")
        self.load_file_action.triggered.connect(self._trigger_load_streamlines)

        # --- Load Anatomical Image Action ---
        self.load_bg_image_action = QAction("Load &Image...", self)
        self.load_bg_image_action.setStatusTip("Load a NIfTI image (.nii, .nii.gz) as background")
        self.load_bg_image_action.triggered.connect(self._trigger_load_anatomical_image)
        self.load_bg_image_action.setEnabled(False)

        self.close_bundle_action = QAction("&Close Bundle", self)
        self.close_bundle_action.setStatusTip("Close the current streamline bundle (also clears image)") # Updated tip
        self.close_bundle_action.triggered.connect(self._close_bundle)
        self.close_bundle_action.setEnabled(False)

        # --- Clear Anatomical Image Action ---
        self.clear_bg_image_action = QAction("Clear Anatomical Image", self)
        self.clear_bg_image_action.setStatusTip("Remove the background anatomical image")
        self.clear_bg_image_action.triggered.connect(self._trigger_clear_anatomical_image)
        self.clear_bg_image_action.setEnabled(False) # Enabled only when image loaded


        self.save_file_action = QAction("&Save As...", self)
        self.save_file_action.setStatusTip("Save the modified streamlines to a trk, tck or trx file")
        self.save_file_action.triggered.connect(self._trigger_save_streamlines)
        self.save_file_action.setEnabled(False)

        self.screenshot_action = QAction("Save &Screenshot", self)
        self.screenshot_action.setStatusTip("Save a screenshot of the current view (bundle and image)")
        self.screenshot_action.setShortcut("Ctrl+P")
        self.screenshot_action.triggered.connect(self._trigger_screenshot)
        self.screenshot_action.setEnabled(False)

        self.exit_action = QAction("&Exit", self)
        self.exit_action.setStatusTip("Exit the application")
        self.exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self.exit_action.triggered.connect(self.close)

        # --- Edit Actions ---
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setStatusTip("Undo the last deletion")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo) # Ctrl+Z
        self.undo_action.triggered.connect(self._perform_undo)
        self.undo_action.setEnabled(False)

        self.redo_action = QAction("&Redo", self)
        self.redo_action.setStatusTip("Redo the last undone deletion")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo) # Ctrl+Y / Ctrl+Shift+Z
        self.redo_action.triggered.connect(self._perform_redo)
        self.redo_action.setEnabled(False)

        # --- View Actions (Coloring) ---
        self.coloring_action_group = QActionGroup(self)
        self.coloring_action_group.setExclusive(True)

        self.color_default_action = QAction("Default Color", self)
        self.color_default_action.setStatusTip("Color streamlines with default gray")
        self.color_default_action.setCheckable(True)
        self.color_default_action.setChecked(True) # Default mode
        self.color_default_action.triggered.connect(lambda: self._set_color_mode(ColorMode.DEFAULT))
        self.coloring_action_group.addAction(self.color_default_action)
        self.color_default_action.setEnabled(False)

        self.color_orientation_action = QAction("Color by Orientation", self)
        self.color_orientation_action.setStatusTip("Color streamlines by local orientation (RGB)")
        self.color_orientation_action.setCheckable(True)
        self.color_orientation_action.triggered.connect(lambda: self._set_color_mode(ColorMode.ORIENTATION))
        self.coloring_action_group.addAction(self.color_orientation_action)
        self.color_orientation_action.setEnabled(False)

        self.color_scalar_action = QAction("Color by Scalar", self)
        self.color_scalar_action.setStatusTip("Color streamlines by the first loaded scalar value per point")
        self.color_scalar_action.setCheckable(True)
        self.color_scalar_action.triggered.connect(lambda: self._set_color_mode(ColorMode.SCALAR))
        self.coloring_action_group.addAction(self.color_scalar_action)
        self.color_scalar_action.setEnabled(False)


        # --- Command Actions ---
        self.clear_select_action = QAction("&Clear Selection", self)
        self.clear_select_action.setStatusTip("Clear the current streamline selection (C)")
        self.clear_select_action.setShortcut("C")
        self.clear_select_action.triggered.connect(self._perform_clear_selection)
        self.clear_select_action.setEnabled(False)

        self.delete_select_action = QAction("&Delete Selection", self)
        self.delete_select_action.setStatusTip("Delete the selected streamlines (D)")
        self.delete_select_action.setShortcut("D")
        self.delete_select_action.triggered.connect(self._perform_delete_selection)
        self.delete_select_action.setEnabled(False)

        self.increase_radius_action = QAction("&Increase Radius", self)
        self.increase_radius_action.setStatusTip(f"Increase the selection sphere radius (+{RADIUS_INCREMENT}mm)")
        self.increase_radius_action.setShortcut("+")
        self.increase_radius_action.triggered.connect(self._increase_radius)
        self.increase_radius_action.setEnabled(False)

        self.decrease_radius_action = QAction("&Decrease Radius", self)
        self.decrease_radius_action.setStatusTip(f"Decrease the selection sphere radius (-{RADIUS_INCREMENT}mm)")
        self.decrease_radius_action.setShortcut("-")
        self.decrease_radius_action.triggered.connect(self._decrease_radius)
        self.decrease_radius_action.setEnabled(False)

        self.hide_sphere_action = QAction("&Hide Selection Sphere", self)
        self.hide_sphere_action.setStatusTip("Hide the blue selection sphere (Esc)")
        self.hide_sphere_action.setShortcut("Esc")
        self.hide_sphere_action.triggered.connect(self._hide_sphere)
        self.hide_sphere_action.setEnabled(True)

        # --- Help Menu ---
        self.about_action = QAction("&About TractEdit...", self)
        self.about_action.setStatusTip("Show information about TractEdit")
        self.about_action.triggered.connect(self._show_about_dialog)

    def _create_menus(self):
        """Creates the main menu bar and populates it with actions."""
        main_bar = self.menuBar()

        # --- File Menu --- 
        file_menu = main_bar.addMenu("&File")
        file_menu.addAction(self.load_file_action)      # Load streamlines
        file_menu.addAction(self.load_bg_image_action)  # Load image (Moved here)
        file_menu.addSeparator()
        file_menu.addAction(self.close_bundle_action)   # Close streamlines (also clears image)
        file_menu.addAction(self.clear_bg_image_action) # Clear image (Moved here)
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
        # Background image actions moved to File menu

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

    def _setup_status_bar(self):
        """Creates and configures the status bar with a permanent widget for bundle/image info."""
        self.status_bar = self.statusBar()
        self.data_info_label = QLabel(" No data loaded ")
        self.data_info_label.setStyleSheet("border: 1px solid grey; padding: 2px;")
        self.status_bar.addPermanentWidget(self.data_info_label)

    def _setup_central_widget(self):
        """Sets up the main central widget which will contain the VTK panel."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.vtk_panel = VTKPanel(parent_widget=self.central_widget, main_window_ref=self)

    def _update_initial_status(self):
        """Sets the initial status message in the VTK panel."""
        date_str = get_formatted_datetime()
        self.vtk_panel.update_status(f"Ready ({date_str}). Load data.")

    def _update_action_states(self):
            """Enables/disables actions based on current application state."""
            has_streamlines = bool(self.streamlines_list)
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

    def _update_bundle_info_display(self):
        """Updates the data information QLabel in the status bar for both streamlines and image."""
        bundle_text = "Bundle: None"
        image_text = "Image: None"

        # Streamline Info (same as before)
        if self.streamlines_list:
            count = len(self.streamlines_list)
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
                 if isinstance(vox_val, (tuple, list, np.ndarray)) and len(vox_val) == 3:
                     vox_str = format_tuple(vox_val, precision=2)
            if 'voxel_order' in header and isinstance(header['voxel_order'], str):
                 order = header['voxel_order']

            bundle_text = (f"Bundle: {filename}{file_type_info} | #: {count} | Dim={dims_str} | "
                           f"VoxSize={vox_str} | Order={order}{scalar_info}")

        # Anatomical Image Info (same as before)
        if self.anatomical_image_data is not None:
            filename = os.path.basename(self.anatomical_image_path) if self.anatomical_image_path else "Unknown"
            shape_str = format_tuple(self.anatomical_image_data.shape, precision=0)
            image_text = f"Image: {filename} | Shape={shape_str}"

        # Combine and Set (same as before)
        separator = " || " if self.streamlines_list and self.anatomical_image_data is not None else " | "
        if not self.streamlines_list and not self.anatomical_image_data:
            final_text = " No data loaded "
        elif self.streamlines_list and self.anatomical_image_data is None:
            final_text = f" {bundle_text} "
        elif not self.streamlines_list and self.anatomical_image_data is not None:
             final_text = f" {image_text} "
        else:
             final_text = f" {bundle_text}{separator}{image_text} "
        self.data_info_label.setText(final_text)


    # --- Undo/Redo Core Logic ---
    def _perform_undo(self):
        """Performs the Undo operation (currently only for streamline deletions)."""
        if self.undo_stack:
            current_state = {'streamlines': list(self.streamlines_list)}
            self.redo_stack.append(current_state)
            if len(self.redo_stack) > MAX_STACK_LEVELS: self.redo_stack.pop(0)

            previous_state = self.undo_stack.pop()
            self.streamlines_list = previous_state['streamlines']
            self.selected_streamline_indices = set()

            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
                self.vtk_panel.update_status(f"Undo successful. Streamlines: {len(self.streamlines_list)}")

            self._update_bundle_info_display()
            self._update_action_states()
        elif self.vtk_panel:
             self.vtk_panel.update_status("Nothing to undo.")

    def _perform_redo(self):
        """Performs the Redo operation (currently only for streamline deletions)."""
        if self.redo_stack:
            current_state = {'streamlines': list(self.streamlines_list)}
            self.undo_stack.append(current_state)
            if len(self.undo_stack) > MAX_STACK_LEVELS: self.undo_stack.pop(0)

            next_state = self.redo_stack.pop()
            self.streamlines_list = next_state['streamlines']
            self.selected_streamline_indices = set()

            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
                self.vtk_panel.update_status(f"Redo successful. Streamlines: {len(self.streamlines_list)}")

            self._update_bundle_info_display()
            self._update_action_states()
        elif self.vtk_panel:
            self.vtk_panel.update_status("Nothing to redo.")

    # --- Command Actions Logic ---
    def _perform_clear_selection(self):
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

    def _perform_delete_selection(self):
        """Deletes the selected streamlines."""
        if not self.selected_streamline_indices:
            if self.vtk_panel: self.vtk_panel.update_status("Delete: No streamlines selected.")
            return
        if self.vtk_panel: self.vtk_panel.update_radius_actor(visible=False)

        num_to_delete = len(self.selected_streamline_indices)
        undo_state = {'streamlines': list(self.streamlines_list)}
        self.undo_stack.append(undo_state)
        if len(self.undo_stack) > MAX_STACK_LEVELS: self.undo_stack.pop(0)
        self.redo_stack = []

        indices_to_delete = self.selected_streamline_indices
        new_streamlines_list = [sl for i, sl in enumerate(self.streamlines_list) if i not in indices_to_delete]

        # --- Handle scalar data (Placeholder - needs proper filtering) ---
        new_scalar_data = None
        if self.scalar_data_per_point:
            # Basic check: if lengths match, assume indices align 
            lengths_ok = True
            for k, v in self.scalar_data_per_point.items():
                if len(v) != len(self.streamlines_list):
                    lengths_ok = False
                    break
            if lengths_ok:
                new_scalar_data = {}
                for key, scalar_list in self.scalar_data_per_point.items():
                     new_scalar_data[key] = [s for i, s in enumerate(scalar_list) if i not in indices_to_delete]
                print("Filtered scalar data (basic).")
            else:
                 print("Warning: Scalar data lengths inconsistent. Cannot filter scalars reliably during deletion.")
                 new_scalar_data = self.scalar_data_per_point 

        # Update state
        self.streamlines_list = new_streamlines_list
        self.scalar_data_per_point = new_scalar_data
        self.selected_streamline_indices = set()

        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor()
            self.vtk_panel.update_status(f"Deleted {num_to_delete} streamlines. Remaining: {len(self.streamlines_list)}.")
        self._update_bundle_info_display()
        self._update_action_states()

    def _increase_radius(self):
        """Increases the selection radius."""
        if not self.streamlines_list: 
            return
        self.selection_radius_3d += RADIUS_INCREMENT
        if self.vtk_panel:
            self.vtk_panel.update_status(f"Selection radius increased to {self.selection_radius_3d:.1f}mm.")
            if self.vtk_panel.radius_actor and self.vtk_panel.radius_actor.GetVisibility():
                center = self.vtk_panel.radius_actor.GetCenter()
                self.vtk_panel.update_radius_actor(center_point=center, radius=self.selection_radius_3d, visible=True)

    def _decrease_radius(self):
        """Decreases the selection radius."""
        if not self.streamlines_list: 
            return
        new_radius = self.selection_radius_3d - RADIUS_INCREMENT
        self.selection_radius_3d = max(MIN_SELECTION_RADIUS, new_radius)
        if self.vtk_panel:
            self.vtk_panel.update_status(f"Selection radius decreased to {self.selection_radius_3d:.1f}mm.")
            if self.vtk_panel.radius_actor and self.vtk_panel.radius_actor.GetVisibility():
                center = self.vtk_panel.radius_actor.GetCenter()
                self.vtk_panel.update_radius_actor(center_point=center, radius=self.selection_radius_3d, visible=True)

    def _hide_sphere(self):
        """Hides the selection sphere."""
        if self.vtk_panel:
            self.vtk_panel.update_radius_actor(visible=False)
            self.vtk_panel.update_status("Selection sphere hidden.")

    # --- View Action Logic ---
    @pyqtSlot(object)
    def _set_color_mode(self, mode):
        """Sets the streamline coloring mode and triggers VTK update."""
        if not isinstance(mode, ColorMode): 
            return
        if not self.streamlines_list:
             self.color_default_action.setChecked(True)
             return

        if self.current_color_mode != mode:
            if mode == ColorMode.SCALAR and not self.active_scalar_name:
                QMessageBox.warning(self, "Coloring Error", "No active scalar data loaded for streamlines.")
                if self.current_color_mode == ColorMode.DEFAULT: self.color_default_action.setChecked(True)
                elif self.current_color_mode == ColorMode.ORIENTATION: self.color_orientation_action.setChecked(True)
                return

            self.current_color_mode = mode
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
                self.vtk_panel.update_status(f"Streamline color mode changed to {mode.name}.")

    # --- GUI Action Methods ---
    def _close_bundle(self):
        """
        Closes the current streamline bundle.
        Also clears anatomical image as a workaround for cleanup crashes.
        """
        if not self.streamlines_list:
            if self.vtk_panel:
                self.vtk_panel.update_status("No bundle open to close.")
            return

        if self.vtk_panel:
            self.vtk_panel.update_status("Closing bundle (also clears image)...") # Status is okay
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
        self.streamlines_list = []
        self.original_trk_header = None
        self.original_trk_affine = None
        self.original_trk_path = None
        self.original_file_extension = None
        self.scalar_data_per_point = None
        self.active_scalar_name = None
        self.undo_stack = []
        self.redo_stack = []
        self.current_color_mode = ColorMode.DEFAULT
        self.color_default_action.setChecked(True)

        # Update VTK (remove main streamline actor)
        if self.vtk_panel:
            self.vtk_panel.update_main_streamlines_actor() # Should remove streamline actor
            self.vtk_panel.update_status("Bundle closed (Image also cleared).") # Keep status clear
            if self.vtk_panel.render_window and self.vtk_panel.render_window.GetInteractor().GetInitialized():
                self.vtk_panel.render_window.Render()

        # Update UI
        self._update_bundle_info_display()
        self._update_action_states()

    # --- Action Trigger Wrappers ---
    def _trigger_load_streamlines(self):
        """Wrapper to call the streamline load function from file_io."""
        file_io.load_streamlines_file(self)

    def _trigger_save_streamlines(self):
        """Wrapper to call the streamline save function from file_io."""
        file_io.save_streamlines_file(self)

    def _trigger_screenshot(self):
        """Wrapper to call the screenshot function in vtk_panel."""
        if not (self.streamlines_list or self.anatomical_image_data):
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
    def _trigger_load_anatomical_image(self):
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

    def _trigger_clear_anatomical_image(self):
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
            if not self.streamlines_list and self.vtk_panel.scene:
                 self.vtk_panel.scene.reset_camera()
                 self.vtk_panel.scene.reset_clipping_range()
            if self.vtk_panel.render_window:
                 self.vtk_panel.render_window.Render()

        self._update_bundle_info_display()
        self._update_action_states()

    # --- Window Close Event ---
    def closeEvent(self, event):
        """Handles the main window close event, prompting if data is loaded."""
        data_loaded = bool(self.streamlines_list or self.anatomical_image_data)
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

    def _cleanup_vtk(self):
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
    def _show_about_dialog(self):
        """Displays the About tractedit information box."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About tractedit")
        about_text = """<b>TractEdit version 1.1.0</b><br><br>
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