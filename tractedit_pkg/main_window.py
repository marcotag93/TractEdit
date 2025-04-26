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

        # --- Initialize Data Variables (Managed by MainWindow) ---
        self.streamlines_list = []
        self.original_trk_header = None # Header dict from loaded file
        self.original_trk_affine = None # Affine matrix (affine_to_rasmm)
        self.original_trk_path = None # Full path
        self.original_file_extension = None # '.trk' or '.tck' or None
        self.scalar_data_per_point = None # Dictionary: {scalar_name: [scalar_array_sl0, ...]}
        self.active_scalar_name = None # Key for the currently active scalar
        self.selected_streamline_indices = set() # Indices of selected streamlines
        self.selection_radius_3d = DEFAULT_SELECTION_RADIUS # Radius for sphere selection

        # Undo/Redo Stacks
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
        self.load_file_action = QAction("&Load trk/tck...", self)
        self.load_file_action.setStatusTip("Load a trk or tck streamline file")
        self.load_file_action.triggered.connect(self._trigger_load_file)

        self.close_bundle_action = QAction("&Close Bundle", self)
        self.close_bundle_action.setStatusTip("Close the current streamline bundle")
        self.close_bundle_action.triggered.connect(self._close_bundle)
        self.close_bundle_action.setEnabled(False)

        self.save_file_action = QAction("&Save As...", self)
        self.save_file_action.setStatusTip("Save the modified streamlines to a trk or tck file")
        self.save_file_action.triggered.connect(self._trigger_save_file)
        self.save_file_action.setEnabled(False)

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
        self.color_scalar_action.setEnabled(False) # Enabled only if scalars loaded

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

        self.screenshot_action = QAction("Save &Screenshot", self)
        self.screenshot_action.setStatusTip("Save a screenshot of the current view (bundle only)")
        self.screenshot_action.setShortcut("Ctrl+P")
        self.screenshot_action.triggered.connect(self._trigger_screenshot)
        self.screenshot_action.setEnabled(True) 

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
        file_menu.addAction(self.load_file_action)
        file_menu.addAction(self.close_bundle_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_file_action)
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
        # Add background image menu items, will implemented in future updates
        # view_menu.addSeparator()
        # view_menu.addAction(self.load_bg_image_action)
        # view_menu.addAction(self.clear_bg_image_action)

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
        """Creates and configures the status bar with a permanent widget for bundle info."""
        self.status_bar = self.statusBar()
        self.bundle_info_label = QLabel(" No bundle loaded ")
        self.bundle_info_label.setStyleSheet("border: 1px solid grey; padding: 2px;")
        self.status_bar.addPermanentWidget(self.bundle_info_label)

    def _setup_central_widget(self):
        """Sets up the main central widget which will contain the VTK panel."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.vtk_panel = VTKPanel(self.central_widget, self)

    def _update_initial_status(self):
        """Sets the initial status message in the VTK panel."""
        date_str = get_formatted_datetime() 
        self.vtk_panel.update_status(f"Ready ({date_str}). Load a trk or tck file.")

    def _update_action_states(self):
        """Enables/disables actions based on current application state."""
        has_data = bool(self.streamlines_list) # check 
        has_selection = bool(self.selected_streamline_indices)
        has_scalars = bool(self.scalar_data_per_point)

        # File Menu
        self.close_bundle_action.setEnabled(has_data)
        self.save_file_action.setEnabled(has_data)

        # Edit Menu
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))

        # View Menu
        self.color_default_action.setEnabled(has_data)
        self.color_orientation_action.setEnabled(has_data)
        self.color_scalar_action.setEnabled(has_data and has_scalars)

        # Commands Menu
        self.clear_select_action.setEnabled(has_selection)
        self.delete_select_action.setEnabled(has_selection)
        self.increase_radius_action.setEnabled(has_data)
        self.decrease_radius_action.setEnabled(has_data)

        # Screenshot
        self.screenshot_action.setEnabled(has_data)

    def _update_bundle_info_display(self):
        """Updates the bundle information QLabel in the status bar.
        Handles missing header fields gracefully for different file types (TRK/TCK)."""

        # Default text if no bundle is loaded
        if not self.streamlines_list or self.original_trk_header is None:
            self.bundle_info_label.setText(" No bundle loaded ")
            return

        # Basic information available for both TRK and TCK
        count = len(self.streamlines_list)
        filename = os.path.basename(self.original_trk_path) if self.original_trk_path else "Unknown"
        file_type_info = f" ({self.original_file_extension.upper()})" if self.original_file_extension else ""
        scalar_info = f" | Scalar: {self.active_scalar_name}" if self.active_scalar_name else ""
        header = self.original_trk_header

        # Initialize display strings for optional fields
        dims_str = "N/A"
        vox_str = "N/A"
        order = "N/A" 

        # --- Process TRK-specific fields only if they exist ---
        if 'dimensions' in header:
            dims_raw = header['dimensions']
            # Attempt to parse only if the key exists
            dims_parsed = file_io.parse_numeric_tuple_from_string(dims_raw, int, 3)
            # Use format_tuple only if parsing was successful (check type, not just value)
            if isinstance(dims_parsed, (tuple, list, np.ndarray)) and len(dims_parsed) == 3:
                dims_str = format_tuple(dims_parsed)

        if 'voxel_sizes' in header:
            vox_sizes_raw = header['voxel_sizes']
            # Attempt to parse only if the key exists
            vox_sizes_parsed = file_io.parse_numeric_tuple_from_string(vox_sizes_raw, float, 3)
            # Use format_tuple only if parsing was successful
            if isinstance(vox_sizes_parsed, (tuple, list, np.ndarray)) and len(vox_sizes_parsed) == 3:
                vox_str = format_tuple(vox_sizes_parsed, precision=2)

        # Get voxel_order if it exists (should be str after header prep, but check anyway)
        if 'voxel_order' in header and isinstance(header['voxel_order'], str):
            order = header['voxel_order']

        # --- Construct the final info text ---
        info_text = (f" File: {filename}{file_type_info} | Count={count} | Dim={dims_str} | "
                    f"VoxSize={vox_str} | Order={order}{scalar_info} ")

        self.bundle_info_label.setText(info_text)

    # --- Undo/Redo Core Logic ---
    def _perform_undo(self):
        """Performs the Undo operation."""
        if self.undo_stack:
            # Push current state to redo stack
            self.redo_stack.append(list(self.streamlines_list))
            # Limit redo stack size
            if len(self.redo_stack) > MAX_STACK_LEVELS: self.redo_stack.pop(0)

            # Restore previous state from undo stack
            self.streamlines_list = self.undo_stack.pop()
            self.selected_streamline_indices = set()

            # Update VTK display via vtk_panel
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
                self.vtk_panel.update_status(f"Undo successful. Streamlines: {len(self.streamlines_list)}")

            # Update UI elements
            self._update_bundle_info_display()
            self._update_action_states() # update_highlight
        elif self.vtk_panel:
             self.vtk_panel.update_status("Nothing to undo.")

    def _perform_redo(self):
        """Performs the Redo operation."""
        if self.redo_stack:
            # Push current state to undo stack
            self.undo_stack.append(list(self.streamlines_list))
            if len(self.undo_stack) > MAX_STACK_LEVELS: self.undo_stack.pop(0)

            # Restore state from redo stack
            self.streamlines_list = self.redo_stack.pop()
            self.selected_streamline_indices = set()

            # Update VTK display via vtk_panel
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
                self.vtk_panel.update_status(f"Redo successful. Streamlines: {len(self.streamlines_list)}")

            # Update UI elements
            self._update_bundle_info_display()
            self._update_action_states()
        elif self.vtk_panel:
            self.vtk_panel.update_status("Nothing to redo.")

    # --- Command Actions Logic (Manages MainWindow state, triggers VTK updates) ---
    def _perform_clear_selection(self):
        """Clears the current streamline selection."""
        if self.vtk_panel: self.vtk_panel.update_radius_actor(visible=False) # Hide sphere

        if self.selected_streamline_indices:
            self.selected_streamline_indices = set()
            if self.vtk_panel:
                self.vtk_panel.update_highlight() # Update VTK display
                self.vtk_panel.update_status("Selection cleared.")
        elif self.vtk_panel:
            self.vtk_panel.update_status("Clear: No active selection.")

    def _perform_delete_selection(self):
        """Deletes the selected streamlines."""
        if self.vtk_panel: self.vtk_panel.update_radius_actor(visible=False) # Hide sphere

        if self.selected_streamline_indices:
            num_to_delete = len(self.selected_streamline_indices)

            # Add current state to undo stack
            self.undo_stack.append(list(self.streamlines_list))
            if len(self.undo_stack) > MAX_STACK_LEVELS: self.undo_stack.pop(0)
            self.redo_stack = [] # Clear redo stack on new action

            # Filter streamlines
            indices_to_delete = self.selected_streamline_indices 
            new_streamlines_list = [
                sl for i, sl in enumerate(self.streamlines_list) if i not in indices_to_delete
            ]

            # TODO: Filter scalar data if present
            if self.scalar_data_per_point and self.active_scalar_name:
                 old_scalar_list = self.scalar_data_per_point[self.active_scalar_name]
                 new_scalar_list = [
                     s for i, s in enumerate(old_scalar_list) if i not in indices_to_delete
                 ]
                 self.scalar_data_per_point[self.active_scalar_name] = new_scalar_list
                 # for key in self.scalar_data_per_point:
                 #    self.scalar_data_per_point[key] = [
                 #        s for i, s in enumerate(self.scalar_data_per_point[key])
                 #        if i not in indices_to_delete
                 #    ]

            # Update state
            self.streamlines_list = new_streamlines_list
            self.selected_streamline_indices = set()

            # Update VTK display and UI
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor() # Also calls update_highlight
                self.vtk_panel.update_status(f"Deleted {num_to_delete} streamlines. Remaining: {len(self.streamlines_list)}.")
            self._update_bundle_info_display()
            self._update_action_states() # update_highlight also calls this
        elif self.vtk_panel:
            self.vtk_panel.update_status("Delete: No streamlines selected.")

    def _increase_radius(self):
        """Increases the selection radius."""
        if not self.streamlines_list: 
            return

        self.selection_radius_3d += RADIUS_INCREMENT
        if self.vtk_panel:
            self.vtk_panel.update_status(f"Selection radius increased.")
            if self.vtk_panel.radius_actor and self.vtk_panel.radius_actor.GetVisibility():
                center = self.vtk_panel.radius_actor.GetCenter()
                self.vtk_panel.update_radius_actor(center_point=center, radius=self.selection_radius_3d, visible=True)

    def _decrease_radius(self):
        """Decreases the selection radius."""
        if not self.streamlines_list: 
            return

        self.selection_radius_3d -= RADIUS_INCREMENT
        if self.selection_radius_3d < MIN_SELECTION_RADIUS:
            self.selection_radius_3d = MIN_SELECTION_RADIUS

        if self.vtk_panel:
            self.vtk_panel.update_status(f"Selection radius decreased.")
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
        if not isinstance(mode, ColorMode): # check
            print(f"Warning: Invalid color mode type received: {type(mode)}")
            return

        if self.current_color_mode != mode:
            if mode == ColorMode.SCALAR and not self.scalar_data_per_point:
                QMessageBox.warning(self, "Coloring Error", "No scalar data loaded for streamlines.")
                # Revert radio button selection to the previous state
                if self.current_color_mode == ColorMode.DEFAULT:
                    self.color_default_action.setChecked(True)
                elif self.current_color_mode == ColorMode.ORIENTATION:
                    self.color_orientation_action.setChecked(True)
                return

            # Update the state
            self.current_color_mode = mode
            print(f"Color mode set to: {mode.name}") 

            # Trigger VTK update via the panel
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor()
                self.vtk_panel.update_status(f"Color mode changed to {mode.name}.")

    # --- GUI Action Methods ---
    def _close_bundle(self):
        """Closes the current bundle and resets the application state."""
        if self.streamlines_list:
            if self.vtk_panel: self.vtk_panel.update_status("Closing bundle...")

            # Reset data state
            self.streamlines_list = []
            self.selected_streamline_indices = set()
            self.original_trk_header = None
            self.original_trk_affine = None
            self.original_trk_path = None
            self.original_file_extension = None
            self.scalar_data_per_point = None
            self.active_scalar_name = None
            self.undo_stack = []
            self.redo_stack = []

            # Reset view state
            self.current_color_mode = ColorMode.DEFAULT
            self.color_default_action.setChecked(True)

            # Update VTK display and UI
            if self.vtk_panel:
                self.vtk_panel.update_main_streamlines_actor() # Will clear actors
                self.vtk_panel.update_radius_actor(visible=False) 
                self.vtk_panel.update_status("Bundle closed.")
            self._update_bundle_info_display()
            self._update_action_states()

        elif self.vtk_panel:
            self.vtk_panel.update_status("No bundle open to close.")

    # --- Action Trigger Wrappers ---
    def _trigger_load_file(self):
        """Wrapper to call the load function from file_io."""
        file_io.load_streamlines_file(self)

    def _trigger_save_file(self):
        """Wrapper to call the save function from file_io."""
        file_io.save_streamlines_file(self)

    def _trigger_screenshot(self):
        """Wrapper to call the screenshot function in vtk_panel."""
        if self.vtk_panel:
            try:
                self.vtk_panel.take_screenshot()
            except AttributeError:
                 QMessageBox.warning(self, "Error", "Screenshot function not available.")
            except Exception as e:
                 QMessageBox.critical(self, "Screenshot Error", f"Could not take screenshot:\n{e}")
        else:
            QMessageBox.warning(self, "Screenshot Error", "VTK panel not initialized.")

    # --- ##TODO - Background Image Methods (Placeholders) ---
    def load_background_image(self):
        QMessageBox.information(self, "Not Implemented", "Loading background image is not implemented yet.")
        pass

    def clear_background_image(self):
        QMessageBox.information(self, "Not Implemented", "Clearing background image is not implemented yet.")
        pass

    # --- Window Close Event ---
    def closeEvent(self, event):
        """Handles the main window close event, prompting if data is loaded."""
        data_loaded = bool(self.streamlines_list)

        if data_loaded:
            reply = QMessageBox.question(self, 'Confirm Quit',
                                        "A streamline bundle is currently loaded.\n"
                                        "Are you sure you want to quit?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                print("Closing tractedit GUI")
                self._cleanup_vtk() 
                event.accept()
            else:
                print("Quit cancelled by user.")
                event.ignore()
        else:
            print("Closing tractedit GUI")
            self._cleanup_vtk()
            event.accept()

    def _cleanup_vtk(self):
        """Safely cleans up VTK resources."""
        if hasattr(self, 'vtk_panel') and self.vtk_panel:
            if hasattr(self.vtk_panel, 'interactor') and self.vtk_panel.interactor:
                try:
                    self.vtk_panel.interactor.TerminateApp()
                except Exception as e:
                    print(f"Error terminating VTK interactor: {e}")
            if hasattr(self.vtk_panel, 'render_window') and self.vtk_panel.render_window:
                try:
                    # Release graphics resources
                    self.vtk_panel.render_window.Finalize()
                except Exception as e:
                    print(f"Error finalizing VTK render window: {e}")

    # --- Help-About dialog ---
    def _show_about_dialog(self):
        """Displays the About tractedit information box."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About tractedit")
        about_text = """<b>Tractedit version 1.2</b><br><br>
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
