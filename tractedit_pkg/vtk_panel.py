# -*- coding: utf-8 -*-

"""
Manages the VTK scene, actors, and interactions for TractEdit.
"""

import os
import numpy as np 
import vtk
import traceback
from PyQt6.QtWidgets import QVBoxLayout, QMessageBox, QApplication
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from fury import window, actor, colormap
from PyQt6.QtWidgets import QFileDialog

# --- Local Imports ---
from .utils import ColorMode

class VTKPanel:
    """
    Manages the VTK rendering window, scene, actors, and interactions.
    """
    def __init__(self, parent_widget, main_window_ref):
        """
        Initializes the VTK panel.

        Args:
            parent_widget: The PyQt widget this panel will reside in.
            main_window_ref: A reference to the main MainWindow instance.
        """
        self.main_window = main_window_ref

        # --- VTK Widget Setup ---
        self.vtk_widget = QVTKRenderWindowInteractor(parent_widget)
        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)
        parent_widget.setLayout(layout)

        # --- FURY/VTK Scene Setup ---
        self.scene = window.Scene()
        self.scene.background((0.1, 0.1, 0.1)) # Dark background

        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.scene) 

        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera()) # TrackballCamera 

        # --- Actor Placeholders ---
        self.radius_actor = None # standard vtkActor
        self.current_radius_actor_radius = None
        self.streamlines_actor = None # FURY actor
        self.highlight_actor = None # FURY actor
        self.status_text_actor = None # vtkTextActor
        self.instruction_text_actor = None # vtkTextActor
        self.axes_actor = None # FURY actor

        # --- Create Initial Scene UI ---
        self._create_scene_ui()

        # --- Setup Interaction Callbacks ---
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0)

        # --- Initialize VTK Widget ---
        self.vtk_widget.Start() 

    def _create_scene_ui(self):
        """Creates the initial UI elements (Axes, Text) within the VTK scene."""
        status_prop = vtk.vtkTextProperty()
        status_prop.SetFontSize(16)
        status_prop.SetColor(0.95, 0.95, 0.95) # Light gray
        status_prop.SetFontFamilyToArial()
        status_prop.SetJustificationToLeft()
        status_prop.SetVerticalJustificationToBottom()

        self.status_text_actor = vtk.vtkTextActor()
        self.status_text_actor.SetTextProperty(status_prop)
        self.status_text_actor.SetInput("Status: Initializing...")
        self.status_text_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        self.status_text_actor.GetPositionCoordinate().SetValue(10, 10)
        self.scene.AddActor2D(self.status_text_actor)

        # Instruction Text Actor
        instr_prop = vtk.vtkTextProperty()
        instr_prop.SetFontSize(14)
        instr_prop.SetColor(1.0, 1.0, 1.0) # White
        instr_prop.SetFontFamilyToArial()
        instr_prop.SetJustificationToLeft()
        instr_prop.SetVerticalJustificationToBottom()

        self.instruction_text_actor = vtk.vtkTextActor()
        self.instruction_text_actor.SetTextProperty(instr_prop)
        instruction_text = "S: Select | D: Del | C: Clear | +/-: Radius | Ctrl+S: Save | Ctrl+Z/Y: Undo/Redo | Esc: Hide Sphere"
        self.instruction_text_actor.SetInput(instruction_text)
        self.instruction_text_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        self.instruction_text_actor.GetPositionCoordinate().SetValue(10, 35) 
        self.scene.AddActor2D(self.instruction_text_actor)

        # Axes Actor (bottom-left corner of the 3D scene)
        self.axes_actor = actor.axes(scale=(20, 20, 20)) # Scale to adjust if needed
        self.scene.add(self.axes_actor) 

    def update_status(self, message):
        """Updates the status text displayed in the VTK window."""
        if self.status_text_actor is None:
            return

        try:
            # Access main_window state for context
            undo_possible = bool(self.main_window.undo_stack)
            redo_possible = bool(self.main_window.redo_stack)
            data_loaded = bool(self.main_window.streamlines_list)

            status_suffix = ""
            if "Deleted" in message and undo_possible:
                status_suffix = " (Ctrl+Z to Undo)"
            elif "Undo successful" in message:
                status_suffix = f" ({len(self.main_window.undo_stack)} undo remaining"
                status_suffix += ", Ctrl+Y to Redo)" if redo_possible else ")"
            elif "Redo successful" in message:
                status_suffix = f" ({len(self.main_window.redo_stack)} redo remaining"
                status_suffix += ", Ctrl+Z to Undo)" if undo_possible else ")"
            elif not data_loaded and undo_possible:
                    status_suffix = " (Ctrl+Z to Undo)"

            # Add radius info if data is loaded
            current_radius = self.main_window.selection_radius_3d
            prefix = f"[Radius: {current_radius:.1f}mm] " if data_loaded else ""

            full_message = f"{prefix}{message}{status_suffix}"
            self.status_text_actor.SetInput(str(full_message))

            # Request render update
            if self.render_window:
                self.render_window.Render()
        except Exception as e:
            print(f"Error updating status text actor: {e}")

    def _ensure_actor_exists(self, radius, center_point):
        sphere_source = vtk.vtkSphereSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())
        self.radius_actor = vtk.vtkActor()
        self.radius_actor.SetMapper(mapper)

        # Set initial properties
        sphere_source.SetRadius(radius)
        sphere_source.SetCenter(0, 0, 0) # Center source at origin, position actor
        self.radius_actor.SetPosition(center_point)
        prop = self.radius_actor.GetProperty()
        prop.SetColor(0.2, 0.5, 1.0) # Blue color
        prop.SetOpacity(0.3) # Semi-transparent
        # prop.SetRepresentationToWireframe() # Optional

        self.scene.AddActor(self.radius_actor) # Use AddActor 
        self.current_radius_actor_radius = radius

    def _update_existing_actor(self, center_point, radius, visible):
        needs_update = False

        # --- Ensure Actor Exists if Needed ---
        if visible and center_point is not None and self.radius_actor is None:
            self._ensure_actor_exists(radius, center_point)
            needs_update = True

        if self.radius_actor is None:
            return needs_update

        current_visibility = self.radius_actor.GetVisibility()
        if visible and not current_visibility:
            self.radius_actor.SetVisibility(1)
            needs_update = True
        elif not visible and current_visibility:
            self.radius_actor.SetVisibility(0)
            needs_update = True

        # Update Center and Radius if visible
        if not visible or center_point is None:
            return needs_update
    
        current_position = np.array(self.radius_actor.GetPosition())
        new_position = np.array(center_point)
        if not np.array_equal(current_position, new_position):
            self.radius_actor.SetPosition(new_position)
            needs_update = True

        # Update Radius 
        if radius == self.current_radius_actor_radius:
            return needs_update

        mapper = self.radius_actor.GetMapper()
        if not mapper:
            return needs_update
    
        if not mapper.GetInputConnection(0, 0):
            print("Warning: Mapper has no input connection to get source from.")
            return needs_update

        source = mapper.GetInputConnection(0, 0).GetProducer()
        if not isinstance(source, vtk.vtkSphereSource):
            print("Warning: Could not get vtkSphereSource to update radius.")
            return needs_update
        
        source.SetRadius(radius)
        self.current_radius_actor_radius = radius
        return True

    def update_radius_actor(self, center_point=None, radius=5.0, visible=False):
        """
        Creates or updates the selection sphere actor using standard VTK objects.
        Optimized to modify existing actor properties (visibility, position, radius)
        when possible, recreating only when necessary (first creation).
        
        """
        if not self.scene:
            return

        # --- Update Existing Actor ---
        needs_update = self._update_existing_actor(center_point, radius, visible)

        # Request render update only if something changed
        if needs_update and self.render_window:
            self.render_window.Render()

    def update_highlight(self):
        """Updates the actor for highlighted/selected streamlines."""
        if not self.scene:
            return

        if self.highlight_actor is not None:
            try:
                self.scene.rm(self.highlight_actor) 
            except ValueError: 
                pass 
            self.highlight_actor = None

        # Ensure selection indices set exists
        if self.main_window.selected_streamline_indices is None:
            self.main_window.selected_streamline_indices = set()

        # Create new actor if there are selected streamlines
        if self.main_window.selected_streamline_indices:
            selected_sl_data = []
            if self.main_window.streamlines_list is not None:
                valid_indices = {
                    idx for idx in self.main_window.selected_streamline_indices
                    if 0 <= idx < len(self.main_window.streamlines_list)
                }
                # Update the main window's set if invalid indices were removed
                if len(valid_indices) != len(self.main_window.selected_streamline_indices):
                    self.main_window.selected_streamline_indices = valid_indices

                # Get the actual streamline data for valid indices
                selected_sl_data = [self.main_window.streamlines_list[idx] for idx in valid_indices]

            # Create the FURY actor
            if selected_sl_data:
                try:
                    self.highlight_actor = actor.line(
                        selected_sl_data,
                        colors=(1, 1, 0),    # Bright Yellow
                        linewidth=4,         # Slightly thicker than main lines
                        opacity=1.0          # Fully opaque
                    )
                    self.scene.add(self.highlight_actor) # Use FURY scene method
                except Exception as e:
                     print(f"Error creating highlight actor: {e}")
                     self.highlight_actor = None

        # Request render update
        if self.render_window:
            self.render_window.Render()

        # Update UI action states (e.g., enable/disable delete/clear)
        self.main_window._update_action_states()
        
    def _calculate_scalar_colors(self, streamlines, scalar_array_list):
        """
        Calculates vertex colors based on a list of scalar arrays per streamline.

        Args:
            streamlines (list): The list of streamline coordinate arrays.
            scalar_array_list (list): List of scalar arrays, one per streamline.

        Returns:
            dict: Dictionary containing 'colors', 'opacity', 'linewidth' for the actor,
                  or None if calculation fails.
        """
        default_params = {'colors': (0.8, 0.8, 0.8), 'opacity': 0.5, 'linewidth': 2}
        default_color_rgba = (128, 128, 128, 255) # Gray uint8

        if not scalar_array_list:
            print("Warning: _calculate_scalar_colors received empty scalar_array_list.")
            return None 

        non_empty_scalars = [arr for arr in scalar_array_list if hasattr(arr, 'size') and arr.size > 0]
        if not non_empty_scalars:
            print("Warning: Active scalar data contains only empty arrays. Using default colors.")
            return default_params

        all_scalars_flat = np.concatenate(non_empty_scalars)
        if not all_scalars_flat.size > 0:
            print("Warning: Scalar data arrays concatenated to empty. Using default colors.")
            return default_params

        # --- Calculate Min/Max and Setup LUT ---
        scalar_min, scalar_max = np.min(all_scalars_flat), np.max(all_scalars_flat)
        lut = vtk.vtkLookupTable()
        # Handle constant scalar value case
        table_min = scalar_min - 0.5 if scalar_min == scalar_max else scalar_min
        table_max = scalar_max + 0.5 if scalar_min == scalar_max else scalar_max
        lut.SetTableRange(table_min, table_max)
        lut.SetHueRange(0.667, 0.0)      # Blue to Red
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(0.4, 1.0)
        lut.Build()

        # --- Generate Colors Per Vertex ---
        vertex_colors = []
        rgb_output = [0.0, 0.0, 0.0] # Reusable buffer for GetColor

        for i, sl in enumerate(streamlines):
            num_points = len(sl)
            # Check if valid scalar data exists for this streamline
            if i < len(scalar_array_list) and scalar_array_list[i] is not None and len(scalar_array_list[i]) == num_points:
                sl_scalars = scalar_array_list[i]
                sl_colors_rgba = np.empty((num_points, 4), dtype=np.uint8)
                # Apply LUT to each point scalar
                for j in range(num_points):
                    lut.GetColor(sl_scalars[j], rgb_output)
                    sl_colors_rgba[j, 0] = int(rgb_output[0] * 255)
                    sl_colors_rgba[j, 1] = int(rgb_output[1] * 255)
                    sl_colors_rgba[j, 2] = int(rgb_output[2] * 255)
                    sl_colors_rgba[j, 3] = 255 # Alpha
                vertex_colors.append(sl_colors_rgba)
            else:
                # Use default color for missing/mismatched scalar data
                vertex_colors.append(np.array([default_color_rgba] * num_points, dtype=np.uint8))

        return {'colors': vertex_colors, 'opacity': 1.0, 'linewidth': 3}

    def _get_streamline_actor_params(self):
        """
        Determines parameters (colors, opacity, linewidth) for the main
        streamlines actor based on the current color mode and data.

        Returns:
            dict: Dictionary containing 'colors', 'opacity', 'linewidth'.
        """
        # --- Default parameters ---
        params = {'colors': (0.8, 0.8, 0.8), 'opacity': 0.5, 'linewidth': 2}
        streamlines = self.main_window.streamlines_list 
        current_mode = self.main_window.current_color_mode

        # --- Mode-Specific Parameters ---
        if current_mode == ColorMode.DEFAULT:
            pass
        elif current_mode == ColorMode.ORIENTATION:
            params['colors'] = colormap.line_colors(streamlines)
            params['opacity'] = 0.8
        elif current_mode == ColorMode.SCALAR:
            print(f"Coloring by scalar: {self.main_window.active_scalar_name}")
            scalar_data = self.main_window.scalar_data_per_point
            active_scalar = self.main_window.active_scalar_name

            # Guard clauses for scalar data availability
            if not scalar_data or not active_scalar:
                 print("Warning: Scalar coloring selected but no scalar data loaded/active. Using default.")
                 return params 

            scalar_array_list = scalar_data.get(active_scalar)
            if not scalar_array_list:
                 print(f"Warning: Active scalar '{active_scalar}' not found in data dict. Using default.")
                 return params

            # Attempt to calculate scalar colors
            scalar_params = self._calculate_scalar_colors(streamlines, scalar_array_list)
            if scalar_params:
                params = scalar_params # Use calculated scalar params if successful
            # If _calculate_scalar_colors returned None or default, params remain default

        else:
            print(f"Warning: Unknown color mode {current_mode}. Using default.")

        return params
    
    def update_main_streamlines_actor(self):
        """Recreates the main streamlines actor based on current color mode."""
        if not self.scene:
            return

        # --- Remove existing actor ---
        if self.streamlines_actor is not None:
            try:
                self.scene.rm(self.streamlines_actor)
            except ValueError:
                pass 
            self.streamlines_actor = None

        # --- Check for data ---
        streamlines = self.main_window.streamlines_list
        if not streamlines:
            if self.render_window:
                self.render_window.Render() 
            return

        # --- Get actor parameters using helper ---
        actor_params = self._get_streamline_actor_params()

        # --- Create the new actor ---
        try:
            self.streamlines_actor = actor.line(
                streamlines,
                colors=actor_params['colors'],
                opacity=actor_params['opacity'],
                linewidth=actor_params['linewidth']
            )
            self.scene.add(self.streamlines_actor)

        # --- Handle potential errors during actor creation ---
        except Exception as e:
            print(f"Error creating main streamlines actor: {e}\n{traceback.format_exc()}")
            QMessageBox.warning(self.main_window, "Actor Error", f"Could not display streamlines with selected coloring: {e}")
            try:
                if self.streamlines_actor:
                    try:
                        self.scene.rm(self.streamlines_actor)
                    except ValueError: pass
                self.streamlines_actor = actor.line(streamlines, colors=(0.8, 0.8, 0.8), opacity=0.5, linewidth=2)
                self.scene.add(self.streamlines_actor)
                print("Successfully created fallback default actor.")
            except Exception as fallback_e:
                print(f"CRITICAL: Error creating fallback streamlines actor: {fallback_e}")
                self.streamlines_actor = None 

        # --- Final updates ---
        if self.render_window:
            self.render_window.Render()

        # Update highlight AFTER main actor is potentially replaced/created
        self.update_highlight()
        
    def _handle_save_shortcut(self):
        """Handles the Ctrl+S save shortcut logic."""
        self.update_radius_actor(visible=False) 
        try:
            self.main_window._trigger_save_file()
        except AttributeError:
            print("Error: Save function not found (ensure it's accessible from MainWindow).")
        except Exception as e:
            print(f"Error during Ctrl+S save: {e}")

    def _handle_radius_change(self, increase=True):
        """Handles increasing or decreasing the selection radius."""
        if increase:
            self.main_window._increase_radius()
        else:
            self.main_window._decrease_radius()

        # Update the visual if sphere is visible
        if self.radius_actor and self.radius_actor.GetVisibility():
            center = self.radius_actor.GetPosition()
            self.update_radius_actor(center_point=center,
                                     radius=self.main_window.selection_radius_3d,
                                     visible=True)

    def _find_streamlines_in_radius(self, center_point, radius, streamlines):
        """
        Finds indices of streamlines intersecting a sphere.

        Args:
            center_point (np.ndarray): The 3D center of the sphere.
            radius (float): The radius of the sphere.
            streamlines (list): List of streamline coordinate arrays.

        Returns:
            set: A set containing the indices of streamlines within the radius.
        """
        indices_in_radius = set()
        radius_sq = radius * radius

        # TODO: Coordinate system check/transform if needed

        for idx, sl in enumerate(streamlines):
            if not isinstance(sl, np.ndarray) or sl.ndim != 2 or sl.shape[1] != 3:
                continue 

            # Check if any point in the streamline is within the sphere
            try:
                diff = sl - center_point # Broadcasting center_point
                dist_sq_all_points = np.sum(diff * diff, axis=1)
                if np.any(dist_sq_all_points < radius_sq):
                    indices_in_radius.add(idx)
            except ValueError as ve: # Handle potential shape mismatches during subtraction
                 print(f"Warning: Skipping streamline {idx} due to shape mismatch or error: {ve}")
                 continue
            except Exception as e: # Catch other potential errors per streamline
                 print(f"Warning: Error processing streamline {idx}: {e}")
                 continue

        return indices_in_radius

    def _toggle_selection(self, indices_to_toggle):
        """
        Toggles the selection state for given indices and updates status/highlight.

        Args:
            indices_to_toggle (set): Set of streamline indices to toggle.
        """
        current_selection = self.main_window.selected_streamline_indices
        added_count = 0
        removed_count = 0

        for idx in indices_to_toggle:
            if idx in current_selection:
                current_selection.remove(idx)
                removed_count += 1
            else:
                current_selection.add(idx)
                added_count += 1

        # Update status and highlight only if the selection actually changed
        if added_count > 0 or removed_count > 0:
            total_selected = len(current_selection)
            status_msg = (f"Radius Sel: Found {len(indices_to_toggle)}. "
                          f"Added {added_count}, Removed {removed_count}. "
                          f"Total Sel: {total_selected}")
            self.update_status(status_msg)
            self.update_highlight() # Updates action states

    def _handle_streamline_selection(self):
        """Handles the logic for selecting streamlines triggered by the 's' key."""
        display_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.005) 
        picker.Pick(display_pos[0], display_pos[1], 0, self.scene)

        # --- Validate Pick Target ---
        picked_actor = picker.GetActor()
        is_streamline_actor = (self.streamlines_actor and picked_actor == self.streamlines_actor) or \
                              (self.highlight_actor and picked_actor == self.highlight_actor)
        if not is_streamline_actor:
            self.update_status("Select ('s'): Please click directly on streamlines.")
            self.update_radius_actor(visible=False) 
            return

        # --- Validate Pick Point ---
        picked_point_id = picker.GetPointId()
        if picked_point_id < 0:
            self.update_status("Select ('s'): No point picked close enough on the actor.")
            self.update_radius_actor(visible=False)
            return

        # --- Validate Pick Position ---
        click_pos_world = picker.GetPickPosition()
        if click_pos_world is None or len(click_pos_world) != 3:
            self.update_status("Select ('s'): Could not get valid 3D pick position.")
            self.update_radius_actor(visible=False)
            return

        # --- Perform Selection Logic ---
        p_center_arr = np.array(click_pos_world)
        radius = self.main_window.selection_radius_3d

        # Update sphere visual *before* finding streamlines
        self.update_radius_actor(center_point=p_center_arr, radius=radius, visible=True)

        # Find streamlines within the radius
        indices_in_radius = self._find_streamlines_in_radius(
            p_center_arr, radius, self.main_window.streamlines_list
        )

        # Toggle selection state and update UI
        if not indices_in_radius and self.main_window.selected_streamline_indices:
             self.update_status("Radius Sel: No streamlines found within radius.")  # Only show "no streamlines found" if a selection existed before
             self._toggle_selection(indices_in_radius) 
        elif indices_in_radius:
             self._toggle_selection(indices_in_radius)
        
    def key_press_callback(self, obj, event_id):
        """Handles key press events forwarded from the VTK interactor."""
        if not self.scene: 
            return

        # --- Get Key Info ---
        key_sym = self.interactor.GetKeySym()
        key = key_sym.lower() if key_sym else ""
        ctrl = self.interactor.GetControlKey() == 1
        shift = self.interactor.GetShiftKey() == 1

        # --- Define Handlers ---
        # Format: (key, ctrl_pressed, shift_pressed): handler_method
        non_data_handlers = {
            ('z', True, False): self.main_window._perform_undo,
            ('y', True, False): self.main_window._perform_redo,
            ('z', True, True): self.main_window._perform_redo, # Ctrl+Shift+Z
            ('escape', False, False): self.main_window._hide_sphere,
            ('s', True, False): self._handle_save_shortcut,
        }
        # Format: key: handler_method (require data loaded)
        data_handlers = {
            's': self._handle_streamline_selection,
            'plus': lambda: self._handle_radius_change(increase=True),
            'equal': lambda: self._handle_radius_change(increase=True),
            'minus': lambda: self._handle_radius_change(increase=False),
            'd': self.main_window._perform_delete_selection,
            'c': self.main_window._perform_clear_selection,
        }
        # Keys that require data but only for status update
        data_required_keys_for_status = {'s', 'plus', 'equal', 'minus', 'd', 'c'}

        # --- Dispatch Non-Data Dependent Keys ---
        handler_key = (key, ctrl, shift)
        if handler_key in non_data_handlers:
            non_data_handlers[handler_key]()
            return 

        # --- Data Loaded Check ---
        data_loaded = bool(self.main_window.streamlines_list)
        if not data_loaded:
            if key in data_required_keys_for_status:
                self.update_status("No data loaded. Load a trk or tck file first.")
            return 

        # --- Dispatch Data Dependent Keys ---
        if key in data_handlers:
            data_handlers[key]()
        # else:
            # print(f"Unrecognized key combination: key='{key}', Ctrl={ctrl}, Shift={shift}")
        
    def take_screenshot(self):
        """Saves a screenshot of the VTK view with an opaque black background, hiding overlays."""
        if not self.render_window or not self.scene:
            QMessageBox.warning(self.main_window, "Screenshot Error", "Render window or scene not available.")
            return

        # Suggest a filename 
        default_filename = "tractedit_screenshot.png"
        if self.main_window.original_trk_path:
            base_name = os.path.splitext(os.path.basename(self.main_window.original_trk_path))[0]
            default_filename = f"{base_name}_screenshot.png"

        # Get output path 
        output_path, _ = QFileDialog.getSaveFileName(
            self.main_window, 
            "Save Screenshot",
            default_filename,
            "PNG Image Files (*.png);;JPEG Image Files (*.jpg *.jpeg);;TIFF Image Files (*.tif *.tiff);;All Files (*.*)" # Added TIFF
        )

        if not output_path:
            self.update_status("Screenshot cancelled.")
            return

        self.update_status(f"Preparing screenshot: {os.path.basename(output_path)}...")
        QApplication.processEvents() 

        # --- Store original states ---
        original_background_color = self.scene.GetBackground()
        original_status_visibility = self.status_text_actor.GetVisibility() if self.status_text_actor else 0
        original_instruction_visibility = self.instruction_text_actor.GetVisibility() if self.instruction_text_actor else 0
        original_axes_visibility = self.axes_actor.GetVisibility() if self.axes_actor else 0
        original_radius_actor_visibility = self.radius_actor.GetVisibility() if self.radius_actor else 0

        try:
            # --- Hide overlays ---
            if self.status_text_actor: self.status_text_actor.SetVisibility(0)
            if self.instruction_text_actor: self.instruction_text_actor.SetVisibility(0)
            if self.axes_actor: self.axes_actor.SetVisibility(0)
            if self.radius_actor: self.radius_actor.SetVisibility(0) 

            # --- Set background to black ---
            self.scene.background((0.0, 0.0, 0.0)) # Black

            # --- Render scene changes ---
            self.render_window.Render()

            # --- Capture Image ---
            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(self.render_window)
            window_to_image_filter.SetInputBufferTypeToRGB() # RGB for opaque background
            window_to_image_filter.ReadFrontBufferOff()
            window_to_image_filter.Update()

            # --- Write Image File ---
            _, output_ext = os.path.splitext(output_path)
            output_ext = output_ext.lower()

            writer = None
            if output_ext == '.png':
                writer = vtk.vtkPNGWriter()
            elif output_ext in ['.jpg', '.jpeg']:
                 writer = vtk.vtkJPEGWriter()
                 writer.SetQuality(95)
                 writer.ProgressiveOn()
            elif output_ext in ['.tif', '.tiff']:
                 writer = vtk.vtkTIFFWriter()
                 writer.SetCompressionToPackBits() # A lossless compression
            else:
                default_ext = ".png"
                QMessageBox.information(self.main_window, "Screenshot Info", f"Unsupported or unknown extension '{output_ext}'. Saving as PNG.")
                output_path = os.path.splitext(output_path)[0] + default_ext
                writer = vtk.vtkPNGWriter()

            writer.SetFileName(output_path)
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()

            self.update_status(f"Screenshot saved: {os.path.basename(output_path)}")

        except Exception as e:
            error_msg = f"Error taking screenshot:\n{e}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self.main_window, "Screenshot Error", error_msg)
            self.update_status(f"Error saving screenshot.")

        finally:
            # --- Restore original states ---
            if original_background_color:
                self.scene.background(original_background_color)

            # Restore overlays visibility
            if self.status_text_actor: self.status_text_actor.SetVisibility(original_status_visibility)
            if self.instruction_text_actor: self.instruction_text_actor.SetVisibility(original_instruction_visibility)
            if self.axes_actor: self.axes_actor.SetVisibility(original_axes_visibility)
            if self.radius_actor: self.radius_actor.SetVisibility(original_radius_actor_visibility)

            # Render again 
            self.render_window.Render()