# -*- coding: utf-8 -*-

"""
Manages the VTK scene, actors, and interactions for TractEdit.
Handles display of streamlines and anatomical image slices.
"""

import os
import numpy as np
import vtk
import traceback
from PyQt6.QtWidgets import QVBoxLayout, QMessageBox, QApplication, QFileDialog
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from fury import window, actor, colormap, ui

# --- Local Imports ---
from .utils import ColorMode

class VTKPanel:
    """
    Manages the VTK rendering window, scene, actors (streamlines, image slices),
    and interactions.
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
        self.scene.background((0.1, 0.1, 0.1))

        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.scene)

        self.interactor = self.render_window.GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # --- Actor Placeholders ---
        self.streamlines_actor = None
        self.highlight_actor = None
        self.radius_actor = None
        self.current_radius_actor_radius = None
        self.axial_slice_actor = None
        self.coronal_slice_actor = None
        self.sagittal_slice_actor = None
        self.status_text_actor = None
        self.instruction_text_actor = None
        self.axes_actor = None

        self._create_scene_ui()

        # --- Setup Interaction Callbacks ---
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self.key_press_callback, 1.0)

        # --- Initialize VTK Widget ---
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self.render_window.Render()


    def _create_scene_ui(self):
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
        instruction_text = "[S] Select [D] Del [C] Clear Select | [+/-] Radius | [Ctrl+S] Save Bundle | [Ctrl+Z/Y] Undo/Redo | [Esc] Hide Sphere"
        self.instruction_text_actor.SetInput(instruction_text)
        self.instruction_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self.instruction_text_actor.GetPositionCoordinate().SetValue(0.01, 0.05)
        self.scene.add(self.instruction_text_actor)

        self.axes_actor = actor.axes(scale=(25, 25, 25))
        self.scene.add(self.axes_actor)

    def update_status(self, message):
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
                data_loaded = bool(self.main_window.streamlines_list)

            status_suffix = ""
            if "Deleted" in message and undo_possible: status_suffix = " (Ctrl+Z to Undo)"
            elif "Undo successful" in message:
                status_suffix = f" ({len(self.main_window.undo_stack)} undo remaining"
                status_suffix += ", Ctrl+Y to Redo)" if redo_possible else ")"
            elif "Redo successful" in message:
                status_suffix = f" ({len(self.main_window.redo_stack)} redo remaining"
                status_suffix += ", Ctrl+Z to Undo)" if undo_possible else ")"

            prefix = f"[Radius: {current_radius:.1f}mm] " if data_loaded else ""
            full_message = f"Status: {prefix}{message}{status_suffix}"
            self.status_text_actor.SetInput(str(full_message))

            if self.render_window and self.render_window.GetInteractor().GetInitialized():
                self.render_window.Render()
        except Exception as e:
            print(f"Error updating status text actor: {e}\n{traceback.format_exc()}")

    # --- Selection Sphere Actor Management (using standard VTK) ---
    def _ensure_radius_actor_exists(self, radius, center_point):
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


    def _update_existing_radius_actor(self, center_point, radius, visible):
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

    def update_radius_actor(self, center_point=None, radius=None, visible=False):
        """Creates or updates the selection sphere actor (standard VTK)."""
        if not self.scene: 
            return
        if radius is None and self.main_window: radius = self.main_window.selection_radius_3d

        needs_render = False
        if visible and center_point is not None and radius is not None:
            if self.radius_actor is None:
                 self._ensure_radius_actor_exists(radius, center_point)
                 if self.radius_actor: # Check creation success
                     self.radius_actor.SetVisibility(1 if visible else 0)
                     needs_render = True
            else:
                 needs_render = self._update_existing_radius_actor(center_point, radius, visible)
        elif self.radius_actor is not None:
             needs_render = self._update_existing_radius_actor(None, None, visible)

        if needs_render and self.render_window and self.render_window.GetInteractor().GetInitialized():
            self.render_window.Render()

    # --- Anatomical Slice Actor Management ---
    def update_anatomical_slices(self):
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
            print("Error: Invalid image data or affine provided for slices.")
            self.clear_anatomical_slices()
            return

        # Ensure data is contiguous float32
        if not image_data.flags['C_CONTIGUOUS']:
            image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        else:
            image_data = image_data.astype(np.float32, copy=False)

        # Get dimensions and extents
        shape = image_data.shape[:3]
        center_slices = [s // 2 for s in shape]
        x_extent = (0, shape[0] - 1)
        y_extent = (0, shape[1] - 1)
        z_extent = (0, shape[2] - 1)

        # --- Clear existing slicer actors ---
        self.clear_anatomical_slices()

        try:
            # --- Determine Value Range ---
            img_min = np.min(image_data)
            img_max = np.max(image_data)

            if not np.isfinite(img_min) or not np.isfinite(img_max):
                print("Warning: Image contains non-finite values. Clamping range.")
                finite_data = image_data[np.isfinite(image_data)]
                if finite_data.size > 0: img_min, img_max = np.min(finite_data), np.max(finite_data)
                else: img_min, img_max = 0.0, 1.0

            # Ensure range has distinct min/max
            if img_max <= img_min: value_range = (img_min - 1.0, img_max + 1.0)
            else: value_range = (float(img_min), float(img_max))


            # --- Create Explicit Grayscale VTK Lookup Table ---
            grayscale_lut = vtk.vtkLookupTable()
            grayscale_lut.SetTableRange(value_range[0], value_range[1])
            grayscale_lut.SetSaturationRange(0, 0)  # Grayscale
            grayscale_lut.SetHueRange(0, 0)
            grayscale_lut.SetValueRange(0, 1)       # Black to White
            grayscale_lut.Build()

            # Slicer parameters
            slicer_opacity = 0.8
            interpolation_mode = 'nearest' # Or 'linear'

            # --- Create Slicer Actors using the explicit LUT ---
            # 1. Axial Slice (Z plane)
            self.axial_slice_actor = actor.slicer(
                image_data, affine=affine,
                lookup_colormap=grayscale_lut, # Pass the created LUT here
                opacity=slicer_opacity, interpolation=interpolation_mode
            )
            self.axial_slice_actor.display_extent(x_extent[0], x_extent[1], y_extent[0], y_extent[1], center_slices[2], center_slices[2])
            self.scene.add(self.axial_slice_actor)

            # 2. Coronal Slice (Y plane)
            self.coronal_slice_actor = actor.slicer(
                image_data, affine=affine,
                lookup_colormap=grayscale_lut, # Pass the created LUT here
                opacity=slicer_opacity, interpolation=interpolation_mode
            )
            self.coronal_slice_actor.display_extent(x_extent[0], x_extent[1], center_slices[1], center_slices[1], z_extent[0], z_extent[1])
            self.scene.add(self.coronal_slice_actor)

            # 3. Sagittal Slice (X plane)
            self.sagittal_slice_actor = actor.slicer(
                image_data, affine=affine,
                lookup_colormap=grayscale_lut, # Pass the created LUT here
                opacity=slicer_opacity, interpolation=interpolation_mode
            )
            self.sagittal_slice_actor.display_extent(center_slices[0], center_slices[0], y_extent[0], y_extent[1], z_extent[0], z_extent[1])
            self.scene.add(self.sagittal_slice_actor)

        except TypeError as te:
             error_msg = f"TypeError during slice actor creation/display: {te}"
             print(error_msg); traceback.print_exc()
             QMessageBox.critical(self.main_window, "Slice Actor TypeError", error_msg)
             self.clear_anatomical_slices()
        except Exception as e:
            error_msg = f"Error during anatomical slice actor creation/addition: {e}"
            print(error_msg); traceback.print_exc()
            QMessageBox.critical(self.main_window, "Slice Actor Error", error_msg)
            self.clear_anatomical_slices()

        # Final render
        if self.render_window and self.render_window.GetInteractor().GetInitialized():
            self.render_window.Render()

    def clear_anatomical_slices(self):
        """Removes anatomical slice actors from the scene."""
        actors_to_remove = [self.axial_slice_actor, self.coronal_slice_actor, self.sagittal_slice_actor]
        removed_count = 0
        for act in actors_to_remove:
            if act is not None and self.scene is not None:
                try:
                    self.scene.rm(act); removed_count += 1
                except (ValueError, AttributeError): pass
                except Exception as e: print(f"Error removing slice actor: {e}")

        self.axial_slice_actor, self.coronal_slice_actor, self.sagittal_slice_actor = None, None, None

        if removed_count > 0 and self.render_window and self.render_window.GetInteractor().GetInitialized():
            self.render_window.Render()


    # --- Streamline Actor Management ---
    def update_highlight(self):
        """Updates the actor for highlighted/selected streamlines."""
        if not self.scene: return

        # Safely Remove Existing Highlight Actor
        actor_removed = False
        if self.highlight_actor is not None:
            try:
                self.scene.rm(self.highlight_actor)
                actor_removed = True
            except (ValueError, AttributeError): actor_removed = True 
            except Exception as e: print(f"  Error removing highlight actor: {e}. Proceeding cautiously.")
            finally: self.highlight_actor = None

        # Check prerequisites
        if not self.main_window or \
           not hasattr(self.main_window, 'selected_streamline_indices') or \
           not self.main_window.streamlines_list:
            if self.main_window: self.main_window._update_action_states()
            return

        selected_indices = self.main_window.selected_streamline_indices
        streamlines = self.main_window.streamlines_list

        # Create new actor only if there's a valid selection
        if selected_indices:
            valid_indices = {idx for idx in selected_indices if 0 <= idx < len(streamlines)}
            selected_sl_data = [streamlines[idx] for idx in valid_indices if idx < len(streamlines)]

            if selected_sl_data:
                try:
                    highlight_linewidth = 6 
                    self.highlight_actor = actor.line(
                        selected_sl_data,
                        colors=(1, 1, 0),          # Bright Yellow
                        linewidth=highlight_linewidth, # Make it thicker
                        opacity=1.0                # Fully opaque
                    )
                    self.scene.add(self.highlight_actor)
                except Exception as e:
                     print(f"Error creating highlight actor: {e}\n{traceback.format_exc()}")
                     self.highlight_actor = None

        # Update UI action states
        if self.main_window: self.main_window._update_action_states()


    def _calculate_scalar_colors(self, streamlines, scalar_array_list):
        """Calculates vertex colors based on a list of scalar arrays per streamline."""
        if not scalar_array_list or not streamlines: 
            return None
        if len(scalar_array_list) != len(streamlines): 
            return None

        all_scalars_flat, valid_indices = [], []
        for i, sl_scalars in enumerate(scalar_array_list):
            if sl_scalars is not None and hasattr(sl_scalars, 'size') and sl_scalars.size > 0 and len(sl_scalars) == len(streamlines[i]):
                all_scalars_flat.append(sl_scalars)
                valid_indices.append(i)

        if not all_scalars_flat: 
            return None
        concatenated_scalars = np.concatenate(all_scalars_flat)
        if not concatenated_scalars.size > 0: 
            return None

        scalar_min, scalar_max = np.min(concatenated_scalars), np.max(concatenated_scalars)
        lut = vtk.vtkLookupTable()
        table_min = scalar_min - 0.5 if scalar_min == scalar_max else scalar_min
        table_max = scalar_max + 0.5 if scalar_min == scalar_max else scalar_max
        lut.SetTableRange(table_min, table_max)
        lut.SetHueRange(0.667, 0.0)
        lut.Build()

        default_color_rgba = np.array([128, 128, 128, 255], dtype=np.uint8)
        vertex_colors = []
        rgb_output = [0.0, 0.0, 0.0]
        scalar_idx = 0
        for i, sl in enumerate(streamlines):
            num_points = len(sl)
            if i in valid_indices:
                sl_scalars = all_scalars_flat[scalar_idx]
                sl_colors_rgba = np.empty((num_points, 4), dtype=np.uint8)
                for j in range(num_points):
                    lut.GetColor(sl_scalars[j], rgb_output)
                    sl_colors_rgba[j, 0:3] = [int(c * 255) for c in rgb_output]
                    sl_colors_rgba[j, 3] = 255
                vertex_colors.append(sl_colors_rgba)
                scalar_idx += 1
            else:
                vertex_colors.append(np.array([default_color_rgba] * num_points, dtype=np.uint8))

        return {'colors': vertex_colors, 'opacity': 1.0, 'linewidth': 3}


    def _get_streamline_actor_params(self):
        """Determines parameters for the main streamlines actor."""
        params = {'colors': (0.8, 0.8, 0.8), 'opacity': 0.5, 'linewidth': 2}
        if not self.main_window or not self.main_window.streamlines_list: 
            return params

        streamlines = self.main_window.streamlines_list
        current_mode = self.main_window.current_color_mode

        if current_mode == ColorMode.ORIENTATION:
            try:
                params['colors'] = colormap.line_colors(streamlines)
                params['opacity'] = 0.8
            except Exception as e: print(f"Error calculating orientation colors: {e}. Using default.")
        elif current_mode == ColorMode.SCALAR:
            scalar_data = self.main_window.scalar_data_per_point
            active_scalar = self.main_window.active_scalar_name
            if scalar_data and active_scalar and active_scalar in scalar_data:
                scalar_array_list = scalar_data.get(active_scalar)
                if scalar_array_list:
                    scalar_params = self._calculate_scalar_colors(streamlines, scalar_array_list)
                    if scalar_params: params = scalar_params

        return params

    def update_main_streamlines_actor(self):
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
            except Exception as e: print(f"  Error removing streamline actor: {e}. Proceeding cautiously.")
            finally: self.streamlines_actor = None

        if not self.main_window or not self.main_window.streamlines_list:
            self.update_highlight()
            return

        streamlines = self.main_window.streamlines_list
        actor_params = self._get_streamline_actor_params()

        try:
            self.streamlines_actor = actor.line(streamlines, **actor_params) # Use dictionary unpacking
            self.scene.add(self.streamlines_actor)
        except Exception as e:
            print(f"Error creating main streamlines actor: {e}\n{traceback.format_exc()}")
            QMessageBox.warning(self.main_window, "Actor Error", f"Could not display streamlines: {e}")
            try:
                if self.streamlines_actor:
                    try: self.scene.rm(self.streamlines_actor)
                    except: 
                        pass
                self.streamlines_actor = actor.line(streamlines, colors=(0.8, 0.8, 0.8), opacity=0.5, linewidth=2)
                self.scene.add(self.streamlines_actor)
                if self.main_window:
                     self.main_window.current_color_mode = ColorMode.DEFAULT
                     self.main_window.color_default_action.setChecked(True)
            except Exception as fallback_e:
                print(f"CRITICAL: Error creating fallback streamlines actor: {fallback_e}")
                self.streamlines_actor = None

        self.update_highlight()
        if self.render_window and self.render_window.GetInteractor().GetInitialized():
            self.render_window.Render()


    # --- Interaction Callbacks ---
    def _handle_save_shortcut(self):
        """Handles the Ctrl+S save shortcut logic (saves streamlines)."""
        if not self.main_window: 
            return
        if not self.main_window.streamlines_list:
             self.update_status("Save shortcut (Ctrl+S): No streamlines loaded to save.")
             return
        self.update_radius_actor(visible=False)
        try: self.main_window._trigger_save_streamlines()
        except AttributeError: self.update_status("Error: Save function not found.")
        except Exception as e: self.update_status(f"Error during save: {e}")

    def _handle_radius_change(self, increase=True):
        """Handles increasing or decreasing the selection radius via main window."""
        if not self.main_window: 
            return
        if not self.main_window.streamlines_list:
             self.update_status("Radius change (+/-): No streamlines loaded.")
             return
        if increase: self.main_window._increase_radius()
        else: self.main_window._decrease_radius()

    def _find_streamlines_in_radius(self, center_point, radius, streamlines):
        """Finds indices of streamlines intersecting a sphere."""
        if not streamlines: 
            return set()
        indices_in_radius = set()
        radius_sq = radius * radius
        for idx, sl in enumerate(streamlines):
            if not isinstance(sl, np.ndarray) or sl.ndim != 2 or sl.shape[1] != 3 or sl.size == 0: 
                continue
            try:
                diff_sq = np.sum((sl - center_point)**2, axis=1)
                if np.min(diff_sq) < radius_sq: indices_in_radius.add(idx)
            except Exception as e: print(f"Warning: Error processing streamline {idx} for selection: {e}")
        return indices_in_radius

    def _toggle_selection(self, indices_to_toggle):
        """Toggles the selection state for given indices and updates status/highlight."""
        if not self.main_window or not hasattr(self.main_window, 'selected_streamline_indices'): 
            return
        current_selection = self.main_window.selected_streamline_indices
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

    def _handle_streamline_selection(self):
        """Handles the logic for selecting streamlines triggered by the 's' key."""
        if not self.scene or not self.main_window or not self.main_window.streamlines_list:
            self.update_status("Select ('s'): No streamlines loaded to select from.")
            self.update_radius_actor(visible=False)
            return

        display_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.005)
        picker.Pick(display_pos[0], display_pos[1], 0, self.render_window.GetRenderers().GetFirstRenderer())

        picked_actor = picker.GetActor()
        click_pos_world = picker.GetPickPosition()

        # Basic check if something potentially valid was picked
        if not picked_actor or not click_pos_world or len(click_pos_world) != 3 or picker.GetPointId() < 0:
            self.update_status("Select ('s'): Please click directly on visible streamlines.")
            self.update_radius_actor(visible=False)
            return

        p_center_arr = np.array(click_pos_world)
        radius = self.main_window.selection_radius_3d
        self.update_radius_actor(center_point=p_center_arr, radius=radius, visible=True)
        indices_in_radius = self._find_streamlines_in_radius(p_center_arr, radius, self.main_window.streamlines_list)

        if not indices_in_radius:
             self.update_status("Radius Sel: No streamlines found within radius at click position.")
             self._toggle_selection(set())
        else:
             self._toggle_selection(indices_in_radius)

    def key_press_callback(self, obj, event_id):
        """Handles key press events forwarded from the VTK interactor."""
        if not self.scene or not self.main_window: 
            return

        key_sym = self.interactor.GetKeySym()
        key = key_sym.lower() if key_sym and isinstance(key_sym, str) else ""
        ctrl = self.interactor.GetControlKey() == 1
        shift = self.interactor.GetShiftKey() == 1

        non_data_handlers = {
            ('z', True, False): self.main_window._perform_undo,
            ('y', True, False): self.main_window._perform_redo,
            ('z', True, True): self.main_window._perform_redo,
            ('escape', False, False): self.main_window._hide_sphere,
            ('s', True, False): self._handle_save_shortcut,
        }
        streamline_data_handlers = {
            's': self._handle_streamline_selection,
            'plus': lambda: self._handle_radius_change(increase=True),
            'equal': lambda: self._handle_radius_change(increase=True),
            'minus': lambda: self._handle_radius_change(increase=False),
            'd': self.main_window._perform_delete_selection,
            'c': self.main_window._perform_clear_selection,
        }
        streamline_keys_for_status = {'s', 'plus', 'equal', 'minus', 'd', 'c'}

        handler_key = (key, ctrl, shift)
        if handler_key in non_data_handlers:
            non_data_handlers[handler_key]()
            if self.interactor and self.interactor.GetInitialized(): self.interactor.Render()
            return

        streamlines_loaded = bool(self.main_window.streamlines_list)
        if not streamlines_loaded:
            if key in streamline_keys_for_status:
                self.update_status(f"Action ('{key}') requires streamlines. Load a trk/tck file first.")
            return

        if key in streamline_data_handlers:
             streamline_data_handlers[key]()
             if self.interactor and self.interactor.GetInitialized(): self.interactor.Render()


    def take_screenshot(self):
        """Saves a screenshot of the VTK view with an opaque black background, hiding UI overlays."""
        if not self.render_window or not self.scene:
            QMessageBox.warning(self.main_window, "Screenshot Error", "Render window or scene not available.")
            return
        if not (self.main_window.streamlines_list or self.main_window.anatomical_image_data):
            QMessageBox.warning(self.main_window, "Screenshot Error", "No data loaded to take screenshot of.")
            return

        default_filename = "tractedit_screenshot.png"
        base_name = "tractedit_view"
        if self.main_window.original_trk_path: base_name = os.path.splitext(os.path.basename(self.main_window.original_trk_path))[0]
        elif self.main_window.anatomical_image_path: base_name = os.path.splitext(os.path.basename(self.main_window.anatomical_image_path))[0]
        default_filename = f"{base_name}_screenshot.png"

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
            writer = None
            if output_ext == '.png': writer = vtk.vtkPNGWriter()
            elif output_ext in ['.jpg', '.jpeg']: writer = vtk.vtkJPEGWriter(); writer.SetQuality(95); writer.ProgressiveOn()
            elif output_ext in ['.tif', '.tiff']: writer = vtk.vtkTIFFWriter(); writer.SetCompressionToPackBits()
            else:
                output_path = os.path.splitext(output_path)[0] + ".png"
                writer = vtk.vtkPNGWriter()
                QMessageBox.warning(self.main_window, "Screenshot Info", f"Unsupported extension '{output_ext}'. Saving as PNG.")

            writer.SetFileName(output_path)
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()
            self.update_status(f"Screenshot saved: {os.path.basename(output_path)}")

        except Exception as e:
            error_msg = f"Error taking screenshot:\n{e}\n\n{traceback.format_exc()}"
            print(error_msg); QMessageBox.critical(self.main_window, "Screenshot Error", error_msg)
            self.update_status(f"Error saving screenshot.")
        finally:
            if original_background: self.scene.background(original_background)
            if self.status_text_actor: self.status_text_actor.SetVisibility(original_status_visibility)
            if self.instruction_text_actor: self.instruction_text_actor.SetVisibility(original_instruction_visibility)
            if self.axes_actor: self.axes_actor.SetVisibility(original_axes_visibility)
            if self.radius_actor: self.radius_actor.SetVisibility(original_radius_actor_visibility)

            if self.render_window and self.render_window.GetInteractor().GetInitialized():
                self.render_window.Render()