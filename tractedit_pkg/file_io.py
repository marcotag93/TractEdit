# -*- coding: utf-8 -*-

"""
Functions for loading and saving streamline files (trk, tck).
"""

import os
import traceback
import ast
import numpy as np
import nibabel as nib
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication

# --- Local Imports ---
from .utils import ColorMode 

# --- Helper Function ---
def parse_numeric_tuple_from_string(value_str, target_type=float, expected_length=None):
    if not isinstance(value_str, str):
        # If it's already a list or tuple, try to convert types and check length
        if isinstance(value_str, (list, tuple)):
            try:
                converted_tuple = tuple(target_type(x) for x in value_str)
                if expected_length is not None and len(converted_tuple) != expected_length:
                    print(f"Warning: Input sequence {value_str} has length {len(converted_tuple)}, expected {expected_length}.")
                return converted_tuple
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert existing sequence {value_str} to {target_type}: {e}")
                return value_str 
        elif isinstance(value_str, np.ndarray):
             return value_str
        else:
             return value_str

    try:
        # Try standard literal evaluation (handles tuples/lists)
        parsed_val = ast.literal_eval(value_str)
        if isinstance(parsed_val, (list, tuple)):
            result = tuple(target_type(x) for x in parsed_val)
            if expected_length is not None and len(result) != expected_length:
                 print(f"Warning: Parsed tuple {result} from '{value_str}' has length {len(result)}, expected {expected_length}.")
                 return value_str 
            return result
        # If literal_eval results in a single number (e.g., string was "1.0")
        elif isinstance(parsed_val, (int, float)):
             result = (target_type(parsed_val),) # Make it a tuple
             if expected_length is not None and len(result) != expected_length:
                 print(f"Warning: Parsed single number {result} from '{value_str}' has length {len(result)}, expected {expected_length}.")
                 return value_str
             return result

    except (ValueError, SyntaxError, TypeError):
        cleaned_str = value_str.strip('()[]') # Remove brackets
        parts = []
        if ',' in cleaned_str:
            parts = [p.strip() for p in cleaned_str.split(',')]
        else:
            parts = cleaned_str.split() # Split by whitespace

        try:
            result = tuple(target_type(p) for p in parts if p) 
            if expected_length is not None and len(result) != expected_length:
                 print(f"Warning: Parsed tuple {result} from splitting '{value_str}' has length {len(result)}, expected {expected_length}.")
                 return value_str 
            if not result: 
                 raise ValueError("Splitting resulted in empty sequence")
            return result
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse '{value_str}' as a tuple of {target_type} after splitting: {e}")
            return value_str 

    # Fallback: return the original string if none of the methods worked
    return value_str

# --- Helper Function for Scalar Loading ---
def _load_scalar_data(trk_file):
    """Attempts to load scalar data from the tractogram file object."""
    scalar_data = None
    active_scalar = None
    if hasattr(trk_file.tractogram, 'data_per_point') and trk_file.tractogram.data_per_point:
        print("Scalar data found in file (data_per_point).")
        try:
            loaded_scalars_dict = trk_file.tractogram.data_per_point
            if loaded_scalars_dict:
                scalar_data = dict(loaded_scalars_dict)
                active_scalar = list(loaded_scalars_dict.keys())[0]
                print(f"Assigned scalars. Active scalar: '{active_scalar}'")
                # scalar_data[active_scalar] = [np.asarray(arr) for arr in scalar_data[active_scalar]]
            else:
                print("Scalar data dictionary (data_per_point) is empty.")
        except Exception as scalar_e:
            print(f"Warning: Could not process scalar data: {scalar_e}")
            scalar_data = None
            active_scalar = None
    else:
        print("No scalar data found in file (data_per_point).")

    return scalar_data, active_scalar

# --- Helper Function for VTK/UI Update ---
def _update_vtk_and_ui_after_load(main_window, status_msg):
    """Updates VTK panel and main window UI elements after loading."""
    if main_window.vtk_panel:
        main_window.vtk_panel.update_main_streamlines_actor()
        if main_window.vtk_panel.scene:
            main_window.vtk_panel.scene.reset_camera()
            main_window.vtk_panel.scene.reset_clipping_range()
        else:
            print("Warning: vtk_panel.scene not available for camera reset.")

        main_window.vtk_panel.update_status(status_msg)

        if main_window.vtk_panel.render_window:
            main_window.vtk_panel.render_window.Render()
        else:
            print("Warning: render_window not available.")
    else:
        print("Error: vtk_panel not available to update actors.")
        print(f"Status: {status_msg}") 

    # Ensure radio button reflects default state
    if hasattr(main_window, 'color_default_action'):
        main_window.color_default_action.setChecked(True)

    main_window._update_action_states() 
    main_window._update_bundle_info_display() 
    
# --- File I/O Functions ---
def load_streamlines_file(main_window):
    """
    Loads a trk or tck file using nibabel.streamlines.load.
    Updates the MainWindow state, including scalar data if present.
    
    Args:
        main_window: The instance of the main application window.
    """
    if not hasattr(main_window, 'vtk_panel') or not main_window.vtk_panel.scene:
        print("Error: Scene not initialized in vtk_panel.")
        QMessageBox.critical(main_window, "Error", "Scene not initialized.")
        return

    file_filter = "Streamline Files (*.trk *.tck);;TrackVis Files (*.trk);;TCK Files (*.tck);;All Files (*.*)"
    start_dir = os.path.dirname(main_window.original_trk_path) if main_window.original_trk_path else ""
    input_path, _ = QFileDialog.getOpenFileName(main_window, "Select Input Streamline File", start_dir, file_filter)

    if not input_path:
        # Use a helper or direct check for vtk_panel status update
        status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
        status_updater("File load cancelled.")
        return

    # Clean existing bundle first
    if main_window.streamlines_list:
        main_window._close_bundle()

    # Update status before potentially long load
    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
    status_updater(f"Loading file: {os.path.basename(input_path)}...")
    QApplication.processEvents() # UI update

    try:
        # --- Load File ---
        trk_file = nib.streamlines.load(input_path, lazy_load=False)
        loaded_streamlines = list(trk_file.streamlines)

        # --- Guard Clause: Check if streamlines were loaded ---
        if not loaded_streamlines:
            print("No streamlines found in file.")
            QMessageBox.information(main_window, "Load Info", "No streamlines found in the selected file.")
            status_updater(f"Loaded 0 streamlines from {os.path.basename(input_path)}")
            main_window._close_bundle() # Reset state
            return # Exit function early

        # --- Assign Core Data to MainWindow ---
        main_window.streamlines_list = loaded_streamlines
        main_window.original_trk_header = trk_file.header.copy() if hasattr(trk_file, 'header') else {}

        # Load Affine (Simplified conditional assignment)
        main_window.original_trk_affine = np.identity(4) # Default
        if hasattr(trk_file, 'tractogram') and hasattr(trk_file.tractogram, 'affine_to_rasmm'):
            main_window.original_trk_affine = trk_file.tractogram.affine_to_rasmm
        else:
            print("Warning: affine_to_rasmm not found in loaded file object. Using identity affine.")

        main_window.original_trk_path = input_path
        _, ext = os.path.splitext(input_path)
        main_window.original_file_extension = ext.lower()

        # Reset state variables
        main_window.selected_streamline_indices = set()
        main_window.undo_stack = []
        main_window.redo_stack = []
        main_window.current_color_mode = ColorMode.DEFAULT

        # --- Load Scalar Data (using helper function) ---
        main_window.scalar_data_per_point, main_window.active_scalar_name = _load_scalar_data(trk_file)

        # --- Update VTK and UI (using helper function) ---
        status_msg = f"Loaded {len(main_window.streamlines_list)} streamlines from {os.path.basename(input_path)}"
        if main_window.active_scalar_name:
            status_msg += f" | Active Scalar: {main_window.active_scalar_name}"
        _update_vtk_and_ui_after_load(main_window, status_msg)

    except Exception as e:
        print(f"Error during file loading or processing: {type(e).__name__}")
        print(f"Error details: {e}")
        print("Traceback (detailed):")
        traceback.print_exc()

        # Simplified error message display
        error_title = "Load Error"
        if isinstance(e, nib.filebasedimages.ImageFileError):
            error_msg = f"Nibabel Error: {e}\n\nIs the file a valid TRK or TCK format?"
        else:
            error_msg = f"Error loading streamline file:\n{type(e).__name__}: {e}\n\nSee console for details."

        try:
            QMessageBox.critical(main_window, error_title, error_msg)
        except Exception as qm_e:
            print(f"ERROR: Could not display QMessageBox: {qm_e}")

        # Update status and attempt to reset state cleanly
        status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
        status_updater(f"Error loading file: {os.path.basename(input_path)}")
        try:
            main_window._close_bundle() # Ensure reset after error
        except Exception as cleanup_e:
            print(f"ERROR during error cleanup (_close_bundle): {cleanup_e}")

def save_streamlines_file(main_window):
    """
    Saves the current streamlines to a trk or tck file, chosen by the user.
    Uses data from the MainWindow state.

    Args:
        main_window: The instance of the main application window.
    """

    # --- Pre-checks ---
    if main_window.streamlines_list is None or not main_window.streamlines_list:
        print("Save Error: No streamline data to save.")
        QMessageBox.warning(main_window, "Save Error", "No streamline data to save.")
        return
    if main_window.original_trk_affine is None:
        print("Save Error: Original affine info missing.")
        QMessageBox.critical(main_window, "Save Error", "Original file affine info missing (needed for saving).")
        return
    if main_window.original_trk_header is None:
        print("Warning: Original header info missing. Saving with minimal header.")
    if main_window.original_file_extension is None:
        print("Save Error: Cannot determine original file format to enforce saving.")
        QMessageBox.critical(main_window, "Save Error", "Could not determine the original file format. Cannot save.")
        return

    # --- Determine save format and suggest path ---
    initial_dir = os.path.dirname(main_window.original_trk_path) if main_window.original_trk_path else ""
    base_name = "modified_bundle"
    if main_window.original_trk_path:
        base_name = f"{os.path.splitext(os.path.basename(main_window.original_trk_path))[0]}_modified"

    required_ext = main_window.original_file_extension # e.g., '.trk' or '.tck'
    if required_ext == '.trk':
        file_filter = "TrackVis TRK Files (*.trk)"
        all_filters = f"{file_filter};;All Files (*.*)" # Keep 'All Files' but main check enforces type
    elif required_ext == '.tck':
        file_filter = "TCK Files (*.tck)"
        all_filters = f"{file_filter};;All Files (*.*)"
    else:
        # Fallback if original extension is somehow invalid (should be caught earlier)
        print(f"Save Error: Unexpected original file extension '{required_ext}'.")
        QMessageBox.critical(main_window, "Save Error", f"Unexpected original file extension: '{required_ext}'. Cannot determine save format.")
        return

    suggested_path = os.path.join(initial_dir, base_name + required_ext)

    # --- Get output path from user ---
    output_path, selected_filter = QFileDialog.getSaveFileName(
        main_window,
        f"Save Modified Streamlines As ({required_ext.upper()} only)", 
        suggested_path,
        all_filters,
        initialFilter=file_filter 
    )

    if not output_path:
        if hasattr(main_window, 'vtk_panel') and main_window.vtk_panel: main_window.vtk_panel.update_status("Save cancelled.")
        else: print("Status: Save cancelled.")
        return

    _, output_ext = os.path.splitext(output_path)
    output_ext = output_ext.lower()

    if output_ext != required_ext:
        error_msg = (f"Format Mismatch: The original file was a '{required_ext}' file. "
                     f"You must save the modified bundle as a '{required_ext}' file.\n\n"
                     f"Chosen path: '{os.path.basename(output_path)}' has the wrong extension ('{output_ext}').")
        print(f"Save Error: {error_msg}")
        QMessageBox.warning(main_window, "Save Format Error", error_msg)
        status_msg = f"Save failed: Incorrect file extension chosen (must be '{required_ext}')."
        if hasattr(main_window, 'vtk_panel') and main_window.vtk_panel: main_window.vtk_panel.update_status(status_msg)
        else: print(f"Status: {status_msg}")
        return 

    # --- Prepare for Save ---
    status_msg = f"Saving {len(main_window.streamlines_list)} streamlines to: {os.path.basename(output_path)}..."
    if hasattr(main_window, 'vtk_panel') and main_window.vtk_panel: main_window.vtk_panel.update_status(status_msg)
    else: print(f"Status: {status_msg}")
    QApplication.processEvents() # Allow UI update
    tractogram_data = main_window.streamlines_list

    try:
        affine_matrix = main_window.original_trk_affine
        if not isinstance(affine_matrix, np.ndarray):
             print(f"Warning: Affine matrix is type {type(affine_matrix)}, expected ndarray. Attempting to use anyway.")
        new_tractogram = nib.streamlines.Tractogram(
            tractogram_data,
            affine_to_rasmm=affine_matrix
        )

        # --- Save based on the output extension ---
        if output_ext == '.trk':
            header = main_window.original_trk_header.copy() if main_window.original_trk_header is not None else {}
            keys_to_clean_if_string = ['voxel_sizes', 'dimensions', 'voxel_to_rasmm']
            expected_types = {'voxel_sizes': float, 'dimensions': int, 'voxel_to_rasmm': float}
            expected_lengths = {'voxel_sizes': 3, 'dimensions': 3, 'voxel_to_rasmm': (4,4)}

            for key in keys_to_clean_if_string:
                if key in header:
                    original_value = header[key]
                    if isinstance(original_value, str):
                        print(f"  - Header field '{key}': Original is string '{original_value}'. Attempting parse...")
                        length = expected_lengths.get(key) 
                        parsed_value = parse_numeric_tuple_from_string(
                            original_value,
                            expected_types.get(key, float),
                            length 
                        )
                        # Update header only if parsing succeeded (result is not a string)
                        if not isinstance(parsed_value, str):
                            print(f"  - Header field '{key}': Parsed string to {type(parsed_value).__name__}: {parsed_value}")
                            header[key] = parsed_value
                        else:
                            print(f"  - Header field '{key}': Parsing failed or length mismatch, kept as string: '{parsed_value}'")
                            
                    ## - Debug ##
                    # elif isinstance(original_value, (np.ndarray, list, tuple)):
                    #     # If already numeric array/list/tuple, assume nibabel handles it. Log type.
                    #     print(f"  - Header field '{key}': Already numeric type ({type(original_value).__name__}). Skipping parse.")
                    # else:
                    #     # Handle other unexpected types if necessary
                    #     print(f"  - Header field '{key}': Unexpected type {type(original_value).__name__}. Skipping parse.")

            # Ensure essential fields (validate types/lengths after potential parsing)
            header['nb_streamlines'] = len(tractogram_data)
            if 'voxel_order' not in header:
                header['voxel_order'] = 'RAS'
                print("  - Added default voxel_order: RAS")

            # Validate voxel_sizes
            vs = header.get('voxel_sizes')
            if vs is None or not isinstance(vs, (tuple, list, np.ndarray)) or len(vs) != 3:
                print(f"  - Warning: Voxel sizes missing or invalid type/length ({type(vs)}). Setting default (1,1,1).")
                header['voxel_sizes'] = (1.0, 1.0, 1.0)
            elif isinstance(vs, np.ndarray):
                 header['voxel_sizes'] = tuple(vs.tolist()) 

            # Validate dimensions
            dim = header.get('dimensions')
            if dim is None or not isinstance(dim, (tuple, list, np.ndarray)) or len(dim) != 3:
                print(f"  - Warning: Dimensions missing or invalid type/length ({type(dim)}). Setting default (1,1,1).")
                header['dimensions'] = (1, 1, 1) 
            elif isinstance(dim, np.ndarray):
                 header['dimensions'] = tuple(dim.astype(int).tolist())

            # Validate voxel_to_rasmm (ensure it's a 4x4 ndarray for nibabel)
            v2r = header.get('voxel_to_rasmm')
            if v2r is None or not isinstance(v2r, np.ndarray) or v2r.shape != (4, 4):
                 print("  - Warning: voxel_to_rasmm missing or invalid shape/type. Saving may fail.")

            trk_file = nib.streamlines.TrkFile(new_tractogram, header=header)
            nib.streamlines.save(trk_file, output_path)
            status_msg = f"File saved successfully (TRK): {os.path.basename(output_path)}"

        elif output_ext == '.tck':
            tck_header = main_window.original_trk_header.copy() if main_window.original_trk_header is not None else {}
            tck_header['count'] = str(len(tractogram_data))
            tck_header.pop('nb_streamlines', None) # TCK doesn't use nb_streamlines
            tck_file = nib.streamlines.TckFile(new_tractogram, header=tck_header)
            nib.streamlines.save(tck_file, output_path)
            status_msg = f"File saved successfully (TCK): {os.path.basename(output_path)}"

        # Update status after success
        if hasattr(main_window, 'vtk_panel') and main_window.vtk_panel: main_window.vtk_panel.update_status(status_msg)
        else: print(f"Status: {status_msg}")

    except Exception as e:
        print(f"Type: {type(e).__name__}")
        print(f"Error: {e}")
        print("Traceback:")
        traceback.print_exc()
        try:
            if isinstance(e, ValueError): error_msg = f"Error saving file (ValueError):\n{e}\n\nCheck console for details. Header issues?"
            else: error_msg = f"Error saving file:\n{type(e).__name__}: {e}\n\nCheck console for details."
            QMessageBox.critical(main_window, "Save Error", error_msg)
        except Exception as qm_e: print(f"ERROR: Could not display QMessageBox: {qm_e}")
        try:
            status_msg = f"Error saving file: {os.path.basename(output_path)}"
            if hasattr(main_window, 'vtk_panel') and main_window.vtk_panel: main_window.vtk_panel.update_status(status_msg)
            else: print(f"Status: {status_msg}")
        except Exception as status_e: print(f"ERROR updating status after save error: {status_e}")
