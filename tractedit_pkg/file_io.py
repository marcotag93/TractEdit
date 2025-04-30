# -*- coding: utf-8 -*-

"""
Functions for loading and saving streamline files (trk, tck)
and loading anatomical image files (NIfTI).
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
            try:
                converted_array = value_str.astype(target_type)
                # For matrix dimensions, expected_length might be a tuple (e.g., (4,4))
                if isinstance(expected_length, tuple):
                    if converted_array.shape != expected_length:
                         print(f"Warning: Input array {value_str} has shape {converted_array.shape}, expected {expected_length}.")
                         return value_str # Return original if shape mismatch
                elif expected_length is not None and len(converted_array) != expected_length:
                     print(f"Warning: Input array {value_str} has length {len(converted_array)}, expected {expected_length}.")
                     return value_str # Return original if length mismatch
                return converted_array # Return the potentially type-converted array
            except Exception as e:
                 print(f"Warning: Could not ensure type {target_type} for ndarray {value_str}: {e}")
                 return value_str 
        else:
             return value_str

    # --- If input is a string, proceed with parsing ---
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
                 return value_str # Return original string on mismatch
            if not result:
                 raise ValueError("Splitting resulted in empty sequence")
            return result
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse '{value_str}' as a tuple of {target_type} after splitting: {e}")
            return value_str # Return original string on failure

    # Fallback: return the original string if none of the methods worked
    print(f"Warning: Could not parse '{value_str}' into expected format. Returning original string.")
    return value_str

# --- Helper Function for Scalar Loading ---
def _load_scalar_data(trk_file):
    """Attempts to load scalar data from the tractogram file object."""
    scalar_data = None
    active_scalar = None
    if hasattr(trk_file.tractogram, 'data_per_point') and trk_file.tractogram.data_per_point:
        print("Scalar data found in file (data_per_point).")
        try:
            loaded_scalars_dict = trk_file.tractogram.data_per_point.copy()
            if loaded_scalars_dict:
                processed_scalars = {}
                for key, value_list in loaded_scalars_dict.items():
                    processed_scalars[key] = [np.asarray(arr) for arr in value_list]

                scalar_data = processed_scalars
                active_scalar = list(loaded_scalars_dict.keys())[0]
                print(f"Assigned scalars. Active scalar: '{active_scalar}'")
            else:
                print("Scalar data dictionary (data_per_point) is empty.")
        except Exception as scalar_e:
            print(f"Warning: Could not process scalar data: {scalar_e}\n{traceback.format_exc()}")
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
        if main_window.vtk_panel.scene and not main_window.anatomical_image_data:
            main_window.vtk_panel.scene.reset_camera()
            main_window.vtk_panel.scene.reset_clipping_range()
        elif not main_window.vtk_panel.scene:
            print("Warning: vtk_panel.scene not available for camera reset.")

        main_window.vtk_panel.update_status(status_msg)

        if main_window.vtk_panel.render_window:
            main_window.vtk_panel.render_window.Render()
        else:
            print("Warning: render_window not available.")
    else:
        print("Error: vtk_panel not available to update actors.")
        print(f"Status: {status_msg}")

    # Ensure radio button reflects default state if no scalars loaded
    if hasattr(main_window, 'color_default_action') and not main_window.active_scalar_name:
        main_window.color_default_action.setChecked(True)

    main_window._update_action_states()
    main_window._update_bundle_info_display()

# --- Anatomical Image Loading Function ---
def load_anatomical_image(main_window):
    """
    Loads a NIfTI image file (.nii, .nii.gz).
    Returns the image data array and affine matrix.

    Args:
        main_window: The instance of the main application window.

    Returns:
        tuple: (numpy.ndarray, numpy.ndarray, str) containing image data,
               affine matrix, and file path, or (None, None, None) on failure/cancel.
    """
    if not hasattr(main_window, 'vtk_panel') or not main_window.vtk_panel.scene:
        print("Error: Scene not initialized in vtk_panel.")
        QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
        return None, None, None

    file_filter = "NIfTI Image Files (*.nii *.nii.gz);;All Files (*.*)"
    start_dir = ""
    if main_window.anatomical_image_path:
        start_dir = os.path.dirname(main_window.anatomical_image_path)
    elif main_window.original_trk_path:
        start_dir = os.path.dirname(main_window.original_trk_path)

    input_path, _ = QFileDialog.getOpenFileName(main_window, "Select Input Anatomical Image File", start_dir, file_filter)

    if not input_path:
        status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
        status_updater("Anatomical image load cancelled.")
        return None, None, None

    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
    status_updater(f"Loading image: {os.path.basename(input_path)}...")
    QApplication.processEvents() 

    try:
        # Load NIfTI file using nibabel
        img = nib.load(input_path)

        # Get data - use get_fdata() for float data potentially scaled
        image_data = img.get_fdata(dtype=np.float32) # Ensure float for VTK/FURY
        image_affine = img.affine

        # Basic validation
        if image_data.ndim < 3:
            raise ValueError(f"Loaded image has only {image_data.ndim} dimensions, expected 3 or more.")
        if image_affine.shape != (4, 4):
             raise ValueError(f"Loaded image affine has shape {image_affine.shape}, expected (4, 4).")

        status_updater(f"Successfully loaded anatomical image: {os.path.basename(input_path)}")
        return image_data, image_affine, input_path

    except FileNotFoundError:
        error_msg = f"Error: Anatomical image file not found:\n{input_path}"
        print(error_msg)
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error: File not found - {os.path.basename(input_path)}")
        return None, None, None
    except nib.filebasedimages.ImageFileError as e:
        error_msg = f"Nibabel Error loading anatomical image:\n{e}\n\nIs '{os.path.basename(input_path)}' a valid NIfTI file?"
        print(error_msg)
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error loading NIfTI: {os.path.basename(input_path)}")
        return None, None, None
    except Exception as e:
        error_msg = f"An unexpected error occurred loading the anatomical image:\n{type(e).__name__}: {e}\n\nPath: {input_path}\n\nSee console for details."
        print(error_msg)
        traceback.print_exc()
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error loading image: {os.path.basename(input_path)}")
        return None, None, None

# --- Streamline File I/O Functions ---
def load_streamlines_file(main_window):
    """
    Loads a trk or tck file using nibabel.streamlines.load.
    Updates the MainWindow state, including scalar data if present.

    Args:
        main_window: The instance of the main application window.
    """
    if not hasattr(main_window, 'vtk_panel') or not main_window.vtk_panel.scene:
        print("Error: Scene not initialized in vtk_panel.")
        QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
        return

    file_filter = "Streamline Files (*.trk *.tck);;TrackVis Files (*.trk);;TCK Files (*.tck);;All Files (*.*)"
    start_dir = ""
    if main_window.original_trk_path:
        start_dir = os.path.dirname(main_window.original_trk_path)
    elif main_window.anatomical_image_path:
        start_dir = os.path.dirname(main_window.anatomical_image_path)

    input_path, _ = QFileDialog.getOpenFileName(main_window, "Select Input Streamline File", start_dir, file_filter)

    if not input_path:
        status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
        status_updater("Streamline file load cancelled.")
        return

    # Clean existing bundle first (if any)
    if main_window.streamlines_list:
        main_window._close_bundle() 

    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
    status_updater(f"Loading streamlines: {os.path.basename(input_path)}...")
    QApplication.processEvents() 

    try:
        # --- Load File ---
        trk_file = nib.streamlines.load(input_path, lazy_load=False)
        loaded_streamlines = list(trk_file.streamlines)

        # --- Guard Clause: Check if streamlines were loaded ---
        if not loaded_streamlines:
            print("No streamlines found in file.")
            QMessageBox.information(main_window, "Load Info", "No streamlines found in the selected file.")
            status_updater(f"Loaded 0 streamlines from {os.path.basename(input_path)}")
            main_window.streamlines_list = []
            main_window.original_trk_header = None
            main_window.original_trk_affine = None
            main_window.original_trk_path = None
            main_window.original_file_extension = None
            main_window.scalar_data_per_point = None
            main_window.active_scalar_name = None
            main_window.selected_streamline_indices = set()
            main_window.undo_stack = []
            main_window.redo_stack = []
            main_window.current_color_mode = ColorMode.DEFAULT
            # Trigger UI updates
            main_window._update_bundle_info_display()
            main_window._update_action_states()
            if main_window.vtk_panel:
                main_window.vtk_panel.update_main_streamlines_actor() # Clears streamline actor
                main_window.vtk_panel.update_highlight() # Clears highlight actor
                main_window.vtk_panel.update_radius_actor(visible=False) # Hides selection sphere
                if main_window.vtk_panel.render_window:
                     main_window.vtk_panel.render_window.Render()
            return 

        # --- Assign Core Data to MainWindow ---
        main_window.streamlines_list = loaded_streamlines
        main_window.original_trk_header = trk_file.header.copy() if hasattr(trk_file, 'header') else {}

        # Load Affine
        main_window.original_trk_affine = np.identity(4) 
        if hasattr(trk_file, 'tractogram') and hasattr(trk_file.tractogram, 'affine_to_rasmm'):
            loaded_affine = trk_file.tractogram.affine_to_rasmm
            if isinstance(loaded_affine, np.ndarray) and loaded_affine.shape == (4, 4):
                main_window.original_trk_affine = loaded_affine
            else:
                print(f"Warning: loaded affine_to_rasmm is not a valid 4x4 numpy array (type: {type(loaded_affine)}). Using identity affine.")
        else:
            print("Warning: affine_to_rasmm not found in loaded file object. Using identity affine.")

        main_window.original_trk_path = input_path
        _, ext = os.path.splitext(input_path)
        main_window.original_file_extension = ext.lower()

        # Reset state variables specific to streamlines
        main_window.selected_streamline_indices = set()
        main_window.undo_stack = []
        main_window.redo_stack = []
        main_window.current_color_mode = ColorMode.DEFAULT

        # --- Load Scalar Data ---
        main_window.scalar_data_per_point, main_window.active_scalar_name = _load_scalar_data(trk_file)

        # --- Update VTK and UI ---
        status_msg = f"Loaded {len(main_window.streamlines_list)} streamlines from {os.path.basename(input_path)}"
        if main_window.active_scalar_name:
            status_msg += f" | Active Scalar: {main_window.active_scalar_name}"
        _update_vtk_and_ui_after_load(main_window, status_msg)

    except Exception as e:
        print(f"Error during streamline loading or processing: {type(e).__name__}")
        print(f"Error details: {e}")
        print("Traceback (detailed):")
        traceback.print_exc()

        error_title = "Load Error"
        if isinstance(e, nib.filebasedimages.ImageFileError):
            error_msg = f"Nibabel Error: {e}\n\nIs the file a valid TRK or TCK format?"
        elif isinstance(e, FileNotFoundError):
            error_msg = f"Error: Streamline file not found:\n{input_path}"
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
            # Reset only streamline-related vars, keep anatomical if loaded
            main_window.streamlines_list = []
            main_window.original_trk_header = None
            main_window.original_trk_affine = None
            main_window.original_trk_path = None
            main_window.original_file_extension = None
            main_window.scalar_data_per_point = None
            main_window.active_scalar_name = None
            main_window.selected_streamline_indices = set()
            main_window.undo_stack = []
            main_window.redo_stack = []
            main_window.current_color_mode = ColorMode.DEFAULT
            if main_window.vtk_panel:
                 main_window.vtk_panel.update_main_streamlines_actor()
                 main_window.vtk_panel.update_highlight()
            main_window._update_bundle_info_display()
            main_window._update_action_states()
        except Exception as cleanup_e:
            print(f"ERROR during error cleanup after streamline load failure: {cleanup_e}")


def _validate_save_prerequisites(main_window):
    """Checks if prerequisites for saving streamlines are met."""
    if main_window.streamlines_list is None or not main_window.streamlines_list:
        print("Save Error: No streamline data to save.")
        QMessageBox.warning(main_window, "Save Error", "No streamline data to save.")
        return False
    if main_window.original_trk_affine is None:
        print("Save Error: Original streamline affine info missing.")
        QMessageBox.critical(main_window, "Save Error", "Original streamline file affine info missing (needed for saving).")
        return False
    if main_window.original_trk_header is None:
        print("Warning: Original streamline header info missing. Saving with minimal header.")
        main_window.original_trk_header = {} # Ensure it's a dict
    if main_window.original_file_extension not in ['.trk', '.tck']:
        print(f"Save Error: Cannot determine original format ('{main_window.original_file_extension}').")
        QMessageBox.critical(main_window, "Save Error", f"Cannot determine original format ('{main_window.original_file_extension}').")
        return False
    return True

def _get_save_path_and_extension(main_window):
    """Gets the output path and validated extension from the user."""
    required_ext = main_window.original_file_extension
    initial_dir = os.path.dirname(main_window.original_trk_path) if main_window.original_trk_path else ""
    base_name = f"{os.path.splitext(os.path.basename(main_window.original_trk_path))[0]}_modified" \
                if main_window.original_trk_path else "modified_bundle"
    suggested_path = os.path.join(initial_dir, base_name + required_ext)

    if required_ext == '.trk':
        file_filter = "TrackVis TRK Files (*.trk)"
    elif required_ext == '.tck':
        file_filter = "TCK Files (*.tck)"
    else: 
        return None, None
    all_filters = f"{file_filter};;All Files (*.*)"

    output_path, _ = QFileDialog.getSaveFileName(
        main_window, f"Save Modified As ({required_ext.upper()} only)",
        suggested_path, all_filters, initialFilter=file_filter
    )

    if not output_path:
        return None, None

    _, output_ext = os.path.splitext(output_path)
    output_ext = output_ext.lower()

    # Enforce correct extension
    if output_ext != required_ext:
        if not output_ext:
            output_path += required_ext
            print(f"Save Info: Appended required extension '{required_ext}'.")
            output_ext = required_ext
        else:
            error_msg = (f"Format Mismatch: Must save as '{required_ext}'. "
                         f"Chosen path '{os.path.basename(output_path)}' has wrong extension ('{output_ext}').")
            QMessageBox.warning(main_window, "Save Format Error", error_msg)
            return None, None 

    return output_path, output_ext

def _prepare_tractogram_and_affine(main_window):
    """Prepares the Tractogram object and validates the affine matrix."""
    tractogram_data = main_window.streamlines_list
    affine_matrix = main_window.original_trk_affine

    if not isinstance(affine_matrix, np.ndarray) or affine_matrix.shape != (4, 4):
        print(f"Warning: Affine matrix invalid. Using identity.")
        affine_matrix = np.identity(4)

    # Handle potential scalar data
    data_per_point_to_save = {}
    if main_window.scalar_data_per_point and main_window.active_scalar_name:
        active_scalar_list = main_window.scalar_data_per_point.get(main_window.active_scalar_name, [])
        if len(active_scalar_list) == len(tractogram_data):
            data_per_point_to_save = main_window.scalar_data_per_point
        else:
            print("Warning: Scalar data length mismatch. Saving without scalar data.")

    new_tractogram = nib.streamlines.Tractogram(
        tractogram_data,
        data_per_point=data_per_point_to_save if data_per_point_to_save else None,
        affine_to_rasmm=affine_matrix
    )
    return new_tractogram

def _prepare_trk_header(base_header, nb_streamlines):
    """Prepares and validates the header dictionary for TRK saving."""
    header = base_header.copy()
    print("Preparing TRK header for saving...")

    # Fields to clean/validate if they exist as strings
    keys_to_clean_if_string = ['voxel_sizes', 'dimensions', 'voxel_to_rasmm']
    expected_types = {'voxel_sizes': float, 'dimensions': int, 'voxel_to_rasmm': float}
    expected_lengths = {'voxel_sizes': 3, 'dimensions': 3, 'voxel_to_rasmm': (4, 4)}

    for key in keys_to_clean_if_string:
        if key not in header: 
            continue
        original_value = header[key]
        is_string = isinstance(original_value, str)
        is_numeric_like = isinstance(original_value, (np.ndarray, list, tuple))

        if not is_string and not is_numeric_like: 
            continue 
        parsed_value = original_value # Default to original
        expected_type = expected_types.get(key, float)
        length_or_shape = expected_lengths.get(key)

        if is_string:
            parsed_value = parse_numeric_tuple_from_string(
                original_value, expected_type, length_or_shape
            )
            if isinstance(parsed_value, str): # Parsing failed
                 print(f"    - Warning: Could not parse string '{original_value}' for key '{key}'. Keeping original.")
                 continue 

        # --- Validation for parsed string or existing numeric types ---
        try:
            valid = True
            current_val = parsed_value
            # Convert list to tuple, ndarray to expected type if needed
            if isinstance(current_val, list):
                current_val = tuple(expected_type(x) for x in current_val)
            elif isinstance(current_val, np.ndarray):
                current_val = current_val.astype(expected_type)

            # Validate shape/length
            if isinstance(length_or_shape, tuple) and isinstance(current_val, np.ndarray):
                if current_val.shape != length_or_shape: valid = False
            elif isinstance(length_or_shape, int):
                 # Ensure it's a tuple (for dimensions, voxel_sizes) not ndarray after conversion
                 if not isinstance(current_val, tuple) or len(current_val) != length_or_shape:
                     # If it was an ndarray, try converting to tuple AFTER type cast
                     if isinstance(parsed_value, np.ndarray) and len(parsed_value) == length_or_shape:
                         current_val = tuple(current_val)
                     else:
                         valid = False

            if valid:
                if is_string: print(f"    - Parsed string for '{key}' to {type(current_val).__name__}: {current_val}.")
                header[key] = current_val # Update header
            else:
                print(f"    - Warning: Validation failed for '{key}' (shape/length/type). Original value: {original_value}")
                header[key] = original_value

        except Exception as conv_e:
            print(f"    - Warning: Error converting/validating field '{key}': {conv_e}. Keeping original.")
            header[key] = original_value 

    # --- Ensure essential fields are present and valid ---
    header['nb_streamlines'] = nb_streamlines
    if 'voxel_order' not in header or not isinstance(header['voxel_order'], str) or len(header['voxel_order']) != 3:
        header['voxel_order'] = 'RAS'
    header['voxel_order'] = header['voxel_order'].upper()

    # Use .get() with default for validation checks
    vs = header.get('voxel_sizes')
    if not isinstance(vs, tuple) or len(vs) != 3:
        header['voxel_sizes'] = (1.0, 1.0, 1.0)

    dim = header.get('dimensions')
    if not isinstance(dim, tuple) or len(dim) != 3:
        header['dimensions'] = (1, 1, 1)

    v2r = header.get('voxel_to_rasmm')
    if not isinstance(v2r, np.ndarray) or v2r.shape != (4, 4):
        header['voxel_to_rasmm'] = np.identity(4, dtype=np.float32)
    else:
        header['voxel_to_rasmm'] = v2r.astype(np.float32) # Ensure float

    return header

def _prepare_tck_header(base_header, nb_streamlines):
    """Prepares the header dictionary for TCK saving."""
    header = base_header.copy() if base_header is not None else {}
    header['count'] = str(nb_streamlines) # TCK expects count as string
    header.pop('nb_streamlines', None) # Remove TRK specific field
    # Ensure other common fields are strings if present
    for key in ['voxel_order', 'dimensions', 'voxel_sizes']:
        if key in header: header[key] = str(header[key])
    return header

def _save_tractogram_file(tractogram, header, output_path, file_ext):
    """Saves the tractogram using nibabel based on the extension."""
    if file_ext == '.trk':
        trk_file = nib.streamlines.TrkFile(tractogram, header=header)
        nib.streamlines.save(trk_file, output_path)
        print("File saved successfully (TRK)")
        return f"File saved successfully (TRK): {os.path.basename(output_path)}"
    elif file_ext == '.tck':
        tck_file = nib.streamlines.TckFile(tractogram, header=header)
        nib.streamlines.save(tck_file, output_path)
        print("File saved successfully (TCK)")
        return f"File saved successfully (TCK): {os.path.basename(output_path)}"
    else:
        raise ValueError(f"Unsupported save extension: {file_ext}")

# --- Main Save Function ---
def save_streamlines_file(main_window):
    """
    Saves the current streamlines to a trk or tck file (refactored).
    """
    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))

    # --- 1. Pre-checks ---
    if not _validate_save_prerequisites(main_window):
        return

    # --- 2. Get Output Path ---
    output_path, output_ext = _get_save_path_and_extension(main_window)
    if not output_path:
        status_updater("Save cancelled.")
        return 

    # --- 3. Prepare Data ---
    status_updater(f"Saving {len(main_window.streamlines_list)} streamlines to: {os.path.basename(output_path)}...")
    QApplication.processEvents() # Allow UI update

    try:
        tractogram = _prepare_tractogram_and_affine(main_window)

        # --- 4. Prepare Header ---
        header = {}
        if output_ext == '.trk':
            header = _prepare_trk_header(main_window.original_trk_header, len(tractogram.streamlines))
        elif output_ext == '.tck':
            header = _prepare_tck_header(main_window.original_trk_header, len(tractogram.streamlines))

        # --- 5. Save File ---
        success_msg = _save_tractogram_file(tractogram, header, output_path, output_ext)
        status_updater(success_msg)

    except Exception as e:
        print(f"Error during file saving:\nType: {type(e).__name__}\nError: {e}")
        traceback.print_exc()
        error_msg = f"Error saving file:\n{type(e).__name__}: {e}\n\nCheck console for details."
        QMessageBox.critical(main_window, "Save Error", error_msg)
        status_updater(f"Error saving file: {os.path.basename(output_path)}")