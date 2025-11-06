# -*- coding: utf-8 -*-

"""
Functions for loading and saving streamline files (trk, tck, trx)
and loading anatomical image files (NIfTI).
"""

import os
import traceback
import ast
import numpy as np
import nibabel as nib
import trx.trx_file_memmap as tbx
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication
from .utils import ColorMode
from typing import Optional, List, Dict, Any, Tuple, Type, Union

# --- Helper Function ---
# This is a mess, will be updated to be cleaner in the future 
def parse_numeric_tuple_from_string(value_str: Any, target_type: Type = float, expected_length: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
    if not isinstance(value_str, str):
        # If it's already a list or tuple, try to convert types and check length
        if isinstance(value_str, (list, tuple)):
            try:
                converted_tuple = tuple(target_type(x) for x in value_str)
                if expected_length is not None and len(converted_tuple) != expected_length:
                    # print(f"Warning: Input sequence {value_str} has length {len(converted_tuple)}, expected {expected_length}.") # debug
                    return value_str # Return original if length mismatch after conversion
                return converted_tuple
            except (ValueError, TypeError): # e.g. target_type(x) fails
                return value_str # Return original on conversion error
        elif isinstance(value_str, np.ndarray):
            try:
                converted_array = value_str.astype(target_type)
                if isinstance(expected_length, tuple): # For matrix shapes
                    if converted_array.shape != expected_length:
                        return value_str
                elif expected_length is not None: # For 1D array/tuple lengths
                        if converted_array.ndim == 1 and len(converted_array) != expected_length:
                            return value_str
                        elif converted_array.ndim != 1 : # Or if not 1D when expected_length is int
                            return value_str
                return converted_array
            except (ValueError, TypeError):
                return value_str # Return original on type conversion error for ndarray
        else: # Not a string, list, tuple, or ndarray
            return value_str

    # --- If input is a string, proceed with parsing ---
    try:
        parsed_val = ast.literal_eval(value_str)
        if isinstance(parsed_val, (list, tuple)):
            result = tuple(target_type(x) for x in parsed_val)
            if expected_length is not None and len(result) != expected_length:
                # print(f"Warning: Parsed tuple {result} from '{value_str}' has length {len(result)}, expected {expected_length}.")
                return value_str # Return original string if length mismatch
            return result
        elif isinstance(parsed_val, (int, float)): 
            result_scalar = target_type(parsed_val)
            # If expected_length is 1, wrap in a tuple
            if expected_length == 1:
                return (result_scalar,)
            # If expected_length is None (e.g. for a single scalar not in a tuple), return scalar
            elif expected_length is None: 
                return result_scalar
            else: # Mismatch if expected_length is other than 1 or None for a single number
                return value_str 
    except (ValueError, SyntaxError, TypeError): # ast.literal_eval failed or target_type(x) failed
        # Fallback: try splitting the string
        cleaned_str = value_str.strip('()[] ') # Added space to strip
        parts = [p.strip() for p in cleaned_str.split(',') if p.strip()] if ',' in cleaned_str else \
                [p.strip() for p in cleaned_str.split() if p.strip()]

        if not parts: # If splitting results in no parts
            return value_str

        try:
            result = tuple(target_type(p) for p in parts)
            if expected_length is not None and len(result) != expected_length:
                # print(f"Warning: Parsed tuple {result} from splitting '{value_str}' has length {len(result)}, expected {expected_length}.")
                return value_str # Return original string if length mismatch
            return result
        except (ValueError, TypeError): # target_type(p) failed
            # print(f"Warning: Could not parse '{value_str}' as a tuple of {target_type} after splitting.")
            return value_str
    # Fallback if ast.literal_eval results in an unexpected type or other issues
    return value_str

# --- Helper Function for VTK/UI Update ---
def _update_vtk_and_ui_after_load(main_window: Any, status_msg: str) -> None:
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
    if hasattr(main_window, 'color_orientation_action'):
        main_window.color_orientation_action.setChecked(True)

    main_window._update_action_states()
    main_window._update_bundle_info_display()

# --- Anatomical Image Loading Function ---
def load_anatomical_image(main_window: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
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
def load_streamlines_file(main_window: Any) -> None:
    """
    Loads a trk, tck, or trx file.
    Updates the MainWindow state.

    Args:
        main_window: The instance of the main application window.
    """
    if not hasattr(main_window, 'vtk_panel') or not main_window.vtk_panel.scene:
        print("Error: Scene not initialized in vtk_panel.")
        QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
        return

    base_filter = "Streamline Files (*.trk *.tck *.trx)"
    all_filters = f"{base_filter};;TrackVis Files (*.trk);;TCK Files (*.tck);;TRX Files (*.trx);;All Files (*.*)"
    
    start_dir = ""
    if main_window.original_trk_path:
        start_dir = os.path.dirname(main_window.original_trk_path)
    elif main_window.anatomical_image_path:
        start_dir = os.path.dirname(main_window.anatomical_image_path)

    input_path, _ = QFileDialog.getOpenFileName(main_window, "Select Input Streamline File", start_dir, all_filters)

    if not input_path:
        status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
        status_updater("Streamline file load cancelled.")
        return

    # Clean existing bundle first (if any)
    if main_window.tractogram_data:
        main_window._close_bundle()

    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: print(f"Status: {msg}"))
    status_updater(f"Loading streamlines: {os.path.basename(input_path)}...")
    QApplication.processEvents()

    try:
        _, ext = os.path.splitext(input_path)
        ext = ext.lower()

        loaded_streamlines_obj: Optional['nib.streamlines.ArraySequence'] = None
        loaded_header: Dict[str, Any] = {}
        loaded_affine: np.ndarray = np.identity(4)
        scalar_data: Optional[Dict[str, 'nib.streamlines.ArraySequence']] = None
        active_scalar: Optional[str] = None
        tractogram_obj: Any = None # To hold the object with .data_per_point
        num_streamlines = 0
        
        # Clear any old trx file reference
        if hasattr(main_window, 'trx_file_reference'):
             try:
                 main_window.trx_file_reference.close()
             except Exception:
                 pass 
        main_window.trx_file_reference = None

        if ext in ['.trk', '.tck']:
            # --- Load File using Nibabel ---
            # lazy_load=True returns a generator for .trk/.tck.
            trk_file = nib.streamlines.load(input_path, lazy_load=True)
            
            # This is the memory-intensive step
            streamlines_list = list(trk_file.streamlines)
            num_streamlines = len(streamlines_list)
            print(f"Info: Loaded {num_streamlines} streamlines from {ext} file into memory.")

            # Store as an ArraySequence for a consistent API
            loaded_streamlines_obj = nib.streamlines.ArraySequence(streamlines_list)
            
            loaded_header = trk_file.header.copy() if hasattr(trk_file, 'header') else {}
            tractogram_obj = trk_file.tractogram # This has the data_per_point

            # Load Affine from Nibabel object
            if hasattr(tractogram_obj, 'affine_to_rasmm'):
                nib_affine = tractogram_obj.affine_to_rasmm
                if isinstance(nib_affine, np.ndarray) and nib_affine.shape == (4, 4):
                    loaded_affine = nib_affine
                else:
                    print(f"Warning: loaded affine_to_rasmm is not a valid 4x4 numpy array (type: {type(nib_affine)}). Using identity affine.")
            else:
                print("Warning: affine_to_rasmm not found in loaded file object. Using identity affine.")
            
            # --- Load Scalar Data ---
            if hasattr(tractogram_obj, 'data_per_point') and tractogram_obj.data_per_point:
                print("Scalar data found in file (data_per_point).")
                # For nibabel, data_per_point is a PerPointDict (dict of lists)
                # We MUST convert it to a dict of ArraySequences for our new model.
                scalar_data = {}
                try:
                    for key, value_list in tractogram_obj.data_per_point.items():
                        scalar_data[key] = nib.streamlines.ArraySequence(value_list)
                    if scalar_data:
                        active_scalar = list(scalar_data.keys())[0]
                        print(f"Assigned scalars. Active scalar: '{active_scalar}'")
                except Exception as scalar_e:
                    print(f"Warning: Could not process scalar data: {scalar_e}\n{traceback.format_exc()}")
                    scalar_data = None
                    active_scalar = None
            else:
                 print("No scalar data found in file (data_per_point).")

        elif ext == '.trx':
            
            # --- Load File using trx-python ---
            trx_obj = tbx.load(input_path)
            
            # trx_obj.streamlines IS a lazy ArraySequence. 
            loaded_streamlines_obj = trx_obj.streamlines # Get streamlines
            num_streamlines = len(loaded_streamlines_obj) # This is fast
            print(f"Info: Loaded {num_streamlines} streamlines from {ext} file (lazy-loaded).")
            
            loaded_header = trx_obj.header.copy() # Get header
            tractogram_obj = trx_obj # This has the data_per_point
            
            # Load Affine from TRX object
            if hasattr(trx_obj, 'affine_to_rasmm'):
                trx_affine = trx_obj.affine_to_rasmm
                if isinstance(trx_affine, np.ndarray) and trx_affine.shape == (4, 4):
                    loaded_affine = trx_affine
                else:
                    print(f"Warning: TRX affine_to_rasmm is not a valid 4x4 numpy array (type: {type(trx_affine)}). Using identity affine.")
            else:
                print("Warning: affine_to_rasmm not found in loaded TRX file. Using identity affine.")

            # --- Load Scalar Data (REFACTORED) ---
            if hasattr(trx_obj, 'data_per_point') and trx_obj.data_per_point:
                print("Scalar data found in file (data_per_point).")
                # For trx-python, data_per_point is *already* a PerPointArraySequenceDict (dict of ArraySequences).
                scalar_data = trx_obj.data_per_point
                if scalar_data:
                    active_scalar = list(scalar_data.keys())[0]
                    print(f"Assigned scalars. Active scalar: '{active_scalar}'")
            else:
                print("No scalar data found in file (data_per_point).")
            
            main_window.trx_file_reference = trx_obj

        else:
            raise ValueError(f"Unsupported file extension: '{ext}'. Only .trk, .tck, and .trx are supported.")


        # --- Guard Clause: Check if streamlines were loaded ---
        if not loaded_streamlines_obj or num_streamlines == 0:
            print("No streamlines found in file.")
            QMessageBox.information(main_window, "Load Info", "No streamlines found in the selected file.")
            status_updater(f"Loaded 0 streamlines from {os.path.basename(input_path)}")
            main_window.tractogram_data = None
            main_window.visible_indices = set()
            main_window.original_trk_header = None
            main_window.original_trk_affine = None
            main_window.original_trk_path = None
            main_window.original_file_extension = None
            main_window.scalar_data_per_point = None
            main_window.active_scalar_name = None
            main_window.selected_streamline_indices = set()
            main_window.undo_stack = []
            main_window.redo_stack = []
            main_window.current_color_mode = ColorMode.ORIENTATION
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
        main_window.tractogram_data = loaded_streamlines_obj
        main_window.visible_indices = set(range(num_streamlines))
        
        main_window.original_trk_header = loaded_header
        main_window.original_trk_affine = loaded_affine
        main_window.original_trk_path = input_path
        main_window.original_file_extension = ext 

        # Reset state variables specific to streamlines
        main_window.selected_streamline_indices = set()
        main_window.undo_stack = []
        main_window.redo_stack = []
        main_window.current_color_mode = ColorMode.ORIENTATION

        # --- Assign Scalar Data ---
        main_window.scalar_data_per_point = scalar_data
        main_window.active_scalar_name = active_scalar

        # --- Update VTK and UI ---
        status_msg = f"Loaded {len(main_window.tractogram_data)} streamlines from {os.path.basename(input_path)}"
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
            main_window.tractogram_data = None
            main_window.visible_indices = set()
            main_window.original_trk_header = None
            main_window.original_trk_affine = None
            main_window.original_trk_path = None
            main_window.original_file_extension = None
            main_window.scalar_data_per_point = None
            main_window.active_scalar_name = None
            main_window.selected_streamline_indices = set()
            main_window.undo_stack = []
            main_window.redo_stack = []
            main_window.current_color_mode = ColorMode.ORIENTATION 
            if main_window.vtk_panel:
                main_window.vtk_panel.update_main_streamlines_actor()
                main_window.vtk_panel.update_highlight()
            main_window._update_bundle_info_display()
            main_window._update_action_states()
        except Exception as cleanup_e:
            print(f"ERROR during error cleanup after streamline load failure: {cleanup_e}")


def _validate_save_prerequisites(main_window: Any) -> bool:
    """Checks if prerequisites for saving streamlines are met."""
    if main_window.tractogram_data is None:
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
    
    if main_window.original_file_extension not in ['.trk', '.tck', '.trx']:
        print(f"Save Error: Cannot determine original format ('{main_window.original_file_extension}').")
        QMessageBox.critical(main_window, "Save Error", f"Cannot determine original format ('{main_window.original_file_extension}').")
        return False
    return True

def _get_save_path_and_extension(main_window: Any) -> Tuple[Optional[str], Optional[str]]:
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
    elif required_ext == '.trx':
        file_filter = "TRX Files (*.trx)"
    else:
        # Fallback or error if extension is invalid or TRX not supported
        QMessageBox.critical(main_window, "Save Error", f"Cannot save: Unknown original format '{required_ext}'.")
        return None, None 
    
    all_filters = f"{file_filter};;All Files (*.*)"

    output_path, _ = QFileDialog.getSaveFileName(
        main_window, f"Save Modified As ({required_ext.upper()} only)",
        suggested_path, all_filters, initialFilter=file_filter
    )

    if not output_path:
        return None, None

    _, output_ext_from_dialog = os.path.splitext(output_path)
    output_ext_from_dialog = output_ext_from_dialog.lower()

    # Enforce correct extension, inform user if corrected
    if output_ext_from_dialog != required_ext:
        old_output_path = output_path
        # Correct the path to have the required extension
        output_path = os.path.splitext(output_path)[0] + required_ext
        if output_ext_from_dialog == "": # No extension was provided
            print(f"Save Info: Appended required extension '{required_ext}'. New path: {output_path}")
        else: # A different extension was provided
            QMessageBox.warning(main_window, "Save Format Corrected",
                                f"File extension was corrected from '{output_ext_from_dialog}' to the required '{required_ext}'.\n"
                                f"Saving as: {os.path.basename(output_path)}")
            print(f"Save Info: Corrected extension from '{output_ext_from_dialog}' to '{required_ext}'. Path changed from '{old_output_path}' to '{output_path}'.")
    return output_path, required_ext

def _prepare_tractogram_and_affine(main_window: Any) -> nib.streamlines.Tractogram:
    """Prepares the Tractogram object and validates the affine matrix."""
    tractogram = main_window.tractogram_data
    # Get the sorted list of indices to save
    indices_to_save = sorted(list(main_window.visible_indices))
    
    # Create a generator for the streamlines to save
    streamlines_to_save_gen = (tractogram[i] for i in indices_to_save)

    affine_matrix = main_window.original_trk_affine

    if not isinstance(affine_matrix, np.ndarray) or affine_matrix.shape != (4, 4):
        print(f"Warning: Affine matrix invalid. Using identity.")
        affine_matrix = np.identity(4)

    # Handle potential scalar data
    data_per_point_to_save = {}
    if main_window.scalar_data_per_point:
        try:
            for key, scalar_sequence in main_window.scalar_data_per_point.items():
                # Use the same generator logic to get scalars for visible indices
                scalars_for_key_gen = (scalar_sequence[i] for i in indices_to_save)
                # We must save it as a list of arrays
                data_per_point_to_save[key] = list(scalars_for_key_gen)
        except Exception as e:
            print(f"Warning: Could not filter scalar data for saving. Saving without scalars. Error: {e}")
            data_per_point_to_save = {}
    
    # Use Nibabel's Tractogram object as a generic container
    new_tractogram = nib.streamlines.Tractogram(
        list(streamlines_to_save_gen),
        data_per_point=data_per_point_to_save if data_per_point_to_save else None,
        affine_to_rasmm=affine_matrix
    )
    return new_tractogram

def _prepare_trk_header(base_header: Dict[str, Any], nb_streamlines: int, anatomical_img_affine: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Prepares and validates the header dictionary for TRK saving.
    If voxel_order is missing in base_header, attempts to derive it from
    anatomical_img_affine, otherwise defaults to 'RAS'.
    """
    header = base_header.copy()
    print("Preparing TRK header for saving...")

    # --- Voxel Order Logic ---
    raw_voxel_order_from_trk = header.get('voxel_order')
    processed_voxel_order_from_trk = None

    if isinstance(raw_voxel_order_from_trk, bytes):
        try:
            processed_voxel_order_from_trk = raw_voxel_order_from_trk.decode('utf-8', errors='strict')
            print(f"      - Info: Decoded 'voxel_order' (bytes: {raw_voxel_order_from_trk}) to string: '{processed_voxel_order_from_trk}'")
        except UnicodeDecodeError:
            print(f"      - Warning: 'voxel_order' field in TRK header (bytes: {raw_voxel_order_from_trk}) could not be decoded. Treating as invalid.")
    elif isinstance(raw_voxel_order_from_trk, str):
        processed_voxel_order_from_trk = raw_voxel_order_from_trk

    is_valid_trk_voxel_order = isinstance(processed_voxel_order_from_trk, str) and len(processed_voxel_order_from_trk) == 3

    if is_valid_trk_voxel_order:
        header['voxel_order'] = processed_voxel_order_from_trk.upper()
        print(f"      - Info: Using existing 'voxel_order' from TRK header: {header['voxel_order']}.")
    else:
        if raw_voxel_order_from_trk is not None:
            print(f"      - Warning: 'voxel_order' from TRK header ('{raw_voxel_order_from_trk}') is invalid or in an unexpected format.")
        else:
            print(f"      - Info: 'voxel_order' missing in TRK header.")

        derived_from_anat = False
        if anatomical_img_affine is not None and \
           isinstance(anatomical_img_affine, np.ndarray) and \
           anatomical_img_affine.shape == (4,4):
            try:
                axcodes = nib.aff2axcodes(anatomical_img_affine)
                derived_vo_str = "".join(axcodes).upper()
                if len(derived_vo_str) == 3:
                    header['voxel_order'] = derived_vo_str
                    derived_from_anat = True
                    print(f"      - Info: Derived 'voxel_order' from loaded anatomical image: {header['voxel_order']}.")
                else:
                    print(f"      - Warning: Could not derive a valid 3-character 'voxel_order' from anatomical image affine (got: '{derived_vo_str}').")
            except Exception as e:
                print(f"      - Warning: Error deriving 'voxel_order' from anatomical image affine: {e}.")

        if not derived_from_anat:
            header['voxel_order'] = 'RAS'
            # Contextual print for defaulting voxel_order
            if raw_voxel_order_from_trk is None and anatomical_img_affine is None:
                print(f"      - Info: 'voxel_order' missing, no anatomical image. Defaulting to 'RAS'.")
            elif not is_valid_trk_voxel_order and anatomical_img_affine is None:
                print(f"      - Info: Original 'voxel_order' invalid/missing, no anatomical image. Defaulting to 'RAS'.")
            else: # Covers cases where derivation from anat failed or anat_img_affine was invalid
                print(f"      - Info: Could not use original or derive 'voxel_order' from anatomical image. Defaulting to 'RAS'.")

    # --- Process other specific TRK header fields ---
    keys_to_process = {
        'voxel_sizes': {'type': float, 'length': 3, 'default': (1.0, 1.0, 1.0)},
        'dimensions': {'type': int, 'length': 3, 'default': (1, 1, 1)}, # Small valid default
        'voxel_to_rasmm': {'type': float, 'shape': (4,4), 'default': np.identity(4, dtype=np.float32)}
    }

    for key, K_props in keys_to_process.items():
        original_value = header.get(key)
        processed_value = original_value
        expected_item_type = K_props['type']
        is_matrix = 'shape' in K_props

        # 1. Decode if bytes
        if isinstance(processed_value, bytes):
            try:
                processed_value = processed_value.decode('utf-8', errors='strict')
            except UnicodeDecodeError:
                print(f"      - Warning: Could not decode bytes for '{key}'. Original value: {original_value}")
                header[key] = K_props['default']
                print(f"      - Info: Set '{key}' to default: {header[key]}")
                continue 

        # 2. Parse if string, or use if already suitable type
        if isinstance(processed_value, str):
            parsed_val = parse_numeric_tuple_from_string(
                processed_value,
                expected_item_type,
                K_props.get('length') or K_props.get('shape') 
            )
            # Check if parse_numeric_tuple_from_string returned the original string (failure)
            if not (isinstance(parsed_val, str) and parsed_val == processed_value):
                processed_value = parsed_val # Successfully parsed
            else:
                # Parsing failed
                print(f"      - Info: Could not parse string '{processed_value}' for '{key}'.")
        
        # 3. Validate and set
        valid_structure = False
        final_value = None

        try:
            if is_matrix: # voxel_to_rasmm
                if isinstance(processed_value, np.ndarray) and processed_value.shape == K_props['shape']:
                    final_value = processed_value.astype(expected_item_type)
                    valid_structure = True
            else: # voxel_sizes, dimensions (tuples)
                if isinstance(processed_value, tuple) and len(processed_value) == K_props['length']:
                    final_value = tuple(expected_item_type(x) for x in processed_value)
                    valid_structure = True
                elif isinstance(processed_value, np.ndarray) and processed_value.ndim == 1 and len(processed_value) == K_props['length']:
                    final_value = tuple(processed_value.astype(expected_item_type))
                    valid_structure = True
        except (ValueError, TypeError) as e: # Catch errors from type conversion 
            print(f"      - Warning: Type conversion error for '{key}' (value: '{processed_value}'): {e}")
            valid_structure = False 

        if valid_structure:
            header[key] = final_value
        else:
            print(f"      - Warning: '{key}' ('{original_value}') was invalid, missing, or failed processing. Defaulted to {K_props['default']}.")
            header[key] = K_props['default']

    header['nb_streamlines'] = nb_streamlines
    header['voxel_order'] = header['voxel_order'].upper()

    return header

def _prepare_tck_header(base_header: Optional[Dict[str, Any]], nb_streamlines: int) -> Dict[str, Any]:
    """Prepares the header dictionary for TCK saving."""
    header = base_header.copy() if base_header is not None else {}
    header['count'] = str(nb_streamlines) 
    header.pop('nb_streamlines', None) 
    for key in ['voxel_order', 'dimensions', 'voxel_sizes']:
        if key in header:
            if isinstance(header[key], bytes):
                try:
                    header[key] = header[key].decode('utf-8', errors='replace')
                except Exception:
                    header[key] = str(header[key]) # Fallback
            header[key] = str(header[key])
    return header

def _prepare_trx_header(base_header: Optional[Dict[str, Any]], nb_streamlines: int) -> Dict[str, Any]:
    """Prepares the header dictionary for TRX saving."""
    header = base_header.copy() if base_header is not None else {}
    header['nb_streamlines'] = nb_streamlines
    
    # Clean up fields
    header.pop('count', None) # TCK specific
    
    return header

def _save_tractogram_file(tractogram: nib.streamlines.Tractogram, header: Dict[str, Any], output_path: str, file_ext: str) -> str:
    """
    Saves the tractogram using nibabel or trx-python based on the extension.
    """
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
    
    elif file_ext == '.trx':        
        # 1. 'tractogram' is our nib.streamlines.Tractogram object.
        #    'header' is our prepared header.
        trx_obj_to_save = tbx.TrxFile.from_lazy_tractogram(
            tractogram, header
        )
        
        # 2. Save the newly created object using tbx.save()
        tbx.save(trx_obj_to_save, output_path)
        print("File saved successfully (TRX)")
        return f"File saved successfully (TRX): {os.path.basename(output_path)}"
    
    else:
        raise ValueError(f"Unsupported save extension: {file_ext}")

# --- Main Save Function ---
def save_streamlines_file(main_window: Any) -> None:
    """
    Saves the current streamlines to a trk, tck, or trx file.
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
    status_updater(f"Saving {len(main_window.visible_indices)} streamlines to: {os.path.basename(output_path)}...")
    QApplication.processEvents() # UI update

    try:
        tractogram = _prepare_tractogram_and_affine(main_window)

        header_to_save: Dict[str, Any] = {}
        if output_ext == '.trk':
            header_to_save = _prepare_trk_header(
                main_window.original_trk_header,
                len(tractogram.streamlines),
                anatomical_img_affine=main_window.anatomical_image_affine
            )
        elif output_ext == '.tck':
            header_to_save = _prepare_tck_header(
                main_window.original_trk_header, len(tractogram.streamlines))
        
        elif output_ext == '.trx':
            header_to_save = _prepare_trx_header(
                main_window.original_trk_header, len(tractogram.streamlines))
            
        # --- 5. Save File ---
        success_msg = _save_tractogram_file(tractogram, header_to_save, output_path, output_ext)
        status_updater(success_msg)

    except Exception as e:
        print(f"Error during file saving:\nType: {type(e).__name__}\nError: {e}")
        traceback.print_exc()
        error_msg = f"Error saving file:\n{type(e).__name__}: {e}\n\nCheck console for details."
        QMessageBox.critical(main_window, "Save Error", error_msg)
        status_updater(f"Error saving file: {os.path.basename(output_path)}")