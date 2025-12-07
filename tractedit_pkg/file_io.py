# -*- coding: utf-8 -*-

"""
Functions for loading and saving streamline files (trk, tck, trx)
and loading anatomical image files (NIfTI).
"""

import os
import ast
import logging
import numpy as np
import nibabel as nib
import trx.trx_file_memmap as tbx
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication, QProgressDialog, QWidget
from .utils import ColorMode
from typing import Optional, List, Dict, Any, Tuple, Type, Union

logger = logging.getLogger(__name__)

# --- Helper Function ---
def parse_numeric_tuple_from_string(
    input_value: Union[str, List, Tuple, np.ndarray, Any], 
    target_type: Type = float, 
    expected_length: Optional[Union[int, Tuple[int, ...]]] = None
) -> Any:
    """
    Parses an input (string, list, tuple, or ndarray) into a tuple/array of a specific numeric type.
    
    If parsing or validation fails, the original input_value is returned.

    Args:
        input_value: The input data to parse. Can be a string representation (e.g. "(1, 2, 3)", "1 2 3"),
                     a sequence, or a numpy array.
        target_type: The desired type for the elements (default: float).
        expected_length: The expected length (int) or shape (tuple) of the result.
    
    Returns:
        The parsed tuple/array cast to target_type, or input_value if parsing/validation fails.
    """
    
    # 1. Handle non-string inputs (Already lists, tuples, or arrays)
    if not isinstance(input_value, str):
        return _process_existing_sequence(input_value, target_type, expected_length)

    # 2. Strategy A: Try safe python syntax parsing (e.g., "(1, 2)", "[1, 2]")
    try:
        parsed_val = ast.literal_eval(input_value)
        return _process_parsed_value(parsed_val, input_value, target_type, expected_length)
    except (ValueError, SyntaxError, TypeError):
        pass

    # 3. Strategy B: Fallback to string splitting (e.g., "1 2 3", "1, 2, 3")
    # Clean brackets and split by comma or whitespace
    cleaned_str = input_value.translate(str.maketrans('', '', '[]()'))
    parts = cleaned_str.replace(',', ' ').split()
    
    if not parts:
        return input_value

    try:
        # Convert parts to target type
        converted = tuple(target_type(p) for p in parts)
        return _validate_length(converted, expected_length, input_value)
    except (ValueError, TypeError):
        return input_value


# --- Helper Functions ---
def _process_existing_sequence(data: Any, dtype: Type, length_req: Any) -> Any:
    """Handles inputs that are already lists, tuples, or numpy arrays."""
    if isinstance(data, (list, tuple)):
        try:
            converted = tuple(dtype(x) for x in data)
            return _validate_length(converted, length_req, data)
        except (ValueError, TypeError):
            return data

    if isinstance(data, np.ndarray):
        try:
            # Check shape/length before casting
            if not _check_numpy_shape(data, length_req):
                return data
            return data.astype(dtype)
        except (ValueError, TypeError):
            return data
            
    return data

def _process_parsed_value(parsed: Any, original: Any, dtype: Type, length_req: Any) -> Any:
    """Handles the result of ast.literal_eval."""
    # Handle Sequences
    if isinstance(parsed, (list, tuple)):
        try:
            converted = tuple(dtype(x) for x in parsed)
            return _validate_length(converted, length_req, original)
        except (ValueError, TypeError):
            return original
            
    # Handle Scalars
    if isinstance(parsed, (int, float)):
        val = dtype(parsed)
        if length_req == 1:
            return (val,)
        elif length_req is None:
            return val
            
    return original

def _validate_length(data: tuple, expected: Optional[Union[int, Tuple[int, ...]]], original: Any) -> Any:
    """Validates that the data tuple matches the expected length."""
    if expected is None:
        return data
    
    # If expected is a tuple (usually for numpy shapes), we only check dimension 0 here for tuples
    if isinstance(expected, tuple):
        # This function primarily handles 1D tuples. Complex matrix strings are rare in this context.
        return data if len(data) == expected[0] else original

    return data if len(data) == expected else original

def _check_numpy_shape(arr: np.ndarray, expected: Optional[Union[int, Tuple[int, ...]]]) -> bool:
    """Validates numpy array shape or length."""
    if expected is None:
        return True
    if isinstance(expected, tuple):
        return arr.shape == expected
    return arr.ndim == 1 and len(arr) == expected

def _resample_streamline(streamline: np.ndarray, nb_points: int = 100) -> np.ndarray:
    """
    Resamples a streamline to a fixed number of points using linear interpolation.
    Necessary for calculating mean or distance matrices.
    """
    if len(streamline) <= 1:
        return streamline
        
    # Calculate cumulative distance along the streamline
    dists = np.sqrt(np.sum(np.diff(streamline, axis=0)**2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_dists[-1]
    
    if total_length == 0:
        return np.repeat(streamline[0][None, :], nb_points, axis=0)
        
    # Generate new distances
    new_dists = np.linspace(0, total_length, nb_points)
    
    # Interpolate X, Y, Z
    new_x = np.interp(new_dists, cum_dists, streamline[:, 0])
    new_y = np.interp(new_dists, cum_dists, streamline[:, 1])
    new_z = np.interp(new_dists, cum_dists, streamline[:, 2])
    
    return np.stack((new_x, new_y, new_z), axis=1)

def _compute_centroid_math(streamlines: List[np.ndarray], nb_points: int = 100) -> np.ndarray:
    """
    Computes the mean streamline (centroid).
    Handles orientation flipping to ensure streamlines align before averaging.
    """
    if not streamlines:
        return None
    
    # 1. Resample all to same number of points
    resampled = [_resample_streamline(s, nb_points) for s in streamlines]
    ref = resampled[0]
    
    aligned_streamlines = [ref]
    
    # 2. Align all subsequent streamlines to the reference (the first one)
    # MDF (Mean Direct Flip) distance logic for alignment
    for i in range(1, len(resampled)):
        s = resampled[i]
        
        # Direct distance
        dist_direct = np.mean(np.linalg.norm(ref - s, axis=1))
        # Flipped distance
        dist_flipped = np.mean(np.linalg.norm(ref - s[::-1], axis=1))
        
        if dist_flipped < dist_direct:
            aligned_streamlines.append(s[::-1])
        else:
            aligned_streamlines.append(s)
            
    # 3. Compute arithmetic mean
    centroid = np.mean(aligned_streamlines, axis=0)
    return centroid

def _compute_medoid_math(streamlines: List[np.ndarray], nb_points: int = 100, parent: Optional[QWidget] = None) -> int:
    """
    Identifies the index of the medoid streamline.
    The medoid is the streamline that minimizes the sum of distances (MDF) to all other streamlines.
    """
    if not streamlines:
        return -1
    
    n = len(streamlines)
    if n == 1:
        return 0
        
    # Resample streamlines for consistent distance calculation
    resampled = np.array([_resample_streamline(s, nb_points) for s in streamlines])
        
    # 2. Compute Distance Matrix (MDF - Minimum Direct Flip)
    # This is O(N^2), but for typical editing bundles it's acceptable.
    # For very large N, this might freeze the UI briefly. Added a progress dialog. 
    dist_matrix = np.zeros((n, n))
    
    # Progress dialog
    progress = QProgressDialog("Calculating Medoid... ", "Cancel", 0, n, parent)
    progress.setWindowTitle("TractEdit") 
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(500) # Only show if calculation takes > 0.5s
    progress.setValue(0)
    
    # O(N^2) Distance calculation
    for i in range(n):
        if progress.wasCanceled():
            return -1 # Return error code if cancelled
        progress.setValue(i)
        
        for j in range(i + 1, n):
            s1 = resampled[i]
            s2 = resampled[j]
            
            # MDF (Mean Direct Flip) Distance
            d_direct = np.mean(np.linalg.norm(s1 - s2, axis=1))
            d_flipped = np.mean(np.linalg.norm(s1 - s2[::-1], axis=1))
            dist = min(d_direct, d_flipped)
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist 
            
    progress.setValue(n) # Ensure bar fills completely
            
    # Sum rows to find total distance for each candidate
    total_dists = np.sum(dist_matrix, axis=1)
    
    # 4. Argmin is the medoid index
    return int(np.argmin(total_dists))

def calculate_and_save_statistic(main_window: Any, method: str) -> None:
    """
    Calculates and saves a statistic (centroid or medoid) of the visible streamlines.
    Unifies logic for calculate_and_save_centroid and calculate_and_save_medoid
    while preserving exact UX and logic.

    Args:
        main_window: The main application window instance.
        method: 'centroid' or 'medoid'.
    """
    # Ensure method is lowercase for logic, capitalize for UI strings
    method = method.lower()
    if method not in ['centroid', 'medoid']:
        logger.error(f"Invalid method for statistic calculation: {method}")
        return

    method_ui = method.capitalize()
    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: logger.info(f"Status: {msg}"))

    # 1. Validation
    if not _validate_save_prerequisites(main_window):
        return
    if not main_window.visible_indices:
        QMessageBox.warning(main_window, "Calculation Error", f"No visible streamlines to calculate {method}.")
        return

    # --- Medoid Specific Safety Check ---
    if method == 'medoid' and len(main_window.visible_indices) > 100000:
        QMessageBox.warning(main_window, "Safety Warning", 
                            f"Too many streamlines selected ({len(main_window.visible_indices)}).\n"
                            "Medoid calculation is computationally intensive (O(N²)) and would freeze the application.\n"
                            "Please reduce the selection to below 100,000 streamlines.")
        status_updater("Medoid calculation aborted (too many streamlines).")
        return
    
    # 2. Extract Visible Data
    tractogram_data = main_window.tractogram_data
    visible_streamlines = [tractogram_data[i] for i in main_window.visible_indices]
    
    # Status Update
    if method == 'medoid':
        status_updater("Calculating medoid (O(N^2) complexity)...")
    else:
        status_updater("Calculating centroid (this may take a moment)...")
        
    QApplication.processEvents()

    try:
        # 3. Calculate
        result_streamline = None
        
        if method == 'centroid':
            result_streamline = _compute_centroid_math(visible_streamlines)
        elif method == 'medoid':
            # Medoid math returns an index, need to check for cancellation (-1)
            medoid_local_index = _compute_medoid_math(visible_streamlines, parent=main_window)
            if medoid_local_index == -1:
                status_updater("Medoid calculation cancelled by user.")
                return
            result_streamline = visible_streamlines[medoid_local_index]

        # 4. Prepare for Saving
        # The result is a single new streamline. Reuse the original affine.
        affine = main_window.original_trk_affine
        
        new_tractogram = nib.streamlines.Tractogram(
            [result_streamline], 
            affine_to_rasmm=affine
        )
        
        # 5. Get Save Path
        original_path = main_window.original_trk_path
        base, ext = os.path.splitext(original_path)
        suggested_name = f"{base}_{method}{ext}"
        
        file_filter = "TrackVis TRK Files (*.trk);;TCK Files (*.tck);;TRX Files (*.trx)"
        output_path, _ = QFileDialog.getSaveFileName(
            main_window, f"Save {method_ui} As", suggested_name, file_filter
        )
        
        if not output_path:
            status_updater(f"{method_ui} save cancelled.")
            return

        _, out_ext = os.path.splitext(output_path)
        
        # 6. Save Logic
        header = {}
        if out_ext.lower() == '.trk':
            header = _prepare_trk_header(main_window.original_trk_header, 1, main_window.anatomical_image_affine)
        elif out_ext.lower() == '.tck':
            header = _prepare_tck_header(main_window.original_trk_header, 1)
        elif out_ext.lower() == '.trx':
            header = _prepare_trx_header(main_window.original_trk_header, 1)
            
        _save_tractogram_file(new_tractogram, header, output_path, out_ext.lower())
        status_updater(f"{method_ui} saved: {os.path.basename(output_path)}")

    except Exception as e:
        logger.error(f"Failed to calculate {method}: {e}", exc_info=True)
        QMessageBox.critical(main_window, "Error", f"Failed to calculate {method}:\n{e}")
        status_updater(f"Error calculating {method}.")
        
# --- Helper Function for VTK/UI Update ---
def _update_vtk_and_ui_after_load(main_window: Any, status_msg: str, render: bool = True) -> None:
    """
    Updates VTK panel and main window UI elements after loading.
    
    Args:
        main_window: Reference to the main window.
        status_msg: Status message to display.
        render: If True, calls update_main_streamlines_actor(). Set to False if
                actor has already been updated (e.g., by auto-skip logic).
    """
    if main_window.vtk_panel:
        # Only update the actor if requested
        if render:
            main_window.vtk_panel.update_main_streamlines_actor()
            
        if main_window.vtk_panel.scene and not main_window.anatomical_image_data:
            main_window.vtk_panel.scene.reset_camera()
            main_window.vtk_panel.scene.reset_clipping_range()
        elif not main_window.vtk_panel.scene:
            logger.warning("vtk_panel.scene not available for camera reset.")

        main_window.vtk_panel.update_status(status_msg)

        if main_window.vtk_panel.render_window:
            main_window.vtk_panel.render_window.Render()
        else:
            logger.warning("render_window not available.")
    else:
        logger.error("Error: vtk_panel not available to update actors.")
        logger.error(f"Status: {status_msg}")

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
        logger.error("Scene not initialized in vtk_panel.")
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
        status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: logger.info(f"Status: {msg}"))
        status_updater("Anatomical image load cancelled.")
        return None, None, None

    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: logger.info(f"Status: {msg}"))
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
        logger.error(error_msg)
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error: File not found - {os.path.basename(input_path)}")
        return None, None, None
    except nib.filebasedimages.ImageFileError as e:
        error_msg = f"Nibabel Error loading anatomical image:\n{e}\n\nIs '{os.path.basename(input_path)}' a valid NIfTI file?"
        logger.error(error_msg)
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error loading NIfTI: {os.path.basename(input_path)}")
        return None, None, None
    except Exception as e:
        error_msg = f"An unexpected error occurred loading the anatomical image:\n{type(e).__name__}: {e}\n\nPath: {input_path}\n\nSee console for details."
        logger.error(error_msg)
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error loading image: {os.path.basename(input_path)}")
        return None, None, None
    
# --- ROI Image Loading Function ---
def load_roi_images(main_window: Any) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Loads multiple NIfTI ROI files (.nii, .nii.gz).
    Returns a list of tuples, where each tuple is: (image data, affine matrix, file path).

    Args:
        main_window: The instance of the main application window.

    Returns:
        List[Tuple[...]]: A list containing data for all successfully loaded ROIs.
    """
    if not hasattr(main_window, 'vtk_panel') or not main_window.vtk_panel.scene:
        logger.error("Error: Scene not initialized in vtk_panel.")
        QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
        return []

    file_filter = "NIfTI Image Files (*.nii *.nii.gz);;All Files (*.*)"
    start_dir = ""
    if main_window.anatomical_image_path:
        start_dir = os.path.dirname(main_window.anatomical_image_path)
    elif main_window.original_trk_path:
        start_dir = os.path.dirname(main_window.original_trk_path)

    # CHANGED: Use getOpenFileNames (plural) to allow multiple selection
    input_paths, _ = QFileDialog.getOpenFileNames(main_window, "Select Input ROI Image File(s)", start_dir, file_filter)

    if not input_paths:
        status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: logger.info(f"Status: {msg}"))
        status_updater("ROI image load cancelled.")
        return []

    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: logger.info(f"Status: {msg}"))
    
    loaded_rois = []

    # CHANGED: Iterate through all selected paths
    for input_path in input_paths:
        status_updater(f"Loading ROI: {os.path.basename(input_path)}...")
        QApplication.processEvents()

        try:
            img = nib.load(input_path)

            image_data = img.get_fdata(dtype=np.float32) # Ensure float
            image_affine = img.affine

            if image_data.ndim < 3:
                logger.warning(f"Skipping {os.path.basename(input_path)}: Has {image_data.ndim} dims, expected 3+.")
                continue
            if image_affine.shape != (4, 4):
                logger.warning(f"Skipping {os.path.basename(input_path)}: Invalid affine shape.")
                continue

            loaded_rois.append((image_data, image_affine, input_path))
            status_updater(f"Successfully loaded ROI: {os.path.basename(input_path)}")

        except Exception as e:
            error_msg = f"Error loading {os.path.basename(input_path)}:\n{type(e).__name__}: {e}"
            logger.error(error_msg)
            # Optional: Show error for specific file but continue loading others
            QMessageBox.warning(main_window, "Load Error", error_msg)

    return loaded_rois

# --- Streamline File I/O Functions ---
def load_streamlines_file(main_window: Any) -> None:
    """
    Loads a trk, tck, or trx file.
    Updates the MainWindow state.
    """
    if not hasattr(main_window, 'vtk_panel') or not main_window.vtk_panel.scene:
        logger.error("Error: Scene not initialized in vtk_panel.")
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
        status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: logger.info(f"Status: {msg}"))
        status_updater("Streamline file load cancelled.")
        return

    # Clean existing bundle first (if any)
    if main_window.tractogram_data:
        main_window._close_bundle()

    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: logger.info(f"Status: {msg}"))
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
        tractogram_obj: Any = None 
        num_streamlines = 0
        
        # Clear any old trx file reference
        if hasattr(main_window, 'trx_file_reference'):
             try:
                 main_window.trx_file_reference.close()
             except Exception:
                 pass 
        main_window.trx_file_reference = None

        if ext in ['.trk', '.tck']:
            # lazy_load=True returns a generator for .trk/.tck.
            trk_file = nib.streamlines.load(input_path, lazy_load=True)
            
            # --- START PROGRESS BAR LOGIC (Added) ---
            total_sl = 0
            if hasattr(trk_file, 'header'):
                h = trk_file.header
                if 'nb_streamlines' in h:
                    try: total_sl = int(h['nb_streamlines'])
                    except: pass
                elif 'count' in h:
                    try: total_sl = int(h['count'])
                    except: pass
            
            progress = QProgressDialog("Loading File... ", "Cancel", 0, total_sl, main_window)
            progress.setWindowTitle("TractEdit")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(500) 
            progress.setValue(0)
            
            if total_sl <= 0: progress.setRange(0, 0) 

            streamlines_list = []
            load_canceled = False

            for i, sl in enumerate(trk_file.streamlines):
                streamlines_list.append(sl)
                # Update every 2000 items to avoid UI overhead
                if i % 2000 == 0:
                    progress.setValue(i)
                    if progress.wasCanceled():
                        load_canceled = True
                        break
            
            progress.setValue(total_sl if total_sl > 0 else len(streamlines_list))
            progress.close()

            if load_canceled:
                status_updater("Load cancelled by user.")
                return
            
            num_streamlines = len(streamlines_list)
            logger.info(f"Loaded {num_streamlines} streamlines from {ext} file into memory.")

            loaded_streamlines_obj = nib.streamlines.ArraySequence(streamlines_list)
            loaded_header = trk_file.header.copy() if hasattr(trk_file, 'header') else {}
            tractogram_obj = trk_file.tractogram 

            if hasattr(tractogram_obj, 'affine_to_rasmm'):
                nib_affine = tractogram_obj.affine_to_rasmm
                if isinstance(nib_affine, np.ndarray) and nib_affine.shape == (4, 4):
                    loaded_affine = nib_affine
                else:
                    logger.warning(f"Loaded affine_to_rasmm is not a valid 4x4 numpy array. Using identity affine.")
            else:
                logger.warning("affine_to_rasmm not found. Using identity affine.")
            
            if hasattr(tractogram_obj, 'data_per_point') and tractogram_obj.data_per_point:
                logger.info("Scalar data found in file (data_per_point).")
                scalar_data = {}
                try:
                    for key, value_list in tractogram_obj.data_per_point.items():
                        scalar_data[key] = nib.streamlines.ArraySequence(value_list)
                    if scalar_data:
                        active_scalar = list(scalar_data.keys())[0]
                        logger.info(f"Assigned scalars. Active scalar: '{active_scalar}'") 
                except Exception:
                    logger.warning(f"Could not process scalar data.", exc_info=True)
                    scalar_data = None
                    active_scalar = None
            else:
                 logger.info("No scalar data found in file.") 

        elif ext == '.trx':
            trx_obj = tbx.load(input_path)
            loaded_streamlines_obj = trx_obj.streamlines
            num_streamlines = len(loaded_streamlines_obj)
            logger.info(f"Loaded {num_streamlines} streamlines from {ext} file (lazy-loaded).") 
            
            loaded_header = trx_obj.header.copy()
            tractogram_obj = trx_obj
            
            if hasattr(trx_obj, 'affine_to_rasmm'):
                trx_affine = trx_obj.affine_to_rasmm
                if isinstance(trx_affine, np.ndarray) and trx_affine.shape == (4, 4):
                    loaded_affine = trx_affine
                else:
                    logger.warning("affine_to_rasmm not found. Using identity affine.") 
            else:
                    logger.warning("affine_to_rasmm not found. Using identity affine.") 

            if hasattr(trx_obj, 'data_per_point') and trx_obj.data_per_point:
                logger.info("Scalar data found in file (data_per_point).")
                scalar_data = trx_obj.data_per_point
                if scalar_data:
                    active_scalar = list(scalar_data.keys())[0]
                    logger.info(f"Assigned scalars. Active scalar: '{active_scalar}'")
            else:
                logger.info("No scalar data found in file.")
            
            main_window.trx_file_reference = trx_obj

        else:
            raise ValueError(f"Unsupported file extension: '{ext}'.")

        if not loaded_streamlines_obj or num_streamlines == 0:
            logger.info("No streamlines found in file.")
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
            
            main_window._update_bundle_info_display()
            main_window._update_action_states()
            if main_window.vtk_panel:
                main_window.vtk_panel.update_main_streamlines_actor()
                main_window.vtk_panel.update_highlight()
                main_window.vtk_panel.update_radius_actor(visible=False)
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

        main_window.selected_streamline_indices = set()
        main_window.undo_stack = []
        main_window.redo_stack = []
        main_window.current_color_mode = ColorMode.ORIENTATION

        main_window.scalar_data_per_point = scalar_data
        main_window.active_scalar_name = active_scalar

        # Calculate skip level BEFORE the initial render.
        # This prevents rendering 500k fibers only to immediately clear them and render 20k.
        should_render_in_update = True
        
        if hasattr(main_window, '_auto_calculate_skip_level'):
             main_window._auto_calculate_skip_level()
             should_render_in_update = False 

        # --- Update VTK and UI ---
        status_msg = f"Loaded {len(main_window.tractogram_data)} streamlines from {os.path.basename(input_path)}"
        if main_window.active_scalar_name:
            status_msg += f" | Active Scalar: {main_window.active_scalar_name}"
        
        # Pass False to render
        _update_vtk_and_ui_after_load(main_window, status_msg, render=should_render_in_update)

    except Exception as e:
        logger.error(f"Error during streamline loading: {e}", exc_info=True)
        
        error_title = "Load Error"
        if isinstance(e, nib.filebasedimages.ImageFileError):
            error_msg = f"Nibabel Error: {e}\n\nIs the file a valid TRK or TCK format?"
        elif isinstance(e, FileNotFoundError):
            error_msg = f"Error: Streamline file not found:\n{input_path}"
        else:
            error_msg = f"Error loading streamline file:\n{type(e).__name__}: {e}\n\nSee console for details."

        try:
            QMessageBox.critical(main_window, error_title, error_msg)
        except Exception:
            pass

        status_updater(f"Error loading file: {os.path.basename(input_path)}")
        try:
            main_window.tractogram_data = None
            if main_window.vtk_panel:
                main_window.vtk_panel.update_main_streamlines_actor()
            main_window._update_bundle_info_display()
            main_window._update_action_states()
        except Exception:
            pass

def _validate_save_prerequisites(main_window: Any) -> bool:
    """Checks if prerequisites for saving streamlines are met."""
    if main_window.tractogram_data is None:
        logger.error("Save Error: No streamline data to save.")
        QMessageBox.warning(main_window, "Save Error", "No streamline data to save.")
        return False
    if main_window.original_trk_affine is None:
        logger.error("Save Error: Original streamline affine info missing.")
        QMessageBox.critical(main_window, "Save Error", "Original streamline file affine info missing (needed for saving).")
        return False
    if main_window.original_trk_header is None:
        logger.warning("Warning: Original streamline header info missing. Saving with minimal header.")
        main_window.original_trk_header = {} # Ensure it's a dict
    
    if main_window.original_file_extension not in ['.trk', '.tck', '.trx']:
        logger.error(f"Save Error: Cannot determine original format ('{main_window.original_file_extension}').")
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
        output_path = os.path.splitext(output_path)[0] + required_ext
        if output_ext_from_dialog == "": # No extension was provided
            logger.info(f"Save Info: Appended required extension '{required_ext}'. New path: {output_path}")
        else:
            QMessageBox.warning(main_window, "Save Format Corrected",
                                f"File extension was corrected from '{output_ext_from_dialog}' to the required '{required_ext}'.\n"
                                f"Saving as: {os.path.basename(output_path)}")
            logger.info(f"Save Info: Corrected extension from '{output_ext_from_dialog}' to '{required_ext}'. Path changed from '{old_output_path}' to '{output_path}'.")
    return output_path, required_ext

def _prepare_tractogram_and_affine(main_window: Any) -> nib.streamlines.Tractogram:
    """Prepares the Tractogram object and validates the affine matrix."""
    tractogram = main_window.tractogram_data
    indices_to_save = sorted(list(main_window.visible_indices))
    
    # Create a generator for the streamlines to save
    streamlines_to_save_gen = (tractogram[i] for i in indices_to_save)

    affine_matrix = main_window.original_trk_affine

    if not isinstance(affine_matrix, np.ndarray) or affine_matrix.shape != (4, 4):
        logger.warning(f"Warning: Affine matrix invalid. Using identity.")
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
            logger.warning(f"Warning: Could not filter scalar data for saving. Saving without scalars. Error: {e}")
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
    logger.info("Preparing TRK header for saving...") 

    # --- Voxel Order Logic ---
    raw_voxel_order_from_trk = header.get('voxel_order')
    processed_voxel_order_from_trk = None

    if isinstance(raw_voxel_order_from_trk, bytes):
        try:
            processed_voxel_order_from_trk = raw_voxel_order_from_trk.decode('utf-8', errors='strict')
            logger.debug(f"Decoded 'voxel_order': '{processed_voxel_order_from_trk}'") 
        except UnicodeDecodeError:
            logger.warning(f"'voxel_order' field in TRK header (bytes: {raw_voxel_order_from_trk}) could not be decoded.") 
    elif isinstance(raw_voxel_order_from_trk, str):
        processed_voxel_order_from_trk = raw_voxel_order_from_trk

    is_valid_trk_voxel_order = isinstance(processed_voxel_order_from_trk, str) and len(processed_voxel_order_from_trk) == 3

    if is_valid_trk_voxel_order:
        header['voxel_order'] = processed_voxel_order_from_trk.upper()
        logger.info(f"      - Info: Using existing 'voxel_order' from TRK header: {header['voxel_order']}.")
    else:
        if raw_voxel_order_from_trk is not None:
            logger.warning(f"      - Warning: 'voxel_order' from TRK header ('{raw_voxel_order_from_trk}') is invalid or in an unexpected format.")
        else:
            logger.warning(f"      - Info: 'voxel_order' missing in TRK header.")

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
                    logger.info(f"      - Info: Derived 'voxel_order' from loaded anatomical image: {header['voxel_order']}.")
                else:
                    logger.warning(f"      - Warning: Could not derive a valid 3-character 'voxel_order' from anatomical image affine (got: '{derived_vo_str}').")
            except Exception as e:
                logger.warning(f"      - Warning: Error deriving 'voxel_order' from anatomical image affine: {e}.")

        if not derived_from_anat:
            header['voxel_order'] = 'RAS'
            if raw_voxel_order_from_trk is None and anatomical_img_affine is None:
                logger.info(f"      - Info: 'voxel_order' missing, no anatomical image. Defaulting to 'RAS'.")
            elif not is_valid_trk_voxel_order and anatomical_img_affine is None:
                logger.info(f"      - Info: Original 'voxel_order' invalid/missing, no anatomical image. Defaulting to 'RAS'.")
            else: # Covers cases where derivation from anat failed or anat_img_affine was invalid
                logger.info(f"      - Info: Could not use original or derive 'voxel_order' from anatomical image. Defaulting to 'RAS'.")

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
                logger.warning(f"      - Warning: Could not decode bytes for '{key}'. Original value: {original_value}")
                header[key] = K_props['default']
                logger.warning(f"      - Info: Set '{key}' to default: {header[key]}")
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
                logger.info(f"      - Info: Could not parse string '{processed_value}' for '{key}'.")
        
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
            logger.warning(f"      - Warning: Type conversion error for '{key}' (value: '{processed_value}'): {e}")
            valid_structure = False 

        if valid_structure:
            header[key] = final_value
        else:
            logger.warning(f"      - Warning: '{key}' ('{original_value}') was invalid, missing, or failed processing. Defaulted to {K_props['default']}.")
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
        logger.info("File saved successfully (TRK)") 
        return f"File saved successfully (TRK): {os.path.basename(output_path)}"
    
    elif file_ext == '.tck':
        tck_file = nib.streamlines.TckFile(tractogram, header=header)
        nib.streamlines.save(tck_file, output_path)
        logger.info("File saved successfully (TCK)")
        return f"File saved successfully (TCK): {os.path.basename(output_path)}"
    
    elif file_ext == '.trx':        
        # 1. 'tractogram' is our nib.streamlines.Tractogram object.
        #    'header' is our prepared header.
        trx_obj_to_save = tbx.TrxFile.from_lazy_tractogram(
            tractogram, header
        )
        
        # 2. Save the newly created object using tbx.save()
        tbx.save(trx_obj_to_save, output_path)
        logger.info("File saved successfully (TRX)")
        return f"File saved successfully (TRX): {os.path.basename(output_path)}"
    
    else:
        raise ValueError(f"Unsupported save extension: {file_ext}")

# --- Main Save Function ---
def save_streamlines_file(main_window: Any) -> None:
    """
    Saves the current streamlines to a trk, tck, or trx file.
    """
    status_updater = getattr(main_window.vtk_panel, 'update_status', lambda msg: logger.info(f"Status: {msg}"))

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
        logger.error(f"Error during file saving:\nType: {type(e).__name__}\nError: {e}")
        error_msg = f"Error saving file:\n{type(e).__name__}: {e}\n\nCheck console for details."
        QMessageBox.critical(main_window, "Save Error", error_msg)
        status_updater(f"Error saving file: {os.path.basename(output_path)}")