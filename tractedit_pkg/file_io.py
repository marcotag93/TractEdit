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
import vtk
from vtk.util import numpy_support
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QApplication,
    QProgressDialog,
    QWidget,
)
from .utils import ColorMode
from typing import Optional, List, Dict, Any, Tuple, Type, Union
from numba import njit

logger = logging.getLogger(__name__)


class StreamlineLoaderThread(QThread):
    """
    Background thread to load streamline files without freezing the GUI.
    """

    progress = pyqtSignal(int, str)  # Signal to update progress bar (percent, message)
    finished = pyqtSignal(dict)  # Signal when loading is done
    error = pyqtSignal(str)  # Signal if an error occurs

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def run(self):
        try:
            _, ext = os.path.splitext(self.input_path)
            ext = ext.lower()
            results = {"path": self.input_path, "ext": ext}

            if ext == ".trk":
                self.progress.emit(20, "Reading TRK file...")
                trk_file = nib.streamlines.TrkFile.load(
                    self.input_path, lazy_load=False
                )
                tractogram_obj = trk_file.tractogram
                loaded_streamlines = tractogram_obj.streamlines

                # Header handling for TRK
                results["header"] = (
                    trk_file.header.copy() if hasattr(trk_file, "header") else {}
                )

            elif ext == ".tck":
                self.progress.emit(20, "Reading TCK file...")
                tck_file = nib.streamlines.TckFile.load(
                    self.input_path, lazy_load=False
                )
                tractogram_obj = tck_file.tractogram
                loaded_streamlines = tractogram_obj.streamlines

                # Header handling for TCK
                results["header"] = (
                    tck_file.header.copy() if hasattr(tck_file, "header") else {}
                )

            if ext in [".trk", ".tck"]:
                # Shared post-processing for TRK/TCK
                # Optimization
                self.progress.emit(50, "Rendering...")
                if (
                    hasattr(loaded_streamlines, "_data")
                    and loaded_streamlines._data.dtype != np.float32
                ):
                    loaded_streamlines._data = loaded_streamlines._data.astype(
                        np.float32, copy=False
                    )

                # Header already set above
                results["streamlines"] = loaded_streamlines

                # Handle Affine
                aff = np.identity(4)
                if hasattr(tractogram_obj, "affine_to_rasmm"):
                    temp_aff = tractogram_obj.affine_to_rasmm
                    if isinstance(temp_aff, np.ndarray) and temp_aff.shape == (4, 4):
                        aff = temp_aff
                results["affine"] = aff

                # Handle Scalars
                scalars = {}
                active_scalar = None
                if (
                    hasattr(tractogram_obj, "data_per_point")
                    and tractogram_obj.data_per_point
                ):
                    for k, v in tractogram_obj.data_per_point.items():
                        scalars[k] = nib.streamlines.ArraySequence(v)
                    if scalars:
                        active_scalar = list(scalars.keys())[0]
                results["scalars"] = scalars
                results["active_scalar"] = active_scalar

                # Geometry (BBox)
                self.progress.emit(70, "Finalizing...")
                # Attempt fast Numba calc
                if hasattr(loaded_streamlines, "_data") and hasattr(
                    loaded_streamlines, "_offsets"
                ):
                    flat_data = loaded_streamlines._data
                    offsets = loaded_streamlines._offsets
                    lengths = loaded_streamlines._lengths
                    results["bboxes"] = _compute_bboxes_numba(
                        flat_data, offsets, lengths
                    )
                else:
                    # Fallback slow calc
                    bboxes = []
                    for sl in loaded_streamlines:
                        if len(sl) > 0:
                            bboxes.append([np.min(sl, axis=0), np.max(sl, axis=0)])
                        else:
                            bboxes.append([np.zeros(3), np.zeros(3)])
                    results["bboxes"] = np.array(bboxes, dtype=np.float32)

            elif ext in [".vtk", ".vtp"]:
                self.progress.emit(10, "Reading VTK file...")

                # Select Reader
                if ext == ".vtp":
                    reader = vtk.vtkXMLPolyDataReader()
                else:
                    reader = vtk.vtkPolyDataReader()

                reader.SetFileName(self.input_path)
                reader.Update()
                poly_data = reader.GetOutput()

                # Extract Points
                self.progress.emit(30, "Parsing geometry...")
                vtk_points = poly_data.GetPoints()
                if vtk_points:
                    points_data = numpy_support.vtk_to_numpy(vtk_points.GetData())
                else:
                    points_data = np.empty((0, 3), dtype=np.float32)

                # Extract Lines (Streamlines)
                vtk_lines = poly_data.GetLines()
                lines_data = numpy_support.vtk_to_numpy(vtk_lines.GetData())

                # Parse VTK Cell Array (Format: [n_points, id1, id2... n_points, id1...])
                loaded_streamlines = []
                idx = 0
                total_cells = poly_data.GetNumberOfLines()

                # Quick progress update loop
                progress_step = max(1, total_cells // 20)
                cell_count = 0

                while idx < len(lines_data):
                    if cell_count % progress_step == 0:
                        self.progress.emit(
                            30 + int(20 * (cell_count / total_cells)),
                            "Parsing geometry...",
                        )

                    n_pts = lines_data[idx]
                    idx += 1

                    point_indices = lines_data[idx : idx + n_pts]
                    idx += n_pts

                    # Fancy indexing to get coordinates
                    streamline = points_data[point_indices]
                    loaded_streamlines.append(streamline.astype(np.float32))
                    cell_count += 1

                # Determine Affine
                # VTK files are usually already in world coordinates. We use Identity.
                aff = np.identity(4)

                results["streamlines"] = loaded_streamlines
                results["header"] = {}  # VTK has no standard header
                results["affine"] = aff

                # Handle Scalars (Point Data)
                self.progress.emit(60, "Reading scalars...")
                scalars = {}
                point_data = poly_data.GetPointData()
                n_arrays = point_data.GetNumberOfArrays()

                for i in range(n_arrays):
                    arr_name = point_data.GetArrayName(i)
                    vtk_arr = point_data.GetArray(i)
                    np_arr = numpy_support.vtk_to_numpy(vtk_arr)

                    # We need to split the scalar array to match streamlines structure
                    scalar_sequence = []
                    current_idx = 0
                    for sl in loaded_streamlines:
                        sl_len = len(sl)
                        scalar_sequence.append(
                            np_arr[current_idx : current_idx + sl_len]
                        )
                        current_idx += sl_len

                    if arr_name:
                        scalars[arr_name] = nib.streamlines.ArraySequence(
                            scalar_sequence
                        )
                    else:
                        scalars[f"Scalar_{i}"] = nib.streamlines.ArraySequence(
                            scalar_sequence
                        )

                results["scalars"] = scalars
                results["active_scalar"] = list(scalars.keys())[0] if scalars else None

                # Geometry (BBox) - Use Numba optimization if possible
                self.progress.emit(80, "Finalizing...")
                try:
                    # ArraySequence creates the flat_data/_offsets structure internally
                    as_streamlines = nib.streamlines.ArraySequence(loaded_streamlines)
                    results["bboxes"] = _compute_bboxes_numba(
                        as_streamlines._data,
                        as_streamlines._offsets,
                        as_streamlines._lengths,
                    )
                except Exception:
                    # Fallback
                    bboxes = []
                    for sl in loaded_streamlines:
                        if len(sl) > 0:
                            bboxes.append([np.min(sl, axis=0), np.max(sl, axis=0)])
                        else:
                            bboxes.append([np.zeros(3), np.zeros(3)])
                    results["bboxes"] = np.array(bboxes, dtype=np.float32)

            elif ext == ".trx":
                self.progress.emit(10, "Loading TRX file...")
                trx_obj = tbx.load(self.input_path)
                results["trx_obj"] = trx_obj
                results["streamlines"] = trx_obj.streamlines
                results["header"] = trx_obj.header.copy()

                # Affine
                aff = np.identity(4)
                if hasattr(trx_obj, "affine_to_rasmm"):
                    temp_aff = trx_obj.affine_to_rasmm
                    if isinstance(temp_aff, np.ndarray) and temp_aff.shape == (4, 4):
                        aff = temp_aff
                results["affine"] = aff

                # Scalars (Basic check)
                scalars = {}
                dpp = getattr(trx_obj, "data_per_vertex", None)
                if dpp:
                    scalars.update(dpp)
                results["scalars"] = scalars
                results["active_scalar"] = list(scalars.keys())[0] if scalars else None

                # Geometry (Progressive)
                self.progress.emit(30, "Finalizing...")
                bboxes = []
                total = len(trx_obj.streamlines)
                step = max(1, total // 50)  # Update every 2%

                for i, sl in enumerate(trx_obj.streamlines):
                    if i % step == 0:
                        pct = 30 + int(60 * (i / total))
                        self.progress.emit(
                            pct, f"Calculating geometry {int(i/total*100)}%..."
                        )

                    if len(sl) > 0:
                        bboxes.append([np.min(sl, axis=0), np.max(sl, axis=0)])
                    else:
                        bboxes.append([np.zeros(3), np.zeros(3)])
                results["bboxes"] = np.array(bboxes, dtype=np.float32)

            else:
                self.error.emit(f"Unsupported file format: {ext}")
                return

            self.progress.emit(100, "Done")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class AnatomicalImageLoaderThread(QThread):
    """
    Background thread to load anatomical images without freezing the GUI.
    Useful for large images like MNI152 0.5mm or MGH 100 micron.
    """

    progress = pyqtSignal(int, str)  # Signal to update progress bar (percent, message)
    finished = pyqtSignal(dict)  # Signal when loading is done
    error = pyqtSignal(str)  # Signal if an error occurs

    def __init__(self, input_path: str):
        super().__init__()
        self.input_path = input_path

    def run(self):
        try:
            self.progress.emit(10, "Loading NIfTI header...")

            # Load the NIfTI file
            img = nib.load(self.input_path)

            # Get file size estimate for progress feedback
            header = img.header
            shape = header.get_data_shape()
            if len(shape) >= 3:
                total_voxels = shape[0] * shape[1] * shape[2]
                size_mb = (total_voxels * 4) / (1024 * 1024)  # float32 = 4 bytes
                self.progress.emit(
                    20,
                    f"Loading {shape[0]}×{shape[1]}×{shape[2]} ({size_mb:.0f} MB)...",
                )
            else:
                self.progress.emit(20, "Loading image data...")

            # This is the slow part - get_fdata loads and converts the data
            self.progress.emit(30, "Reading image data (this may take a while)...")
            image_data = img.get_fdata(dtype=np.float32)

            self.progress.emit(80, "Validating...")
            image_affine = img.affine

            # Basic validation
            if image_data.ndim < 3:
                self.error.emit(
                    f"Loaded image has only {image_data.ndim} dimensions, expected 3 or more."
                )
                return
            if image_affine.shape != (4, 4):
                self.error.emit(
                    f"Loaded image affine has shape {image_affine.shape}, expected (4, 4)."
                )
                return

            self.progress.emit(100, "Done")
            self.finished.emit(
                {
                    "data": image_data,
                    "affine": image_affine,
                    "path": self.input_path,
                }
            )

        except FileNotFoundError:
            self.error.emit(f"File not found: {self.input_path}")
        except nib.filebasedimages.ImageFileError as e:
            self.error.emit(f"Invalid NIfTI file: {e}")
        except Exception as e:
            self.error.emit(f"Error loading image: {type(e).__name__}: {e}")


# Numba Optimized Helper
@njit(nogil=True)
def _compute_bboxes_numba(
    flat_data: np.ndarray, offsets: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """
    Calculates bounding boxes for streamlines using Numba for high performance.

    Args:
        flat_data: The flattened coordinates array (N_total_points, 3).
        offsets: Array of start indices for each streamline.
        lengths: Array of point counts for each streamline.

    Returns:
        A (N_streamlines, 2, 3) array containing [min_coords, max_coords] for each streamline.
    """
    n_streamlines = len(lengths)
    bboxes = np.zeros((n_streamlines, 2, 3), dtype=np.float32)

    for i in range(n_streamlines):
        start = offsets[i]
        length = lengths[i]

        if length == 0:
            continue

        # Initialize min/max with the first point of the streamline
        first_idx = start
        min_x = flat_data[first_idx, 0]
        max_x = flat_data[first_idx, 0]
        min_y = flat_data[first_idx, 1]
        max_y = flat_data[first_idx, 1]
        min_z = flat_data[first_idx, 2]
        max_z = flat_data[first_idx, 2]

        # Iterate over the rest of the points
        for j in range(1, length):
            idx = start + j
            x = flat_data[idx, 0]
            y = flat_data[idx, 1]
            z = flat_data[idx, 2]

            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            if z < min_z:
                min_z = z
            if z > max_z:
                max_z = z

        bboxes[i, 0, 0] = min_x
        bboxes[i, 0, 1] = min_y
        bboxes[i, 0, 2] = min_z
        bboxes[i, 1, 0] = max_x
        bboxes[i, 1, 1] = max_y
        bboxes[i, 1, 2] = max_z

    return bboxes


# Helper Functions
def parse_numeric_tuple_from_string(
    input_value: Union[str, List, Tuple, np.ndarray, Any],
    target_type: Type = float,
    expected_length: Optional[Union[int, Tuple[int, ...]]] = None,
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
        return _process_parsed_value(
            parsed_val, input_value, target_type, expected_length
        )
    except (ValueError, SyntaxError, TypeError):
        pass

    # 3. Strategy B: Fallback to string splitting (e.g., "1 2 3", "1, 2, 3")
    # Clean brackets and split by comma or whitespace
    cleaned_str = input_value.translate(str.maketrans("", "", "[]()"))
    parts = cleaned_str.replace(",", " ").split()

    if not parts:
        return input_value

    try:
        # Convert parts to target type
        converted = tuple(target_type(p) for p in parts)
        return _validate_length(converted, expected_length, input_value)
    except (ValueError, TypeError):
        return input_value


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


def _process_parsed_value(
    parsed: Any, original: Any, dtype: Type, length_req: Any
) -> Any:
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


def _validate_length(
    data: tuple, expected: Optional[Union[int, Tuple[int, ...]]], original: Any
) -> Any:
    """Validates that the data tuple matches the expected length."""
    if expected is None:
        return data

    # If expected is a tuple (usually for numpy shapes), we only check dimension 0 here for tuples
    if isinstance(expected, tuple):
        return data if len(data) == expected[0] else original

    return data if len(data) == expected else original


def _check_numpy_shape(
    arr: np.ndarray, expected: Optional[Union[int, Tuple[int, ...]]]
) -> bool:
    """Validates numpy array shape or length."""
    if expected is None:
        return True
    if isinstance(expected, tuple):
        return arr.shape == expected
    return arr.ndim == 1 and len(arr) == expected


@njit(nogil=True)
def _resample_streamline_numba(streamline: np.ndarray, nb_points: int) -> np.ndarray:
    """
    Numba-optimized streamline resampling using linear interpolation.
    """
    n_pts = streamline.shape[0]
    if n_pts <= 1:
        result = np.empty((nb_points, 3), dtype=np.float64)
        for i in range(nb_points):
            result[i, 0] = streamline[0, 0]
            result[i, 1] = streamline[0, 1]
            result[i, 2] = streamline[0, 2]
        return result

    # Calculate cumulative distances
    cum_dists = np.zeros(n_pts, dtype=np.float64)
    for i in range(1, n_pts):
        dx = streamline[i, 0] - streamline[i - 1, 0]
        dy = streamline[i, 1] - streamline[i - 1, 1]
        dz = streamline[i, 2] - streamline[i - 1, 2]
        cum_dists[i] = cum_dists[i - 1] + np.sqrt(dx * dx + dy * dy + dz * dz)

    total_length = cum_dists[-1]
    if total_length == 0:
        result = np.empty((nb_points, 3), dtype=np.float64)
        for i in range(nb_points):
            result[i, 0] = streamline[0, 0]
            result[i, 1] = streamline[0, 1]
            result[i, 2] = streamline[0, 2]
        return result

    # Generate new distances and interpolate
    result = np.empty((nb_points, 3), dtype=np.float64)
    for i in range(nb_points):
        target_dist = total_length * i / (nb_points - 1)

        # Find segment containing target_dist
        seg_idx = 0
        for j in range(1, n_pts):
            if cum_dists[j] >= target_dist:
                seg_idx = j - 1
                break
            seg_idx = j - 1

        # Interpolate within segment
        seg_start = cum_dists[seg_idx]
        seg_end = cum_dists[seg_idx + 1] if seg_idx + 1 < n_pts else seg_start
        seg_len = seg_end - seg_start

        if seg_len > 0:
            t = (target_dist - seg_start) / seg_len
        else:
            t = 0.0

        result[i, 0] = streamline[seg_idx, 0] + t * (
            streamline[seg_idx + 1, 0] - streamline[seg_idx, 0]
        )
        result[i, 1] = streamline[seg_idx, 1] + t * (
            streamline[seg_idx + 1, 1] - streamline[seg_idx, 1]
        )
        result[i, 2] = streamline[seg_idx, 2] + t * (
            streamline[seg_idx + 1, 2] - streamline[seg_idx, 2]
        )

    return result


@njit(nogil=True, parallel=True)
def _compute_centroid_numba(
    resampled: np.ndarray,
) -> np.ndarray:
    """
    Numba-optimized centroid computation with MDF alignment.

    Args:
        resampled: Array of shape (N, nb_points, 3) containing resampled streamlines.

    Returns:
        Centroid streamline of shape (nb_points, 3).
    """
    n = resampled.shape[0]
    nb_points = resampled.shape[1]

    if n == 0:
        return np.zeros((nb_points, 3), dtype=np.float64)

    # Reference is first streamline
    ref = resampled[0]

    # Aligned streamlines accumulator (for mean)
    centroid = np.zeros((nb_points, 3), dtype=np.float64)

    # Add reference
    for k in range(nb_points):
        centroid[k, 0] = ref[k, 0]
        centroid[k, 1] = ref[k, 1]
        centroid[k, 2] = ref[k, 2]

    # Align and accumulate remaining streamlines
    for i in range(1, n):
        s = resampled[i]

        # Compute direct distance
        d_direct = 0.0
        for k in range(nb_points):
            dx = ref[k, 0] - s[k, 0]
            dy = ref[k, 1] - s[k, 1]
            dz = ref[k, 2] - s[k, 2]
            d_direct += np.sqrt(dx * dx + dy * dy + dz * dz)
        d_direct /= nb_points

        # Compute flipped distance
        d_flipped = 0.0
        for k in range(nb_points):
            flipped_k = nb_points - 1 - k
            dx = ref[k, 0] - s[flipped_k, 0]
            dy = ref[k, 1] - s[flipped_k, 1]
            dz = ref[k, 2] - s[flipped_k, 2]
            d_flipped += np.sqrt(dx * dx + dy * dy + dz * dz)
        d_flipped /= nb_points

        # Add aligned streamline to centroid
        if d_flipped < d_direct:
            # Use flipped
            for k in range(nb_points):
                flipped_k = nb_points - 1 - k
                centroid[k, 0] += s[flipped_k, 0]
                centroid[k, 1] += s[flipped_k, 1]
                centroid[k, 2] += s[flipped_k, 2]
        else:
            # Use direct
            for k in range(nb_points):
                centroid[k, 0] += s[k, 0]
                centroid[k, 1] += s[k, 1]
                centroid[k, 2] += s[k, 2]

    # Divide by n to get mean
    for k in range(nb_points):
        centroid[k, 0] /= n
        centroid[k, 1] /= n
        centroid[k, 2] /= n

    return centroid


def _resample_streamline(streamline: np.ndarray, nb_points: int = 100) -> np.ndarray:
    """
    Resamples a streamline to a fixed number of points using linear interpolation.
    Uses Numba-optimized implementation for performance.
    """
    if len(streamline) <= 1:
        return np.repeat(streamline[0][None, :], nb_points, axis=0)
    return _resample_streamline_numba(streamline.astype(np.float64), nb_points)


def _compute_centroid_math(
    streamlines: List[np.ndarray], nb_points: int = 100
) -> np.ndarray:
    """
    Computes the mean streamline (centroid) using Numba-optimized functions.
    Handles orientation flipping to ensure streamlines align before averaging.
    """
    if not streamlines:
        return None

    # Resample all streamlines
    resampled = np.array(
        [
            _resample_streamline_numba(s.astype(np.float64), nb_points)
            for s in streamlines
        ],
        dtype=np.float64,
    )

    # Compute centroid using Numba
    return _compute_centroid_numba(resampled)


@njit(nogil=True, parallel=True)
def _compute_mdf_distance_matrix(resampled: np.ndarray) -> np.ndarray:
    """
    Computes MDF (Minimum Direct Flip) distance matrix using Numba.

    Args:
        resampled: Array of shape (N, nb_points, 3) containing resampled streamlines.

    Returns:
        Distance matrix of shape (N, N).
    """
    n = resampled.shape[0]
    nb_points = resampled.shape[1]
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            # Direct distance
            d_direct = 0.0
            for k in range(nb_points):
                dx = resampled[i, k, 0] - resampled[j, k, 0]
                dy = resampled[i, k, 1] - resampled[j, k, 1]
                dz = resampled[i, k, 2] - resampled[j, k, 2]
                d_direct += np.sqrt(dx * dx + dy * dy + dz * dz)
            d_direct /= nb_points

            # Flipped distance
            d_flipped = 0.0
            for k in range(nb_points):
                flipped_k = nb_points - 1 - k
                dx = resampled[i, k, 0] - resampled[j, flipped_k, 0]
                dy = resampled[i, k, 1] - resampled[j, flipped_k, 1]
                dz = resampled[i, k, 2] - resampled[j, flipped_k, 2]
                d_flipped += np.sqrt(dx * dx + dy * dy + dz * dz)
            d_flipped /= nb_points

            # MDF = minimum of direct and flipped
            dist = d_direct if d_direct < d_flipped else d_flipped
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def _compute_medoid_math(
    streamlines: List[np.ndarray],
    nb_points: int = 100,
    parent: Optional[QWidget] = None,
) -> int:
    """
    Identifies the index of the medoid streamline.
    The medoid is the streamline that minimizes the sum of distances (MDF) to all other streamlines.
    Uses Numba-optimized distance calculation for performance.
    """
    if not streamlines:
        return -1

    n = len(streamlines)
    if n == 1:
        return 0

    # Progress dialog for resampling phase
    progress = QProgressDialog("Resampling streamlines...", "Cancel", 0, 100, parent)
    progress.setWindowTitle("TractEdit - Medoid Calculation")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    progress.show()
    QApplication.processEvents()

    # Resample streamlines for consistent distance calculation
    resampled_list = []
    for i, s in enumerate(streamlines):
        if progress.wasCanceled():
            return -1
        resampled_list.append(_resample_streamline(s, nb_points))
        if i % max(1, n // 20) == 0:  # Update every 5%
            progress.setValue(int(40 * i / n))
            QApplication.processEvents()

    resampled = np.array(resampled_list, dtype=np.float64)

    if progress.wasCanceled():
        return -1

    # Compute distance matrix using Numba (this is the fast part)
    progress.setLabelText("Computing distance matrix (Numba-optimized)...")
    progress.setValue(50)
    QApplication.processEvents()

    dist_matrix = _compute_mdf_distance_matrix(resampled)

    progress.setValue(90)
    QApplication.processEvents()

    # Sum rows to find total distance for each candidate
    total_dists = np.sum(dist_matrix, axis=1)

    progress.setValue(100)

    # Argmin is the medoid index
    return int(np.argmin(total_dists))


def calculate_and_save_statistic(main_window: Any, method: str) -> None:
    """
    Calculates and saves a statistic (centroid or medoid) of the visible streamlines.
    Unifies logic for calculate_and_save_centroid and calculate_and_save_medoid
    while preserving UX and logic.

    Args:
        main_window: The main application window instance.
        method: 'centroid' or 'medoid'.
    """
    method = method.lower()
    if method not in ["centroid", "medoid"]:
        logger.error(f"Invalid method for statistic calculation: {method}")
        return

    method_ui = method.capitalize()
    status_updater = getattr(
        main_window.vtk_panel,
        "update_status",
        lambda msg: logger.info(f"Status: {msg}"),
    )

    # Validation
    if not _validate_save_prerequisites(main_window):
        return
    if not main_window.visible_indices:
        QMessageBox.warning(
            main_window,
            "Calculation Error",
            f"No visible streamlines to calculate {method}.",
        )
        return

    # Safety Check for Medoid
    if method == "medoid" and len(main_window.visible_indices) > 100000:
        QMessageBox.warning(
            main_window,
            "Safety Warning",
            f"Too many streamlines selected ({len(main_window.visible_indices)}).\n"
            "Medoid calculation is computationally intensive (O(N²)) and would freeze the application.\n"
            "Please reduce the selection to below 100,000 streamlines.",
        )
        status_updater("Medoid calculation aborted (too many streamlines).")
        return

    # Extract Visible Data
    tractogram_data = main_window.tractogram_data
    visible_streamlines = [tractogram_data[i] for i in main_window.visible_indices]

    # Status Update
    if method == "medoid":
        status_updater("Calculating medoid (O(N^2) complexity)...")
    else:
        status_updater("Calculating centroid (this may take a moment)...")

    QApplication.processEvents()

    try:
        # Calculate
        result_streamline = None

        if method == "centroid":
            result_streamline = _compute_centroid_math(visible_streamlines)
        elif method == "medoid":
            # Medoid math returns an index, need to check for cancellation (-1)
            medoid_local_index = _compute_medoid_math(
                visible_streamlines, parent=main_window
            )
            if medoid_local_index == -1:
                status_updater("Medoid calculation cancelled by user.")
                return
            result_streamline = visible_streamlines[medoid_local_index]

        # Prepare for Saving
        affine = main_window.original_trk_affine

        new_tractogram = nib.streamlines.Tractogram(
            [result_streamline], affine_to_rasmm=affine
        )

        # Get Save Path
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

        # Save Logic
        header = {}
        if out_ext.lower() == ".trk":
            header = _prepare_trk_header(
                main_window.original_trk_header, 1, main_window.anatomical_image_affine
            )
        elif out_ext.lower() == ".tck":
            header = _prepare_tck_header(main_window.original_trk_header, 1)
        elif out_ext.lower() == ".trx":
            header = _prepare_trx_header(
                main_window.original_trk_header,
                1,
                anatomical_img_affine=main_window.anatomical_image_affine,
            )

        _save_tractogram_file(new_tractogram, header, output_path, out_ext.lower())
        status_updater(f"{method_ui} saved: {os.path.basename(output_path)}")

    except Exception as e:
        logger.error(f"Failed to calculate {method}: {e}", exc_info=True)
        QMessageBox.critical(
            main_window, "Error", f"Failed to calculate {method}:\n{e}"
        )
        status_updater(f"Error calculating {method}.")


# Helper Function for VTK/UI Update
def _update_vtk_and_ui_after_load(
    main_window: Any, status_msg: str, render: bool = True
) -> None:
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

        if main_window.vtk_panel.scene and main_window.anatomical_image_data is None:
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
    if hasattr(main_window, "color_orientation_action"):
        main_window.color_orientation_action.setChecked(True)

    main_window._update_action_states()
    main_window._update_bundle_info_display()
    main_window._update_data_panel_display()  # Update tree widget after load

    # If parcellation data is loaded, calculate intersections for the new bundle
    if (
        hasattr(main_window, "parcellation_data")
        and main_window.parcellation_data is not None
        and hasattr(main_window, "connectivity_manager")
    ):
        try:
            main_window.connectivity_manager.recalculate_all_intersections()
        except Exception as e:
            logger.warning(f"Could not recalculate parcellation intersection: {e}")


# Anatomical Image Loading Function
def load_anatomical_image(
    main_window: Any,
    file_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Loads a NIfTI image file (.nii, .nii.gz).
    Returns the image data array and affine matrix.

    Args:
        main_window: The instance of the main application window.
        file_path: Optional path to the file to load. If None, opens a dialog.

    Returns:
        tuple: (numpy.ndarray, numpy.ndarray, str) containing image data,
               affine matrix, and file path, or (None, None, None) on failure/cancel.
    """
    if not hasattr(main_window, "vtk_panel") or not main_window.vtk_panel.scene:
        logger.error("Scene not initialized in vtk_panel.")
        QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
        return None, None, None

    if file_path:
        input_path = file_path
    else:
        file_filter = "NIfTI Image Files (*.nii *.nii.gz);;All Files (*.*)"
        start_dir = ""
        if main_window.anatomical_image_path:
            start_dir = os.path.dirname(main_window.anatomical_image_path)
        elif main_window.original_trk_path:
            start_dir = os.path.dirname(main_window.original_trk_path)

        input_path, _ = QFileDialog.getOpenFileName(
            main_window, "Select Input Anatomical Image File", start_dir, file_filter
        )

    if not input_path:
        status_updater = getattr(
            main_window.vtk_panel,
            "update_status",
            lambda msg: logger.info(f"Status: {msg}"),
        )
        status_updater("Anatomical image load cancelled.")
        return None, None, None

    status_updater = getattr(
        main_window.vtk_panel,
        "update_status",
        lambda msg: logger.info(f"Status: {msg}"),
    )
    status_updater(f"Loading image: {os.path.basename(input_path)}...")
    QApplication.processEvents()

    try:
        # Load NIfTI
        img = nib.load(input_path)
        image_data = img.get_fdata(dtype=np.float32)  # Ensure float for VTK/FURY
        image_affine = img.affine

        # Basic validation
        if image_data.ndim < 3:
            raise ValueError(
                f"Loaded image has only {image_data.ndim} dimensions, expected 3 or more."
            )
        if image_affine.shape != (4, 4):
            raise ValueError(
                f"Loaded image affine has shape {image_affine.shape}, expected (4, 4)."
            )

        status_updater(
            f"Successfully loaded anatomical image: {os.path.basename(input_path)}"
        )
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


# ROI Image Loading Function
def load_roi_images(
    main_window: Any,
    file_paths: Optional[List[str]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Loads multiple NIfTI ROI files (.nii, .nii.gz).
    Returns a list of tuples, where each tuple is: (image data, affine matrix, file path).

    Args:
        main_window: The instance of the main application window.
        file_paths: Optional list of file paths to load. If None, opens a dialog.

    Returns:
        List[Tuple[...]]: A list containing data for all successfully loaded ROIs.
    """
    if not hasattr(main_window, "vtk_panel") or not main_window.vtk_panel.scene:
        logger.error("Error: Scene not initialized in vtk_panel.")
        QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
        return []

    file_filter = "NIfTI Image Files (*.nii *.nii.gz);;All Files (*.*)"
    start_dir = ""
    if main_window.anatomical_image_path:
        start_dir = os.path.dirname(main_window.anatomical_image_path)
    elif main_window.original_trk_path:
        start_dir = os.path.dirname(main_window.original_trk_path)

    # getOpenFileNames to allow multiple selection
    if file_paths:
        input_paths = file_paths
    else:
        input_paths, _ = QFileDialog.getOpenFileNames(
            main_window, "Select Input ROI Image File(s)", start_dir, file_filter
        )

    if not input_paths:
        status_updater = getattr(
            main_window.vtk_panel,
            "update_status",
            lambda msg: logger.info(f"Status: {msg}"),
        )
        status_updater("ROI image load cancelled.")
        return []

    status_updater = getattr(
        main_window.vtk_panel,
        "update_status",
        lambda msg: logger.info(f"Status: {msg}"),
    )

    loaded_rois = []

    # Iterate through all selected paths
    for input_path in input_paths:
        status_updater(f"Loading ROI: {os.path.basename(input_path)}...")
        QApplication.processEvents()

        try:
            img = nib.load(input_path)
            image_data = img.get_fdata(dtype=np.float32).astype(
                np.uint8
            )  # load as float32 for get_fdata, then convert to uint8 for optimization
            image_affine = img.affine

            if image_data.ndim < 3:
                logger.warning(
                    f"Skipping {os.path.basename(input_path)}: Has {image_data.ndim} dims, expected 3+."
                )
                continue
            if image_affine.shape != (4, 4):
                logger.warning(
                    f"Skipping {os.path.basename(input_path)}: Invalid affine shape."
                )
                continue

            loaded_rois.append((image_data, image_affine, input_path))
            status_updater(f"Successfully loaded ROI: {os.path.basename(input_path)}")

        except Exception as e:
            error_msg = f"Error loading {os.path.basename(input_path)}:\n{type(e).__name__}: {e}"
            logger.error(error_msg)
            QMessageBox.warning(main_window, "Load Error", error_msg)

    return loaded_rois


# Streamline File I/O Functions
def load_streamlines_file(
    main_window: Any, keep_image: bool = False, file_path: Optional[str] = None
) -> None:
    """
    Loads a streamline file using a background thread and a progress bar.
    """
    if not hasattr(main_window, "vtk_panel") or not main_window.vtk_panel.scene:
        QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
        return

    # Get File Path
    if file_path:
        input_path = file_path
    else:
        base_filter = "Streamline Files (*.trk *.tck *.trx *.vtk *.vtp)"
        all_filters = f"{base_filter};;TrackVis Files (*.trk);;TCK Files (*.tck);;TRX Files (*.trx);;VTK Files (*.vtk *.vtp);;All Files (*.*)"
        start_dir = os.path.dirname(
            main_window.original_trk_path or main_window.anatomical_image_path or ""
        )

        input_path, _ = QFileDialog.getOpenFileName(
            main_window, "Select Input Streamline File", start_dir, all_filters
        )
    if not input_path:
        return

    # Close existing bundle to clean up state
    if main_window.tractogram_data:
        main_window._close_bundle(keep_image=keep_image)

    # Setup Progress Dialog (Modal)
    progress = QProgressDialog("Initializing...", "Cancel", 0, 100, main_window)
    progress.setWindowTitle("Loading Bundle")
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setMinimumWidth(350)

    # Apply custom style to match VTK panel theme and color (#05B8CC)
    progress.setStyleSheet(
        """
        QProgressDialog {
            background-color: #2b2b2b;
            color: #dddddd;
        }
        QLabel {
            color: #dddddd;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 4px;
            background-color: #333;
            color: white;
            text-align: center;
            font-size: 12px;
            height: 25px;
        }
        QProgressBar::chunk {
            background-color: #05B8CC;
            border-radius: 3px;
        }
        QPushButton {
            background-color: #444;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px 12px;
        }
        QPushButton:hover {
            background-color: #555;
        }
    """
    )

    progress.setValue(0)
    progress.show()

    # Create and Configure Thread
    loader_thread = StreamlineLoaderThread(input_path)

    def on_progress(val, msg):
        progress.setValue(val)
        progress.setLabelText(msg)

    def on_error(msg):
        progress.cancel()
        QMessageBox.critical(main_window, "Load Error", f"Error loading file:\n{msg}")

    def on_finished(data):
        try:
            progress.setValue(100)

            # Apply Data to MainWindow
            main_window.tractogram_data = data["streamlines"]
            main_window.streamline_bboxes = data["bboxes"]

            main_window.original_trk_header = data["header"]
            main_window.original_trk_affine = data["affine"]
            main_window.original_trk_path = data["path"]
            main_window.original_file_extension = data["ext"]

            main_window.trx_file_reference = data.get("trx_obj", None)
            main_window.scalar_data_per_point = data["scalars"]
            main_window.active_scalar_name = data["active_scalar"]

            # Initialize Logic State
            total_fibers = len(main_window.tractogram_data)
            main_window.manual_visible_indices = set(range(total_fibers))
            main_window.visible_indices = set(range(total_fibers))

            # Reset Caches
            main_window.roi_states = {}
            main_window.roi_intersection_cache = {}
            main_window.roi_highlight_indices = set()

            main_window.selected_streamline_indices = set()
            main_window.unified_undo_stack = []
            main_window.unified_redo_stack = []
            main_window.current_color_mode = ColorMode.ORIENTATION

            # Check for empty file
            if not main_window.tractogram_data or len(main_window.tractogram_data) == 0:
                QMessageBox.information(
                    main_window, "Load Info", "No streamlines found in file."
                )
                main_window._close_bundle()
                return

            # Auto Skip Calculation
            should_render = True
            if hasattr(main_window, "_auto_calculate_skip_level"):
                main_window._auto_calculate_skip_level()
                should_render = False

            # Finalize UI
            status_msg = f"Loaded {len(main_window.tractogram_data)} streamlines from {os.path.basename(data['path'])}"
            _update_vtk_and_ui_after_load(main_window, status_msg, render=should_render)

        except Exception as e:
            logger.error(f"Error in on_finished: {e}", exc_info=True)
            QMessageBox.critical(
                main_window, "Load Error", f"Error finalizing load:\n{e}"
            )
            # Attempt to cleanup if possible
            if hasattr(main_window, "_close_bundle"):
                # Don't recurse if close bundle fails
                try:
                    main_window._close_bundle(keep_image=True)
                except Exception:
                    pass

    # Connect Signals
    loader_thread.progress.connect(on_progress)
    loader_thread.error.connect(on_error)
    loader_thread.finished.connect(on_finished)
    progress.canceled.connect(loader_thread.terminate)

    # Keep reference
    main_window._loader_thread = loader_thread

    # Start
    loader_thread.start()


def _validate_save_prerequisites(main_window: Any) -> bool:
    """Checks if prerequisites for saving streamlines are met."""
    if main_window.tractogram_data is None:
        logger.error("Save Error: No streamline data to save.")
        QMessageBox.warning(main_window, "Save Error", "No streamline data to save.")
        return False
    if main_window.original_trk_affine is None:
        logger.error("Save Error: Original streamline affine info missing.")
        QMessageBox.critical(
            main_window,
            "Save Error",
            "Original streamline file affine info missing (needed for saving).",
        )
        return False
    if main_window.original_trk_header is None:
        logger.warning(
            "Warning: Original streamline header info missing. Saving with minimal header."
        )
        main_window.original_trk_header = {}  # Ensure it's a dict

    if main_window.original_file_extension not in [
        ".trk",
        ".tck",
        ".trx",
        ".vtk",
        ".vtp",
    ]:
        logger.error(
            f"Save Error: Cannot determine original format ('{main_window.original_file_extension}')."
        )
        QMessageBox.critical(
            main_window,
            "Save Error",
            f"Cannot determine original format ('{main_window.original_file_extension}').",
        )
        return False
    return True


def _get_save_path_and_extension(
    main_window: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Gets the output path and validated extension from the user."""
    initial_dir = (
        os.path.dirname(main_window.original_trk_path)
        if main_window.original_trk_path
        else ""
    )
    base_name = (
        f"{os.path.splitext(os.path.basename(main_window.original_trk_path))[0]}_modified"
        if main_window.original_trk_path
        else "modified_bundle"
    )

    # Map extensions to filters
    ext_to_filter = {
        ".trk": "TrackVis TRK Files (*.trk)",
        ".tck": "MRtrix TCK Files (*.tck)",
        ".trx": "TRX Files (*.trx)",
        ".vtk": "Legacy VTK Files (*.vtk)",
        ".vtp": "XML VTK Files (*.vtp)",
    }

    # Create comprehensive filter string
    filters_list = list(ext_to_filter.values()) + ["All Files (*.*)"]
    all_filters = ";;".join(filters_list)

    # Determine initial filter based on original extension
    initial_filter = ext_to_filter.get(
        main_window.original_file_extension, "All Files (*.*)"
    )

    # Default output filename with original extension
    suggested_path = os.path.join(
        initial_dir, base_name + main_window.original_file_extension
    )

    output_path, selected_filter = QFileDialog.getSaveFileName(
        main_window,
        "Save Modified Streamlines",
        suggested_path,
        all_filters,
        initialFilter=initial_filter,
    )

    if not output_path:
        return None, None

    _, output_ext = os.path.splitext(output_path)
    output_ext = output_ext.lower()

    # Auto-detect extension from filter if missing
    if not output_ext:
        # Reverse map filter to extension
        filter_to_ext = {v: k for k, v in ext_to_filter.items()}
        inferred_ext = filter_to_ext.get(selected_filter)

        if inferred_ext:
            output_path += inferred_ext
            output_ext = inferred_ext
            logger.info(
                f"Save Info: Appended extension '{output_ext}' based on filter."
            )
        else:
            # Fallback to original if "All Files" was selected and no extension typed
            output_path += main_window.original_file_extension
            output_ext = main_window.original_file_extension
            logger.info(f"Save Info: Appended default extension '{output_ext}'.")

    # Validate supported extension
    supported_extensions = {".trk", ".tck", ".trx", ".vtk", ".vtp"}
    if output_ext not in supported_extensions:
        QMessageBox.warning(
            main_window,
            "Save Warning",
            f"Unsupported file extension '{output_ext}'.\nSaving as {main_window.original_file_extension} instead.",
        )
        output_path = (
            os.path.splitext(output_path)[0] + main_window.original_file_extension
        )
        output_ext = main_window.original_file_extension

    return output_path, output_ext


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
                data_per_point_to_save[key] = list(scalars_for_key_gen)
        except Exception as e:
            logger.warning(
                f"Warning: Could not filter scalar data for saving. Saving without scalars. Error: {e}"
            )
            data_per_point_to_save = {}

    # Use Nibabel's Tractogram object as a generic container
    new_tractogram = nib.streamlines.Tractogram(
        list(streamlines_to_save_gen),
        data_per_point=data_per_point_to_save if data_per_point_to_save else None,
        affine_to_rasmm=affine_matrix,
    )
    return new_tractogram


def _prepare_trk_header(
    base_header: Dict[str, Any],
    nb_streamlines: int,
    anatomical_img_affine: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Prepares and validates the header dictionary for TRK saving.
    If voxel_order is missing in base_header, attempts to derive it from
    anatomical_img_affine, otherwise defaults to 'RAS'.
    """
    header = base_header.copy()
    logger.info("Preparing TRK header for saving...")
    if anatomical_img_affine is not None:
        logger.debug(f"Anatomical affine provided. Type: {type(anatomical_img_affine)}")

    # Voxel Order Logic
    raw_voxel_order_from_trk = header.get("voxel_order")
    processed_voxel_order_from_trk = None

    if isinstance(raw_voxel_order_from_trk, bytes):
        try:
            processed_voxel_order_from_trk = raw_voxel_order_from_trk.decode(
                "utf-8", errors="strict"
            )
            logger.debug(f"Decoded 'voxel_order': '{processed_voxel_order_from_trk}'")
        except UnicodeDecodeError:
            logger.warning(
                f"'voxel_order' field in TRK header (bytes: {raw_voxel_order_from_trk}) could not be decoded."
            )
    elif isinstance(raw_voxel_order_from_trk, str):
        processed_voxel_order_from_trk = raw_voxel_order_from_trk

    is_valid_trk_voxel_order = (
        isinstance(processed_voxel_order_from_trk, str)
        and len(processed_voxel_order_from_trk) == 3
    )

    if is_valid_trk_voxel_order:
        header["voxel_order"] = processed_voxel_order_from_trk.upper()
        logger.info(
            f"      - Info: Using existing 'voxel_order' from TRK header: {header['voxel_order']}."
        )
    else:
        if raw_voxel_order_from_trk is not None:
            logger.warning(
                f"      - Warning: 'voxel_order' from TRK header ('{raw_voxel_order_from_trk}') is invalid or in an unexpected format."
            )
        else:
            logger.warning(f"      - Info: 'voxel_order' missing in TRK header.")

        derived_from_anat = False
        if (
            anatomical_img_affine is not None
            and isinstance(anatomical_img_affine, np.ndarray)
            and anatomical_img_affine.shape == (4, 4)
        ):
            try:
                axcodes = nib.aff2axcodes(anatomical_img_affine)
                derived_vo_str = "".join(axcodes).upper()
                if len(derived_vo_str) == 3:
                    header["voxel_order"] = derived_vo_str
                    derived_from_anat = True
                    logger.info(
                        f"      - Info: Derived 'voxel_order' from loaded anatomical image: {header['voxel_order']}."
                    )
                else:
                    logger.warning(
                        f"      - Warning: Could not derive a valid 3-character 'voxel_order' from anatomical image affine (got: '{derived_vo_str}')."
                    )
            except Exception as e:
                logger.warning(
                    f"      - Warning: Error deriving 'voxel_order' from anatomical image: {e}"
                )

        if not derived_from_anat:
            header["voxel_order"] = "RAS"
            logger.info(
                f"      - Info: Defaulting 'voxel_order' to 'RAS' (Standard fallback)."
            )

    # If converting from TCK/VTK -> TRK, we might lack voxel_to_rasmm
    # If an anatomical image is available, use its affine
    if "voxel_to_rasmm" not in header or header["voxel_to_rasmm"] is None:
        if anatomical_img_affine is not None:
            header["voxel_to_rasmm"] = anatomical_img_affine
            logger.info(
                "      - Info: Populated 'voxel_to_rasmm' from anatomical image."
            )
        else:
            header["voxel_to_rasmm"] = np.eye(4)
            logger.warning(
                "      - Warning: 'voxel_to_rasmm' missing and no anatomical image. Using Identity."
            )

    # Process other specific TRK header fields
    keys_to_process = {
        "voxel_sizes": {"type": float, "length": 3, "default": (1.0, 1.0, 1.0)},
        "dimensions": {
            "type": int,
            "length": 3,
            "default": (1, 1, 1),
        },  # Small valid default
        "voxel_to_rasmm": {
            "type": float,
            "shape": (4, 4),
            "default": np.identity(4, dtype=np.float32),
        },
    }

    for key, K_props in keys_to_process.items():
        original_value = header.get(key)
        processed_value = original_value
        expected_item_type = K_props["type"]
        is_matrix = "shape" in K_props

        # Decode if bytes
        if isinstance(processed_value, bytes):
            try:
                processed_value = processed_value.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                logger.warning(
                    f"      - Warning: Could not decode bytes for '{key}'. Original value: {original_value}"
                )
                header[key] = K_props["default"]
                logger.warning(f"      - Info: Set '{key}' to default: {header[key]}")
                continue

        # Parse if string, or use if already suitable type
        if isinstance(processed_value, str):
            parsed_val = parse_numeric_tuple_from_string(
                processed_value,
                expected_item_type,
                K_props.get("length") or K_props.get("shape"),
            )
            # Check if parse_numeric_tuple_from_string returned the original string (failure)
            if not (isinstance(parsed_val, str) and parsed_val == processed_value):
                processed_value = parsed_val
            else:
                logger.info(
                    f"      - Info: Could not parse string '{processed_value}' for '{key}'."
                )

        # Validate and set
        valid_structure = False
        final_value = None

        try:
            if is_matrix:  # voxel_to_rasmm
                if (
                    isinstance(processed_value, np.ndarray)
                    and processed_value.shape == K_props["shape"]
                ):
                    final_value = processed_value.astype(expected_item_type)
                    valid_structure = True
            else:  # voxel_sizes, dimensions (tuples)
                if (
                    isinstance(processed_value, tuple)
                    and len(processed_value) == K_props["length"]
                ):
                    final_value = tuple(expected_item_type(x) for x in processed_value)
                    valid_structure = True
                elif (
                    isinstance(processed_value, np.ndarray)
                    and processed_value.ndim == 1
                    and len(processed_value) == K_props["length"]
                ):
                    final_value = tuple(processed_value.astype(expected_item_type))
                    valid_structure = True
        except (ValueError, TypeError) as e:  # Catch errors from type conversion
            logger.warning(
                f"      - Warning: Type conversion error for '{key}' (value: '{processed_value}'): {e}"
            )
            valid_structure = False

        if valid_structure:
            header[key] = final_value
        else:
            # If missing, try to fill from anatomical image if available
            if anatomical_img_affine is not None:
                if key == "voxel_sizes":
                    header[key] = tuple(nib.affines.voxel_sizes(anatomical_img_affine))
                    logger.info(f"      - Info: Derived '{key}' from anatomical image.")
                    continue
                ## TODO - handle dimensions

            logger.warning(
                f"      - Warning: '{key}' ('{original_value}') was invalid, missing, or failed processing. Defaulted to {K_props['default']}."
            )
            header[key] = K_props["default"]

    header["nb_streamlines"] = nb_streamlines
    header["voxel_order"] = header["voxel_order"].upper()

    return header


def _prepare_tck_header(
    base_header: Optional[Dict[str, Any]], nb_streamlines: int
) -> Dict[str, Any]:
    """Prepares the header dictionary for TCK saving."""
    # Start with a clean header to avoid TRK-specific fields (like 'magic') polluting TCK
    header = {}
    header["count"] = str(nb_streamlines)

    if base_header:
        # TCK headers are flexible key-value pairs.
        # Some TRK metadata are preserved as strings.
        keys_to_preserve = ["voxel_order", "dimensions", "voxel_sizes"]

        for key in keys_to_preserve:
            if key in base_header:
                val = base_header[key]
                # Convert complex types to string representation
                if isinstance(val, (tuple, list, np.ndarray)):
                    val_str = " ".join(map(str, np.array(val).flatten()))
                    header[key] = val_str
                elif isinstance(val, bytes):
                    try:
                        header[key] = val.decode("utf-8", errors="replace")
                    except Exception:
                        header[key] = str(val)
                else:
                    header[key] = str(val)

    return header


def _prepare_trx_header(
    base_header: Optional[Dict[str, Any]],
    nb_streamlines: int,
    anatomical_img_affine: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Prepares the header dictionary for TRX saving.
    Ensures essential reference fields (affine, dimensions) are present.
    """
    header = base_header.copy() if base_header is not None else {}
    header["nb_streamlines"] = nb_streamlines

    # Clean up TCK specific
    header.pop("count", None)

    # Ensure voxel_to_rasmm exists
    if "voxel_to_rasmm" not in header or header["voxel_to_rasmm"] is None:
        if anatomical_img_affine is not None:
            header["voxel_to_rasmm"] = anatomical_img_affine
        else:
            header["voxel_to_rasmm"] = np.eye(4)

    # Validate and fix dimensions - must be tuple of 3 integers
    dims = header.get("dimensions")
    valid_dims = None

    if dims is not None:
        # Handle string representation like "(182, 218, 182)"
        if isinstance(dims, str):
            parsed = parse_numeric_tuple_from_string(dims, int, 3)
            if isinstance(parsed, tuple) and len(parsed) == 3:
                valid_dims = parsed
        elif isinstance(dims, (tuple, list)) and len(dims) == 3:
            try:
                valid_dims = tuple(int(x) for x in dims)
            except (ValueError, TypeError):
                pass
        elif isinstance(dims, np.ndarray) and dims.size == 3:
            try:
                valid_dims = tuple(int(x) for x in dims.flatten())
            except (ValueError, TypeError):
                pass

    if valid_dims is None:
        header["dimensions"] = (1, 1, 1)
        logger.warning(
            "TRX header: 'dimensions' invalid or missing, defaulting to (1,1,1)"
        )
    else:
        header["dimensions"] = valid_dims

    # Validate and fix voxel_sizes - must be tuple of 3 floats
    vox_sizes = header.get("voxel_sizes")
    valid_vox_sizes = None

    if vox_sizes is not None:
        if isinstance(vox_sizes, str):
            parsed = parse_numeric_tuple_from_string(vox_sizes, float, 3)
            if isinstance(parsed, tuple) and len(parsed) == 3:
                valid_vox_sizes = parsed
        elif isinstance(vox_sizes, (tuple, list)) and len(vox_sizes) == 3:
            try:
                valid_vox_sizes = tuple(float(x) for x in vox_sizes)
            except (ValueError, TypeError):
                pass
        elif isinstance(vox_sizes, np.ndarray) and vox_sizes.size == 3:
            try:
                valid_vox_sizes = tuple(float(x) for x in vox_sizes.flatten())
            except (ValueError, TypeError):
                pass

    if valid_vox_sizes is None:
        if anatomical_img_affine is not None:
            try:
                header["voxel_sizes"] = tuple(
                    nib.affines.voxel_sizes(anatomical_img_affine)
                )
            except Exception:
                header["voxel_sizes"] = (1.0, 1.0, 1.0)
        else:
            header["voxel_sizes"] = (1.0, 1.0, 1.0)
    else:
        header["voxel_sizes"] = valid_vox_sizes

    # Ensure voxel_order exists (TRX might use it or ignore it, but good to have)
    if "voxel_order" not in header:
        header["voxel_order"] = "RAS"

    return header


def _create_vtk_polydata_from_tractogram(
    tractogram: nib.streamlines.Tractogram,
) -> vtk.vtkPolyData:
    """
    Converts a Nibabel Tractogram to vtkPolyData (lines).
    Assumes streamlines are in RASMM (world space).
    """
    poly_data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Flatten points
    if hasattr(tractogram.streamlines, "_data"):
        # Fast path if ArraySequence
        all_points = tractogram.streamlines._data
        offsets = tractogram.streamlines._offsets
        lengths = tractogram.streamlines._lengths
    else:
        # Slow path
        all_points = np.concatenate(tractogram.streamlines)
        lengths = [len(s) for s in tractogram.streamlines]
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))

    # Set Points
    vtk_points_array = numpy_support.numpy_to_vtk(all_points, deep=True)
    points.SetData(vtk_points_array)
    poly_data.SetPoints(points)

    # Set Lines (Connectivity)
    # VTK CellArray needs [n_pts, id0, id1..., n_pts, id0...]; in Numpy for speed
    n_streamlines = len(lengths)
    total_points = len(all_points)

    # Size of connectivity array = total_points + n_streamlines (headers)
    connectivity = np.empty(total_points + n_streamlines, dtype=np.int64)

    current_conn_idx = 0
    current_pt_idx = 0

    # Vectorized approach to build connectivity is complex, using mixed approach:
    for i in range(n_streamlines):
        l = lengths[i]
        connectivity[current_conn_idx] = l
        # Create range for points
        connectivity[current_conn_idx + 1 : current_conn_idx + 1 + l] = np.arange(
            current_pt_idx, current_pt_idx + l
        )

        current_conn_idx += l + 1
        current_pt_idx += l

    # Safe legacy approach
    cell_array = vtk.vtkCellArray()
    # Handle int64 vs int32 for VTK ID types
    if vtk.vtkIdTypeArray().GetDataTypeSize() == 4:
        connectivity = connectivity.astype(np.int32)

    vtk_ids = numpy_support.numpy_to_vtk(
        connectivity, deep=True, array_type=vtk.vtkIdTypeArray().GetDataType()
    )
    cell_array.SetCells(n_streamlines, vtk_ids)
    poly_data.SetLines(cell_array)

    # Add Scalars (Point Data)
    if tractogram.data_per_point:
        for key, seq in tractogram.data_per_point.items():
            # Flatten
            if hasattr(seq, "_data"):
                flat_scalar = seq._data
            else:
                flat_scalar = np.concatenate(seq)

            vtk_arr = numpy_support.numpy_to_vtk(flat_scalar, deep=True)
            vtk_arr.SetName(key)
            poly_data.GetPointData().AddArray(vtk_arr)

    return poly_data


def _save_tractogram_file(
    tractogram: nib.streamlines.Tractogram,
    header: Dict[str, Any],
    output_path: str,
    file_ext: str,
) -> str:
    """
    Saves the tractogram using nibabel or trx-python based on the extension.
    """
    if file_ext == ".trk":
        trk_file = nib.streamlines.TrkFile(tractogram, header=header)
        nib.streamlines.save(trk_file, output_path)
        logger.info("File saved successfully (TRK)")
        return f"File saved successfully (TRK): {os.path.basename(output_path)}"

    elif file_ext == ".tck":
        tck_file = nib.streamlines.TckFile(tractogram, header=header)
        nib.streamlines.save(tck_file, output_path)
        logger.info("File saved successfully (TCK)")
        return f"File saved successfully (TCK): {os.path.basename(output_path)}"

    elif file_ext == ".trx":
        # Create a valid reference object (Nifti1Image) for TRX
        # TRX requires a reference that defines the space (affine, dimensions)
        affine = header.get("voxel_to_rasmm", np.eye(4))
        dimensions = header.get("dimensions", (1, 1, 1))
        voxel_sizes = header.get("voxel_sizes", (1.0, 1.0, 1.0))

        # Create a dummy Nifti header
        nifti_header = nib.Nifti1Header()
        nifti_header.set_data_shape(dimensions)
        nifti_header.set_zooms(voxel_sizes)
        nifti_header.set_qform(affine)
        nifti_header.set_sform(affine)

        # Create dummy image (empty data, just for reference)
        dummy_data = np.empty(dimensions, dtype=np.int8)
        reference_img = nib.Nifti1Image(dummy_data, affine, header=nifti_header)

        # Create TRX object using the valid reference
        trx_obj_to_save = tbx.TrxFile.from_lazy_tractogram(tractogram, reference_img)

        # Save the newly created object using tbx.save()
        tbx.save(trx_obj_to_save, output_path)
        logger.info("File saved successfully (TRX)")
        return f"File saved successfully (TRX): {os.path.basename(output_path)}"

    elif file_ext in [
        ".vtk",
        ".vtp",
    ]:  # VTK format doesn't have a dedicated affine field.

        # Apply affine to coordinates if not identity
        if not np.allclose(tractogram.affine_to_rasmm, np.eye(4)):
            # Nibabel apply_affine is convenient here
            streamlines_world = list(
                nib.streamlines.transform_streamlines(
                    tractogram.streamlines, tractogram.affine_to_rasmm
                )
            )
            # Create a temporary tractogram in RASMM (Identity affine) for saving
            temp_tractogram = nib.streamlines.Tractogram(
                streamlines_world,
                data_per_point=tractogram.data_per_point,
                affine_to_rasmm=np.eye(4),
            )
            poly_data = _create_vtk_polydata_from_tractogram(temp_tractogram)
        else:
            poly_data = _create_vtk_polydata_from_tractogram(tractogram)

        # Write
        if file_ext == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetDataModeToBinary()
        else:
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileTypeToBinary()

        writer.SetFileName(output_path)
        writer.SetInputData(poly_data)
        writer.Write()

        logger.info(f"File saved successfully ({file_ext.upper()})")
        return f"File saved successfully ({file_ext.upper()}): {os.path.basename(output_path)}"

    else:
        raise ValueError(f"Unsupported save extension: {file_ext}")


# Main Save Function
def save_streamlines_file(main_window: Any) -> None:
    """
    Saves the current streamlines to a trk, tck, or trx file.
    """
    status_updater = getattr(
        main_window.vtk_panel,
        "update_status",
        lambda msg: logger.info(f"Status: {msg}"),
    )

    # Pre-checks
    if not _validate_save_prerequisites(main_window):
        return

    # Get Output Path
    output_path, output_ext = _get_save_path_and_extension(main_window)
    if not output_path:
        status_updater("Save cancelled.")
        return

    # Prepare Data
    status_updater(
        f"Saving {len(main_window.visible_indices)} streamlines to: {os.path.basename(output_path)}..."
    )
    QApplication.processEvents()  # UI update

    try:
        tractogram = _prepare_tractogram_and_affine(main_window)

        header_to_save: Dict[str, Any] = {}
        if output_ext == ".trk":
            header_to_save = _prepare_trk_header(
                main_window.original_trk_header,
                len(tractogram.streamlines),
                anatomical_img_affine=main_window.anatomical_image_affine,
            )
        elif output_ext == ".tck":
            header_to_save = _prepare_tck_header(
                main_window.original_trk_header, len(tractogram.streamlines)
            )

        elif output_ext == ".trx":
            header_to_save = _prepare_trx_header(
                main_window.original_trk_header,
                len(tractogram.streamlines),
                anatomical_img_affine=main_window.anatomical_image_affine,
            )

        # Save File
        logger.info(f"Saving to {output_ext}...")
        if output_ext == ".trk":
            logger.debug(f"TRK Header Keys: {list(header_to_save.keys())}")
            if "voxel_to_rasmm" in header_to_save:
                logger.debug(
                    f"voxel_to_rasmm type: {type(header_to_save['voxel_to_rasmm'])}"
                )
                logger.debug(
                    f"voxel_to_rasmm shape: {header_to_save['voxel_to_rasmm'].shape}"
                )
            logger.debug(f"Tractogram affine type: {type(tractogram.affine_to_rasmm)}")

        success_msg = _save_tractogram_file(
            tractogram, header_to_save, output_path, output_ext
        )
        status_updater(success_msg)

    except Exception as e:
        logger.error(f"Error during file saving:\nType: {type(e).__name__}\nError: {e}")
        error_msg = (
            f"Error saving file:\n{type(e).__name__}: {e}\n\nCheck console for details."
        )
        QMessageBox.critical(main_window, "Save Error", error_msg)
        status_updater(f"Error saving file: {os.path.basename(output_path)}")
