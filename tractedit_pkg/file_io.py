# -*- coding: utf-8 -*-

"""
Functions for loading and saving streamline files (trk, tck, trx)
and loading anatomical image files (NIfTI).
"""

# ============================================================================
# Imports
# ============================================================================

import os
import logging
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
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

logger = logging.getLogger(__name__)


##TODO - This will be updated. Fury doesn't support memmap images
# so to handle anatomical images in 3D main view we'll need to replace entirely Fury with VTK
# We tried an hybrid approach by maintaining 2D high-res for 2D panels and downsampled images for the 3D view but it is messy
# For now we avoid crashes with high-res images by downsampling both 3D and 2D views, in future we'll replace Fury with VTK

# Auto-downsampling constants
# 512³ (~134M voxels)
MAX_VOXELS = 512**3  # ~134M voxels - target max for display

# LRU cache size for memory-mapped slices (number of slices to cache)
MMAP_SLICE_CACHE_SIZE = 64

# Medoid calculation constants
# Binary search in resampling kernels makes exact medoid feasible for larger
# bundles.  SAMPLING_THRESHOLD controls when to switch from exact (N×N) to
# approximate (N×k) distance computation.  SAMPLE_SIZE controls the k value
# for the approximate path — larger k improves accuracy at the cost of time.
MEDOID_SAMPLING_THRESHOLD = 20000  # Exact medoid up to this count; above uses sampling
MEDOID_SAMPLE_SIZE = 2000  # Samples for approximate medoid (higher = more accurate)

# Batch size for distance computation — controls cancellation granularity.
# Each batch is dispatched in parallel, then cancellation is checked.
_MEDOID_DISTANCE_BATCH = 5000

# Centroid safety limit — centroid is O(N) but runs synchronously on the main
# thread.  The resampled array is (N, 100, 3) float64 ≈ 2.3 KB per streamline;
# at 500k that's ~1.1 GB.
CENTROID_MAX_STREAMLINES = 500_000


# ============================================================================
# Memory-Mapped Image Wrapper
# ============================================================================


class MemoryMappedImage:
    """
    Memory-mapped NIfTI image wrapper for efficient on-demand slice extraction.

    Uses nibabel's memory-mapping to avoid loading the entire volume into RAM.
    Provides cached full-resolution slices for 2D panel display.
    Data is kept in canonical RAS+ orientation (from as_closest_canonical).
    """

    def __init__(self, img: nib.Nifti1Image):
        """
        Initialize memory-mapped image wrapper.

        Args:
            img: NiBabel image object (will be accessed via memory-mapping).
        """
        self._img = img
        self._affine = img.affine.copy()
        self._shape = img.shape[:3]

        # Create cached slice getter
        self._get_slice_cached = self._create_cached_slice_getter()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the 3D shape of the image."""
        return self._shape

    @property
    def affine(self) -> np.ndarray:
        """Return the affine matrix."""
        return self._affine

    def _create_cached_slice_getter(self):
        """Create an LRU-cached slice getter function."""
        from functools import lru_cache

        @lru_cache(maxsize=MMAP_SLICE_CACHE_SIZE)
        def get_slice_cached(axis: str, index: int) -> np.ndarray:
            """
            Extract a single slice from the memory-mapped image.

            Args:
                axis: 'x' (sagittal), 'y' (coronal), or 'z' (axial).
                index: Slice index along the specified axis.

            Returns:
                2D numpy array (float32) of the slice data.
            """
            # Use dataobj for memory-mapped access (no full load)
            dataobj = self._img.dataobj

            if axis == "x":
                # Sagittal slice
                slice_data = np.asarray(dataobj[index, :, :])
            elif axis == "y":
                # Coronal slice
                slice_data = np.asarray(dataobj[:, index, :])
            elif axis == "z":
                # Axial slice
                slice_data = np.asarray(dataobj[:, :, index])
            else:
                raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'.")

            # Ensure contiguous float32 for VTK
            return np.ascontiguousarray(slice_data, dtype=np.float32)

        return get_slice_cached

    def get_slice(self, axis: str, index: int) -> np.ndarray:
        """
        Get a cached full-resolution slice.

        Args:
            axis: 'x' (sagittal), 'y' (coronal), or 'z' (axial).
            index: Slice index along the specified axis.

        Returns:
            2D numpy array (float32) of the slice data.
        """
        # Clamp index to valid range
        axis_map = {"x": 0, "y": 1, "z": 2}
        max_idx = self._shape[axis_map[axis]] - 1
        index = max(0, min(index, max_idx))
        return self._get_slice_cached(axis, index)

    def get_value_range(self) -> Tuple[float, float]:
        """
        Get the min/max value range by sampling the image.

        Uses a sampling approach to avoid loading the full volume.
        """
        # Sample slices at 25%, 50%, 75% through each axis
        samples = []
        for axis in ["x", "y", "z"]:
            axis_map = {"x": 0, "y": 1, "z": 2}
            size = self._shape[axis_map[axis]]
            for pct in [0.25, 0.5, 0.75]:
                idx = int(size * pct)
                slice_data = self.get_slice(axis, idx)
                samples.append(slice_data)

        all_samples = np.concatenate([s.ravel() for s in samples])
        finite_samples = all_samples[np.isfinite(all_samples)]

        if finite_samples.size > 0:
            return float(np.min(finite_samples)), float(np.max(finite_samples))
        return 0.0, 1.0

    def clear_cache(self):
        """Clear the slice cache."""
        self._get_slice_cached.cache_clear()


def _align_if_oblique(
    img: nib.Nifti1Image, input_path: str, status_updater: callable
) -> nib.Nifti1Image:
    """
    Checks for and corrects oblique image affines (shears/rotations from gantry tilt).
    Ensures orthogonal layout so FURY renders slices aligned to world X,Y,Z axes.
    """
    from nibabel.funcs import as_closest_canonical
    from nibabel.processing import resample_to_output

    img = as_closest_canonical(img)

    if hasattr(nib.affines, "obliquity"):
        obliquity = nib.affines.obliquity(img.affine)
        is_oblique = np.any(np.abs(obliquity) > 1e-4)
    else:
        rot_part = img.affine[:3, :3]
        is_oblique = not np.allclose(rot_part, np.diag(np.diag(rot_part)), atol=1e-4)

    if is_oblique:
        if status_updater:
            status_updater("Aligning oblique image grid...")
        logger.info(f"Image {input_path} is oblique. Resampling to orthogonal grid.")
        img = resample_to_output(img)

    return img


def _maybe_downsample_image(
    img: nib.Nifti1Image,
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Checks if an image is too large and downsamples if needed.

    Data is kept in its canonical RAS+ orientation (from as_closest_canonical).
    Radiological display convention is handled at the rendering level.

    Args:
        img: NiBabel image object.
        progress_callback: Optional callback(percent, message) for progress updates.

    Returns:
        Tuple of (image_data, image_affine, was_downsampled).
    """
    shape = np.array(img.shape[:3])
    total_voxels = shape[0] * shape[1] * shape[2]

    if total_voxels <= MAX_VOXELS:
        # Image is small enough, no downsampling needed
        if progress_callback:
            progress_callback(50, "Loading image data...")
        return img.get_fdata(dtype=np.float32), img.affine.copy(), False

    # Image is too large - calculate striding step
    if progress_callback:
        progress_callback(
            30, f"Image too large ({shape[0]}×{shape[1]}×{shape[2]}), downsampling..."
        )

    # Calculate step to reach ~256³ voxels
    target_size = int(MAX_VOXELS ** (1 / 3))  # ~256
    step = max(1, int(np.ceil(max(shape) / target_size)))

    if progress_callback:
        progress_callback(40, "Loading original data...")

    # Load the original data (kept in canonical RAS+ orientation)
    original_data = img.get_fdata(dtype=np.float32)
    original_affine = img.affine.copy()

    if progress_callback:
        progress_callback(60, f"Downsampling with step={step}...")

    # Downsample using striding first (fast operation)
    # Without this, FURY/VTK will be extremely slow on the non-contiguous strided view
    resampled_data = np.ascontiguousarray(original_data[::step, ::step, ::step])

    if progress_callback:
        progress_callback(75, "Applying anti-aliasing filter...")

    # Anti-aliasing: Apply mild Gaussian blur AFTER downsampling ##TODO - to add in 'Settings' menu
    resampled_data = gaussian_filter(resampled_data, sigma=0.8, mode="nearest")

    # Adjust affine: multiply the voxel-step columns by the step size
    new_affine = original_affine.copy()
    new_affine[:3, :3] *= step

    if progress_callback:
        progress_callback(90, "Done resampling...")

    new_shape = resampled_data.shape[:3]
    logger.info(
        f"Downsampled image from {tuple(shape)} to {new_shape} " f"(step: {step})"
    )

    return resampled_data, new_affine, True


# ============================================================================
# Background Loader Threads
# ============================================================================


class StreamlineLoaderThread(QThread):
    """
    Background thread to load streamline files without freezing the GUI.

    Supports cooperative cancellation via :meth:`cancel`.  The loading
    loop checks the ``_cancelled`` flag at each major stage and exits
    early when requested, avoiding the use of ``QThread.terminate()``.
    """

    progress = pyqtSignal(int, str)  # Signal to update progress bar (percent, message)
    finished = pyqtSignal(dict)  # Signal when loading is done
    error = pyqtSignal(str)  # Signal if an error occurs

    def __init__(self, input_path: str) -> None:
        super().__init__()
        self.input_path = input_path
        self._cancelled: bool = False

    def cancel(self) -> None:
        """Request cooperative cancellation of the loading operation."""
        self._cancelled = True

    def run(self) -> None:
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
                if self._cancelled:
                    return

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
                if self._cancelled:
                    return
                self.progress.emit(70, "Finalizing...")
                # Attempt fast AOT calc
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

                # Select reader
                if ext == ".vtp":
                    reader = vtk.vtkXMLPolyDataReader()
                else:
                    reader = vtk.vtkPolyDataReader()

                reader.SetFileName(self.input_path)

                # Forward fine-grained I/O progress from VTK to the Qt dialog
                def _on_reader_progress(caller, event):
                    pct = 10 + int(caller.GetProgress() * 18)  # 0–1 → 10–28 %
                    self.progress.emit(pct, "Reading VTK file...")

                observer_tag = reader.AddObserver(
                    "ProgressEvent", _on_reader_progress
                )
                reader.Update()
                reader.RemoveObserver(observer_tag)
                if self._cancelled:
                    return
                poly_data = reader.GetOutput()

                # Extract point coordinates as a contiguous float32 array
                self.progress.emit(30, "Parsing geometry...")
                vtk_points = poly_data.GetPoints()
                if vtk_points is not None:
                    points_data = numpy_support.vtk_to_numpy(
                        vtk_points.GetData()
                    ).astype(np.float32, copy=False)
                else:
                    points_data = np.empty((0, 3), dtype=np.float32)

                vtk_lines = poly_data.GetLines()

                if poly_data.GetNumberOfLines() == 0:
                    as_streamlines = nib.streamlines.ArraySequence()
                else:
                    offsets_arr = numpy_support.vtk_to_numpy(
                        vtk_lines.GetOffsetsArray()
                    ).astype(np.intp)
                    connectivity_arr = numpy_support.vtk_to_numpy(
                        vtk_lines.GetConnectivityArray()
                    )

                    # Single vectorised lookup
                    flat_coords = points_data[connectivity_arr].astype(
                        np.float32, copy=False
                    )

                    # Build ArraySequence directly from pre-computed flat arrays.
                    lengths_arr = np.diff(offsets_arr).astype(np.intp)
                    as_streamlines = nib.streamlines.ArraySequence()
                    as_streamlines._data = flat_coords
                    as_streamlines._offsets = offsets_arr[:-1].astype(np.intp)
                    as_streamlines._lengths = lengths_arr

                # Determine Affine
                # VTK files are already in world coordinates; use identity.
                aff = np.identity(4)
                results["streamlines"] = as_streamlines
                results["header"] = {}  # VTK has no standard header
                results["affine"] = aff

                # Handle Scalars (Point Data)
                self.progress.emit(60, "Reading scalars...")
                scalars = {}
                point_data_vtk = poly_data.GetPointData()
                n_arrays = point_data_vtk.GetNumberOfArrays()

                for i in range(n_arrays):
                    arr_name = point_data_vtk.GetArrayName(i)
                    vtk_arr = point_data_vtk.GetArray(i)
                    np_arr = numpy_support.vtk_to_numpy(vtk_arr)

                    # Vectorised split using the same offsets as the coordinate array
                    split_at = as_streamlines._offsets[1:]
                    scalar_parts = np.split(np_arr, split_at)

                    key = arr_name if arr_name else f"Scalar_{i}"
                    scalars[key] = nib.streamlines.ArraySequence(scalar_parts)

                results["scalars"] = scalars
                results["active_scalar"] = list(scalars.keys())[0] if scalars else None

                # Geometry (BBox)
                if self._cancelled:
                    return
                self.progress.emit(80, "Finalizing...")
                try:
                    results["bboxes"] = _compute_bboxes_numba(
                        as_streamlines._data,
                        as_streamlines._offsets,
                        as_streamlines._lengths,
                    )
                except (ImportError, TypeError, ValueError):
                    logger.debug("AOT bbox computation failed, using fallback.")
                    bboxes = [
                        (
                            [np.min(sl, axis=0), np.max(sl, axis=0)]
                            if len(sl) > 0
                            else [np.zeros(3), np.zeros(3)]
                        )
                        for sl in as_streamlines
                    ]
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

                if self._cancelled:
                    return

                # Normalise coordinate dtype to float32 (matches TRK/TCK behaviour).
                streamlines = trx_obj.streamlines
                if (
                    hasattr(streamlines, "_data")
                    and streamlines._data.dtype != np.float32
                ):
                    streamlines._data = streamlines._data.astype(
                        np.float32, copy=False
                    )

                # Geometry - Bounding box calculation
                # TRX streamlines are ArraySequence with _data, _offsets, _lengths
                if self._cancelled:
                    return
                self.progress.emit(30, "Computing bounding boxes...")
                streamlines = trx_obj.streamlines
                n_streamlines = len(streamlines)

                # Phase 1 optimisation: try loading cached bboxes from
                # TRX data_per_streamline before recomputing from scratch.
                cached_bboxes = _try_load_cached_bboxes(
                    trx_obj, n_streamlines
                )

                if cached_bboxes is not None:
                    results["bboxes"] = cached_bboxes
                elif (
                    hasattr(streamlines, "_data")
                    and hasattr(streamlines, "_offsets")
                    and hasattr(streamlines, "_lengths")
                ):
                    # Fast path: AOT-optimized computation
                    results["bboxes"] = _compute_bboxes_numba(
                        streamlines._data,
                        streamlines._offsets,
                        streamlines._lengths,
                    )
                else:
                    # Fallback for unexpected data structures
                    bboxes = []
                    for sl in streamlines:
                        if len(sl) > 0:
                            bboxes.append(
                                [np.min(sl, axis=0), np.max(sl, axis=0)]
                            )
                        else:
                            bboxes.append([np.zeros(3), np.zeros(3)])
                    results["bboxes"] = np.array(
                        bboxes, dtype=np.float32
                    )

            else:
                self.error.emit(f"Unsupported file format: {ext}")
                return

            if self._cancelled:
                return

            # Emit 99 % (not 100 %) so that QProgressDialog's autoClose does
            # not fire here.  The dialog will be closed by on_finished() only
            # after the actor build and first VTK render have completed.
            self.progress.emit(99, "Building visualization...")
            self.finished.emit(results)

        except (OSError, ValueError, TypeError, IndexError, MemoryError) as e:
            self.error.emit(str(e))


class MedoidCalculationThread(QThread):
    """
    Background thread to calculate medoid without freezing the GUI.
    Allows for responsive cancellation and progress updates.
    """

    progress = pyqtSignal(int, str)  # Signal to update progress (percent, message)
    result_ready = pyqtSignal(int)  # Signal when done (medoid index, -1 if cancelled)
    error = pyqtSignal(str)  # Signal if an error occurs

    def __init__(self, streamlines: List[np.ndarray], nb_points: int = 100):
        super().__init__()
        self.streamlines = streamlines
        self.nb_points = nb_points
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the computation."""
        self._cancelled = True

    def run(self):
        try:
            n = len(self.streamlines)
            if n == 0:
                self.result_ready.emit(-1)
                return
            if n == 1:
                self.result_ready.emit(0)
                return

            # Prepare data for batch processing (0-10%)
            self.progress.emit(0, "Preparing streamlines...")

            # Convert streamlines to ArraySequence for efficient flat data access
            as_streamlines = nib.streamlines.ArraySequence(self.streamlines)
            flat_data = np.ascontiguousarray(as_streamlines._data.astype(np.float64))
            offsets = np.ascontiguousarray(as_streamlines._offsets.astype(np.int64))
            lengths = np.ascontiguousarray(as_streamlines._lengths.astype(np.int64))

            if self._cancelled:
                self.result_ready.emit(-1)
                return

            # Batch Resampling with parallel Numba (10-40%)
            self.progress.emit(10, "Batch resampling (parallel)...")
            resampled = _resample_batch_numba(
                flat_data, offsets, lengths, self.nb_points
            )

            if self._cancelled:
                self.result_ready.emit(-1)
                return

            # Import AOT kernels and batched parallel dispatcher
            from tractedit_pkg._numba_aot import (
                compute_distances_chunk as _distances_kernel,
                compute_mdf_rows_chunk as _mdf_rows_kernel,
            )
            from tractedit_pkg._numba_aot._parallel_wrappers import (
                parallel_chunks_range,
            )

            batch = _MEDOID_DISTANCE_BATCH

            # Distance Computation (40-90%)
            # For large bundles, use sampling-based approximate medoid
            if n > MEDOID_SAMPLING_THRESHOLD:
                # Approximate medoid using random sampling
                sample_size = min(MEDOID_SAMPLE_SIZE, n // 3)
                self.progress.emit(
                    40,
                    f"Computing distances (sampling {sample_size} of {n})...",
                )

                # Generate random sample indices
                np.random.seed(42)  # Reproducible results
                sample_indices = np.random.choice(n, sample_size, replace=False).astype(
                    np.int64
                )

                if self._cancelled:
                    self.result_ready.emit(-1)
                    return

                # Batched distance computation with cancellation checkpoints
                distances = np.zeros((n, sample_size), dtype=np.float64)
                for batch_start in range(0, n, batch):
                    if self._cancelled:
                        self.result_ready.emit(-1)
                        return
                    batch_end = min(batch_start + batch, n)
                    pct = 40 + int(45 * batch_start / n)
                    self.progress.emit(
                        pct,
                        f"Computing distances ({batch_start:,}/{n:,})...",
                    )
                    parallel_chunks_range(
                        _distances_kernel,
                        batch_start,
                        batch_end,
                        resampled,
                        sample_indices,
                        distances,
                    )

                # Sum distances to find approximate medoid
                self.progress.emit(85, "Finding approximate medoid...")
                total_dists = np.sum(distances, axis=1)

            else:
                # Exact medoid for smaller bundles
                self.progress.emit(40, "Computing distance matrix...")
                dist_matrix = np.zeros((n, n), dtype=np.float64)

                # Batched row computation with cancellation checkpoints
                for batch_start in range(0, n, batch):
                    if self._cancelled:
                        self.result_ready.emit(-1)
                        return
                    batch_end = min(batch_start + batch, n)
                    pct = 40 + int(45 * batch_start / n)
                    self.progress.emit(
                        pct,
                        f"Computing distances ({batch_start:,}/{n:,})...",
                    )
                    parallel_chunks_range(
                        _mdf_rows_kernel,
                        batch_start,
                        batch_end,
                        resampled,
                        dist_matrix,
                    )

                self.progress.emit(85, "Finding medoid...")
                total_dists = np.sum(dist_matrix, axis=1)

            # Find Medoid (85-100%)
            medoid_idx = int(np.argmin(total_dists))

            self.progress.emit(100, "Done")
            self.result_ready.emit(medoid_idx)

        except (ValueError, IndexError, TypeError, MemoryError) as e:
            self.error.emit(str(e))


class AnatomicalImageLoaderThread(QThread):
    """
    Background thread to load anatomical images without freezing the GUI.
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

            # Load the NIfTI file (lazy - data not loaded yet)
            img = nib.load(self.input_path)

            img = _align_if_oblique(
                img, self.input_path, lambda msg: self.progress.emit(15, msg)
            )

            # Get file size estimate for progress feedback
            header = img.header
            shape = header.get_data_shape()
            if len(shape) >= 3:
                total_voxels = shape[0] * shape[1] * shape[2]
                size_mb = (total_voxels * 4) / (1024 * 1024)  # float32 = 4 bytes
                self.progress.emit(
                    20,
                    f"Checking {shape[0]}×{shape[1]}×{shape[2]} ({size_mb:.0f} MB)...",
                )
            else:
                self.progress.emit(20, "Loading image data...")

            # Use auto-downsampling for large images
            def progress_callback(percent, message):
                self.progress.emit(percent, message)

            image_data, image_affine, was_downsampled = _maybe_downsample_image(
                img, progress_callback
            )

            self.progress.emit(85, "Creating memory-mapped accessor...")

            # Create memory-mapped image for full-resolution 2D slicing
            mmap_image = MemoryMappedImage(img)

            self.progress.emit(90, "Validating...")

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

            status_msg = "Done"
            if was_downsampled:
                status_msg = f"Done (downsampled to {image_data.shape[0]}×{image_data.shape[1]}×{image_data.shape[2]})"

            self.progress.emit(100, status_msg)
            self.finished.emit(
                {
                    "data": image_data,
                    "affine": image_affine,
                    "path": self.input_path,
                    "was_downsampled": was_downsampled,
                    "mmap_image": mmap_image,
                }
            )

        except FileNotFoundError:
            self.error.emit(f"File not found: {self.input_path}")
        except nib.filebasedimages.ImageFileError as e:
            self.error.emit(f"Invalid NIfTI file: {e}")
        except (OSError, ValueError, MemoryError) as e:
            self.error.emit(f"Error loading image: {type(e).__name__}: {e}")


# ============================================================================
# AOT-Compiled Functions
# ============================================================================


# _compute_bboxes_numba — AOT chunk + ThreadPool wrapper
from tractedit_pkg._numba_aot._parallel_wrappers import (
    compute_bboxes as _compute_bboxes_numba,
)

# Key used to store/retrieve cached bounding boxes inside TRX files.
_TRX_BBOX_KEY = "_tractedit_bboxes"


# ============================================================================
# TRX Bbox Caching Helpers (Phase 1)
# ============================================================================


def _try_load_cached_bboxes(
    trx_obj: "tbx.TrxFile",
    n_streamlines: int,
) -> Optional[np.ndarray]:
    """
    Attempt to load cached bounding boxes from a TRX file's
    ``data_per_streamline``.

    The bboxes are stored as a flat ``(N, 6)`` float32 array
    (``[min_x, min_y, min_z, max_x, max_y, max_z]``) and are reshaped
    to the application-standard ``(N, 2, 3)`` layout on load.

    Args:
        trx_obj: The loaded TRX file object (memmap-backed).
        n_streamlines: Expected number of streamlines for shape validation.

    Returns:
        Bounding boxes as ``np.ndarray`` of shape ``(N, 2, 3)``
        and dtype ``float32``, or ``None`` if the cache is absent,
        corrupted, or has the wrong shape.
    """
    try:
        dps = getattr(trx_obj, "data_per_streamline", None)
        if dps is None or _TRX_BBOX_KEY not in dps:
            return None

        raw = dps[_TRX_BBOX_KEY]
        if not isinstance(raw, np.ndarray):
            logger.warning(
                "TRX bbox cache: unexpected type %s, recomputing.",
                type(raw).__name__,
            )
            return None

        if raw.shape != (n_streamlines, 6):
            logger.warning(
                "TRX bbox cache: shape mismatch (expected (%d, 6), got %s), "
                "recomputing.",
                n_streamlines,
                raw.shape,
            )
            return None

        # Reshape from flat (N, 6) → (N, 2, 3) expected by the app
        cached = np.asarray(raw, dtype=np.float32).reshape(-1, 2, 3)
        logger.info(
            "TRX: loaded cached bounding boxes from "
            "data_per_streamline (%d streamlines, skipped computation).",
            n_streamlines,
        )
        return cached

    except (ValueError, KeyError, TypeError, IndexError) as exc:
        logger.warning(
            "TRX bbox cache: failed to read (%s), recomputing.", exc
        )
        return None


def _embed_cached_bboxes(
    trx_obj: "tbx.TrxFile",
    bboxes: Optional[np.ndarray],
) -> None:
    """
    Embed precomputed bounding boxes into a TRX object before saving.

    The bboxes are flattened from the application-standard ``(N, 2, 3)``
    layout to ``(N, 6)`` float32 for TRX ``data_per_streamline`` storage.

    Args:
        trx_obj: The TRX object that will be saved.
        bboxes: Bounding boxes of shape ``(N, 2, 3)``.  If ``None`` or
            empty, no embedding is performed.
    """
    if bboxes is None or bboxes.size == 0:
        return

    try:
        flat = bboxes.reshape(-1, 6).astype(np.float32)
        trx_obj.data_per_streamline[_TRX_BBOX_KEY] = flat
        logger.debug(
            "TRX: embedded cached bounding boxes (%d streamlines).",
            flat.shape[0],
        )
    except (ValueError, TypeError, IndexError) as exc:
        # Non-fatal: saving continues without the cache.
        logger.warning(
            "TRX: failed to embed cached bboxes (%s). "
            "Saving continues without bbox cache.",
            exc,
        )


def _extract_visible_bboxes(main_window: Any) -> Optional[np.ndarray]:
    """
    Extract bounding boxes for the currently visible streamlines.

    Returns a contiguous ``(M, 2, 3)`` float32 array indexed by the
    sorted visible indices, or ``None`` if bboxes are not available.

    Args:
        main_window: The MainWindow instance holding application state.

    Returns:
        Bounding boxes array or ``None``.
    """
    if (
        main_window.streamline_bboxes is None
        or not main_window.visible_indices
    ):
        return None

    indices = sorted(main_window.visible_indices)
    return main_window.streamline_bboxes[indices]


# ============================================================================
# TRX Native Save (Phase 2)
# ============================================================================


def _save_trx_native(
    trx_source: "tbx.TrxFile",
    visible_indices: set,
    bboxes: Optional[np.ndarray],
    output_path: str,
) -> str:
    """
    Save a TRX file using the native trx-python API.

    Uses ``TrxFile.select(copy_safe=True)`` to extract visible streamlines
    without Python-level iteration, then embeds cached bounding boxes
    as ``data_per_streamline`` before saving.

    Parameters
    ----------
    trx_source : TrxFile
        The original loaded TRX object (memmap-backed).
    visible_indices : set of int
        Indices of streamlines to include in the saved file.
    bboxes : ndarray of shape ``(N_total, 2, 3)`` or ``None``
        Full bounding box array for all streamlines.  Only the rows
        corresponding to *visible_indices* will be embedded.
    output_path : str
        Destination file path.

    Returns
    -------
    str
        Success message.

    Raises
    ------
    Exception
        Any failure is propagated to the caller so that
        ``save_streamlines_file`` can fall back to the generic path.
    """
    indices_arr = np.sort(
        np.fromiter(visible_indices, dtype=np.int64, count=len(visible_indices))
    )
    logger.info(
        "TRX native save: selecting %d / %d streamlines.",
        len(indices_arr),
        len(trx_source.streamlines),
    )

    trx_subset = trx_source.select(
        indices_arr, keep_group=False, copy_safe=True
    )

    # Embed cached bboxes for fast reload (Phase 1 integration)
    if bboxes is not None:
        try:
            saved_bboxes = bboxes[indices_arr]  # (M, 2, 3)
            _embed_cached_bboxes(trx_subset, saved_bboxes)
        except (ValueError, IndexError, TypeError) as exc:
            # Non-fatal: continue saving without bbox cache
            logger.warning(
                "TRX native save: failed to embed bboxes (%s). "
                "Saving continues without bbox cache.",
                exc,
            )

    tbx.save(trx_subset, output_path)
    logger.info(
        "File saved successfully (TRX, native path): %s",
        os.path.basename(output_path),
    )
    return (
        f"File saved successfully (TRX): {os.path.basename(output_path)}"
    )


# ============================================================================
# Helper Functions
# ============================================================================


def parse_numeric_tuple_from_string(
    input_value: Union[str, List, Tuple, np.ndarray, Any],
    target_type: Type = float,
    expected_length: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Any:
    """
    Parses an input (string, list, tuple, or ndarray) into a tuple/array of a specific numeric type.

    If parsing or validation fails, the original input_value is returned.

    Args:
        input_value: The input data to parse. Can be a string like "(1, 2, 3)" or "1 2 3",
                     a sequence, or a numpy array.
        target_type: The desired type for the elements (default: float).
        expected_length: The expected length (int) or shape (tuple) of the result.

    Returns:
        The parsed tuple/array cast to target_type, or input_value if parsing/validation fails.
    """

    # 1. Handle non-string inputs (Already lists, tuples, or arrays)
    if not isinstance(input_value, str):
        return _process_existing_sequence(input_value, target_type, expected_length)

    # 2. Parse by stripping brackets and splitting on commas/whitespace.
    # This handles formats like "(1, 2)", "[1, 2]", "1 2 3", "1, 2, 3"
    # without using ast.literal_eval
    cleaned_str = input_value.translate(str.maketrans("", "", "[]()"))
    parts = cleaned_str.replace(",", " ").split()

    if not parts:
        return input_value

    # Helper: parse a single token to the target type.
    # When target_type is int, parse as float first to handle "1.9" -> 1.
    def _convert(token: str) -> target_type:
        if target_type is int:
            return int(float(token))
        return target_type(token)

    # Single scalar value
    if len(parts) == 1:
        try:
            val = _convert(parts[0])
            if expected_length == 1:
                return (val,)
            return val
        except (ValueError, TypeError):
            return input_value

    # Multiple values — convert to tuple
    try:
        converted = tuple(_convert(p) for p in parts)
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


# _resample_streamline_numba — AOT-compiled (see _numba_aot/build_aot.py)
from tractedit_pkg._numba_aot import (
    resample_streamline as _resample_streamline_numba,
)


# _resample_batch_numba — AOT chunk + ThreadPool wrapper
from tractedit_pkg._numba_aot._parallel_wrappers import (
    resample_batch as _resample_batch_numba,
)


# _compute_centroid_numba — AOT-compiled (see _numba_aot/build_aot.py)
from tractedit_pkg._numba_aot import (
    compute_centroid as _compute_centroid_numba,
)


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


# _compute_mdf_distance_matrix — AOT chunk + ThreadPool wrapper
from tractedit_pkg._numba_aot._parallel_wrappers import (
    compute_mdf_distance_matrix as _compute_mdf_distance_matrix,
)

# _compute_distances_to_samples — AOT chunk + ThreadPool wrapper
from tractedit_pkg._numba_aot._parallel_wrappers import (
    compute_distances_to_samples as _compute_distances_to_samples,
)


def _finalize_statistic_save(
    main_window: Any, result_streamline: np.ndarray, method: str
) -> None:
    """
    Helper to save the calculated statistic (centroid/medoid) to a file.
    """
    method_ui = method.capitalize()
    status_updater = getattr(
        main_window.vtk_panel,
        "update_status",
        lambda msg: logger.info(f"Status: {msg}"),
    )

    try:
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

    except (OSError, ValueError, TypeError) as e:
        logger.error(f"Failed to save {method}: {e}", exc_info=True)
        QMessageBox.critical(main_window, "Error", f"Failed to save {method}:\n{e}")
        status_updater(f"Error saving {method}.")


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

    # Safety Check for Centroid
    if method == "centroid" and len(main_window.visible_indices) > CENTROID_MAX_STREAMLINES:
        QMessageBox.warning(
            main_window,
            "Safety Warning",
            f"Too many streamlines selected ({len(main_window.visible_indices):,}).\n"
            f"Centroid calculation runs on the main thread and would freeze\n"
            f"the application with this many streamlines.\n"
            f"Please reduce the selection to below {CENTROID_MAX_STREAMLINES:,} streamlines.",
        )
        status_updater("Centroid calculation aborted (too many streamlines).")
        return

    # Extract Visible Data
    tractogram_data = main_window.tractogram_data
    visible_streamlines = [tractogram_data[i] for i in main_window.visible_indices]

    if method == "centroid":
        status_updater("Calculating centroid (this may take a moment)...")
        QApplication.processEvents()
        try:
            result_streamline = _compute_centroid_math(visible_streamlines)
            _finalize_statistic_save(main_window, result_streamline, method)
        except (ValueError, IndexError, TypeError, MemoryError) as e:
            logger.error(f"Failed to calculate centroid: {e}", exc_info=True)
            QMessageBox.critical(
                main_window, "Error", f"Failed to calculate centroid:\n{e}"
            )
            status_updater("Error calculating centroid.")

    elif method == "medoid":
        # Setup Progress Dialog
        progress = QProgressDialog("Initializing...", "Cancel", 0, 100, main_window)
        progress.setWindowTitle("TractEdit - Medoid Calculation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        # Create Thread
        thread = MedoidCalculationThread(visible_streamlines)
        main_window._medoid_thread = thread  # Keep reference

        def on_progress(val, msg):
            if not progress.wasCanceled():
                progress.setValue(val)
                progress.setLabelText(msg)

        def on_result(idx):
            progress.close()
            if idx == -1:
                status_updater("Medoid calculation cancelled.")
            else:
                result_streamline = visible_streamlines[idx]
                _finalize_statistic_save(main_window, result_streamline, method)

        def on_error(msg):
            progress.close()
            QMessageBox.critical(
                main_window, "Error", f"Medoid calculation failed:\n{msg}"
            )

        def on_thread_finished():
            """Clean up after QThread has fully stopped."""
            t = getattr(main_window, "_medoid_thread", None)
            if t is not None:
                t.wait()
                del main_window._medoid_thread

        thread.progress.connect(on_progress)
        thread.result_ready.connect(
            on_result, type=Qt.ConnectionType.QueuedConnection
        )
        thread.error.connect(on_error)
        # QThread's built-in finished fires after run() has fully exited
        thread.finished.connect(
            on_thread_finished, type=Qt.ConnectionType.QueuedConnection
        )
        progress.canceled.connect(thread.cancel)

        thread.start()


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
        except (ValueError, IndexError, TypeError) as e:
            logger.warning(f"Could not recalculate parcellation intersection: {e}")


# Anatomical Image Loading Function
def load_anatomical_image(
    main_window: Any,
    file_path: Optional[str] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[str],
    Optional["MemoryMappedImage"],
]:
    """
    Loads a NIfTI image file (.nii, .nii.gz).
    Returns the image data array, affine matrix, and memory-mapped accessor.

    Args:
        main_window: The instance of the main application window.
        file_path: Optional path to the file to load. If None, opens a dialog.

    Returns:
        tuple: (image_data, affine, path, mmap_image) or (None, None, None, None).
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
        return None, None, None, None

    status_updater = getattr(
        main_window.vtk_panel,
        "update_status",
        lambda msg: logger.info(f"Status: {msg}"),
    )
    status_updater(f"Loading image: {os.path.basename(input_path)}...")
    QApplication.processEvents()

    try:
        # Load NIfTI (lazy - data not loaded yet)
        img = nib.load(input_path)

        img = _align_if_oblique(img, input_path, status_updater)

        # Use auto-downsampling for large images
        image_data, image_affine, was_downsampled = _maybe_downsample_image(img)

        # Create memory-mapped accessor for full-resolution 2D slicing
        mmap_image = MemoryMappedImage(img)

        # Basic validation
        if image_data.ndim < 3:
            raise ValueError(
                f"Loaded image has only {image_data.ndim} dimensions, expected 3 or more."
            )
        if image_affine.shape != (4, 4):
            raise ValueError(
                f"Loaded image affine has shape {image_affine.shape}, expected (4, 4)."
            )

        if was_downsampled:
            status_updater(
                f"Loaded {os.path.basename(input_path)} (downsampled to {image_data.shape[0]}×{image_data.shape[1]}×{image_data.shape[2]})"
            )
        else:
            status_updater(
                f"Successfully loaded anatomical image: {os.path.basename(input_path)}"
            )
        return image_data, image_affine, input_path, mmap_image

    except FileNotFoundError:
        error_msg = f"Error: Anatomical image file not found:\n{input_path}"
        logger.error(error_msg)
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error: File not found - {os.path.basename(input_path)}")
        return None, None, None, None
    except nib.filebasedimages.ImageFileError as e:
        error_msg = f"Nibabel Error loading anatomical image:\n{e}\n\nIs '{os.path.basename(input_path)}' a valid NIfTI file?"
        logger.error(error_msg)
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error loading NIfTI: {os.path.basename(input_path)}")
        return None, None, None, None
    except (OSError, ValueError, MemoryError) as e:
        error_msg = f"An unexpected error occurred loading the anatomical image:\n{type(e).__name__}: {e}\n\nPath: {input_path}\n\nSee console for details."
        logger.error(error_msg)
        QMessageBox.critical(main_window, "Load Error", error_msg)
        status_updater(f"Error loading image: {os.path.basename(input_path)}")
        return None, None, None, None


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

            img = _align_if_oblique(img, input_path, status_updater)

            image_data_raw = img.get_fdata()
            image_affine = img.affine

            # Preserve data integrity while optimizing memory
            is_integer_data = np.allclose(image_data_raw, np.round(image_data_raw))

            if is_integer_data:
                # Round to handle floating point artifacts (e.g., 1.9999999 -> 2)
                image_data_rounded = np.round(image_data_raw)
                min_val = image_data_rounded.min()
                max_val = image_data_rounded.max()

                # Choose smallest appropriate integer type
                if min_val >= 0 and max_val <= 255:
                    image_data = image_data_rounded.astype(np.uint8)
                elif min_val >= 0 and max_val <= 65535:
                    image_data = image_data_rounded.astype(np.uint16)
                elif min_val >= -32768 and max_val <= 32767:
                    image_data = image_data_rounded.astype(np.int16)
                else:
                    image_data = image_data_rounded.astype(np.int32)

                logger.debug(
                    f"ROI '{os.path.basename(input_path)}' loaded as {image_data.dtype} "
                    f"(range: {min_val:.0f} to {max_val:.0f})"
                )
            else:
                # Preserve floating point for probability maps, partial volume estimates, etc.
                image_data = image_data_raw.astype(np.float32)
                logger.debug(
                    f"ROI '{os.path.basename(input_path)}' loaded as float32 "
                    f"(contains non-integer values, range: {image_data_raw.min():.4f} to {image_data_raw.max():.4f})"
                )

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

        except (OSError, ValueError, MemoryError) as e:
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

    # Apply theme-aware style
    if hasattr(main_window, "theme_manager"):
        progress.setStyleSheet(main_window.theme_manager.get_progress_dialog_style())
    else:
        # Fallback to dark style if theme manager not available
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
            # Apply Data to MainWindow
            main_window.tractogram_data = data["streamlines"]
            main_window.streamline_bboxes = data["bboxes"]

            main_window.original_trk_header = data["header"]
            main_window.original_trk_affine = data["affine"]
            main_window.original_trk_path = data["path"]
            main_window.original_file_extension = data["ext"]

            # Close any previously open TRX file before assigning the new one
            main_window._close_trx_file()
            main_window.trx_file_reference = data.get("trx_obj", None)
            main_window.scalar_data_per_point = data["scalars"]
            main_window.active_scalar_name = data["active_scalar"]

            # Initialize Logic State
            total_fibers = len(main_window.tractogram_data)
            main_window.manual_visible_indices = set(range(total_fibers))
            main_window.visible_indices = set(range(total_fibers))
            main_window._visibility_version += 1

            # Reset Caches
            main_window.roi_states = {}
            main_window.roi_intersection_cache = {}
            main_window.roi_highlight_indices = set()

            main_window.selected_streamline_indices = set()
            main_window._inversion_active = False
            main_window._inversion_keeper_indices = set()
            if main_window.vtk_panel:
                main_window.vtk_panel.clear_invert_contour()
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
                # Reset user override so auto-calc works for the new bundle
                main_window._skip_user_disabled = False
                main_window._auto_calculate_skip_level()
                should_render = False

            # Finalize UI + first VTK render
            status_msg = f"Loaded {len(main_window.tractogram_data)} streamlines from {os.path.basename(data['path'])}"
            _update_vtk_and_ui_after_load(main_window, status_msg, render=should_render)

            # Close the progress dialog only after the actor is built and the scene has been rendered
            progress.setValue(100)

        except (OSError, ValueError, TypeError, AttributeError, IndexError) as e:
            progress.setValue(100)  # Ensure the dialog always closes on error
            logger.error(f"Error in on_finished: {e}", exc_info=True)
            QMessageBox.critical(
                main_window, "Load Error", f"Error finalizing load:\n{e}"
            )
            # Attempt to cleanup if possible
            if hasattr(main_window, "_close_bundle"):
                # Don't recurse if close bundle fails — intentionally broad
                try:
                    main_window._close_bundle(keep_image=True)
                except Exception:
                    logger.debug("Cleanup after load failure also failed.")

    # Connect Signals
    loader_thread.progress.connect(on_progress)
    loader_thread.error.connect(on_error)
    loader_thread.finished.connect(
        on_finished, type=Qt.ConnectionType.QueuedConnection
    )
    progress.canceled.connect(loader_thread.cancel)

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
        except (ValueError, IndexError, KeyError, TypeError) as e:
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
            except (ValueError, TypeError) as e:
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
    # Start with a clean header to avoid TRK-specific fields polluting TCK
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
                    except (UnicodeDecodeError, ValueError):
                        logger.debug("Failed to decode header field '%s'.", key)
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
            except (ValueError, np.linalg.LinAlgError):
                logger.debug("Failed to compute voxel sizes from affine, using default.")
                header["voxel_sizes"] = (1.0, 1.0, 1.0)
        else:
            header["voxel_sizes"] = (1.0, 1.0, 1.0)
    else:
        header["voxel_sizes"] = valid_vox_sizes

    # Ensure voxel_order exists
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
    bboxes_to_embed: Optional[np.ndarray] = None,
) -> str:
    """
    Saves the tractogram using nibabel or trx-python based on the extension.

    Args:
        tractogram: The nibabel Tractogram to save.
        header: Header dictionary with format-specific metadata.
        output_path: Destination file path.
        file_ext: File extension (e.g., '.trk', '.tck', '.trx').
        bboxes_to_embed: Optional bounding boxes array of shape (N, 2, 3)
            to embed in the TRX file as cached metadata. Ignored for
            non-TRX formats.

    Returns:
        Success message string.
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
        trx_obj_to_save = tbx.TrxFile.from_lazy_tractogram(
            tractogram, reference_img
        )

        # Phase 1 optimisation: embed cached bounding boxes so that
        # subsequent loads can skip the bbox computation step entirely.
        _embed_cached_bboxes(trx_obj_to_save, bboxes_to_embed)

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

    # ------------------------------------------------------------------
    # Phase 2: TRX-native fast path
    # ------------------------------------------------------------------
    if output_ext == ".trx" and main_window.trx_file_reference is not None:
        try:
            success_msg = _save_trx_native(
                trx_source=main_window.trx_file_reference,
                visible_indices=main_window.visible_indices,
                bboxes=main_window.streamline_bboxes,
                output_path=output_path,
            )
            status_updater(success_msg)
            return
        except (OSError, ValueError, TypeError) as e:
            logger.warning(
                "TRX native save failed, falling back to generic path: %s",
                e,
            )
            # Fall through to generic path below

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
            tractogram,
            header_to_save,
            output_path,
            output_ext,
            bboxes_to_embed=_extract_visible_bboxes(main_window),
        )
        status_updater(success_msg)

    except (OSError, ValueError, TypeError) as e:
        logger.error(f"Error during file saving:\nType: {type(e).__name__}\nError: {e}")
        error_msg = (
            f"Error saving file:\n{type(e).__name__}: {e}\n\nCheck console for details."
        )
        QMessageBox.critical(main_window, "Save Error", error_msg)
        status_updater(f"Error saving file: {os.path.basename(output_path)}")
