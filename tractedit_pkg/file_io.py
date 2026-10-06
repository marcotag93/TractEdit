# -*- coding: utf-8 -*-

"""
Functions for loading and saving streamline files (trk, tck, trx)
and loading anatomical image files (NIfTI).
"""

# ============================================================================
# Imports
# ============================================================================

import hashlib
import logging
import os
import zlib
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import trx.trx_file_memmap as tbx
import vtk
from vtk.util import numpy_support
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QApplication,
    QProgressDialog,
    QWidget,
)
from .utils import ColorMode, write_vtk_polydata
from .tractogram_metadata import (
    add_vtk_metadata,
    ensure_metadata_supported,
    extract_vtk_metadata,
)
from .reference_grid import (
    ReferenceGrid,
    validate_affine as _validate_affine,
    validate_volume_geometry as _validate_volume_geometry,
)
from .transactional_io import transactional_save
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from .data_contracts import AnatomicalImageLoadResult, StreamlineLoadResult

logger = logging.getLogger(__name__)


def _register_background_worker(
    main_window: Any,
    worker_attr: str,
    generation_attr: str,
    worker: Any,
) -> int:
    """Register one owned worker and invalidate any preceding operation."""
    previous = getattr(main_window, worker_attr, None)
    if previous is not None and previous is not worker:
        cancel = getattr(previous, "cancel", None)
        if callable(cancel):
            cancel()
        progress_dialog = getattr(previous, "progress_dialog", None)
        if progress_dialog is not None:
            try:
                progress_dialog.close()
            except RuntimeError:
                logger.debug("Previous worker progress dialog is already closed.")

    generation = getattr(main_window, generation_attr, 0) + 1
    setattr(main_window, generation_attr, generation)
    setattr(main_window, worker_attr, worker)

    workers = getattr(main_window, "_background_workers", None)
    if workers is None:
        workers = []
        setattr(main_window, "_background_workers", workers)
    workers.append(worker)
    return generation


def _worker_is_current(
    main_window: Any,
    worker_attr: str,
    generation_attr: str,
    worker: Any,
    generation: int,
) -> bool:
    return (
        getattr(main_window, worker_attr, None) is worker
        and getattr(main_window, generation_attr, 0) == generation
        and not getattr(worker, "is_cancelled", False)
    )


def _release_background_worker(
    main_window: Any,
    worker_attr: str,
    worker: Any,
) -> None:
    workers = getattr(main_window, "_background_workers", None)
    if workers is not None:
        try:
            workers.remove(worker)
        except ValueError:
            pass
    if getattr(main_window, worker_attr, None) is worker:
        setattr(main_window, worker_attr, None)
    if hasattr(worker, "progress_dialog"):
        progress_dialog = worker.progress_dialog
        if progress_dialog is not None:
            try:
                progress_dialog.deleteLater()
            except RuntimeError:
                logger.debug("Worker progress dialog was already deleted.")
        worker.progress_dialog = None


def _release_finished_worker(
    main_window: Any,
    worker_attr: str,
    worker: Any,
) -> None:
    if worker.has_pending_result:
        return
    if worker.isRunning():
        QTimer.singleShot(
            10,
            lambda: _release_finished_worker(main_window, worker_attr, worker),
        )
        return
    _release_background_worker(main_window, worker_attr, worker)


def _release_deferred_trx_owners(main_window: Any) -> None:
    deferred = getattr(main_window, "_deferred_trx_owners", [])
    workers = getattr(main_window, "_background_workers", [])
    remaining = []
    for owner in deferred:
        in_use = any(
            getattr(worker, "trx_owner", None) is owner
            and (
                worker.isRunning()
                or getattr(worker, "is_consuming_result", False)
                or getattr(worker, "has_pending_result", False)
            )
            for worker in workers
        )
        if in_use:
            remaining.append(owner)
            continue
        try:
            owner.close()
        except OSError:
            logger.warning("Failed to close retired TRX owner.", exc_info=True)
            remaining.append(owner)
    main_window._deferred_trx_owners = remaining


def _retire_trx_owner(main_window: Any, owner: Any) -> None:
    if owner is None:
        return
    workers = getattr(main_window, "_background_workers", [])
    users = [
        worker for worker in workers if getattr(worker, "trx_owner", None) is owner
    ]
    for worker in users:
        worker.cancel()
        progress_dialog = getattr(worker, "progress_dialog", None)
        if progress_dialog is not None:
            try:
                progress_dialog.close()
            except RuntimeError:
                logger.debug("Previous worker progress dialog is already closed.")
    if any(
        worker.isRunning()
        or getattr(worker, "is_consuming_result", False)
        or getattr(worker, "has_pending_result", False)
        for worker in users
    ):
        deferred = getattr(main_window, "_deferred_trx_owners", None)
        if deferred is None:
            deferred = []
            main_window._deferred_trx_owners = deferred
        if not any(item is owner for item in deferred):
            deferred.append(owner)
        return
    try:
        owner.close()
    except OSError:
        logger.warning("Failed to close retired TRX owner.", exc_info=True)
        deferred = getattr(main_window, "_deferred_trx_owners", None)
        if deferred is None:
            deferred = []
            main_window._deferred_trx_owners = deferred
        if not any(item is owner for item in deferred):
            deferred.append(owner)


_BUNDLE_STATE_ATTRIBUTES = (
    "tractogram_data",
    "_tractogram_data_version",
    "streamline_bboxes",
    "original_trk_header",
    "original_trk_affine",
    "original_trk_path",
    "original_file_extension",
    "tractogram_reference_grid",
    "trx_file_reference",
    "scalar_data_per_point",
    "data_per_streamline",
    "active_scalar_name",
    "manual_visible_indices",
    "visible_indices",
    "_visibility_version",
    "roi_states",
    "roi_intersection_cache",
    "roi_highlight_indices",
    "selected_streamline_indices",
    "_inversion_active",
    "_inversion_keeper_indices",
    "unified_undo_stack",
    "unified_redo_stack",
    "current_color_mode",
    "_skip_user_disabled",
    "render_stride",
    "_last_visibility_version",
    "_last_render_stride",
    "_last_color_mode",
    "_last_active_scalar",
    "_last_tube_mode",
    "_last_bundle_opacity",
    "render_as_tubes",
    "bundle_opacity",
    "bundle_is_visible",
    "scalar_min_val",
    "scalar_max_val",
    "scalar_data_min",
    "scalar_data_max",
    "scalar_range_initialized",
    "anatomical_image_data",
    "anatomical_image_affine",
    "anatomical_image_path",
    "anatomical_mmap_image",
    "anatomical_reference_grid",
    "image_is_visible",
)


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


def _legacy_medoid_sample_indices(count: int, sample_size: int) -> np.ndarray:
    """Return the historical seeded sample without mutating global RNG state."""
    random_state = np.random.RandomState(42)
    return random_state.choice(count, sample_size, replace=False).astype(np.int64)


def _historical_mdf_distance(
    resampled: np.ndarray, first: int, second: int
) -> float:
    """Replay the former row kernel's arithmetic for one pair."""
    points = resampled.shape[1]
    direct = 0.0
    for point in range(points):
        dx = resampled[first, point, 0] - resampled[second, point, 0]
        dy = resampled[first, point, 1] - resampled[second, point, 1]
        dz = resampled[first, point, 2] - resampled[second, point, 2]
        direct += np.sqrt(dx * dx + dy * dy + dz * dz)
    direct /= points
    flipped = 0.0
    for point in range(points):
        other = points - 1 - point
        dx = resampled[first, point, 0] - resampled[second, other, 0]
        dy = resampled[first, point, 1] - resampled[second, other, 1]
        dz = resampled[first, point, 2] - resampled[second, other, 2]
        flipped += np.sqrt(dx * dx + dy * dy + dz * dz)
    flipped /= points
    return direct if direct < flipped else flipped


def _historical_mdf_row_total(
    resampled: np.ndarray, row_index: int, sample_indices: Optional[np.ndarray]
) -> float:
    """Use one bounded row and NumPy's historical row reduction."""
    columns = range(len(resampled)) if sample_indices is None else sample_indices
    row = np.zeros(len(columns), dtype=np.float64)
    for column_index, other in enumerate(columns):
        other = int(other)
        if row_index == other:
            continue
        first, second = (
            (min(row_index, other), max(row_index, other))
            if sample_indices is None
            else (row_index, other)
        )
        row[column_index] = _historical_mdf_distance(resampled, first, second)
    return float(np.sum(row))


def _refine_historical_medoid_candidates(
    resampled: np.ndarray,
    totals: np.ndarray,
    sample_indices: Optional[np.ndarray],
    batch_size: int,
    cancel_check: Optional[Callable[[], bool]],
) -> Optional[np.ndarray]:
    """Re-evaluate every row whose summation error could change the winner."""
    if not len(totals) or not np.all(np.isfinite(totals)):
        return totals
    maximum = float(np.max(np.abs(totals)))
    if maximum == 0:
        return totals

    # Both reductions sum nonnegative, identical pair distances. Gamma bounds
    # each positive floating sum; the factor four covers both summation trees
    # and the final merge of at most eight private worker arrays.
    terms = (len(resampled) if sample_indices is None else len(sample_indices))
    batches = (len(resampled) + batch_size - 1) // batch_size
    additions = terms + batches + resampled.shape[1] + 40
    unit_roundoff = np.finfo(np.float64).eps / 2
    gamma = additions * unit_roundoff / (1 - additions * unit_roundoff)
    radius = (4 * gamma + 16 * unit_roundoff) * maximum
    candidates = np.flatnonzero(totals <= float(np.min(totals)) + 2 * radius)
    if len(candidates) == 1:
        return totals
    run_start = None
    refined_runs = {}
    if len(candidates) > 16:
        # Consecutive bit-identical streamlines have identical historical
        # distance rows: every other index is on the same side of the run,
        # and all within-run distances are zero. Reuse that exact row sum.
        run_start = np.empty(len(resampled), dtype=np.int64)
        first = 0
        for index in range(len(resampled)):
            if index and not np.array_equal(
                resampled[index], resampled[index - 1]
            ):
                first = index
            run_start[index] = first
    for index in candidates:
        if cancel_check is not None and cancel_check():
            return None
        representative = int(run_start[index]) if run_start is not None else int(index)
        if representative not in refined_runs:
            refined_runs[representative] = _historical_mdf_row_total(
                resampled, representative, sample_indices
            )
        totals[index] = refined_runs[representative]
    return totals


def _compute_medoid_totals(
    resampled: np.ndarray,
    sample_indices: Optional[np.ndarray] = None,
    batch_size: int = _MEDOID_DISTANCE_BATCH,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Optional[np.ndarray]:
    """Compute exact or sampled MDF row totals with bounded working memory."""
    from tractedit_pkg._numba_aot import compute_sampled_mdf_totals_chunk
    from tractedit_pkg._numba_aot._parallel_wrappers import (
        accumulate_mdf_totals_range,
        parallel_chunks_range,
    )

    resampled = np.ascontiguousarray(resampled, dtype=np.float64)
    count = len(resampled)
    totals = np.zeros(count, dtype=np.float64)
    if batch_size < 1:
        raise ValueError("Medoid batch size must be positive.")

    if sample_indices is not None:
        sample_indices = np.ascontiguousarray(sample_indices, dtype=np.int64)

    for batch_start in range(0, count, batch_size):
        if cancel_check is not None and cancel_check():
            return None
        batch_end = min(batch_start + batch_size, count)
        if progress_callback is not None:
            progress_callback(batch_start, count)
        if sample_indices is None:
            totals += accumulate_mdf_totals_range(resampled, batch_start, batch_end)
        else:
            parallel_chunks_range(
                compute_sampled_mdf_totals_chunk,
                batch_start,
                batch_end,
                resampled,
                sample_indices,
                totals,
            )
    return _refine_historical_medoid_candidates(
        resampled, totals, sample_indices, batch_size, cancel_check
    )


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


def _canonicalize_image(
    img: nib.Nifti1Image, input_path: str, status_updater: callable
) -> nib.Nifti1Image:
    """
    Reorient an image to the closest canonical voxel order without resampling.

    NiBabel reorders/flips the data proxy and composes the affine accordingly,
    so every voxel retains its original scanner-RAS position. Oblique rotation
    and shear are deliberately preserved for the rendering transform.
    """
    from .reference_grid import canonicalize_nifti

    original_orientation = nib.aff2axcodes(img.affine)
    canonical = canonicalize_nifti(img)
    if nib.aff2axcodes(canonical.affine) != original_orientation:
        logger.info(
            "Image %s reoriented from %s to RAS+ without resampling.",
            input_path,
            original_orientation,
        )
    return canonical


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
        progress_callback(40, "Loading display samples...")

    original_affine = img.affine.copy()

    if progress_callback:
        progress_callback(60, f"Downsampling with step={step}...")

    source_proxy = img.dataobj
    resampled_data = np.ascontiguousarray(
        source_proxy[::step, ::step, ::step],
        dtype=np.float32,
    )

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
    done = pyqtSignal()

    def __init__(self, input_path: str) -> None:
        super().__init__()
        self.input_path = input_path
        self._cancelled: bool = False
        self._trx_owner: Optional["tbx.TrxFile"] = None
        self._result_published = False
        self._result_pending = False
        self.is_consuming_result = False

    def cancel(self) -> None:
        """Request cooperative cancellation of the loading operation."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancelled

    @property
    def has_pending_result(self) -> bool:
        return self._result_pending

    def complete_result(self) -> None:
        self._result_pending = False
        self.is_consuming_result = False

    def begin_result(self) -> None:
        self.is_consuming_result = True

    def take_trx_owner(
        self,
        result: "StreamlineLoadResult",
    ) -> Optional["tbx.TrxFile"]:
        """Transfer the loaded TRX owner to the result consumer."""
        owner = result.get("trx_obj")
        if owner is None or owner is not self._trx_owner:
            return None
        self._trx_owner = None
        return owner

    def discard_result(self, result: "StreamlineLoadResult") -> None:
        """Release resources held by an unpublished or rejected result."""
        self.complete_result()
        owner = self.take_trx_owner(result)
        if owner is not None:
            try:
                owner.close()
            except (OSError, RuntimeError, AttributeError):
                logger.warning("Failed to close a rejected TRX result.")

    def discard_pending_result(self) -> None:
        """Release an emitted TRX result that was never consumed."""
        self.complete_result()
        if self._trx_owner is not None:
            try:
                self._trx_owner.close()
            except (OSError, RuntimeError, AttributeError):
                logger.warning("Failed to close a pending TRX result.")
            self._trx_owner = None

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
                temp_aff = getattr(tractogram_obj, "affine_to_rasmm", None)
                if temp_aff is not None:
                    aff = np.asarray(temp_aff)
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
                results["data_per_streamline"] = {
                    key: np.asarray(values)
                    for key, values in tractogram_obj.data_per_streamline.items()
                }

                if self._cancelled:
                    return
                self.progress.emit(70, "Finalizing...")
                results["bboxes"] = _validate_and_compute_bboxes(
                    loaded_streamlines,
                    aff,
                    scalars,
                )

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

                observer_tag = reader.AddObserver("ProgressEvent", _on_reader_progress)
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
                offsets_arr = np.zeros(1, dtype=np.intp)
                connectivity_arr = np.empty(0, dtype=np.intp)
                lengths_arr = np.empty(0, dtype=np.intp)

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
                scalars, data_per_streamline = extract_vtk_metadata(
                    poly_data,
                    connectivity_arr,
                    offsets_arr.astype(np.intp, copy=False),
                    lengths_arr,
                )

                results["scalars"] = scalars
                results["active_scalar"] = list(scalars.keys())[0] if scalars else None
                results["data_per_streamline"] = data_per_streamline

                if self._cancelled:
                    return
                self.progress.emit(80, "Finalizing...")
                results["bboxes"] = _validate_and_compute_bboxes(
                    as_streamlines,
                    aff,
                    scalars,
                )

            elif ext == ".trx":
                self.progress.emit(10, "Loading TRX file...")
                try:
                    trx_obj = tbx.load(self.input_path)
                except (OSError, ValueError, TypeError, IndexError, MemoryError):
                    raise
                except Exception as exc:
                    # TRX/ZIP parsers raise several exception types for invalid bytes.
                    raise ValueError(f"Invalid TRX file: {exc}") from exc
                self._trx_owner = trx_obj
                results["trx_obj"] = trx_obj
                results["streamlines"] = trx_obj.streamlines
                results["header"] = trx_obj.header.copy()

                # Affine
                aff = np.identity(4)
                temp_aff = getattr(trx_obj, "affine_to_rasmm", None)
                if temp_aff is not None:
                    aff = np.asarray(temp_aff)
                results["affine"] = aff

                # Scalars (Basic check)
                scalars = {}
                dpp = getattr(trx_obj, "data_per_vertex", None)
                if dpp:
                    scalars.update(dpp)
                results["scalars"] = scalars
                results["active_scalar"] = list(scalars.keys())[0] if scalars else None
                results["data_per_streamline"] = {
                    key: values
                    for key, values in trx_obj.data_per_streamline.items()
                    if key != _TRX_BBOX_KEY
                }

                if self._cancelled:
                    return

                # Normalise coordinate dtype to float32 (matches TRK/TCK behaviour).
                streamlines = trx_obj.streamlines
                if (
                    hasattr(streamlines, "_data")
                    and streamlines._data.dtype != np.float32
                ):
                    streamlines._data = streamlines._data.astype(np.float32, copy=False)

                # Geometry - Bounding box calculation
                # TRX streamlines are ArraySequence with _data, _offsets, _lengths
                if self._cancelled:
                    return
                self.progress.emit(30, "Computing bounding boxes...")
                streamlines = trx_obj.streamlines
                n_streamlines = len(streamlines)

                # Phase 1 optimisation: try loading cached bboxes from
                # TRX data_per_streamline before recomputing from scratch.
                cached_bboxes = _try_load_cached_bboxes(trx_obj, n_streamlines)

                results["bboxes"] = _validate_and_compute_bboxes(
                    streamlines,
                    aff,
                    scalars,
                    cached_bboxes=cached_bboxes,
                )

            else:
                self._result_pending = True
                self.error.emit(f"Unsupported file format: {ext}")
                return

            if self._cancelled:
                return

            reference_grid = ReferenceGrid.from_header(
                results["header"],
                provenance=f"bundle:{ext}",
            )
            results["reference_grid"] = reference_grid

            # Emit 99 % (not 100 %) so that QProgressDialog's autoClose does
            # not fire here.  The dialog will be closed by on_finished() only
            # after the actor build and first VTK render have completed.
            self.progress.emit(99, "Building visualization...")
            self._result_published = True
            self._result_pending = True
            self.finished.emit(results)

        except (OSError, ValueError, TypeError, IndexError, MemoryError) as e:
            self._result_pending = True
            self.error.emit(str(e))
        finally:
            if not self._result_published and self._trx_owner is not None:
                try:
                    self._trx_owner.close()
                except OSError:
                    logger.debug("Failed to close an unpublished TRX result.")
                self._trx_owner = None
            self.done.emit()


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
        self._result_pending = False
        self.is_consuming_result = False

    def cancel(self):
        """Request cancellation of the computation."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancelled

    @property
    def has_pending_result(self) -> bool:
        return self._result_pending

    def begin_result(self) -> None:
        self.is_consuming_result = True

    def complete_result(self) -> None:
        self._result_pending = False
        self.is_consuming_result = False

    def _publish_result(self, index: int) -> None:
        self._result_pending = True
        self.result_ready.emit(index)

    def _publish_error(self, message: str) -> None:
        self._result_pending = True
        self.error.emit(message)

    def run(self):
        try:
            n = len(self.streamlines)
            if n == 0:
                self._publish_result(-1)
                return
            if n == 1:
                self._publish_result(0)
                return

            # Prepare data for batch processing (0-10%)
            self.progress.emit(0, "Preparing streamlines...")

            # Convert streamlines to ArraySequence for efficient flat data access
            as_streamlines = nib.streamlines.ArraySequence(self.streamlines)
            flat_data = np.ascontiguousarray(as_streamlines._data.astype(np.float64))
            offsets = np.ascontiguousarray(as_streamlines._offsets.astype(np.int64))
            lengths = np.ascontiguousarray(as_streamlines._lengths.astype(np.int64))

            if self._cancelled:
                self._publish_result(-1)
                return

            # Batch Resampling with parallel Numba (10-40%)
            self.progress.emit(10, "Batch resampling (parallel)...")
            resampled = _resample_batch_numba(
                flat_data, offsets, lengths, self.nb_points
            )

            if self._cancelled:
                self._publish_result(-1)
                return

            sample_indices = None
            if n > MEDOID_SAMPLING_THRESHOLD:
                sample_size = min(MEDOID_SAMPLE_SIZE, n // 3)
                self.progress.emit(
                    40,
                    f"Computing distances (sampling {sample_size} of {n})...",
                )
                sample_indices = _legacy_medoid_sample_indices(n, sample_size)
            else:
                self.progress.emit(40, "Computing distance matrix...")

            def report_progress(batch_start, total):
                percent = 40 + int(45 * batch_start / total)
                self.progress.emit(
                    percent,
                    f"Computing distances ({batch_start:,}/{total:,})...",
                )

            total_dists = _compute_medoid_totals(
                resampled,
                sample_indices=sample_indices,
                cancel_check=lambda: self._cancelled,
                progress_callback=report_progress,
            )
            if total_dists is None:
                self._publish_result(-1)
                return

            status = (
                "Finding approximate medoid..."
                if sample_indices is not None
                else "Finding medoid..."
            )
            self.progress.emit(85, status)

            # Find Medoid (85-100%)
            medoid_idx = int(np.argmin(total_dists))

            self.progress.emit(100, "Done")
            self._publish_result(medoid_idx)

        except (ValueError, IndexError, TypeError, MemoryError) as e:
            self._publish_error(str(e))


class AnatomicalImageLoaderThread(QThread):
    """
    Background thread to load anatomical images without freezing the GUI.
    """

    progress = pyqtSignal(int, str)  # Signal to update progress bar (percent, message)
    finished = pyqtSignal(dict)  # Signal when loading is done
    error = pyqtSignal(str)  # Signal if an error occurs
    done = pyqtSignal()

    def __init__(self, input_path: str):
        super().__init__()
        self.input_path = input_path
        self._cancelled = False
        self._result: Optional["AnatomicalImageLoadResult"] = None
        self._result_pending = False
        self.is_consuming_result = False

    def cancel(self) -> None:
        """Request cooperative cancellation of the image load."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancelled

    @property
    def has_pending_result(self) -> bool:
        return self._result_pending

    def complete_result(self) -> None:
        self._result_pending = False
        self.is_consuming_result = False

    def begin_result(self) -> None:
        self.is_consuming_result = True

    def take_result(
        self,
        result: "AnatomicalImageLoadResult",
    ) -> Optional["AnatomicalImageLoadResult"]:
        """Transfer an image result to its GUI consumer."""
        if result is not self._result:
            return None
        self._result = None
        return result

    def discard_result(self, result: "AnatomicalImageLoadResult") -> None:
        """Release an image result rejected by its GUI consumer."""
        self.complete_result()
        owned_result = self.take_result(result)
        if owned_result is not None:
            mmap_image = owned_result.get("mmap_image")
            if mmap_image is not None:
                try:
                    mmap_image.clear_cache()
                except (OSError, RuntimeError, AttributeError):
                    logger.warning("Failed to clear a rejected image result.")

    def discard_pending_result(self) -> None:
        """Release an emitted image result that was never consumed."""
        self.complete_result()
        if self._result is not None:
            mmap_image = self._result.get("mmap_image")
            if mmap_image is not None:
                try:
                    mmap_image.clear_cache()
                except (OSError, RuntimeError, AttributeError):
                    logger.warning("Failed to clear a pending image result.")
            self._result = None

    def run(self):
        mmap_image = None
        published = False
        try:
            self.progress.emit(10, "Loading NIfTI header...")

            # Load the NIfTI file (lazy - data not loaded yet)
            img = nib.load(self.input_path)
            if self._cancelled:
                return

            _validate_volume_geometry(img.dataobj, img.affine, "Anatomical image")

            img = _canonicalize_image(
                img, self.input_path, lambda msg: self.progress.emit(15, msg)
            )
            if self._cancelled:
                return

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
            if self._cancelled:
                return
            reference_grid = ReferenceGrid.from_nifti(
                img,
                provenance=f"anatomical:{self.input_path}",
            )

            self.progress.emit(85, "Creating memory-mapped accessor...")

            # Create memory-mapped image for full-resolution 2D slicing
            mmap_image = MemoryMappedImage(img)
            if self._cancelled:
                return

            self.progress.emit(90, "Validating...")

            _validate_volume_geometry(
                image_data,
                image_affine,
                "Anatomical image",
            )

            status_msg = "Done"
            if was_downsampled:
                status_msg = f"Done (downsampled to {image_data.shape[0]}×{image_data.shape[1]}×{image_data.shape[2]})"

            self.progress.emit(100, status_msg)
            if self._cancelled:
                return
            self._result = {
                "data": image_data,
                "affine": image_affine,
                "path": self.input_path,
                "was_downsampled": was_downsampled,
                "mmap_image": mmap_image,
                "reference_grid": reference_grid,
            }
            self._result_pending = True
            self.finished.emit(self._result)
            published = True

        except FileNotFoundError:
            self._result_pending = True
            self.error.emit(f"File not found: {self.input_path}")
        except nib.filebasedimages.ImageFileError as e:
            self._result_pending = True
            self.error.emit(f"Invalid NIfTI file: {e}")
        except (OSError, ValueError, MemoryError) as e:
            self._result_pending = True
            self.error.emit(f"Error loading image: {type(e).__name__}: {e}")
        finally:
            if not published and mmap_image is not None:
                try:
                    mmap_image.clear_cache()
                except (OSError, AttributeError):
                    logger.debug("Failed to clear an unpublished image accessor.")
            self.done.emit()


# ============================================================================
# AOT-Compiled Functions
# ============================================================================


# _compute_bboxes_numba — AOT chunk + ThreadPool wrapper
from tractedit_pkg._numba_aot._parallel_wrappers import (
    compute_bboxes as _compute_bboxes_numba,
    compute_bboxes_prevalidated as _compute_bboxes_prevalidated,
    validate_streamlines as _validate_streamlines_numba,
)


def _packed_streamline_arrays(
    streamlines: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from .input_validation import packed_streamline_arrays

    return packed_streamline_arrays(streamlines)


def _array_is_finite(array: np.ndarray, block_bytes: int = 4 * 1024 * 1024) -> bool:
    from .input_validation import array_is_finite

    return array_is_finite(array, block_bytes)


def _validate_scalar_data(
    scalar_data: Dict[str, Any],
    streamline_lengths: np.ndarray,
) -> None:
    for name, sequence in scalar_data.items():
        if len(sequence) != len(streamline_lengths):
            raise ValueError(
                f"Scalar {name!r} has {len(sequence)} streamlines; "
                f"expected {len(streamline_lengths)}."
            )
        scalar_lengths = getattr(sequence, "_lengths", None)
        if scalar_lengths is not None and not np.array_equal(
            np.asarray(scalar_lengths, dtype=np.int64), streamline_lengths
        ):
            raise ValueError(f"Scalar {name!r} does not match streamline lengths.")
        values = getattr(sequence, "_data", None)
        if values is not None:
            if not _array_is_finite(np.asarray(values)):
                raise ValueError(f"Scalar {name!r} contains non-finite values.")
            continue
        for index, item in enumerate(sequence):
            array = np.asarray(item)
            if len(array) != streamline_lengths[index]:
                raise ValueError(
                    f"Scalar {name!r} does not match streamline {index}."
                )
            if not _array_is_finite(array):
                raise ValueError(f"Scalar {name!r} contains non-finite values.")


def _validate_and_compute_bboxes(
    streamlines: Any,
    affine: np.ndarray,
    scalar_data: Dict[str, Any],
    cached_bboxes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Validate one loaded tractogram and return authoritative bounding boxes."""
    _validate_affine(affine, "Tractogram")
    packed = _packed_streamline_arrays(streamlines)
    if packed is None:
        bboxes = []
        lengths = []
        for index, streamline in enumerate(streamlines):
            coordinates = np.asarray(streamline)
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError(
                    f"Streamline {index} must have shape (N, 3), "
                    f"got {coordinates.shape}."
                )
            if len(coordinates) == 0:
                raise ValueError(f"Tractogram contains an empty streamline at index {index}.")
            if len(coordinates) == 1:
                raise ValueError(
                    f"Tractogram contains a single-point streamline at index {index}."
                )
            if not _array_is_finite(coordinates):
                raise ValueError(
                    f"Tractogram contains non-finite coordinates at index {index}."
                )
            minimum = np.min(coordinates, axis=0)
            maximum = np.max(coordinates, axis=0)
            bboxes.append([minimum, maximum])
            lengths.append(len(coordinates))
        _validate_scalar_data(scalar_data, np.asarray(lengths, dtype=np.int64))
        return np.asarray(bboxes, dtype=np.float32)

    data, offsets, lengths = packed
    _validate_scalar_data(scalar_data, lengths)
    if cached_bboxes is None:
        bboxes = _compute_bboxes_prevalidated(
            data,
            offsets,
            lengths,
        )
        nonfinite = np.flatnonzero(~np.all(np.isfinite(bboxes), axis=(1, 2)))
        if nonfinite.size:
            raise ValueError(
                "Tractogram contains non-finite coordinates at index "
                f"{nonfinite[0]}."
            )
        return bboxes

    status = _validate_streamlines_numba(
        data,
        offsets,
        lengths,
        _validated=True,
    )
    nonfinite = np.flatnonzero(status == 1)
    if nonfinite.size:
        raise ValueError(
            f"Tractogram contains non-finite coordinates at index {nonfinite[0]}."
        )
    return cached_bboxes

_TRX_BBOX_KEY = "_tractedit_bboxes"
_TRX_BBOX_CACHE_VERSION = 1
_TRX_BBOX_CACHE_VERSION_KEY = "TRACTEDIT_BBOX_CACHE_VERSION"
_TRX_BBOX_CACHE_DIGEST_KEY = "TRACTEDIT_BBOX_CACHE_DIGEST"
_TRX_BBOX_GEOMETRY_DIGEST_KEY = "TRACTEDIT_BBOX_GEOMETRY_SAMPLE_DIGEST"
_TRX_BBOX_SAMPLE_STREAMLINES = 64
_TRX_BBOX_SAMPLE_POINTS = 16
_TRX_BBOX_HEADER_KEYS = (
    _TRX_BBOX_CACHE_VERSION_KEY,
    _TRX_BBOX_CACHE_DIGEST_KEY,
    _TRX_BBOX_GEOMETRY_DIGEST_KEY,
)


# ============================================================================
# TRX Bbox Caching Helpers (Phase 1)
# ============================================================================


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=limit, dtype=np.int64)


def _bbox_cache_digest(flat_bboxes: np.ndarray) -> str:
    canonical = np.ascontiguousarray(flat_bboxes, dtype="<f4")
    checksum = zlib.crc32(np.asarray(canonical.shape, dtype="<u8").tobytes())
    checksum = zlib.crc32(canonical, checksum)
    return f"{checksum:08x}"


def _bbox_geometry_sample_digest(
    streamlines: Any,
    n_streamlines: int,
) -> Optional[str]:
    if streamlines is None or len(streamlines) != n_streamlines:
        return None

    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([n_streamlines], dtype="<u8").tobytes())
    streamline_indices = _sample_indices(
        n_streamlines,
        _TRX_BBOX_SAMPLE_STREAMLINES,
    )
    for streamline_index in streamline_indices:
        points = np.asarray(streamlines[int(streamline_index)])
        if points.ndim != 2 or points.shape[1] != 3:
            return None
        point_indices = _sample_indices(len(points), _TRX_BBOX_SAMPLE_POINTS)
        digest.update(
            np.asarray(
                [int(streamline_index), len(points)],
                dtype="<u8",
            ).tobytes()
        )
        digest.update(np.asarray(point_indices, dtype="<u8").tobytes())
        digest.update(np.ascontiguousarray(points[point_indices], dtype="<f4"))
    return digest.hexdigest()


def _discard_cached_bboxes(trx_obj: "tbx.TrxFile") -> None:
    data_per_streamline = getattr(trx_obj, "data_per_streamline", None)
    if data_per_streamline is not None:
        data_per_streamline.pop(_TRX_BBOX_KEY, None)
    header = getattr(trx_obj, "header", None)
    if header is not None:
        for key in _TRX_BBOX_HEADER_KEYS:
            header.pop(key, None)


def _try_load_cached_bboxes(
    trx_obj: "tbx.TrxFile",
    n_streamlines: int,
    *,
    validate_geometry: bool = False,
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
        validate_geometry: Recompute every bounding box and require exact
            equality. The default keeps the bounded cache-hit path.

    Returns:
        Bounding boxes as ``np.ndarray`` of shape ``(N, 2, 3)``
        and dtype ``float32``, or ``None`` if the cache is absent,
        corrupted, or has the wrong shape.
    """
    try:
        dps = getattr(trx_obj, "data_per_streamline", None)
        if dps is None or _TRX_BBOX_KEY not in dps:
            return None

        header = getattr(trx_obj, "header", None)
        if (
            header is None
            or header.get(_TRX_BBOX_CACHE_VERSION_KEY) != _TRX_BBOX_CACHE_VERSION
        ):
            logger.warning(
                "TRX bbox cache: unsupported or missing schema, recomputing."
            )
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

        if not np.issubdtype(raw.dtype, np.number):
            logger.warning("TRX bbox cache: nonnumeric data, recomputing.")
            return None

        flat = np.ascontiguousarray(raw, dtype=np.float32)
        cached = flat.reshape(-1, 2, 3)
        if not np.all(np.isfinite(cached)):
            logger.warning("TRX bbox cache: nonfinite bounds, recomputing.")
            return None
        if not np.all(cached[:, 0] <= cached[:, 1]):
            logger.warning("TRX bbox cache: inverted bounds, recomputing.")
            return None

        expected_cache_digest = header.get(_TRX_BBOX_CACHE_DIGEST_KEY)
        if (
            not isinstance(expected_cache_digest, str)
            or _bbox_cache_digest(flat) != expected_cache_digest
        ):
            logger.warning("TRX bbox cache: payload provenance mismatch, recomputing.")
            return None

        expected_geometry_digest = header.get(_TRX_BBOX_GEOMETRY_DIGEST_KEY)
        geometry_digest = _bbox_geometry_sample_digest(
            getattr(trx_obj, "streamlines", None),
            n_streamlines,
        )
        if (
            not isinstance(expected_geometry_digest, str)
            or geometry_digest != expected_geometry_digest
        ):
            logger.warning("TRX bbox cache: geometry provenance mismatch, recomputing.")
            return None

        if validate_geometry:
            streamlines = trx_obj.streamlines
            computed = _compute_bboxes_numba(
                streamlines._data,
                streamlines._offsets,
                streamlines._lengths,
            )
            if not np.array_equal(cached, computed):
                logger.warning(
                    "TRX bbox cache: full geometry validation failed, recomputing."
                )
                return None

        logger.info(
            "TRX: loaded cached bounding boxes from "
            "data_per_streamline (%d streamlines, skipped computation).",
            n_streamlines,
        )
        return cached

    except (AttributeError, ValueError, KeyError, TypeError, IndexError) as exc:
        logger.warning("TRX bbox cache: failed to read (%s), recomputing.", exc)
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
        streamlines = getattr(trx_obj, "streamlines", None)
        n_streamlines = len(streamlines) if streamlines is not None else -1
        if bboxes.shape != (n_streamlines, 2, 3):
            raise ValueError(f"expected ({n_streamlines}, 2, 3), got {bboxes.shape}")
        flat = np.ascontiguousarray(bboxes.reshape(-1, 6), dtype=np.float32)
        cached = flat.reshape(-1, 2, 3)
        if not np.all(np.isfinite(cached)) or not np.all(cached[:, 0] <= cached[:, 1]):
            raise ValueError("bounding boxes must be finite and ordered")
        geometry_digest = _bbox_geometry_sample_digest(streamlines, n_streamlines)
        if geometry_digest is None:
            raise ValueError("streamline geometry is unavailable")
        header = getattr(trx_obj, "header", None)
        if header is None:
            raise ValueError("TRX header is unavailable")

        trx_obj.data_per_streamline[_TRX_BBOX_KEY] = flat
        header[_TRX_BBOX_CACHE_VERSION_KEY] = _TRX_BBOX_CACHE_VERSION
        header[_TRX_BBOX_CACHE_DIGEST_KEY] = _bbox_cache_digest(flat)
        header[_TRX_BBOX_GEOMETRY_DIGEST_KEY] = geometry_digest
        logger.debug(
            "TRX: embedded cached bounding boxes (%d streamlines).",
            flat.shape[0],
        )
    except (AttributeError, ValueError, TypeError, IndexError) as exc:
        _discard_cached_bboxes(trx_obj)
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
    if main_window.streamline_bboxes is None or not main_window.visible_indices:
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
    without Python-level iteration, retains intersecting source groups, then
    embeds cached bounding boxes as ``data_per_streamline`` before saving.

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

    keep_groups = bool(getattr(trx_source, "groups", None))
    trx_subset = None
    try:
        trx_subset = trx_source.select(
            indices_arr, keep_group=keep_groups, copy_safe=True
        )

        if bboxes is not None:
            try:
                saved_bboxes = bboxes[indices_arr]
                _embed_cached_bboxes(trx_subset, saved_bboxes)
            except (ValueError, IndexError, TypeError) as exc:
                logger.warning(
                    "TRX native save: failed to embed bboxes (%s). "
                    "Saving continues without bbox cache.",
                    exc,
                )
        else:
            _discard_cached_bboxes(trx_subset)

        transactional_save(
            output_path,
            lambda staged_path: tbx.save(trx_subset, staged_path),
        )
    finally:
        if trx_subset is not None:
            close = getattr(trx_subset, "close", None)
            if callable(close):
                close()
    logger.info(
        "File saved successfully (TRX, native path): %s",
        os.path.basename(output_path),
    )
    return f"File saved successfully (TRX): {os.path.basename(output_path)}"


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
        reference_grid = _main_window_reference_grid(main_window)
        if out_ext.lower() == ".trk":
            header = _prepare_trk_header(
                main_window.original_trk_header,
                1,
                main_window.anatomical_image_affine,
                reference_grid=reference_grid,
            )
        elif out_ext.lower() == ".tck":
            header = _prepare_tck_header(main_window.original_trk_header, 1)
        elif out_ext.lower() == ".trx":
            header = _prepare_trx_header(
                main_window.original_trk_header,
                1,
                anatomical_img_affine=main_window.anatomical_image_affine,
                reference_grid=reference_grid,
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
    if (
        method == "centroid"
        and len(main_window.visible_indices) > CENTROID_MAX_STREAMLINES
    ):
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
        thread.trx_owner = getattr(main_window, "trx_file_reference", None)
        thread.progress_dialog = progress
        generation = _register_background_worker(
            main_window,
            "_medoid_thread",
            "_medoid_generation",
            thread,
        )

        def on_progress(val, msg):
            if (
                _worker_is_current(
                    main_window,
                    "_medoid_thread",
                    "_medoid_generation",
                    thread,
                    generation,
                )
                and not progress.wasCanceled()
            ):
                progress.setValue(val)
                progress.setLabelText(msg)

        def on_result(idx):
            if not _worker_is_current(
                main_window,
                "_medoid_thread",
                "_medoid_generation",
                thread,
                generation,
            ):
                progress.close()
                visible_streamlines.clear()
                thread.complete_result()
                _release_finished_worker(main_window, "_medoid_thread", thread)
                QTimer.singleShot(0, lambda: _release_deferred_trx_owners(main_window))
                return
            thread.begin_result()
            try:
                progress.close()
                if idx == -1:
                    status_updater("Medoid calculation cancelled.")
                else:
                    result_streamline = visible_streamlines[idx]
                    _finalize_statistic_save(main_window, result_streamline, method)
            finally:
                result_streamline = None
                visible_streamlines.clear()
                thread.complete_result()
                _release_finished_worker(main_window, "_medoid_thread", thread)
                QTimer.singleShot(0, lambda: _release_deferred_trx_owners(main_window))

        def on_error(msg):
            if not _worker_is_current(
                main_window,
                "_medoid_thread",
                "_medoid_generation",
                thread,
                generation,
            ):
                progress.close()
                visible_streamlines.clear()
                thread.complete_result()
                _release_finished_worker(main_window, "_medoid_thread", thread)
                QTimer.singleShot(0, lambda: _release_deferred_trx_owners(main_window))
                return
            thread.begin_result()
            try:
                progress.close()
                QMessageBox.critical(
                    main_window, "Error", f"Medoid calculation failed:\n{msg}"
                )
            finally:
                visible_streamlines.clear()
                thread.complete_result()
                _release_finished_worker(main_window, "_medoid_thread", thread)
                QTimer.singleShot(0, lambda: _release_deferred_trx_owners(main_window))

        def on_thread_finished():
            """Clean up after QThread has fully stopped."""
            _release_finished_worker(main_window, "_medoid_thread", thread)
            _release_deferred_trx_owners(main_window)

        thread.progress.connect(on_progress)
        thread.result_ready.connect(on_result, type=Qt.ConnectionType.QueuedConnection)
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

        _validate_volume_geometry(img.dataobj, img.affine, "Anatomical image")

        img = _canonicalize_image(img, input_path, status_updater)

        # Use auto-downsampling for large images
        image_data, image_affine, was_downsampled = _maybe_downsample_image(img)

        # Create memory-mapped accessor for full-resolution 2D slicing
        mmap_image = MemoryMappedImage(img)

        _validate_volume_geometry(image_data, image_affine, "Anatomical image")

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

            _validate_volume_geometry(img.dataobj, img.affine, "ROI")

            img = _canonicalize_image(img, input_path, status_updater)

            # Apply proxy scaling exactly once, retaining the effective dtype.
            # Own a writable array for editing; never round or narrow values.
            image_data = np.asarray(img.dataobj).copy()
            image_affine = img.affine

            _validate_volume_geometry(image_data, image_affine, "ROI")

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
    loader_thread.progress_dialog = progress
    generation = _register_background_worker(
        main_window,
        "_loader_thread",
        "_bundle_load_generation",
        loader_thread,
    )
    completed = False

    def on_progress(val, msg):
        if not _worker_is_current(
            main_window,
            "_loader_thread",
            "_bundle_load_generation",
            loader_thread,
            generation,
        ):
            return
        progress.setValue(val)
        progress.setLabelText(msg)

    def on_error(msg):
        nonlocal completed
        if not _worker_is_current(
            main_window,
            "_loader_thread",
            "_bundle_load_generation",
            loader_thread,
            generation,
        ):
            progress.close()
            loader_thread.complete_result()
            _release_finished_worker(main_window, "_loader_thread", loader_thread)
            return
        completed = True
        loader_thread.begin_result()
        progress.close()
        QMessageBox.critical(main_window, "Load Error", f"Error loading file:\n{msg}")
        loader_thread.complete_result()
        _release_finished_worker(main_window, "_loader_thread", loader_thread)

    def on_finished(data: "StreamlineLoadResult") -> None:
        nonlocal completed
        if completed or not _worker_is_current(
            main_window,
            "_loader_thread",
            "_bundle_load_generation",
            loader_thread,
            generation,
        ):
            progress.close()
            loader_thread.discard_result(data)
            _release_finished_worker(main_window, "_loader_thread", loader_thread)
            return

        completed = True
        loader_thread.begin_result()
        new_trx_owner = loader_thread.take_trx_owner(data)
        streamlines = data.get("streamlines")
        if streamlines is None or len(streamlines) == 0:
            if new_trx_owner is not None:
                try:
                    new_trx_owner.close()
                except (OSError, RuntimeError, AttributeError):
                    logger.warning("Failed to close an empty TRX result.")
            progress.close()
            QMessageBox.information(
                main_window, "Load Info", "No streamlines found in file."
            )
            loader_thread.complete_result()
            _release_finished_worker(main_window, "_loader_thread", loader_thread)
            return

        previous_state = {
            name: getattr(main_window, name)
            for name in _BUNDLE_STATE_ATTRIBUTES
            if hasattr(main_window, name)
        }
        previous_trx_owner = previous_state.get("trx_file_reference")
        previous_image = previous_state.get("anatomical_image_data")
        old_mmap = previous_state.get("anatomical_mmap_image")
        action_state = {}
        for action_name in (
            "color_default_action",
            "color_orientation_action",
            "color_scalar_action",
        ):
            action = getattr(main_window, action_name, None)
            if action is not None:
                action_state[action_name] = action.isChecked()
        skip_state = {}
        skip_checkbox = getattr(main_window, "skip_checkbox", None)
        if skip_checkbox is not None:
            skip_state["checked"] = skip_checkbox.isChecked()
        skip_spinbox = getattr(main_window, "skip_spinbox", None)
        if skip_spinbox is not None:
            skip_state["enabled"] = skip_spinbox.isEnabled()
            skip_state["value"] = skip_spinbox.value()
        clear_image = (
            not keep_image
            and previous_state.get("tractogram_data") is not None
            and previous_image is not None
        )
        committed = False
        try:
            if clear_image:
                main_window.anatomical_image_data = None
                main_window.anatomical_image_affine = None
                main_window.anatomical_image_path = None
                main_window.anatomical_mmap_image = None
                main_window.anatomical_reference_grid = None
                main_window.image_is_visible = True
                main_window.vtk_panel.clear_anatomical_slices()

            # Apply Data to MainWindow
            main_window.tractogram_data = streamlines
            main_window._tractogram_data_version = (
                getattr(main_window, "_tractogram_data_version", 0) + 1
            )
            main_window.streamline_bboxes = data["bboxes"]

            main_window.original_trk_header = data["header"]
            main_window.original_trk_affine = data["affine"]
            main_window.original_trk_path = data["path"]
            main_window.original_file_extension = data["ext"]
            main_window.tractogram_reference_grid = data.get("reference_grid")

            main_window.trx_file_reference = new_trx_owner
            main_window.scalar_data_per_point = data["scalars"]
            main_window.data_per_streamline = data["data_per_streamline"]
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
            if hasattr(main_window, "bundle_is_visible"):
                main_window.bundle_is_visible = True
            if hasattr(main_window, "render_as_tubes"):
                main_window.render_as_tubes = False
            if hasattr(main_window, "scalar_range_initialized"):
                main_window.scalar_range_initialized = False

            # Auto Skip Calculation
            should_render = True
            if hasattr(main_window, "_auto_calculate_skip_level"):
                # Reset user override so auto-calc works for the new bundle
                main_window._skip_user_disabled = False
                main_window._auto_calculate_skip_level()
                should_render = False
            if loader_thread.is_cancelled:
                raise RuntimeError("Bundle load cancelled.")

            # Finalize UI + first VTK render
            status_msg = (
                f"Loaded {len(main_window.tractogram_data)} streamlines from "
                f"{os.path.basename(data['path'])}"
            )
            _update_vtk_and_ui_after_load(main_window, status_msg, render=should_render)
            main_window.vtk_panel.update_highlight()
            if loader_thread.is_cancelled:
                raise RuntimeError("Bundle load cancelled.")

            progress.setValue(100)
            committed = True
        except Exception as e:
            # Restore authoritative state before any UI or VTK recovery, which
            # can itself allocate or fail during an out-of-memory condition.
            for name, value in previous_state.items():
                setattr(main_window, name, value)
            logger.error("Error in on_finished: %s", e, exc_info=True)
            try:
                progress.close()
            except Exception:
                logger.debug("Failed to close rejected bundle progress dialog.")
            for action_name, checked in action_state.items():
                try:
                    action = getattr(main_window, action_name)
                    was_blocked = action.blockSignals(True)
                    try:
                        action.setChecked(checked)
                    finally:
                        action.blockSignals(was_blocked)
                except Exception:
                    logger.debug("Failed to restore bundle action %s.", action_name)
            if skip_checkbox is not None:
                try:
                    was_blocked = skip_checkbox.blockSignals(True)
                    try:
                        skip_checkbox.setChecked(skip_state["checked"])
                    finally:
                        skip_checkbox.blockSignals(was_blocked)
                except Exception:
                    logger.debug("Failed to restore bundle skip checkbox.")
            if skip_spinbox is not None:
                try:
                    was_blocked = skip_spinbox.blockSignals(True)
                    try:
                        skip_spinbox.setEnabled(skip_state["enabled"])
                        skip_spinbox.setValue(skip_state["value"])
                    finally:
                        skip_spinbox.blockSignals(was_blocked)
                except Exception:
                    logger.debug("Failed to restore bundle skip level.")
            if getattr(main_window, "vtk_panel", None):
                if clear_image:
                    try:
                        main_window.vtk_panel.update_anatomical_slices()
                    except Exception:
                        logger.debug("Failed to restore the previous image actor.")
                try:
                    main_window.vtk_panel.update_main_streamlines_actor(force=True)
                except Exception:
                    logger.debug("Failed to restore the previous bundle actor.")
                try:
                    main_window.vtk_panel.update_highlight()
                except Exception:
                    logger.debug("Failed to restore the previous highlight actor.")
                if previous_state.get("_inversion_active", False):
                    try:
                        main_window.vtk_panel.update_invert_contour()
                    except Exception:
                        logger.debug("Failed to restore the inversion contour.")
            for update_name in (
                "_update_bundle_info_display",
                "_update_action_states",
                "_update_data_panel_display",
            ):
                update = getattr(main_window, update_name, None)
                if callable(update):
                    try:
                        update()
                    except Exception:
                        logger.debug("Failed to refresh previous bundle UI.")
            if not loader_thread.is_cancelled:
                try:
                    QMessageBox.critical(
                        main_window, "Load Error", f"Error finalizing load:\n{e}"
                    )
                except Exception:
                    logger.debug("Failed to display bundle load error.")
        finally:
            # Ownership was transferred out of the loader. Rejected results
            # must be disposed even if a secondary rollback operation fails.
            try:
                if (
                    not committed
                    and new_trx_owner is not None
                    and new_trx_owner is not previous_trx_owner
                ):
                    try:
                        new_trx_owner.close()
                    except Exception:
                        logger.warning("Failed to close rejected TRX replacement.")
            finally:
                loader_thread.complete_result()
                _release_finished_worker(main_window, "_loader_thread", loader_thread)

        if committed:
            if previous_trx_owner is not new_trx_owner:
                try:
                    _retire_trx_owner(main_window, previous_trx_owner)
                except (OSError, RuntimeError, AttributeError):
                    logger.warning("Failed to retire the previous TRX owner.")
            if clear_image and old_mmap is not None:
                try:
                    old_mmap.clear_cache()
                except (OSError, AttributeError):
                    logger.warning("Failed to clear the previous image cache.")
            if previous_state.get("tractogram_data") is not None:
                remove_odf = getattr(main_window, "_remove_odf_data", None)
                if (
                    callable(remove_odf)
                    and getattr(main_window, "odf_data", None) is not None
                ):
                    try:
                        remove_odf()
                    except (OSError, RuntimeError, ValueError, AttributeError):
                        logger.warning("Failed to reset the previous ODF view.")
            try:
                if hasattr(main_window, "geo_lines_action"):
                    main_window.geo_lines_action.setChecked(True)
                scalar_toolbar = getattr(main_window, "scalar_toolbar", None)
                if scalar_toolbar is not None:
                    scalar_toolbar.setVisible(False)
            except (RuntimeError, AttributeError):
                logger.warning("Failed to reset the previous bundle controls.")

    def on_done():
        _release_finished_worker(main_window, "_loader_thread", loader_thread)

    # Connect Signals
    loader_thread.progress.connect(on_progress)
    loader_thread.error.connect(on_error)
    loader_thread.finished.connect(on_finished, type=Qt.ConnectionType.QueuedConnection)
    progress.canceled.connect(loader_thread.cancel)
    if hasattr(loader_thread, "done"):
        loader_thread.done.connect(on_done, type=Qt.ConnectionType.QueuedConnection)

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
    indices_to_save = sorted(main_window.visible_indices)
    indices_array = np.asarray(indices_to_save, dtype=np.intp)
    streamline_count = len(tractogram)

    streamlines_to_save = [tractogram[index] for index in indices_to_save]

    affine_matrix = main_window.original_trk_affine

    if not isinstance(affine_matrix, np.ndarray) or affine_matrix.shape != (4, 4):
        logger.warning(f"Warning: Affine matrix invalid. Using identity.")
        affine_matrix = np.identity(4)

    data_per_point_to_save = {}
    if main_window.scalar_data_per_point:
        for key, scalar_sequence in main_window.scalar_data_per_point.items():
            if len(scalar_sequence) != streamline_count:
                raise ValueError(
                    f"Per-point metadata field '{key}' has "
                    f"{len(scalar_sequence)} items; expected {streamline_count}."
                )
            selected_scalars = [scalar_sequence[index] for index in indices_to_save]
            if any(
                len(values) != len(streamline)
                for values, streamline in zip(
                    selected_scalars, streamlines_to_save, strict=True
                )
            ):
                raise ValueError(
                    f"Per-point metadata field '{key}' does not match point counts."
                )
            data_per_point_to_save[key] = selected_scalars

    data_per_streamline_to_save = {}
    source_data_per_streamline = getattr(main_window, "data_per_streamline", None)
    if source_data_per_streamline:
        for key, values in source_data_per_streamline.items():
            array = np.asarray(values)
            if array.ndim not in (1, 2) or len(array) != streamline_count:
                raise ValueError(
                    f"Per-streamline metadata field '{key}' has invalid shape "
                    f"{array.shape}; expected {streamline_count} items."
                )
            data_per_streamline_to_save[key] = array[indices_array]

    new_tractogram = nib.streamlines.Tractogram(
        streamlines_to_save,
        data_per_point=data_per_point_to_save if data_per_point_to_save else None,
        data_per_streamline=(
            data_per_streamline_to_save if data_per_streamline_to_save else None
        ),
        affine_to_rasmm=affine_matrix,
    )
    return new_tractogram


def _main_window_reference_grid(main_window: Any) -> Optional[ReferenceGrid]:
    reference_grid = getattr(main_window, "anatomical_reference_grid", None)
    if reference_grid is not None:
        return reference_grid
    reference_grid = getattr(main_window, "tractogram_reference_grid", None)
    if reference_grid is not None:
        return reference_grid
    bboxes = getattr(main_window, "streamline_bboxes", None)
    if bboxes is None or len(bboxes) == 0:
        return None
    return ReferenceGrid.from_bounds(
        np.min(bboxes[:, 0], axis=0),
        np.max(bboxes[:, 1], axis=0),
        affine=getattr(main_window, "original_trk_affine", None),
        provenance="synthetic:bundle-bounds",
    )


def _prepare_trk_header(
    base_header: Dict[str, Any],
    nb_streamlines: int,
    anatomical_img_affine: Optional[np.ndarray] = None,
    reference_grid: Optional[ReferenceGrid] = None,
) -> Dict[str, Any]:
    """
    Prepares and validates the header dictionary for TRK saving.
    If voxel_order is missing in base_header, attempts to derive it from
    anatomical_img_affine, otherwise defaults to 'RAS'.
    """
    header = base_header.copy()
    source_grid = ReferenceGrid.from_header(header, provenance="tractogram-header")
    selected_grid = source_grid or reference_grid
    if selected_grid is not None:
        header.update(selected_grid.header_fields())
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
    reference_grid: Optional[ReferenceGrid] = None,
) -> Dict[str, Any]:
    """
    Prepares the header dictionary for TRX saving.
    Ensures essential reference fields (affine, dimensions) are present.
    """
    header = base_header.copy() if base_header is not None else {}
    source_grid = ReferenceGrid.from_header(header, provenance="tractogram-header")
    selected_grid = source_grid or reference_grid
    if selected_grid is not None:
        header.update(selected_grid.header_fields())
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
                logger.debug(
                    "Failed to compute voxel sizes from affine, using default."
                )
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

    add_vtk_metadata(poly_data, tractogram)

    return poly_data


def _save_tractogram_file(
    tractogram: nib.streamlines.Tractogram,
    header: Dict[str, Any],
    output_path: str,
    file_ext: str,
    bboxes_to_embed: Optional[np.ndarray] = None,
    *,
    _transactional: bool = True,
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
    ensure_metadata_supported(
        file_ext,
        tractogram.data_per_point,
        tractogram.data_per_streamline,
    )

    if _transactional:
        transactional_save(
            output_path,
            lambda staged_path: _save_tractogram_file(
                tractogram,
                header,
                staged_path,
                file_ext,
                bboxes_to_embed=bboxes_to_embed,
                _transactional=False,
            ),
        )
        format_name = file_ext.removeprefix(".").upper()
        return (
            f"File saved successfully ({format_name}): "
            f"{os.path.basename(output_path)}"
        )

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
        reference_grid = ReferenceGrid.from_header(
            header,
            provenance="trx-save-header",
        )
        if reference_grid is None:
            dimensions = header.get("dimensions", (1, 1, 1))
            affine = header.get("voxel_to_rasmm", np.eye(4))
            reference_grid = ReferenceGrid(
                affine=affine,
                shape=dimensions,
                provenance="synthetic:trx-save-fallback",
            )
        streamlines = tractogram.streamlines
        if hasattr(streamlines, "_data"):
            nb_vertices = len(streamlines._data)
        else:
            nb_vertices = sum(len(streamline) for streamline in streamlines)
        trx_reference = reference_grid.trx_reference(
            nb_vertices=nb_vertices,
            nb_streamlines=len(streamlines),
        )

        trx_obj_to_save = None
        try:
            trx_obj_to_save = tbx.TrxFile.from_lazy_tractogram(
                tractogram,
                trx_reference,
            )

            _embed_cached_bboxes(trx_obj_to_save, bboxes_to_embed)
            tbx.save(trx_obj_to_save, output_path)
        finally:
            if trx_obj_to_save is not None:
                close = getattr(trx_obj_to_save, "close", None)
                if callable(close):
                    close()
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
                data_per_streamline=tractogram.data_per_streamline,
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

        write_vtk_polydata(writer, poly_data, output_path)

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

    trx_source = main_window.trx_file_reference
    source_groups = getattr(trx_source, "groups", None)
    source_data_per_group = getattr(trx_source, "data_per_group", None)
    try:
        ensure_metadata_supported(
            output_ext,
            main_window.scalar_data_per_point,
            getattr(main_window, "data_per_streamline", None),
            source_groups,
            source_data_per_group,
        )
    except ValueError as exc:
        QMessageBox.critical(main_window, "Save Error", str(exc))
        status_updater(f"Error saving file: {os.path.basename(output_path)}")
        return

    # Prepare Data
    status_updater(
        f"Saving {len(main_window.visible_indices)} streamlines to: {os.path.basename(output_path)}..."
    )
    QApplication.processEvents()  # UI update

    # ------------------------------------------------------------------
    # Phase 2: TRX-native fast path
    # ------------------------------------------------------------------
    if output_ext == ".trx" and trx_source is not None:
        try:
            success_msg = _save_trx_native(
                trx_source=trx_source,
                visible_indices=main_window.visible_indices,
                bboxes=main_window.streamline_bboxes,
                output_path=output_path,
            )
            status_updater(success_msg)
            return
        except (OSError, ValueError, TypeError, IndexError, KeyError) as e:
            logger.warning(
                "TRX native save failed, falling back to generic path: %s",
                e,
            )
            if source_groups or source_data_per_group:
                error_msg = (
                    "TRX native save failed. A generic fallback would discard "
                    "group metadata, so no fallback was attempted."
                )
                QMessageBox.critical(main_window, "Save Error", error_msg)
                status_updater(f"Error saving file: {os.path.basename(output_path)}")
                return

    try:
        tractogram = _prepare_tractogram_and_affine(main_window)
        reference_grid = _main_window_reference_grid(main_window)

        header_to_save: Dict[str, Any] = {}
        if output_ext == ".trk":
            header_to_save = _prepare_trk_header(
                main_window.original_trk_header,
                len(tractogram.streamlines),
                anatomical_img_affine=main_window.anatomical_image_affine,
                reference_grid=reference_grid,
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
                reference_grid=reference_grid,
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
