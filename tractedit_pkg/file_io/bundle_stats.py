import os
import numpy as np
import nibabel as nib
import logging
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QProgressDialog
from ..workers import CancellableWorkerThread
from typing import Any, Callable, List, Optional
from tractedit_pkg._numba_aot import (
    resample_streamline as _resample_streamline_numba,
    compute_centroid as _compute_centroid_numba,
)
from .bundle_gui import _validate_save_prerequisites
from .tractogram import (
    _main_window_reference_grid,
    _prepare_trk_header,
    _prepare_tck_header,
    _prepare_trx_header,
)
from .anatomical import _status_updater
from .. import file_io

logger = logging.getLogger(__name__)


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
    return file_io._refine_historical_medoid_candidates(
        resampled, totals, sample_indices, batch_size, cancel_check
    )


class MedoidCalculationThread(CancellableWorkerThread):
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

            if self.is_cancelled:
                self._publish_result(-1)
                return

            # Batch Resampling with parallel Numba (10-40%)
            self.progress.emit(10, "Batch resampling (parallel)...")
            resampled = file_io._resample_batch_numba(
                flat_data, offsets, lengths, self.nb_points
            )

            if self.is_cancelled:
                self._publish_result(-1)
                return

            sample_indices = None
            if n > file_io.MEDOID_SAMPLING_THRESHOLD:
                sample_size = min(file_io.MEDOID_SAMPLE_SIZE, n // 3)
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
                cancel_check=self._cancelled.is_set,
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
    status_updater = _status_updater(main_window)

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
        output_path, _ = file_io.QFileDialog.getSaveFileName(
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

        file_io._save_tractogram_file(new_tractogram, header, output_path, out_ext.lower())
        status_updater(f"{method_ui} saved: {os.path.basename(output_path)}")

    except (OSError, ValueError, TypeError) as e:
        logger.error(f"Failed to save {method}: {e}", exc_info=True)
        file_io.QMessageBox.critical(main_window, "Error", f"Failed to save {method}:\n{e}")
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

    status_updater = _status_updater(main_window)

    # Validation
    if not _validate_save_prerequisites(main_window):
        return
    if not main_window.visible_indices:
        file_io.QMessageBox.warning(
            main_window,
            "Calculation Error",
            f"No visible streamlines to calculate {method}.",
        )
        return

    # Safety Check for Medoid
    if method == "medoid" and len(main_window.visible_indices) > 100000:
        file_io.QMessageBox.warning(
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
        file_io.QMessageBox.warning(
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
        file_io.QApplication.processEvents()
        try:
            result_streamline = _compute_centroid_math(visible_streamlines)
            _finalize_statistic_save(main_window, result_streamline, method)
        except (ValueError, IndexError, TypeError, MemoryError) as e:
            logger.error(f"Failed to calculate centroid: {e}", exc_info=True)
            file_io.QMessageBox.critical(
                main_window, "Error", f"Failed to calculate centroid:\n{e}"
            )
            status_updater("Error calculating centroid.")

    elif method == "medoid":
        # Setup Progress Dialog
        progress = file_io.QProgressDialog("Initializing...", "Cancel", 0, 100, main_window)
        progress.setWindowTitle("TractEdit - Medoid Calculation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        # Create Thread
        thread = file_io.MedoidCalculationThread(visible_streamlines)
        thread.trx_owner = getattr(main_window, "trx_file_reference", None)
        thread.progress_dialog = progress
        generation = file_io._register_background_worker(
            main_window,
            "_medoid_thread",
            "_medoid_generation",
            thread,
        )

        def on_progress(val, msg):
            if (
                file_io._worker_is_current(
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
            if not file_io._worker_is_current(
                main_window,
                "_medoid_thread",
                "_medoid_generation",
                thread,
                generation,
            ):
                progress.close()
                visible_streamlines.clear()
                thread.complete_result()
                file_io._release_finished_worker(main_window, "_medoid_thread", thread)
                QTimer.singleShot(0, lambda: file_io._release_deferred_trx_owners(main_window))
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
                file_io._release_finished_worker(main_window, "_medoid_thread", thread)
                QTimer.singleShot(0, lambda: file_io._release_deferred_trx_owners(main_window))

        def on_error(msg):
            if not file_io._worker_is_current(
                main_window,
                "_medoid_thread",
                "_medoid_generation",
                thread,
                generation,
            ):
                progress.close()
                visible_streamlines.clear()
                thread.complete_result()
                file_io._release_finished_worker(main_window, "_medoid_thread", thread)
                QTimer.singleShot(0, lambda: file_io._release_deferred_trx_owners(main_window))
                return
            thread.begin_result()
            try:
                progress.close()
                file_io.QMessageBox.critical(
                    main_window, "Error", f"Medoid calculation failed:\n{msg}"
                )
            finally:
                visible_streamlines.clear()
                thread.complete_result()
                file_io._release_finished_worker(main_window, "_medoid_thread", thread)
                QTimer.singleShot(0, lambda: file_io._release_deferred_trx_owners(main_window))

        def on_thread_finished():
            """Clean up after QThread has fully stopped."""
            file_io._release_finished_worker(main_window, "_medoid_thread", thread)
            file_io._release_deferred_trx_owners(main_window)

        thread.progress.connect(on_progress)
        thread.result_ready.connect(on_result, type=Qt.ConnectionType.QueuedConnection)
        thread.error.connect(on_error)
        # QThread's built-in finished fires after run() has fully exited
        thread.finished.connect(
            on_thread_finished, type=Qt.ConnectionType.QueuedConnection
        )
        progress.canceled.connect(thread.cancel)

        thread.start()
