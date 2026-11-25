from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_contracts import StreamlineLoadResult

import os
import numpy as np
import nibabel as nib
import trx.trx_file_memmap as tbx
import vtk
import logging
from contextlib import ExitStack
from vtk.util import numpy_support
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QProgressDialog
from ..workers import CancellableWorkerThread
from ..ui.theme_manager import ThemeManager
from ..file_dialogs import choose_input
from ..utils import ColorMode
from ..tractogram_metadata import ensure_metadata_supported, extract_vtk_metadata
from ..reference_grid import ReferenceGrid
from typing import Any, Dict, Optional, Tuple
from .anatomical import _update_vtk_and_ui_after_load
from .bbox_cache import (
    _validate_and_compute_bboxes,
    _TRX_BBOX_KEY,
    _try_load_cached_bboxes,
    _extract_visible_bboxes,
)
from .tractogram import (
    load_tractogram_sync,
    _save_trx_native,
    _main_window_reference_grid,
    _prepare_trk_header,
    _prepare_tck_header,
    _prepare_trx_header,
)
from .anatomical import _status_updater
from .. import file_io

logger = logging.getLogger(__name__)


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
    "scalar_display",
    "manual_visible_indices",
    "visible_indices",
    "_visibility_version",
    "_session_render_override",
    "_rendered_streamline_indices",
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
)


class StreamlineLoaderThread(CancellableWorkerThread):
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
        self._trx_owner: Optional["tbx.TrxFile"] = None
        self._result_published = False

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

            if ext in (".trk", ".tck"):
                self.progress.emit(20, f"Reading {ext[1:].upper()} file...")
                source = load_tractogram_sync(self.input_path)
                tractogram_obj = source.tractogram
                loaded_streamlines = tractogram_obj.streamlines
                results["header"] = source.header.copy() if hasattr(source, "header") else {}

                if self.is_cancelled:
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

                if self.is_cancelled:
                    return
                self.progress.emit(70, "Finalizing...")
                results["bboxes"] = _validate_and_compute_bboxes(
                    loaded_streamlines,
                    aff,
                    scalars,
                )

            elif ext in [".vtk", ".vtp"]:
                self.progress.emit(10, "Reading VTK file...")

                # Forward fine-grained I/O progress from VTK to the Qt dialog
                def _on_reader_progress(caller, event):
                    pct = 10 + int(caller.GetProgress() * 18)  # 0–1 → 10–28 %
                    self.progress.emit(pct, "Reading VTK file...")

                poly_data = load_tractogram_sync(
                    self.input_path, on_vtk_progress=_on_reader_progress
                )
                if self.is_cancelled:
                    return

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

                if self.is_cancelled:
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
                    trx_obj = load_tractogram_sync(self.input_path)
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

                if self.is_cancelled:
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
                if self.is_cancelled:
                    return
                self.progress.emit(30, "Computing bounding boxes...")
                streamlines = trx_obj.streamlines
                n_streamlines = len(streamlines)

                # Phase 1 optimisation
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

            if self.is_cancelled:
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


def load_streamlines_file(
    main_window: Any, *, file_path: Optional[str] = None
) -> None:
    """
    Loads a streamline file using a background thread and a progress bar.
    """
    if not hasattr(main_window, "vtk_panel") or not main_window.vtk_panel.scene:
        file_io.QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
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

        input_path, _ = choose_input(
            file_io.QFileDialog.getOpenFileName,
            main_window, "Select Input Streamline File", start_dir, all_filters
        )
    if not input_path:
        return

    # Setup Progress Dialog (Modal)
    progress = file_io.QProgressDialog("Initializing...", "Cancel", 0, 100, main_window)
    progress.setWindowTitle("Loading Bundle")
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setMinimumWidth(350)

    manager = getattr(main_window, "theme_manager", None)
    progress.setStyleSheet(
        manager.get_progress_dialog_style() if manager else ThemeManager.dialog_style(True)
    )

    progress.setValue(0)
    progress.show()

    # Create and Configure Thread
    loader_thread = file_io.StreamlineLoaderThread(input_path)
    mapping = getattr(main_window, "surface_mapping_manager", None)
    loader_thread.progress_dialog = progress
    generation = file_io._register_background_worker(
        main_window,
        "_loader_thread",
        "_bundle_load_generation",
        loader_thread,
    )
    completed = False

    def on_progress(val, msg):
        if completed or not file_io._worker_is_current(
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
        if not file_io._worker_is_current(
            main_window,
            "_loader_thread",
            "_bundle_load_generation",
            loader_thread,
            generation,
        ):
            progress.close()
            loader_thread.complete_result()
            file_io._release_finished_worker(main_window, "_loader_thread", loader_thread)
            return
        completed = True
        loader_thread.begin_result()
        progress.close()
        file_io.QMessageBox.critical(main_window, "Load Error", f"Error loading file:\n{msg}")
        loader_thread.complete_result()
        file_io._release_finished_worker(main_window, "_loader_thread", loader_thread)

    def on_finished(data: "StreamlineLoadResult") -> None:
        nonlocal completed
        if completed or not file_io._worker_is_current(
            main_window,
            "_loader_thread",
            "_bundle_load_generation",
            loader_thread,
            generation,
        ):
            progress.close()
            loader_thread.discard_result(data)
            file_io._release_finished_worker(main_window, "_loader_thread", loader_thread)
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
            file_io.QMessageBox.information(
                main_window, "Load Info", "No streamlines found in file."
            )
            loader_thread.complete_result()
            file_io._release_finished_worker(main_window, "_loader_thread", loader_thread)
            return

        previous_state = {
            name: getattr(main_window, name)
            for name in _BUNDLE_STATE_ATTRIBUTES
            if hasattr(main_window, name)
        }
        previous_trx_owner = previous_state.get("trx_file_reference")
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
        committed = False
        scene_rollback = ExitStack()
        try:
            main_window.vtk_panel.stage_bundle_scene(scene_rollback)
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
            main_window.scalar_display = None

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
            from ..logic.state_manager import retain_polyline_history

            retain_polyline_history(main_window)
            main_window.current_color_mode = ColorMode.ORIENTATION
            if hasattr(main_window, "bundle_is_visible"):
                main_window.bundle_is_visible = True
            if hasattr(main_window, "render_as_tubes"):
                main_window.render_as_tubes = False
            if hasattr(main_window, "scalar_range_initialized"):
                main_window.scalar_range_initialized = False

            # Auto Skip Calculation
            if hasattr(main_window, "_auto_calculate_skip_level"):
                # Reset user override so auto-calc works for the new bundle
                main_window._skip_user_disabled = False
                main_window._auto_calculate_skip_level(update_actor=False)
            if loader_thread.is_cancelled:
                raise RuntimeError("Bundle load cancelled.")

            # Finalize UI + first VTK render
            status_msg = (
                f"Loaded {len(main_window.tractogram_data)} streamlines from "
                f"{os.path.basename(data['path'])}"
            )
            main_window.vtk_panel.update_main_streamlines_actor(loading=True)
            if getattr(main_window, "parcellation_data", None) is not None:
                main_window.connectivity_manager.stage_bundle_replacement(scene_rollback)
            _update_vtk_and_ui_after_load(main_window, status_msg)
            if loader_thread.is_cancelled:
                raise RuntimeError("Bundle load cancelled.")

            progress.reset()
            scene_rollback.pop_all()
            committed = True
        except Exception as e:
            was_cancelled = loader_thread.is_cancelled
            # Restore authoritative state before any UI or VTK recovery, which
            # can itself allocate or fail during an out-of-memory condition.
            for name, value in previous_state.items():
                setattr(main_window, name, value)
            for name in ("_rendered_streamline_indices", "_session_render_override"):
                if name not in previous_state and hasattr(main_window, name):
                    delattr(main_window, name)
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
                try:
                    scene_rollback.close()
                    main_window.vtk_panel.render_window.Render()
                except Exception:
                    logger.warning("Failed to restore the previous scene.", exc_info=True)
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
            if not was_cancelled:
                try:
                    file_io.QMessageBox.critical(
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
                file_io._release_finished_worker(main_window, "_loader_thread", loader_thread)

        if committed:
            if previous_trx_owner is not new_trx_owner:
                try:
                    file_io._retire_trx_owner(main_window, previous_trx_owner)
                except (OSError, RuntimeError, AttributeError):
                    logger.warning("Failed to retire the previous TRX owner.")
            if mapping is not None:
                mapping.invalidate()
            if getattr(main_window, "odf_data", None) is not None:
                main_window._odf_tunnel_indices = None
                main_window._odf_tunnel_source = None
                main_window._session_deferred_odf_owner = None
                main_window.view_odf_tunnel_action.setChecked(False)
                main_window._toggle_odf_tunnel(False)
            try:
                if hasattr(main_window, "geo_lines_action"):
                    main_window.geo_lines_action.setChecked(True)
                scalar_toolbar = getattr(main_window, "scalar_toolbar", None)
                if scalar_toolbar is not None:
                    scalar_toolbar.setVisible(False)
            except (RuntimeError, AttributeError):
                logger.warning("Failed to reset the previous bundle controls.")

    def on_done():
        file_io._release_finished_worker(main_window, "_loader_thread", loader_thread)

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
        file_io.QMessageBox.warning(main_window, "Save Error", "No streamline data to save.")
        return False
    if main_window.original_trk_affine is None:
        logger.error("Save Error: Original streamline affine info missing.")
        file_io.QMessageBox.critical(
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
        file_io.QMessageBox.critical(
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

    output_path, selected_filter = file_io.QFileDialog.getSaveFileName(
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
        file_io.QMessageBox.warning(
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


def save_streamlines_file(main_window: Any) -> None:
    """
    Saves the current streamlines to a trk, tck, or trx file.
    """
    status_updater = _status_updater(main_window)

    # Pre-checks
    if not _validate_save_prerequisites(main_window):
        return

    # Get Output Path
    output_path, output_ext = file_io._get_save_path_and_extension(main_window)
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
        file_io.QMessageBox.critical(main_window, "Save Error", str(exc))
        status_updater(f"Error saving file: {os.path.basename(output_path)}")
        return

    # Prepare Data
    status_updater(
        f"Saving {len(main_window.visible_indices)} streamlines to: {os.path.basename(output_path)}..."
    )
    file_io.QApplication.processEvents()  # UI update

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
                file_io.QMessageBox.critical(main_window, "Save Error", error_msg)
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

        success_msg = file_io._save_tractogram_file(
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
        file_io.QMessageBox.critical(main_window, "Save Error", error_msg)
        status_updater(f"Error saving file: {os.path.basename(output_path)}")
