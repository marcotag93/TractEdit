# -*- coding: utf-8 -*-
"""Session worker thread and modal save/open orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
import time

from PyQt6.QtCore import QEventLoop, Qt, pyqtSignal

from .. import session_manager
from ...file_dialogs import choose_input
from ...render_batch import defer_render, request_render
from ...session.io import (
    SessionCancelled,
    SessionError,
    SourceResolutionError,
    checkpoint,
    timestamped_path,
)
from ...session.state import SOURCE_FIELDS
from ...workers import CancellableWorkerThread, create_progress_dialog, run_modal_worker

logger = logging.getLogger(__name__)
SAVE_FILE_FILTER = "TractEdit sessions (*.tractedit-session)"
OPEN_FILE_FILTER = "TractEdit sessions (*.tractedit-session *.tractedit-session-*)"


class SessionWorker(CancellableWorkerThread):
    """One cancellable operation; no Qt widgets or VTK work in run()."""

    progress = pyqtSignal(str)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.result = None
        self.error = None
        self.loader = None
        self._last_progress = 0.0

    @property
    def cancelled(self):
        return self.is_cancelled

    def cancel(self):
        super().cancel()
        if self.loader is not None:
            self.loader.cancel()

    def set_loader(self, loader):
        self.loader = loader
        if loader is not None and self.cancelled:
            loader.cancel()

    def report(self, message):
        now = time.monotonic()
        if now - self._last_progress >= 0.05:
            self.progress.emit(message)
            self._last_progress = now

    def run(self):
        try:
            self.result = self.operation(self)
        except Exception as error:
            self.error = error


class SessionManager:
    """Own dialog flow; scene data and the active session path live on MainWindow."""

    def __init__(self, main_window):
        self.mw = main_window

    def _idle(self, *, except_worker=None):
        mw = self.mw
        if (
            getattr(mw, "_session_busy", False)
            or getattr(getattr(mw, "mesh_polyline_manager", None), "io_busy", False)
            or getattr(mw, "_mesh_commit_active", False)
            or getattr(mw, "_mesh_loader_thread", None) is not None
            or getattr(getattr(mw, "mesh_manager", None), "texture_busy", False)
            or any(
                worker.isRunning()
                or getattr(worker, "has_pending_result", False)
                or getattr(worker, "is_consuming_result", False)
                for worker in getattr(mw, "_background_workers", []) if worker is not except_worker
            )
        ):
            raise SessionError("Wait for the current operation to finish")
        if mw.vtk_panel.is_drawing_active:
            raise SessionError(
                "Finish the current ROI drawing before saving or opening a session"
            )

    def _run(self, title, operation, discard_result=None, *, trx_owner=None):
        mw = self.mw
        worker = session_manager.SessionWorker(operation)
        if trx_owner is not None:
            worker.trx_owner = trx_owner
            mw._background_workers.append(worker)
        dialog = create_progress_dialog(
            mw, title, factory=session_manager.QProgressDialog
        )
        dialog.setAutoClose(False)
        dialog.canceled.connect(worker.cancel)
        worker.progress.connect(dialog.setLabelText, Qt.ConnectionType.QueuedConnection)
        mw._session_busy = True
        enabled = mw.isEnabled()
        mw.setEnabled(False)
        try:
            dialog.setEnabled(True)
            run_modal_worker(worker, dialog)
            if worker.error is not None:
                raise worker.error
            if worker.cancelled and discard_result is not None:
                discard_result(worker.result)
                raise SessionCancelled()
            return worker.result
        finally:
            dialog.close()
            dialog.deleteLater()
            mw.setEnabled(enabled)
            mw._session_busy = False
            if trx_owner is not None:
                from ...file_io import _release_deferred_trx_owners

                mw._background_workers.remove(worker)
                _release_deferred_trx_owners(mw)
            worker.deleteLater()

    def save(self):
        try:
            if self.mw.mesh_polyline_manager.active:
                raise SessionError(
                    "Finish or cancel the current polyline before saving a session"
                )
            self._idle()
            base = getattr(self.mw, "session_path", None)
            suggestion = timestamped_path(base or "session.tractedit-session")
            chosen, _ = session_manager.QFileDialog.getSaveFileName(
                self.mw,
                "Save Session",
                str(suggestion),
                SAVE_FILE_FILTER,
            )
            if not chosen:
                return False
            selected = Path(chosen)
            destination = (
                selected
                if selected.name == suggestion.name and not selected.exists()
                else timestamped_path(chosen)
            )
            state = session_manager.capture_state(self.mw)
            views = session_manager.capture_views(self.mw)
            session_manager.validate_state(state)
            session_manager.validate_views(views)

            def operation(worker):
                def cancelled():
                    return worker.cancelled

                sources = session_manager.associate_sources(
                    self.mw, cancelled, worker.report, worker.set_loader
                )
                document = {"sources": sources, "state": state, "views": views}
                session_manager.write_session(
                    destination, document, cancelled, worker.report
                )
                return str(destination)

            self.mw.session_path = self._run("Saving session...", operation)
            for key, identity in state["roi_ids"].items():
                self.mw.roi_layers[key]["session_id"] = identity
            self.mw.vtk_panel.update_status(
                f"Session saved: {Path(self.mw.session_path).name}"
            )
            return True
        except SessionCancelled:
            return False
        except Exception as error:
            logger.exception("Session save failed")
            session_manager.QMessageBox.critical(self.mw, "Session Error", str(error))
            return False

    def open(self, path=None):
        candidate = None
        scene_dialog = None
        locked = False
        sources = {}
        try:
            if self.mw.mesh_polyline_manager.active:
                raise SessionError(
                    "Finish or cancel the current polyline before opening a session"
                )
            mapping = getattr(self.mw, "surface_mapping_manager", None)
            self._idle(except_worker=getattr(mapping, "_worker", None))
            if path is None:
                path, _ = choose_input(
                    session_manager.QFileDialog.getOpenFileName,
                    self.mw, "Open Session", "", OPEN_FILE_FILTER
                )
            if not path:
                return False
            save_first = False
            if (
                any(
                    getattr(self.mw, field) is not None
                    for _, field in SOURCE_FIELDS.values()
                )
                or self.mw.roi_layers
                or self.mw.mesh_layers
                or self.mw.standalone_polylines
            ):
                choice = session_manager.QMessageBox.question(
                    self.mw,
                    "Replace Current Session",
                    "Save the current session before opening another?",
                    session_manager.QMessageBox.StandardButton.Save
                    | session_manager.QMessageBox.StandardButton.Discard
                    | session_manager.QMessageBox.StandardButton.Cancel,
                    session_manager.QMessageBox.StandardButton.Cancel,
                )
                if choice == session_manager.QMessageBox.StandardButton.Cancel:
                    return False
                save_first = choice == session_manager.QMessageBox.StandardButton.Save

            if mapping is not None:
                mapping.stop_for_restore()
            self._idle()
            # TOCTOU recheck after the modal prompt above.
            if self.mw.mesh_polyline_manager.active:
                raise SessionError("Finish the current polyline before opening a session")
            if save_first and not self.save():
                return False

            def operation(worker):
                current = (
                    document
                    if document is not None
                    else session_manager.read_session(
                        path, lambda: worker.cancelled, worker.report, verify=False
                    )
                )
                session_manager.resolve_session_sources(
                    current, path, replacements, lambda: worker.cancelled, worker.report
                )
                return session_manager.prepare_session(current, worker)

            document = None
            replacements = {}
            while True:
                try:
                    document, sources = self._run(
                        "Opening session...",
                        operation,
                        discard_result=lambda result: session_manager.dispose_sources(
                            result[1]
                        ),
                    )
                    break
                except SourceResolutionError as error:
                    document = error.document
                    for role, reason in error.issues.items():
                        label = role
                        if role.startswith("mesh:"):
                            label = next(
                                record["display_name"]
                                for record in document["state"]["mesh_layers"]
                                if "mesh:" + record["id"] == role
                            ) + f" ({role})"
                        replacement, _ = choose_input(
                            session_manager.QFileDialog.getOpenFileName,
                            self.mw,
                            f"Locate {label} source — identical content required\n"
                            f"{reason}",
                            str(Path(path).resolve().parent),
                            "All files (*)",
                        )
                        if not replacement:
                            return False
                        replacements[role] = replacement
            from ...main_window import MainWindow

            previous_enabled = self.mw.isEnabled()
            locked = True
            self.mw._session_busy = True
            self.mw.setEnabled(False)
            scene_dialog = session_manager.QProgressDialog(
                "Preparing scene...", "Cancel", 0, 0, self.mw
            )
            scene_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            scene_dialog.setAutoClose(False)
            scene_dialog.setAutoReset(False)
            scene_dialog.setMinimumDuration(0)
            scene_dialog.setEnabled(True)
            scene_dialog.show()

            def scene_progress(message):
                scene_dialog.setLabelText(message)
                session_manager.QApplication.processEvents()

            candidate = MainWindow()
            candidate._session_busy = True
            with defer_render(candidate.vtk_panel):
                session_manager.populate_window(
                    candidate,
                    document,
                    sources,
                    cancelled=scene_dialog.wasCanceled,
                    progress=scene_progress,
                )
                candidate.session_path = str(Path(path).resolve())
                candidate.show()
                session_manager.restore_mesh_tree(candidate, document["views"])
                session_manager.restore_cameras(candidate, document["views"])
                candidate.vtk_panel.update_status(f"Session restored: {Path(path).name}")
                candidate.vtk_panel._render_all()
                request_render(candidate.vtk_panel, candidate.vtk_panel.render_window)
            checkpoint(scene_dialog.wasCanceled)
            if self.mw.appimage_integration.working:
                scene_dialog.setLabelText("Waiting for desktop integration...")
                loop = QEventLoop()
                self.mw.appimage_integration.idle.connect(loop.quit)
                scene_dialog.canceled.connect(loop.quit)
                if (
                    self.mw.appimage_integration.working
                    and not scene_dialog.wasCanceled()
                ):
                    loop.exec()
                checkpoint(scene_dialog.wasCanceled)
            scene_dialog.close()
            candidate._session_busy = False
            candidate._update_action_states()
            app = session_manager.QApplication.instance()
            app._tractedit_session_window = candidate
            if getattr(app, "_initial_window", None) is self.mw:
                app.release_initial_window()
            candidate = None
            self.mw.hide()
            try:
                self.mw._cleanup_resources()
                self.mw._cleanup_vtk()
            except Exception:
                logger.exception("Could not fully retire the previous session window")
            self.mw.deleteLater()
            app._tractedit_session_window.update_notifier.start()
            app._tractedit_session_window.appimage_integration.start()
            return True
        except SessionCancelled:
            return False
        except Exception as error:
            logger.exception("Session restore failed")
            if scene_dialog is not None:
                scene_dialog.close()
            session_manager.QMessageBox.critical(self.mw, "Session Error", str(error))
            return False
        finally:
            if scene_dialog is not None:
                scene_dialog.close()
                scene_dialog.deleteLater()
            if candidate is not None:
                try:
                    candidate.hide()
                    candidate._cleanup_resources()
                    candidate._cleanup_vtk()
                except Exception:
                    logger.exception("Could not fully dispose the candidate session")
                finally:
                    candidate.deleteLater()
            try:
                session_manager.dispose_sources(sources)
            finally:
                if locked:
                    self.mw._session_busy = False
                    self.mw.setEnabled(previous_enabled)
