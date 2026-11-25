"""Receive Finder document requests during and after GUI startup."""

import os
from pathlib import Path
import subprocess
import sys

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

from . import SESSION_EXTENSION
from .cli import NIFTI_SUFFIXES, gui_document_paths

_DOCUMENT_SUFFIXES = (".trk", ".trx", ".tck", ".vtk", ".vtp", ".obj", SESSION_EXTENSION) + NIFTI_SUFFIXES
_USER_INPUT = frozenset((
    QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress, QEvent.Type.Shortcut,
    QEvent.Type.Drop, QEvent.Type.TouchBegin, QEvent.Type.Wheel,
    QEvent.Type.NativeGesture,
))


def _document_identity(path):
    try:
        stat = os.stat(path)
    except OSError:
        return os.path.abspath(path)
    return stat.st_dev, stat.st_ino


class DocumentApplication(QApplication):
    def __init__(self, argv):
        self._pending = []
        self._initial_window = None
        super().__init__(argv)

    def reserve_initial_window(self, window):
        self._initial_window = window
        window.destroyed.connect(self.release_initial_window)
        self.installEventFilter(self)

    def release_initial_window(self):
        self.removeEventFilter(self)
        self._initial_window = None

    def eventFilter(self, receiver, event):
        if event.type() in _USER_INPUT:
            self.release_initial_window()
        return super().eventFilter(receiver, event)

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen:
            path = event.file()
            if path.lower().endswith(_DOCUMENT_SUFFIXES):
                if self._pending is not None:
                    self._pending.append(path)
                else:
                    self._open_document(path)
                return True
        return super().event(event)

    def finish_startup(self, bundle, anatomical=None):
        seen = {_document_identity(path) for path in (bundle, anatomical) if path}
        pending, self._pending = self._pending, None
        for path in pending:
            identity = _document_identity(path)
            if identity not in seen:
                seen.add(identity)
                self._open_document(path)

    def _open_document(self, path):
        window = self._initial_window
        if window is not None:
            self.release_initial_window()
            if not (
                not window.isVisible() or getattr(window, "_shutdown_requested", False)
                or self.activeModalWidget() or window._background_workers
                or window._session_busy or window.session_path
                or window.unified_undo_stack or window.roi_layers
                or window.tractogram_data is not None
                or window.anatomical_image_data is not None
                or window.parcellation_data is not None or window.odf_data is not None
                or window.mesh_layers or window.standalone_polylines
            ):
                window.load_initial_files(*gui_document_paths(path, None))
                return
        self._open_separately(path)

    def _open_separately(self, path):
        command = [sys.executable]
        if not getattr(sys, "frozen", False):
            command.append(str(Path(__file__).resolve().parents[1] / "main.py"))
        try:
            subprocess.Popen(
                [*command, "--", path],
                env={**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"},
            )
        except OSError as error:
            QMessageBox.warning(None, "Open document", str(error))
