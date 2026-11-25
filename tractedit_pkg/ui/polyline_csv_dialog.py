# -*- coding: utf-8 -*-
"""Bounded CSV preview and explicit coordinate/layout choices."""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from .. import workers
from ..mesh.polyline_csv import MAX_LINE_BYTES, SIGNATURE, _metadata
from ..workers import CancellableWorkerThread


class _PreviewThread(CancellableWorkerThread):
    ready = pyqtSignal(object)

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def run(self) -> None:
        rows, metadata, error = [], None, ""
        try:
            with Path(self.path).open("rb") as stream:
                for number in range(12):
                    if self._cancelled.is_set():
                        return
                    raw = stream.readline(MAX_LINE_BYTES + 1)
                    if not raw:
                        break
                    if len(raw) > MAX_LINE_BYTES:
                        raise ValueError(f"Line {number + 1}: Line byte limit exceeded")
                    line = raw.decode("utf-8-sig" if number == 0 else "utf-8")
                    if line.strip().startswith(SIGNATURE):
                        if metadata is not None:
                            raise ValueError("Repeated polyline metadata")
                        metadata = _metadata(line.strip()[len(SIGNATURE) :].strip())
                    rows.append(line.rstrip("\r\n"))
        except Exception as exception:
            error = str(exception)
        if not self._cancelled.is_set():
            self.ready.emit((rows, metadata, error))


class PolylineImportDialog(QDialog):
    def __init__(self, parent, paths: tuple[str, ...]) -> None:
        super().__init__(parent)
        self.mw = parent
        self._worker = None
        self.selected_paths = paths
        self.preview_path = next(
            (path for path in paths if Path(path).suffix.lower() != ".vtp"), None,
        )
        self._valid_preview = self.preview_path is None
        layout = QVBoxLayout(self)
        filename = Path(self.preview_path or paths[0]).name
        summary = filename if len(paths) == 1 else f"{len(paths)} files selected"
        if len(paths) > 1 and self.preview_path is not None:
            summary += f" (preview: {filename})"
        selection = QLabel()
        selection.setTextFormat(Qt.TextFormat.PlainText)
        selection.setText(summary)
        selection.setToolTip("\n".join(paths))
        layout.addWidget(selection)
        if self.preview_path is None:
            self.setWindowTitle("Import VTP Polyline")
            layout.addWidget(QLabel("Each VTP line cell becomes a separate polyline."))
        elif any(Path(path).suffix.lower() == ".vtp" for path in paths):
            self.setWindowTitle("Import Polylines")
            layout.addWidget(QLabel("Each VTP uses its own SPACE declaration."))
            layout.addWidget(QLabel("CSV/TXT settings"))
        else:
            self.setWindowTitle("Import CSV/Text Polyline")
        form = QFormLayout()
        if self.preview_path is not None:
            self._add_csv_controls(form)
        self.space_combo = QComboBox()
        if self.preview_path is None:
            self.space_combo.addItem("From file", "from_file")
        else:
            self.space_combo.addItem("Choose coordinate space...", None)
        self.space_combo.addItem("RAS (mm)", "RAS")
        self.space_combo.addItem("LPS (mm)", "LPS")
        form.addRow("Coordinates:", self.space_combo)
        layout.addLayout(form)
        if self.preview_path is not None:
            self.metadata_label = QLabel()
            self.metadata_label.setTextFormat(Qt.TextFormat.PlainText)
            self.metadata_label.setWordWrap(True)
            layout.addWidget(self.metadata_label)
            self.preview = QPlainTextEdit()
            self.preview.setReadOnly(True)
            self.preview.setMaximumBlockCount(12)
            self.preview.setAccessibleName("First twelve file lines")
            layout.addWidget(self.preview)
        note = QLabel(
            "Coordinates must already share the "
            "subject's physical space. RAS/LPS selection performs no registration."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        self.space_combo.currentIndexChanged.connect(self._update_ok)
        self._update_ok()
        if self.preview_path is not None:
            self.resize(560, 420)
            self._start_preview(self.preview_path)

    def _add_csv_controls(self, form: QFormLayout) -> None:
        self.delimiter_combo = QComboBox()
        for label, value in (
            ("Comma", ","),
            ("Semicolon", ";"),
            ("Tab", "\t"),
            ("Whitespace", "whitespace"),
        ):
            self.delimiter_combo.addItem(label, value)
        form.addRow("Delimiter:", self.delimiter_combo)
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("XYZ (indexed header recognized)", None)
        self.layout_combo.addItem("Index, X, Y, Z", True)
        form.addRow("Columns:", self.layout_combo)
        self.extra_columns_checkbox = QCheckBox("Ignore columns after XYZ")
        form.addRow(self.extra_columns_checkbox)
        self.count_checkbox = QCheckBox("Initial point count")
        form.addRow(self.count_checkbox)

    def _start_preview(self, path: str) -> None:
        worker = _PreviewThread(path)
        self._worker = worker
        workers.register_background_worker(
            self.mw,
            "_polyline_preview_thread",
            "_polyline_preview_generation",
            worker,
        )
        worker.ready.connect(self._show_preview, Qt.ConnectionType.QueuedConnection)
        mw = self.mw

        def release():
            worker.wait()
            workers.release_background_worker(mw, "_polyline_preview_thread", worker)
            worker.deleteLater()

        worker.finished.connect(release, Qt.ConnectionType.QueuedConnection)
        worker.start()

    def _show_preview(self, result) -> None:
        worker = self.sender()
        if worker is not self._worker:
            return
        rows, metadata, error = result
        self.preview.setPlainText("\n".join(rows))
        self._valid_preview = not error and bool(rows)
        locked = metadata is not None and len(self.selected_paths) <= 1
        self.space_combo.setEnabled(not locked)
        if locked:
            self.space_combo.setCurrentIndex(self.space_combo.findData("RAS"))
        self.metadata_label.setText(
            error
            or (
                "TractEdit metadata: RAS, mm; presentation retained."
                if locked
                else "Choose RAS or LPS explicitly."
            )
        )
        self._update_ok()

    def _update_ok(self) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self._valid_preview and self.space_combo.currentData() is not None,
        )

    def options(self) -> dict:
        if self.preview_path is None:
            return {"space": self.space_combo.currentData()}
        return {
            "delimiter": self.delimiter_combo.currentData(),
            "indexed": self.layout_combo.currentData(),
            "ignore_extra_columns": self.extra_columns_checkbox.isChecked(),
            "point_count_header": self.count_checkbox.isChecked(),
            "space": self.space_combo.currentData(),
        }

    def done(self, result: int) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        super().done(result)
