# -*- coding: utf-8 -*-
"""TRX inventory, lazy value inspection and component colouring controls."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableView,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import workers
from ..tractogram_arrays import (
    COLOR_MAPS, DataArray, array_statistics, collect_arrays, colour_range,
)
from ..utils import ColorMode, signals_blocked
from ..workers import CancellableWorkerThread


class ArrayTableModel(QAbstractTableModel):
    """Read only the requested cells, retaining original streamline IDs."""

    def __init__(self, array, ids, current=lambda: True, parent=None):
        super().__init__(parent)
        self.array = array
        self.ids = ids
        self.current = current
        self.point_id = ids[0] if len(ids) else None

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid() or not self.current() or self.array is None:
            return 0
        if self.array.association == "point":
            return (
                len(self.array.values[self.point_id])
                if self.point_id is not None
                else 0
            )
        return len(self.ids)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid() or self.array is None:
            return 0
        if self.array.association == "membership":
            return 1
        return self.array.components + (2 if self.array.association == "point" else 1)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role not in (
            Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole
        ) or self.array is None:
            return None
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        if section == 0:
            return (
                "Streamline index (0-based)"
                if self.array.association != "group" else "Entry"
            )
        if self.array.association == "point":
            if section == 1:
                return "Vertex index (0-based, within streamline)"
            section -= 1
        return "Value" if self.array.components == 1 else f"Component {section}"

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if (
            not index.isValid()
            or not self.current()
            or self.array is None
            or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole)
        ):
            return None
        row, column = index.row(), index.column()
        if row >= self.rowCount() or column >= self.columnCount():
            return None
        array = self.array
        if array.association == "membership":
            return str(int(array.values[self.ids[row]]))
        if array.association == "point":
            if column == 0:
                return str(self.point_id)
            if column == 1:
                return str(row)
            value = np.asarray(array.values[self.point_id])[row].reshape(-1)[column - 2]
        else:
            original = self.ids[row]
            if column == 0:
                return str(original)
            value = np.asarray(array.values[original]).reshape(-1)[column - 1]
        return str(value.item() if isinstance(value, np.generic) else value)

    def clear(self):
        self.beginResetModel()
        self.array = None
        self.ids = ()
        self.endResetModel()


class _Histogram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = None
        self.setMinimumHeight(90)
        self.setAccessibleName("Value distribution")

    def paintEvent(self, event):
        if self.counts is None or not np.any(self.counts):
            return
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.palette().highlight())
        width = self.width() / len(self.counts)
        maximum = int(self.counts.max())
        for index, count in enumerate(self.counts):
            height = int(count) * (self.height() - 4) / maximum
            painter.drawRect(
                int(index * width),
                self.height() - int(height),
                max(1, int(width) - 1),
                int(height),
            )


class _StatisticsWorker(CancellableWorkerThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, array, component, ids, owner, parent, histogram=True):
        super().__init__(parent)
        self.array = array
        self.component = component
        self.ids = ids
        self.trx_owner = owner
        self.histogram = histogram

    def run(self):
        try:
            result = array_statistics(
                self.array,
                self.component,
                self.ids,
                self._cancelled.is_set,
                histogram=self.histogram,
            )
            if result is not None and not self.is_cancelled:
                self.ready.emit(result)
        except (ValueError, TypeError, IndexError, MemoryError, OverflowError) as error:
            if not self.is_cancelled:
                self.failed.emit(str(error))
        finally:
            self.array = None
            self.ids = ()


class DataInspector(QDialog):
    def __init__(self, panel, array):
        super().__init__(panel.mw)
        self.panel = panel
        self.array = array
        self.source = panel.source
        self.worker = None
        self._closed = False
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Inspect values — {array.name}")
        self.resize(760, 520)
        layout = QVBoxLayout(self)
        description = QLabel(panel.description(array))
        description.setTextFormat(Qt.TextFormat.PlainText)
        description.setWordWrap(True)
        layout.addWidget(description)
        controls = QHBoxLayout()
        self.scope = QComboBox()
        self.scope.addItems(["All", "Visible", "Selected visible"])
        self.scope.setAccessibleName("Inspection scope")
        self.scope.setToolTip(
            "All includes every streamline in the loaded source. Visible uses "
            "logical visibility after deletions and filters; Selected visible "
            "also requires selection. Both include streamlines omitted from "
            "rendering by Skip and do not depend on hiding the bundle."
        )
        self.scope.setCurrentIndex(1)
        self.scope.setEnabled(array.association in ("point", "streamline"))
        controls.addWidget(QLabel("Scope:"))
        controls.addWidget(self.scope)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        controls.addWidget(refresh)
        self.point_id = QSpinBox()
        self.point_id.setAccessibleName("Streamline index (0-based)")
        self.point_id.setToolTip(
            "Index in the loaded source, starting at 0; not a persistent "
            "identifier across files or exports."
        )
        self.point_id.setRange(0, max(0, len(panel.mw.tractogram_data) - 1))
        self.point_id.setVisible(array.association == "point")
        point_label = QLabel("Streamline index (0-based):")
        point_label.setVisible(array.association == "point")
        controls.addWidget(point_label)
        controls.addWidget(self.point_id)
        layout.addLayout(controls)
        self.model = ArrayTableModel(array, (), self.current, self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setDefaultSectionSize(125)
        layout.addWidget(self.table, 1)
        statistics_controls = QHBoxLayout()
        self.component = QSpinBox()
        self.component.setRange(1, array.components)
        self.component.setAccessibleName("Histogram component")
        self.component.setToolTip(
            "Components are numbered from 1 in the interface; "
            "source component indices start at 0."
        )
        statistics_controls.addWidget(QLabel("Component:"))
        statistics_controls.addWidget(self.component)
        compute = QPushButton("Compute statistics")
        compute.setEnabled(array.association != "membership")
        compute.clicked.connect(self.compute_statistics)
        statistics_controls.addWidget(compute)
        cancel = QPushButton("Cancel calculation")
        cancel.clicked.connect(self.cancel_statistics)
        statistics_controls.addWidget(cancel)
        layout.addLayout(statistics_controls)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.histogram = _Histogram()
        layout.addWidget(self.histogram)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.scope.currentIndexChanged.connect(self.refresh)
        self.point_id.valueChanged.connect(self.refresh_points)
        self.component.valueChanged.connect(self.clear_statistics)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_source)
        self.timer.start(200)
        self.refresh()

    def current(self):
        return not self._closed and self.panel.is_current(self.source)

    def check_source(self):
        if not self.current():
            self.close()

    def clear_statistics(self):
        self.cancel_statistics()
        self.histogram.counts = None
        self.histogram.update()
        self.summary.setText(
            "Statistics are calculated on request. Streamline indices are "
            "0-based positions in the loaded source, not persistent identifiers."
        )

    def refresh(self):
        if not self.current():
            return
        self.clear_statistics()
        self.model.beginResetModel()
        self.model.ids = self.panel.scope_ids(self.array, self.scope.currentText())
        self.model.point_id = self.model.ids[0] if len(self.model.ids) else None
        self.model.endResetModel()
        if self.model.point_id is not None:
            with signals_blocked(self.point_id):
                self.point_id.setValue(self.model.point_id)
        if not len(self.model.ids):
            self.summary.setText("No entries in this scope.")

    def refresh_points(self):
        self.model.beginResetModel()
        value = self.point_id.value()
        self.model.point_id = value if value in self.model.ids else None
        self.model.endResetModel()

    def cancel_statistics(self):
        if self.worker is not None:
            self.worker.cancel()
            self.worker = None
            self.summary.setText("Calculation cancelled.")

    def compute_statistics(self):
        if not self.current():
            return
        self.cancel_statistics()
        self.summary.setText("Calculating…")
        self.worker = self.panel.statistics(
            self.array,
            self.component.value() - 1,
            self.model.ids,
            self.show_statistics,
            self.summary.setText,
        )

    def show_statistics(self, result):
        if not self.current() or self.worker is None or self.worker.is_cancelled:
            return
        if result["count"]:
            low = result["min"]
            high = result["max"]
            low_text = str(low) if isinstance(low, int) else f"{low:.8g}"
            high_text = str(high) if isinstance(high, int) else f"{high:.8g}"
            self.summary.setText(
                f"Finite values: {result['count']:,} · Non-finite values: "
                f"{result['missing']:,} · Min: {low_text} · "
                f"Max: {high_text}"
            )
        else:
            self.summary.setText(
                f"No finite values · Non-finite values: {result['missing']:,}"
            )
        self.histogram.counts = result["histogram"]
        self.histogram.setToolTip(self.summary.text())
        self.histogram.update()

    def closeEvent(self, event):
        self._closed = True
        self.cancel_statistics()
        self.timer.stop()
        self.model.clear()
        self.array = None
        self.source = None
        if self in self.panel.dialogs:
            self.panel.dialogs.remove(self)
        super().closeEvent(event)


class TractogramDataPanel(QWidget):
    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.mw = mw
        self.source = None
        self.fields = {}
        self.dialogs = []
        self.selected = None
        layout = QFormLayout(self)
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setWordWrap(True)
        layout.addRow(self.title)
        self.component = QSpinBox()
        self.component.setAccessibleName("Array component")
        self.component.valueChanged.connect(self.cancel_colour)
        layout.addRow("Component:", self.component)
        self.colormap = QComboBox()
        self.colormap.addItems(COLOR_MAPS)
        self.colormap.setAccessibleName("Array colour map")
        self.colormap.currentIndexChanged.connect(self.cancel_colour)
        layout.addRow("Colour map:", self.colormap)
        self.colour = QPushButton("Color by this data")
        self.colour.clicked.connect(self.color_by_data)
        layout.addRow(self.colour)
        inspect = QPushButton("Inspect values…")
        inspect.clicked.connect(self.inspect_values)
        layout.addRow(inspect)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addRow(self.message)
        self.hide()

    def is_current(self, source):
        return (
            source is not None
            and source[0] is self.mw.tractogram_data
            and source[1] is self.mw.trx_file_reference
            and not getattr(self.mw, "_shutdown_requested", False)
        )

    def sync(self):
        if not self.is_current(self.source):
            for dialog in self.dialogs[:]:
                dialog.close()
            self.cancel_colour()
            self.selected = None
            self.hide()
        self.source = (self.mw.tractogram_data, self.mw.trx_file_reference)

    def retire(self, owner):
        for dialog in self.dialogs[:]:
            if dialog.source is not None and dialog.source[1] is owner:
                dialog.close()
        if self.source is not None and self.source[1] is owner:
            self.cancel_colour()
            self.fields.clear()
            self.selected = None
            self.source = None
            self.hide()

    @staticmethod
    def key(array):
        return array.association, array.group, array.name

    @staticmethod
    def description(array):
        rows, components = array.shape
        association = {
            "point": "Per vertex (DPV)",
            "streamline": "Per streamline (DPS)",
            "group": "Per group (DPG)",
            "membership": "Group membership",
        }[array.association]
        return f"{array.name}\n{association} · {rows:,} × {components} · {array.dtype}"

    def build_information(self, parent, arrays=None):
        if arrays is None:
            arrays = collect_arrays(self.mw)
        owner = self.mw.trx_file_reference
        header = (
            owner.header if owner is not None
            else getattr(self.mw, "original_trk_header", None)
        ) or {}
        streamlines = self.mw.tractogram_data
        lengths = getattr(streamlines, "_lengths", None)
        vertices = (
            int(lengths.sum(dtype=np.int64)) if lengths is not None
            else sum(len(line) for line in streamlines)
        )
        info = QTreeWidgetItem(parent, ["Information"])
        rows = {
            "Streamlines": f"{len(streamlines):,}",
            "Visible streamlines": f"{len(self.mw.visible_indices):,}",
            "Vertices": f"{vertices:,}",
            "Data arrays": str(len(arrays)),
            "Groups": str(len(getattr(owner, "groups", {}) or {})),
            "Coordinates": (
                "RAS (mm), assumed; source convention unverified"
                if getattr(self.mw, "original_file_extension", None) in (".vtk", ".vtp")
                else "RAS (mm)"
            ),
        }
        for label, value in rows.items():
            QTreeWidgetItem(info, [f"{label}: {value}"])
        metadata = QTreeWidgetItem(info, ["Header"])
        for name, value in header.items():
            text = (
                np.array2string(value) if isinstance(value, np.ndarray) else str(value)
            )
            item = QTreeWidgetItem(metadata, [f"{name}: {text}"])
            tooltip = (
                np.array2string(value, threshold=value.size)
                if isinstance(value, np.ndarray) else text
            )
            item.setToolTip(0, f"{name}: {tooltip}")
        if not header:
            text = "No header metadata available"
            item = QTreeWidgetItem(metadata, [text])
            item.setToolTip(0, text)
        pending = [(info, ())]
        while pending:
            item, path = pending.pop()
            item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "tractogram_information",
                "source": id(streamlines),
                "row": path,
            })
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            pending.extend(
                (item.child(index), (*path, index))
                for index in range(item.childCount())
            )

    def update_visible_count(self):
        tree = self.mw.data_tree_widget
        for index in range(tree.topLevelItemCount()):
            root = tree.topLevelItem(index)
            if root.text(0) != "Tractogram":
                continue
            for bundle_index in range(root.childCount()):
                bundle = root.child(bundle_index)
                for child_index in range(bundle.childCount()):
                    info = bundle.child(child_index)
                    data = info.data(0, Qt.ItemDataRole.UserRole)
                    if (
                        isinstance(data, dict)
                        and data.get("type") == "tractogram_information"
                        and data.get("source") == id(self.mw.tractogram_data)
                    ):
                        with signals_blocked(tree):
                            info.child(1).setText(
                                0,
                                f"Visible streamlines: {len(self.mw.visible_indices):,}",
                            )
                        return

    def build_tree(self, parent):
        self.sync()
        arrays = collect_arrays(self.mw)
        self.fields = {self.key(array): array for array in arrays}
        self.build_information(parent, arrays)
        owner = self.mw.trx_file_reference
        if any(array.association != "group" for array in arrays):
            root = QTreeWidgetItem(parent, ["Data arrays"])
            root.setExpanded(True)
            for association, label in (
                ("point", "Per vertex (DPV)"),
                ("streamline", "Per streamline (DPS)"),
            ):
                fields = [array for array in arrays if array.association == association]
                if fields:
                    group = QTreeWidgetItem(root, [label])
                    group.setExpanded(len(fields) <= 10)
                    for array in fields:
                        self.add_field(group, array)
        groups = getattr(owner, "groups", {}) or {}
        if groups:
            root = QTreeWidgetItem(parent, ["Groups"])
            for name, values in groups.items():
                array = DataArray("membership", name, values, name)
                self.fields[self.key(array)] = array
                item = self.add_field(
                    root, array, f"{name} · {len(values):,} streamlines"
                )
                for field in arrays:
                    if field.association == "group" and field.group == name:
                        self.add_field(item, field)
        pending = [parent.child(index) for index in range(parent.childCount())]
        while pending:
            item = pending.pop()
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            pending.extend(item.child(index) for index in range(item.childCount()))

    def add_field(self, parent, array, label=None):
        label = label or (
            f"{array.name} · {array.components} "
            f"{'component' if array.components == 1 else 'components'}"
        )
        item = QTreeWidgetItem(parent, [label])
        item.setData(
            0,
            Qt.ItemDataRole.UserRole,
            {
                "type": "tractogram_array",
                "association": array.association,
                "group": array.group,
                "name": array.name,
            },
        )
        item.setToolTip(0, self.description(array))
        return item

    def select(self, data):
        self.hide()
        if not isinstance(data, dict) or data.get("type") != "tractogram_array":
            return False
        self.sync()
        array = self.fields.get((data["association"], data["group"], data["name"]))
        if array is None:
            return True
        self.cancel_colour()
        self.selected = array
        self.title.setText(self.description(array))
        display = getattr(self.mw, "scalar_display", None)
        active = (
            display is not None
            and self.mw.active_scalar_name == array.name
            and display["association"] == array.association
        )
        with signals_blocked(self.component, self.colormap):
            self.component.setRange(1, array.components)
            self.component.setValue(display["component"] + 1 if active else 1)
            self.colormap.setCurrentText(
                display["colormap"] if active else COLOR_MAPS[0]
            )
        self.component.setToolTip(
            f"Components 1–{array.components} correspond to source component "
            f"indices 0–{array.components - 1}."
        )
        self.colour.setEnabled(array.association in ("point", "streamline"))
        self.show()
        return True

    def scope_ids(self, array, scope):
        if array.association not in ("point", "streamline"):
            return range(len(array.values))
        count = len(self.mw.tractogram_data)
        if scope == "All":
            return range(count)
        ids = self.mw.visible_indices
        if scope == "Selected visible":
            ids = ids & self.mw.selected_streamline_indices
        return range(count) if len(ids) == count else sorted(ids)

    def inspect_values(self):
        if self.selected is not None and self.is_current(self.source):
            dialog = DataInspector(self, self.selected)
            self.dialogs.append(dialog)
            dialog.show()

    def statistics(self, array, component, ids, ready, failed, *, histogram=True):
        source = self.source
        previous = getattr(self.mw, "_array_statistics_thread", None)
        if previous is not None:
            for dialog in self.dialogs:
                if dialog.worker is previous:
                    dialog.cancel_statistics()
            if getattr(self, "_colour_worker", None) is previous:
                self.cancel_colour()
                self.message.setText("Colour calculation cancelled.")
        worker = _StatisticsWorker(array, component, ids, source[1], self.mw, histogram)
        generation = workers.register_background_worker(
            self.mw,
            "_array_statistics_thread",
            "_array_statistics_generation",
            worker,
        )

        def current():
            return self.is_current(source) and workers.worker_is_current(
                self.mw,
                "_array_statistics_thread",
                "_array_statistics_generation",
                worker,
                generation,
            )

        worker.ready.connect(
            lambda result: ready(result) if current() else None,
            Qt.ConnectionType.QueuedConnection,
        )
        worker.failed.connect(
            lambda error: failed(error) if current() else None,
            Qt.ConnectionType.QueuedConnection,
        )

        def finished():
            if worker.isRunning():
                QTimer.singleShot(10, finished)
                return
            workers.release_background_worker(
                self.mw, "_array_statistics_thread", worker
            )
            workers.release_deferred_trx_owners(self.mw)
            worker.deleteLater()

        worker.finished.connect(finished, Qt.ConnectionType.QueuedConnection)
        worker.start()
        return worker

    def cancel_colour(self):
        worker = getattr(self, "_colour_worker", None)
        if worker is not None:
            worker.cancel()
        self._colour_worker = None
        if hasattr(self, "message"):
            self.message.clear()

    def color_by_data(self):
        if (
            self.selected is None
            or not self.is_current(self.source)
            or self.selected.association not in ("point", "streamline")
        ):
            return
        self.cancel_colour()
        array = self.selected
        component = self.component.value() - 1
        colormap = self.colormap.currentText()
        self.message.setText("Calculating colour range…")

        def apply(result):
            if not result["count"]:
                self.message.setText("No finite values available for colouring.")
                return
            bounds = colour_range(result["min"], result["max"])
            if bounds is None:
                self.message.setText("The colour scale cannot represent this range.")
                return
            mw = self.mw
            mw.active_scalar_name = array.name
            mw.scalar_display = {
                "association": array.association,
                "component": component,
                "colormap": colormap,
            }
            mw.scalar_data_min, mw.scalar_data_max = bounds
            mw.scalar_min_val = mw.scalar_data_min
            mw.scalar_max_val = mw.scalar_data_max
            mw.scalar_range_initialized = True
            mw.scalar_manager.update_scalar_range_widgets()
            with signals_blocked(mw.color_scalar_action):
                mw.color_scalar_action.setChecked(True)
            if mw.current_color_mode == ColorMode.SCALAR:
                mw.vtk_panel.update_main_streamlines_actor(force=True)
            else:
                mw._set_color_mode(ColorMode.SCALAR)
            self.message.setText(f"Colouring by component {component + 1}.")

        self._colour_worker = self.statistics(
            array,
            component,
            range(len(self.mw.tractogram_data)),
            apply,
            self.message.setText,
            histogram=False,
        )
