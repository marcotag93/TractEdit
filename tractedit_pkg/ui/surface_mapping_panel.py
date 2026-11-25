# -*- coding: utf-8 -*-
"""Non-modal analysis parameters and presentation of the owned mapping result."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QSize, Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QTreeWidgetItem,
    QWidget,
)

from ..surface_mapping import SurfaceMappingParameters
from ..utils import signals_blocked

if TYPE_CHECKING:
    from ..data_contracts import MeshLayer
    from ..main_window import MainWindow


METHODS = {
    "endpoint_proximity": "Endpoint proximity",
    "surface_intersections": "Surface intersections",
}
SCOPES = {
    "visible": "Visible streamlines",
    "selected_visible": "Selected visible streamlines",
}


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _summary_table(title: str, rows: list[tuple[str, str]]) -> str:
    heading = (
        f'<tr><td colspan="2"><b>{escape(title)}</b></td></tr>' if title else ""
    )
    cells = "".join(
        f'<tr><td>{escape(name)}</td><td align="right">{escape(value)}</td></tr>'
        for name, value in rows
    )
    return (
        '<table width="100%" cellspacing="0" cellpadding="1">'
        + heading + cells + '</table>'
    )


class SurfaceMappingPanel(QWidget):
    def __init__(self, main_window: MainWindow) -> None:
        super().__init__(main_window)
        self.setObjectName("surface_mapping_panel")
        self.hide()
        self.mw = main_window
        self.mesh_id = None
        self._shutdown = False
        self._scope_revision = None
        self._scope_count = 0
        self._recipe_identity = None
        self._job_identity = None
        self._result_identity = None
        self._intersection_distance = 2.0
        self.root_item = None
        self._rows = []
        self._groups = {}
        self._expansion = {}
        self.target_label = _label()
        self.status_label = _label()
        self.method_group = QButtonGroup(self)
        self.method_group.setExclusive(False)
        self.method_buttons = {}
        for value, text in METHODS.items():
            button = QCheckBox(text)
            self.method_group.addButton(button)
            self.method_buttons[value] = button
        self.method_buttons["endpoint_proximity"].setChecked(True)
        self.qc_checkbox = QCheckBox("Diagnostics (QC)")
        self.qc_label = _label()
        self.qc_label.setText("Diagnostics (QC)")
        self.qc_checkbox.setAccessibleName("Diagnostics (QC)")
        self.qc_checkbox.setToolTip(
            "Ambiguous endpoint assignments; contacts, ambiguities "
            "and coplanar overlaps."
        )
        self.qc_label.setBuddy(self.qc_checkbox)
        self.scope_combo = QComboBox()
        self.scope_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.scope_combo.setMinimumContentsLength(8)
        for value, text in SCOPES.items():
            self.scope_combo.addItem(text, value)
        self.scope_label = _label()
        self.distance_edit = QLineEdit("2.0")
        self.distance_label = QLabel("Maximum distance (mm)")
        self.distance_label.setBuddy(self.distance_edit)
        self.distance_edit.setAccessibleName("Maximum distance (mm)")
        self.constraints_label = _label(
            "Inputs must already be aligned in RAS mm. Geometric proximity "
            "does not establish an anatomical termination. Distance is an "
            "operational threshold; 0 allows contact within numerical tolerance."
        )
        self.parameters_label = _label()
        self.run_button = QPushButton("Compute mapping")
        self.cancel_button = QPushButton("Cancel computation")

        self.colors_label = _label(
            "Orange spheres: assigned endpoints. Cyan outlines: crossings. "
            "Magenta: diagnostics, "
            "excluded from the quantitative map."
        )
        self.visibility_label = _label()
        self.result_header = _label()
        self.result_label = _label()
        self.geometry_label = _label()
        self.result_label.setTextFormat(Qt.TextFormat.RichText)
        self.geometry_label.setTextFormat(Qt.TextFormat.RichText)
        self.result_label.setToolTip(
            "Endpoint order: storage\n"
            "Intersection ambiguities: points + intervals\n"
            "Endpoint radius: 0.12 mm\n"
            "Crossing half-width: 0.12 mm\n"
            "Interval radius: 0.06 mm"
        )
        self.method_results = {mode: _label() for mode in METHODS}
        self.sampling_label = _label()
        for label in (*self.method_results.values(), self.sampling_label):
            label.setTextFormat(Qt.TextFormat.RichText)
        self.export_button = QPushButton("Export…")
        self.remove_button = QPushButton("Remove mapping")
        self._controls = [
            *self.method_buttons.values(), self.qc_checkbox, self.scope_combo,
            self.scope_label, self.distance_edit, self.distance_label,
            self.constraints_label, self.parameters_label, self.run_button,
            self.cancel_button, self.colors_label, self.visibility_label,
            self.result_header, self.result_label, self.geometry_label,
            *self.method_results.values(), self.sampling_label, self.status_label,
            self.target_label, self.qc_label,
            self.export_button, self.remove_button,
        ]
        for control in self._controls:
            control.setParent(self)

        manager = self.mw.surface_mapping_manager
        self.method_group.buttonToggled.connect(self._method_changed)
        self.scope_combo.currentIndexChanged.connect(self.refresh)
        self.distance_edit.textChanged.connect(self.refresh)
        self.run_button.clicked.connect(self.run_mapping)
        self.cancel_button.clicked.connect(manager.cancel)
        self.qc_checkbox.toggled.connect(
            lambda enabled: manager.set_display("qc_enabled", enabled)
        )
        self.export_button.clicked.connect(manager.open_export_dialog)
        self.remove_button.clicked.connect(lambda: manager.remove(self.mesh_id))
        manager.changed.connect(self.refresh)
        self.tree = main_window.data_panel_manager.mesh_inspector
        self.tree.model().modelAboutToBeReset.connect(self.detach)
        self.tree.viewport().installEventFilter(self)
        self.refresh()

    def detach(self) -> None:
        if self.root_item is not None:
            self._expansion = {
                name: item.isExpanded() for name, item in self._groups.items()
            }
        for control in self._controls:
            control.setParent(self)
        for item, _ in self._rows:
            self.tree.removeItemWidget(item, 0)
        self.root_item = None
        self._rows.clear()
        self._groups.clear()

    def attach(self, root: QTreeWidgetItem) -> None:
        if self.root_item is root:
            return
        previous = self.root_item
        self.detach()
        if previous is not None:
            previous.takeChildren()
            previous.setChildIndicatorPolicy(
                QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
            )
            previous.setExpanded(False)
        self.root_item = root
        root.takeChildren()
        tree = self.tree

        def group(parent, name):
            item = QTreeWidgetItem(parent, [name])
            item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "surface_mapping_control", "id": self.mesh_id,
            })
            item.setExpanded(self._expansion.get(name, name == "Parameters"))
            self._groups[name] = item
            return item

        def row(parent, control, label=None):
            item = QTreeWidgetItem(parent)
            item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "surface_mapping_control", "id": self.mesh_id,
            })
            host = QWidget()
            layout = QFormLayout(host) if label is not None else QVBoxLayout(host)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            if label is not None:
                layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
                layout.addRow(label, control)
                if isinstance(label, QWidget):
                    label.show()
            else:
                layout.addWidget(control)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            control.show()
            tree.setItemWidget(item, 0, host)
            self._rows.append((item, control))
            return item

        for control in self.method_buttons.values():
            row(root, control)
        row(root, self.qc_checkbox)
        parameters = group(root, "Parameters")
        row(parameters, self.scope_combo, "Streamlines")
        row(parameters, self.scope_label, "In scope")
        self._distance_row = row(parameters, self.distance_edit, self.distance_label)
        row(parameters, self.constraints_label)
        self._changed_row = row(root, self.parameters_label)
        row(root, self.run_button)
        self._cancel_row = row(root, self.cancel_button)
        row(root, self.status_label)
        result = group(root, "Results")
        row(result, self.visibility_label)
        row(result, self.result_header)
        for mode, label in self.method_results.items():
            row(group(result, METHODS[mode]), label)
        row(group(result, "Sampling"), self.sampling_label)
        row(group(result, "Surface QC"), self.geometry_label)
        row(result, self.colors_label)

    def isHidden(self) -> bool:
        return self.root_item is None or not self.root_item.isExpanded()

    def close(self) -> bool:
        if self.root_item is not None:
            self.root_item.setExpanded(False)
        return True

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Resize and self.root_item is not None:
            self._update_row_sizes()
        return super().eventFilter(watched, event)

    def _update_row_sizes(self) -> None:
        tree = self.tree
        for item, control in self._rows:
            host = tree.itemWidget(item, 0)
            if host is None:
                continue
            depth, parent = 0, item.parent()
            while parent is not None:
                depth += 1
                parent = parent.parent()
            width = max(100, tree.viewport().width() - (depth + 1) * tree.indentation() - 8)
            layout = host.layout()
            height = (
                layout.totalHeightForWidth(width)
                if layout.hasHeightForWidth() else layout.sizeHint().height()
            )
            height = max(24, height + 8)
            host.setFixedHeight(height)
            item.setSizeHint(0, QSize(0, height))

    def _method_changed(self, button: QAbstractButton, checked: bool) -> None:
        manager = self.mw.surface_mapping_manager
        if manager.mesh_id == self.mesh_id and (
            manager.current or manager.recipe_ids is not None
        ):
            mode = next(
                key for key, control in self.method_buttons.items() if control is button
            )
            if manager.set_method_enabled(mode, checked):
                return
        self.refresh()

    def _load_parameters(self) -> None:
        manager = self.mw.surface_mapping_manager
        parameters = (
            manager.parameters if manager.mesh_id == self.mesh_id else None
        ) or SurfaceMappingParameters()
        with signals_blocked(
            *self.method_buttons.values(),
            self.scope_combo,
            self.distance_edit,
        ):
            for mode, button in self.method_buttons.items():
                enabled = mode in parameters.modes
                if manager.mesh_id == self.mesh_id and not manager.busy:
                    presentation = manager.presentation
                    if presentation is not None:
                        enabled = presentation.method_enabled(mode)
                    elif manager.recipe_ids is not None:
                        name = (
                            "endpoint_enabled" if mode == "endpoint_proximity"
                            else "intersections_enabled"
                        )
                        enabled = enabled and manager.recipe_display[name]
                button.setChecked(enabled)
            self.scope_combo.setCurrentIndex(
                self.scope_combo.findData(parameters.scope)
            )
            self.distance_edit.setText(str(parameters.max_distance_mm))
        self._intersection_distance = (
            parameters.max_distance_mm
            if parameters.mode == "surface_intersections"
            else 2.0
        )

    def set_target(self, mesh_id: str | None) -> None:
        if mesh_id != self.mesh_id:
            self.mesh_id = mesh_id
            self._recipe_identity = None
            self._load_parameters()
        self.refresh()

    def parameters(self) -> SurfaceMappingParameters:
        endpoint = self.method_buttons["endpoint_proximity"].isChecked()
        intersections = self.method_buttons["surface_intersections"].isChecked()
        if not endpoint and not intersections:
            raise ValueError("Choose at least one mapping method")
        return SurfaceMappingParameters(
            mode=(
                "both" if endpoint and intersections
                else "endpoint_proximity" if endpoint else "surface_intersections"
            ),
            scope=self.scope_combo.currentData(),
            max_distance_mm=(
                float(self.distance_edit.text())
                if endpoint
                else self._intersection_distance
            ),
        )

    def _count_scope(self, parameters: SurfaceMappingParameters) -> None:
        manager = self.mw.surface_mapping_manager
        saved = (
            manager.mesh_id == self.mesh_id
            and manager.recipe_ids is not None
            and manager.parameters is not None
            and parameters.scope == manager.parameters.scope
            and (
                "endpoint_proximity" not in parameters.modes
                or parameters.max_distance_mm == manager.parameters.max_distance_mm
            )
        )
        revision = (
            self.mesh_id,
            parameters.scope,
            self.mw._visibility_version,
            manager.selection_version,
            id(self.mw.tractogram_data),
            id(manager.recipe_ids) if saved else None,
        )
        if revision != self._scope_revision:
            visible = self.mw.visible_indices
            if saved:
                self._scope_count = len(manager.recipe_ids)
            elif parameters.scope == "visible":
                self._scope_count = len(visible)
            else:
                self._scope_count = sum(
                    index in visible for index in self.mw.selected_streamline_indices
                )
            self._scope_revision = revision
        source = "saved scope IDs" if saved else "live; independent of skip"
        self.scope_label.setText(f"{self._scope_count:,} ({source})")

    def refresh(self, *args: object) -> None:
        if self._shutdown or getattr(self.mw, "_shutdown_requested", False):
            return
        if self.isHidden():
            if self.mesh_id not in self.mw.mesh_layers:
                self.mesh_id = None
                for control in (
                    self.run_button, self.cancel_button,
                    self.export_button, self.remove_button,
                ):
                    control.setEnabled(False)
            return
        manager = self.mw.surface_mapping_manager
        if manager.busy and manager._worker.layer["id"] in self.mw.mesh_layers:
            job_mesh = manager._worker.layer["id"]
            if job_mesh != self.mesh_id:
                self.mesh_id = job_mesh
                self._recipe_identity = None
                self._load_parameters()
            if id(manager._worker) != self._job_identity:
                self._job_identity = id(manager._worker)
                self._load_parameters()
        else:
            self._job_identity = None
        if self.mesh_id not in self.mw.mesh_layers:
            self.mesh_id = None
        owns = self.mesh_id is not None and self.mesh_id == manager.mesh_id
        result_identity = (
            id(manager.bundle) if owns and manager.bundle is not None else None
        )
        if not manager.busy and result_identity != self._result_identity:
            self._result_identity = result_identity
            if result_identity is not None:
                with signals_blocked(*self.method_buttons.values()):
                    for mode, button in self.method_buttons.items():
                        button.setChecked(manager.presentation.method_enabled(mode))
        recipe_identity = (
            id(manager.recipe_ids) if owns and manager.recipe_ids is not None else None
        )
        if recipe_identity != self._recipe_identity:
            self._recipe_identity = recipe_identity
            if recipe_identity is not None:
                self._load_parameters()
        layer = self.mw.mesh_layers.get(self.mesh_id)
        self.target_label.setText(
            layer["display_name"] if layer is not None else "Choose a surface mesh"
        )
        reason = manager.unavailable_reason(self.mesh_id)
        parameters = None
        try:
            parameters = self.parameters()
            self._count_scope(parameters)
            if not self._scope_count and not reason:
                reason = "The selected scope is empty."
        except (ValueError, TypeError, OverflowError):
            reason = (
                "Enter a finite, nonnegative maximum distance in mm."
                if any(button.isChecked() for button in self.method_buttons.values())
                else "Choose at least one mapping method."
            )
            self.scope_label.setText(
                "Enter valid parameters"
                if any(button.isChecked() for button in self.method_buttons.values())
                else "Choose a mapping method"
            )
        blocked = (
            manager.busy
            or manager._incompatible()
            or manager._exporting
            or getattr(self.mw, "_mesh_commit_active", False)
            or layer is None
        )
        for control in (
            *self.method_buttons.values(),
            self.scope_combo,
            self.distance_edit,
        ):
            control.setEnabled(not blocked)
        endpoint = self.method_buttons["endpoint_proximity"].isChecked()
        self.distance_edit.setVisible(endpoint)
        self.distance_label.setVisible(endpoint)
        self._distance_row.setHidden(not endpoint)
        has_analysis = owns and (
            manager.bundle is not None
            or manager.stale
            or manager.recipe_ids is not None
        )
        self.run_button.setText(
            "Recompute mapping" if has_analysis else "Compute mapping"
        )
        self.run_button.setEnabled(not reason and not blocked)
        self.cancel_button.setVisible(manager.busy)
        self._cancel_row.setHidden(not manager.busy)
        self.cancel_button.setEnabled(manager.busy and not manager._worker.is_cancelled)
        current = owns and manager.current
        status = (
            "Cancelling computation…"
            if manager.busy and manager._worker.is_cancelled
            else (
                "Recompute required"
                if owns and manager.stale
                else (
                    "Current result · Complete"
                    if current
                    else (
                        manager.message
                        if owns or manager.busy
                        else "Ready to compute mapping."
                    )
                )
            )
        )
        if reason:
            status += "\n" + reason
        if manager.mesh_id is not None and not owns and not manager.busy:
            other = self.mw.mesh_layers.get(manager.mesh_id)
            if other is not None:
                status += f"\nExisting mapping: {other['display_name']}."
        self.status_label.setText(status)
        changed = (
            has_analysis and parameters is not None
            and (
                parameters.scope != manager.parameters.scope
                or (
                    "endpoint_proximity" in parameters.modes
                    and parameters.max_distance_mm != manager.parameters.max_distance_mm
                )
                or not set(parameters.modes).issubset(manager.parameters.modes)
            )
        )
        self.parameters_label.setText(
            "Parameters changed — compute to apply" if changed else ""
        )
        self.parameters_label.setVisible(bool(changed))
        self._changed_row.setHidden(not changed)
        self._refresh_result(current, layer)
        actions = owns and manager.result_actions_enabled
        self.export_button.setEnabled(bool(actions))
        self.remove_button.setEnabled(bool(has_analysis and not blocked))
        self._update_row_sizes()

    def _refresh_result(self, current: bool, layer: MeshLayer | None) -> None:
        manager = self.mw.surface_mapping_manager
        qc_enabled = bool(
            current and not manager._incompatible() and not manager._exporting
        )
        self.qc_checkbox.setEnabled(qc_enabled)
        self.qc_label.setEnabled(qc_enabled)
        if not current:
            self.visibility_label.setText("No current mapping")
            self.result_header.setText("No current result")
            self.result_label.setText("Compute required")
            self.geometry_label.clear()
            self.sampling_label.clear()
            for mode, label in self.method_results.items():
                label.clear()
                self._groups[METHODS[mode]].setHidden(True)
            if manager.mesh_id == self.mesh_id and manager.parameters is not None:
                self._set_qc_control(manager.recipe_display["qc_enabled"])
                if manager.recipe_ids is not None:
                    self.result_header.setText(
                        "Saved recipe · "
                        + " + ".join(METHODS[mode] for mode in manager.parameters.modes)
                    )
            return
        bundle, presentation = manager.bundle, manager.presentation
        parameters = bundle.result.request.parameters
        endpoint = parameters.mode == "endpoint_proximity"
        visibility = (
            "Mapping · visible" if presentation.visible else "Mapping · hidden (tree)"
        )
        if not layer["visible"]:
            visibility += " · Target mesh hidden"
        elif not layer["presentation"]["surface"]["enabled"]:
            visibility += " · Surface hidden"
        self.visibility_label.setText(visibility)
        self._set_qc_control(presentation.qc_enabled)
        self.result_header.setText(
            f"{layer['display_name']} · "
            + " + ".join(METHODS[mode] for mode in manager.parameters.modes)
            + "\n"
            + (
                f"Maximum distance: {parameters.max_distance_mm} mm\n"
                if endpoint
                else ""
            )
            + SCOPES[parameters.scope]
        )
        d = bundle.result.diagnostics
        sections = [
            _summary_table("", [("Streamlines", f"{d.requested_streamlines:,}")])
        ]
        bundles = [bundle]
        if bundle.companion is not None:
            bundles.append(bundle.companion)
        for mode in METHODS:
            self._groups[METHODS[mode]].setHidden(mode not in manager.parameters.modes)
        markers = []
        for part in bundles:
            mode = part.result.request.parameters.mode
            diagnostics = part.result.diagnostics
            if mode == "endpoint_proximity":
                total = diagnostics.evaluated_endpoints
                rows = [
                    ("Endpoints", f"{total:,}"),
                    ("Assigned", f"{diagnostics.assigned_endpoints:,} / {total:,}"),
                    (
                        "Beyond distance",
                        f"{diagnostics.beyond_distance_endpoints:,} / {total:,}",
                    ),
                    ("Ambiguous", f"{diagnostics.ambiguous_endpoints:,} / {total:,}"),
                    ("Invalid", f"{diagnostics.invalid_endpoints:,} / {total:,}"),
                ]
                marker_name = "Endpoints + QC"
            else:
                rows = [
                    ("Crossing events", f"{diagnostics.crossing_events:,}"),
                    (
                        "Unique streamlines",
                        f"{diagnostics.crossing_streamlines:,} / "
                        f"{diagnostics.requested_streamlines:,}",
                    ),
                    ("Contacts", f"{diagnostics.touch_events:,}"),
                    ("Coplanar intervals", f"{diagnostics.coplanar_intervals:,}"),
                    ("Ambiguous events", f"{diagnostics.ambiguous_events:,}"),
                    ("Invalid events", f"{diagnostics.invalid_events:,}"),
                ]
                marker_name = "Crossings + QC"
            sections.append(_summary_table(METHODS[mode], rows))
            self.method_results[mode].setText(_summary_table("", rows))
            markers.append((
                marker_name, f"{len(part.marker_points):,} / {part.marker_count:,}"
            ))
            if mode == "surface_intersections":
                markers.append((
                    "Interval segments",
                    f"{len(part.overlap_points):,} / {part.overlap_segment_count:,}",
                ))
        sections.append(_summary_table("Markers · sample / total", markers))
        self.sampling_label.setText(_summary_table("Sample / total", markers))
        self.result_label.setText("<br>".join(sections))
        g = bundle.geometry.diagnostics
        self.geometry_label.setText(_summary_table("Surface QC", [
            ("Valid triangles", f"{g.valid_triangles:,} / {g.derived_triangles:,}"),
            ("Degenerate", f"{g.degenerate_triangles:,}"),
            ("Boundary edges", f"{g.boundary_edges:,}"),
            ("Nonmanifold edges", f"{g.nonmanifold_edges:,}"),
            ("Duplicate faces", f"{g.duplicate_faces:,}"),
            ("Nonplanar polygons", f"{g.nonplanar_polygons:,}"),
        ]))

    def _set_qc_control(self, enabled: bool) -> None:
        with signals_blocked(self.qc_checkbox):
            self.qc_checkbox.setChecked(enabled)

    def run_mapping(self) -> None:
        try:
            self.mw.surface_mapping_manager.start(self.mesh_id, self.parameters())
        except (ValueError, TypeError, OverflowError):
            self.refresh()

    def shutdown(self) -> None:
        if not self._shutdown:
            self._shutdown = True
            self.mw.surface_mapping_manager.changed.disconnect(self.refresh)
            self.tree.viewport().removeEventFilter(self)
            self.tree.model().modelAboutToBeReset.disconnect(self.detach)
            self.detach()
            self.mesh_id = None
