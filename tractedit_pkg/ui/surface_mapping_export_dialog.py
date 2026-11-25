# -*- coding: utf-8 -*-
"""Local scientific-package and figure destinations for a current analysis."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from ..surface_mapping.export import output_paths, validate_destinations
from ..visualization.surface_mapping_export import FigureOptions


class SurfaceMappingExportDialog(QDialog):
    def __init__(self, manager):
        super().__init__(manager.mw)
        self.manager = manager
        self.setWindowTitle("Export surface mapping")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        self.scientific = QCheckBox("Scientific data (VTP + CSV + JSON)")
        self.scientific.setChecked(True)
        self.figure = QCheckBox("Figure (PNG + JSON)")
        layout.addWidget(self.scientific)
        layout.addWidget(self.figure)
        form = QFormLayout()
        self.scene_combo = QComboBox()
        self.scene_combo.addItem("Mapping only", "mapping_only")
        self.scene_combo.addItem("Current 3D scene", "current_scene")
        form.addRow("Figure content", self.scene_combo)
        self.width, self.height = QSpinBox(), QSpinBox()
        for control in (self.width, self.height):
            control.setRange(1, 65536)
            control.setValue(2048)
        form.addRow("Width (pixels)", self.width)
        form.addRow("Height (pixels)", self.height)
        self.alpha = QCheckBox("Transparent background (RGBA)")
        self.alpha.setEnabled(False)
        self.alpha.setToolTip(
            "RGBA capture is not verified on this backend. Figures use an opaque black background."
        )
        form.addRow(self.alpha)
        layout.addLayout(form)
        bundle = manager.bundle
        self._token = manager._token
        bundles = [bundle] if bundle.companion is None else [bundle, bundle.companion]
        self.full_markers = QCheckBox("Require full marker detail")
        self._full_markers_available = all(
            item.marker_count == len(item.marker_points)
            and item.overlap_segment_count == len(item.overlap_points)
            for item in bundles
        )
        self.full_markers.setChecked(self._full_markers_available)
        self.full_markers.setToolTip(
            "Available only when all markers and overlap segments fit the verified display budget. Scientific data always contains the complete result."
        )
        layout.addWidget(self.full_markers)
        self.note = QLabel(
            f"Current camera; proportions preserved with padding.\n"
            f"Markers: {sum(len(item.marker_points) for item in bundles):,}/"
            f"{sum(item.marker_count for item in bundles):,}; "
            f"overlap segments: {sum(len(item.overlap_points) for item in bundles):,}/"
            f"{sum(item.overlap_segment_count for item in bundles):,}.\n"
            "The scalar map and scientific data are complete. Figure markers retain the reported display sampling. Context streamlines retain their current skip."
            + (
                " Both methods are exported; intersection files use "
                "the _surface_intersections prefix."
                if bundle.companion is not None else ""
            )
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self.save_export)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.scientific.toggled.connect(self.refresh)
        self.figure.toggled.connect(self.refresh)
        manager.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        self.save_button.setEnabled(
            self.manager._token == self._token
            and self.manager.result_actions_enabled
            and (self.scientific.isChecked() or self.figure.isChecked())
        )
        for control in (self.scene_combo, self.width, self.height):
            control.setEnabled(self.figure.isChecked())
        self.full_markers.setEnabled(
            self.figure.isChecked() and self._full_markers_available
        )

    def _check_result(self):
        manager = self.manager
        manager.check_current()
        if manager._token != self._token or not manager.result_actions_enabled:
            raise ValueError("Recompute required: surface mapping is no longer current")

    def save_export(self):
        self.manager.check_current()
        self.refresh()
        if not self.save_button.isEnabled():
            return
        prefix, _ = QFileDialog.getSaveFileName(
            self,
            "Choose surface mapping output prefix",
            "surface_mapping",
            "All files (*)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not prefix:
            return
        try:
            manager = self.manager
            self._check_result()
            paths = output_paths(
                prefix,
                manager.bundle,
                self.scientific.isChecked(),
                self.figure.isChecked(),
            )
            validate_destinations(paths, manager._export_sources(), overwrite=True)
            existing = [path.name for path in paths.values() if path.exists()]
            if (
                existing
                and QMessageBox.question(
                    self,
                    "Replace surface mapping export",
                    "Replace the following files as one export set?\n\n"
                    + "\n".join(existing),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
            options = (
                FigureOptions(
                    self.width.value(),
                    self.height.value(),
                    self.scene_combo.currentData(),
                    require_full_markers=self.full_markers.isChecked(),
                )
                if self.figure.isChecked()
                else None
            )
            self._check_result()
            result = manager.export_to(
                prefix,
                scientific=self.scientific.isChecked(),
                figure_options=options,
                overwrite=bool(existing),
            )
            if result is not None:
                manager.mw.vtk_panel.update_status(
                    f"Surface mapping exported: {Path(prefix).name}"
                )
                self.accept()
        except Exception as error:
            QMessageBox.critical(self, "Surface mapping export error", str(error))
