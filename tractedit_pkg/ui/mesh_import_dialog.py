"""Explicit coordinate choice, ready for the separate surface import action."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..data_contracts import MeshCoordinates, MeshSpaceMetadata
from ..mesh.model import MeshCoordinateError, resolve_mesh_coordinates


class MeshImportDialog(QDialog):
    """Collect coordinates only; readers and publication belong to the manager."""

    def __init__(
        self, metadata: MeshSpaceMetadata, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._metadata = metadata.copy()
        self.setWindowTitle("Mesh coordinates")
        layout = QVBoxLayout(self)
        self.requirements_label = QLabel(
            "Input coordinates must be in millimetres and already in the same "
            "subject space as the tractogram and optional T1w. Voxel coordinates, "
            "metres, tkregister/surface-RAS and other templates require external "
            "conversion or registration. No automatic alignment is performed."
        )
        self.requirements_label.setWordWrap(True)
        layout.addWidget(self.requirements_label)
        layout.addWidget(QLabel("Coordinate system"))
        self.coordinate_combo = QComboBox()
        for label, choice in (
            ("From file", "from_file"), ("RAS (mm)", "RAS"), ("LPS (mm)", "LPS")
        ):
            self.coordinate_combo.addItem(label, choice)
        layout.addWidget(self.coordinate_combo)
        self.message_label = QLabel()
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.coordinate_combo.currentIndexChanged.connect(self._update_coordinates)
        self._update_coordinates()

    @property
    def coordinates(self) -> MeshCoordinates:
        return resolve_mesh_coordinates(
            self._metadata, self.coordinate_combo.currentData()
        )

    def _update_coordinates(self) -> None:
        try:
            coordinates = self.coordinates
        except MeshCoordinateError as error:
            self.message_label.setText(str(error))
            valid = False
        else:
            valid = True
            space = coordinates["coordinate_system"]
            if coordinates["coordinate_provenance"]["overrides_metadata"]:
                declarations = repr(self._metadata["declarations"])
                message = (
                    f"File SPACE metadata {declarations}: "
                    f"explicit {space} takes precedence."
                )
            else:
                message = f"Input: {space} (mm). Display: scanner-RAS (mm)."
            self.message_label.setText(message)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    def accept(self) -> None:
        try:
            self.coordinates
        except MeshCoordinateError:
            return
        super().accept()
