# -*- coding: utf-8 -*-

"""
Data panel controller for TractEdit UI.

Owns the dockable data panel widget, drag-and-drop tree widget, item selection
and check-state slots, opacity handling, and 4D image volume selection.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6 import sip
from PyQt6.QtCore import QMimeData, QSignalBlocker, Qt
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...mesh.model import validate_mesh_parent
from ...mesh.presentation import MESH_NORMAL_DENSITY_RANGE
from ...utils import ColorMode, signals_blocked
from ..theme_manager import ThemeManager
from ..tractogram_data import TractogramDataPanel
from .menus import DataPanelMenusMixin
from .mesh_inspector import MeshInspectorMixin, _MeshTargetLabel
from .tree import DataPanelTreeMixin

if TYPE_CHECKING:
    from ...main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# Data Tree Widget & Data Panel Manager
# ============================================================================


class _DataTreeWidget(QTreeWidget):
    _MESH_MIME = "application/x-tractedit-mesh-id"

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self.mw = main_window
        self.setDragDropMode(QTreeWidget.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setAutoExpandDelay(700)

    def mimeTypes(self) -> list[str]:
        return [self._MESH_MIME]

    def startDrag(self, supported_actions) -> None:
        items = self.selectedItems()
        if len(items) != 1 or not self.mw.mesh_manager._can_mutate():
            return
        data = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict) or data.get("type") != "mesh":
            return
        mesh_id = data.get("id")
        if mesh_id not in self.mw.mesh_layers:
            return
        mime = QMimeData()
        mime.setData(self._MESH_MIME, mesh_id.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        try:
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            drag.deleteLater()

    def _drop_target(self, event) -> tuple[str, str | None] | None:
        if (
            event.source() is not self
            or not self.mw.mesh_manager._can_mutate()
            or not event.mimeData().hasFormat(self._MESH_MIME)
            or not event.possibleActions() & Qt.DropAction.MoveAction
        ):
            return None
        try:
            mesh_id = bytes(event.mimeData().data(self._MESH_MIME)).decode("utf-8")
        except UnicodeDecodeError:
            return None
        item = self.itemAt(event.position().toPoint())
        if item is None:
            return None
        data = item.data(0, Qt.ItemDataRole.UserRole)
        root = item
        while root.parent() is not None:
            root = root.parent()
        if (root.text(0) == "Meshes" and 0 <= event.position().x() < self.indentation()
                and self.mw.mesh_layers.get(mesh_id, {}).get("parent_id") is not None):
            parent_id = None
        elif isinstance(data, dict) and data.get("type") == "mesh":
            parent_id = data.get("id")
        elif item.parent() is None and item.text(0) == "Meshes":
            parent_id = None
        else:
            return None
        try:
            validate_mesh_parent(self.mw.mesh_layers, mesh_id, parent_id)
        except ValueError:
            return None
        return mesh_id, parent_id

    def dragEnterEvent(self, event) -> None:
        if event.source() is self and event.mimeData().hasFormat(self._MESH_MIME):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        super().dragMoveEvent(event)
        target = self._drop_target(event)
        if target is not None and (
            self.dropIndicatorPosition() == QTreeWidget.DropIndicatorPosition.OnItem
            or (target[1] is None and event.position().x() < self.indentation())
        ):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        target = self._drop_target(event)
        event.ignore()
        if target is None:
            return
        if (self.dropIndicatorPosition() != QTreeWidget.DropIndicatorPosition.OnItem
                and not (target[1] is None and event.position().x() < self.indentation())):
            return
        if self.mw.data_panel_manager._set_mesh_parent(*target):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()


class DataPanelManager(DataPanelTreeMixin, DataPanelMenusMixin, MeshInspectorMixin):
    """
    Manages the data panel dock widget and its tree view.

    This class handles:
    - Creating the dockable data panel
    - Building and debouncing tree updates
    - Handling item selection changes
    - Updating opacity based on selected items
    - Context menus for ROI, mesh, ODF, image, and parcellation operations
    - Inspecting mesh and polyline properties
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the data panel manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def create_data_panel(self) -> None:
        """Creates the dockable panel for listing loaded data."""
        mw = self.mw

        mw.data_dock_widget = QDockWidget("Data Panel", mw)
        mw.data_dock_widget.setObjectName("DataPanelDock")
        mw.data_dock_widget.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Container Widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tree Widget
        mw.data_tree_widget = _DataTreeWidget(mw)
        mw.data_tree_widget.setHeaderLabels(["Loaded Data"])
        mw.data_tree_widget.setMinimumWidth(200)
        mw.data_tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        mw.data_tree_widget.customContextMenuRequested.connect(
            self.on_data_panel_context_menu
        )
        mw.data_tree_widget.itemChanged.connect(self.on_data_panel_item_changed)
        mw.data_tree_widget.itemSelectionChanged.connect(self.on_data_item_selected)
        mw.data_tree_widget.itemDoubleClicked.connect(
            self.on_data_panel_item_double_clicked
        )

        # Highlighting Style Sheet
        mw.data_tree_widget.setStyleSheet(ThemeManager.tree_style(False))

        layout.addWidget(mw.data_tree_widget)
        self.mesh_inspector = QTreeWidget(container)
        self.mesh_inspector.setHeaderLabel("Mesh properties")
        self.mesh_inspector.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.mesh_inspector.customContextMenuRequested.connect(
            lambda position: self.on_data_panel_context_menu(position, self.mesh_inspector)
        )
        self.mesh_inspector.itemExpanded.connect(self._on_mapping_expanded)
        self.mesh_inspector.itemChanged.connect(self.on_data_panel_item_changed)
        self.mesh_inspector.itemSelectionChanged.connect(self._on_inspector_selection)
        self.inspected_mesh_id = None
        self._inspected_layer = None
        self._component_states = {}
        self.mesh_inspector.hide()
        layout.addWidget(self.mesh_inspector)
        self.mesh_properties = QWidget(container)
        properties_layout = QVBoxLayout(self.mesh_properties)
        self.mesh_target_label = _MeshTargetLabel()
        self.mesh_target_label.setTextFormat(Qt.TextFormat.PlainText)
        self.mesh_target_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )
        properties_layout.addWidget(self.mesh_target_label)
        self.mesh_color = QPushButton("Change color...")
        self.mesh_color.clicked.connect(self._change_mesh_color)
        properties_layout.addWidget(self.mesh_color)
        self.mesh_opacity = QSlider(Qt.Orientation.Horizontal)
        self.mesh_opacity.setRange(0, 100)
        self.mesh_opacity.setAccessibleName("Surface opacity")
        self.mesh_opacity.valueChanged.connect(self.on_opacity_slider_changed)
        self.mesh_opacity_label = QLabel("Opacity")
        self.mesh_opacity_label.setBuddy(self.mesh_opacity)
        properties_layout.addWidget(self.mesh_opacity_label)
        properties_layout.addWidget(self.mesh_opacity)
        size_layout = QHBoxLayout()
        self.mesh_size_label = QLabel()
        self.mesh_size_spinbox = QDoubleSpinBox()
        self.mesh_size_spinbox.setDecimals(1)
        self.mesh_size_spinbox.setSingleStep(0.5)
        self.mesh_size_spinbox.setSuffix(" px")
        self.mesh_size_spinbox.setKeyboardTracking(False)
        self.mesh_size_spinbox.valueChanged.connect(self._on_mesh_size_changed)
        self.mesh_size_label.setBuddy(self.mesh_size_spinbox)
        size_layout.addWidget(self.mesh_size_label)
        size_layout.addWidget(self.mesh_size_spinbox)
        properties_layout.addLayout(size_layout)
        density_layout = QHBoxLayout()
        self.mesh_density_label = QLabel("&Density:")
        self.mesh_density_spinbox = QDoubleSpinBox()
        self.mesh_density_spinbox.setRange(*MESH_NORMAL_DENSITY_RANGE)
        self.mesh_density_spinbox.setDecimals(1)
        self.mesh_density_spinbox.setSingleStep(1)
        self.mesh_density_spinbox.setSuffix(" %")
        self.mesh_density_spinbox.setKeyboardTracking(False)
        self.mesh_density_spinbox.setAccessibleName("Normal density")
        self.mesh_density_spinbox.setToolTip(
            "Percentage of nondegenerate derived faces, 0.1–100%.",
        )
        self.mesh_density_label.setBuddy(self.mesh_density_spinbox)
        self.mesh_density_spinbox.valueChanged.connect(self._on_mesh_density_changed)
        density_layout.addWidget(self.mesh_density_label)
        density_layout.addWidget(self.mesh_density_spinbox)
        properties_layout.addLayout(density_layout)
        self.mesh_normals_count_label = QLabel()
        properties_layout.addWidget(self.mesh_normals_count_label)
        self.mesh_properties.hide()
        layout.addWidget(self.mesh_properties)
        self._polyline_target = None
        self._polyline_callbacks = []
        self.polyline_properties = QWidget(container)
        polyline_layout = QVBoxLayout(self.polyline_properties)
        self.polyline_target_label = _MeshTargetLabel()
        self.polyline_target_label.setTextFormat(Qt.TextFormat.PlainText)
        polyline_layout.addWidget(self.polyline_target_label)
        self.polyline_color = QPushButton("Change color...")
        self.polyline_color.clicked.connect(self._change_polyline_color)
        polyline_layout.addWidget(self.polyline_color)
        for field, label, maximum in (
            ("width", "Line width", 10),
            ("size", "Point size", 20),
        ):
            row = QHBoxLayout()
            text = QLabel(label + ":")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(10, maximum * 10)
            slider.setAccessibleName(label + " slider")
            spinbox = QDoubleSpinBox()
            spinbox.setRange(1, maximum)
            spinbox.setDecimals(1)
            spinbox.setSingleStep(0.5)
            spinbox.setSuffix(" px")
            spinbox.setKeyboardTracking(False)
            spinbox.setAccessibleName(label)
            spinbox.setToolTip(f"Screen pixels, 1–{maximum}; geometry is unchanged.")
            text.setBuddy(spinbox)
            row.addWidget(text)
            row.addWidget(slider)
            row.addWidget(spinbox)
            setattr(self, f"polyline_{field}_slider", slider)
            setattr(self, f"polyline_{field}_spinbox", spinbox)
            polyline_layout.addLayout(row)
        self.polyline_show_points = QCheckBox("Show points")
        polyline_layout.addWidget(self.polyline_show_points)
        self.polyline_properties.setEnabled(False)
        self.polyline_properties.hide()
        layout.addWidget(self.polyline_properties)
        self.tractogram_data = TractogramDataPanel(mw, container)
        layout.addWidget(self.tractogram_data)
        mw.data_dock_widget.setWidget(container)
        mw.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, mw.data_dock_widget)

    def on_data_item_selected(self) -> None:
        """Updates the opacity slider based on the selected item."""
        mw = self.mw
        self.tractogram_data.sync()
        self.tractogram_data.hide()
        self.mesh_properties.hide()
        self.update_polyline_properties()
        items = mw.data_tree_widget.selectedItems()
        owner = items[0] if items else mw.data_tree_widget.currentItem()
        data = owner.data(0, Qt.ItemDataRole.UserRole) if owner is not None else None
        mesh_id = (
            data["id"] if isinstance(data, dict) and data.get("type") == "mesh" else None
        )
        if not items and self.mesh_inspector.selectedItems():
            with QSignalBlocker(self.mesh_inspector):
                self.mesh_inspector.clearSelection()
        self._show_mesh_inspector(mesh_id if self._polyline_target is None else None)
        if self._polyline_target is not None:
            return

        if mesh_id is not None:
            self.update_mesh_properties()
            return

        items = mw.data_tree_widget.selectedItems()
        if not items:
            mw.opacity_slider.setEnabled(False)
            return

        item = items[0]

        if self.tractogram_data.select(item.data(0, Qt.ItemDataRole.UserRole)):
            mw.opacity_slider.setEnabled(False)
            return

        parent = item.parent()
        if parent and parent.text(0) == "Scalars":
            scalar_name = item.text(0)

            if (
                mw.tractogram_data
                and mw.scalar_data_per_point
                and scalar_name in mw.scalar_data_per_point
            ):
                mw.active_scalar_name = scalar_name
                mw.scalar_display = None

                for i in range(parent.childCount()):
                    child = parent.child(i)
                    font = child.font(0)
                    font.setBold(child.text(0) == scalar_name)
                    child.setFont(0, font)

                mw.scalar_manager.update_scalar_data_range()
                mw.scalar_range_initialized = True

                if mw.current_color_mode != ColorMode.SCALAR:
                    with signals_blocked(mw.color_scalar_action):
                        mw.color_scalar_action.setChecked(True)
                    mw._set_color_mode(ColorMode.SCALAR)
                else:
                    if mw.vtk_panel:
                        mw.vtk_panel.update_main_streamlines_actor()

            mw.opacity_slider.setEnabled(False)
            return

        item_data = item.data(0, Qt.ItemDataRole.UserRole)

        if not item_data or not isinstance(item_data, dict):
            mw.opacity_slider.setEnabled(False)
            return

        itype = item_data.get("type")
        val = 1.0

        with signals_blocked(mw.opacity_slider):
            if itype == "bundle":
                val = mw.bundle_opacity
                mw.opacity_slider.setEnabled(True)
            elif itype == "image":
                val = mw.image_opacity
                mw.opacity_slider.setEnabled(True)
            elif itype == "mesh":
                layer = mw.mesh_layers.get(item_data.get("id"))
                if layer is None:
                    mw.opacity_slider.setEnabled(False)
                    return
                val = layer["opacity"]
                mw.opacity_slider.setEnabled(not mw._session_busy)
            elif itype == "roi":
                path = item_data.get("path")
                val = mw.roi_opacities.get(path, 0.5)
                mw.opacity_slider.setEnabled(True)

                # Set as current drawing ROI
                mw.current_drawing_roi = path
                if mw.vtk_panel:
                    status_msg = f"Selected ROI: {path}"
                    if mw.is_drawing_mode:
                        status_msg += " (Drawing Enabled)"
                    elif getattr(mw, "is_eraser_mode", False):
                        status_msg += " (Eraser Enabled)"
                    elif getattr(mw, "is_sphere_mode", False):
                        status_msg += " (Sphere Mode Enabled)"
                    elif getattr(mw, "is_rectangle_mode", False):
                        status_msg += " (Rectangle Mode Enabled)"
                    mw.vtk_panel.update_progress_status(status_msg)
            else:
                mw.opacity_slider.setEnabled(False)

            mw.opacity_slider.setValue(int(val * 100))

    def on_data_panel_item_double_clicked(
        self, item: QTreeWidgetItem, column: int
    ) -> None:
        if column != 0:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict) or data.get("type") != "tractogram_array":
            return
        key = (data.get("association"), data.get("group"), data.get("name"))
        if key not in self.tractogram_data.fields:
            return
        self.tractogram_data.select(data)
        self.tractogram_data.inspect_values()

    def on_opacity_slider_changed(self, value: int) -> None:
        """Updates the opacity of the selected item."""
        mw = self.mw
        float_val = value / 100.0
        if self.inspected_mesh_id is not None:
            target = self._selected_mesh_component()
            if target is not None and target[1] == "surface":
                mw.mesh_manager.set_opacity(target[0], float_val)
            return

        items = mw.data_tree_widget.selectedItems()
        if not items:
            return
        item = items[0]
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data:
            return

        itype = item_data.get("type")

        if itype == "bundle":
            mw.bundle_opacity = float_val
            if mw.vtk_panel:
                mw.vtk_panel.set_streamlines_opacity(float_val)

        elif itype == "image":
            mw.image_opacity = float_val
            if mw.vtk_panel:
                mw.vtk_panel.set_anatomical_opacity(float_val)

        elif itype == "roi":
            path = item_data.get("path")
            if path:
                mw.roi_opacities[path] = float_val
                if mw.vtk_panel:
                    mw.vtk_panel.set_roi_opacity(path, float_val)

    def on_data_panel_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """
        Slot triggered when an item in the data panel (QTreeWidget) is changed,
        e.g., a checkbox is toggled.
        """
        mw = self.mw

        if column != 0:
            return

        # Get the data we stored to identify the item
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data or not isinstance(item_data, dict):
            return

        # Get the new check state
        is_checked = item.checkState(0) == Qt.CheckState.Checked
        item_type = item_data.get("type")

        if item_type == "standalone_polyline":
            mw.mesh_polyline_manager.set_visible(None, item_data["id"], is_checked)
            return

        if item_type in ("mesh_polyline", "mesh_polygon"):
            mw.mesh_polyline_manager.set_visible(
                item_data["id"], item_data["component"], is_checked,
            )
            return

        if item_type == "surface_mapping":
            if item_data.get("id") == mw.surface_mapping_manager.mesh_id:
                mw.surface_mapping_manager.set_display("visible", is_checked)
            return

        if item_type in ("mesh", "mesh_component"):
            mesh_id = item_data.get("id")
            component = item_data.get("component")
            try:
                if component == "texture":
                    loading = mw.mesh_manager.texture_status(mesh_id) == "loading"
                    mw.mesh_manager.request_texture(
                        mesh_id, is_checked and not loading,
                    )
                elif component is not None:
                    pending = mw.mesh_manager.overlay_pending(mesh_id, component)
                    mw.mesh_manager.set_component_visible(
                        mesh_id, component, is_checked and not pending,
                    )
                else:
                    mw.mesh_manager.set_visible(mesh_id, is_checked)
            except (RuntimeError, ValueError, AttributeError, KeyError, MemoryError):
                logger.exception("Could not update mesh visibility")
                self.update_mesh_items(mesh_id)
            else:
                if not sip.isdeleted(item) and item.treeWidget() is not None:
                    with QSignalBlocker(item.treeWidget()):
                        self._sync_mesh_item(item)
            return

        tree = mw.data_tree_widget
        if tree is None:
            return
        blocked = tree.blockSignals(True)

        try:
            if item_type == "bundle":
                mw._toggle_bundle_visibility(is_checked)

            elif item_type == "image":
                mw._toggle_image_visibility(is_checked)

            elif item_type == "roi":
                path = item_data.get("path")
                if path:
                    mw._toggle_roi_visibility(path, is_checked)

            elif item_type == "parcellation":
                # Toggle entire parcellation overlay
                mw._toggle_parcellation_overlay(is_checked)
                # Update the menu action state to match
                if hasattr(mw, "view_parcellation_action"):
                    mw.view_parcellation_action.setChecked(is_checked)

            elif item_type == "parcellation_region":
                # Toggle individual region visibility
                label = item_data.get("label")
                if label is not None:
                    mw._toggle_parcellation_region(label, is_checked)

            elif item_type == "odf_tunnel":
                # Toggle ODF tunnel visibility without recomputing
                mw._toggle_odf_tunnel_visibility(is_checked)

        except (RuntimeError, ValueError, AttributeError, KeyError) as e:
            logger.error(f"Error handling item visibility change: {e}", exc_info=True)
        finally:
            tree.blockSignals(blocked)

    def _set_mesh_parent(self, mesh_id: str, parent_id: str | None) -> bool:
        if not self.mw.mesh_manager._can_mutate():
            return False
        try:
            self.mw.mesh_manager.set_parent(mesh_id, parent_id)
        except (ValueError, RuntimeError, MemoryError) as error:
            logger.exception("Could not move mesh")
            QMessageBox.warning(self.mw, "Move mesh", str(error))
            return False
        return True

    def select_image_volume(self) -> None:
        image = self.mw.anatomical_mmap_image
        if image is None or len(image.source_shape) != 4:
            return
        count = image.source_shape[3]
        number, accepted = QInputDialog.getInt(
            self.mw, "Select volume", f"Volume (1–{count}):",
            image.volume_index + 1, 1, count,
        )
        if accepted:
            self._change_image_display(number - 1, False)

    def _change_image_display(self, volume_index, rgb) -> None:
        mw = self.mw
        if mw._session_busy or mw._shutdown_requested:
            return
        image = mw.anatomical_mmap_image
        if image is None or (image.volume_index == volume_index and image.rgb == rgb):
            return
        mw._load_initial_anat_threaded(
            mw.anatomical_image_path,
            anatomical_display={"volume_index": volume_index, "rgb": rgb},
        )
