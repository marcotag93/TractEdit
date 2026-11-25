# -*- coding: utf-8 -*-

"""
Mesh and polyline inspector mixin for the TractEdit data panel.

Manages the mesh component tree, mesh information rows, mesh property controls,
and polyline property controls.
"""

from __future__ import annotations

import logging
from itertools import chain
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QSignalBlocker, Qt
from PyQt6.QtWidgets import QLabel, QTreeWidgetItem

from ...mesh.presentation import (
    ADVANCED_MESH_COMPONENTS,
    MESH_COMPONENT_LABELS,
    MESH_LINE_WIDTH_RANGE,
    MESH_NORMAL_LENGTH_RANGE,
    MESH_POINT_SIZE_RANGE,
)
from ...utils import signals_blocked

if TYPE_CHECKING:
    from ...main_window import MainWindow

logger = logging.getLogger(__name__)


class _MeshTargetLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._full_text = ""

    def setText(self, text: str) -> None:
        self._full_text = text
        self.setAccessibleName(text)
        self._elide_text()

    def _elide_text(self) -> None:
        super().setText(self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, self.contentsRect().width(),
        ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide_text()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._elide_text()


class MeshInspectorMixin:
    """Mixin implementing mesh and polyline inspection for DataPanelManager."""

    mw: "MainWindow"

    def component_states(self):
        from ...logic.mesh_manager import _tree_items

        for item in _tree_items(self.mesh_inspector):
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and "component" in data:
                key = (data["id"], data["component"])
                self._component_states[key] = (item.isExpanded(), item.isSelected())
        return self._component_states

    def _show_mesh_inspector(self, mesh_id: str | None) -> None:
        from ...logic.mesh_manager import _tree_items

        layer = self.mw.mesh_layers.get(mesh_id)
        if layer is not self._inspected_layer:
            states = self.component_states()
            self._component_states = {
                key: (flags[0], False) for key, flags in states.items()
                if key[0] in self.mw.mesh_layers
            }
            with QSignalBlocker(self.mesh_inspector):
                self.mesh_inspector.clear()
                self.inspected_mesh_id = mesh_id
                self._inspected_layer = layer
                if layer is not None:
                    self.mesh_inspector.setHeaderLabel(
                        f"Properties: {layer['display_name']}"
                    )
                    appearance = QTreeWidgetItem(self.mesh_inspector, ["Appearance"])
                    appearance.setData(0, Qt.ItemDataRole.UserRole, {
                        "type": "mesh_group", "id": mesh_id, "component": "appearance",
                    })
                    self.build_mesh_component_items(appearance, mesh_id)
                    root = self.mesh_inspector.invisibleRootItem()
                    self.mw.surface_mapping_manager.sync_tree(root, mesh_id)
                    self._build_mesh_information_items(root, mesh_id)
                    for item in _tree_items(self.mesh_inspector):
                        data = item.data(0, Qt.ItemDataRole.UserRole)
                        if data and "component" in data:
                            flags = self._component_states.get((mesh_id, data["component"]))
                            if flags:
                                item.setExpanded(flags[0])
        self.mesh_inspector.setVisible(layer is not None)

    def build_mesh_component_items(
        self, parent: QTreeWidgetItem, mesh_id: str,
    ) -> None:
        surface = None
        for component, text in (
            ("surface", "Surface"), ("texture", "Texture"),
            ("edges", "Edges / Wireframe"), ("vertices", "Vertices"),
        ):
            owner = surface if component == "texture" else parent
            item = QTreeWidgetItem(owner, [text])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "mesh_component", "id": mesh_id, "component": component,
            })
            if component == "surface":
                surface = item
            self._sync_mesh_item(item)
        parent.setExpanded(True)
        advanced = QTreeWidgetItem(parent, ["Advanced"])
        advanced.setFlags(advanced.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        advanced.setData(0, Qt.ItemDataRole.UserRole, {
            "type": "mesh_group", "id": mesh_id, "component": "advanced",
        })
        for component in ADVANCED_MESH_COMPONENTS:
            item = QTreeWidgetItem(advanced, [MESH_COMPONENT_LABELS[component]])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "mesh_component", "id": mesh_id, "component": component,
            })
            self._sync_mesh_item(item)

    def _on_mapping_expanded(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and data.get("type") == "surface_mapping":
            self.mw.surface_mapping_manager.open_panel(data["id"])

    def _build_mesh_information_items(
        self, parent: QTreeWidgetItem, mesh_id: str,
    ) -> None:
        from ...mesh.overlays import mesh_ras_bounds

        layer = self.mw.mesh_layers[mesh_id]
        provenance = layer["coordinate_provenance"]
        metadata = provenance["metadata"]
        choice = (
            "from file" if provenance["choice"] == "from_file" else
            "metadata override" if provenance["overrides_metadata"] else
            "explicit choice"
        )
        bounds = mesh_ras_bounds(layer)
        bounds_text = "; ".join(
            f"{axis} [{bounds[i]:.3f}, {bounds[i + 1]:.3f}]"
            for axis, i in zip("XYZ", (0, 2, 4))
        )
        rows = [
            ("vertices", f"Vertices: {layer['point_count']:,}", ""),
            ("polygons", f"Polygons: {layer['polygon_count']:,}", ""),
        ]
        if layer["strip_count"]:
            rows.append(("strips", f"Triangle strips: {layer['strip_count']:,}", ""))
        rows.extend((
            ("source", f"Source coordinates: {layer['coordinate_system']} (mm)"
             f" — {choice}", f"File SPACE: {metadata['status']}\n"
             f"Declarations: {metadata['declarations']}\n"
             "No registration or unit conversion is performed."),
            ("display", "Display coordinates: RAS (mm)", ""),
            ("bounds", "Bounds RAS (mm): " + bounds_text,
             "Source geometry bounds in RAS millimetres:\n" + "; ".join(
                 f"{axis} [{bounds[i]:.17g}, {bounds[i + 1]:.17g}]"
                 for axis, i in zip("XYZ", (0, 2, 4))
             )),
            ("texture", "", ""),
        ))
        group = QTreeWidgetItem(parent, ["Information"])
        group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        group.setData(0, Qt.ItemDataRole.UserRole, {
            "type": "mesh_group", "id": mesh_id, "component": "information",
        })
        for field, text, detail in rows:
            item = QTreeWidgetItem(group, [text])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "mesh_information", "id": mesh_id,
                "component": "information_" + field,
            })
            item.setToolTip(0, detail or text)
            if field == "texture":
                self._sync_mesh_information(item)

    def _sync_mesh_information(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data["component"] != "information_texture":
            return
        key = data["id"]
        layer = self.mw.mesh_layers.get(key)
        if layer is None:
            return
        _, checked, _, message = self.mw.mesh_manager.texture_control_state(key)
        status = self.mw.mesh_manager.texture_status(key)
        if status == "loading":
            text = "Loading..."
        elif status == "unavailable" or not layer["path"].lower().endswith(".obj"):
            text = "Unavailable"
        elif checked:
            text = "ON" if self.mw.vtk_panel.mesh_texture_enabled(key) else "ON (deferred)"
        else:
            text = (
                "OFF (not examined)"
                if status == "unloaded" and "texture" not in layer else "OFF"
            )
        item.setText(0, "Texture: " + text)
        item.setToolTip(0, message)

    def _sync_mesh_item(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        layer = self.mw.mesh_layers.get(data["id"])
        if layer is None:
            return
        component = data.get("component")
        editable = not (self.mw._session_busy or self.mw._shutdown_requested)
        flags = item.flags()
        item.setFlags(
            flags | Qt.ItemFlag.ItemIsUserCheckable if editable
            else flags & ~Qt.ItemFlag.ItemIsUserCheckable
        )
        if component == "texture":
            state = self.mw.mesh_manager.texture_control_state(data["id"])
            enabled, checked, text, message = state
            item.setDisabled(not enabled)
            item.setText(0, text)
            item.setToolTip(0, message)
        else:
            checked = (layer["visible"] if component is None
                       else layer["presentation"][component]["enabled"])
            if component is not None:
                message = {
                    "boundary_edges": (
                        "Open borders of the loaded topology; derived material/UV "
                        "splits are ignored. Coincident source points, including "
                        "native reader splits, are not welded."
                    ),
                    "bounding_box": (
                        "Axis-aligned bounds of the source geometry in RAS millimetres."
                    ),
                    "surface_normals": (
                        "Geometric face normals "
                        "(derived triangles for polygons/strips)."
                        " Directions follow vertex winding; no outward orientation "
                        "is assumed."
                    ),
                }.get(component, (
                    "3D mesh representation; independent of the other components."
                ))
                if component in ADVANCED_MESH_COMPONENTS:
                    pending = self.mw.mesh_manager.overlay_pending(
                        data["id"], component,
                    )
                    item.setText(0, MESH_COMPONENT_LABELS[component] + (
                        " (preparing...)" if pending else ""
                    ))
                    if pending:
                        message += "\nPreparing overlay. Click again to cancel."
                item.setToolTip(0, message)
        item.setCheckState(
            0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
        )

    def update_mesh_items(self, mesh_id: str | None = None) -> None:
        """Synchronize resident rows without keeping handles across tree rebuilds."""
        from ...logic.mesh_manager import _tree_items

        tree = self.mw.data_tree_widget
        with QSignalBlocker(tree), QSignalBlocker(self.mesh_inspector):
            for item in chain(
                _tree_items(tree, mesh_only=True), _tree_items(self.mesh_inspector),
            ):
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if (
                    isinstance(data, dict)
                    and data.get("type") in ("mesh", "mesh_component", "mesh_information")
                    and (mesh_id is None or data["id"] == mesh_id)
                ):
                    if data["type"] == "mesh_information":
                        self._sync_mesh_information(item)
                    else:
                        self._sync_mesh_item(item)
        target = self._selected_mesh_component()
        if mesh_id is None or target is not None and target[0] == mesh_id:
            self.update_mesh_properties()
        self.mw.surface_mapping_manager.refresh_panel()

    def _selected_mesh_component(self) -> tuple[str, str] | None:
        items = self.mesh_inspector.selectedItems()
        if not items and self.mesh_inspector.currentItem() is None:
            items = self.mw.data_tree_widget.selectedItems()
        data = items[0].data(0, Qt.ItemDataRole.UserRole) if items else None
        if (
            isinstance(data, dict)
            and data.get("type") in ("mesh", "mesh_component")
            and data.get("id") in self.mw.mesh_layers
        ):
            return data["id"], data.get("component", "surface")
        return None

    def _on_inspector_selection(self) -> None:
        tree = self.mw.data_tree_widget
        owner = tree.currentItem()
        if owner is not None and self.mesh_inspector.selectedItems() and not owner.isSelected():
            data = owner.data(0, Qt.ItemDataRole.UserRole)
            if data == {"type": "mesh", "id": self.inspected_mesh_id}:
                with QSignalBlocker(tree):
                    owner.setSelected(True)
        self.update_mesh_properties()

    def _change_mesh_color(self) -> None:
        target = self._selected_mesh_component()
        if target is not None and target[1] in MESH_COMPONENT_LABELS:
            self.mw.mesh_manager.change_color(*target)

    def update_mesh_properties(self) -> None:
        target = self._selected_mesh_component()
        if target is None or target[1] == "texture":
            self.mesh_properties.hide()
            if target is not None or self.inspected_mesh_id is not None:
                self.mw.opacity_slider.setEnabled(False)
            return
        mesh_id, component = target
        layer = self.mw.mesh_layers[mesh_id]
        style = layer if component == "surface" else layer["presentation"][component]
        label = f"{MESH_COMPONENT_LABELS[component]} — {layer['display_name']}"
        self.mesh_target_label.setText(label)
        self.mesh_target_label.setToolTip(label + "\n" + layer["path"])
        self.mesh_properties.setEnabled(
            not (self.mw._session_busy or self.mw._shutdown_requested)
        )
        normals = component == "surface_normals"
        has_size = component != "surface"
        self.mesh_properties.show()
        self.mesh_opacity.setVisible(not has_size)
        self.mesh_opacity_label.setVisible(not has_size)
        if not has_size:
            with signals_blocked(self.mesh_opacity):
                self.mesh_opacity.setValue(round(layer["opacity"] * 100))
        self.mesh_size_label.setVisible(has_size)
        self.mesh_size_spinbox.setVisible(has_size)
        self.mesh_density_label.setVisible(normals)
        self.mesh_density_spinbox.setVisible(normals)
        self.mesh_normals_count_label.setVisible(normals)
        if has_size:
            field = ("length_mm" if normals else
                     "point_size" if component == "vertices" else "line_width")
            bounds = {"length_mm": MESH_NORMAL_LENGTH_RANGE,
                      "point_size": MESH_POINT_SIZE_RANGE,
                      "line_width": MESH_LINE_WIDTH_RANGE}[field]
            name = {"length_mm": "Length", "point_size": "Point size",
                    "line_width": "Line width"}[field]
            self.mesh_size_label.setText("&" + name + ":")
            self.mesh_size_spinbox.setAccessibleName(name)
            self.mesh_size_spinbox.setSuffix(" mm" if normals else " px")
            self.mesh_size_spinbox.setToolTip(
                f"{label}\n{'Millimetres' if normals else 'Screen pixels'}, "
                f"{bounds[0]:g}–{bounds[1]:g}."
            )
            with signals_blocked(self.mesh_size_spinbox):
                self.mesh_size_spinbox.setRange(*bounds)
                self.mesh_size_spinbox.setValue(style[field])
        if normals:
            from ...mesh.overlays import MAX_MESH_NORMAL_GLYPHS, normal_glyph_count

            with signals_blocked(self.mesh_density_spinbox):
                self.mesh_density_spinbox.setValue(style["density"])
            data = self.mw.vtk_panel.mesh_overlay_data(layer, component)
            text = f"Maximum {MAX_MESH_NORMAL_GLYPHS:,} arrows; prepared on first use."
            if data is not None:
                count = normal_glyph_count(data, style["density"])
                text = f"{count:,} arrows / {data.face_count:,} valid faces"
                if data.face_count * style["density"] / 100 > MAX_MESH_NORMAL_GLYPHS:
                    text += f" (limit {MAX_MESH_NORMAL_GLYPHS:,})"
            self.mesh_normals_count_label.setText(text)
            self.mesh_normals_count_label.setWordWrap(True)
        with signals_blocked(self.mw.opacity_slider):
            self.mw.opacity_slider.setEnabled(
                component == "surface" and self.mesh_properties.isEnabled()
            )
            if component == "surface":
                self.mw.opacity_slider.setValue(round(layer["opacity"] * 100))

    def _on_mesh_size_changed(self, value: float) -> None:
        target = self._selected_mesh_component()
        if target is None or target[1] in ("surface", "texture"):
            return
        mesh_id, component = target
        field = ("length_mm" if component == "surface_normals" else
                 "point_size" if component == "vertices" else "line_width")
        try:
            self.mw.mesh_manager.set_component_style(
                mesh_id, component, **{field: value},
            )
        except (RuntimeError, ValueError, AttributeError, KeyError):
            logger.exception("Could not update mesh component size")
        finally:
            self.update_mesh_properties()

    def _on_mesh_density_changed(self, value: float) -> None:
        target = self._selected_mesh_component()
        if target is not None and target[1] == "surface_normals":
            try:
                self.mw.mesh_manager.set_component_style(*target, density=value)
            except (RuntimeError, ValueError, AttributeError, KeyError):
                logger.exception("Could not update mesh normal density")
            finally:
                self.update_mesh_properties()

    def _selected_polyline(self):
        manager = self.mw.mesh_polyline_manager
        if manager.active:
            return manager.mw.mesh_polyline_mesh_id, None, manager._draft_style
        items = self.mw.data_tree_widget.selectedItems()
        data = items[0].data(0, Qt.ItemDataRole.UserRole) if items else None
        if isinstance(data, dict) and data.get("type") in (
            "mesh_polyline",
            "mesh_polygon",
            "standalone_polyline",
        ):
            mesh_id = (
                data["id"]
                if data["type"] in ("mesh_polyline", "mesh_polygon")
                else None
            )
            identity = data["component"]
            record = manager._record(mesh_id, identity)
            if record is not None:
                return mesh_id, identity, record
        return None

    def _change_polyline_color(self) -> None:
        target = self._selected_polyline()
        if target is not None and target[1] is not None:
            self.mw.mesh_polyline_manager.change_color(target[0], target[1])

    def update_polyline_properties(self) -> None:
        target = self._selected_polyline()
        previous = self._polyline_target
        changed = (
            previous is None
            or target is None
            or previous[:2] != target[:2]
            or previous[2] is not target[2]
        )
        if changed:
            for signal, callback in self._polyline_callbacks:
                signal.disconnect(callback)
            self._polyline_callbacks.clear()
            self._polyline_target = target
        self.polyline_properties.setVisible(target is not None)
        self.polyline_properties.setEnabled(
            target is not None and self.mw.mesh_manager._can_mutate()
        )
        if target is None:
            return
        self.mesh_properties.hide()
        self.mw.opacity_slider.setEnabled(False)
        mesh_id, identity, record = target
        owner = self.mw.mesh_layers.get(mesh_id)
        association = owner["display_name"] if owner is not None else "Standalone"
        polygon_editor = self.mw.mesh_polyline_manager._polygon
        is_polygon_draft = (
            identity is None
            and polygon_editor is not None
            and polygon_editor.layer is not None
        )
        name = (
            "Surface polygon draft" if is_polygon_draft
            else "Polyline draft" if identity is None
            else record["name"]
        )
        self.polyline_target_label.setText(f"{name} ({association})")
        self.polyline_color.setVisible(identity is not None)
        with signals_blocked(
            self.polyline_width_slider,
            self.polyline_width_spinbox,
            self.polyline_size_slider,
            self.polyline_size_spinbox,
            self.polyline_show_points,
        ):
            self.polyline_width_slider.setValue(round(record["line_width"] * 10))
            self.polyline_width_spinbox.setValue(record["line_width"])
            self.polyline_size_slider.setValue(round(record["point_size"] * 10))
            self.polyline_size_spinbox.setValue(record["point_size"])
            self.polyline_show_points.setChecked(record["show_points"])
        if changed:

            def connect(signal, callback):
                signal.connect(callback)
                self._polyline_callbacks.append((signal, callback))

            for field, name in (("width", "line_width"), ("size", "point_size")):
                slider = getattr(self, f"polyline_{field}_slider")
                spinbox = getattr(self, f"polyline_{field}_spinbox")
                connect(
                    slider.valueChanged,
                    lambda value, name=name, target=target: self._set_polyline_style(
                        target, **{name: value / 10}
                    ),
                )
                connect(
                    spinbox.valueChanged,
                    lambda value, name=name, target=target: self._set_polyline_style(
                        target, **{name: value}, final=True
                    ),
                )
                connect(
                    slider.sliderReleased,
                    lambda target=target: self._set_polyline_style(target, final=True),
                )
            connect(
                self.polyline_show_points.toggled,
                lambda value, target=target: self._set_polyline_style(
                    target, show_points=value, final=True
                ),
            )

    def _set_polyline_style(self, target, **style) -> None:
        if target is not self._polyline_target:
            return
        manager = self.mw.mesh_polyline_manager
        mesh_id, identity, record = target
        current = (
            manager._draft_style
            if identity is None and manager.active
            else manager._record(mesh_id, identity)
        )
        if current is not record:
            self.update_polyline_properties()
            return
        manager.set_style(mesh_id, identity, **style)
        self.update_polyline_properties()
