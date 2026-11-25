# -*- coding: utf-8 -*-

"""
Data panel tree builder mixin and helper functions.

Constructs and updates the QTreeWidget hierarchy for tractograms, anatomical
images, ROIs, meshes, ODFs, and FreeSurfer parcellations while preserving
expansion, selection, and scroll state across debounced rebuilds.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Set, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QStyle, QTreeWidgetItem

from ...mesh.model import validate_mesh_forest
from ...utils import format_tuple, iter_tree_items as _tree_items, signals_blocked

if TYPE_CHECKING:
    from ...main_window import MainWindow

logger = logging.getLogger(__name__)


def _tree_item_key(item):
    """Disambiguate mapping rows that share the same UserRole payload."""
    data = item.data(0, Qt.ItemDataRole.UserRole)
    if not isinstance(data, dict):
        return item.text(0)
    key = tuple(sorted(data.items()))
    if data.get("type") == "surface_mapping_control":
        path = []
        parent = item.parent()
        while parent is not None:
            path.append(parent.indexOfChild(item))
            parent_data = parent.data(0, Qt.ItemDataRole.UserRole) or {}
            if parent_data.get("type") != "surface_mapping_control":
                break
            item, parent = parent, parent.parent()
        key += (("row", tuple(reversed(path))),)
    return key

_PRESERVED_TREE_TYPES = (
    "mesh",
    "mesh_component",
    "mesh_group",
    "mesh_information",
    "mesh_polygon",
    "mesh_polygons",
    "mesh_polyline",
    "mesh_polylines",
    "standalone_polyline",
    "standalone_polylines",
    "surface_mapping",
    "surface_mapping_control",
    "tractogram_array",
    "tractogram_information",
)
_SELECTED_MESH_TYPES = (
    "mesh",
    "mesh_component",
    "mesh_group",
    "mesh_information",
    "mesh_polygon",
    "mesh_polygons",
    "mesh_polyline",
    "mesh_polylines",
    "standalone_polyline",
    "standalone_polylines",
    "tractogram_array",
    "tractogram_information",
)


def classify_region_hemisphere(label_name: str) -> str:
    """
    Classifies a FreeSurfer region name into hemisphere.

    Args:
        label_name: The region name from FreeSurfer parcellation.

    Returns:
        'left', 'right', or 'other'
    """
    name_lower = label_name.lower()

    # Left hemisphere indicators
    if any(x in name_lower for x in ["left-", "ctx-lh-", "-lh-", "lh-", "lh_"]):
        return "left"

    # Right hemisphere indicators
    if any(x in name_lower for x in ["right-", "ctx-rh-", "-rh-", "rh-", "rh_"]):
        return "right"

    # Bilateral/midline/other structures
    return "other"


def image_volume_label(mw: "MainWindow") -> str:
    """Return the volume/RGB suffix for a 4D anatomical image, or empty string."""
    image = mw.anatomical_mmap_image
    if image is None or len(getattr(image, "source_shape", ())) != 4:
        return ""
    if image.rgb:
        return " | RGB"
    return f" | Volume {image.volume_index + 1}/{image.source_shape[3]}"


class DataPanelTreeMixin:
    """Mixin implementing debounced tree construction for DataPanelManager."""

    mw: "MainWindow"

    def update_data_panel_display(self) -> None:
        """
        Updates the QTreeWidget in the data panel dock.

        Uses debouncing to prevent expensive rebuilds during rapid-fire calls.
        Multiple calls within 100ms are coalesced into a single update.
        """
        mw = self.mw
        if not mw.data_tree_widget:
            return

        # Initialize debounce timer on first use
        if mw._data_panel_debounce_timer is None:
            mw._data_panel_debounce_timer = QTimer(mw)
            mw._data_panel_debounce_timer.setSingleShot(True)
            mw._data_panel_debounce_timer.timeout.connect(
                mw._perform_data_panel_update
            )

        # Mark that an update is pending and restart the timer
        mw._data_panel_update_pending = True
        mw._data_panel_debounce_timer.start(100)  # 100ms debounce delay

    def perform_data_panel_update(self, *, refresh_selection: bool = False) -> None:
        """
        Performs the actual data panel update (debounced).

        This is called by the debounce timer after the delay expires.
        Blocks signals during rebuild to prevent cascading itemChanged events.
        Delegates tree construction to focused sub-methods for maintainability.
        """
        mw = self.mw
        if not mw._data_panel_update_pending:
            return

        mw._data_panel_update_pending = False

        if not mw.data_tree_widget:
            return

        # Block signals during rebuild to prevent cascading callbacks
        blocked = mw.data_tree_widget.blockSignals(True)
        updates_enabled = mw.data_tree_widget.updatesEnabled()
        mw.data_tree_widget.setUpdatesEnabled(False)
        try:
            selected_mesh_data = None
            current_mesh_key = None
            mesh_expansion = {}
            meshes_expanded = None
            meshes_current = False
            meshes_selected = False
            for item in mw.data_tree_widget.selectedItems():
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(data, dict) and data.get("type") in _SELECTED_MESH_TYPES:
                    selected_mesh_data = data
            current = mw.data_tree_widget.currentItem()
            for item in _tree_items(mw.data_tree_widget):
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(data, dict) and data.get("type") in _PRESERVED_TREE_TYPES:
                    key = _tree_item_key(item)
                    mesh_expansion[key] = (item.isExpanded(), item.isSelected())
                    if item is current:
                        current_mesh_key = key
            scroll = (
                mw.data_tree_widget.verticalScrollBar().value(),
                mw.data_tree_widget.horizontalScrollBar().value(),
            )
            for i in range(mw.data_tree_widget.topLevelItemCount()):
                item = mw.data_tree_widget.topLevelItem(i)
                if item.text(0) == "Meshes":
                    meshes_expanded = item.isExpanded()
                    meshes_current = item is current
                    meshes_selected = item.isSelected()

            # Save expansion state before clearing
            expanded_items = set()
            for i in range(mw.data_tree_widget.topLevelItemCount()):
                item = mw.data_tree_widget.topLevelItem(i)
                if item.text(0) != "Meshes":
                    self._collect_expanded_items(item, "", expanded_items)

            mw.data_tree_widget.clear()
            mw._pending_expanded_items = expanded_items

            # Build each section of the data panel tree
            self._build_tractogram_tree_item()
            self._build_image_tree_item()
            self._build_roi_tree_items()
            mw._build_mesh_tree_items()
            mw.mesh_polyline_manager.build_tree(mw.data_tree_widget, None)
            self._build_odf_tree_item()
            self._build_parcellation_tree_item()

            mw.data_tree_widget.resizeColumnToContents(0)

            # Restore expansion state
            if mw._pending_expanded_items:
                for i in range(mw.data_tree_widget.topLevelItemCount()):
                    item = mw.data_tree_widget.topLevelItem(i)
                    if item.text(0) != "Meshes":
                        self._restore_expanded_items(
                            item, "", mw._pending_expanded_items
                        )
                mw._pending_expanded_items = set()
            if current_mesh_key is not None:
                for item in _tree_items(mw.data_tree_widget):
                    if _tree_item_key(item) == current_mesh_key:
                        mw.data_tree_widget.setCurrentItem(item)
            for i in range(mw.data_tree_widget.topLevelItemCount()):
                root = mw.data_tree_widget.topLevelItem(i)
                if root.text(0) != "Meshes":
                    continue
                if meshes_current:
                    mw.data_tree_widget.setCurrentItem(root)
                root.setSelected(meshes_selected)
                if meshes_expanded is not None:
                    root.setExpanded(meshes_expanded)
            for item in _tree_items(mw.data_tree_widget):
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(data, dict) and data.get("type") in _PRESERVED_TREE_TYPES:
                    key = _tree_item_key(item)
                    if key in mesh_expansion:
                        item.setSelected(mesh_expansion[key][1])
                        item.setExpanded(mesh_expansion[key][0])
            mw.data_tree_widget.doItemsLayout()
            mw.data_tree_widget.verticalScrollBar().setValue(scroll[0])
            mw.data_tree_widget.horizontalScrollBar().setValue(scroll[1])
        finally:
            mw.data_tree_widget.setUpdatesEnabled(updates_enabled)
            mw.data_tree_widget.blockSignals(blocked)

        # Force visual update to ensure checkbox states are immediately reflected
        if mw.data_tree_widget.viewport():
            mw.data_tree_widget.viewport().update()
        if (
            refresh_selection
            or mw.mesh_layers
            or selected_mesh_data is not None
            or self.inspected_mesh_id is not None
        ):
            self.on_data_item_selected()

    def build_mesh_tree_items(self) -> None:
        """Build surface items using UUID identity, independently of file names."""
        mw = self.mw
        if not mw.mesh_layers:
            return
        validate_mesh_forest(mw.mesh_layers)
        root = QTreeWidgetItem(mw.data_tree_widget, ["Meshes"])
        root.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))
        items = {}
        for mesh_id, layer in mw.mesh_layers.items():
            item = QTreeWidgetItem([layer["display_name"]])
            items[mesh_id] = item
            item.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_FileIcon))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(0, Qt.ItemDataRole.UserRole, {"type": "mesh", "id": mesh_id})
            item.setCheckState(
                0,
                Qt.CheckState.Checked if layer["visible"] else Qt.CheckState.Unchecked,
            )
            provenance = layer["coordinate_provenance"]
            item.setToolTip(
                0,
                f"Path: {layer['path']}\nPoints: {layer['point_count']}\n"
                f"Polygons: {layer['polygon_count']}\nStrips: {layer['strip_count']}\n"
                f"Coordinates: {layer['coordinate_system']} (mm)\n"
                f"Choice: {provenance['choice']}\n"
                f"File SPACE: {provenance['metadata']['declarations']}",
            )
        for mesh_id, layer in mw.mesh_layers.items():
            parent_id = layer.get("parent_id")
            parent = root if parent_id is None else items[parent_id]
            parent.addChild(items[mesh_id])
        for mesh_id, item in items.items():
            mw.mesh_polyline_manager.build_tree(item, mesh_id)
        root.setExpanded(True)

    def _build_tractogram_tree_item(self) -> None:
        """Builds the Tractogram section of the data panel tree."""
        mw = self.mw
        if mw.tractogram_data is None:
            return

        tractogram_header = QTreeWidgetItem(mw.data_tree_widget, ["Tractogram"])
        tractogram_header.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))

        bundle_name = (
            os.path.basename(mw.original_trk_path)
            if mw.original_trk_path
            else "Loaded Bundle"
        )

        bundle_item = QTreeWidgetItem(tractogram_header, [bundle_name])
        bundle_item.setFlags(bundle_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        bundle_state = (
            Qt.CheckState.Checked if mw.bundle_is_visible else Qt.CheckState.Unchecked
        )
        bundle_item.setCheckState(0, bundle_state)
        bundle_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "bundle"})

        count = len(mw.visible_indices)
        ext = (
            mw.original_file_extension.upper()
            if mw.original_file_extension
            else "TRK"
        )
        dims = "N/A"
        if mw.tractogram_reference_grid is not None:
            dims = format_tuple(mw.tractogram_reference_grid.shape, precision=0)
        elif mw.original_trk_header and "dimensions" in mw.original_trk_header:
            dims = format_tuple(mw.original_trk_header["dimensions"], precision=0)

        tooltip_text = f"Type: {ext}\nCount: {count}\nDimensions: {dims}"
        bundle_item.setToolTip(0, tooltip_text)

        if mw.original_file_extension != ".trx":
            self.tractogram_data.build_information(bundle_item)
        if mw.original_file_extension == ".trx":
            self.tractogram_data.build_tree(bundle_item)
        elif mw.scalar_data_per_point:
            scalars_root = QTreeWidgetItem(bundle_item, ["Scalars"])
            scalars_root.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))
            for scalar_name in mw.scalar_data_per_point.keys():
                scalar_item = QTreeWidgetItem(scalars_root, [scalar_name])
                if scalar_name == mw.active_scalar_name:
                    font = scalar_item.font(0)
                    font.setBold(True)
                    scalar_item.setFont(0, font)

        bundle_item.setExpanded(True)
        tractogram_header.setExpanded(True)

    def _build_image_tree_item(self) -> None:
        """Builds the Anatomical Image section of the data panel tree."""
        mw = self.mw
        if mw.anatomical_image_data is None:
            return

        image_header = QTreeWidgetItem(mw.data_tree_widget, ["Anatomical Image"])
        image_header.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))

        image_name = (
            os.path.basename(mw.anatomical_image_path)
            if mw.anatomical_image_path
            else "Loaded Image"
        )

        volume_label = mw._image_volume_label()
        image_item = QTreeWidgetItem(image_header, [image_name + volume_label])
        image_item.setFlags(image_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        image_state = (
            Qt.CheckState.Checked if mw.image_is_visible else Qt.CheckState.Unchecked
        )
        image_item.setCheckState(0, image_state)
        image_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "image"})
        shape_str = format_tuple(mw.anatomical_image_data.shape, precision=0)
        image_item.setToolTip(
            0,
            f"Path: {mw.anatomical_image_path}\nShape: {shape_str}"
            + (
                "\nRight-click to select a volume or RGB display."
                if volume_label
                else ""
            ),
        )
        image_header.setExpanded(True)

    def _build_roi_tree_items(self) -> None:
        """Builds the ROI Layers section of the data panel tree."""
        mw = self.mw
        if not mw.roi_layers:
            return

        roi_root_item = QTreeWidgetItem(mw.data_tree_widget, ["ROI Layers"])
        roi_root_item.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))

        for path, roi_info in mw.roi_layers.items():
            roi_name = roi_info.get("display_name", os.path.basename(path))

            state_str = ""
            if path in mw.roi_states:
                if mw.roi_states[path].get("select"):
                    state_str = " [SELECT]"
                elif mw.roi_states[path].get("include"):
                    state_str = " [INCLUDE]"
                elif mw.roi_states[path].get("exclude"):
                    state_str = " [EXCLUDE]"

            display_text = f"{roi_name}{state_str}"
            roi_item = QTreeWidgetItem(roi_root_item, [display_text])

            # Color indicator
            roi_color = roi_info.get("color", (1.0, 0.0, 0.0))
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            c_r = int(roi_color[0] * 255)
            c_g = int(roi_color[1] * 255)
            c_b = int(roi_color[2] * 255)
            color = QColor(c_r, c_g, c_b)

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(2, 2, 12, 12)
            painter.end()

            roi_item.setIcon(0, QIcon(pixmap))

            roi_item.setFlags(roi_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            is_visible = mw.roi_visibility.get(path, True)
            roi_state = Qt.CheckState.Checked if is_visible else Qt.CheckState.Unchecked
            roi_item.setCheckState(0, roi_state)
            roi_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "roi", "path": path})

            shape_str = format_tuple(roi_info["data"].shape, precision=0)
            roi_item.setToolTip(0, f"Path: {path}\nShape: {shape_str}")

        roi_root_item.setExpanded(True)

    def _build_odf_tree_item(self) -> None:
        """Builds the ODF Data section of the data panel tree."""
        mw = self.mw
        if mw.odf_data is None:
            return

        odf_header = QTreeWidgetItem(mw.data_tree_widget, ["ODF Data"])
        odf_header.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))

        odf_name = os.path.basename(mw.odf_path) if mw.odf_path else "Loaded ODF"

        odf_file_item = QTreeWidgetItem(odf_header, [odf_name])
        odf_file_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "odf_data"})
        shape_str = format_tuple(mw.odf_data.shape, precision=0)
        odf_file_item.setToolTip(
            0,
            f"Path: {mw.odf_path}\n"
            f"Shape: {shape_str}\n"
            f"SH Order: {mw.odf_sh_order}",
        )

        # ODF Tunnel View item (checkable for visibility toggle)
        if mw.vtk_panel and (
            mw.vtk_panel.odf_actor is not None
            or getattr(mw, "_session_deferred_odf", False)
        ):
            tunnel_item = QTreeWidgetItem(odf_header, ["ODF Tunnel View"])
            tunnel_item.setFlags(tunnel_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            tunnel_state = (
                Qt.CheckState.Checked
                if mw.odf_tunnel_is_visible
                else Qt.CheckState.Unchecked
            )
            tunnel_item.setCheckState(0, tunnel_state)
            tunnel_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "odf_tunnel"})
            tunnel_item.setToolTip(
                0,
                "Toggle visibility of the ODF Tunnel View.\nRight-click to remove.",
            )

        odf_header.setExpanded(True)

    def _build_parcellation_tree_item(self) -> None:
        """Builds the FreeSurfer Parcellation section of the data panel tree."""
        mw = self.mw
        if mw.parcellation_data is None:
            return

        parcellation_header = QTreeWidgetItem(
            mw.data_tree_widget, ["FreeSurfer Parcellation"]
        )
        parcellation_header.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))

        parc_name = (
            os.path.basename(mw.parcellation_path)
            if mw.parcellation_path
            else "Loaded Parcellation"
        )

        parc_item = QTreeWidgetItem(parcellation_header, [parc_name])
        parc_item.setFlags(parc_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

        parc_visible = mw._parcellation_overlay_visible
        parc_state = Qt.CheckState.Checked if parc_visible else Qt.CheckState.Unchecked
        parc_item.setCheckState(0, parc_state)
        parc_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "parcellation"})

        shape_str = format_tuple(mw.parcellation_data.shape, precision=0)
        if mw._parcellation_label_count is None:
            mw._parcellation_label_count = np.count_nonzero(
                np.unique(mw.parcellation_data)
            )
        n_labels = mw._parcellation_label_count
        parc_item.setToolTip(
            0,
            f"Path: {mw.parcellation_path}\n"
            f"Shape: {shape_str}\n"
            f"Total Labels: {n_labels}",
        )

        # Connected Regions submenu — organized by hemisphere
        connected_labels = mw.parcellation_connected_labels
        if connected_labels:
            self._build_connected_regions_tree(parc_item, connected_labels)

        parc_item.setExpanded(True)
        parcellation_header.setExpanded(True)

    def _build_connected_regions_tree(
        self,
        parent_item: QTreeWidgetItem,
        connected_labels: Set[int],
    ) -> None:
        """Builds the Connected Regions sub-tree under a parcellation item."""
        mw = self.mw
        connected_item = QTreeWidgetItem(
            parent_item, [f"Connected Regions ({len(connected_labels)})"]
        )
        connected_item.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))

        main_labels = mw.parcellation_main_labels

        # Classify regions by hemisphere
        left_regions: List[Tuple[int, str]] = []
        right_regions: List[Tuple[int, str]] = []
        other_regions: List[Tuple[int, str]] = []

        for label in connected_labels:
            label_name = mw.parcellation_labels.get(int(label), f"Region_{label}")
            hemisphere = classify_region_hemisphere(label_name)
            if hemisphere == "left":
                left_regions.append((label, label_name))
            elif hemisphere == "right":
                right_regions.append((label, label_name))
            else:
                other_regions.append((label, label_name))

        left_regions.sort(key=lambda x: x[1].lower())
        right_regions.sort(key=lambda x: x[1].lower())
        other_regions.sort(key=lambda x: x[1].lower())

        # Build hemisphere folders
        hemisphere_data = [
            ("Left Hemisphere", left_regions),
            ("Right Hemisphere", right_regions),
            ("Bilateral/Other", other_regions),
        ]
        for folder_name, regions in hemisphere_data:
            if not regions:
                continue
            folder = QTreeWidgetItem(
                connected_item, [f"{folder_name} ({len(regions)})"]
            )
            folder.setIcon(0, mw._tree_icon(QStyle.StandardPixmap.SP_DirIcon))
            self._add_region_items(folder, regions, main_labels)
            folder.setExpanded(False)

        connected_item.setExpanded(False)

    def _add_region_items(
        self,
        parent_item: QTreeWidgetItem,
        regions: List[Tuple[int, str]],
        main_labels: Set[int],
        limit: int = 100,
    ) -> None:
        """Adds individual region items to a hemisphere folder."""
        mw = self.mw
        parc_states = mw.parcellation_region_states

        for label, label_name in regions[:limit]:
            display_name = label_name
            state = parc_states.get(int(label), {})
            if state.get("include"):
                display_name = f"{label_name} [INC]"
            elif state.get("exclude"):
                display_name = f"{label_name} [EXC]"

            region_item = QTreeWidgetItem(parent_item, [display_name])
            region_item.setFlags(region_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

            has_actor = int(label) in main_labels
            region_visible = mw.parcellation_region_visibility.get(
                int(label), has_actor
            )
            region_state = (
                Qt.CheckState.Checked if region_visible else Qt.CheckState.Unchecked
            )
            region_item.setCheckState(0, region_state)

            region_item.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {"type": "parcellation_region", "label": int(label)},
            )

            actor_status = "Active (has actor)" if has_actor else "On-demand"
            region_item.setToolTip(
                0,
                f"Label ID: {label}\nStatus: {actor_status}\nToggle to show/hide",
            )

        if len(regions) > limit:
            more_item = QTreeWidgetItem(
                parent_item,
                [f"... and {len(regions) - limit} more regions"],
            )
            more_item.setDisabled(True)

    def _collect_expanded_items(
        self, item: QTreeWidgetItem, path: str, expanded_set: set
    ) -> None:
        """Recursively collects paths of expanded items."""
        # Build path using first part of item text (before any brackets/tags)
        item_text = item.text(0).split(" [")[0].split(" (")[0]
        current_path = f"{path}/{item_text}" if path else item_text

        if item.isExpanded():
            expanded_set.add(current_path)

        for i in range(item.childCount()):
            self._collect_expanded_items(item.child(i), current_path, expanded_set)

    def _restore_expanded_items(
        self, item: QTreeWidgetItem, path: str, expanded_set: set
    ) -> None:
        """Recursively restores expansion state of items."""
        # Build path using first part of item text (before any brackets/tags)
        item_text = item.text(0).split(" [")[0].split(" (")[0]
        current_path = f"{path}/{item_text}" if path else item_text

        if current_path in expanded_set:
            item.setExpanded(True)

        for i in range(item.childCount()):
            self._restore_expanded_items(item.child(i), current_path, expanded_set)

    def refresh_image_volume_label(self) -> None:
        """Update the image entry without losing tree selection or expansion."""
        mw = self.mw
        tree = mw.data_tree_widget
        with signals_blocked(tree):
            for index in range(tree.topLevelItemCount()):
                parent = tree.topLevelItem(index)
                for child_index in range(parent.childCount()):
                    item = parent.child(child_index)
                    metadata = item.data(0, Qt.ItemDataRole.UserRole)
                    if isinstance(metadata, dict) and metadata.get("type") == "image":
                        name = os.path.basename(mw.anatomical_image_path)
                        item.setText(0, name + mw._image_volume_label())
                        item.setToolTip(
                            0,
                            f"Path: {mw.anatomical_image_path}\n"
                            f"Shape: {mw.anatomical_image_data.shape}\n"
                            "Right-click to select a volume or RGB display.",
                        )
                        return
