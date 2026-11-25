# -*- coding: utf-8 -*-

"""
Context menu builders and region visibility actions for the data panel.

Decomposes context menu dispatch by tree item type, parameterizes Logic Mode
and hemisphere folder menus, and unifies region visibility toggling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup

from ...mesh.model import mesh_subtree_postorder
from ...mesh.presentation import MESH_COMPONENT_LABELS
from ...render_batch import batch_render
from ...utils import signals_blocked
from .tree import classify_region_hemisphere

if TYPE_CHECKING:
    from ...main_window import MainWindow

logger = logging.getLogger(__name__)

_MESH_CONTEXT_TYPES = (
    "mesh",
    "mesh_component",
    "mesh_group",
    "mesh_information",
    "mesh_polygon",
    "mesh_polygons",
    "mesh_polyline",
    "mesh_polylines",
    "surface_mapping",
    "surface_mapping_control",
)
_HEMISPHERE_FOLDERS = (
    ("Left Hemisphere", "left", "Left"),
    ("Right Hemisphere", "right", "Right"),
    ("Bilateral/Other", "other", "Bilateral"),
)
_HEMISPHERE_NAMES = {
    "left": "Left Hemisphere",
    "right": "Right Hemisphere",
    "other": "Bilateral/Other",
}


class DataPanelMenusMixin:
    """Mixin implementing context menus and region/ODF helpers for DataPanelManager."""

    mw: "MainWindow"

    def _new_menu(self):
        from .. import data_panel as _dp_mod

        return _dp_mod.QMenu(self.mw)

    def _build_logic_mode_submenu(
        self,
        menu,
        options: Sequence[tuple[str, bool, Callable[[], None]]],
    ) -> None:
        """Build an exclusive QActionGroup 'Logic Mode' submenu."""
        mw = self.mw
        logic_menu = menu.addMenu("Logic Mode")
        logic_group = QActionGroup(mw)
        logic_group.setExclusive(True)
        for label, is_checked, callback in options:
            action = QAction(label, mw)
            action.setCheckable(True)
            action.setChecked(is_checked)
            action.triggered.connect(lambda _checked=False, cb=callback: cb())
            logic_group.addAction(action)
            logic_menu.addAction(action)

    def on_data_panel_context_menu(self, position, tree=None) -> None:
        """Grouped Logic Modes with Radio Buttons."""
        mw = self.mw

        tree = mw.data_tree_widget if tree is None else tree
        item = tree.itemAt(position)
        if not item:
            return

        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        item_type = item_data.get("type") if isinstance(item_data, dict) else None

        if item_type == "tractogram_array":
            self._show_tractogram_array_menu(item_data, position)
            return

        if item_type == "image":
            self._show_image_menu(position)
            return

        if item_type in ("standalone_polyline", "standalone_polylines"):
            self._show_standalone_polyline_menu(item_data, position)
            return

        if item_type in _MESH_CONTEXT_TYPES:
            self._show_mesh_menu(item_data, tree, position)
            return

        if item_type == "roi":
            self._show_roi_menu(item_data, position)
            return

        if item_type == "odf_tunnel":
            self._show_odf_tunnel_menu(position)
            return

        if item_type == "odf_data":
            self._show_odf_data_menu(position)
            return

        if item_type == "parcellation":
            self._show_parcellation_menu(position)
            return

        item_text = item.text(0)
        if item_text.startswith("Connected Regions"):
            self._show_connected_regions_menu(position)
            return

        for prefix, hemisphere, display_label in _HEMISPHERE_FOLDERS:
            if item_text.startswith(prefix):
                self._show_hemisphere_menu(hemisphere, display_label, position)
                return

        if item_type == "parcellation_region":
            self._show_parcellation_region_menu(item_data, position)

    def _show_tractogram_array_menu(self, item_data: dict, position) -> None:
        mw = self.mw
        self.tractogram_data.select(item_data)
        menu = self._new_menu()
        action = menu.addAction("Inspect values...")
        action.triggered.connect(self.tractogram_data.inspect_values)
        try:
            menu.exec(mw.data_tree_widget.mapToGlobal(position))
        finally:
            menu.deleteLater()

    def _show_image_menu(self, position) -> None:
        mw = self.mw
        image = mw.anatomical_mmap_image
        if image is None or len(image.source_shape) != 4:
            return
        menu = self._new_menu()
        volume_action = menu.addAction("Select volume...")
        volume_action.triggered.connect(self.select_image_volume)
        if image.source_shape[3] == 3:
            rgb_action = menu.addAction("Display as RGB")
            rgb_action.setCheckable(True)
            rgb_action.setChecked(image.rgb)
            rgb_action.triggered.connect(
                lambda checked: self._change_image_display(image.volume_index, checked)
            )
        try:
            menu.exec(mw.data_tree_widget.mapToGlobal(position))
        finally:
            menu.deleteLater()

    def _show_standalone_polyline_menu(self, item_data: dict, position) -> None:
        mw = self.mw
        if not mw.mesh_manager._can_mutate():
            return
        menu = self._new_menu()
        try:
            mw.mesh_polyline_manager.add_menu_actions(menu, item_data)
            menu.exec(mw.data_tree_widget.mapToGlobal(position))
        finally:
            menu.deleteLater()

    def _show_mesh_menu(self, item_data: dict, tree, position) -> None:
        mw = self.mw
        mesh_id = item_data.get("id")
        if (
            mesh_id not in mw.mesh_layers
            or mw._session_busy
            or mw._shutdown_requested
        ):
            return
        menu = self._new_menu()
        if mw.mesh_polyline_manager.add_menu_actions(menu, item_data):
            try:
                menu.exec(tree.viewport().mapToGlobal(position))
            finally:
                menu.deleteLater()
            return
        mapping_node = item_data["type"] in (
            "surface_mapping", "surface_mapping_control",
        )
        mw.surface_mapping_manager.add_menu_actions(
            menu, mesh_id, mapping_node=mapping_node,
        )
        if mapping_node:
            try:
                menu.exec(tree.viewport().mapToGlobal(position))
            finally:
                menu.deleteLater()
            return
        component = item_data.get("component", "surface")
        if component == "texture":
            load_action = menu.addAction("Load/Replace Texture...")
            load_action.setEnabled(mw.mesh_manager.can_load_texture(mesh_id))
            load_action.triggered.connect(
                lambda: mw.mesh_manager.load_texture(mesh_id)
            )
        if "texture" in mw.mesh_layers[mesh_id] or component == "texture":
            reload_action = menu.addAction("Reload Texture...")
            reload_action.setEnabled(
                not mw.mesh_manager.texture_busy
                and mw.mesh_manager._can_mutate()
                and mw.mesh_layers[mesh_id]["presentation"]["surface"]["enabled"]
            )
            reload_action.triggered.connect(
                lambda: mw.mesh_manager.request_texture(mesh_id, True, retry=True)
            )
        if component in MESH_COMPONENT_LABELS:
            color_action = menu.addAction("Change Color...")
            color_action.triggered.connect(
                lambda: mw.mesh_manager.change_color(mesh_id, component)
            )
        fit_action = menu.addAction("Fit mesh in view")
        fit_action.triggered.connect(lambda: mw.mesh_manager.fit(mesh_id))
        if item_data["type"] == "mesh":
            if mw.mesh_layers[mesh_id].get("parent_id") is not None:
                top_action = menu.addAction("Move to top level")
                top_action.triggered.connect(
                    lambda: self._set_mesh_parent(mesh_id, None)
                )
        descendants = len(mesh_subtree_postorder(mw.mesh_layers, mesh_id)) - 1
        remove_action = menu.addAction(
            f"Remove mesh and descendants ({descendants})"
            if descendants else "Remove mesh"
        )
        remove_action.triggered.connect(lambda: mw.mesh_manager.remove(mesh_id))
        try:
            menu.exec(tree.viewport().mapToGlobal(position))
        finally:
            menu.deleteLater()

    def _show_roi_menu(self, item_data: dict, position) -> None:
        mw = self.mw
        roi_path = item_data.get("path")

        # Ensure state dict exists
        if roi_path not in mw.roi_states:
            mw.roi_states[roi_path] = {
                "select": False,
                "include": False,
                "exclude": False,
            }

        state = mw.roi_states[roi_path]
        is_none = not state["select"] and not state["include"] and not state["exclude"]

        menu = self._new_menu()
        self._build_logic_mode_submenu(
            menu,
            (
                ("None (Reset)", is_none, lambda: mw._set_roi_logic_mode(roi_path, "none")),
                (
                    "Select (Show Intersecting)",
                    state["select"],
                    lambda: mw._set_roi_logic_mode(roi_path, "select"),
                ),
                (
                    "Include (Filter to Intersecting)",
                    state["include"],
                    lambda: mw._set_roi_logic_mode(roi_path, "include"),
                ),
                (
                    "Exclude (Filter Out Intersecting)",
                    state["exclude"],
                    lambda: mw._set_roi_logic_mode(roi_path, "exclude"),
                ),
            ),
        )

        menu.addSeparator()

        save_action = QAction("Save ROI As...", mw)
        save_action.triggered.connect(
            lambda: mw.roi_manager.save_roi_action(roi_path)
        )
        menu.addAction(save_action)

        rename_action = QAction("Rename ROI...", mw)
        rename_action.triggered.connect(lambda: mw._rename_roi_action(roi_path))
        menu.addAction(rename_action)

        change_color_action = QAction("Change Color...", mw)
        change_color_action.triggered.connect(
            lambda: mw.roi_manager.change_roi_color_action(roi_path)
        )
        menu.addAction(change_color_action)

        remove_action = QAction("Remove ROI Layer", mw)
        remove_action.triggered.connect(
            lambda: mw._remove_roi_layer_action(roi_path)
        )
        menu.addAction(remove_action)

        menu.exec(mw.data_tree_widget.mapToGlobal(position))

    def _show_odf_tunnel_menu(self, position) -> None:
        mw = self.mw
        menu = self._new_menu()

        remove_tunnel_action = QAction("Remove Tunnel View", mw)
        remove_tunnel_action.setStatusTip("Remove the ODF tunnel visualization")
        remove_tunnel_action.triggered.connect(self._remove_odf_tunnel_view)
        menu.addAction(remove_tunnel_action)

        menu.exec(mw.data_tree_widget.mapToGlobal(position))

    def _show_odf_data_menu(self, position) -> None:
        mw = self.mw
        menu = self._new_menu()

        show_tunnel_action = QAction("Show ODF Tunnel", mw)
        show_tunnel_action.setStatusTip(
            f"Show ODF glyphs masked by current bundle (< {mw.MAX_ODF_STREAMLINES} fibers)"
        )
        show_tunnel_action.setCheckable(True)
        show_tunnel_action.setChecked(mw.view_odf_tunnel_action.isChecked())
        show_tunnel_action.setEnabled(mw.view_odf_tunnel_action.isEnabled())
        show_tunnel_action.triggered.connect(
            lambda checked: mw.view_odf_tunnel_action.trigger()
        )
        menu.addAction(show_tunnel_action)

        menu.addSeparator()

        remove_odf_action = QAction("Remove ODF Data", mw)
        remove_odf_action.setStatusTip("Remove ODF data and tunnel visualization")
        remove_odf_action.triggered.connect(mw._remove_odf_data)
        menu.addAction(remove_odf_action)

        menu.exec(mw.data_tree_widget.mapToGlobal(position))

    def _show_parcellation_menu(self, position) -> None:
        mw = self.mw
        menu = self._new_menu()

        recalc_action = QAction("Calculate Intersection", mw)
        recalc_action.setStatusTip(
            "Recalculate parcellation intersections based on current visible streamlines"
        )
        recalc_action.triggered.connect(
            lambda: mw.connectivity_manager.recalculate_all_intersections()
        )
        menu.addAction(recalc_action)

        menu.addSeparator()

        remove_action = QAction("Remove Parcellation", mw)
        remove_action.setStatusTip("Remove all parcellation data and free cache")
        remove_action.triggered.connect(
            lambda: mw.connectivity_manager.remove_parcellation()
        )
        menu.addAction(remove_action)

        menu.exec(mw.data_tree_widget.mapToGlobal(position))

    def _show_connected_regions_menu(self, position) -> None:
        mw = self.mw
        menu = self._new_menu()

        show_all_action = QAction("Show All Regions", mw)
        show_all_action.triggered.connect(lambda: self._toggle_regions(True))
        menu.addAction(show_all_action)

        hide_all_action = QAction("Hide All Regions", mw)
        hide_all_action.triggered.connect(lambda: self._toggle_regions(False))
        menu.addAction(hide_all_action)

        menu.addSeparator()

        clear_filters_action = QAction("Clear All Region Filters", mw)
        clear_filters_action.triggered.connect(self._clear_all_region_filters)
        menu.addAction(clear_filters_action)

        menu.exec(mw.data_tree_widget.mapToGlobal(position))

    def _show_hemisphere_menu(
        self, hemisphere: str, display_label: str, position
    ) -> None:
        mw = self.mw
        menu = self._new_menu()

        show_all_action = QAction(f"Show All {display_label} Regions", mw)
        show_all_action.triggered.connect(
            lambda: self._toggle_regions(True, hemisphere=hemisphere)
        )
        menu.addAction(show_all_action)

        hide_all_action = QAction(f"Hide All {display_label} Regions", mw)
        hide_all_action.triggered.connect(
            lambda: self._toggle_regions(False, hemisphere=hemisphere)
        )
        menu.addAction(hide_all_action)

        menu.exec(mw.data_tree_widget.mapToGlobal(position))

    def _show_parcellation_region_menu(self, item_data: dict, position) -> None:
        mw = self.mw
        label = item_data.get("label")
        if label is None:
            return

        menu = self._new_menu()
        parc_states = getattr(mw, "parcellation_region_states", {})
        state = parc_states.get(label, {"include": False, "exclude": False})
        is_included = state.get("include", False)
        is_excluded = state.get("exclude", False)
        is_none = not is_included and not is_excluded

        self._build_logic_mode_submenu(
            menu,
            (
                (
                    "None (Remove Filter)",
                    is_none,
                    lambda l=label: mw.connectivity_manager.set_region_logic_mode(
                        l, "none"
                    ),
                ),
                (
                    "Include (Keep Intersecting)",
                    is_included,
                    lambda l=label: mw.connectivity_manager.set_region_logic_mode(
                        l, "include"
                    ),
                ),
                (
                    "Exclude (Remove Intersecting)",
                    is_excluded,
                    lambda l=label: mw.connectivity_manager.set_region_logic_mode(
                        l, "exclude"
                    ),
                ),
            ),
        )

        menu.addSeparator()

        save_action = QAction("Save Region As...", mw)
        save_action.setStatusTip("Save this region as a NIfTI file")
        save_action.triggered.connect(
            lambda checked, l=label: mw.connectivity_manager.save_region_as_nifti(l)
        )
        menu.addAction(save_action)

        menu.exec(mw.data_tree_widget.mapToGlobal(position))

    def _toggle_regions(
        self, visible: bool, hemisphere: str | None = None
    ) -> None:
        """Toggle visibility of all or hemisphere-filtered parcellation regions."""
        mw = self.mw

        if not hasattr(mw, "parcellation_connected_labels"):
            return

        connected_labels = getattr(mw, "parcellation_connected_labels", set())
        if not connected_labels:
            return

        if not hasattr(mw, "parcellation_region_visibility"):
            mw.parcellation_region_visibility = {}

        if hemisphere is None:
            target_labels = [int(label) for label in connected_labels]
        else:
            target_labels = []
            for label in connected_labels:
                label_name = mw.parcellation_labels.get(int(label), f"Region_{label}")
                if classify_region_hemisphere(label_name) == hemisphere:
                    target_labels.append(int(label))
            if not target_labels:
                return

        for label in target_labels:
            mw.parcellation_region_visibility[label] = visible
            mw._toggle_parcellation_region(label, visible, batch_mode=True)

        if mw.data_tree_widget:
            with signals_blocked(mw.data_tree_widget):
                mw._update_data_panel_display()

        status = "shown" if visible else "hidden"
        count = len(target_labels)
        if hemisphere is None:
            mw.vtk_panel.update_status(f"All {count} regions {status}")
        else:
            hemisphere_name = _HEMISPHERE_NAMES.get(hemisphere, hemisphere)
            mw.vtk_panel.update_status(f"{hemisphere_name}: {count} regions {status}")

    def _toggle_all_regions(self, visible: bool) -> None:
        """Toggles visibility of all connected parcellation regions."""
        self._toggle_regions(visible)

    def _toggle_hemisphere_regions(self, hemisphere: str, visible: bool) -> None:
        """Toggles visibility of parcellation regions in a specific hemisphere."""
        self._toggle_regions(visible, hemisphere=hemisphere)

    def _clear_all_region_filters(self) -> None:
        """Clears all parcellation region include/exclude filters."""
        mw = self.mw

        mw.parcellation_region_states = {}

        mw.roi_manager._apply_filters_with_skip_protection()

        # Update data panel with signals blocked
        if mw.data_tree_widget:
            with signals_blocked(mw.data_tree_widget):
                mw._update_data_panel_display()

        mw.vtk_panel.update_status("All region filters cleared")

    @batch_render(lambda self: self.mw.vtk_panel)
    def _remove_odf_tunnel_view(self) -> None:
        """
        Removes only the ODF tunnel visualization, keeping the ODF data loaded.

        This allows users to remove the tunnel view without losing the ODF data,
        enabling them to re-enable the tunnel view later from the View menu.
        """
        mw = self.mw

        try:
            # Remove the ODF actor from the scene
            if mw.vtk_panel:
                mw.vtk_panel.remove_odf_actor()

            # Reset visibility state
            mw.odf_tunnel_is_visible = False

            # Sync the menu action state
            with signals_blocked(mw.view_odf_tunnel_action):
                mw.view_odf_tunnel_action.setChecked(False)

            # Update the data panel
            mw._update_data_panel_display()

            if mw.vtk_panel:
                mw.vtk_panel.update_status("ODF Tunnel View removed")

        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Error removing ODF tunnel view: {e}", exc_info=True)
