# -*- coding: utf-8 -*-

"""
Data panel manager for TractEdit UI.

Handles the dockable data panel displaying loaded data items
(bundles, images, ROIs, scalars) and their interaction handlers.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QAction, QActionGroup

from ..utils import ColorMode

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# Data Panel Manager Class
# ============================================================================


class DataPanelManager:
    """
    Manages the data panel dock widget and its tree view.

    This class handles:
    - Creating the dockable data panel
    - Handling item selection changes
    - Updating opacity based on selected items
    - Context menu for ROI operations
    - Updating the display when data changes
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
        mw.data_tree_widget = QTreeWidget(mw)
        mw.data_tree_widget.setHeaderLabels(["Loaded Data"])
        mw.data_tree_widget.setMinimumWidth(200)
        mw.data_tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        mw.data_tree_widget.customContextMenuRequested.connect(
            mw._on_data_panel_context_menu
        )
        mw.data_tree_widget.itemChanged.connect(mw._on_data_panel_item_changed)
        mw.data_tree_widget.itemSelectionChanged.connect(mw._on_data_item_selected)

        # Highlighting Style Sheet
        mw.data_tree_widget.setStyleSheet(
            """
            QTreeWidget {
                outline: 0; /* Removes the dotted focus line */
            }
            QTreeWidget::item {
                padding: 4px;
                border-radius: 4px; /* Subtle rounded corners */
            }
            
            /* --- SELECTION STATES --- */
            QTreeWidget::item:selected {
                background-color: #4a6984; /* Desaturated Steel Blue */
                color: white;
                border: none;
            }
            QTreeWidget::item:selected:!active {
                background-color: #4a6984; /* Keep same color when window loses focus */
                color: white;
                border: none;
            }

            /* --- HOVER STATES --- */
            QTreeWidget::item:hover {
                /* Ultra-subtle tint. No white/bright flash. */
                background-color: rgba(0, 0, 0, 0.03); 
                border: none;
            }
            QTreeWidget::item:selected:hover {
                background-color: #557ba0; /* Slight feedback on the blue selection itself */
                color: white;
            }
        """
        )

        layout.addWidget(mw.data_tree_widget)
        mw.data_dock_widget.setWidget(container)
        mw.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, mw.data_dock_widget)

    def on_data_item_selected(self) -> None:
        """Updates the opacity slider based on the selected item."""
        mw = self.mw

        items = mw.data_tree_widget.selectedItems()
        if not items:
            mw.opacity_slider.setEnabled(False)
            return

        item = items[0]

        parent = item.parent()
        if parent and parent.text(0) == "Scalars":
            scalar_name = item.text(0)

            if (
                mw.tractogram_data
                and mw.scalar_data_per_point
                and scalar_name in mw.scalar_data_per_point
            ):
                mw.active_scalar_name = scalar_name

                for i in range(parent.childCount()):
                    child = parent.child(i)
                    font = child.font(0)
                    font.setBold(child.text(0) == scalar_name)
                    child.setFont(0, font)

                mw._update_scalar_data_range()
                mw.scalar_range_initialized = True

                if mw.current_color_mode != ColorMode.SCALAR:
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

        mw.opacity_slider.blockSignals(True)  # Prevent feedback

        if itype == "bundle":
            val = mw.bundle_opacity
            mw.opacity_slider.setEnabled(True)
        elif itype == "image":
            val = mw.image_opacity
            mw.opacity_slider.setEnabled(True)
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
                mw.vtk_panel.update_status(status_msg)
        else:
            mw.opacity_slider.setEnabled(False)

        mw.opacity_slider.setValue(int(val * 100))
        mw.opacity_slider.blockSignals(False)

    def on_opacity_slider_changed(self, value: int) -> None:
        """Updates the opacity of the selected item."""
        mw = self.mw
        float_val = value / 100.0

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

    @pyqtSlot(QTreeWidgetItem, int)
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

        # Block signals to prevent recursion
        if mw.data_tree_widget:
            mw.data_tree_widget.blockSignals(True)

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

        except Exception as e:
            logger.error(f"Error handling item visibility change: {e}", exc_info=True)
        finally:
            # Unblock signals
            if mw.data_tree_widget:
                mw.data_tree_widget.blockSignals(False)

    def on_data_panel_context_menu(self, position) -> None:
        """Grouped Logic Modes with Radio Buttons."""
        mw = self.mw

        item = mw.data_tree_widget.itemAt(position)
        if not item:
            return

        item_data = item.data(0, Qt.ItemDataRole.UserRole)

        if item_data and isinstance(item_data, dict) and item_data.get("type") == "roi":
            roi_path = item_data.get("path")

            # Ensure state dict exists
            if roi_path not in mw.roi_states:
                mw.roi_states[roi_path] = {
                    "select": False,
                    "include": False,
                    "exclude": False,
                }

            menu = QMenu(mw)

            # Logic Mode Section
            logic_menu = menu.addMenu("Logic Mode")

            # Create action group for exclusive selection
            logic_group = QActionGroup(mw)
            logic_group.setExclusive(True)

            # None (reset) action
            none_action = QAction("None (Reset)", mw)
            none_action.setCheckable(True)
            none_action.setChecked(
                not mw.roi_states[roi_path]["select"]
                and not mw.roi_states[roi_path]["include"]
                and not mw.roi_states[roi_path]["exclude"]
            )
            none_action.triggered.connect(
                lambda: mw._set_roi_logic_mode(roi_path, "none")
            )
            logic_group.addAction(none_action)
            logic_menu.addAction(none_action)

            # Select action
            select_action = QAction("Select (Show Intersecting)", mw)
            select_action.setCheckable(True)
            select_action.setChecked(mw.roi_states[roi_path]["select"])
            select_action.triggered.connect(
                lambda: mw._set_roi_logic_mode(roi_path, "select")
            )
            logic_group.addAction(select_action)
            logic_menu.addAction(select_action)

            # Include action
            include_action = QAction("Include (Filter to Intersecting)", mw)
            include_action.setCheckable(True)
            include_action.setChecked(mw.roi_states[roi_path]["include"])
            include_action.triggered.connect(
                lambda: mw._set_roi_logic_mode(roi_path, "include")
            )
            logic_group.addAction(include_action)
            logic_menu.addAction(include_action)

            # Exclude action
            exclude_action = QAction("Exclude (Filter Out Intersecting)", mw)
            exclude_action.setCheckable(True)
            exclude_action.setChecked(mw.roi_states[roi_path]["exclude"])
            exclude_action.triggered.connect(
                lambda: mw._set_roi_logic_mode(roi_path, "exclude")
            )
            logic_group.addAction(exclude_action)
            logic_menu.addAction(exclude_action)

            menu.addSeparator()

            # Save ROI Action
            save_action = QAction("Save ROI As...", mw)
            save_action.triggered.connect(lambda: mw._save_roi_action(roi_path))
            menu.addAction(save_action)

            # Rename ROI Action
            rename_action = QAction("Rename ROI...", mw)
            rename_action.triggered.connect(lambda: mw._rename_roi_action(roi_path))
            menu.addAction(rename_action)

            # Standard Actions
            change_color_action = QAction("Change Color...", mw)
            change_color_action.triggered.connect(
                lambda: mw._change_roi_color_action(roi_path)
            )
            menu.addAction(change_color_action)

            remove_action = QAction("Remove ROI Layer", mw)
            remove_action.triggered.connect(
                lambda: mw._remove_roi_layer_action(roi_path)
            )
            menu.addAction(remove_action)

            menu.exec(mw.data_tree_widget.mapToGlobal(position))

        # Context menu for ODF Tunnel View
        elif (
            item_data
            and isinstance(item_data, dict)
            and item_data.get("type") == "odf_tunnel"
        ):
            menu = QMenu(mw)

            # Remove Tunnel View action
            remove_tunnel_action = QAction("Remove Tunnel View", mw)
            remove_tunnel_action.setStatusTip("Remove the ODF tunnel visualization")
            remove_tunnel_action.triggered.connect(self._remove_odf_tunnel_view)
            menu.addAction(remove_tunnel_action)

            menu.exec(mw.data_tree_widget.mapToGlobal(position))

        # Context menu for ODF Data (the parent file item)
        elif (
            item_data
            and isinstance(item_data, dict)
            and item_data.get("type") == "odf_data"
        ):
            menu = QMenu(mw)

            # Show ODF Tunnel action (same as View menu)
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

            # Remove ODF Data action
            remove_odf_action = QAction("Remove ODF Data", mw)
            remove_odf_action.setStatusTip("Remove ODF data and tunnel visualization")
            remove_odf_action.triggered.connect(mw._remove_odf_data)
            menu.addAction(remove_odf_action)

            menu.exec(mw.data_tree_widget.mapToGlobal(position))

        # Context menu for parcellation filename
        elif (
            item_data
            and isinstance(item_data, dict)
            and item_data.get("type") == "parcellation"
        ):
            menu = QMenu(mw)

            # Calculate Intersection action
            recalc_action = QAction("Calculate Intersection", mw)
            recalc_action.setStatusTip(
                "Recalculate parcellation intersections based on current visible streamlines"
            )
            recalc_action.triggered.connect(
                lambda: mw.connectivity_manager.recalculate_all_intersections()
            )
            menu.addAction(recalc_action)

            menu.addSeparator()

            # Remove Parcellation action
            remove_action = QAction("Remove Parcellation", mw)
            remove_action.setStatusTip("Remove all parcellation data and free cache")
            remove_action.triggered.connect(
                lambda: mw.connectivity_manager.remove_parcellation()
            )
            menu.addAction(remove_action)

            menu.exec(mw.data_tree_widget.mapToGlobal(position))

        # Context menu for Connected Regions folder
        elif item.text(0).startswith("Connected Regions"):
            menu = QMenu(mw)

            show_all_action = QAction("Show All Regions", mw)
            show_all_action.triggered.connect(lambda: self._toggle_all_regions(True))
            menu.addAction(show_all_action)

            hide_all_action = QAction("Hide All Regions", mw)
            hide_all_action.triggered.connect(lambda: self._toggle_all_regions(False))
            menu.addAction(hide_all_action)

            menu.addSeparator()

            # Clear all region filters
            clear_filters_action = QAction("Clear All Region Filters", mw)
            clear_filters_action.triggered.connect(self._clear_all_region_filters)
            menu.addAction(clear_filters_action)

            menu.exec(mw.data_tree_widget.mapToGlobal(position))

        # Context menu for Left Hemisphere folder
        elif item.text(0).startswith("Left Hemisphere"):
            menu = QMenu(mw)

            show_all_action = QAction("Show All Left Regions", mw)
            show_all_action.triggered.connect(
                lambda: self._toggle_hemisphere_regions("left", True)
            )
            menu.addAction(show_all_action)

            hide_all_action = QAction("Hide All Left Regions", mw)
            hide_all_action.triggered.connect(
                lambda: self._toggle_hemisphere_regions("left", False)
            )
            menu.addAction(hide_all_action)

            menu.exec(mw.data_tree_widget.mapToGlobal(position))

        # Context menu for Right Hemisphere folder
        elif item.text(0).startswith("Right Hemisphere"):
            menu = QMenu(mw)

            show_all_action = QAction("Show All Right Regions", mw)
            show_all_action.triggered.connect(
                lambda: self._toggle_hemisphere_regions("right", True)
            )
            menu.addAction(show_all_action)

            hide_all_action = QAction("Hide All Right Regions", mw)
            hide_all_action.triggered.connect(
                lambda: self._toggle_hemisphere_regions("right", False)
            )
            menu.addAction(hide_all_action)

            menu.exec(mw.data_tree_widget.mapToGlobal(position))

        # Context menu for Bilateral/Other folder
        elif item.text(0).startswith("Bilateral/Other"):
            menu = QMenu(mw)

            show_all_action = QAction("Show All Bilateral Regions", mw)
            show_all_action.triggered.connect(
                lambda: self._toggle_hemisphere_regions("other", True)
            )
            menu.addAction(show_all_action)

            hide_all_action = QAction("Hide All Bilateral Regions", mw)
            hide_all_action.triggered.connect(
                lambda: self._toggle_hemisphere_regions("other", False)
            )
            menu.addAction(hide_all_action)

            menu.exec(mw.data_tree_widget.mapToGlobal(position))

        # Context menu for individual parcellation regions
        elif (
            item_data
            and isinstance(item_data, dict)
            and item_data.get("type") == "parcellation_region"
        ):
            label = item_data.get("label")
            if label is not None:
                menu = QMenu(mw)
                logic_menu = menu.addMenu("Logic Mode")

                logic_group = QActionGroup(mw)
                logic_group.setExclusive(True)

                # Get current state
                parc_states = getattr(mw, "parcellation_region_states", {})
                state = parc_states.get(label, {"include": False, "exclude": False})
                is_included = state.get("include", False)
                is_excluded = state.get("exclude", False)
                is_none = not is_included and not is_excluded

                # None action - removes this region's filter
                none_action = QAction("None (Remove Filter)", mw)
                none_action.setCheckable(True)
                none_action.setChecked(is_none)
                none_action.triggered.connect(
                    lambda checked, l=label: mw.connectivity_manager.set_region_logic_mode(
                        l, "none"
                    )
                )
                logic_group.addAction(none_action)
                logic_menu.addAction(none_action)

                # Include action - applies to this region
                include_action = QAction("Include (Keep Intersecting)", mw)
                include_action.setCheckable(True)
                include_action.setChecked(is_included)
                include_action.triggered.connect(
                    lambda checked, l=label: mw.connectivity_manager.set_region_logic_mode(
                        l, "include"
                    )
                )
                logic_group.addAction(include_action)
                logic_menu.addAction(include_action)

                # Exclude action - applies to this region
                exclude_action = QAction("Exclude (Remove Intersecting)", mw)
                exclude_action.setCheckable(True)
                exclude_action.setChecked(is_excluded)
                exclude_action.triggered.connect(
                    lambda checked, l=label: mw.connectivity_manager.set_region_logic_mode(
                        l, "exclude"
                    )
                )
                logic_group.addAction(exclude_action)
                logic_menu.addAction(exclude_action)

                menu.addSeparator()

                # Save Region As action
                save_action = QAction("Save Region As...", mw)
                save_action.setStatusTip("Save this region as a NIfTI file")
                save_action.triggered.connect(
                    lambda checked, l=label: mw.connectivity_manager.save_region_as_nifti(
                        l
                    )
                )
                menu.addAction(save_action)

                menu.exec(mw.data_tree_widget.mapToGlobal(position))

    def _toggle_all_regions(self, visible: bool) -> None:
        """Toggles visibility of all connected parcellation regions."""
        mw = self.mw

        if not hasattr(mw, "parcellation_connected_labels"):
            return

        connected_labels = getattr(mw, "parcellation_connected_labels", set())
        if not connected_labels:
            return

        # Update visibility for all connected regions
        if not hasattr(mw, "parcellation_region_visibility"):
            mw.parcellation_region_visibility = {}

        # Toggle all connected regions (uses on-demand actor creation)
        count = 0
        for label in connected_labels:
            label = int(label)
            mw.parcellation_region_visibility[label] = visible

            # Use the toggle method with batch_mode to skip per-region renders
            mw._toggle_parcellation_region(label, visible, batch_mode=True)
            count += 1

        # Single render after all changes
        if mw.vtk_panel:
            try:
                mw.vtk_panel.render_window.Render()
            except Exception:
                pass

        # Update data panel with signals blocked to prevent callback loops
        if mw.data_tree_widget:
            mw.data_tree_widget.blockSignals(True)
            mw._update_data_panel_display()
            mw.data_tree_widget.blockSignals(False)

        status = "shown" if visible else "hidden"
        mw.vtk_panel.update_status(f"All {count} regions {status}")

    def _toggle_hemisphere_regions(self, hemisphere: str, visible: bool) -> None:
        """
        Toggles visibility of parcellation regions in a specific hemisphere.

        Args:
            hemisphere: 'left', 'right', or 'other'
            visible: True to show, False to hide
        """
        mw = self.mw

        if not hasattr(mw, "parcellation_connected_labels"):
            return

        connected_labels = getattr(mw, "parcellation_connected_labels", set())
        if not connected_labels:
            return

        # Initialize visibility dict if not exists
        if not hasattr(mw, "parcellation_region_visibility"):
            mw.parcellation_region_visibility = {}

        # Filter regions by hemisphere
        matching_labels = []
        for label in connected_labels:
            label_name = mw.parcellation_labels.get(int(label), f"Region_{label}")
            if mw._classify_region_hemisphere(label_name) == hemisphere:
                matching_labels.append(int(label))

        if not matching_labels:
            return

        # Toggle visibility for matching regions
        count = 0
        for label in matching_labels:
            mw.parcellation_region_visibility[label] = visible

            # Use the toggle method with batch_mode to skip per-region renders
            mw._toggle_parcellation_region(label, visible, batch_mode=True)
            count += 1

        # Single render after all changes
        if mw.vtk_panel:
            try:
                mw.vtk_panel.render_window.Render()
            except Exception:
                pass

        # Update data panel with signals blocked to prevent callback loops
        if mw.data_tree_widget:
            mw.data_tree_widget.blockSignals(True)
            mw._update_data_panel_display()
            mw.data_tree_widget.blockSignals(False)

        hemisphere_name = {
            "left": "Left Hemisphere",
            "right": "Right Hemisphere",
            "other": "Bilateral/Other",
        }.get(hemisphere, hemisphere)
        status = "shown" if visible else "hidden"
        mw.vtk_panel.update_status(f"{hemisphere_name}: {count} regions {status}")

    def _clear_all_region_filters(self) -> None:
        """Clears all parcellation region include/exclude filters."""
        mw = self.mw

        # Clear all region filter states
        mw.parcellation_region_states = {}

        # Re-apply logic filters
        mw.roi_manager.apply_logic_filters()

        # Update data panel with signals blocked
        if mw.data_tree_widget:
            mw.data_tree_widget.blockSignals(True)
            mw._update_data_panel_display()
            mw.data_tree_widget.blockSignals(False)

        mw.vtk_panel.update_status("All region filters cleared")

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
            mw.view_odf_tunnel_action.blockSignals(True)
            mw.view_odf_tunnel_action.setChecked(False)
            mw.view_odf_tunnel_action.blockSignals(False)

            # Update the data panel
            mw._update_data_panel_display()

            if mw.vtk_panel:
                mw.vtk_panel.update_status("ODF Tunnel View removed")

        except Exception as e:
            logger.error(f"Error removing ODF tunnel view: {e}", exc_info=True)

    def update_data_panel_display(self) -> None:
        """
        Updates the QTreeWidget in the data panel dock.

        Delegates to MainWindow._update_data_panel_display() for the canonical
        implementation with full features (tooltips, state labels, disabled items).
        """
        self.mw._update_data_panel_display()

    def update_bundle_info_display(self) -> None:
        """
        Updates the status label and data panel.

        Delegates to MainWindow._update_bundle_info_display() for consistency.
        """
        self.mw._update_bundle_info_display()
