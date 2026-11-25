# -*- coding: utf-8 -*-

"""
Actions manager for TractEdit UI.

Handles creation of QAction objects, menus, and action state management.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMenuBar
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup

from ..utils import ColorMode, RADIUS_INCREMENT
from ..tractogram_arrays import active_array
from ..logic.state_manager import ActionType

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# Actions Manager Class
# ============================================================================


class ActionsManager:
    """
    Manages QAction objects and menu creation for the main window.

    This class handles:
    - Creating all QAction objects for menus and toolbars
    - Creating and populating the main menu bar
    - Updating action enabled/disabled states based on application state
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the actions manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def _add_action(
        self, attr: str, text: str, slot: Callable, *,
        shortcut: str | QKeySequence | QKeySequence.StandardKey | None = None,
        status_tip: str | None = None, checkable: bool = False, enabled: bool = True,
    ) -> None:
        action = QAction(text, self.mw)
        setattr(self.mw, attr, action)
        if status_tip is not None:
            action.setStatusTip(status_tip)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        action.setEnabled(enabled)

    def create_actions(self) -> None:
        """Creates QAction objects used in menus and potentially toolbars."""
        mw = self.mw

        # File Actions
        self._add_action("open_session_action", "Open Session...", mw._open_session)
        self._add_action("save_session_action", "Save Session", mw._save_session)

        self._add_action(
            "load_file_action", "&Load Tractogram...", mw._trigger_load_streamlines,
            status_tip="Open or replace the tractogram and keep other loaded data",
        )

        self._add_action(
            "load_mesh_action", "Load &Mesh...", mw._trigger_load_mesh,
            status_tip="Load a surface or VTP polylines in subject space",
        )

        self._add_action(
            "load_texture_action", "Load/Replace Texture...",
            lambda: mw.mesh_manager.load_texture(),
            status_tip="Choose a PNG or JPEG image for an OBJ material",
        )

        self._add_action(
            "import_polyline_csv_action", "Import Polyline...",
            lambda: mw.mesh_polyline_manager.trigger_import(),
            status_tip="Import CSV coordinates or VTP lines in subject RAS or LPS millimetres",
        )

        self._add_action(
            "import_surface_polygon_action", "Import Surface Polygon...",
            lambda: mw.mesh_polyline_manager.polygon_editor().import_pair(),
            status_tip="Import a paired VTP or OBJ surface polygon onto a loaded mesh",
            enabled=False,
        )

        # Load Anatomical Image Action
        self._add_action(
            "load_bg_image_action", "Load &Image...", mw._trigger_load_anatomical_image,
            status_tip="Load a NIfTI image (.nii, .nii.gz) as background",
            enabled=False,
        )

        # ODF Actions
        self._add_action(
            "load_odf_action", "Load &ODF (SH)...", mw._trigger_load_odf,
            status_tip="Load a NIfTI file containing Spherical Harmonics coefficients",
        )

        self._add_action(
            "view_odf_tunnel_action", "Show ODF &Tunnel", mw._toggle_odf_tunnel,
            status_tip=f"Show ODF glyphs masked by current bundle (< {mw.MAX_ODF_STREAMLINES} fibers)",
            checkable=True, enabled=False,
        )

        # Load Parcellation Action
        self._add_action(
            "load_parcellation_action", "Load &Parcellation...",
            lambda: mw.connectivity_manager.load_parcellation(),
            status_tip="Load a FreeSurfer parcellation/segmentation (aparc+aseg, etc.)",
        )

        # Show Parcellation Overlay Action
        self._add_action(
            "view_parcellation_action", "Show &Parcellation Overlay",
            mw._toggle_parcellation_overlay,
            status_tip="Show 3D parcellation with regions connected by streamlines highlighted",
            checkable=True, enabled=False,
        )

        # Close Bundle Action
        self._add_action(
            "close_bundle_action", "&Close Bundle", lambda: mw._close_bundle(keep_image=True),
            status_tip="Close the current streamline bundle (keeps image)",
            enabled=False,
        )

        # Clear Anatomical Image Action
        self._add_action(
            "clear_bg_image_action", "Clear Anatomical Image",
            mw._trigger_clear_anatomical_image,
            status_tip="Remove the background anatomical image",
            enabled=False,
        )

        # Load ROI Action
        self._add_action(
            "load_roi_action", "Load &ROI...", mw._trigger_load_roi,
            status_tip="Load a NIfTI image (.nii, .nii.gz) as an ROI overlay",
            enabled=False,
        )

        # New ROI Action
        self._add_action(
            "new_roi_action", "New &ROI", mw._trigger_new_roi,
            status_tip="Create a new empty ROI for manual drawing",
            enabled=False,
        )

        # Draw Mode Action (for toolbar)
        self._add_action(
            "draw_mode_action", "Draw Mode", mw.drawing_modes_manager.toggle_draw_mode,
            status_tip="Toggle drawing mode for manual ROI editing (1)",
            checkable=True, shortcut="1", enabled=False,
        )

        # Erase Mode Action (for toolbar)
        self._add_action(
            "erase_mode_action", "Erase Mode", mw.drawing_modes_manager.toggle_erase_mode,
            status_tip="Toggle eraser mode to erase ROI voxels (2)",
            checkable=True, shortcut="2", enabled=False,
        )

        # Sphere Mode Action (for toolbar)
        self._add_action(
            "sphere_mode_action", "Sphere Mode", mw.drawing_modes_manager.toggle_sphere_mode,
            status_tip="Toggle sphere drawing mode (3)",
            checkable=True, shortcut="3", enabled=False,
        )

        # Rectangle Mode Action (for toolbar)
        self._add_action(
            "rectangle_mode_action", "Rectangle Mode", mw.drawing_modes_manager.toggle_rectangle_mode,
            status_tip="Toggle rectangle drawing mode (4)",
            checkable=True, shortcut="4", enabled=False,
        )

        # Clear All ROIs Action
        self._add_action(
            "clear_all_rois_action", "Clear All ROIs", mw._trigger_clear_all_rois,
            status_tip="Remove all loaded ROI overlays",
            enabled=False,
        )

        # Clear All Data Action
        self._add_action(
            "clear_all_data_action", "Clear &All", mw._trigger_clear_all_data,
            status_tip="Clear all loaded data (Streamlines, Image, ROIs)",
            enabled=False,
        )

        # Save Streamlines Action
        self._add_action(
            "save_file_action", "&Save As...", mw._trigger_save_streamlines,
            status_tip="Save the modified streamlines to a trk, tck or trx file",
            enabled=False,
        )

        # Screenshot Action
        self._add_action(
            "screenshot_action", "Save &Screenshot", mw._trigger_screenshot,
            status_tip="Save a screenshot of the current view (bundle and image)",
            shortcut="Ctrl+P", enabled=False,
        )

        # Export to HTML Action
        self._add_action(
            "export_html_action", "[Experimental] Export to &HTML...", mw._trigger_export_html,
            status_tip="Export an interactive 3D visualization to a self-contained HTML file",
            enabled=False,
        )

        # Calculate Centroid Action
        self._add_action(
            "calc_centroid_action", "Calculate &Centroid", mw._trigger_calculate_centroid,
            status_tip="Calculate and save the centroid (mean) of the current bundle",
            enabled=False,
        )

        # Calculate Medoid Action
        self._add_action(
            "calc_medoid_action", "Calculate &Medoid", mw._trigger_calculate_medoid,
            status_tip="Calculate and save the medoid (geometric median) of the current bundle",
            enabled=False,
        )

        # Compute Connectivity Action
        self._add_action(
            "compute_connectivity_action", "Compute &Connectivity Matrix...",
            lambda: mw.connectivity_manager.compute_and_export(),
            status_tip="Compute and save structural connectivity matrix from streamlines and parcellation",
            enabled=False,
        )

        # Exit Action
        self._add_action(
            "exit_action", "&Exit", mw.close,
            status_tip="Exit the application",
            shortcut=QKeySequence("Ctrl+Q"),
        )

        # Save Density Map Action
        self._add_action(
            "save_density_map_action", "Save &Density Map...", mw._trigger_save_density_map,
            status_tip="Generate and save a density map (TDI) of the current visible bundle",
            enabled=False,
        )

        # Geometry Lines/Tubes
        mw.geometry_action_group = QActionGroup(mw)
        mw.geometry_action_group.setExclusive(True)

        self._add_action(
            "geo_lines_action", "Render as &Lines",
            lambda: mw._set_geometry_mode(as_tubes=False),
            status_tip="Render streamlines as simple lines (Faster)",
            checkable=True, enabled=False,
        )
        mw.geo_lines_action.setChecked(True)
        mw.geometry_action_group.addAction(mw.geo_lines_action)

        self._add_action(
            "geo_tubes_action", "Render as &Tubes",
            lambda: mw._set_geometry_mode(as_tubes=True),
            status_tip="Render streamlines as 3D tubes (Slower, High Quality)",
            checkable=True, enabled=False,
        )
        mw.geometry_action_group.addAction(mw.geo_tubes_action)

        # Edit Actions
        self._add_action(
            "undo_action", "&Undo", mw._perform_undo,
            status_tip="Undo the last edit",
            shortcut=QKeySequence.StandardKey.Undo, enabled=False,
        )

        self._add_action(
            "redo_action", "&Redo", mw._perform_redo,
            status_tip="Redo the last undone edit",
            shortcut=QKeySequence.StandardKey.Redo, enabled=False,
        )

        # View Actions (Coloring)
        mw.coloring_action_group = QActionGroup(mw)
        mw.coloring_action_group.setExclusive(True)

        self._add_action(
            "color_default_action", "&Greyscale Color",
            lambda: mw._set_color_mode(ColorMode.DEFAULT),
            status_tip="Color streamlines with greyscale",
            checkable=True, enabled=False,
        )
        mw.color_default_action.setChecked(False)
        mw.coloring_action_group.addAction(mw.color_default_action)

        self._add_action(
            "color_orientation_action", "Color by Orientation",
            lambda: mw._set_color_mode(ColorMode.ORIENTATION),
            status_tip="Color streamlines by local orientation (RGB)",
            checkable=True, enabled=False,
        )
        mw.color_orientation_action.setChecked(True)
        mw.coloring_action_group.addAction(mw.color_orientation_action)

        self._add_action(
            "color_scalar_action", "Color by Scalar",
            lambda: mw._set_color_mode(ColorMode.SCALAR),
            status_tip="Color streamlines using the active data component",
            checkable=True, enabled=False,
        )
        mw.coloring_action_group.addAction(mw.color_scalar_action)

        # Command Actions
        self._add_action(
            "clear_select_action", "&Clear Selection",
            lambda: mw.state_manager.perform_clear_selection(),
            status_tip="Clear the current streamline selection (C)",
            shortcut="C", enabled=False,
        )

        # Delete Selection Action
        self._add_action(
            "delete_select_action", "&Delete Selection",
            lambda: mw.state_manager.perform_delete_selection(),
            status_tip="Delete the selected streamlines (D)",
            shortcut="D", enabled=False,
        )

        # Reset Camera Action
        self._add_action(
            "reset_camera_action", "&Reset Camera", mw._perform_reset_camera,
            status_tip="Reset the 3D camera view",
        )

        # Increase/Decrease Radius Actions
        self._add_action(
            "increase_radius_action", "&Increase Radius",
            lambda: mw.state_manager.increase_radius(),
            status_tip=f"Increase the selection sphere radius (+{RADIUS_INCREMENT}mm)",
            shortcut="+", enabled=False,
        )

        self._add_action(
            "decrease_radius_action", "&Decrease Radius",
            lambda: mw.state_manager.decrease_radius(),
            status_tip=f"Decrease the selection sphere radius (-{RADIUS_INCREMENT}mm)",
            shortcut="-", enabled=False,
        )

        self._add_action(
            "hide_sphere_action", "&Hide Selection Sphere",
            lambda: mw.state_manager.hide_sphere(),
            status_tip="Hide the blue selection sphere (Esc)",
            shortcut="Esc",
        )

        # Settings Menu - Theme Actions
        self._add_action(
            "theme_light_action", "&Light", mw._set_theme_light,
            status_tip="Use light theme",
            checkable=True,
        )

        self._add_action(
            "theme_dark_action", "&Dark", mw._set_theme_dark,
            status_tip="Use dark theme",
            checkable=True,
        )

        # Auto-Fill Action (Settings)
        self._add_action(
            "auto_fill_action", "Auto-fill Voxels", mw._toggle_auto_fill,
            status_tip="Automatically fill the interior of drawn ROI polygons",
            checkable=True,
        )

        self._add_action(
            "theme_system_action", "&System", mw._set_theme_system,
            status_tip="Follow system theme",
            checkable=True,
        )

        # Create action group for mutual exclusivity
        mw.theme_action_group = QActionGroup(mw)
        mw.theme_action_group.addAction(mw.theme_light_action)
        mw.theme_action_group.addAction(mw.theme_dark_action)
        mw.theme_action_group.addAction(mw.theme_system_action)
        mw.theme_action_group.setExclusive(True)

        # Help Menu
        self._add_action(
            "about_action", "&About TractEdit...", mw._show_about_dialog,
            status_tip="Show information about TractEdit",
        )

    def create_menus(self) -> None:
        """Creates the main menu bar and populates it with actions."""
        mw = self.mw
        main_bar: QMenuBar = mw.menuBar()

        # File Menu
        file_menu = main_bar.addMenu("&File")
        file_menu.addAction(mw.load_file_action)
        file_menu.addAction(mw.load_bg_image_action)
        file_menu.addAction(mw.load_mesh_action)
        file_menu.addAction(mw.load_texture_action)
        file_menu.addAction(mw.import_polyline_csv_action)
        file_menu.addAction(mw.import_surface_polygon_action)
        file_menu.addAction(mw.load_roi_action)
        file_menu.addAction(mw.new_roi_action)
        file_menu.addAction(mw.load_odf_action)
        file_menu.addAction(mw.load_parcellation_action)

        file_menu.addSeparator()
        file_menu.addAction(mw.open_session_action)
        file_menu.addAction(mw.save_session_action)
        file_menu.addSeparator()
        file_menu.addAction(mw.calc_centroid_action)
        file_menu.addAction(mw.calc_medoid_action)
        file_menu.addAction(mw.compute_connectivity_action)
        file_menu.addSeparator()
        file_menu.addAction(mw.close_bundle_action)
        file_menu.addAction(mw.clear_bg_image_action)
        file_menu.addAction(mw.clear_all_rois_action)
        file_menu.addAction(mw.clear_all_data_action)
        file_menu.addSeparator()
        file_menu.addAction(mw.save_file_action)
        file_menu.addAction(mw.save_density_map_action)
        file_menu.addAction(mw.screenshot_action)
        file_menu.addAction(mw.export_html_action)
        file_menu.addSeparator()
        file_menu.addAction(mw.exit_action)

        # Edit Menu
        edit_menu = main_bar.addMenu("&Edit")
        edit_menu.addAction(mw.undo_action)
        edit_menu.addAction(mw.redo_action)

        # View Menu
        view_menu = main_bar.addMenu("&View")

        # Color Sub-menu
        color_menu = view_menu.addMenu("Streamline Color")
        color_menu.addAction(mw.color_default_action)
        color_menu.addAction(mw.color_orientation_action)
        color_menu.addAction(mw.color_scalar_action)

        # Geometry Sub-menu
        geo_menu = view_menu.addMenu("Streamline &Geometry")
        geo_menu.addAction(mw.geo_lines_action)
        geo_menu.addAction(mw.geo_tubes_action)

        # ODF Tunnel View
        view_menu.addSeparator()
        view_menu.addAction(mw.view_odf_tunnel_action)
        view_menu.addAction(mw.view_parcellation_action)

        # Dock Panel Toggle
        if mw.data_dock_widget:
            mw.toggle_data_panel_action = mw.data_dock_widget.toggleViewAction()
            mw.toggle_data_panel_action.setText("Data Panel")
            mw.toggle_data_panel_action.setStatusTip("Show/Hide the Data Panel")
            view_menu.addSeparator()
            view_menu.addAction(mw.toggle_data_panel_action)

        # Commands Menu
        commands_menu = main_bar.addMenu("&Commands")
        commands_menu.addAction(mw.reset_camera_action)
        commands_menu.addAction(mw.screenshot_action)
        commands_menu.addSeparator()
        commands_menu.addAction(mw.clear_select_action)
        commands_menu.addAction(mw.delete_select_action)
        commands_menu.addSeparator()
        commands_menu.addAction(mw.increase_radius_action)
        commands_menu.addAction(mw.decrease_radius_action)
        commands_menu.addSeparator()
        commands_menu.addAction(mw.hide_sphere_action)

        # Settings Menu
        settings_menu = main_bar.addMenu("&Settings")

        settings_menu.addAction(mw.auto_fill_action)
        settings_menu.addSeparator()

        # Theme Sub-menu
        theme_menu = settings_menu.addMenu("&Theme")
        theme_menu.addAction(mw.theme_light_action)
        theme_menu.addAction(mw.theme_dark_action)
        theme_menu.addAction(mw.theme_system_action)

        # Set initial checked state based on current theme
        if hasattr(mw, "theme_manager"):
            from .theme_manager import ThemeMode

            current_mode = mw.theme_manager.current_mode
            if current_mode == ThemeMode.LIGHT:
                mw.theme_light_action.setChecked(True)
            elif current_mode == ThemeMode.DARK:
                mw.theme_dark_action.setChecked(True)
            else:
                mw.theme_system_action.setChecked(True)
        else:
            # Default to system if theme manager not yet initialized
            mw.theme_system_action.setChecked(True)

        # Shortcuts Menu
        shortcuts_menu = main_bar.addMenu("&Shortcuts")

        shortcut_sections = (
            ("Selection Tools", (
                ("s", "Add streamlines at cursor to selection"),
                ("Shift + s", "Remove streamlines at cursor from selection"),
                ("i", "Invert selection"),
                ("d", "Delete selection"),
                ("c", "Clear selection"),
                ("+ / =", "Increase sphere radius"),
                ("-", "Decrease sphere radius"),
            )),
            ("Slice Navigation", (
                ("↑ / ↓", "Axial (Z-axis)"),
                ("← / →", "Sagittal (X-axis)"),
                ("Ctrl + ↑ / ↓", "Coronal (Y-axis)"),
                ("Shift + Scroll", "Slice nav on hovered 2D panel"),
            )),
            ("ROI Drawing", (
                ("1", "Toggle pencil draw mode"),
                ("2", "Toggle eraser mode"),
                ("3", "Toggle sphere mode"),
                ("4", "Toggle rectangle mode"),
            )),
            ("General", (
                ("Ctrl + S", "Save As"),
                ("Ctrl + Z", "Undo"),
                ("Ctrl + Y", "Redo"),
                ("Ctrl + P", "Screenshot"),
                ("Ctrl + Q", "Quit"),
            )),
        )
        for section, entries in shortcut_sections:
            shortcuts_menu.addSection(section)
            for shortcut, label in entries:
                action = QAction(f"{shortcut}  :  {label}", mw)
                action.setEnabled(False)
                shortcuts_menu.addAction(action)

        # Help Menu
        help_menu = main_bar.addMenu("&Help")
        mw.help_menu = help_menu
        help_menu.aboutToShow.connect(mw.update_notifier.dismiss_tip)
        help_menu.aboutToShow.connect(mw.appimage_integration.dismiss_tip)
        help_menu.addAction(mw.update_notifier.check_action)
        help_menu.addAction(mw.update_notifier.release_action)
        help_menu.addAction(mw.update_notifier.auto_action)
        help_menu.addAction(mw.appimage_integration.menu_separator)
        help_menu.addAction(mw.appimage_integration.integrate_action)
        help_menu.addAction(mw.appimage_integration.remove_action)
        help_menu.addSeparator()
        help_menu.addAction(mw.about_action)

    def update_action_states(self) -> None:
        """Enables/disables actions based on current application state."""
        mw = self.mw

        has_streamlines = mw.tractogram_data is not None
        has_odf = mw.odf_data is not None
        has_selection = bool(mw.selected_streamline_indices)
        has_scalars = bool(mw.scalar_data_per_point)
        if getattr(mw, "scalar_display", None) is not None:
            field = active_array(mw)
            has_scalars = (
                field is not None
                and 0 <= mw.scalar_display["component"] < field.components
            )
        has_image = mw.anatomical_image_data is not None
        has_meshes = bool(mw.mesh_layers)
        has_any_data = (
            has_streamlines or has_image or has_meshes or bool(mw.standalone_polylines)
        )
        mesh_loading = mw._mesh_loader_thread is not None
        session_busy = mw._session_busy or mw._shutdown_requested
        texture_busy = getattr(getattr(mw, "mesh_manager", None), "texture_busy", False)
        workers_busy = mesh_loading or texture_busy or any(
            worker.isRunning()
            or getattr(worker, "has_pending_result", False)
            or getattr(worker, "is_consuming_result", False)
            for worker in mw._background_workers
        )

        # File Menu
        mw.load_bg_image_action.setEnabled(True)
        mw.load_mesh_action.setEnabled(not session_busy and not mesh_loading)
        manager = getattr(mw, "mesh_manager", None)
        mw.load_texture_action.setEnabled(
            manager is not None and manager.can_load_texture()
        )
        mw.import_polyline_csv_action.setEnabled(mw.mesh_polyline_manager.can_import())
        mw.import_surface_polygon_action.setEnabled(
            has_meshes and mw.mesh_polyline_manager.can_import()
        )
        mw.open_session_action.setEnabled(not session_busy and not workers_busy)
        mw.save_session_action.setEnabled(
            not session_busy and not workers_busy
        )
        mw.save_session_action.setStatusTip(
            "Save the current editing session"
        )
        mw.view_odf_tunnel_action.setEnabled(has_odf and has_streamlines)
        mw.load_roi_action.setEnabled(has_image)
        mw.new_roi_action.setEnabled(has_image)
        mw.draw_mode_action.setEnabled(bool(mw.roi_layers))
        mw.erase_mode_action.setEnabled(bool(mw.roi_layers))
        mw.sphere_mode_action.setEnabled(bool(mw.roi_layers))
        mw.rectangle_mode_action.setEnabled(bool(mw.roi_layers))
        mw.close_bundle_action.setEnabled(has_streamlines)
        mw.clear_bg_image_action.setEnabled(has_image)
        mw.clear_all_rois_action.setEnabled(bool(mw.roi_layers))
        mw.clear_all_data_action.setEnabled(
            (has_any_data or mesh_loading or mw.mesh_polyline_manager.io_busy)
            and not session_busy
        )
        mw.calc_centroid_action.setEnabled(has_streamlines)
        mw.calc_medoid_action.setEnabled(has_streamlines)
        mw.save_file_action.setEnabled(has_streamlines)
        mw.save_density_map_action.setEnabled(has_streamlines)
        mw.screenshot_action.setEnabled(has_any_data)
        mw.export_html_action.setEnabled(has_streamlines or has_image)

        # Connectivity matrix requires both streamlines and parcellation
        has_parcellation = mw.parcellation_data is not None
        mw.compute_connectivity_action.setEnabled(has_streamlines and has_parcellation)
        mw.view_parcellation_action.setEnabled(has_streamlines and has_parcellation)

        # Edit Menu
        if mw.mesh_polyline_manager.active:
            mw.undo_action.setEnabled(bool(mw.mesh_polyline_points))
            mw.redo_action.setEnabled(mw.mesh_polyline_manager.can_redo_point)
        else:
            for command, stack in (
                (mw.undo_action, mw.unified_undo_stack),
                (mw.redo_action, mw.unified_redo_stack),
            ):
                command.setEnabled(bool(stack) and (
                    stack[-1].get("action_type") != ActionType.POLYLINE_EDIT
                    or mw.mesh_polyline_manager.can_replay_history()
                ))

        # View Menu - Streamline Colors
        mw.color_default_action.setEnabled(has_streamlines)
        mw.color_orientation_action.setEnabled(has_streamlines)
        mw.color_scalar_action.setEnabled(has_streamlines and has_scalars)

        # Commands Menu
        mw.clear_select_action.setEnabled(has_selection)
        mw.delete_select_action.setEnabled(has_selection)
        mw.increase_radius_action.setEnabled(has_streamlines)
        mw.decrease_radius_action.setEnabled(has_streamlines)

        # Geometry Menu
        mw.geo_lines_action.setEnabled(has_streamlines)
        mw.geo_tubes_action.setEnabled(has_streamlines)
        if has_meshes:
            mw.data_panel_manager.update_mesh_items()
        mw.data_panel_manager.update_polyline_properties()
