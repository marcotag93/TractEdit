# -*- coding: utf-8 -*-

"""
Actions manager for TractEdit UI.

Handles creation of QAction objects, menus, and action state management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup

from ..utils import ColorMode, RADIUS_INCREMENT

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


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

    def create_actions(self) -> None:
        """Creates QAction objects used in menus and potentially toolbars."""
        mw = self.mw

        # File Actions
        mw.load_file_action = QAction("&Load trk/tck/trx/vtk/vtp...", mw)
        mw.load_file_action.setStatusTip("Load a trk, tck or trx streamline file")
        mw.load_file_action.triggered.connect(mw._trigger_load_streamlines)

        # Replace Bundle Action
        mw.replace_bundle_action = QAction("&Replace Bundle...", mw)
        mw.replace_bundle_action.setStatusTip(
            "Replace the current bundle without removing the anatomical image"
        )
        mw.replace_bundle_action.triggered.connect(mw._trigger_replace_bundle)

        # Load Anatomical Image Action
        mw.load_bg_image_action = QAction("Load &Image...", mw)
        mw.load_bg_image_action.setStatusTip(
            "Load a NIfTI image (.nii, .nii.gz) as background"
        )
        mw.load_bg_image_action.triggered.connect(mw._trigger_load_anatomical_image)
        mw.load_bg_image_action.setEnabled(False)

        # ODF Actions
        mw.load_odf_action = QAction("Load &ODF (SH)...", mw)
        mw.load_odf_action.setStatusTip(
            "Load a NIfTI file containing Spherical Harmonics coefficients"
        )
        mw.load_odf_action.triggered.connect(mw._trigger_load_odf)

        mw.view_odf_tunnel_action = QAction("Show ODF &Tunnel", mw)
        mw.view_odf_tunnel_action.setStatusTip(
            f"Show ODF glyphs masked by current bundle (< {mw.MAX_ODF_STREAMLINES} fibers)"
        )
        mw.view_odf_tunnel_action.setCheckable(True)
        mw.view_odf_tunnel_action.triggered.connect(mw._toggle_odf_tunnel)
        mw.view_odf_tunnel_action.setEnabled(False)

        # Load Parcellation Action
        mw.load_parcellation_action = QAction("Load &Parcellation...", mw)
        mw.load_parcellation_action.setStatusTip(
            "Load a FreeSurfer parcellation/segmentation (aparc+aseg, etc.)"
        )
        mw.load_parcellation_action.triggered.connect(mw._trigger_load_parcellation)

        # Show Parcellation Overlay Action
        mw.view_parcellation_action = QAction("Show &Parcellation Overlay", mw)
        mw.view_parcellation_action.setStatusTip(
            "Show 3D parcellation with regions connected by streamlines highlighted"
        )
        mw.view_parcellation_action.setCheckable(True)
        mw.view_parcellation_action.triggered.connect(mw._toggle_parcellation_overlay)
        mw.view_parcellation_action.setEnabled(False)

        # Close Bundle Action
        mw.close_bundle_action = QAction("&Close Bundle", mw)
        mw.close_bundle_action.setStatusTip(
            "Close the current streamline bundle (keeps image)"
        )
        mw.close_bundle_action.triggered.connect(
            lambda: mw._close_bundle(keep_image=True)
        )
        mw.close_bundle_action.setEnabled(False)

        # Clear Anatomical Image Action
        mw.clear_bg_image_action = QAction("Clear Anatomical Image", mw)
        mw.clear_bg_image_action.setStatusTip("Remove the background anatomical image")
        mw.clear_bg_image_action.triggered.connect(mw._trigger_clear_anatomical_image)
        mw.clear_bg_image_action.setEnabled(False)

        # Load ROI Action
        mw.load_roi_action = QAction("Load &ROI...", mw)
        mw.load_roi_action.setStatusTip(
            "Load a NIfTI image (.nii, .nii.gz) as an ROI overlay"
        )
        mw.load_roi_action.triggered.connect(mw._trigger_load_roi)
        mw.load_roi_action.setEnabled(False)

        # New ROI Action
        mw.new_roi_action = QAction("New &ROI", mw)
        mw.new_roi_action.setStatusTip("Create a new empty ROI for manual drawing")
        mw.new_roi_action.triggered.connect(mw._trigger_new_roi)
        mw.new_roi_action.setEnabled(False)

        # Draw Mode Action (for toolbar)
        mw.draw_mode_action = QAction("Draw Mode", mw)
        mw.draw_mode_action.setStatusTip(
            "Toggle drawing mode for manual ROI editing (1)"
        )
        mw.draw_mode_action.setCheckable(True)
        mw.draw_mode_action.setShortcut("1")
        mw.draw_mode_action.triggered.connect(mw._toggle_draw_mode)
        mw.draw_mode_action.setEnabled(False)

        # Erase Mode Action (for toolbar)
        mw.erase_mode_action = QAction("Erase Mode", mw)
        mw.erase_mode_action.setStatusTip("Toggle eraser mode to erase ROI voxels (2)")
        mw.erase_mode_action.setCheckable(True)
        mw.erase_mode_action.setShortcut("2")
        mw.erase_mode_action.triggered.connect(mw._toggle_erase_mode)
        mw.erase_mode_action.setEnabled(False)

        # Sphere Mode Action (for toolbar)
        mw.sphere_mode_action = QAction("Sphere Mode", mw)
        mw.sphere_mode_action.setStatusTip("Toggle sphere drawing mode (3)")
        mw.sphere_mode_action.setCheckable(True)
        mw.sphere_mode_action.setShortcut("3")
        mw.sphere_mode_action.triggered.connect(mw._toggle_sphere_mode)
        mw.sphere_mode_action.setEnabled(False)

        # Rectangle Mode Action (for toolbar)
        mw.rectangle_mode_action = QAction("Rectangle Mode", mw)
        mw.rectangle_mode_action.setStatusTip("Toggle rectangle drawing mode (4)")
        mw.rectangle_mode_action.setCheckable(True)
        mw.rectangle_mode_action.setShortcut("4")
        mw.rectangle_mode_action.triggered.connect(mw._toggle_rectangle_mode)
        mw.rectangle_mode_action.setEnabled(False)

        # Clear All ROIs Action
        mw.clear_all_rois_action = QAction("Clear All ROIs", mw)
        mw.clear_all_rois_action.setStatusTip("Remove all loaded ROI overlays")
        mw.clear_all_rois_action.triggered.connect(mw._trigger_clear_all_rois)
        mw.clear_all_rois_action.setEnabled(False)

        # Clear All Data Action
        mw.clear_all_data_action = QAction("Clear &All", mw)
        mw.clear_all_data_action.setStatusTip(
            "Clear all loaded data (Streamlines, Image, ROIs)"
        )
        mw.clear_all_data_action.triggered.connect(mw._trigger_clear_all_data)
        mw.clear_all_data_action.setEnabled(False)

        # Save Streamlines Action
        mw.save_file_action = QAction("&Save As...", mw)
        mw.save_file_action.setStatusTip(
            "Save the modified streamlines to a trk, tck or trx file"
        )
        mw.save_file_action.triggered.connect(mw._trigger_save_streamlines)
        mw.save_file_action.setEnabled(False)

        # Screenshot Action
        mw.screenshot_action = QAction("Save &Screenshot", mw)
        mw.screenshot_action.setStatusTip(
            "Save a screenshot of the current view (bundle and image)"
        )
        mw.screenshot_action.setShortcut("Ctrl+P")
        mw.screenshot_action.triggered.connect(mw._trigger_screenshot)
        mw.screenshot_action.setEnabled(False)

        # Export to HTML Action
        mw.export_html_action = QAction("[Experimental] Export to &HTML...", mw)
        mw.export_html_action.setStatusTip(
            "Export an interactive 3D visualization to a self-contained HTML file"
        )
        mw.export_html_action.triggered.connect(mw._trigger_export_html)
        mw.export_html_action.setEnabled(False)

        # Calculate Centroid Action
        mw.calc_centroid_action = QAction("Calculate &Centroid", mw)
        mw.calc_centroid_action.setStatusTip(
            "Calculate and save the centroid (mean) of the current bundle"
        )
        mw.calc_centroid_action.triggered.connect(mw._trigger_calculate_centroid)
        mw.calc_centroid_action.setEnabled(False)

        # Calculate Medoid Action
        mw.calc_medoid_action = QAction("Calculate &Medoid", mw)
        mw.calc_medoid_action.setStatusTip(
            "Calculate and save the medoid (geometric median) of the current bundle"
        )
        mw.calc_medoid_action.triggered.connect(mw._trigger_calculate_medoid)
        mw.calc_medoid_action.setEnabled(False)

        # Compute Connectivity Action
        mw.compute_connectivity_action = QAction("Compute &Connectivity Matrix...", mw)
        mw.compute_connectivity_action.setStatusTip(
            "Compute and save structural connectivity matrix from streamlines and parcellation"
        )
        mw.compute_connectivity_action.triggered.connect(
            mw._trigger_compute_connectivity
        )
        mw.compute_connectivity_action.setEnabled(False)

        # Exit Action
        mw.exit_action = QAction("&Exit", mw)
        mw.exit_action.setStatusTip("Exit the application")
        mw.exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        mw.exit_action.triggered.connect(mw.close)

        # Save Density Map Action
        mw.save_density_map_action = QAction("Save &Density Map...", mw)
        mw.save_density_map_action.setStatusTip(
            "Generate and save a density map (TDI) of the current visible bundle"
        )
        mw.save_density_map_action.triggered.connect(mw._trigger_save_density_map)
        mw.save_density_map_action.setEnabled(False)

        # Geometry Lines/Tubes
        mw.geometry_action_group = QActionGroup(mw)
        mw.geometry_action_group.setExclusive(True)

        mw.geo_lines_action = QAction("Render as &Lines", mw)
        mw.geo_lines_action.setStatusTip("Render streamlines as simple lines (Faster)")
        mw.geo_lines_action.setCheckable(True)
        mw.geo_lines_action.setChecked(True)
        mw.geo_lines_action.triggered.connect(
            lambda: mw._set_geometry_mode(as_tubes=False)
        )
        mw.geometry_action_group.addAction(mw.geo_lines_action)
        mw.geo_lines_action.setEnabled(False)

        mw.geo_tubes_action = QAction("Render as &Tubes", mw)
        mw.geo_tubes_action.setStatusTip(
            "Render streamlines as 3D tubes (Slower, High Quality)"
        )
        mw.geo_tubes_action.setCheckable(True)
        mw.geo_tubes_action.triggered.connect(
            lambda: mw._set_geometry_mode(as_tubes=True)
        )
        mw.geometry_action_group.addAction(mw.geo_tubes_action)
        mw.geo_tubes_action.setEnabled(False)

        # Edit Actions
        mw.undo_action = QAction("&Undo", mw)
        mw.undo_action.setStatusTip("Undo the last deletion")
        mw.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        mw.undo_action.triggered.connect(mw._perform_undo)
        mw.undo_action.setEnabled(False)

        mw.redo_action = QAction("&Redo", mw)
        mw.redo_action.setStatusTip("Redo the last undone deletion")
        mw.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        mw.redo_action.triggered.connect(mw._perform_redo)
        mw.redo_action.setEnabled(False)

        # View Actions (Coloring)
        mw.coloring_action_group = QActionGroup(mw)
        mw.coloring_action_group.setExclusive(True)

        mw.color_default_action = QAction("&Greyscale Color", mw)
        mw.color_default_action.setStatusTip("Color streamlines with greyscale")
        mw.color_default_action.setCheckable(True)
        mw.color_default_action.setChecked(False)
        mw.color_default_action.triggered.connect(
            lambda: mw._set_color_mode(ColorMode.DEFAULT)
        )
        mw.coloring_action_group.addAction(mw.color_default_action)
        mw.color_default_action.setEnabled(False)

        mw.color_orientation_action = QAction("Color by Orientation", mw)
        mw.color_orientation_action.setStatusTip(
            "Color streamlines by local orientation (RGB)"
        )
        mw.color_orientation_action.setCheckable(True)
        mw.color_orientation_action.setChecked(True)
        mw.color_orientation_action.triggered.connect(
            lambda: mw._set_color_mode(ColorMode.ORIENTATION)
        )
        mw.coloring_action_group.addAction(mw.color_orientation_action)
        mw.color_orientation_action.setEnabled(False)

        mw.color_scalar_action = QAction("Color by Scalar", mw)
        mw.color_scalar_action.setStatusTip(
            "Color streamlines by the first loaded scalar value per point"
        )
        mw.color_scalar_action.setCheckable(True)
        mw.color_scalar_action.triggered.connect(
            lambda: mw._set_color_mode(ColorMode.SCALAR)
        )
        mw.coloring_action_group.addAction(mw.color_scalar_action)
        mw.color_scalar_action.setEnabled(False)

        # Command Actions
        mw.clear_select_action = QAction("&Clear Selection", mw)
        mw.clear_select_action.setStatusTip(
            "Clear the current streamline selection (C)"
        )
        mw.clear_select_action.setShortcut("C")
        mw.clear_select_action.triggered.connect(mw._perform_clear_selection)
        mw.clear_select_action.setEnabled(False)

        # Delete Selection Action
        mw.delete_select_action = QAction("&Delete Selection", mw)
        mw.delete_select_action.setStatusTip("Delete the selected streamlines (D)")
        mw.delete_select_action.setShortcut("D")
        mw.delete_select_action.triggered.connect(mw._perform_delete_selection)
        mw.delete_select_action.setEnabled(False)

        # Reset Camera Action
        mw.reset_camera_action = QAction("&Reset Camera", mw)
        mw.reset_camera_action.setStatusTip("Reset the 3D camera view")
        mw.reset_camera_action.triggered.connect(mw._perform_reset_camera)
        mw.reset_camera_action.setEnabled(True)

        # Increase/Decrease Radius Actions
        mw.increase_radius_action = QAction("&Increase Radius", mw)
        mw.increase_radius_action.setStatusTip(
            f"Increase the selection sphere radius (+{RADIUS_INCREMENT}mm)"
        )
        mw.increase_radius_action.setShortcut("+")
        mw.increase_radius_action.triggered.connect(mw._increase_radius)
        mw.increase_radius_action.setEnabled(False)

        mw.decrease_radius_action = QAction("&Decrease Radius", mw)
        mw.decrease_radius_action.setStatusTip(
            f"Decrease the selection sphere radius (-{RADIUS_INCREMENT}mm)"
        )
        mw.decrease_radius_action.setShortcut("-")
        mw.decrease_radius_action.triggered.connect(mw._decrease_radius)
        mw.decrease_radius_action.setEnabled(False)

        mw.hide_sphere_action = QAction("&Hide Selection Sphere", mw)
        mw.hide_sphere_action.setStatusTip("Hide the blue selection sphere (Esc)")
        mw.hide_sphere_action.setShortcut("Esc")
        mw.hide_sphere_action.triggered.connect(mw._hide_sphere)
        mw.hide_sphere_action.setEnabled(True)

        # Help Menu
        mw.about_action = QAction("&About TractEdit...", mw)
        mw.about_action.setStatusTip("Show information about TractEdit")
        mw.about_action.triggered.connect(mw._show_about_dialog)

    def create_menus(self) -> None:
        """Creates the main menu bar and populates it with actions."""
        mw = self.mw
        main_bar: QMenuBar = mw.menuBar()

        # File Menu
        file_menu = main_bar.addMenu("&File")
        file_menu.addAction(mw.load_file_action)
        file_menu.addAction(mw.load_bg_image_action)
        file_menu.addAction(mw.replace_bundle_action)
        file_menu.addAction(mw.load_roi_action)
        file_menu.addAction(mw.new_roi_action)
        file_menu.addAction(mw.load_odf_action)
        file_menu.addAction(mw.load_parcellation_action)

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

        # Help Menu
        help_menu = main_bar.addMenu("&Help")
        help_menu.addAction(mw.about_action)

        # Shortcuts Submenu List
        shortcuts_menu = help_menu.addMenu("Keyboard &Shortcuts")

        # Helper to add static text items
        def add_shortcut_item(text: str) -> None:
            act = QAction(text, mw)
            act.setEnabled(False)
            shortcuts_menu.addAction(act)

        # Selection Group
        shortcuts_menu.addSection("Selection Tools")
        add_shortcut_item("s  :  Select/Deselect at cursor")
        add_shortcut_item("d  :  Delete selection")
        add_shortcut_item("c  :  Clear selection")
        add_shortcut_item("+ / =  :  Increase sphere radius")
        add_shortcut_item("-  :  Decrease sphereradius")

        # Navigation Group
        shortcuts_menu.addSection("Slice Navigation")
        add_shortcut_item("↑ / ↓  :  Axial (Z-axis)")
        add_shortcut_item("← / →  :  Sagittal (X-axis)")
        add_shortcut_item("Ctrl + ↑ / ↓  :  Coronal (Y-axis)")

        # ROI Drawing Group
        shortcuts_menu.addSection("ROI Drawing")
        add_shortcut_item("1  :  Toggle pencil draw mode")
        add_shortcut_item("2  :  Toggle eraser mode")
        add_shortcut_item("3  :  Toggle sphere mode")
        add_shortcut_item("4  :  Toggle rectangle mode")

        # General Group
        shortcuts_menu.addSection("General")
        add_shortcut_item("Ctrl + S  :  Save As")
        add_shortcut_item("Ctrl + Z  :  Undo")
        add_shortcut_item("Ctrl + Y  :  Redo")
        add_shortcut_item("Ctrl + P  :  Screenshot")
        add_shortcut_item("Ctrl + Q  :  Quit")

        help_menu.addSeparator()
        help_menu.addAction(mw.about_action)

    def update_action_states(self) -> None:
        """Enables/disables actions based on current application state."""
        mw = self.mw

        has_streamlines = mw.tractogram_data is not None
        has_odf = mw.odf_data is not None
        has_selection = bool(mw.selected_streamline_indices)
        has_scalars = bool(mw.scalar_data_per_point)
        has_image = mw.anatomical_image_data is not None
        has_any_data = has_streamlines or has_image

        # File Menu
        mw.load_bg_image_action.setEnabled(True)
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
        mw.clear_all_data_action.setEnabled(has_any_data)
        mw.calc_centroid_action.setEnabled(has_streamlines)
        mw.calc_medoid_action.setEnabled(has_streamlines)
        mw.save_file_action.setEnabled(has_streamlines)
        mw.save_density_map_action.setEnabled(has_streamlines)
        mw.screenshot_action.setEnabled(has_any_data)
        mw.export_html_action.setEnabled(has_any_data)

        # Connectivity matrix requires both streamlines and parcellation
        has_parcellation = mw.parcellation_data is not None
        mw.compute_connectivity_action.setEnabled(has_streamlines and has_parcellation)
        mw.view_parcellation_action.setEnabled(has_streamlines and has_parcellation)

        # Edit Menu
        mw.undo_action.setEnabled(bool(mw.unified_undo_stack))
        mw.redo_action.setEnabled(bool(mw.unified_redo_stack))

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
