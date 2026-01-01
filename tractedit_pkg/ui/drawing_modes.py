# -*- coding: utf-8 -*-

"""
Drawing modes manager for TractEdit UI.

Handles drawing mode toggles (pencil, eraser, sphere, rectangle)
and their related UI updates.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from ..utils import ROI_COLORS

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# Drawing Modes Manager Class
# ============================================================================


class DrawingModesManager:
    """
    Manages drawing mode toggles for the main window.

    This class handles:
    - Toggling between different drawing modes (pencil, eraser, sphere, rectangle)
    - Ensuring mutual exclusivity between modes
    - Updating button styles based on active mode
    - Creating new ROIs for drawing
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the drawing modes manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def _get_button_style_default(self) -> str:
        """Returns the default button style from theme manager."""
        if hasattr(self.mw, "theme_manager"):
            return self.mw.theme_manager.get_button_style_default()
        # Fallback light style if theme manager not available
        return """
            QToolButton {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                padding: 2px;
                margin-left: 5px;
            }
            QToolButton:hover {
                background-color: #e8e8e8;
                border: 1px solid #b0b0b0;
            }
            QToolButton:checked {
                background-color: #d0d0d0;
                border: 1px inset #a0a0a0;
                padding: 3px 1px 1px 3px;
            }
        """

    def _get_button_style_active(self) -> str:
        """Returns the active button style from theme manager."""
        if hasattr(self.mw, "theme_manager"):
            return self.mw.theme_manager.get_button_style_active()
        # Fallback active style if theme manager not available
        return """
            QToolButton {
                background-color: rgb(0, 188, 212);
                border: 1px solid #00ACC1;
                border-radius: 5px;
                padding: 2px;
                margin-left: 5px;
            }
            QToolButton:hover {
                background-color: rgb(0, 172, 193);
                border: 1px solid #0097A7;
            }
            QToolButton:checked {
                background-color: rgb(0, 151, 167);
                border: 1px inset #00838F;
                padding: 3px 1px 1px 3px;
            }
        """

    def trigger_new_roi(self) -> None:
        """Creates a new empty ROI image for manual drawing."""
        mw = self.mw

        if mw.anatomical_image_data is None:
            QMessageBox.warning(
                mw,
                "No Image",
                "Please load an anatomical image first.",
            )
            return

        # Create empty array with same shape as anatomical image
        roi_shape = mw.anatomical_image_data.shape
        new_roi_data = np.zeros(roi_shape, dtype=np.uint8)

        # Generate unique name
        mw.manual_roi_counter += 1
        roi_name = f"manual_roi_{mw.manual_roi_counter}"

        # Use anatomical image affine
        roi_affine = mw.anatomical_image_affine.copy()

        # Store in roi_layers
        inv_affine = np.linalg.inv(roi_affine)
        T_main_to_roi = np.dot(inv_affine, mw.anatomical_image_affine)

        # Assign distinct color
        color_idx = (mw.manual_roi_counter - 1) % len(ROI_COLORS)
        roi_color = ROI_COLORS[color_idx]

        mw.roi_layers[roi_name] = {
            "data": new_roi_data,
            "affine": roi_affine,
            "inv_affine": inv_affine,
            "T_main_to_roi": T_main_to_roi,
            "color": roi_color,
        }

        mw.roi_visibility[roi_name] = True
        mw.roi_opacities[roi_name] = 1.0

        # Initialize ROI state
        mw.roi_states[roi_name] = {
            "select": False,
            "include": False,
            "exclude": False,
        }

        # Update VTK status
        if mw.vtk_panel:
            mw.vtk_panel.update_status(
                f"Created new ROI: {roi_name} (enable drawing mode and draw to see it)"
            )

        # Set as current drawing target
        mw.current_drawing_roi = roi_name

        # Update UI
        mw._update_action_states()
        mw._update_bundle_info_display()
        mw._update_data_panel_display()  # Show new ROI in data panel

    def toggle_draw_mode(self, checked: bool) -> None:
        """Toggles between drawing mode and navigation mode."""
        mw = self.mw

        # Guard: Prevent activation when no ROIs exist
        if checked and not mw.roi_layers:
            mw.draw_mode_action.setChecked(False)
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    "Draw Mode: No ROI available. Create or load an ROI first."
                )
            return

        mw.is_drawing_mode = checked

        # Turn off other modes if draw mode is activated
        if checked:
            self._deactivate_eraser_mode()
            self._deactivate_sphere_mode()
            self._deactivate_rectangle_mode()

        # Update button style
        if checked:
            mw.draw_mode_button.setStyleSheet(self._get_button_style_active())
        else:
            mw.draw_mode_button.setStyleSheet(self._get_button_style_default())

        # Update VTK panel
        if mw.vtk_panel:
            mw.vtk_panel.set_drawing_mode(checked)

            if checked:
                # Select a default ROI to draw on if none selected
                if not mw.current_drawing_roi and mw.roi_layers:
                    mw.current_drawing_roi = next(iter(mw.roi_layers.keys()))

                if mw.current_drawing_roi:
                    mw.vtk_panel.update_status(
                        f"Drawing mode ENABLED. Drawing on: {mw.current_drawing_roi} (Click or Click+Drag to draw)"
                    )
                else:
                    mw.vtk_panel.update_status(
                        "Drawing mode ENABLED. No ROI selected. Create a new ROI first."
                    )
                    mw.draw_mode_action.setChecked(False)
                    mw.is_drawing_mode = False
            else:
                mw.vtk_panel.update_status(
                    "Drawing mode DISABLED. Navigation mode active."
                )

    def toggle_erase_mode(self, checked: bool) -> None:
        """Toggles between eraser mode and navigation mode."""
        mw = self.mw

        # Guard: Prevent activation when no ROIs exist
        if checked and not mw.roi_layers:
            mw.erase_mode_action.setChecked(False)
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    "Erase Mode: No ROI available. Create or load an ROI first."
                )
            return

        mw.is_eraser_mode = checked

        # Turn off other modes if eraser mode is activated
        if checked:
            self._deactivate_draw_mode()
            self._deactivate_sphere_mode()
            self._deactivate_rectangle_mode()

        # Update button style
        if checked:
            mw.erase_mode_button.setStyleSheet(self._get_button_style_active())
        else:
            mw.erase_mode_button.setStyleSheet(self._get_button_style_default())

        # Update VTK panel
        if mw.vtk_panel:
            mw.vtk_panel.set_drawing_mode(checked, is_eraser=True)

            if checked:
                if not mw.current_drawing_roi and mw.roi_layers:
                    mw.current_drawing_roi = next(iter(mw.roi_layers.keys()))

                if mw.current_drawing_roi:
                    mw.vtk_panel.update_status(
                        f"Eraser mode ENABLED. Erasing from: {mw.current_drawing_roi} (Click or Click+Drag to erase)"
                    )
                else:
                    mw.vtk_panel.update_status(
                        "Eraser mode ENABLED. No ROI selected. Create a new ROI first."
                    )
                    mw.erase_mode_action.setChecked(False)
                    mw.is_eraser_mode = False
            else:
                mw.vtk_panel.update_status(
                    "Eraser mode DISABLED. Navigation mode active."
                )

    def toggle_sphere_mode(self, checked: bool) -> None:
        """Toggles between sphere drawing mode and navigation mode."""
        mw = self.mw

        # Guard: Prevent activation when no ROIs exist
        if checked and not mw.roi_layers:
            mw.sphere_mode_action.setChecked(False)
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    "Sphere Mode: No ROI available. Create or load an ROI first."
                )
            return

        # Turn off other modes
        if checked:
            self._deactivate_draw_mode()
            self._deactivate_eraser_mode()
            self._deactivate_rectangle_mode()

        # Update button style
        if checked:
            mw.sphere_mode_button.setStyleSheet(self._get_button_style_active())
        else:
            mw.sphere_mode_button.setStyleSheet(self._get_button_style_default())

        # Show/hide sphere radius control
        if hasattr(mw, "sphere_radius_container"):
            mw.sphere_radius_container.setVisible(checked)

            # Sync spinbox with current sphere radius if one exists
            if (
                checked
                and mw.vtk_panel
                and hasattr(mw.vtk_panel, "sphere_params_per_roi")
                and mw.current_drawing_roi in mw.vtk_panel.sphere_params_per_roi
            ):
                current_radius = mw.vtk_panel.sphere_params_per_roi[
                    mw.current_drawing_roi
                ].get("radius", 5.0)
                mw.sphere_radius_spinbox.blockSignals(True)
                mw.sphere_radius_spinbox.setValue(current_radius)
                mw.sphere_radius_spinbox.blockSignals(False)

        # Update VTK panel
        if mw.vtk_panel:
            try:
                mw.vtk_panel.set_drawing_mode(
                    checked, is_eraser=False, is_sphere=checked
                )
            except TypeError:
                mw.vtk_panel.set_drawing_mode(checked, is_eraser=False)

            if checked:
                mw.is_sphere_mode = True
                if not mw.current_drawing_roi and mw.roi_layers:
                    mw.current_drawing_roi = next(iter(mw.roi_layers.keys()))

                if mw.current_drawing_roi:
                    mw.vtk_panel.update_status(
                        f"Sphere Mode Active - Drag to draw sphere on ROI: {mw.current_drawing_roi}"
                    )
                else:
                    mw.vtk_panel.update_status("Sphere Mode Active. No ROI selected.")
            else:
                mw.is_sphere_mode = False
                mw.vtk_panel.update_status("Navigation Mode")

    def toggle_rectangle_mode(self, checked: bool) -> None:
        """Toggles between rectangle drawing mode and navigation mode."""
        mw = self.mw

        # Guard: Prevent activation when no ROIs exist
        if checked and not mw.roi_layers:
            mw.rectangle_mode_action.setChecked(False)
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    "Rectangle Mode: No ROI available. Create or load an ROI first."
                )
            return

        # Turn off other modes
        if checked:
            self._deactivate_draw_mode()
            self._deactivate_eraser_mode()
            self._deactivate_sphere_mode()

        # Update button style
        if checked:
            mw.rectangle_mode_button.setStyleSheet(self._get_button_style_active())
        else:
            mw.rectangle_mode_button.setStyleSheet(self._get_button_style_default())

        # Update VTK panel
        if mw.vtk_panel:
            try:
                mw.vtk_panel.set_drawing_mode(
                    checked, is_eraser=False, is_sphere=False, is_rectangle=checked
                )
            except TypeError:
                mw.vtk_panel.set_drawing_mode(checked, is_eraser=False)

            if checked:
                mw.is_rectangle_mode = True
                if not mw.current_drawing_roi and mw.roi_layers:
                    mw.current_drawing_roi = next(iter(mw.roi_layers.keys()))

                if mw.current_drawing_roi:
                    mw.vtk_panel.update_status(
                        f"Rectangle Mode Active - Drag to draw rectangle on ROI: {mw.current_drawing_roi}"
                    )
                else:
                    mw.vtk_panel.update_status(
                        "Rectangle Mode Active. No ROI selected."
                    )
            else:
                mw.is_rectangle_mode = False
                mw.vtk_panel.update_status("Navigation Mode")

    def reset_all_drawing_modes(self) -> None:
        """Resets all drawing modes to inactive state."""
        mw = self.mw

        # Reset Drawing Mode
        mw.is_drawing_mode = False
        mw.draw_mode_action.setChecked(False)
        mw.draw_mode_button.setStyleSheet(self._get_button_style_default())

        # Reset Eraser Mode
        mw.is_eraser_mode = False
        mw.erase_mode_action.setChecked(False)
        mw.erase_mode_button.setStyleSheet(self._get_button_style_default())

        # Reset Sphere Mode
        if hasattr(mw, "is_sphere_mode"):
            mw.is_sphere_mode = False
        mw.sphere_mode_action.setChecked(False)
        mw.sphere_mode_button.setStyleSheet(self._get_button_style_default())
        if hasattr(mw, "sphere_radius_container"):
            mw.sphere_radius_container.setVisible(False)

        # Reset Rectangle Mode
        if hasattr(mw, "is_rectangle_mode"):
            mw.is_rectangle_mode = False
        mw.rectangle_mode_action.setChecked(False)
        mw.rectangle_mode_button.setStyleSheet(self._get_button_style_default())

        # Reset current drawing ROI
        mw.current_drawing_roi = None

        # Update VTK panel
        if mw.vtk_panel:
            mw.vtk_panel.set_drawing_mode(
                False, is_eraser=False, is_sphere=False, is_rectangle=False
            )

    def _deactivate_draw_mode(self) -> None:
        """Deactivates draw mode without triggering the toggle handler."""
        mw = self.mw
        if mw.is_drawing_mode:
            mw.is_drawing_mode = False
            mw.draw_mode_action.setChecked(False)
            mw.draw_mode_button.setStyleSheet(self._get_button_style_default())

    def _deactivate_eraser_mode(self) -> None:
        """Deactivates eraser mode without triggering the toggle handler."""
        mw = self.mw
        if mw.is_eraser_mode:
            mw.is_eraser_mode = False
            mw.erase_mode_action.setChecked(False)
            mw.erase_mode_button.setStyleSheet(self._get_button_style_default())

    def _deactivate_sphere_mode(self) -> None:
        """Deactivates sphere mode without triggering the toggle handler."""
        mw = self.mw
        if getattr(mw, "is_sphere_mode", False):
            mw.is_sphere_mode = False
            mw.sphere_mode_action.setChecked(False)
            mw.sphere_mode_button.setStyleSheet(self._get_button_style_default())
            if hasattr(mw, "sphere_radius_container"):
                mw.sphere_radius_container.setVisible(False)

    def _deactivate_rectangle_mode(self) -> None:
        """Deactivates rectangle mode without triggering the toggle handler."""
        mw = self.mw
        if getattr(mw, "is_rectangle_mode", False):
            mw.is_rectangle_mode = False
            mw.rectangle_mode_action.setChecked(False)
            mw.rectangle_mode_button.setStyleSheet(self._get_button_style_default())
