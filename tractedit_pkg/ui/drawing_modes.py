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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from ..utils import ROI_COLORS, signals_blocked

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# Drawing Modes Manager Class
# ============================================================================


@dataclass(frozen=True)
class _ModeSpec:
    flag: str
    enabled: str
    disabled: str
    no_target: str


_MODE_SPECS = {
    "draw": _ModeSpec(
        "is_drawing_mode",
        "Drawing mode ENABLED. Drawing on: {} (Click or Click+Drag to draw)",
        "Drawing mode DISABLED. Navigation mode active.",
        "Drawing mode ENABLED. No ROI selected. Create a new ROI first.",
    ),
    "erase": _ModeSpec(
        "is_eraser_mode",
        "Eraser mode ENABLED. Erasing from: {} (Click or Click+Drag to erase)",
        "Eraser mode DISABLED. Navigation mode active.",
        "Eraser mode ENABLED. No ROI selected. Create a new ROI first.",
    ),
    "sphere": _ModeSpec(
        "is_sphere_mode", "Sphere Mode Active - Drag to draw sphere on ROI: {}",
        "Navigation Mode", "Sphere Mode Active. No ROI selected.",
    ),
    "rectangle": _ModeSpec(
        "is_rectangle_mode", "Rectangle Mode Active - Drag to draw rectangle on ROI: {}",
        "Navigation Mode", "Rectangle Mode Active. No ROI selected.",
    ),
}


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
        return self.mw.theme_manager.get_button_style_default()

    def _get_button_style_active(self) -> str:
        """Returns the active button style from theme manager."""
        return self.mw.theme_manager.get_button_style_active()

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
        roi_shape = mw.anatomical_image_data.shape[:3]
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
        self._toggle_mode("draw", checked)

    def toggle_erase_mode(self, checked: bool) -> None:
        self._toggle_mode("erase", checked)

    def toggle_sphere_mode(self, checked: bool) -> None:
        self._toggle_mode("sphere", checked)

    def toggle_rectangle_mode(self, checked: bool) -> None:
        self._toggle_mode("rectangle", checked)

    def _toggle_mode(self, key: str, checked: bool) -> None:
        mw = self.mw
        spec = _MODE_SPECS[key]
        if checked and not mw.roi_layers:
            getattr(mw, f"{key}_mode_action").setChecked(False)
            if mw.vtk_panel:
                mw.vtk_panel.update_status(
                    f"{key.title()} Mode: No ROI available. Create or load an ROI first."
                )
            return

        freehand = key in ("draw", "erase")
        if freehand:
            setattr(mw, spec.flag, checked)
        if checked:
            for other, other_spec in _MODE_SPECS.items():
                if other != key and getattr(mw, other_spec.flag, False):
                    self._deactivate_mode(other)

        style = (
            self._get_button_style_active() if checked else self._get_button_style_default()
        )
        getattr(mw, f"{key}_mode_button").setStyleSheet(style)
        if key == "sphere":
            self._update_sphere_radius(checked)

        panel = mw.vtk_panel
        if not panel:
            return
        if key == "draw":
            panel.set_drawing_mode(checked)
        elif key == "erase":
            panel.set_drawing_mode(checked, is_eraser=True)
        else:
            try:
                if key == "sphere":
                    panel.set_drawing_mode(checked, is_eraser=False, is_sphere=checked)
                else:
                    panel.set_drawing_mode(
                        checked, is_eraser=False, is_sphere=False, is_rectangle=checked
                    )
            except TypeError:
                panel.set_drawing_mode(checked, is_eraser=False)
            setattr(mw, spec.flag, checked)

        if not checked:
            panel.update_status(spec.disabled)
            return
        if not mw.current_drawing_roi and mw.roi_layers:
            mw.current_drawing_roi = next(iter(mw.roi_layers))
        if mw.current_drawing_roi:
            panel.update_status(spec.enabled.format(mw.current_drawing_roi))
        else:
            panel.update_status(spec.no_target)
            if freehand:
                getattr(mw, f"{key}_mode_action").setChecked(False)
                setattr(mw, spec.flag, False)

    def _update_sphere_radius(self, checked: bool) -> None:
        mw = self.mw
        if not hasattr(mw, "sphere_radius_container"):
            return
        mw.sphere_radius_container.setVisible(checked)
        if (
            checked
            and mw.vtk_panel
            and hasattr(mw.vtk_panel, "sphere_params_per_roi")
            and mw.current_drawing_roi in mw.vtk_panel.sphere_params_per_roi
        ):
            params = mw.vtk_panel.sphere_params_per_roi[mw.current_drawing_roi]
            with signals_blocked(mw.sphere_radius_spinbox):
                mw.sphere_radius_spinbox.setValue(params.get("radius", 5.0))

    def _deactivate_mode(self, key: str) -> None:
        mw = self.mw
        flag = _MODE_SPECS[key].flag
        if key in ("draw", "erase") or hasattr(mw, flag):
            setattr(mw, flag, False)
        getattr(mw, f"{key}_mode_action").setChecked(False)
        getattr(mw, f"{key}_mode_button").setStyleSheet(self._get_button_style_default())
        if key == "sphere" and hasattr(mw, "sphere_radius_container"):
            mw.sphere_radius_container.setVisible(False)

    def reset_all_drawing_modes(self) -> None:
        """Resets all drawing modes to inactive state."""
        for key in _MODE_SPECS:
            self._deactivate_mode(key)
        mw = self.mw
        mw.current_drawing_roi = None
        if mw.vtk_panel:
            mw.vtk_panel.set_drawing_mode(
                False, is_eraser=False, is_sphere=False, is_rectangle=False
            )
