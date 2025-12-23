# -*- coding: utf-8 -*-

"""
Scalar Manager for TractEdit application.

Handles scalar-related operations including:
- Slider/float value conversion
- Scalar data range calculation
- Scalar range widget updates
- VTK update triggering
- RAS coordinate display and parsing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt6.QtCore import pyqtSlot

from ..utils import ColorMode, SLIDER_PRECISION

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class ScalarManager:
    """
    Manages scalar-related operations.

    This class handles:
    - Float to integer slider value conversion
    - Scalar data range calculation
    - Widget synchronization for scalar controls
    - VTK update triggering for scalar visualization
    - RAS coordinate display and input parsing
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the scalar manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def float_to_int_slider(self, float_val: float) -> int:
        """Maps a float value from the data range to the slider's integer range."""
        mw = self.mw
        data_min = mw.scalar_data_min
        data_max = mw.scalar_data_max

        if (data_max - data_min) == 0:
            return 0

        # Clamp value to be within the data range
        float_val = max(data_min, min(data_max, float_val))

        percent = (float_val - data_min) / (data_max - data_min)
        return int(round(percent * SLIDER_PRECISION))

    def int_slider_to_float(self, slider_val: int) -> float:
        """Maps an integer slider value back to the float data range."""
        mw = self.mw
        data_min = mw.scalar_data_min
        data_max = mw.scalar_data_max

        if (data_max - data_min) == 0:
            return data_min

        percent = float(slider_val) / SLIDER_PRECISION
        return data_min + percent * (data_max - data_min)

    def update_scalar_data_range(self) -> None:
        """Calculates the min/max range from the active scalar data."""
        mw = self.mw

        if not mw.active_scalar_name or not mw.scalar_data_per_point:
            logger.info("Scalar range: No active scalar data to calculate range from.")
            return

        scalar_sequence = mw.scalar_data_per_point.get(mw.active_scalar_name)
        if not scalar_sequence:
            logger.info("Scalar range: Active scalar list is empty.")
            return

        try:
            valid_scalars = (s for s in scalar_sequence if s is not None and s.size > 0)

            all_scalars_flat = np.concatenate(list(valid_scalars))
            if all_scalars_flat.size == 0:
                logger.info("Scalar range: Concatenated scalar data is empty.")
                return

            data_min_val = np.min(all_scalars_flat)
            data_max_val = np.max(all_scalars_flat)

            # Handle edge case where all data is the same value
            if data_min_val == data_max_val:
                mw.scalar_data_min = data_min_val - 0.5
                mw.scalar_data_max = data_max_val + 0.5
            else:
                mw.scalar_data_min = data_min_val
                mw.scalar_data_max = data_max_val

            mw.scalar_min_val = mw.scalar_data_min
            mw.scalar_max_val = mw.scalar_data_max

            self.update_scalar_range_widgets()

        except Exception as e:
            logger.warning(f"Error calculating scalar data range: {e}")
            mw.scalar_data_min = 0.0
            mw.scalar_data_max = 1.0
            mw.scalar_min_val = 0.0
            mw.scalar_max_val = 1.0
            self.update_scalar_range_widgets()

    def update_scalar_range_widgets(self) -> None:
        """Updates the spinbox and slider widgets with current range and values."""
        mw = self.mw

        if not mw.scalar_min_spinbox or not mw.scalar_max_spinbox:
            return

        # Block signals to prevent feedback loops
        mw.scalar_min_spinbox.blockSignals(True)
        mw.scalar_max_spinbox.blockSignals(True)
        mw.scalar_min_slider.blockSignals(True)
        mw.scalar_max_slider.blockSignals(True)

        # Set the allowed range for the spinboxes
        mw.scalar_min_spinbox.setRange(mw.scalar_data_min, mw.scalar_data_max)
        mw.scalar_max_spinbox.setRange(mw.scalar_data_min, mw.scalar_data_max)

        # Set the current values
        mw.scalar_min_spinbox.setValue(mw.scalar_min_val)
        mw.scalar_max_spinbox.setValue(mw.scalar_max_val)

        # Set the slider values
        mw.scalar_min_slider.setValue(self.float_to_int_slider(mw.scalar_min_val))
        mw.scalar_max_slider.setValue(self.float_to_int_slider(mw.scalar_max_val))

        # Unblock signals
        mw.scalar_min_spinbox.blockSignals(False)
        mw.scalar_max_spinbox.blockSignals(False)
        mw.scalar_min_slider.blockSignals(False)
        mw.scalar_max_slider.blockSignals(False)

    def slider_value_changed(self, slider_val: int) -> None:
        """
        Slot for when slider value changes.
        Updates the corresponding spinbox, but does NOT trigger VTK update.
        """
        mw = self.mw
        float_val = self.int_slider_to_float(slider_val)

        if mw.sender() == mw.scalar_min_slider:
            mw.scalar_min_val = float_val
            mw.scalar_min_spinbox.blockSignals(True)
            mw.scalar_min_spinbox.setValue(float_val)
            mw.scalar_min_spinbox.blockSignals(False)
            # Ensure min slider doesn't cross max slider
            if slider_val > mw.scalar_max_slider.value():
                mw.scalar_max_slider.blockSignals(True)
                mw.scalar_max_slider.setValue(slider_val)
                mw.scalar_max_slider.blockSignals(False)

        elif mw.sender() == mw.scalar_max_slider:
            mw.scalar_max_val = float_val
            mw.scalar_max_spinbox.blockSignals(True)
            mw.scalar_max_spinbox.setValue(float_val)
            mw.scalar_max_spinbox.blockSignals(False)
            # Ensure max slider doesn't cross min slider
            if slider_val < mw.scalar_min_slider.value():
                mw.scalar_min_slider.blockSignals(True)
                mw.scalar_min_slider.setValue(slider_val)
                mw.scalar_min_slider.blockSignals(False)

    def spinbox_value_changed(self) -> None:
        """
        Slot for when spinbox editing is finished.
        Updates sliders and triggers VTK update.
        """
        mw = self.mw
        min_val = mw.scalar_min_spinbox.value()
        max_val = mw.scalar_max_spinbox.value()

        # Ensure min <= max
        if min_val > max_val:
            if mw.sender() == mw.scalar_min_spinbox:
                max_val = min_val
            else:
                min_val = max_val

        mw.scalar_min_val = min_val
        mw.scalar_max_val = max_val

        self.update_scalar_range_widgets()
        self.trigger_vtk_update()

    def reset_scalar_range(self) -> None:
        """Slot to reset the scalar range to the data's full range."""
        mw = self.mw
        mw.scalar_min_val = mw.scalar_data_min
        mw.scalar_max_val = mw.scalar_data_max
        self.update_scalar_range_widgets()
        self.trigger_vtk_update()

    def trigger_vtk_update(self) -> None:
        """
        Validates range and triggers the (slow) VTK actor update.
        Called on slider release or spinbox edit finished.
        """
        mw = self.mw

        # Validation
        min_val = mw.scalar_min_val
        max_val = mw.scalar_max_val

        if min_val > max_val:
            mw.scalar_min_val = max_val
            min_val = max_val

        # Update widgets one last time to be sure they are synced
        self.update_scalar_range_widgets()

        # Trigger Update
        if mw.vtk_panel and mw.current_color_mode == ColorMode.SCALAR:
            mw.vtk_panel.update_main_streamlines_actor()
            mw.vtk_panel.update_status(
                f"Scalar range set to: [{min_val:.3f}, {max_val:.3f}]"
            )

    def update_ras_coordinate_display(self, ras_coords: Optional[np.ndarray]) -> None:
        """
        Updates the RAS coordinate QLineEdit from the VTK panel.
        Called by vtk_panel._update_slow_slice_components.
        """
        mw = self.mw

        if not mw.ras_coordinate_input:
            return

        # Block signals to prevent _on_ras_coordinate_entered from firing
        mw.ras_coordinate_input.blockSignals(True)

        if ras_coords is not None and len(ras_coords) == 3:
            display_x = -ras_coords[0]

            coord_str = f"{display_x:.2f}, {ras_coords[1]:.2f}, {ras_coords[2]:.2f}"
            mw.ras_coordinate_input.setText(coord_str)
        else:
            mw.ras_coordinate_input.setText("--, --, --")

        # Unblock signals
        mw.ras_coordinate_input.blockSignals(False)

    @pyqtSlot()
    def on_ras_coordinate_entered(self) -> None:
        """
        Parses the RAS coordinate QLineEdit and tells VTKPanel to move the slices.
        """
        mw = self.mw

        if not mw.ras_coordinate_input or not mw.vtk_panel:
            return

        # Check if an image is loaded (for the affine)
        if mw.anatomical_image_data is None:
            mw.vtk_panel.update_status(
                "Error: Cannot set RAS, no anatomical image loaded."
            )
            self.update_ras_coordinate_display(None)
            return

        text_value = mw.ras_coordinate_input.text()

        # Parse the text
        try:
            # Split by comma or space
            parts = text_value.replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError(f"Expected 3 coordinates, got {len(parts)}")

            ras_x_input = float(parts[0])
            ras_y = float(parts[1])
            ras_z = float(parts[2])

            # Negate the X-coordinate
            internal_ras_x = -ras_x_input
            ras_coords = np.array([internal_ras_x, ras_y, ras_z])

            # Send to VTKPanel
            mw.vtk_panel.set_slices_from_ras(ras_coords)

        except (ValueError, TypeError) as e:
            mw.vtk_panel.update_status(
                f"Error: Invalid RAS format. Use 'x, y, z'. ({e})"
            )

            # Revert text to whatever the vtk_panel currently thinks is the coordinate
            current_ras = None
            if mw.vtk_panel.current_slice_indices["x"] is not None:
                c = mw.vtk_panel.current_slice_indices
                main_affine = mw.anatomical_image_affine
                if main_affine is not None:
                    current_ras = mw.vtk_panel._voxel_to_world(
                        [c["x"], c["y"], c["z"]], main_affine
                    )

            self.update_ras_coordinate_display(current_ras)
