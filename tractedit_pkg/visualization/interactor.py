# -*- coding: utf-8 -*-

"""
Custom interactor style for 2D orthogonal views.

Provides navigation, drawing, and zoom functionality for the axial,
coronal, and sagittal slice views.
"""

# ============================================================================
# Imports
# ============================================================================

from typing import TYPE_CHECKING
import vtk

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel


# ============================================================================
# Custom Interactor Style Class
# ============================================================================


class CustomInteractorStyle2D(vtk.vtkInteractorStyleImage):
    """
    Custom interactor style for the 2D views.

    Overrides default vtkInteractorStyleImage behavior:
    - Left-click-drag: Navigates slices.
    - Right-click-drag: Zooms (default).
    - Middle-click-drag: Pans (default).
    """

    def __init__(self, vtk_panel_ref: "VTKPanel") -> None:
        """
        Initialize the custom interactor style.

        Args:
            vtk_panel_ref: A reference to the main VTKPanel instance.
        """
        super().__init__()
        self.vtk_panel: "VTKPanel" = vtk_panel_ref

        # Track Ctrl+Scroll radius preview state
        self._is_adjusting_radius = False
        self._preview_radius = 0.0
        self._preview_view_type = "axial"

        # Add event listeners for default VTK events
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.OnLeftButtonDown)
        self.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, self.OnLeftButtonUp)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.OnMouseMove)
        self.AddObserver(
            vtk.vtkCommand.MouseWheelForwardEvent, self.OnMouseWheelForward
        )
        self.AddObserver(
            vtk.vtkCommand.MouseWheelBackwardEvent, self.OnMouseWheelBackward
        )
        self.AddObserver(vtk.vtkCommand.KeyReleaseEvent, self.OnKeyRelease)

    def OnLeftButtonDown(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles left mouse button press."""
        # Call the VTKPanel's navigation or drawing logic
        if self.vtk_panel:
            if self.vtk_panel.is_drawing_mode:
                # Drawing mode - set active flag and draw
                self.vtk_panel.is_drawing_active = True
                self.vtk_panel.draw_stroke_count = 0  # Reset counter for new stroke
                self.vtk_panel._handle_draw_on_2d(self.GetInteractor())
            else:
                # Navigation mode - enable fast render for smooth interaction
                self.vtk_panel.is_navigating_2d = True
                self.vtk_panel._enable_fast_render()
                self.vtk_panel._navigate_2d_view(self.GetInteractor(), event_id)

    def OnLeftButtonUp(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles left mouse button release."""
        if self.vtk_panel:
            # If we were drawing, update the ROI visualization now
            if (
                self.vtk_panel.is_drawing_active
                and self.vtk_panel.main_window.current_drawing_roi
            ):
                self.vtk_panel._finish_drawing()

            # Disable fast render mode if we were navigating
            was_navigating = self.vtk_panel.is_navigating_2d

            self.vtk_panel.is_navigating_2d = False
            self.vtk_panel.is_drawing_active = False  # Stop drawing
            self.vtk_panel._update_slow_slice_components()

            # Restore full quality rendering after navigation
            if was_navigating:
                self.vtk_panel._disable_fast_render()

    def OnMouseMove(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles mouse move."""
        if self.vtk_panel:
            if self.vtk_panel.is_drawing_mode and self.vtk_panel.is_drawing_active:
                # Drawing while dragging (continuous)
                self.vtk_panel._handle_draw_on_2d(self.GetInteractor())
            elif self.vtk_panel.is_navigating_2d:
                # If we are in navigation mode, call the logic
                self.vtk_panel._navigate_2d_view(self.GetInteractor(), event_id)
            else:
                super().OnMouseMove()
        else:
            super().OnMouseMove()

    def OnMouseWheelForward(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles mouse wheel forward."""
        if self.vtk_panel:
            interactor = self.GetInteractor()
            is_ctrl = interactor.GetControlKey()
            if is_ctrl and getattr(self.vtk_panel, "is_sphere_mode", False):
                # Determine view type
                view_type = "axial"
                if interactor == self.vtk_panel.coronal_interactor:
                    view_type = "coronal"
                elif interactor == self.vtk_panel.sagittal_interactor:
                    view_type = "sagittal"

                # Update preview radius (increase by 0.5 mm)
                self._adjust_preview_radius(0.5, view_type)
            else:
                super().OnMouseWheelForward()
                # Update scale bar after zoom
                self._update_scale_bar_for_interactor(interactor)
        else:
            super().OnMouseWheelForward()

    def OnMouseWheelBackward(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles mouse wheel backward."""
        if self.vtk_panel:
            interactor = self.GetInteractor()
            is_ctrl = interactor.GetControlKey()
            if is_ctrl and getattr(self.vtk_panel, "is_sphere_mode", False):
                # Determine view type
                view_type = "axial"
                if interactor == self.vtk_panel.coronal_interactor:
                    view_type = "coronal"
                elif interactor == self.vtk_panel.sagittal_interactor:
                    view_type = "sagittal"

                # Update preview radius (decrease by 0.5 mm)
                self._adjust_preview_radius(-0.5, view_type)
            else:
                super().OnMouseWheelBackward()
                # Update scale bar after zoom
                self._update_scale_bar_for_interactor(interactor)
        else:
            super().OnMouseWheelBackward()

    def _update_scale_bar_for_interactor(self, interactor) -> None:
        """Update scale bar for the view associated with this interactor."""
        if not self.vtk_panel:
            return
        if not self.vtk_panel.scale_bar_manager.is_initialized():
            return

        if interactor == self.vtk_panel.axial_interactor:
            self.vtk_panel.scale_bar_manager.update_view("axial")
        elif interactor == self.vtk_panel.coronal_interactor:
            self.vtk_panel.scale_bar_manager.update_view("coronal")
        elif interactor == self.vtk_panel.sagittal_interactor:
            self.vtk_panel.scale_bar_manager.update_view("sagittal")

    def _adjust_preview_radius(self, delta: float, view_type: str) -> None:
        """
        Adjusts the preview radius without applying to actual ROI.

        Shows a yellow circle preview at the sphere's location.
        """
        mw = self.vtk_panel.main_window
        if not mw or not mw.current_drawing_roi:
            return

        roi_name = mw.current_drawing_roi
        if not hasattr(self.vtk_panel, "sphere_params_per_roi"):
            return
        if roi_name not in self.vtk_panel.sphere_params_per_roi:
            return

        roi_params = self.vtk_panel.sphere_params_per_roi[roi_name]

        # Initialize preview state on first scroll
        if not self._is_adjusting_radius:
            self._is_adjusting_radius = True
            self._preview_radius = roi_params.get("radius", 5.0)
            self._preview_view_type = view_type

        # Update preview radius
        self._preview_radius = max(0.5, self._preview_radius + delta)

        # Get sphere center for preview
        center_3d = roi_params["center"].copy()
        stored_view_type = roi_params.get("view_type", "axial")

        # Undo radiological X-flip for 2D preview display (stored center is in 3D world coords)
        center_display = center_3d.copy()
        if stored_view_type in ["axial", "coronal"]:
            center_display[0] = -center_display[0]

        # Get the appropriate 2D scene
        if stored_view_type == "axial":
            scene = self.vtk_panel.axial_scene
        elif stored_view_type == "coronal":
            scene = self.vtk_panel.coronal_scene
        else:
            scene = self.vtk_panel.sagittal_scene

        if scene:
            # Show yellow circle preview
            self.vtk_panel.drawing_manager._show_radius_preview(
                center_display, self._preview_radius, stored_view_type, scene
            )

        # Update spinbox to show current preview value
        if hasattr(mw, "sphere_radius_spinbox"):
            mw.sphere_radius_spinbox.blockSignals(True)
            mw.sphere_radius_spinbox.setValue(self._preview_radius)
            mw.sphere_radius_spinbox.blockSignals(False)

        self.vtk_panel.update_status(
            f"(Preview) Radius: {self._preview_radius:.1f} mm - Release Ctrl to apply"
        )

    def OnKeyRelease(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Handles key release - applies changes when Ctrl is released."""
        if not self.vtk_panel:
            return

        interactor = self.GetInteractor()
        key = interactor.GetKeySym()

        # Check if Ctrl was released while in preview mode
        if key in ("Control_L", "Control_R"):
            # Check sphere mode
            is_moving_sphere = getattr(self.vtk_panel, "is_moving_sphere", False)
            is_moving_rectangle = getattr(self.vtk_panel, "is_moving_rectangle", False)

            if self._is_adjusting_radius or is_moving_sphere:
                self._apply_sphere_preview()

            if is_moving_rectangle:
                self._apply_rectangle_preview()

    def _apply_sphere_preview(self) -> None:
        """Applies the previewed sphere changes (radius and/or position)."""
        mw = self.vtk_panel.main_window
        if not mw:
            return

        # If we were moving (dragging), trigger finish_drawing
        # If we only scrolled (resize), use spinbox method
        is_moving = getattr(self.vtk_panel, "is_moving_sphere", False)

        if is_moving:
            # Trigger finish drawing to apply the move+resize
            self.vtk_panel._finish_drawing()
        elif self._is_adjusting_radius:
            # Only resized via scroll - use spinbox method
            if hasattr(mw, "_on_sphere_radius_changed"):
                mw._on_sphere_radius_changed()

        # Reset sphere state
        self._is_adjusting_radius = False
        self._preview_radius = 0.0
        self.vtk_panel.is_moving_sphere = False

    def _apply_rectangle_preview(self) -> None:
        """Applies the previewed rectangle changes (position)."""
        mw = self.vtk_panel.main_window
        if not mw:
            return

        is_moving = getattr(self.vtk_panel, "is_moving_rectangle", False)

        if is_moving:
            # Trigger finish drawing to apply the move
            self.vtk_panel._finish_drawing()

        # Reset rectangle state
        self.vtk_panel.is_moving_rectangle = False
