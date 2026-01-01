# -*- coding: utf-8 -*-

"""
Scale bar overlay for 2D orthogonal view panels.

Provides minimalistic scale bars that automatically adjust to the current
zoom level and display appropriate units (mm, µm, nm).
"""

# ============================================================================
# Imports
# ============================================================================

import logging
from typing import Dict, Tuple

import vtk

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Nice scale values to round to (in mm)
NICE_SCALES_MM = [
    0.000001, 0.000002, 0.000005,  # nm range
    0.00001, 0.00002, 0.00005,     # 10s of nm
    0.0001, 0.0002, 0.0005,        # 100s of nm
    0.001, 0.002, 0.005,           # µm range
    0.01, 0.02, 0.05,              # 10s of µm
    0.1, 0.2, 0.5,                 # 100s of µm
    1, 2, 5,                       # mm range
    10, 20, 50,                    # cm range
    100, 200, 500,                 # 10s of cm
]


# ============================================================================
# Helper Functions
# ============================================================================


def _find_nice_scale(target_mm: float) -> float:
    """
    Find the nearest 'nice' scale value.

    Args:
        target_mm: Target scale in millimeters.

    Returns:
        A nice round number close to the target.
    """
    if target_mm <= 0:
        return 1.0

    # Find the closest nice value
    best = NICE_SCALES_MM[0]
    best_ratio = abs(target_mm / best - 1.0)

    for scale in NICE_SCALES_MM:
        ratio = abs(target_mm / scale - 1.0)
        if ratio < best_ratio:
            best = scale
            best_ratio = ratio

    return best


def _format_scale_label(size_mm: float) -> str:
    """
    Format the scale size with appropriate units.

    Args:
        size_mm: Size in millimeters.

    Returns:
        Formatted string with value and unit.
    """
    if size_mm >= 1.0:
        # Use mm
        if size_mm == int(size_mm):
            return f"{int(size_mm)} mm"
        return f"{size_mm:.1f} mm"
    elif size_mm >= 0.001:
        # Use µm (micrometers)
        um = size_mm * 1000
        if um == int(um):
            return f"{int(um)} µm"
        return f"{um:.1f} µm"
    else:
        # Use nm (nanometers)
        nm = size_mm * 1000000
        if nm == int(nm):
            return f"{int(nm)} nm"
        return f"{nm:.1f} nm"


# ============================================================================
# Scale Bar Actor Class
# ============================================================================


class ScaleBarActor:
    """
    A scale bar overlay for 2D views.

    Displays a horizontal line with a centered label showing the
    physical size and units. Automatically adjusts to zoom level.
    """

    def __init__(
        self,
        position: Tuple[int, int] = (20, 20),
        target_width_pixels: int = 80,
        color: Tuple[float, float, float] = (0.9, 0.9, 0.9),
        font_size: int = 10,
    ) -> None:
        """
        Initialize the scale bar.

        Args:
            position: (x, y) position in pixels from bottom-left corner.
            target_width_pixels: Approximate width of scale bar in pixels.
            color: RGB color tuple (0-1 range).
            font_size: Font size for the label.
        """
        self._position = position
        self._target_width = target_width_pixels
        self._color = color
        self._font_size = font_size
        self._current_scale_mm: float = 1.0
        self._current_width_pixels: int = target_width_pixels

        # Create the line actor (scale bar)
        self._line_source = vtk.vtkLineSource()
        self._line_mapper = vtk.vtkPolyDataMapper2D()
        self._line_mapper.SetInputConnection(self._line_source.GetOutputPort())

        self._line_actor = vtk.vtkActor2D()
        self._line_actor.SetMapper(self._line_mapper)
        self._line_actor.GetProperty().SetColor(*color)
        self._line_actor.GetProperty().SetLineWidth(2.0)

        # Create end caps 
        self._left_cap = vtk.vtkLineSource()
        self._right_cap = vtk.vtkLineSource()

        self._caps_append = vtk.vtkAppendPolyData()
        self._caps_append.AddInputConnection(self._left_cap.GetOutputPort())
        self._caps_append.AddInputConnection(self._right_cap.GetOutputPort())

        self._caps_mapper = vtk.vtkPolyDataMapper2D()
        self._caps_mapper.SetInputConnection(self._caps_append.GetOutputPort())

        self._caps_actor = vtk.vtkActor2D()
        self._caps_actor.SetMapper(self._caps_mapper)
        self._caps_actor.GetProperty().SetColor(*color)
        self._caps_actor.GetProperty().SetLineWidth(2.0)

        # Create the text label
        self._text_actor = vtk.vtkTextActor()
        self._text_actor.GetTextProperty().SetColor(*color)
        self._text_actor.GetTextProperty().SetFontSize(font_size)
        self._text_actor.GetTextProperty().SetFontFamilyToArial()
        self._text_actor.GetTextProperty().SetJustificationToCentered()
        self._text_actor.GetTextProperty().SetVerticalJustificationToBottom()
        self._text_actor.GetTextProperty().BoldOff()
        self._text_actor.GetTextProperty().ShadowOff()

        # Initial update
        self._update_geometry()

    def _update_geometry(self) -> None:
        """Update the scale bar geometry based on current settings."""
        x, y = self._position
        w = self._current_width_pixels
        cap_height = 6  # Height of end caps in pixels

        # Main horizontal line
        self._line_source.SetPoint1(x, y, 0)
        self._line_source.SetPoint2(x + w, y, 0)

        # Left cap (vertical line)
        self._left_cap.SetPoint1(x, y - cap_height // 2, 0)
        self._left_cap.SetPoint2(x, y + cap_height // 2, 0)

        # Right cap (vertical line)
        self._right_cap.SetPoint1(x + w, y - cap_height // 2, 0)
        self._right_cap.SetPoint2(x + w, y + cap_height // 2, 0)

        # Text position (centered above the bar)
        self._text_actor.SetPosition(x + w // 2, y + 4)

        # Update label
        label = _format_scale_label(self._current_scale_mm)
        self._text_actor.SetInput(label)

    def update_scale(
        self,
        camera: vtk.vtkCamera,
        renderer: vtk.vtkRenderer,
    ) -> None:
        """
        Update the scale bar based on current camera zoom.

        Args:
            camera: The VTK camera for this view.
            renderer: The VTK renderer for this view.
        """
        if camera is None or renderer is None:
            return

        try:
            # Get viewport dimensions
            viewport_size = renderer.GetSize()
            if viewport_size[1] == 0:
                return

            viewport_height = viewport_size[1]

            # For orthographic projection, ParallelScale is half the height
            # in world units
            parallel_scale = camera.GetParallelScale()
            if parallel_scale <= 0:
                return

            # World units per pixel
            world_per_pixel = (2.0 * parallel_scale) / viewport_height

            # Target physical size for our desired pixel width
            target_size_mm = world_per_pixel * self._target_width

            # Find nice round number
            nice_scale = _find_nice_scale(target_size_mm)

            # Calculate actual pixel width for this nice scale
            actual_width = nice_scale / world_per_pixel

            # Clamp to reasonable range
            actual_width = max(30, min(150, actual_width))

            # Update if changed significantly
            if (
                abs(nice_scale - self._current_scale_mm) > 1e-9
                or abs(actual_width - self._current_width_pixels) > 1
            ):
                self._current_scale_mm = nice_scale
                self._current_width_pixels = int(actual_width)
                self._update_geometry()

        except Exception as e:
            logger.debug(f"Error updating scale bar: {e}")

    def add_to_renderer(self, renderer: vtk.vtkRenderer) -> None:
        """
        Add the scale bar actors to a renderer.

        Args:
            renderer: The VTK renderer to add actors to.
        """
        renderer.AddActor2D(self._line_actor)
        renderer.AddActor2D(self._caps_actor)
        renderer.AddActor2D(self._text_actor)

    def remove_from_renderer(self, renderer: vtk.vtkRenderer) -> None:
        """
        Remove the scale bar actors from a renderer.

        Args:
            renderer: The VTK renderer to remove actors from.
        """
        renderer.RemoveActor2D(self._line_actor)
        renderer.RemoveActor2D(self._caps_actor)
        renderer.RemoveActor2D(self._text_actor)

    def set_visibility(self, visible: bool) -> None:
        """
        Set visibility of all scale bar components.

        Args:
            visible: True to show, False to hide.
        """
        vis = 1 if visible else 0
        self._line_actor.SetVisibility(vis)
        self._caps_actor.SetVisibility(vis)
        self._text_actor.SetVisibility(vis)

    def set_color(self, color: Tuple[float, float, float]) -> None:
        """
        Set the color of the scale bar and text.

        Args:
            color: RGB tuple (0-1 range).
        """
        self._color = color
        self._line_actor.GetProperty().SetColor(*color)
        self._caps_actor.GetProperty().SetColor(*color)
        self._text_actor.GetTextProperty().SetColor(*color)


# ============================================================================
# Scale Bar Manager Class
# ============================================================================


class ScaleBarManager:
    """Manages scale bars for all 2D orthogonal views."""

    def __init__(self) -> None:
        """Initialize the scale bar manager."""
        self._scale_bars: Dict[str, ScaleBarActor] = {}
        self._renderers: Dict[str, vtk.vtkRenderer] = {}
        self._cameras: Dict[str, vtk.vtkCamera] = {}
        self._initialized = False

    def initialize(
        self,
        axial_renderer: vtk.vtkRenderer,
        coronal_renderer: vtk.vtkRenderer,
        sagittal_renderer: vtk.vtkRenderer,
        axial_camera: vtk.vtkCamera,
        coronal_camera: vtk.vtkCamera,
        sagittal_camera: vtk.vtkCamera,
    ) -> None:
        """
        Initialize scale bars for all three views.

        Args:
            axial_renderer: Renderer for axial view (or its overlay).
            coronal_renderer: Renderer for coronal view (or its overlay).
            sagittal_renderer: Renderer for sagittal view (or its overlay).
            axial_camera: Camera for axial view.
            coronal_camera: Camera for coronal view.
            sagittal_camera: Camera for sagittal view.
        """
        # Store references
        self._renderers = {
            "axial": axial_renderer,
            "coronal": coronal_renderer,
            "sagittal": sagittal_renderer,
        }
        self._cameras = {
            "axial": axial_camera,
            "coronal": coronal_camera,
            "sagittal": sagittal_camera,
        }

        # Create scale bars with consistent styling
        position = (15, 15)  # Bottom-left corner
        color = (0.85, 0.85, 0.85)  # Light gray

        for view in ["axial", "coronal", "sagittal"]:
            scale_bar = ScaleBarActor(
                position=position,
                target_width_pixels=80,
                color=color,
                font_size=11,
            )
            scale_bar.add_to_renderer(self._renderers[view])
            self._scale_bars[view] = scale_bar

        self._initialized = True
        logger.debug("Scale bars initialized for all 2D views")

    def update_all(self) -> None:
        """Update all scale bars based on current camera states."""
        if not self._initialized:
            return

        for view in ["axial", "coronal", "sagittal"]:
            self.update_view(view)

    def update_view(self, view: str) -> None:
        """
        Update scale bar for a specific view.

        Args:
            view: 'axial', 'coronal', or 'sagittal'.
        """
        if not self._initialized:
            return

        scale_bar = self._scale_bars.get(view)
        renderer = self._renderers.get(view)
        camera = self._cameras.get(view)

        if scale_bar and renderer and camera:
            scale_bar.update_scale(camera, renderer)

    def set_visibility(self, visible: bool) -> None:
        """
        Set visibility for all scale bars.

        Args:
            visible: True to show, False to hide.
        """
        for scale_bar in self._scale_bars.values():
            scale_bar.set_visibility(visible)

    def clear(self) -> None:
        """Remove all scale bars and clear references."""
        for view, scale_bar in self._scale_bars.items():
            renderer = self._renderers.get(view)
            if renderer and scale_bar:
                scale_bar.remove_from_renderer(renderer)

        self._scale_bars.clear()
        self._renderers.clear()
        self._cameras.clear()
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if scale bars are initialized."""
        return self._initialized
