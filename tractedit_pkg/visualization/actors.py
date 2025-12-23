# -*- coding: utf-8 -*-

"""
Actor factory functions for TractEdit visualization.

Provides factory functions for creating VTK/FURY actors used in the
visualization of streamlines, anatomical slices, ROIs, ODFs, and UI overlays.
"""

import logging
from typing import Optional, List, Tuple, Any, Dict

import numpy as np
import vtk
from fury import actor, window

logger = logging.getLogger(__name__)


# =============================================================================
# Streamline Actors
# =============================================================================


def create_streamline_actor(
    streamlines_list: List[np.ndarray],
    colors: Any,
    opacity: float = 1.0,
    linewidth: float = 2.0,
    use_tubes: bool = False,
    tube_radius: float = 0.15,
    use_lod: bool = False,
) -> Optional[vtk.vtkActor]:
    """
    Creates a streamline actor using FURY's line or streamtube.

    Args:
        streamlines_list: List of streamline arrays (Nx3 each).
        colors: Color specification - tuple (R,G,B), or array of colors.
        opacity: Actor opacity (0.0 to 1.0).
        linewidth: Line width in pixels (for lines) or tube radius (for tubes).
        use_tubes: If True, render as tubes; otherwise as lines.
        tube_radius: Radius for tube rendering.
        use_lod: If True, use Level of Detail for smoother interaction.

    Returns:
        VTK actor for the streamlines, or None if creation fails.
    """
    if not streamlines_list:
        return None

    try:
        if use_tubes:
            return actor.streamtube(
                streamlines_list,
                colors=colors,
                opacity=opacity,
                linewidth=tube_radius,
                lod=use_lod,
            )
        else:
            return actor.line(
                streamlines_list,
                colors=colors,
                opacity=opacity,
                linewidth=linewidth,
                lod=use_lod,
            )
    except Exception as e:
        logger.error(f"Error creating streamline actor: {e}")
        return None


def create_highlight_actor(
    streamlines_list: List[np.ndarray],
    color: Tuple[float, float, float] = (1.0, 1.0, 0.0),
    linewidth: float = 3.0,
    opacity: float = 0.8,
) -> Optional[vtk.vtkActor]:
    """
    Creates a highlight actor for selected streamlines.

    Args:
        streamlines_list: List of streamline arrays to highlight.
        color: RGB color tuple (default yellow).
        linewidth: Line width in pixels.
        opacity: Actor opacity.

    Returns:
        VTK actor for highlighted streamlines, or None if creation fails.
    """
    if not streamlines_list:
        return None

    try:
        highlight_actor = actor.line(
            streamlines_list,
            colors=color,
            linewidth=linewidth,
            opacity=opacity,
            depth_cue=False,
        )

        # Use mapper offset to ensure it draws on top
        mapper = highlight_actor.GetMapper()
        if mapper:
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -10)

        return highlight_actor
    except Exception as e:
        logger.error(f"Error creating highlight actor: {e}")
        return None


def create_roi_highlight_actor(
    streamlines_list: List[np.ndarray],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    linewidth: float = 4.0,
    opacity: float = 1.0,
) -> Optional[vtk.vtkActor]:
    """
    Creates a red highlight actor for ROI-selected streamlines.

    Args:
        streamlines_list: List of streamline arrays to highlight.
        color: RGB color tuple (default red).
        linewidth: Line width in pixels.
        opacity: Actor opacity.

    Returns:
        VTK actor for ROI-highlighted streamlines, or None if creation fails.
    """
    if not streamlines_list:
        return None

    try:
        roi_actor = actor.line(
            streamlines_list,
            colors=color,
            linewidth=linewidth,
            opacity=opacity,
            depth_cue=False,
        )

        # Stronger offset to draw on top of yellow highlights
        mapper = roi_actor.GetMapper()
        if mapper:
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -20)

        return roi_actor
    except Exception as e:
        logger.error(f"Error creating ROI highlight actor: {e}")
        return None


# =============================================================================
# Slice Actors
# =============================================================================


def create_slicer_actor(
    data: np.ndarray,
    affine: np.ndarray,
    value_range: Tuple[float, float],
    opacity: float = 1.0,
    interpolation: str = "nearest",
) -> Optional[vtk.vtkActor]:
    """
    Creates a FURY slicer actor for anatomical or ROI data.

    Args:
        data: 3D numpy array of image data.
        affine: 4x4 affine transformation matrix.
        value_range: (min, max) value range for colormap.
        opacity: Actor opacity.
        interpolation: Interpolation mode ("nearest" or "linear").

    Returns:
        VTK slicer actor, or None if creation fails.
    """
    try:
        return actor.slicer(
            data,
            affine=affine,
            value_range=value_range,
            opacity=opacity,
            interpolation=interpolation,
        )
    except Exception as e:
        logger.error(f"Error creating slicer actor: {e}")
        return None


# =============================================================================
# ODF Actors
# =============================================================================


def create_odf_actor(
    odf_amplitudes: np.ndarray,
    sphere: Any,
    affine: np.ndarray,
    scale: float = 0.5,
    extent: Optional[Tuple[int, ...]] = None,
) -> Optional[vtk.vtkActor]:
    """
    Creates an ODF glyph actor using FURY's odf_slicer.

    Args:
        odf_amplitudes: 4D array (x, y, z, N_vertices) of SF amplitudes.
        sphere: The SimpleSphere object used for reconstruction.
        affine: The affine of the ODF volume.
        scale: Scale factor for ODF glyphs.
        extent: Optional display extent (min_x, max_x, min_y, max_y, min_z, max_z).

    Returns:
        VTK actor for ODF visualization, or None if creation fails.
    """
    if odf_amplitudes is None:
        return None

    try:
        odf_actor = actor.odf_slicer(
            odf_amplitudes,
            sphere=sphere,
            affine=affine,
            scale=scale,
            norm=True,
            global_cm=False,
        )

        if extent is not None:
            odf_actor.display_extent(*extent)

        return odf_actor
    except Exception as e:
        logger.error(f"Error creating ODF actor: {e}")
        return None


# =============================================================================
# Selection Sphere Actor
# =============================================================================


def create_selection_sphere(
    radius: float,
    center: Tuple[float, float, float],
    color: Tuple[float, float, float] = (0.2, 0.5, 1.0),
    opacity: float = 0.3,
) -> vtk.vtkActor:
    """
    Creates a wireframe sphere actor for selection visualization.

    Args:
        radius: Sphere radius in world units.
        center: Center point (x, y, z).
        color: RGB color tuple.
        opacity: Sphere opacity.

    Returns:
        VTK sphere actor.
    """
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(radius)
    sphere_source.SetCenter(0, 0, 0)
    sphere_source.SetPhiResolution(16)
    sphere_source.SetThetaResolution(16)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere_source.GetOutputPort())

    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(mapper)
    sphere_actor.SetPosition(center)

    prop = sphere_actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetRepresentationToWireframe()
    prop.SetLineWidth(1.0)

    sphere_actor.SetVisibility(0)

    return sphere_actor


# =============================================================================
# UI Text Actors
# =============================================================================


def create_status_text_actor(
    initial_text: str = "Status: Initializing...",
    font_size: int = 14,
    color: Tuple[float, float, float] = (0.95, 0.95, 0.95),
    position: Tuple[float, float] = (0.01, 0.01),
) -> vtk.vtkTextActor:
    """
    Creates a status text actor for the VTK scene.

    Args:
        initial_text: Initial status message.
        font_size: Font size in points.
        color: RGB text color.
        position: Normalized display coordinates (0-1, 0-1).

    Returns:
        VTK text actor for status display.
    """
    text_prop = vtk.vtkTextProperty()
    text_prop.SetFontSize(font_size)
    text_prop.SetColor(*color)
    text_prop.SetFontFamilyToArial()
    text_prop.SetJustificationToLeft()
    text_prop.SetVerticalJustificationToBottom()

    text_actor = vtk.vtkTextActor()
    text_actor.SetTextProperty(text_prop)
    text_actor.SetInput(initial_text)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    text_actor.GetPositionCoordinate().SetValue(*position)

    return text_actor


def create_instruction_text_actor(
    font_size: int = 12,
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    position: Tuple[float, float] = (0.01, 0.05),
) -> vtk.vtkTextActor:
    """
    Creates an instruction text actor showing keyboard shortcuts.

    Args:
        font_size: Font size in points.
        color: RGB text color.
        position: Normalized display coordinates.

    Returns:
        VTK text actor for instructions.
    """
    text_prop = vtk.vtkTextProperty()
    text_prop.SetFontSize(font_size)
    text_prop.SetColor(*color)
    text_prop.SetFontFamilyToArial()
    text_prop.SetJustificationToLeft()
    text_prop.SetVerticalJustificationToBottom()

    text_actor = vtk.vtkTextActor()
    text_actor.SetTextProperty(text_prop)

    instruction_text = (
        "Selection: [S] Select | [D] Del | [C] Clear | [+/-] Radius | [Esc] Hide Sphere\n"
        "File/Edit: [Ctrl+S] Save | [Ctrl+Z] Undo | [Ctrl+Y] Redo"
    )
    text_actor.SetInput(instruction_text)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    text_actor.GetPositionCoordinate().SetValue(*position)

    return text_actor


def create_2d_label_actor(
    text: str,
    position: Tuple[float, float],
    font_size: int = 14,
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    h_align: str = "Center",
    v_align: str = "Center",
) -> vtk.vtkTextActor:
    """
    Creates a 2D text label actor for orientation markers.

    Args:
        text: Label text (e.g., "A", "P", "L", "R", "S", "I").
        position: Normalized display coordinates (0-1, 0-1).
        font_size: Font size in points.
        color: RGB text color.
        h_align: Horizontal alignment ("Left", "Center", "Right").
        v_align: Vertical alignment ("Top", "Center", "Bottom").

    Returns:
        VTK text actor for the label.
    """
    text_prop = vtk.vtkTextProperty()
    text_prop.SetFontSize(font_size)
    text_prop.SetColor(*color)
    text_prop.SetFontFamilyToArial()
    text_prop.BoldOff()
    text_prop.ShadowOff()

    # Set Horizontal Alignment
    if h_align == "Left":
        text_prop.SetJustificationToLeft()
    elif h_align == "Center":
        text_prop.SetJustificationToCentered()
    elif h_align == "Right":
        text_prop.SetJustificationToRight()

    # Set Vertical Alignment
    if v_align == "Top":
        text_prop.SetVerticalJustificationToTop()
    elif v_align == "Center":
        text_prop.SetVerticalJustificationToCentered()
    elif v_align == "Bottom":
        text_prop.SetVerticalJustificationToBottom()

    text_actor = vtk.vtkTextActor()
    text_actor.SetTextProperty(text_prop)
    text_actor.SetInput(text)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    text_actor.GetPositionCoordinate().SetValue(*position)

    return text_actor


# =============================================================================
# Axes and Orientation Actors
# =============================================================================


def create_axes_actor(scale: Tuple[float, float, float] = (25, 25, 25)) -> vtk.vtkActor:
    """
    Creates an axes actor using FURY.

    Args:
        scale: Scale for each axis (x, y, z).

    Returns:
        VTK actor for axes visualization.
    """
    return actor.axes(scale=scale)


def create_orientation_cube() -> vtk.vtkAnnotatedCubeActor:
    """
    Creates an annotated cube actor for 3D orientation display.

    Returns:
        VTK annotated cube actor with R/L/A/P/S/I labels.
    """
    cube_actor = vtk.vtkAnnotatedCubeActor()
    cube_actor.SetXPlusFaceText("R")  # Right
    cube_actor.SetXMinusFaceText("L")  # Left
    cube_actor.SetYPlusFaceText("A")  # Anterior
    cube_actor.SetYMinusFaceText("P")  # Posterior
    cube_actor.SetZPlusFaceText("S")  # Superior
    cube_actor.SetZMinusFaceText("I")  # Inferior

    # Configure text properties
    prop = cube_actor.GetTextEdgesProperty()
    prop.SetColor(0.9, 0.9, 0.9)
    prop.SetLineWidth(1.0)

    # Configure cube face properties
    cube_actor.GetCubeProperty().SetColor(0.3, 0.3, 0.3)

    # Configure face text properties
    face_props = [
        cube_actor.GetXPlusFaceProperty(),
        cube_actor.GetXMinusFaceProperty(),
        cube_actor.GetYPlusFaceProperty(),
        cube_actor.GetYMinusFaceProperty(),
        cube_actor.GetZPlusFaceProperty(),
        cube_actor.GetZMinusFaceProperty(),
    ]
    for face_prop in face_props:
        face_prop.SetColor(0.9, 0.9, 0.9)
        face_prop.SetInterpolationToFlat()

    return cube_actor


def create_orientation_widget(
    cube_actor: vtk.vtkAnnotatedCubeActor,
    interactor: vtk.vtkRenderWindowInteractor,
    viewport: Tuple[float, float, float, float] = (0.85, 0.0, 1.0, 0.15),
) -> vtk.vtkOrientationMarkerWidget:
    """
    Creates an orientation marker widget with the given cube actor.

    Args:
        cube_actor: Annotated cube actor to use as marker.
        interactor: VTK render window interactor.
        viewport: Viewport coordinates (xmin, ymin, xmax, ymax).

    Returns:
        VTK orientation marker widget.
    """
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(cube_actor)
    widget.SetInteractor(interactor)
    widget.SetViewport(*viewport)
    widget.SetEnabled(1)
    widget.InteractiveOff()

    return widget


# =============================================================================
# Scalar Bar Actor
# =============================================================================


def create_scalar_bar(
    lut: vtk.vtkLookupTable,
    title: str,
    position: Tuple[float, float] = (0.88, 0.25),
    width: float = 0.1,
    height: float = 0.5,
) -> vtk.vtkScalarBarActor:
    """
    Creates a scalar bar actor for colormap display.

    Args:
        lut: VTK lookup table for colors.
        title: Title for the scalar bar.
        position: Position in normalized display coordinates.
        width: Width as fraction of display.
        height: Height as fraction of display.

    Returns:
        VTK scalar bar actor.
    """
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)
    scalar_bar.SetTitle(title + "\n")

    # Positioning
    scalar_bar.SetPosition(*position)
    scalar_bar.SetWidth(width)
    scalar_bar.SetHeight(height)

    # Title Styling
    title_prop = vtk.vtkTextProperty()
    title_prop.SetColor(0.9, 0.9, 0.9)
    title_prop.SetFontSize(10)
    title_prop.SetFontFamilyToArial()
    title_prop.BoldOn()
    title_prop.ShadowOn()
    title_prop.SetJustificationToRight()

    # Label Styling
    label_prop = vtk.vtkTextProperty()
    label_prop.SetColor(0.8, 0.8, 0.8)
    label_prop.SetFontSize(10)
    label_prop.SetFontFamilyToArial()
    label_prop.ShadowOff()

    scalar_bar.SetTitleTextProperty(title_prop)
    scalar_bar.SetLabelTextProperty(label_prop)

    # Formatting
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetLabelFormat("%-.4g")
    scalar_bar.UnconstrainedFontSizeOn()
    scalar_bar.SetTextPositionToPrecedeScalarBar()

    return scalar_bar


# =============================================================================
# Crosshair Actors
# =============================================================================


def create_crosshair_components() -> Dict[str, Any]:
    """
    Creates the VTK components for a crosshair (two perpendicular lines).

    Returns:
        Dictionary containing:
        - 'line1': vtkLineSource for first line
        - 'line2': vtkLineSource for second line
        - 'appender': vtkAppendPolyData combining both lines
        - 'mapper': vtkPolyDataMapper for the combined geometry
    """
    line1 = vtk.vtkLineSource()
    line2 = vtk.vtkLineSource()

    appender = vtk.vtkAppendPolyData()
    appender.AddInputConnection(line1.GetOutputPort())
    appender.AddInputConnection(line2.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(appender.GetOutputPort())
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)

    return {
        "line1": line1,
        "line2": line2,
        "appender": appender,
        "mapper": mapper,
    }


def create_crosshair_actor(
    mapper: vtk.vtkPolyDataMapper,
    color: Tuple[float, float, float] = (1.0, 1.0, 0.0),
    line_width: float = 1.0,
    opacity: float = 0.8,
) -> vtk.vtkActor:
    """
    Creates a crosshair actor from a mapper.

    Args:
        mapper: VTK mapper for the crosshair geometry.
        color: RGB color tuple (default yellow).
        line_width: Line width in pixels.
        opacity: Actor opacity.

    Returns:
        VTK actor for the crosshair.
    """
    crosshair_actor = vtk.vtkActor()
    crosshair_actor.SetMapper(mapper)
    crosshair_actor.GetProperty().SetColor(*color)
    crosshair_actor.GetProperty().SetLineWidth(line_width)
    crosshair_actor.GetProperty().SetOpacity(opacity)

    return crosshair_actor
