# -*- coding: utf-8 -*-

"""
Actor factory functions for TractEdit visualization.

Provides factory functions for creating VTK/FURY actors used in the
visualization of meshes, anatomical slices, ROIs, and UI overlays.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import vtk
from fury import actor
from ..data_contracts import MeshComponent, MeshLayer

logger = logging.getLogger(__name__)


def polyline_display_data(points) -> tuple[vtk.vtkPolyData, vtk.vtkPolyData]:
    """Share RAS points; only display markers omit a repeated closing vertex."""
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vertices = vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(np.ascontiguousarray(values), deep=True))
    dtype = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32

    def cells(count):
        result = vtk.vtkCellArray()
        offsets = np.array([0, count] if count else [0], dtype=dtype)
        result.SetData(
            numpy_to_vtkIdTypeArray(offsets, deep=True),
            numpy_to_vtkIdTypeArray(np.arange(count, dtype=dtype), deep=True),
        )
        return result

    line_data = vtk.vtkPolyData()
    line_data.SetPoints(vertices)
    line_data.SetLines(cells(len(values) if len(values) > 1 else 0))
    count = len(values)
    if count > 1 and np.array_equal(values[0], values[-1]):
        count -= 1
    marker_data = vtk.vtkPolyData()
    marker_data.SetPoints(vertices)
    marker_data.SetVerts(cells(count))
    return line_data, marker_data


def create_mesh_polyline_actor(
    points, color, *, line_width=3.0, point_size=7.0, show_points=False,
    allow_duplicate_adjacent=False,
) -> vtk.vtkActor:
    from ..mesh.polylines import validated_color, validated_points, validated_style

    values = validated_points(points, allow_duplicate_adjacent=allow_duplicate_adjacent)
    color = validated_color(color)
    style = validated_style(line_width, point_size, show_points)
    line_data, marker_data = polyline_display_data(values)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(line_data)
    mapper.ScalarVisibilityOff()
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)
    result = vtk.vtkActor()
    result.SetMapper(mapper)
    result.PickableOff()
    result.GetProperty().SetColor(*color)
    result.GetProperty().SetLineWidth(style["line_width"])
    result.GetProperty().SetPointSize(style["point_size"])
    result.GetProperty().LightingOff()
    marker_mapper = vtk.vtkPolyDataMapper()
    marker_mapper.SetInputData(marker_data)
    marker_mapper.ScalarVisibilityOff()
    marker_mapper.SetResolveCoincidentTopologyToPolygonOffset()
    marker_mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-2)
    markers = vtk.vtkActor()
    markers.SetMapper(marker_mapper)
    markers.PickableOff()
    markers.GetProperty().SetColor(*color)
    markers.GetProperty().SetPointSize(style["point_size"])
    markers.GetProperty().LightingOff()
    markers.SetVisibility(style["show_points"])
    result.polyline_markers = markers
    return result


def create_mesh_actor(
    polydata: vtk.vtkPolyData,
    source_to_ras: np.ndarray,
    color: tuple[float, float, float] = (0.8, 0.8, 0.8),
    opacity: float = 1.0,
    *, mapper=None,
) -> vtk.vtkActor:
    """Present validated surface data in scanner-RAS without copying geometry.

    The caller retains immutable source data. Only RAS/LPS millimetre
    presentation is supported; anatomy and radiological mirrors play no role.
    """
    matrix = np.asarray(source_to_ras, dtype=np.float64)
    if matrix.shape != (4, 4) or not (
        np.array_equal(matrix, np.eye(4))
        or np.array_equal(matrix, np.diag([-1.0, -1.0, 1.0, 1.0]))
    ):
        raise ValueError("Mesh source_to_ras must be the RAS identity or LPS flip")
    rgb = np.asarray(color, dtype=np.float64)
    if rgb.shape != (3,) or not np.all(np.isfinite(rgb) & (rgb >= 0) & (rgb <= 1)):
        raise ValueError("Mesh color must contain three finite values in [0, 1]")
    if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("Mesh opacity must be finite and in [0, 1]")

    if mapper is None:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mapper)
    mesh_actor.PickableOff()
    mesh_actor.GetProperty().SetColor(*rgb)
    mesh_actor.GetProperty().SetOpacity(opacity)
    mesh_actor.GetProperty().BackfaceCullingOff()
    user_matrix = vtk.vtkMatrix4x4()
    user_matrix.DeepCopy(matrix.ravel())
    mesh_actor.SetUserMatrix(user_matrix)
    return mesh_actor


def create_mesh_component_actor(
    layer: MeshLayer, component: MeshComponent,
) -> vtk.vtkActor:
    """Build one unlit overlay from source geometry, never from UV display data."""
    from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
    from ..mesh.presentation import validate_mesh_component_style

    if component not in ("edges", "vertices"):
        raise ValueError("Expected an edges or vertices overlay")
    preference = layer["presentation"][component]
    size_field = "line_width" if component == "edges" else "point_size"
    style = validate_mesh_component_style(component, {
        "color": preference["color"], size_field: preference[size_field],
    })
    source = layer["polydata"]
    if component == "vertices":
        # A single polyvertex includes every loaded point, including unused ones.
        points = vtk.vtkPolyData()
        points.SetPoints(source.GetPoints())
        count = source.GetNumberOfPoints()
        cells = vtk.vtkCellArray()
        cells.SetData(
            numpy_to_vtkIdTypeArray(np.array([0, count], dtype=np.int64), deep=True),
            numpy_to_vtkIdTypeArray(np.arange(count, dtype=np.int64), deep=True),
        )
        points.SetVerts(cells)
        source = points
    overlay = create_mesh_actor(source, layer["source_to_ras"], style["color"], 1.0)
    properties = overlay.GetProperty()
    properties.LightingOff()
    if component == "edges":
        properties.SetRepresentationToWireframe()
        properties.SetLineWidth(style["line_width"])
    else:
        properties.SetPointSize(style["point_size"])
    # Per-actor raster depth bias avoids VTK's process-global polygon offset mode.
    bias = "1.0" if component == "edges" else "2.0"
    overlay.GetShaderProperty().AddFragmentShaderReplacement(
        "//VTK::Depth::Impl", True,
        "//VTK::Depth::Impl\n"
        f"gl_FragDepth = gl_FragCoord.z - {bias} / 65000.0;\n", False,
    )
    from ..mesh.overlays import mesh_interaction_preview

    preview = mesh_interaction_preview(layer["polydata"], component)
    overlay._mesh_full_mapper = overlay.GetMapper()
    if preview is not None:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(preview)
        mapper.ScalarVisibilityOff()
        overlay._mesh_interaction_mapper = mapper
    overlay._mesh_preview_base_mapper = getattr(
        overlay, "_mesh_interaction_mapper", overlay.GetMapper(),
    )
    overlay._mesh_preview_levels = {}
    return overlay


def create_mesh_advanced_actor(layer: MeshLayer, component: MeshComponent, data=None):
    """Create a single non-pickable overlay; glyph size is in world millimetres."""
    from ..mesh.overlays import mesh_ras_bounds, sampled_normal_polydata

    preference = layer["presentation"][component]
    if component == "bounding_box":
        outline = vtk.vtkOutlineSource()
        outline.SetBounds(mesh_ras_bounds(layer))
        outline.Update()
        overlay = create_mesh_actor(
            outline.GetOutput(), np.eye(4), preference["color"], 1,
        )
    elif component == "boundary_edges":
        overlay = create_mesh_actor(
            data.polydata, layer["source_to_ras"], preference["color"], 1,
        )
        from ..mesh.overlays import mesh_interaction_preview

        preview = mesh_interaction_preview(data.polydata, component)
        overlay._mesh_full_mapper = overlay.GetMapper()
        if preview is not None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(preview)
            mapper.ScalarVisibilityOff()
            overlay._mesh_interaction_mapper = mapper
        overlay._mesh_preview_base_mapper = getattr(
            overlay, "_mesh_interaction_mapper", overlay.GetMapper(),
        )
        overlay._mesh_preview_levels = {}
    elif component == "surface_normals":
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(6)
        arrow.SetShaftResolution(6)
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetSourceConnection(arrow.GetOutputPort())
        mapper.SetInputData(sampled_normal_polydata(data, preference["density"]))
        mapper.SetOrientationArray("face_normals")
        mapper.SetOrientationModeToDirection()
        mapper.OrientOn()
        mapper.ScalingOn()
        mapper.SetScaleModeToNoDataScaling()
        mapper.SetScaleFactor(preference["length_mm"])
        mapper.ScalarVisibilityOff()
        overlay = vtk.vtkActor()
        overlay.SetMapper(mapper)
        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy(layer["source_to_ras"].ravel())
        overlay.SetUserMatrix(matrix)
        overlay.GetProperty().SetColor(*preference["color"])
        overlay.PickableOff()
    else:
        raise ValueError("Unknown Advanced mesh component")
    overlay.UseBoundsOff()
    overlay.GetProperty().LightingOff()
    if "line_width" in preference:
        overlay.GetProperty().SetLineWidth(preference["line_width"])
    if component == "boundary_edges":
        overlay.GetShaderProperty().AddFragmentShaderReplacement(
            "//VTK::Depth::Impl", True,
            "//VTK::Depth::Impl\ngl_FragDepth = gl_FragCoord.z - 1.5 / 65000.0;\n",
            False,
        )
    return overlay


# =============================================================================
# Slice Actors
# =============================================================================


def apply_slicer_affine(
    slicer_actor: vtk.vtkActor,
    affine: np.ndarray,
    radiological: bool = False,
) -> None:
    """Apply a NIfTI voxel-to-RAS affine without resampling the slicer input."""
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("Slicer affine must be a finite 4x4 matrix.")

    if radiological:
        mirror_x = np.diag([-1.0, 1.0, 1.0, 1.0])
        affine = mirror_x @ affine

    vtk_matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, float(affine[row, column]))

    # Keep all spatial state in one matrix so copies compose identically.
    slicer_actor.SetOrigin(0.0, 0.0, 0.0)
    slicer_actor.SetPosition(0.0, 0.0, 0.0)
    slicer_actor.SetOrientation(0.0, 0.0, 0.0)
    slicer_actor.SetScale(1.0, 1.0, 1.0)
    slicer_actor.SetUserMatrix(vtk_matrix)


def create_slicer_actor(
    data: np.ndarray,
    affine: np.ndarray,
    value_range: Tuple[float, float],
    opacity: float = 1.0,
    interpolation: str = "nearest",
    radiological: bool = False,
) -> Optional[vtk.vtkActor]:
    """
    Creates a FURY slicer actor for anatomical or ROI data.

    Args:
        data: 3D numpy array of image data.
        affine: 4x4 NIfTI voxel-to-RAS affine transformation matrix.
        value_range: (min, max) value range for colormap.
        opacity: Actor opacity.
        interpolation: Interpolation mode ("nearest" or "linear").
        radiological: Mirror scanner-RAS X after applying the affine.

    Returns:
        VTK slicer actor, or None if creation fails.
    """
    try:
        if data.ndim == 4 and data.shape[-1] == 3:
            from ..image_volume import rgb_display_values

            data = rgb_display_values(data)
            value_range = (0, 255)
        slicer_actor = actor.slicer(
            data,
            affine=np.eye(4),
            value_range=value_range,
            opacity=opacity,
            interpolation=interpolation,
        )
        apply_slicer_affine(slicer_actor, affine, radiological=radiological)
        return slicer_actor
    except (RuntimeError, ValueError, AttributeError) as e:
        logger.error(f"Error creating slicer actor: {e}")
        return None


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
    cube_actor: vtk.vtkProp3D,
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
