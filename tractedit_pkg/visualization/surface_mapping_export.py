# -*- coding: utf-8 -*-
"""Figure capture using the GUI's existing OpenGL context and detached pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import vtkCamera, vtkWindowToImageFilter
from vtkmodules.vtkRenderingOpenGL2 import vtkTextureObject
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor

FIGURE_MEMORY_BUDGET = 512 * 1024**2
FIGURE_MAX_DIMENSION = 8192


@dataclass(frozen=True)
class FigureOptions:
    width: int = 2048
    height: int = 2048
    scene: str = "mapping_only"
    transparent: bool = False
    require_full_markers: bool = False

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1 for value in (self.width, self.height)
        ):
            raise ValueError("Figure dimensions must be positive integer pixels")
        if self.scene not in ("mapping_only", "current_scene"):
            raise ValueError("Choose Mapping only or Current 3D scene")
        if self.transparent:
            raise ValueError(
                "RGBA capture has not been verified on this backend; use an opaque background"
            )


def validate_figure_size(options, backend_maximum):
    estimate = options.width * options.height * 48
    maximum = min(FIGURE_MAX_DIMENSION, backend_maximum)
    if maximum < 1:
        raise ValueError("Could not verify the OpenGL capture size limit")
    if max(options.width, options.height) > maximum or estimate > FIGURE_MEMORY_BUDGET:
        raise ValueError(
            f"Figure exceeds capture limits ({maximum} pixels per axis, {FIGURE_MEMORY_BUDGET // 1024**2} MiB estimated working memory). Requested estimate: {estimate // 1024**2} MiB."
        )
    return estimate


def presentation_metadata(presentation):
    record = {
        name: bool(getattr(presentation, name))
        for name in (
            "visible",
            "markers_enabled",
            "qc_enabled",
            "map_enabled",
            "density_enabled",
            "endpoint_enabled",
            "intersections_enabled",
        )
    }
    record.update(marker_radius_mm=0.12, overlap_radius_mm=0.06, position_offset_mm=0.0)
    from .surface_mapping import ENDPOINT_COLOR, CROSSING_COLOR

    record["marker_colors"] = {
        "endpoint_proximity": list(ENDPOINT_COLOR),
        "surface_intersections": list(CROSSING_COLOR),
        "diagnostics": [0.85, 0.2, 0.8],
    }
    record["marker_shapes"] = {
        "endpoint_proximity": "sphere",
        "surface_intersections": "wireframe cube",
    }
    record["crossing_half_width_mm"] = 0.12
    if presentation.map_actor is not None:
        lut = presentation.lut
        record.update(
            colormap={
                "name": "VTK hue ramp",
                "rgba": [
                    list(lut.GetTableValue(i))
                    for i in range(lut.GetNumberOfTableValues())
                ],
            },
            scalar_range=list(presentation.map_actor.GetMapper().GetScalarRange()),
            scale="log10" if lut.GetScale() else "linear",
            missing_color=list(lut.GetNanColor()),
            scalar_units=(
                "endpoint contributions/mm^2"
                if presentation.density_enabled
                else "endpoint contributions"
            ),
        )
    return record


def capture_figure(panel, presentation, options: FigureOptions):
    app = QApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        raise RuntimeError(
            "Capture must run on the GUI thread owning the OpenGL context"
        )
    bundles = [presentation.bundle]
    if presentation.bundle.companion is not None:
        bundles.append(presentation.bundle.companion)
    if options.require_full_markers and any(
        bundle.marker_count != len(bundle.marker_points)
        or bundle.overlap_segment_count != len(bundle.overlap_points)
        for bundle in bundles
    ):
        raise ValueError(
            "Full marker detail exceeds the verified display budget; use the reported sampled figure or export complete scientific data"
        )
    window, scene = panel.render_window, panel.scene
    window.MakeCurrent()
    backend_maximum = vtkTextureObject.GetMaximumTextureSize(window)
    estimate = validate_figure_size(options, backend_maximum)
    original_size, viewport = window.GetSize(), scene.GetViewport()
    if min(original_size) <= 0 or viewport[2] <= viewport[0] or viewport[3] <= viewport[1]:
        raise ValueError("A visible 3D viewport is required to capture a figure")
    offscreen_buffers = window.GetUseOffScreenBuffers()
    aspect = (
        original_size[0]
        * (viewport[2] - viewport[0])
        / (original_size[1] * (viewport[3] - viewport[1]))
    )
    width = min(options.width, max(1, round(options.height * aspect)))
    height = min(options.height, max(1, round(options.width / aspect)))
    camera = scene.GetActiveCamera()
    saved_camera = vtkCamera()
    saved_camera.DeepCopy(camera)
    renderers = list(window.GetRenderers())
    props = list(scene.GetViewProps())
    visibility = [(prop, prop.GetVisibility()) for prop in props]
    mappers = [
        (actor, actor.GetMapper())
        for components in panel._mesh_component_actors.values()
        for actor in components.values()
        if hasattr(actor, "GetMapper")
    ]
    background, background2 = scene.GetBackground(), scene.GetBackground2()
    gradient, alpha = scene.GetGradientBackground(), scene.GetBackgroundAlpha()
    interacting = panel._is_interacting
    update_rate = window.GetDesiredUpdateRate()
    timer = getattr(panel, "_mesh_interaction_timer", None)
    timer_remaining = (
        timer.remainingTime() if timer is not None and timer.isActive() else -1
    )
    legend = presentation.scalar_bar
    capture_legend = None
    legend_colors = (
        None
        if legend is None
        else (
            legend.GetTitleTextProperty().GetColor(),
            legend.GetLabelTextProperty().GetColor(),
        )
    )
    metadata = {
        "format": "tractedit-surface-mapping-figure",
        "schema": 1,
        "width": options.width,
        "height": options.height,
        "channels": "RGB",
        "background": [0.0, 0.0, 0.0],
        "alpha": False,
        "scene": options.scene,
        "padding": {
            "policy": "aspect-preserving centred padding",
            "content_width": width,
            "content_height": height,
        },
        "camera": {
            name: getattr(camera, "Get" + name)()
            for name in (
                "Position",
                "FocalPoint",
                "ViewUp",
                "ClippingRange",
                "ParallelProjection",
                "ParallelScale",
                "ViewAngle",
                "WindowCenter",
                "ViewShear",
            )
        },
        "presentation": presentation_metadata(presentation),
        "marker_count": sum(len(bundle.marker_points) for bundle in bundles),
        "marker_total": sum(bundle.marker_count for bundle in bundles),
        "overlap_segments": sum(len(bundle.overlap_points) for bundle in bundles),
        "overlap_segment_total": sum(
            bundle.overlap_segment_count for bundle in bundles
        ),
        "methods": [bundle.result.request.parameters.mode for bundle in bundles],
        "context_streamline_stride": getattr(panel.main_window, "render_stride", 1),
        "mesh_overlay_detail": "full",
        "scalar_map_detail": "complete",
        "backend": {
            "class": window.GetClassName(),
            "maximum_texture_dimension": backend_maximum,
        },
        "estimated_capture_bytes": estimate,
        "full_marker_detail_required": options.require_full_markers,
    }
    try:
        presentation._capture_override = True
        if timer is not None:
            timer.stop()
        panel._is_interacting = False
        panel._set_mesh_interaction_preview(False)
        window.SetDesiredUpdateRate(0.001)
        for renderer in renderers:
            if renderer is not scene:
                window.RemoveRenderer(renderer)
        scene.SetViewport(0, 0, 1, 1)
        scene.SetBackground(0, 0, 0)
        scene.SetGradientBackground(False)
        scene.SetBackgroundAlpha(1.0)
        if options.scene == "mapping_only":
            for prop in props:
                prop.SetVisibility(False)
            show_map = (
                presentation.map_enabled and presentation.endpoint_enabled
                and presentation.map_actor is not None
            )
            if show_map:
                presentation.map_actor.SetVisibility(True)
                legend.SetVisibility(True)
            else:
                surface = panel.mesh_actors.get(presentation.mesh_id)
                texture = panel._mesh_texture_presentations.get(presentation.mesh_id)
                if surface is not None:
                    surface.SetVisibility(True)
                if texture is not None and texture.enabled:
                    for part in texture.parts:
                        part.actor.SetVisibility(True)
            for actor, enabled in presentation.marker_visibility():
                if actor is not None:
                    actor.SetVisibility(enabled)
        else:
            for name in ("status_text_actor", "instruction_text_actor", "radius_actor"):
                prop = getattr(panel, name, None)
                if prop is not None:
                    prop.SetVisibility(False)
        if legend is not None:
            legend.GetTitleTextProperty().SetColor(1, 1, 1)
            legend.GetLabelTextProperty().SetColor(1, 1, 1)
            if legend.GetVisibility():
                capture_legend = vtkScalarBarActor()
                capture_legend.SetLookupTable(presentation.lut)
                capture_legend.SetTitle(
                    "Endpoint\ncontributions\nper mm²"
                    if presentation.density_enabled
                    else "Endpoint\ncontributions"
                )
                capture_legend.SetPosition(0.06, 0.2)
                capture_legend.SetWidth(0.07)
                capture_legend.SetHeight(0.5)
                capture_legend.SetNumberOfLabels(4)
                capture_legend.SetLabelFormat("%.3g")
                capture_legend.SetUnconstrainedFontSize(True)
                capture_legend.GetTitleTextProperty().SetFontSize(
                    max(12, min(32, height // 35))
                )
                capture_legend.GetLabelTextProperty().SetFontSize(
                    max(11, min(28, height // 42))
                )
                legend.SetVisibility(False)
                scene.AddActor2D(capture_legend)
        metadata["legend"] = {
            "visible": capture_legend is not None,
            "title": capture_legend.GetTitle() if capture_legend is not None else None,
        }
        metadata["included_props"] = [
            {
                "class": prop.GetClassName(),
                "label": prop.GetObjectName() if hasattr(prop, "GetObjectName") else "",
            }
            for prop in scene.GetViewProps()
            if prop.GetVisibility()
        ]
        metadata["elements"] = {
            "target_mesh_id": presentation.mesh_id,
            "mesh_ids": [
                identity
                for identity, actor in panel.mesh_actors.items()
                if actor.GetVisibility()
            ],
            "streamlines_visible": bool(
                panel.streamlines_actor and panel.streamlines_actor.GetVisibility()
            ),
            "anatomical_slices": [
                name
                for name in (
                    "axial_slice_actor",
                    "coronal_slice_actor",
                    "sagittal_slice_actor",
                )
                if getattr(panel, name, None) is not None
                and getattr(panel, name).GetVisibility()
            ],
            "roi_layers": [
                name
                for name, actors in panel.roi_slice_actors.items()
                if any(
                    actor.GetVisibility()
                    for actor in actors.values()
                    if hasattr(actor, "GetVisibility")
                )
            ],
            "endpoint_map": bool(
                presentation.map_actor and presentation.map_actor.GetVisibility()
            ),
            "associated_markers": bool(
                presentation.marker_actor and presentation.marker_actor.GetVisibility()
                or presentation.crossing_actor
                and presentation.crossing_actor.GetVisibility()
            ),
            "crossing_markers": bool(
                presentation.crossing_actor
                and presentation.crossing_actor.GetVisibility()
                or (
                    presentation.bundle.result.request.parameters.mode
                    == "surface_intersections"
                    and presentation.marker_actor
                    and presentation.marker_actor.GetVisibility()
                )
            ),
            "qc_markers": bool(
                presentation.qc_actor and presentation.qc_actor.GetVisibility()
                or presentation.intersection_qc_actor
                and presentation.intersection_qc_actor.GetVisibility()
            ),
            "overlap_intervals": bool(
                presentation.overlap_actor
                and presentation.overlap_actor.GetVisibility()
            ),
        }
        metadata["marker_detail"] = (
            "complete"
            if all(
                bundle.marker_count == len(bundle.marker_points) for bundle in bundles
            )
            else "deterministically sampled"
        )
        # Keep capture size independent of the embedded native view.
        window.SetUseOffScreenBuffers(True)
        window.SetSize(width, height)
        window.Render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.ShouldRerenderOff()
        capture.Update()
        raw = capture.GetOutput()
        if raw.GetDimensions()[:2] != (width, height):
            raise RuntimeError(
                "Backend did not provide the requested capture dimensions: "
                f"requested {(width, height)}, got {raw.GetDimensions()[:2]}"
            )
        pixels = np.zeros((options.height, options.width, 3), dtype=np.uint8)
        x, y = (options.width - width) // 2, (options.height - height) // 2
        pixels[y : y + height, x : x + width] = vtk_to_numpy(
            raw.GetPointData().GetScalars()
        ).reshape(height, width, 3)
        image = vtkImageData()
        image.SetDimensions(options.width, options.height, 1)
        image.GetPointData().SetScalars(numpy_to_vtk(pixels.reshape(-1, 3), deep=True))
        return image, metadata
    finally:
        if capture_legend is not None:
            scene.RemoveActor2D(capture_legend)
        for prop, visible in visibility:
            prop.SetVisibility(visible)
        for actor, mapper in mappers:
            actor.SetMapper(mapper)
        scene.SetBackground(background)
        scene.SetBackground2(background2)
        scene.SetGradientBackground(gradient)
        scene.SetBackgroundAlpha(alpha)
        scene.SetViewport(viewport)
        camera.DeepCopy(saved_camera)
        window.SetSize(original_size)
        window.SetUseOffScreenBuffers(offscreen_buffers)
        window.RemoveRenderer(scene)
        for renderer in renderers:
            window.AddRenderer(renderer)
        window.SetDesiredUpdateRate(update_rate)
        panel._is_interacting = interacting
        if legend_colors is not None:
            legend.GetTitleTextProperty().SetColor(legend_colors[0])
            legend.GetLabelTextProperty().SetColor(legend_colors[1])
        try:
            window.Render()
        finally:
            presentation._capture_override = False
            if timer_remaining >= 0:
                timer.start(max(1, timer_remaining))
