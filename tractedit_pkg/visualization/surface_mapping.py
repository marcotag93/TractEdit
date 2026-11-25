# -*- coding: utf-8 -*-
"""Independent mapping actors; source material and scientific arrays stay intact."""

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkGlyph3DMapper, vtkPolyDataMapper
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor


MARKER_RADIUS_MM = 0.12
OVERLAP_RADIUS_MM = 0.06
ENDPOINT_COLOR = (1.0, 0.55, 0.08)
CROSSING_COLOR = (0.1, 0.7, 1.0)


def _points_data(points):
    data, vertices = vtkPolyData(), vtkPoints()
    vertices.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    data.SetPoints(vertices)
    return data


def _markers(points, color, *, crossing=False):
    if not len(points):
        return None
    if crossing:
        source = vtkCubeSource()
        source.SetXLength(2 * MARKER_RADIUS_MM)
        source.SetYLength(2 * MARKER_RADIUS_MM)
        source.SetZLength(2 * MARKER_RADIUS_MM)
    else:
        source = vtkSphereSource()
        source.SetRadius(MARKER_RADIUS_MM)
        source.SetThetaResolution(8)
        source.SetPhiResolution(6)
    mapper = vtkGlyph3DMapper()
    mapper.SetInputData(_points_data(points))
    mapper.SetSourceConnection(source.GetOutputPort())
    mapper.ScalingOff()
    mapper.ScalarVisibilityOff()
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    if crossing:
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(2.0)
    actor.PickableOff()
    return actor


class SurfaceMappingPresentation:
    """At most six actors plus a legend, with normal depth testing.

    Glyph centres are exact RAS locations. Endpoints use 0.12 mm spheres and
    crossings use wireframe cubes with 0.12 mm half-width to distinguish overlap.
    No global coincident-topology state is changed.
    The map replaces the target surface visually, so it needs no depth offset.
    """

    def __init__(self, panel, bundle):
        self.panel, self.bundle = panel, bundle
        self.visible = True
        self.markers_enabled = True
        self.qc_enabled = False
        self.map_enabled = False
        self.density_enabled = False
        self.endpoint_enabled = (
            bundle.result.request.parameters.mode == "endpoint_proximity"
        )
        self.intersections_enabled = (
            bundle.result.request.parameters.mode == "surface_intersections"
            or bundle.companion is not None
        )
        self.mesh_id = bundle.result.request.mesh_id
        self.map_actor = None
        self.scalar_bar = None
        assigned = bundle.marker_classes == 0
        color = ENDPOINT_COLOR if self.endpoint_enabled else CROSSING_COLOR
        self.marker_actor = _markers(
            bundle.marker_points[assigned], color, crossing=not self.endpoint_enabled
        )
        self.qc_actor = _markers(bundle.marker_points[~assigned], (0.85, 0.2, 0.8))
        self.crossing_actor = None
        self.intersection_qc_actor = None
        if bundle.companion is not None:
            companion = bundle.companion
            crossing = companion.marker_classes == 0
            self.crossing_actor = _markers(
                companion.marker_points[crossing], CROSSING_COLOR, crossing=True
            )
            self.intersection_qc_actor = _markers(
                companion.marker_points[~crossing], (0.85, 0.2, 0.8)
            )
        intervals = bundle.companion if bundle.companion is not None else bundle
        self.overlap_actor = None
        if len(intervals.overlap_points):
            data = _points_data(intervals.overlap_points.reshape(-1, 3))
            lines = vtkCellArray()
            count = len(intervals.overlap_points)
            lines.SetData(
                numpy_to_vtkIdTypeArray(
                    np.arange(count + 1, dtype=np.int64) * 2, deep=True
                ),
                numpy_to_vtkIdTypeArray(
                    np.arange(count * 2, dtype=np.int64), deep=True
                ),
            )
            data.SetLines(lines)
            tubes = vtkTubeFilter()
            tubes.SetInputData(data)
            tubes.SetRadius(OVERLAP_RADIUS_MM)
            tubes.SetNumberOfSides(6)
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(tubes.GetOutputPort())
            self.overlap_actor = vtkActor()
            self.overlap_actor.SetMapper(mapper)
            self.overlap_actor.GetProperty().SetColor(0.85, 0.2, 0.8)
            self.overlap_actor.PickableOff()
        if bundle.result.diagnostics.assigned_endpoints:
            self._create_map()
        self.actors = [
            actor
            for actor in (
                self.marker_actor,
                self.qc_actor,
                self.crossing_actor,
                self.intersection_qc_actor,
                self.overlap_actor,
                self.map_actor,
            )
            if actor is not None
        ]
        self._observer = None

    def method_enabled(self, mode):
        return (
            self.endpoint_enabled if mode == "endpoint_proximity"
            else self.intersections_enabled
        )

    def marker_visibility(self):
        primary = self.method_enabled(self.bundle.result.request.parameters.mode)
        return (
            (self.marker_actor, self.markers_enabled and primary),
            (self.crossing_actor, self.markers_enabled and self.intersections_enabled),
            (self.qc_actor, self.qc_enabled and primary),
            (
                self.intersection_qc_actor,
                self.qc_enabled and self.intersections_enabled,
            ),
            (self.overlap_actor, self.qc_enabled and self.intersections_enabled),
        )

    def _create_map(self):
        self.map_data = vtkPolyData()
        self.map_data.ShallowCopy(self.bundle.geometry.polydata)
        self.lut = vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        self.lut.SetHueRange(0.66, 0.0)
        self.lut.SetSaturationRange(0.9, 0.9)
        self.lut.SetValueRange(0.85, 1.0)
        self.lut.SetNanColor(0.5, 0.5, 0.5, 1.0)
        self.lut.Build()
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(self.map_data)
        mapper.SetLookupTable(self.lut)
        mapper.SetScalarModeToUsePointData()
        mapper.InterpolateScalarsBeforeMappingOn()
        self.map_actor = vtkActor()
        self.map_actor.SetMapper(mapper)
        self.map_actor.PickableOff()
        self.map_actor.GetProperty().LightingOff()
        self.scalar_bar = vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut)
        self.scalar_bar.SetPosition(0.04, 0.28)
        self.scalar_bar.SetWidth(0.035)
        self.scalar_bar.SetHeight(0.35)
        self.scalar_bar.SetNumberOfLabels(4)
        self.set_density(False)

    def set_density(self, enabled):
        self.density_enabled = bool(enabled)
        if self.map_actor is None:
            return
        values = (
            self.bundle.result.endpoint_density
            if enabled
            else self.bundle.result.endpoint_contributions
        )
        scalar = numpy_to_vtk(values)
        scalar.SetName(
            "endpoint_contributions_per_mm2" if enabled else "endpoint_contributions"
        )
        self.map_data.GetPointData().SetScalars(scalar)
        low, high = (
            self.bundle.density_range if enabled else self.bundle.contribution_range
        )
        if high <= low:
            low, high = 0.0, max(1.0, high)
        self.map_actor.GetMapper().SetScalarRange(low, high)
        self.lut.SetTableRange(low, high)
        self.scalar_bar.SetTitle(
            "Endpoint\ncontributions/mm²" if enabled else "Endpoint\ncontributions"
        )

    def publish(self):
        try:
            for actor in self.actors:
                self.panel.scene.add(actor)
            if self.scalar_bar is not None:
                self.panel.scene.AddActor2D(self.scalar_bar)
            self._observer = self.panel.render_window.AddObserver(
                "StartEvent", self.sync
            )
            self.sync()
        except BaseException:
            self.close()
            raise

    def render(self):
        self.sync()
        self._synced_for_render = True
        try:
            self.panel.render_window.Render()
        finally:
            self._synced_for_render = False

    def sync(self, *args):
        if args and getattr(self, "_synced_for_render", False):
            return
        if getattr(self, "_capture_override", False):
            return
        manager = self.panel.main_window.surface_mapping_manager
        pending = getattr(self.panel.main_window, "_loader_thread", None)
        stale = manager.bundle is self.bundle and not manager.current
        replacing = stale and getattr(pending, "is_consuming_result", False)
        if stale and not replacing:
            manager.invalidate()
            return
        layer = self.panel.main_window.mesh_layers.get(self.mesh_id)
        if layer is None:
            return
        visible = bool(self.visible and layer["visible"] and not replacing)
        surface_visible = bool(
            layer["visible"] and layer["presentation"]["surface"]["enabled"]
        )
        show_map = bool(
            visible
            and surface_visible
            and self.map_enabled
            and self.endpoint_enabled
            and self.map_actor is not None
        )
        surface = self.panel.mesh_actors.get(self.mesh_id)
        if surface is not None:
            surface.SetVisibility(surface_visible and not show_map)
        texture = self.panel._mesh_texture_presentations.get(self.mesh_id)
        if texture is not None:
            for part in texture.parts:
                part.actor.SetVisibility(
                    surface_visible and not show_map and texture.enabled
                )
        for actor, show in self.marker_visibility():
            if actor is not None:
                actor.SetVisibility(visible and show)
        if self.map_actor is not None:
            self.map_actor.SetVisibility(show_map)
            self.map_actor.GetProperty().SetOpacity(layer["opacity"])
            self.scalar_bar.SetVisibility(show_map)
            background = self.panel.scene.GetBackground()
            color = (0.0, 0.0, 0.0) if sum(background) > 1.5 else (1.0, 1.0, 1.0)
            self.scalar_bar.GetTitleTextProperty().SetColor(*color)
            self.scalar_bar.GetLabelTextProperty().SetColor(*color)
        self.panel._sync_mesh_transparency()

    def close(self):
        if self._observer is not None:
            self.panel.render_window.RemoveObserver(self._observer)
            self._observer = None
        for actor in self.actors:
            self.panel.scene.rm(actor)
            self.panel._release_mesh_component(actor)
            actor.SetMapper(None)
        self.actors.clear()
        if self.scalar_bar is not None:
            self.panel.scene.RemoveActor2D(self.scalar_bar)
        self.map_actor = self.marker_actor = self.qc_actor = self.overlap_actor = None
        self.crossing_actor = self.intersection_qc_actor = None
        self.scalar_bar = None
        self.map_data = None
        layer = self.panel.main_window.mesh_layers.get(self.mesh_id)
        if layer is not None and self.mesh_id in self.panel.mesh_actors:
            self.panel.set_mesh_layer_visibility(
                self.mesh_id, layer["visible"], layer=layer, render=False
            )
