# -*- coding: utf-8 -*-

"""
Mesh scene manager for TractEdit visualization.

Manages 3D mesh surface actors, overlay components, depth peeling / ordered
transparency policy, camera bounds observers, and texture presentations.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import vtk
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

from ..data_contracts import (
    MeshComponent,
    MeshComponentStyle,
    MeshLayer,
    MeshTexturePreparation,
)

if TYPE_CHECKING:
    from ..mesh_texture import MeshTextureBudgets
    from .vtk_panel import VTKPanel


def _vtk_panel_module():
    from . import vtk_panel

    return vtk_panel


# ============================================================================
# Mesh Scene Manager Class
# ============================================================================


class MeshSceneManager:
    """Manages 3D mesh layers, component overlays, transparency, and textures."""

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        self.panel = vtk_panel

    def set_interaction_multisamples(self, samples: int) -> None:
        """Keep peeling requirements and the interaction quality setting in sync."""
        panel = self.panel
        if panel._mesh_render_settings is not None:
            panel._mesh_render_settings["multisamples"] = samples
            panel.render_window.SetMultiSamples(0)
        else:
            panel.render_window.SetMultiSamples(samples)

    def sync_mesh_transparency(self) -> None:
        """Use ordered transparency only while a visible mesh part needs it."""
        panel = self.panel
        presentations = getattr(panel, "_mesh_texture_presentations", {})
        translucent = any(
            actor.GetVisibility() and actor.GetProperty().GetOpacity() > 0.0
            and (
                actor.GetProperty().GetOpacity() < 1.0
                or (
                    (presentation := presentations.get(key)) is not None
                    and presentation.enabled and presentation.has_transparency
                )
            )
            for key, actor in panel.mesh_actors.items()
        )
        if not translucent:
            mapping = getattr(panel.main_window, "surface_mapping_manager", None)
            presentation = getattr(mapping, "presentation", None)
            mapped_surface = getattr(presentation, "map_actor", None)
            translucent = bool(
                mapped_surface is not None and mapped_surface.GetVisibility()
                and 0.0 < mapped_surface.GetProperty().GetOpacity() < 1.0
            )
        if not translucent:
            translucent = any(
                actor.GetVisibility() and 0.0 < actor.GetProperty().GetOpacity() < 1.0
                for components in getattr(panel, "_mesh_component_actors", {}).values()
                for actor in components.values()
            )
        if translucent and panel._mesh_render_settings is None:
            panel._mesh_render_settings = {
                "multisamples": panel.render_window.GetMultiSamples(),
                "depth_peeling": panel.scene.GetUseDepthPeeling(),
                "maximum_peels": panel.scene.GetMaximumNumberOfPeels(),
                "occlusion_ratio": panel.scene.GetOcclusionRatio(),
            }
            panel.render_window.SetMultiSamples(0)
            panel.scene.SetUseDepthPeeling(True)
            panel.scene.SetMaximumNumberOfPeels(100)
            panel.scene.SetOcclusionRatio(0.0)
        elif not translucent and panel._mesh_render_settings is not None:
            settings = panel._mesh_render_settings
            panel.render_window.SetMultiSamples(settings["multisamples"])
            panel.scene.SetUseDepthPeeling(settings["depth_peeling"])
            panel.scene.SetMaximumNumberOfPeels(settings["maximum_peels"])
            panel.scene.SetOcclusionRatio(settings["occlusion_ratio"])
            panel._mesh_render_settings = None

    def add_mesh_layer(self, layer: MeshLayer, *, render: bool = True) -> vtk.vtkActor:
        """Add a surface to 3D only; the MainWindow owns its data and properties."""
        panel = self.panel
        vp_mod = _vtk_panel_module()
        key = layer["id"]
        if key in panel.mesh_actors:
            raise ValueError(f"Mesh actor already exists: {key}")
        mesh_actor = vp_mod.create_mesh_actor(
            layer["polydata"], layer["source_to_ras"], layer["color"], layer["opacity"]
        )
        mesh_actor.SetVisibility(
            layer["visible"] and layer["presentation"]["surface"]["enabled"]
        )
        has_data = bool(panel.mesh_actors or panel.roi_slice_actors) or any(
            getattr(panel.main_window, name, None) is not None
            for name in (
                "tractogram_data",
                "anatomical_image_data",
                "parcellation_data",
                "odf_data",
            )
        )
        camera = panel.scene.GetActiveCamera()
        previous_camera = vtk.vtkCamera()
        previous_camera.DeepCopy(camera)
        try:
            panel.scene.add(mesh_actor)
            panel.mesh_actors[key] = mesh_actor
            if any(
                layer["presentation"][name]["enabled"]
                for name in ("edges", "vertices", "bounding_box")
            ):
                panel.set_mesh_layer_visibility(
                    key, layer["visible"], layer=layer, render=False,
                )
            panel._sync_mesh_transparency()
            if not has_data and layer["visible"]:
                panel.scene.ResetCamera(mesh_actor.GetBounds())
            panel.scene.reset_clipping_range()
            if render:
                panel.render_window.Render()
        except Exception:
            panel._discard_mesh_components(key)
            panel.mesh_actors.pop(key, None)
            try:
                panel.scene.rm(mesh_actor)
            finally:
                panel._sync_mesh_transparency()
                camera.DeepCopy(previous_camera)
            raise
        return mesh_actor

    def remove_mesh_layer(self, key: str, *, render: bool = True) -> None:
        """Remove the named mesh and its derived rendering resources."""
        panel = self.panel
        mesh_actor = panel.mesh_actors.get(key)
        if mesh_actor is None:
            return
        panel.scene.rm(mesh_actor)
        panel.discard_mesh_texture(key)
        panel._discard_mesh_components(key)
        del panel.mesh_actors[key]
        panel._sync_mesh_transparency()
        panel.scene.reset_clipping_range()
        if render:
            panel.render_window.Render()

    def release_mesh_component(self, actor: vtk.vtkActor) -> None:
        panel = self.panel
        for retained in getattr(actor, "_mesh_preview_levels", {}).values():
            if isinstance(panel.render_window, vtk.vtkWindow):
                retained.ReleaseGraphicsResources(panel.render_window)
            retained.RemoveAllInputs()
        for name in ("_mesh_preview_levels", "_mesh_preview_input"):
            if hasattr(actor, name):
                delattr(actor, name)
        for name in (
            "_mesh_interaction_mapper",
            "_mesh_full_mapper",
            "_mesh_preview_base_mapper",
        ):
            retained = getattr(actor, name, None)
            if retained is not None:
                if isinstance(panel.render_window, vtk.vtkWindow):
                    retained.ReleaseGraphicsResources(panel.render_window)
                retained.RemoveAllInputs()
                delattr(actor, name)
        mapper = actor.GetMapper()
        if isinstance(panel.render_window, vtk.vtkWindow):
            actor.ReleaseGraphicsResources(panel.render_window)
        actor.SetMapper(None)
        if mapper is not None:
            mapper.RemoveAllInputs()

    def discard_mesh_components(self, key: str) -> None:
        panel = self.panel
        getattr(panel, "_mesh_overlay_data", {}).pop(key, None)
        registry = getattr(panel, "_mesh_component_actors", {})
        for overlay in registry.pop(key, {}).values():
            panel.scene.rm(overlay)
            panel._release_mesh_component(overlay)
        panel._sync_mesh_camera_observers()

    def sync_mesh_camera_observers(self) -> None:
        from ..mesh.presentation import ADVANCED_MESH_COMPONENTS

        panel = self.panel
        panel._sync_mesh_interaction_budget()
        active = any(
            (name in ADVANCED_MESH_COMPONENTS
             or hasattr(actor, "_mesh_interaction_mapper")) and actor.GetVisibility()
            for components in getattr(panel, "_mesh_component_actors", {}).values()
            for name, actor in components.items()
        )
        observers = getattr(panel, "_mesh_camera_observers", ())
        if active and not observers:
            panel._mesh_camera_observers = (
                panel.scene.AddObserver("ResetCameraEvent", panel._reset_mesh_camera),
                panel.scene.AddObserver(
                    "ResetCameraClippingRangeEvent", panel._reset_mesh_clipping,
                ),
            )
        elif not active and observers:
            for observer in observers:
                panel.scene.RemoveObserver(observer)
            panel._mesh_camera_observers = ()

    def mesh_scene_bounds(self, *, clipping: bool) -> tuple[float, ...] | None:
        from ..mesh.overlays import mesh_ras_bounds
        from ..mesh.presentation import ADVANCED_MESH_COMPONENTS

        panel = self.panel
        bounds = [panel.scene.ComputeVisiblePropBounds()]
        for key, components in panel._mesh_component_actors.items():
            visible = [
                actor for name, actor in components.items()
                if (name in ADVANCED_MESH_COMPONENTS
                    or hasattr(actor, "_mesh_full_mapper"))
                and actor.GetVisibility()
            ]
            if not visible:
                continue
            if clipping:
                for actor in visible:
                    mapper = getattr(actor, "_mesh_full_mapper", actor.GetMapper())
                    local = mapper.GetBounds()
                    if local is None or any(local[i] > local[i + 1] for i in (0, 2, 4)):
                        continue
                    matrix = actor.GetMatrix()
                    corners = np.array([
                        matrix.MultiplyPoint((x, y, z, 1))[:3]
                        for x in local[:2] for y in local[2:4] for z in local[4:6]
                    ])
                    bounds.append(tuple(np.column_stack((
                        corners.min(axis=0), corners.max(axis=0),
                    )).ravel()))
            else:
                layer = getattr(panel.main_window, "mesh_layers", {}).get(key)
                bounds.append(
                    mesh_ras_bounds(layer) if layer is not None
                    else panel.mesh_actors[key].GetBounds()
                )
        valid = [
            np.asarray(value).reshape(3, 2) for value in bounds
            if value is not None and all(value[i] <= value[i + 1] for i in (0, 2, 4))
        ]
        if not valid:
            return None
        values = np.stack(valid)
        return tuple(np.column_stack((
            values[:, :, 0].min(axis=0),
            values[:, :, 1].max(axis=0),
        )).ravel())

    def reset_mesh_camera(self, caller: Any, event: Any) -> None:
        panel = self.panel
        bounds = panel._mesh_scene_bounds(clipping=False)
        if bounds is not None:
            panel.scene.ResetCamera(bounds)
            panel._reset_mesh_clipping(caller, event)

    def reset_mesh_clipping(self, caller: Any, event: Any) -> None:
        panel = self.panel
        bounds = panel._mesh_scene_bounds(clipping=True)
        if bounds is not None:
            panel.scene.ResetCameraClippingRange(bounds)

    def mesh_overlay_data(self, layer: MeshLayer, component: str):
        from ..mesh.overlays import mesh_geometry_key

        cached = getattr(self.panel, "_mesh_overlay_data", {}).get(
            layer["id"], {}
        ).get(component)
        if cached is not None and cached[0] == mesh_geometry_key(layer):
            return cached[1]
        return None

    def publish_mesh_overlay(
        self, layer: MeshLayer, component: str, data, *, render: bool = True,
    ) -> None:
        """Publish prepared CPU geometry and its actor atomically in the Qt thread."""
        from ..mesh.overlays import mesh_geometry_key

        panel = self.panel
        vp_mod = _vtk_panel_module()
        key = layer["id"]
        registry = getattr(panel, "_mesh_overlay_data", None)
        if registry is None:
            registry = panel._mesh_overlay_data = {}
        existing = registry.get(key, {})
        actors = getattr(panel, "_mesh_component_actors", None)
        if actors is None:
            actors = panel._mesh_component_actors = {}
        previous_actors = actors.get(key, {})
        previous = previous_actors.get(component)
        clipping = panel.scene.GetActiveCamera().GetClippingRange()
        candidate = vp_mod.create_mesh_advanced_actor(layer, component, data)
        candidate.SetVisibility(
            layer["visible"] and layer["presentation"][component]["enabled"],
        )
        was_visible = previous.GetVisibility() if previous is not None else False
        try:
            panel.scene.add(candidate)
            if previous is not None:
                previous.SetVisibility(False)
            actors[key] = {**previous_actors, component: candidate}
            registry[key] = {**existing, component: (mesh_geometry_key(layer), data)}
            panel._sync_mesh_camera_observers()
            panel.scene.reset_clipping_range()
            if render:
                panel.render_window.Render()
        except Exception:
            panel.scene.rm(candidate)
            panel._release_mesh_component(candidate)
            if previous_actors:
                actors[key] = previous_actors
            else:
                actors.pop(key, None)
            if previous is not None:
                previous.SetVisibility(was_visible)
            if existing:
                registry[key] = existing
            else:
                registry.pop(key, None)
            panel._sync_mesh_camera_observers()
            panel.scene.GetActiveCamera().SetClippingRange(clipping)
            raise
        if previous is not None:
            panel.scene.rm(previous)
            panel._release_mesh_component(previous)

    def set_mesh_layer_visibility(
        self, key: str, visible: bool, *, layer: MeshLayer | None = None,
        render: bool = True,
    ) -> bool:
        """Apply component visibility atomically and report whether it rendered."""
        panel = self.panel
        vp_mod = _vtk_panel_module()
        mesh_actor = panel.mesh_actors.get(key)
        if mesh_actor is None:
            return False
        if layer is None:
            layer = getattr(panel.main_window, "mesh_layers", {}).get(key)
        registry = getattr(panel, "_mesh_component_actors", None)
        if registry is None:
            registry = panel._mesh_component_actors = {}
        existing = registry.get(key, {})
        presentation = getattr(panel, "_mesh_texture_presentations", {}).get(key)
        actors = [mesh_actor, *existing.values()]
        if presentation is not None:
            actors.extend(part.actor for part in presentation.parts)
        previous = [(actor, actor.GetVisibility()) for actor in actors]
        clipping = panel.scene.GetActiveCamera().GetClippingRange()
        created = {}
        try:
            if visible and layer is not None:
                for component in (
                    "edges", "vertices", "bounding_box", "boundary_edges",
                    "surface_normals",
                ):
                    if (
                        layer["presentation"][component]["enabled"]
                        and component not in existing
                    ):
                        if component in ("edges", "vertices"):
                            overlay = vp_mod.create_mesh_component_actor(layer, component)
                        else:
                            data = None
                            if component != "bounding_box":
                                data = panel.mesh_overlay_data(layer, component)
                                if data is None:
                                    continue
                            overlay = vp_mod.create_mesh_advanced_actor(
                                layer, component, data
                            )
                        created[component] = overlay
                        panel.scene.add(overlay)
            components = {**existing, **created}
            surface_visible = bool(visible and (
                layer is None or layer["presentation"]["surface"]["enabled"]
            ))
            mesh_actor.SetVisibility(surface_visible)
            if presentation is not None:
                for part in presentation.parts:
                    part.actor.SetVisibility(surface_visible and presentation.enabled)
            for component, overlay in components.items():
                overlay.SetVisibility(bool(visible and (
                    layer is None or layer["presentation"][component]["enabled"]
                )))
            if components:
                registry[key] = components
            panel._sync_mesh_camera_observers()
            panel._sync_mesh_transparency()
            changed = created or any(
                actor.GetVisibility() != was_visible for actor, was_visible in previous
            )
            if changed:
                panel.scene.reset_clipping_range()
            if render and changed:
                panel.render_window.Render()
                return True
            return False
        except Exception:
            for overlay in created.values():
                panel.scene.rm(overlay)
                panel._release_mesh_component(overlay)
            if existing:
                registry[key] = existing
            else:
                registry.pop(key, None)
            for previous_actor, was_visible in previous:
                previous_actor.SetVisibility(was_visible)
            panel._sync_mesh_camera_observers()
            panel._sync_mesh_transparency()
            panel.scene.GetActiveCamera().SetClippingRange(clipping)
            raise

    def set_mesh_component_style(
        self, key: str, component: MeshComponent, style: MeshComponentStyle,
        *, render: bool = True,
    ) -> None:
        """Update a resident overlay without constructing an inactive component."""
        from ..mesh.presentation import validate_mesh_component_style

        panel = self.panel
        values = validate_mesh_component_style(component, style)
        components = getattr(panel, "_mesh_component_actors", {}).get(key, {})
        overlay = components.get(component)
        if overlay is None:
            return
        properties = overlay.GetProperty()
        previous = vtk.vtkProperty()
        previous.DeepCopy(properties)
        mapper = overlay.GetMapper()
        normals = component == "surface_normals"
        previous_input = mapper.GetInput() if normals else None
        previous_length = mapper.GetScaleFactor() if normals else None
        previous_clipping = (
            panel.scene.GetActiveCamera().GetClippingRange() if normals else None
        )
        try:
            if "color" in values:
                properties.SetColor(*values["color"])
            if "line_width" in values:
                properties.SetLineWidth(values["line_width"])
            if "point_size" in values:
                properties.SetPointSize(values["point_size"])
            if normals:
                if "color" in values:
                    mapper.Modified()
                if "length_mm" in values:
                    mapper.SetScaleFactor(values["length_mm"])
                if "density" in values:
                    from ..mesh.overlays import sampled_normal_polydata

                    layer = panel.main_window.mesh_layers[key]
                    mapper.SetInputData(sampled_normal_polydata(
                        panel.mesh_overlay_data(layer, component), values["density"],
                    ))
                if overlay.GetVisibility():
                    panel.scene.reset_clipping_range()
            if render and overlay.GetVisibility():
                panel.render_window.Render()
        except Exception:
            properties.DeepCopy(previous)
            if normals:
                mapper.SetInputData(previous_input)
                mapper.SetScaleFactor(previous_length)
                mapper.Modified()
                panel.scene.GetActiveCamera().SetClippingRange(previous_clipping)
            raise

    def set_mesh_opacity(self, key: str, value: float, *, render: bool = True) -> None:
        """Change only surface opacity; source data and the camera stay intact."""
        panel = self.panel
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Mesh opacity must be finite and in [0, 1]")
        mesh_actor = panel.mesh_actors.get(key)
        if mesh_actor is None:
            return
        mesh_actor.GetProperty().SetOpacity(value)
        presentation = getattr(panel, "_mesh_texture_presentations", {}).get(key)
        if presentation is not None:
            for part in presentation.parts:
                part.actor.GetProperty().SetOpacity(value)
        panel._sync_mesh_transparency()
        if render:
            panel.render_window.Render()

    def set_mesh_color(
        self, key: str, color: tuple[float, float, float], *, render: bool = True
    ) -> None:
        """Update the uniform surface colour without rebuilding its mapper."""
        panel = self.panel
        mesh_actor = panel.mesh_actors.get(key)
        if mesh_actor is None:
            return
        presentation = getattr(panel, "_mesh_texture_presentations", {}).get(key)
        if presentation is not None:
            presentation.uniform_color = tuple(color)
            for part in presentation.parts:
                if not part.textured:
                    part.actor.GetProperty().SetColor(*color)
        if presentation is None or not presentation.enabled:
            mesh_actor.GetProperty().SetColor(*color)
        if render:
            panel.render_window.Render()

    def maximum_texture_size(self) -> int:
        panel = self.panel
        vp_mod = _vtk_panel_module()
        panel.render_window.MakeCurrent()
        maximum = vp_mod.vtk.vtkTextureObject.GetMaximumTextureSize(panel.render_window)
        if maximum <= 0:
            raise ValueError("Could not query the OpenGL texture dimension limit")
        return maximum

    def mesh_texture_enabled(self, key: str) -> bool:
        presentation = getattr(self.panel, "_mesh_texture_presentations", {}).get(key)
        return presentation is not None and presentation.enabled

    def set_mesh_texture(
        self, layer: MeshLayer, preparation: MeshTexturePreparation | None,
        enabled: bool, *, gl_maximum: int | None = None,
        budgets: "MeshTextureBudgets | None" = None, render: bool = True,
    ) -> None:
        """Atomically switch all layer parts; retain resources for warm toggles."""
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            raise RuntimeError("Texture rendering requires the Qt thread")
        from .mesh_texture import create_mesh_texture_presentation, derived_uv_bytes
        from ..mesh_texture import DEFAULT_BUDGETS, MeshTextureError

        panel = self.panel
        key = layer["id"]
        mesh_actor = panel.mesh_actors[key]
        registry = getattr(panel, "_mesh_texture_presentations", None)
        if registry is None:
            registry = panel._mesh_texture_presentations = {}
        previous = registry.get(key)
        candidate = previous
        if enabled and (previous is None or previous.preparation is not preparation):
            resident = preparation["costs"]["resident_estimate_bytes"]
            if previous is not None:
                resident += previous.preparation["costs"]["resident_estimate_bytes"]
                preparations = (previous.preparation, preparation)
                buffers = {
                    id(image["pixels"]): image["pixels"].nbytes
                    for result in preparations for image in result["images"].values()
                }
                buffers.update({
                    id(result["face_uv"]): result["face_uv"].nbytes
                    for result in preparations if "face_uv" in result
                })
                transient = sum(buffers.values()) + sum(
                    derived_uv_bytes(layer, result) for result in preparations
                )
                if transient > (budgets or DEFAULT_BUDGETS).temporary_cpu_bytes:
                    raise MeshTextureError(
                        "texture_budget",
                        "Existing and new presentations exceed the CPU budget",
                    )
            if resident > (budgets or DEFAULT_BUDGETS).resident_estimate_bytes:
                raise MeshTextureError(
                    "texture_budget",
                    "Existing and new textures exceed the resident estimate",
                )
            if gl_maximum is None:
                gl_maximum = panel.maximum_texture_size()
            if type(gl_maximum) is not int or gl_maximum <= 0:
                raise ValueError("Texture dimension limit must be a positive integer")
            candidate = create_mesh_texture_presentation(
                layer, preparation,
                previous.uniform_mapper
                if previous is not None else mesh_actor.GetMapper(),
                gl_maximum=gl_maximum, budgets=budgets or DEFAULT_BUDGETS,
            )
            candidate.uniform_lighting = (
                previous.uniform_lighting if previous is not None
                else mesh_actor.GetProperty().GetLighting()
            )
        if candidate is None:
            return
        old_mapper, old_texture = mesh_actor.GetMapper(), mesh_actor.GetTexture()
        old_color = mesh_actor.GetProperty().GetColor()
        old_lighting = mesh_actor.GetProperty().GetLighting()
        old_force = mesh_actor.GetForceTranslucent()
        old_enabled = candidate.enabled
        removed, added = [], []
        try:
            if previous is not None and previous is not candidate:
                for part in previous.parts:
                    removed.append(part.actor)
                    panel.scene.rm(part.actor)
            if candidate is not previous:
                for part in candidate.parts:
                    added.append(part.actor)
                    panel.scene.add(part.actor)
            for part in candidate.parts:
                part.actor.GetProperty().SetOpacity(
                    mesh_actor.GetProperty().GetOpacity()
                )
                part.actor.SetVisibility(bool(enabled and mesh_actor.GetVisibility()))
            mesh_actor.SetMapper(
                candidate.mapper if enabled else candidate.uniform_mapper
            )
            mesh_actor.SetTexture(candidate.texture if enabled else None)
            mesh_actor.GetProperty().SetLighting(
                False if enabled and candidate.texture is not None
                else candidate.uniform_lighting
            )
            mesh_actor.GetProperty().SetColor(
                *(candidate.diffuse_color if enabled else candidate.uniform_color)
            )
            primary_alpha = candidate.primary_has_transparency
            if primary_alpha is None:
                primary_alpha = candidate.has_transparency
            mesh_actor.SetForceTranslucent(bool(enabled and primary_alpha))
            candidate.enabled = bool(enabled)
            registry[key] = candidate
            panel._sync_mesh_transparency()
            if render:
                panel.render_window.Render()
        except Exception:
            for actor in added:
                panel.scene.rm(actor)
            for actor in removed:
                panel.scene.add(actor)
            for part in candidate.parts:
                part.actor.SetVisibility(
                    bool(old_enabled and mesh_actor.GetVisibility())
                )
            mesh_actor.SetMapper(old_mapper)
            mesh_actor.SetTexture(old_texture)
            mesh_actor.GetProperty().SetColor(*old_color)
            mesh_actor.GetProperty().SetLighting(old_lighting)
            mesh_actor.SetForceTranslucent(old_force)
            candidate.enabled = old_enabled
            if previous is None:
                registry.pop(key, None)
            else:
                registry[key] = previous
            panel._sync_mesh_transparency()
            if candidate is not previous:
                candidate.release(panel.render_window)
            raise
        if previous is not None and previous is not candidate:
            previous.release(panel.render_window)

    def discard_mesh_texture(self, key: str) -> None:
        """Retire inactive cache resources and restore the uniform actor."""
        panel = self.panel
        presentation = getattr(panel, "_mesh_texture_presentations", {}).pop(key, None)
        if presentation is None:
            return
        mesh_actor = panel.mesh_actors.get(key)
        for part in presentation.parts:
            panel.scene.rm(part.actor)
        if mesh_actor is not None:
            mesh_actor.SetTexture(None)
            mesh_actor.SetMapper(presentation.uniform_mapper)
            mesh_actor.GetProperty().SetColor(*presentation.uniform_color)
            mesh_actor.GetProperty().SetLighting(presentation.uniform_lighting)
            mesh_actor.ForceTranslucentOff()
        presentation.release(panel.render_window)
        panel._sync_mesh_transparency()

    def fit_mesh_in_view(self, key: str, *, render: bool = True) -> None:
        """Frame one surface's RAS bounds, preserving the viewing direction."""
        panel = self.panel
        mesh_actor = panel.mesh_actors.get(key)
        if mesh_actor is None:
            return
        presentation = getattr(panel, "_mesh_texture_presentations", {}).get(key)
        if presentation is not None and presentation.enabled:
            full_actor = vtk.vtkActor()
            full_actor.SetMapper(presentation.uniform_mapper)
            full_actor.SetUserMatrix(mesh_actor.GetUserMatrix())
            bounds = full_actor.GetBounds()
        else:
            bounds = mesh_actor.GetBounds()
        panel.scene.ResetCamera(bounds)
        panel.scene.reset_clipping_range()
        if render:
            panel.render_window.Render()
