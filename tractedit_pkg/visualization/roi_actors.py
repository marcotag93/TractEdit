# -*- coding: utf-8 -*-

"""
ROI actor manager for TractEdit visualization.

Manages 2D/3D ROI slicer actors, 3D contour/sphere/rectangle actors,
LUT coloring, opacity, visibility, and slice-extent synchronization.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)

_ROI_SLICE_ACTOR_KEYS = (
    "axial_3d",
    "coronal_3d",
    "sagittal_3d",
    "axial_2d",
    "coronal_2d",
    "sagittal_2d",
)


def _vtk_panel_module():
    from . import vtk_panel

    return vtk_panel


def _crop_roi_for_contour(
    data: np.ndarray, affine: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the occupied ROI bounds with a one-voxel background margin."""
    positive = data > 0
    occupied_axes = [
        np.flatnonzero(np.any(positive, axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))
    ]
    if any(axis.size == 0 for axis in occupied_axes):
        return None

    lower = np.array([axis[0] for axis in occupied_axes], dtype=int)
    upper = np.array([axis[-1] + 1 for axis in occupied_axes], dtype=int)
    lower = np.maximum(lower - 1, 0)
    upper = np.minimum(upper + 1, np.asarray(data.shape))
    slices = tuple(slice(start, stop) for start, stop in zip(lower, upper))

    translation = np.eye(4)
    translation[:3, 3] = lower
    return data[slices], affine @ translation


# ============================================================================
# ROI Scene Manager Class
# ============================================================================


class ROISceneManager:
    """Manages 2D and 3D ROI actors owned by a VTKPanel."""

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        self.panel = vtk_panel

    def update_roi_data_in_place(self, key: str, data: np.ndarray) -> bool:
        """Update the shared FURY slicer source without replacing its actors."""
        panel = self.panel
        actor_set = panel.roi_slice_actors.get(key)
        if not actor_set or data.ndim != 3:
            return False
        data_min = np.min(data)
        data_max = np.max(data)
        if not np.isfinite([data_min, data_max]).all():
            return False
        if data_min < 0 or data_max > 1:
            return False

        pipelines = []
        for actor_key in _ROI_SLICE_ACTOR_KEYS:
            actor_obj = actor_set.get(actor_key)
            if actor_obj is None:
                return False

            top_algorithm = actor_obj.GetMapper().GetInputAlgorithm()
            source_algorithm = top_algorithm
            while source_algorithm.GetInputAlgorithm() is not None:
                source_algorithm = source_algorithm.GetInputAlgorithm()

            source_image = source_algorithm.GetOutputDataObject(0)
            if not isinstance(source_image, vtk.vtkImageData):
                return False
            if tuple(source_image.GetDimensions()) != tuple(data.shape):
                return False
            pipelines.append((top_algorithm, source_image))

        volume = np.ascontiguousarray(np.swapaxes(data, 0, 2))
        vtk_array = numpy_support.numpy_to_vtk(volume.ravel(), deep=False)

        unique_sources = {}
        unique_pipelines = {}
        for top_algorithm, source_image in pipelines:
            unique_sources[id(source_image)] = source_image
            unique_pipelines[id(top_algorithm)] = top_algorithm

        for source_image in unique_sources.values():
            source_image.GetPointData().SetScalars(vtk_array)
            source_image.Modified()
        actor_set["axial_3d"]._roi_scalar_buffer = volume
        for top_algorithm in unique_pipelines.values():
            top_algorithm.Update()

        for actor_key in _ROI_SLICE_ACTOR_KEYS:
            actor_obj = actor_set[actor_key]
            actor_obj.GetMapper().Modified()
            actor_obj.Modified()
        return True

    def update_roi_layer(
        self, key: str, data: np.ndarray, affine: np.ndarray, render: bool = True
    ) -> None:
        """Update an ROI layer, optionally leaving the final render to the caller."""
        panel = self.panel
        old_actors = panel.roi_slice_actors.get(key)
        if old_actors and panel._update_roi_data_in_place(key, data):
            panel._sync_roi_3d_actor(key, data, affine, render=False)
        else:
            panel.add_roi_layer(key, data, affine, render=False)
        if old_actors and panel.roi_slice_actors.get(key) is not old_actors:
            panel._remove_actor_set(old_actors)
        if render:
            panel._render_all()

    def set_roi_opacity(self, key: str, value: float) -> None:
        """Sets the opacity for a specific ROI layer."""
        panel = self.panel
        if key in panel.roi_slice_actors:
            for act in panel.roi_slice_actors[key].values():
                if act:
                    act.GetProperty().SetOpacity(value)

            panel._render_all()

    def update_slice_extents(
        self, slice_moved: Dict[str, bool], new_x: int, new_y: int, new_z: int
    ) -> None:
        """Synchronize all ROI 3D/2D slicer extents with moved anatomical slices."""
        panel = self.panel
        if not panel.main_window or not panel.roi_slice_actors or not panel.image_extents["x"]:
            return

        c_x = (panel.image_extents["x"][0] + panel.image_extents["x"][1]) / 2.0
        c_y = (panel.image_extents["y"][0] + panel.image_extents["y"][1]) / 2.0
        c_z = (panel.image_extents["z"][0] + panel.image_extents["z"][1]) / 2.0

        plane_specs = (
            ("x", 0, "sagittal", np.array([new_x, c_y, c_z, 1.0])),
            ("y", 1, "coronal", np.array([c_x, new_y, c_z, 1.0])),
            ("z", 2, "axial", np.array([c_x, c_y, new_z, 1.0])),
        )

        for key, actor_dict in panel.roi_slice_actors.items():
            roi_info = panel.main_window.roi_layers.get(key)
            if not roi_info or "T_main_to_roi" not in roi_info:
                continue

            T_main_to_roi = roi_info["T_main_to_roi"]
            roi_shape = roi_info["data"].shape
            roi_exts = (
                (0, roi_shape[0] - 1),
                (0, roi_shape[1] - 1),
                (0, roi_shape[2] - 1),
            )

            try:
                for axis_name, axis_idx, view_name, plane_center in plane_specs:
                    if not slice_moved[axis_name]:
                        continue
                    roi_vox_c = T_main_to_roi.dot(plane_center)
                    low, high = roi_exts[axis_idx]
                    new_roi_idx = max(low, min(int(round(roi_vox_c[axis_idx])), high))
                    extent = [
                        roi_exts[0][0], roi_exts[0][1],
                        roi_exts[1][0], roi_exts[1][1],
                        roi_exts[2][0], roi_exts[2][1],
                    ]
                    extent[2 * axis_idx] = new_roi_idx
                    extent[2 * axis_idx + 1] = new_roi_idx
                    for suffix in ("3d", "2d"):
                        act = actor_dict.get(f"{view_name}_{suffix}")
                        if act:
                            act.display_extent(*extent)
            except (RuntimeError, ValueError, KeyError) as e:
                logger.error(f"Error updating ROI layer {key} slice: {e}")

    def remove_roi_actor_entries(self, key: str, *actor_keys: str) -> None:
        """Remove selected ROI actor entries from their scenes and registry."""
        panel = self.panel
        actor_set = panel.roi_slice_actors.get(key)
        if not actor_set:
            return

        removed = {}
        for actor_key in actor_keys:
            actor_obj = actor_set.pop(actor_key, None)
            if isinstance(actor_obj, vtk.vtkProp):
                removed[actor_key] = actor_obj
        if removed:
            panel._remove_actor_set(removed)

    def sync_roi_3d_actor(
        self,
        key: str,
        data: np.ndarray,
        affine: np.ndarray,
        render: bool,
    ) -> None:
        """Synchronize the ROI's sphere, rectangle, or contour representation."""
        panel = self.panel
        actor_set = panel.roi_slice_actors.get(key)
        if actor_set is None:
            return

        assigned_color = panel.main_window.roi_layers[key].get("color", (1.0, 0.0, 0.0))
        visibility = int(panel.main_window.roi_visibility.get(key, True))

        if key in panel.sphere_params_per_roi:
            panel._remove_roi_actor_entries(
                key, "rectangle_3d", "rectangle_points", "contour_3d"
            )
            params = panel.sphere_params_per_roi[key]
            panel.drawing_manager.update_3d_sphere_visuals(
                key, params["center"], params["radius"], render=render
            )
            sphere_actor = actor_set.get("sphere_3d")
            if sphere_actor is not None:
                sphere_actor.SetVisibility(visibility)
            return

        if key in panel.rectangle_params_per_roi:
            panel._remove_roi_actor_entries(
                key, "sphere_3d", "sphere_source", "contour_3d"
            )
            params = panel.rectangle_params_per_roi[key]
            panel.drawing_manager.update_3d_rectangle_visuals(
                key,
                np.asarray(params["start"]),
                np.asarray(params["end"]),
                params.get("view_type", "axial"),
                render=render,
                corners=params.get("corners"),
            )
            rectangle_actor = actor_set.get("rectangle_3d")
            if rectangle_actor is not None:
                rectangle_actor.SetVisibility(visibility)
            return

        panel._remove_roi_actor_entries(
            key,
            "sphere_3d",
            "sphere_source",
            "rectangle_3d",
            "rectangle_points",
        )
        cropped = _crop_roi_for_contour(data, affine)
        old_contour = actor_set.get("contour_3d")
        if cropped is None:
            panel._remove_roi_actor_entries(key, "contour_3d")
            return

        cropped_data, cropped_affine = cropped
        try:
            contour_actor = _vtk_panel_module().actor.contour_from_roi(
                cropped_data,
                affine=cropped_affine,
                color=assigned_color,
                opacity=0.5,
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning(f"Could not create 3D contour for ROI {key}: {e}")
            return

        if contour_actor is None:
            return
        contour_actor.SetVisibility(visibility)
        panel.scene.add(contour_actor)
        actor_set["contour_3d"] = contour_actor
        if old_contour is not None and old_contour is not contour_actor:
            panel._remove_actor_set({"contour_3d": old_contour})

    def add_roi_layer(
        self, key: str, data: np.ndarray, affine: np.ndarray, render: bool = True
    ) -> None:
        """Creates and adds slicer actors for a new ROI layer (separating 2D/3D actors)."""
        panel = self.panel
        vp_mod = _vtk_panel_module()
        if (
            not panel.scene
            or not panel.main_window
            or panel.main_window.anatomical_image_affine is None
        ):
            logger.error("Cannot add ROI layer: Main scene or image not initialized.")
            return

        data_min = np.min(data)
        data_max = np.max(data)

        # If data is uniform (e.g., all zeros), set range to (0, 1).
        # This ensures 0 maps to the start of the LUT (Transparent) and 1 maps to the end (Color).
        if data_max <= data_min:
            value_range = (0, 1)
        else:
            value_range = (data_min, data_max)

        # Get opacity from MainWindow state
        slicer_opacity = 0.5  # Default
        if (
            hasattr(panel.main_window, "roi_opacities")
            and key in panel.main_window.roi_opacities
        ):
            slicer_opacity = panel.main_window.roi_opacities[key]

        interpolation_mode = "nearest"

        # Get current slice indices
        c_idx = panel.current_slice_indices
        if c_idx["x"] is None:
            panel.update_status("Error: Slices not initialized.")
            return

        # Calculate initial ROI slice indices
        main_img_affine = panel.main_window.anatomical_image_affine
        roi_info = panel.main_window.roi_layers.get(key)
        if not roi_info or "inv_affine" not in roi_info:
            return

        inv_roi_affine = roi_info["inv_affine"]
        c_x, c_y, c_z = c_idx["x"], c_idx["y"], c_idx["z"]
        world_c = panel._voxel_to_world([c_x, c_y, c_z], main_img_affine)
        roi_vox_c = panel._world_to_voxel(world_c, inv_roi_affine)
        roi_x, roi_y, roi_z = (
            int(round(roi_vox_c[0])),
            int(round(roi_vox_c[1])),
            int(round(roi_vox_c[2])),
        )

        # Get ROI extents
        roi_shape = data.shape
        roi_x_ext = (0, roi_shape[0] - 1)
        roi_y_ext = (0, roi_shape[1] - 1)
        roi_z_ext = (0, roi_shape[2] - 1)
        roi_x = max(roi_x_ext[0], min(roi_x, roi_x_ext[1]))
        roi_y = max(roi_y_ext[0], min(roi_y, roi_y_ext[1]))
        roi_z = max(roi_z_ext[0], min(roi_z, roi_z_ext[1]))

        try:
            ax_3d = vp_mod.create_slicer_actor(
                data,
                affine,
                value_range,
                slicer_opacity,
                interpolation_mode,
            )
            if ax_3d is None:
                raise RuntimeError("Could not create ROI slicer actor.")
            cor_3d = ax_3d.copy()
            sag_3d = ax_3d.copy()
            ax_2d = ax_3d.copy()
            cor_2d = ax_3d.copy()
            sag_2d = ax_3d.copy()

            for slicer_actor in (ax_3d, cor_3d, sag_3d, sag_2d):
                vp_mod.apply_slicer_affine(slicer_actor, affine)
            for slicer_actor in (ax_2d, cor_2d):
                vp_mod.apply_slicer_affine(slicer_actor, affine, radiological=True)

            is_visible = True
            if panel.main_window:
                is_visible = panel.main_window.roi_visibility.get(key, True)
            vis_flag = 1 if is_visible else 0

            for act in [ax_3d, cor_3d, sag_3d, ax_2d, cor_2d, sag_2d]:
                act.SetVisibility(vis_flag)

            # Set initial display extents for ALL actors
            for ax in [ax_3d, ax_2d]:
                ax.display_extent(
                    roi_x_ext[0], roi_x_ext[1], roi_y_ext[0], roi_y_ext[1], roi_z, roi_z
                )
            for cor in [cor_3d, cor_2d]:
                cor.display_extent(
                    roi_x_ext[0], roi_x_ext[1], roi_y, roi_y, roi_z_ext[0], roi_z_ext[1]
                )
            for sag in [sag_3d, sag_2d]:
                sag.display_extent(
                    roi_x,
                    roi_x,
                    roi_y_ext[0],
                    roi_y_ext[1],
                    roi_z_ext[0],
                    roi_z_ext[1],
                )

            # Retain compositing order when a non-binary edit rebuilds a layer.
            previous = panel.roi_slice_actors.get(key, {}).get("axial_2d")
            if previous is not None:
                layer = previous.GetProperty().GetLayerNumber()
            else:
                layer = 1 + max(
                    (
                        actors["axial_2d"].GetProperty().GetLayerNumber()
                        for actors in panel.roi_slice_actors.values()
                        if actors.get("axial_2d") is not None
                    ),
                    default=0,
                )

            # Store in expanded dictionary
            panel.roi_slice_actors[key] = {
                "axial_3d": ax_3d,
                "coronal_3d": cor_3d,
                "sagittal_3d": sag_3d,
                "axial_2d": ax_2d,
                "coronal_2d": cor_2d,
                "sagittal_2d": sag_2d,
            }

            # Image stacks blend in layer order, not physical depth order.
            panel._add_slice_to_stack("axial", ax_2d, layer)
            panel._add_slice_to_stack("coronal", cor_2d, layer)
            panel._add_slice_to_stack("sagittal", sag_2d, layer)

            # Get assigned color from MainWindow (or default to Red)
            assigned_color = (1.0, 0.0, 0.0)
            if panel.main_window and key in panel.main_window.roi_layers:
                assigned_color = panel.main_window.roi_layers[key].get(
                    "color", (1.0, 0.0, 0.0)
                )

            panel._sync_roi_3d_actor(key, data, affine, render=False)

            panel.set_roi_layer_color(key, assigned_color, render=False)

            # Explicitly force sagittal actor to update (fixes display issue)
            if sag_2d:
                mapper = sag_2d.GetMapper()
                if mapper:
                    mapper.Modified()
                sag_2d.Modified()

            if render:
                panel._render_all()

        except (RuntimeError, ValueError, AttributeError, KeyError) as e:
            logger.error("Error creating ROI slicer actors:", exc_info=True)
            vp_mod.QMessageBox.critical(
                panel.main_window,
                "ROI Actor Error",
                f"Could not create slicer actors for ROI:\n{e}",
            )
            if key in panel.main_window.roi_layers:
                del panel.main_window.roi_layers[key]
            if key in panel.roi_slice_actors:
                del panel.roi_slice_actors[key]

    def apply_display_correction(self, roi_key: str) -> None:
        """Applies display correction only to 2D ROI slices."""
        panel = self.panel
        if roi_key not in panel.roi_slice_actors:
            return

        roi_info = panel.main_window.roi_layers.get(roi_key)
        if not roi_info or "affine" not in roi_info:
            return

        vp_mod = _vtk_panel_module()
        roi_actors = panel.roi_slice_actors[roi_key]

        for actor_type in ("axial_2d", "coronal_2d"):
            actor_obj = roi_actors.get(actor_type)
            if actor_obj:
                vp_mod.apply_slicer_affine(
                    actor_obj, roi_info["affine"], radiological=True
                )

    def set_roi_layer_visibility(self, key: str, visible: bool) -> None:
        """Sets visibility for all 2D and 3D actors of an ROI."""
        panel = self.panel
        if key not in panel.roi_slice_actors:
            return

        vis_flag = 1 if visible else 0
        changed = False

        # Iterate over all actors stored for this key (skip non-vtkProp entries)
        for act in panel.roi_slice_actors[key].values():
            if not act or not isinstance(act, vtk.vtkProp):
                continue
            if act.GetVisibility() != vis_flag:
                act.SetVisibility(vis_flag)
                changed = True

        if changed:
            for stack in getattr(panel, "_slice_stacks_2d", {}).values():
                panel._update_stack_active_layer(stack)
            panel._render_all()

    def remove_roi_layer(self, key: str) -> None:
        """Removes a specific ROI layer's actors from all scenes."""
        panel = self.panel
        if key not in panel.roi_slice_actors:
            return

        panel._remove_actor_set(panel.roi_slice_actors[key])

        del panel.roi_slice_actors[key]
        panel._render_all()

    def clear_all_roi_layers(self) -> None:
        """Removes all ROI slice actors from all scenes."""
        panel = self.panel
        if not panel.roi_slice_actors:
            return

        for actor_dict in panel.roi_slice_actors.values():
            panel._remove_actor_set(actor_dict)

        panel.roi_slice_actors.clear()
        panel._render_all()

    def set_roi_layer_color(
        self,
        key: str,
        color: Tuple[float, float, float],
        render: bool = True,
    ) -> None:
        """Sets the color (tint) for a specific ROI layer."""
        panel = self.panel
        if key not in panel.roi_slice_actors:
            return

        r, g, b = color
        updated_mappers = set()
        for act_key, act in panel.roi_slice_actors[key].items():
            if not act:
                continue

            # Skip non-actor entries (e.g., 'sphere_source', 'rectangle_points')
            if act_key in ("sphere_source", "rectangle_points"):
                continue

            # Handle Sphere Actor (SetColor directly)
            if act_key == "sphere_3d":
                act.GetProperty().SetColor(r, g, b)
                continue

            # Handle Rectangle Actor (SetColor directly with edge color)
            if act_key == "rectangle_3d":
                prop = act.GetProperty()
                prop.SetColor(r, g, b)
                prop.SetEdgeColor(r, g, b)
                continue

            # Handle Contour Actor (SetColor directly)
            if act_key == "contour_3d":
                act.GetProperty().SetColor(r, g, b)
                continue

            # Handle Slicer Actors (Use LUT)
            prop = act.GetProperty()
            color_mapper = act.GetMapper().GetInputAlgorithm()
            value_range = (0.0, 1.0)
            if isinstance(color_mapper, vtk.vtkImageMapToColors):
                if color_mapper in updated_mappers:
                    continue
                updated_mappers.add(color_mapper)
                source_lut = color_mapper.GetLookupTable()
                if source_lut is not None:
                    value_range = source_lut.GetTableRange()
            else:
                color_window = prop.GetColorWindow()
                color_level = prop.GetColorLevel()
                value_range = (
                    color_level - color_window / 2,
                    color_level + color_window / 2,
                )

            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(256)
            lut.SetTableRange(*value_range)
            lut.Build()
            for i in range(256):
                alpha = 0.0 if i == 0 else 1.0
                lut.SetTableValue(i, r * i / 255, g * i / 255, b * i / 255, alpha)

            if isinstance(color_mapper, vtk.vtkImageMapToColors):
                color_mapper.SetOutputFormatToRGBA()
                color_mapper.SetLookupTable(lut)
                color_mapper.Update()
            else:
                prop.SetLookupTable(lut)

        if render:
            panel._render_all()
