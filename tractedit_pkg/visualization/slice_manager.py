# -*- coding: utf-8 -*-

"""
Anatomical slice and 2D navigation manager for TractEdit visualization.

Declares the explicit axis/orientation/radiological-flip specification for the
three orthogonal 2D views and manages anatomical slice actors, slice indices,
RAS-to-voxel navigation, and slow slice UI updates.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import vtk

from ..render_batch import batch_render
from . import coordinates

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrthoViewSpec:
    """Explicit plane orientation, radiological mirror, and camera spec."""

    name: str
    axis: str
    slice_axis: int
    view_sign: int
    view_up_axis: int
    radiological: bool


ORTHO_VIEW_SPECS: Tuple[OrthoViewSpec, ...] = (
    OrthoViewSpec(
        name="axial",
        axis="z",
        slice_axis=2,
        view_sign=1,
        view_up_axis=1,
        radiological=True,
    ),
    OrthoViewSpec(
        name="coronal",
        axis="y",
        slice_axis=1,
        view_sign=-1,
        view_up_axis=2,
        radiological=True,
    ),
    OrthoViewSpec(
        name="sagittal",
        axis="x",
        slice_axis=0,
        view_sign=1,
        view_up_axis=2,
        radiological=False,
    ),
)


def _vtk_panel_module():
    from . import vtk_panel

    return vtk_panel


def _slice_extent_for_axis(
    x_ext: Tuple[int, int],
    y_ext: Tuple[int, int],
    z_ext: Tuple[int, int],
    slice_axis: int,
    slice_index: int,
) -> Tuple[int, int, int, int, int, int]:
    """Build a 6-element display extent fixing one voxel axis to slice_index."""
    extent = [x_ext[0], x_ext[1], y_ext[0], y_ext[1], z_ext[0], z_ext[1]]
    extent[2 * slice_axis] = slice_index
    extent[2 * slice_axis + 1] = slice_index
    return tuple(extent)


# ============================================================================
# Slice Manager Class
# ============================================================================


class SliceManager:
    """Manages anatomical slice actors and 2D/3D slice navigation."""

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        self.panel = vtk_panel

    def set_anatomical_slice_visibility(self, visible: bool) -> None:
        """Sets visibility for anatomical slices and associated crosshairs."""
        panel = self.panel
        vis_flag = 1 if visible else 0

        for suffix in ("_slice_actor", "_slice_actor_2d", "_crosshair_actor"):
            for spec in ORTHO_VIEW_SPECS:
                act = getattr(panel, f"{spec.name}{suffix}", None)
                if act:
                    act.SetVisibility(vis_flag)

        for stack in getattr(panel, "_slice_stacks_2d", {}).values():
            panel._update_stack_active_layer(stack)
        panel._render_all()

    def initialize_3d_camera(self) -> None:
        """Position the 3D camera to frame the anatomical image properly.

        Lets VTK compute optimal centering from all slice actor bounds,
        then overrides the viewing angle to an anterior-facing coronal view.
        """
        panel = self.panel
        if not panel.scene or not panel.coronal_slice_actor:
            return

        # Let VTK center on the visible geometry (all 3 slice actors)
        panel.scene.reset_camera()

        cam = panel.scene.GetActiveCamera()
        fp = cam.GetFocalPoint()

        # Compute camera distance from the image extent
        bounds = panel.coronal_slice_actor.GetBounds()
        extent_x = abs(bounds[1] - bounds[0])
        extent_z = abs(bounds[5] - bounds[4])
        max_extent = max(extent_x, extent_z)

        affine = panel.main_window.anatomical_image_affine
        direction, view_up = coordinates.camera_frame_from_affine(
            affine, slice_axis=1, view_sign=1, view_up_axis=2
        )

        # Anterior-facing coronal view relative to the displayed image axes.
        cam.SetPosition(*(np.asarray(fp) + direction * max_extent * 2.0))
        cam.SetViewUp(*view_up)

        panel.scene.reset_clipping_range()

    @batch_render(lambda self: self.panel)
    def update_anatomical_slices(self, preserve_view: bool = False) -> None:
        """Creates or updates the FURY slicer actors for anatomical image."""
        panel = self.panel
        vp_mod = _vtk_panel_module()
        if not panel.scene:
            return
        if not panel.main_window or panel.main_window.anatomical_image_data is None:
            panel.clear_anatomical_slices()
            return

        image_data = panel.main_window.anatomical_image_data
        affine = panel.main_window.anatomical_image_affine

        # Basic checks
        if image_data is None or affine is None or image_data.ndim < 3:
            logger.error("Invalid image data or affine provided for slices.")
            panel.clear_anatomical_slices()
            return

        # Ensure data is contiguous float32
        if not image_data.flags["C_CONTIGUOUS"]:
            image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        else:
            image_data = image_data.astype(np.float32, copy=False)

        # Get dimensions and extents
        shape = image_data.shape[:3]

        panel.image_shape_vox = shape
        panel.image_extents["x"] = (0, shape[0] - 1)
        panel.image_extents["y"] = (0, shape[1] - 1)
        panel.image_extents["z"] = (0, shape[2] - 1)

        # Store full-resolution shape from mmap image if available
        mmap_img = getattr(panel.main_window, "anatomical_mmap_image", None)
        if mmap_img is not None:
            mmap_shape = mmap_img.shape
            panel.mmap_image_shape = mmap_shape
            panel.mmap_image_extents["x"] = (0, mmap_shape[0] - 1)
            panel.mmap_image_extents["y"] = (0, mmap_shape[1] - 1)
            panel.mmap_image_extents["z"] = (0, mmap_shape[2] - 1)
            logger.info(
                f"Mmap image available: full-res shape {mmap_shape}, "
                f"display shape {shape}"
            )
        else:
            panel.mmap_image_shape = None
            panel.mmap_image_extents = {"x": None, "y": None, "z": None}

        # Initialize or keep current slice indices
        for axis_idx, axis in enumerate(("x", "y", "z")):
            if (
                panel.current_slice_indices[axis] is None
                or panel.current_slice_indices[axis] > panel.image_extents[axis][1]
            ):
                panel.current_slice_indices[axis] = shape[axis_idx] // 2

        current_x = panel.current_slice_indices["x"]
        current_y = panel.current_slice_indices["y"]
        current_z = panel.current_slice_indices["z"]
        x_extent = panel.image_extents["x"]
        y_extent = panel.image_extents["y"]
        z_extent = panel.image_extents["z"]

        # Clear existing slicer actors
        panel.clear_anatomical_slices(reset_state=False)

        try:
            if image_data.ndim == 4 and image_data.shape[-1] == 3:
                value_range = (0.0, 255.0)
            else:
                # Determine Value Range using percentile-based windowing
                #   min = 5th percentile
                #   max = 99th percentile, then expanded by 20%
                finite_data = image_data[np.isfinite(image_data)]
                if finite_data.size > 0:
                    img_min = float(np.percentile(finite_data, 5))
                    img_max = float(np.percentile(finite_data, 99))
                    img_max = img_max + (img_max - img_min) / 5.0
                else:
                    img_min, img_max = 0.0, 1.0

                # Ensure range has distinct min/max
                if img_max <= img_min:
                    value_range = (img_min - 1.0, img_max + 1.0)
                else:
                    value_range = (float(img_min), float(img_max))

            # Get opacity from MainWindow state
            slicer_opacity = getattr(panel.main_window, "image_opacity", 1.0)
            interpolation_mode = "nearest"

            # One native-grid pipeline is shared by all anatomical views.
            panel.axial_slice_actor = vp_mod.create_slicer_actor(
                image_data,
                affine,
                value_range,
                slicer_opacity,
                interpolation_mode,
            )
            if panel.axial_slice_actor is None:
                raise RuntimeError("Could not create anatomical slicer actor.")
            panel.coronal_slice_actor = panel.axial_slice_actor.copy()
            panel.sagittal_slice_actor = panel.axial_slice_actor.copy()
            panel.axial_slice_actor_2d = panel.axial_slice_actor.copy()
            panel.coronal_slice_actor_2d = panel.axial_slice_actor.copy()
            panel.sagittal_slice_actor_2d = panel.axial_slice_actor.copy()

            for spec in ORTHO_VIEW_SPECS:
                actor_3d = getattr(panel, f"{spec.name}_slice_actor")
                actor_2d = getattr(panel, f"{spec.name}_slice_actor_2d")
                vp_mod.apply_slicer_affine(actor_3d, affine)
                vp_mod.apply_slicer_affine(
                    actor_2d, affine, radiological=spec.radiological
                )
                actor_2d.GetProperty().SetOpacity(1.0)

            for spec in ORTHO_VIEW_SPECS:
                actor_3d = getattr(panel, f"{spec.name}_slice_actor")
                actor_2d = getattr(panel, f"{spec.name}_slice_actor_2d")
                extent = _slice_extent_for_axis(
                    x_extent,
                    y_extent,
                    z_extent,
                    spec.slice_axis,
                    panel.current_slice_indices[spec.axis],
                )
                actor_3d.display_extent(*extent)
                panel.scene.add(actor_3d)
                actor_2d.display_extent(*extent)
                if getattr(panel, f"{spec.name}_scene", None):
                    panel._add_slice_to_stack(spec.name, actor_2d, 0)

            # Create crosshair actors
            panel._create_or_update_crosshairs()

        except TypeError as te:
            error_msg = f"TypeError during slice actor creation/display: {te}"
            logger.error(error_msg, exc_info=True)
            if preserve_view:
                panel.clear_anatomical_slices(reset_state=False)
                raise
            vp_mod.QMessageBox.critical(
                panel.main_window, "Slice Actor TypeError", error_msg
            )
            panel.clear_anatomical_slices()
        except (RuntimeError, ValueError, AttributeError) as e:
            error_msg = f"Error during anatomical slice actor creation/addition: {e}"
            logger.error(error_msg, exc_info=True)
            if preserve_view:
                panel.clear_anatomical_slices(reset_state=False)
                raise
            vp_mod.QMessageBox.critical(
                panel.main_window, "Slice Actor Error", error_msg
            )
            panel.clear_anatomical_slices()

        if not preserve_view:
            panel._setup_ortho_cameras()
            panel._initialize_3d_camera()

        # Initialize scale bars for 2D views
        try:
            panel.scale_bar_manager.initialize(
                axial_renderer=panel.axial_overlay_renderer,
                coronal_renderer=panel.coronal_overlay_renderer,
                sagittal_renderer=panel.sagittal_overlay_renderer,
                axial_camera=panel.axial_scene.GetActiveCamera(),
                coronal_camera=panel.coronal_scene.GetActiveCamera(),
                sagittal_camera=panel.sagittal_scene.GetActiveCamera(),
            )
            panel.scale_bar_manager.update_all()
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to initialize scale bars: {e}")

        # Final render
        panel._render_all()
        panel.update_status(f"Slice View: X={current_x} Y={current_y} Z={current_z}")

    def clear_anatomical_slices(self, reset_state: bool = True) -> None:
        """Removes anatomical slice actors from all active scenes and resets slice state."""
        panel = self.panel
        panel._clear_crosshairs()

        actors_to_remove = [
            act
            for suffix in ("_slice_actor", "_slice_actor_2d")
            for spec in ORTHO_VIEW_SPECS
            if (act := getattr(panel, f"{spec.name}{suffix}", None)) is not None
        ]

        scenes_to_check = [
            s
            for s in (
                panel.scene,
                *(getattr(panel, f"{spec.name}_scene", None) for spec in ORTHO_VIEW_SPECS),
            )
            if s is not None
        ]

        scene_changed = False

        for slice_actor in actors_to_remove:
            panel._remove_slice_from_stacks(slice_actor)
            for scene in scenes_to_check:
                try:
                    scene.rm(slice_actor)
                    scene_changed = True
                except (ValueError, AttributeError):
                    pass
                except RuntimeError as e:
                    logger.error(f"Error removing slice actor: {e}", exc_info=True)

        # Reset actor references
        for suffix in ("_slice_actor", "_slice_actor_2d"):
            for spec in ORTHO_VIEW_SPECS:
                setattr(panel, f"{spec.name}{suffix}", None)

        # Clear scale bars
        panel.scale_bar_manager.clear()

        if reset_state:
            panel.current_slice_indices = {"x": None, "y": None, "z": None}
            panel.image_shape_vox = None
            panel.image_extents = {"x": None, "y": None, "z": None}
            panel.mmap_image_shape = None
            panel.mmap_image_extents = {"x": None, "y": None, "z": None}

        if scene_changed:
            panel._render_all()

    def set_anatomical_opacity(self, value: float) -> None:
        """Sets the opacity of the anatomical slices."""
        panel = self.panel
        for spec in ORTHO_VIEW_SPECS:
            act = getattr(panel, f"{spec.name}_slice_actor", None)
            if act:
                act.GetProperty().SetOpacity(value)
        panel.render_window.Render()

    @batch_render(lambda self: self.panel)
    def move_slice(self, axis: str, direction: int) -> None:
        """Moves the specified slice ('x', 'y', 'z') by 'direction' (+1 or -1)."""
        panel = self.panel
        if (
            not panel.image_shape_vox
            or axis not in panel.current_slice_indices
            or panel.current_slice_indices[axis] is None
        ):
            return

        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_index = axis_map[axis]

        current_index = panel.current_slice_indices[axis]
        max_index = panel.image_shape_vox[axis_index] - 1

        new_index = max(0, min(current_index + direction, max_index))

        if new_index == current_index:
            panel.update_status(
                f"Slice View: Reached {axis.upper()} limit ({new_index})."
            )
            return

        panel.set_slice_indices(**{axis: new_index})
        panel._update_slow_slice_components()

    def set_slice_indices(
        self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None
    ) -> None:
        """Sets slice indices and updates anatomical slices, ROIs, and crosshairs."""
        panel = self.panel
        if (
            not panel.image_shape_vox
            or not panel.axial_slice_actor
            or not panel.coronal_slice_actor
            or not panel.sagittal_slice_actor
        ):
            return

        current = panel.current_slice_indices

        new_x = x if x is not None else current["x"]
        new_y = y if y is not None else current["y"]
        new_z = z if z is not None else current["z"]

        new_x = max(panel.image_extents["x"][0], min(new_x, panel.image_extents["x"][1]))
        new_y = max(panel.image_extents["y"][0], min(new_y, panel.image_extents["y"][1]))
        new_z = max(panel.image_extents["z"][0], min(new_z, panel.image_extents["z"][1]))

        if new_x == current["x"] and new_y == current["y"] and new_z == current["z"]:
            return

        panel.current_slice_indices = {"x": new_x, "y": new_y, "z": new_z}

        try:
            x_ext = panel.image_extents["x"]
            y_ext = panel.image_extents["y"]
            z_ext = panel.image_extents["z"]
            new_vals = {"x": new_x, "y": new_y, "z": new_z}
            slice_moved = {"x": False, "y": False, "z": False}

            for spec in reversed(ORTHO_VIEW_SPECS):
                val = new_vals[spec.axis]
                if val != current[spec.axis]:
                    extent = _slice_extent_for_axis(
                        x_ext, y_ext, z_ext, spec.slice_axis, val
                    )
                    getattr(panel, f"{spec.name}_slice_actor").display_extent(*extent)
                    act_2d = getattr(panel, f"{spec.name}_slice_actor_2d", None)
                    if act_2d:
                        act_2d.display_extent(*extent)
                    slice_moved[spec.axis] = True

            # Update all ROI slicers
            panel.roi_scene_manager.update_slice_extents(
                slice_moved, new_x, new_y, new_z
            )

            panel._create_or_update_crosshairs()

            # Update 2D cameras for moved slices
            if slice_moved["x"]:
                panel._update_sagittal_camera()
            if slice_moved["y"]:
                panel._update_coronal_camera()
            if slice_moved["z"]:
                panel._update_axial_camera()

            panel._render_all()

        except (RuntimeError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"Error in set_slice_indices: {e}", exc_info=True)

    def update_slow_slice_components(self, *, render: bool = True) -> None:
        """Updates status text and RAS coordinates after a drag-navigate finishes."""
        panel = self.panel

        def _set_ras_display(coords):
            scalar_mgr = getattr(panel.main_window, "scalar_manager", None)
            if scalar_mgr is not None:
                scalar_mgr.update_ras_coordinate_display(coords)
            elif hasattr(panel.main_window, "update_ras_coordinate_display"):
                panel.main_window.update_ras_coordinate_display(coords)

        if (
            panel.main_window is None
            or any(value is None for value in panel.current_slice_indices.values())
            or not panel.image_extents["x"]
        ):
            if panel.main_window:
                _set_ras_display(None)
            return

        c = panel.current_slice_indices

        status_msg = (
            f"Slice View: X={c['x']}/{panel.image_extents['x'][1]}  "
            f"Y={c['y']}/{panel.image_extents['y'][1]}  "
            f"Z={c['z']}/{panel.image_extents['z'][1]}"
        )
        panel.update_status(status_msg, render=render)

        try:
            main_affine = panel.main_window.anatomical_image_affine
            if main_affine is not None:
                world_coord = panel._voxel_to_world(
                    [c["x"], c["y"], c["z"]], main_affine
                )
                _set_ras_display(world_coord)
            else:
                _set_ras_display(None)
        except (ValueError, IndexError, np.linalg.LinAlgError, AttributeError) as e:
            logger.error(f"Error updating RAS coordinate display: {e}")
            _set_ras_display(None)

    def _ras_to_clamped_voxel(self, ras_coords) -> Tuple[int, int, int]:
        """Convert RAS world coordinates to integer voxel indices clamped to extents."""
        panel = self.panel
        inv_affine = np.linalg.inv(panel.main_window.anatomical_image_affine)
        voxel_pos_float = panel._world_to_voxel(ras_coords, inv_affine)
        clamped = [
            max(
                panel.image_extents[axis][0],
                min(int(round(voxel_pos_float[idx])), panel.image_extents[axis][1]),
            )
            for idx, axis in enumerate(("x", "y", "z"))
        ]
        return clamped[0], clamped[1], clamped[2]

    @batch_render(lambda self: self.panel)
    def set_slices_from_ras(self, ras_coords: np.ndarray) -> None:
        """Moves the slice crosshairs to the specified RAS coordinate."""
        panel = self.panel
        if (
            panel.main_window is None
            or panel.main_window.anatomical_image_affine is None
            or not all(panel.image_extents.values())
            or panel.image_extents["x"] is None
        ):
            panel.update_status(
                "Error: Cannot set slices from RAS, no anatomical image loaded."
            )
            return

        try:
            new_x, new_y, new_z = self._ras_to_clamped_voxel(ras_coords)
            panel.set_slice_indices(x=new_x, y=new_y, z=new_z)
            panel._update_slow_slice_components()
        except (ValueError, IndexError, np.linalg.LinAlgError, TypeError) as e:
            panel.update_status(f"Error setting slice from RAS: {e}")
            logger.error("Error setting slice from RAS:", exc_info=True)

    def navigate_2d_view(self, obj: vtk.vtkObject, event_id: str) -> None:
        """Picks the 3D coordinate from a 2D view and updates the slice indices."""
        panel = self.panel
        if (
            panel.main_window is None
            or panel.main_window.anatomical_image_data is None
            or None in panel.current_slice_indices.values()
        ):
            panel.is_navigating_2d = False  # avoid freeze
            return

        interactor = obj
        if not interactor:
            return
        display_pos = interactor.GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
        if not renderer:
            return

        picker.Pick(display_pos[0], display_pos[1], 0, renderer)

        if picker.GetCellId() < 0:
            return

        world_pos = list(picker.GetPickPosition())

        matched_spec = next(
            (
                spec
                for spec in ORTHO_VIEW_SPECS
                if interactor == getattr(panel, f"{spec.name}_interactor", None)
            ),
            None,
        )
        # Reverse the X mirror composed into the 2D radiological actor matrix.
        if matched_spec is not None and matched_spec.radiological:
            world_pos[0] = -world_pos[0]

        try:
            new_x, new_y, new_z = self._ras_to_clamped_voxel(world_pos)
            target = {"x": new_x, "y": new_y, "z": new_z}
            if matched_spec is not None:
                target[matched_spec.axis] = panel.current_slice_indices[matched_spec.axis]
            panel.set_slice_indices(**target)
        except (ValueError, IndexError, np.linalg.LinAlgError, RuntimeError) as e:
            logger.error("Error during 2D window click navigation:", exc_info=True)
            panel.update_status("Error navigating with click.")
