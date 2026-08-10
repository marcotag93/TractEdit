# -*- coding: utf-8 -*-
"""Tests for efficient ROI slice and contour updates."""

from types import SimpleNamespace
from unittest.mock import Mock

import nibabel as nib
import numpy as np
from fury import actor, window
from vtk.util import numpy_support

from tractedit_pkg.visualization.drawing import DrawingManager
from tractedit_pkg.visualization.vtk_panel import VTKPanel, _crop_roi_for_contour


SLICE_ACTOR_KEYS = (
    "axial_3d",
    "coronal_3d",
    "sagittal_3d",
    "axial_2d",
    "coronal_2d",
    "sagittal_2d",
)


def _make_panel(data: np.ndarray) -> tuple[VTKPanel, str, np.ndarray]:
    key = "manual_roi_1"
    affine = np.array(
        [
            [0.5, 0.0, 0.0, -10.0],
            [0.0, 0.75, 0.0, 20.0],
            [0.0, 0.0, 1.25, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    panel = VTKPanel.__new__(VTKPanel)
    panel.scene = window.Scene()
    panel.axial_scene = window.Scene()
    panel.coronal_scene = window.Scene()
    panel.sagittal_scene = window.Scene()
    panel.current_slice_indices = {"x": 3, "y": 3, "z": 3}
    panel.roi_slice_actors = {}
    panel.sphere_params_per_roi = {}
    panel.rectangle_params_per_roi = {}
    panel.main_window = SimpleNamespace(
        anatomical_image_affine=affine,
        roi_layers={
            key: {
                "data": data,
                "affine": affine,
                "inv_affine": np.linalg.inv(affine),
                "color": (1.0, 0.0, 0.0),
            }
        },
        roi_opacities={key: 0.5},
        roi_visibility={key: True},
    )
    panel._render_all = lambda: None
    panel.set_roi_layer_color = lambda *args, **kwargs: None
    panel._apply_display_correction = lambda *args, **kwargs: None
    panel.drawing_manager = DrawingManager(panel)
    return panel, key, affine


def _source_image(actor_obj):
    algorithm = actor_obj.GetMapper().GetInputAlgorithm()
    while algorithm.GetInputAlgorithm() is not None:
        algorithm = algorithm.GetInputAlgorithm()
    return algorithm.GetOutputDataObject(0)


def test_roi_slice_actors_share_one_pipeline():
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    panel, key, affine = _make_panel(data)

    panel.add_roi_layer(key, data, affine, render=False)

    algorithms = [
        panel.roi_slice_actors[key][actor_key].GetMapper().GetInputAlgorithm()
        for actor_key in SLICE_ACTOR_KEYS
    ]
    assert all(algorithm is algorithms[0] for algorithm in algorithms[1:])


def test_initial_roi_color_updates_the_2d_slicer_lookup_table():
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    data[3, 3, 3] = 1
    panel, key, affine = _make_panel(data)
    color = (0.2, 0.4, 0.6)
    panel.main_window.roi_layers[key]["color"] = color
    panel.set_roi_layer_color = VTKPanel.set_roi_layer_color.__get__(panel)

    panel.add_roi_layer(key, data, affine, render=False)

    for actor_key in ("axial_2d", "coronal_2d", "sagittal_2d"):
        color_mapper = (
            panel.roi_slice_actors[key][actor_key].GetMapper().GetInputAlgorithm()
        )
        lookup_table = color_mapper.GetLookupTable()
        np.testing.assert_allclose(lookup_table.GetTableRange(), (0.0, 1.0))
        table_value = lookup_table.GetTableValue(255)
        np.testing.assert_allclose(table_value, (*color, 1.0))

        mapped_scalars = numpy_support.vtk_to_numpy(
            color_mapper.GetOutputDataObject(0).GetPointData().GetScalars()
        )
        assert mapped_scalars.shape[1] == 4
        assert np.any(mapped_scalars[:, 3] == 0)


def test_drawing_actor_creation_does_not_override_stored_roi_color(monkeypatch):
    key = "manual_roi_1"
    stored_color = (1.0, 0.0, 0.0)
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    affine = np.eye(4)
    interactor = Mock()
    interactor.GetEventPosition.return_value = (10, 20)
    panel = SimpleNamespace(
        main_window=SimpleNamespace(
            current_drawing_roi=key,
            roi_layers={key: {"data": data, "affine": affine, "color": stored_color}},
            roi_visibility={key: False},
        ),
        roi_slice_actors={},
        axial_interactor=interactor,
        coronal_interactor=object(),
        sagittal_interactor=object(),
        axial_scene=object(),
        drawing_preview_points=[],
        add_roi_layer=Mock(),
        set_roi_layer_color=Mock(),
    )
    picker = Mock()
    picker.GetPickPosition.return_value = (1.0, 2.0, 3.0)
    monkeypatch.setattr(
        "tractedit_pkg.visualization.drawing.vtk.vtkWorldPointPicker",
        lambda: picker,
    )
    manager = DrawingManager(panel)
    manager._update_drawing_preview = Mock()

    manager.handle_draw_on_2d(interactor)

    panel.add_roi_layer.assert_called_once_with(key, data, affine)
    panel.set_roi_layer_color.assert_not_called()
    assert panel.main_window.roi_layers[key]["color"] == stored_color


def test_initial_nonred_roi_color_is_visible_in_2d_render():
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    panel, key, affine = _make_panel(data)
    panel.main_window.roi_layers[key]["color"] = (0.0, 1.0, 0.0)
    panel.set_roi_layer_color = VTKPanel.set_roi_layer_color.__get__(panel)
    panel.add_roi_layer(key, data, affine, render=False)

    data[2:5, 2:5, 3] = 1
    panel.update_roi_layer(key, data, affine)
    panel.axial_scene.reset_camera()
    image = window.snapshot(panel.axial_scene, size=(200, 200), offscreen=True)

    green_pixels = (image[:, :, 1] > 20) & (image[:, :, 1] > image[:, :, 0])
    assert np.count_nonzero(green_pixels) > 0


def test_roi_update_reuses_slice_actors_and_updates_source_scalars():
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    panel, key, affine = _make_panel(data)
    panel.add_roi_layer(key, data, affine, render=False)
    original_actors = {
        actor_key: panel.roi_slice_actors[key][actor_key]
        for actor_key in SLICE_ACTOR_KEYS
    }

    data[2, 3, 4] = 1
    panel.update_roi_layer(key, data, affine)

    for actor_key, actor_obj in original_actors.items():
        assert panel.roi_slice_actors[key][actor_key] is actor_obj

    source = _source_image(original_actors["axial_3d"])
    flat = numpy_support.vtk_to_numpy(source.GetPointData().GetScalars())
    restored = np.swapaxes(flat.reshape(9, 8, 7), 0, 2)
    np.testing.assert_array_equal(restored, data)
    np.testing.assert_array_equal(original_actors["axial_3d"].resliced_array(), data)
    assert original_actors["axial_3d"]._roi_scalar_buffer.flags.c_contiguous


def test_nonbinary_roi_update_rebuilds_value_range_pipeline():
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    data[2, 3, 4] = 3
    panel, key, affine = _make_panel(data)
    panel.add_roi_layer(key, data, affine, render=False)
    original_actor = panel.roi_slice_actors[key]["axial_3d"]

    data[2, 3, 4] = 4
    panel.update_roi_layer(key, data, affine)

    assert panel.roi_slice_actors[key]["axial_3d"] is not original_actor


def test_empty_roi_skips_contour_creation(monkeypatch):
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    panel, key, affine = _make_panel(data)

    def fail_contour(*args, **kwargs):
        raise AssertionError("empty ROI should not create a contour")

    monkeypatch.setattr(
        "tractedit_pkg.visualization.vtk_panel.actor.contour_from_roi",
        fail_contour,
    )

    panel.add_roi_layer(key, data, affine, render=False)

    assert "contour_3d" not in panel.roi_slice_actors[key]


def test_contour_crop_preserves_voxel_world_coordinates(monkeypatch):
    data = np.zeros((20, 21, 22), dtype=np.uint8)
    data[8:11, 9:13, 10:15] = 1
    panel, key, affine = _make_panel(data)
    captured = {}

    def capture_contour(cropped_data, *, affine, color, opacity):
        captured["data"] = cropped_data.copy()
        captured["affine"] = affine.copy()
        return None

    monkeypatch.setattr(
        "tractedit_pkg.visualization.vtk_panel.actor.contour_from_roi",
        capture_contour,
    )

    panel.add_roi_layer(key, data, affine, render=False)

    assert captured["data"].shape == (5, 6, 7)
    original_world = affine @ np.array([8.0, 9.0, 10.0, 1.0])
    cropped_world = captured["affine"] @ np.array([1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(cropped_world, original_world)


def test_cropped_contour_matches_full_volume_geometry():
    data = np.zeros((20, 21, 22), dtype=np.uint8)
    data[8:11, 9:13, 10:15] = 1
    _, _, affine = _make_panel(data)
    cropped_data, cropped_affine = _crop_roi_for_contour(data, affine)

    full_actor = actor.contour_from_roi(data, affine=affine)
    cropped_actor = actor.contour_from_roi(cropped_data, affine=cropped_affine)
    full_actor.GetMapper().Update()
    cropped_actor.GetMapper().Update()

    np.testing.assert_allclose(cropped_actor.GetBounds(), full_actor.GetBounds())
    assert (
        cropped_actor.GetMapper().GetInput().GetNumberOfPoints()
        == full_actor.GetMapper().GetInput().GetNumberOfPoints()
    )


def test_freehand_drawing_replaces_mode_specific_3d_actor_with_contour():
    data = np.zeros((20, 21, 22), dtype=np.uint8)
    panel, key, affine = _make_panel(data)
    panel.sphere_params_per_roi[key] = {
        "center": np.array([1.0, 2.0, 3.0]),
        "radius": 2.0,
        "view_type": "axial",
    }
    panel.add_roi_layer(key, data, affine, render=False)
    panel.main_window.current_drawing_roi = key
    panel.main_window.auto_fill_voxels = False
    panel.main_window.draw_brush_size = 1
    panel.drawing_preview_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    panel.current_drawing_view_type = "axial"
    panel.preview_line_actor = None
    panel.is_sphere_mode = False
    panel.is_rectangle_mode = False
    panel.is_eraser_mode = False
    panel.update_status = lambda *args: None
    panel.drawing_manager._world_to_voxel_points = Mock(
        return_value=np.array([[5.0, 10.0, 11.0], [14.0, 10.0, 11.0]])
    )

    panel.drawing_manager.finish_drawing()

    assert key not in panel.sphere_params_per_roi
    assert "sphere_3d" not in panel.roi_slice_actors[key]
    contour_actor = panel.roi_slice_actors[key]["contour_3d"]
    contour_actor.GetMapper().Update()
    assert contour_actor.GetMapper().GetInput().GetNumberOfPoints() > 0


def test_sphere_update_reuses_slice_and_sphere_pipelines():
    data = np.zeros((20, 21, 22), dtype=np.uint8)
    panel, key, affine = _make_panel(data)
    panel.sphere_params_per_roi[key] = {
        "center": np.array([1.0, 2.0, 3.0]),
        "radius": 2.0,
        "view_type": "axial",
    }
    panel.add_roi_layer(key, data, affine, render=False)
    slice_actors = {
        actor_key: panel.roi_slice_actors[key][actor_key]
        for actor_key in SLICE_ACTOR_KEYS
    }
    sphere_source = panel.roi_slice_actors[key]["sphere_source"]

    panel.sphere_params_per_roi[key]["center"] = np.array([4.0, 5.0, 6.0])
    panel.sphere_params_per_roi[key]["radius"] = 3.5
    panel.update_roi_layer(key, data, affine)

    for actor_key, actor_obj in slice_actors.items():
        assert panel.roi_slice_actors[key][actor_key] is actor_obj
    assert panel.roi_slice_actors[key]["sphere_source"] is sphere_source
    assert sphere_source.GetCenter() == (4.0, 5.0, 6.0)
    assert sphere_source.GetRadius() == 3.5


def test_rectangle_update_reuses_slice_and_rectangle_pipelines():
    data = np.zeros((20, 21, 22), dtype=np.uint8)
    panel, key, affine = _make_panel(data)
    panel.rectangle_params_per_roi[key] = {
        "start": np.array([1.0, 2.0, 3.0]),
        "end": np.array([4.0, 5.0, 3.0]),
        "view_type": "axial",
    }
    panel.add_roi_layer(key, data, affine, render=False)
    slice_actors = {
        actor_key: panel.roi_slice_actors[key][actor_key]
        for actor_key in SLICE_ACTOR_KEYS
    }
    rectangle_actor = panel.roi_slice_actors[key]["rectangle_3d"]
    rectangle_points = panel.roi_slice_actors[key]["rectangle_points"]

    panel.rectangle_params_per_roi[key]["start"] = np.array([2.0, 3.0, 4.0])
    panel.rectangle_params_per_roi[key]["end"] = np.array([6.0, 8.0, 4.0])
    panel.update_roi_layer(key, data, affine)

    for actor_key, actor_obj in slice_actors.items():
        assert panel.roi_slice_actors[key][actor_key] is actor_obj
    assert panel.roi_slice_actors[key]["rectangle_3d"] is rectangle_actor
    assert panel.roi_slice_actors[key]["rectangle_points"] is rectangle_points
    assert rectangle_points.GetPoint(0) == (2.0, 3.0, 4.0)
    assert rectangle_points.GetPoint(2) == (6.0, 8.0, 4.0)


def test_sphere_rasterization_uses_physical_radius_on_anisotropic_grid():
    data = np.zeros((25, 25, 25), dtype=np.uint8)
    panel, key, affine = _make_panel(data)
    center_voxel = np.array([10.0, 10.0, 10.0])
    center_world = (affine @ np.append(center_voxel, 1.0))[:3]

    panel.drawing_manager._rasterize_sphere_at_position(
        key,
        data,
        center_world,
        2.0,
        data.shape,
        "axial",
        1,
    )

    assert data[6, 10, 10] == 1
    assert data[14, 10, 10] == 1
    assert data[10, 10, 9] == 1
    assert data[10, 10, 11] == 1
    assert data[10, 10, 8] == 0
    assert data[10, 10, 12] == 0

    selected_voxels = np.argwhere(data > 0)
    selected_world = nib.affines.apply_affine(affine, selected_voxels)
    distances = np.linalg.norm(selected_world - center_world, axis=1)
    assert np.all(distances <= 2.0 + 1e-12)
