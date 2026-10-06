"""Spatial contracts shared by ROI display, drawing, and filtering."""

from types import SimpleNamespace
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from tractedit_pkg._numba_aot._parallel_wrappers import (
    batch_check_sphere_intersection,
)
from tractedit_pkg.logic.roi_manager import (
    ROIManager,
)
from tractedit_pkg.logic.state_manager import ActionType, StateManager
from tractedit_pkg.visualization.drawing import (
    DrawingManager,
)
from tractedit_pkg.visualization.selection import SelectionManager
from tests.test_roi_rendering import _make_panel, _vtk_matrix_to_numpy


OBLIQUE_AFFINE = np.array(
    [
        [1.5, 0.3, 0.1, -12.0],
        [0.2, 0.9, -0.2, 8.0],
        [0.0, 0.1, 2.0, 20.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _set_panel_affine(panel, key, affine):
    panel.main_window.anatomical_image_affine = affine
    panel.main_window.roi_layers[key]["affine"] = affine
    panel.main_window.roi_layers[key]["inv_affine"] = np.linalg.inv(affine)


@pytest.mark.parametrize("slice_index", [0, 3, 6])
@pytest.mark.parametrize(
    "affine", [np.eye(4), OBLIQUE_AFFINE], ids=["identity", "oblique"]
)
def test_sagittal_roi_plane_matches_anatomical_plane(slice_index, affine):
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    panel, key, _ = _make_panel(data)
    _set_panel_affine(panel, key, affine)
    panel.current_slice_indices["x"] = slice_index

    panel.add_roi_layer(key, data, affine, render=False)

    for actor_key in ("sagittal_3d", "sagittal_2d"):
        extent = panel.roi_slice_actors[key][actor_key].GetDisplayExtent()
        assert extent[0:2] == (slice_index, slice_index)


def test_sagittal_roi_plane_tracks_slice_navigation():
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    panel, key, affine = _make_panel(data)
    panel.main_window.roi_layers[key]["T_main_to_roi"] = np.eye(4)
    panel.add_roi_layer(key, data, affine, render=False)
    panel.image_shape_vox = data.shape
    panel.image_extents = {
        "x": (0, data.shape[0] - 1),
        "y": (0, data.shape[1] - 1),
        "z": (0, data.shape[2] - 1),
    }
    panel.axial_slice_actor = Mock()
    panel.coronal_slice_actor = Mock()
    panel.sagittal_slice_actor = Mock()
    panel.axial_slice_actor_2d = Mock()
    panel.coronal_slice_actor_2d = Mock()
    panel.sagittal_slice_actor_2d = Mock()
    panel._create_or_update_crosshairs = Mock()
    panel._update_sagittal_camera = Mock()
    panel._update_coronal_camera = Mock()
    panel._update_axial_camera = Mock()

    for slice_index in (0, 6, 3):
        panel.set_slice_indices(x=slice_index)
        extent = panel.roi_slice_actors[key]["sagittal_2d"].GetDisplayExtent()
        assert extent[0:2] == (slice_index, slice_index)


@pytest.mark.parametrize("view_type", ["axial", "coronal", "sagittal"])
def test_drawing_landmark_roundtrips_through_display_actor(view_type):
    data = np.zeros((7, 8, 9), dtype=np.uint8)
    panel, key, _ = _make_panel(data)
    _set_panel_affine(panel, key, OBLIQUE_AFFINE)
    panel.add_roi_layer(key, data, OBLIQUE_AFFINE, render=False)
    voxel = np.array([3.0, 4.0, 5.0, 1.0])
    actor = panel.roi_slice_actors[key][f"{view_type}_2d"]
    display_point = (_vtk_matrix_to_numpy(actor.GetMatrix()) @ voxel)[:3]
    panel.drawing_preview_points = [display_point]

    actual = panel.drawing_manager._world_to_voxel_points(
        key,
        np.linalg.inv(OBLIQUE_AFFINE),
        data.shape,
        view_type,
    )

    np.testing.assert_allclose(actual[0], voxel[:3], atol=1e-12)




RECTANGLE_POINTS = {
    "axial": np.array([[2.0, 3.0, 4.0], [5.0, 7.0, 4.0]]),
    "coronal": np.array([[2.0, 4.0, 3.0], [5.0, 4.0, 7.0]]),
    "sagittal": np.array([[4.0, 2.0, 3.0], [4.0, 6.0, 7.0]]),
}
NORMAL_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


@pytest.mark.parametrize("view_type", ["axial", "coronal", "sagittal"])
def test_rectangle_is_one_voxel_plane_with_affine_world_corners(view_type):
    data = np.zeros((10, 11, 12), dtype=np.uint8)
    panel, key, _ = _make_panel(data)
    _set_panel_affine(panel, key, OBLIQUE_AFFINE)
    voxel_points = RECTANGLE_POINTS[view_type]
    panel.drawing_preview_points = nib.affines.apply_affine(
        OBLIQUE_AFFINE, voxel_points
    )

    changed = panel.drawing_manager._rasterize_rectangle(
        key, data, voxel_points, data.shape, view_type
    )

    assert changed
    selected = np.argwhere(data > 0)
    axis = NORMAL_AXIS[view_type]
    assert np.unique(selected[:, axis]).tolist() == [int(voxel_points[0, axis])]
    params = panel.rectangle_params_per_roi[key]
    np.testing.assert_array_equal(params["voxel_min"], voxel_points.min(axis=0))
    np.testing.assert_array_equal(params["voxel_max"], voxel_points.max(axis=0))
    corners_voxel = nib.affines.apply_affine(
        np.linalg.inv(OBLIQUE_AFFINE), params["corners"]
    )
    assert np.allclose(corners_voxel[:, axis], voxel_points[0, axis])


def test_oblique_rectangle_actor_uses_rasterized_world_corners():
    data = np.zeros((10, 11, 12), dtype=np.uint8)
    panel, key, _ = _make_panel(data)
    _set_panel_affine(panel, key, OBLIQUE_AFFINE)
    voxel_points = RECTANGLE_POINTS["axial"]
    panel.drawing_preview_points = nib.affines.apply_affine(
        OBLIQUE_AFFINE, voxel_points
    )
    panel.drawing_manager._rasterize_rectangle(
        key, data, voxel_points, data.shape, "axial"
    )

    panel.add_roi_layer(key, data, OBLIQUE_AFFINE, render=False)

    params = panel.rectangle_params_per_roi[key]
    points = panel.roi_slice_actors[key]["rectangle_points"]
    actual = np.array([points.GetPoint(index) for index in range(4)])
    np.testing.assert_allclose(actual, params["corners"], atol=1e-6)


def test_sparse_segment_intersects_oblique_rectangle_prism():
    voxel_line = np.array([[-1.0, 4.0, 4.0], [9.0, 4.0, 4.0]])
    world_line = nib.affines.apply_affine(OBLIQUE_AFFINE, voxel_line)
    window = SimpleNamespace(
        tractogram_data=[world_line],
        streamline_bboxes=np.array([[world_line.min(axis=0), world_line.max(axis=0)]]),
        visible_indices={0},
        _visibility_version=0,
    )
    manager = SelectionManager(SimpleNamespace(main_window=window))

    result = manager.find_streamlines_in_oriented_box(
        OBLIQUE_AFFINE,
        np.array([4, 3, 4]),
        np.array([4, 6, 4]),
        check_all=True,
    )

    assert result == {0}


def test_analytic_sphere_tangency_keeps_open_boundary_contract():
    points = np.array([[-2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    result = batch_check_sphere_intersection(
        points,
        np.array([0, 2], dtype=np.int64),
        np.zeros(3),
        1.0,
    )

    assert not result[0]


@pytest.mark.parametrize("query_kind", ["sphere", "rectangle"])
def test_roi_recomputation_preserves_analytic_query_kind(query_kind):
    line = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    panel = SimpleNamespace(
        sphere_params_per_roi={},
        rectangle_params_per_roi={},
        _find_streamlines_in_radius=Mock(return_value={0}),
        _find_streamlines_in_oriented_box=Mock(return_value={0}),
        update_status=Mock(),
        update_progress_bar=Mock(),
    )
    if query_kind == "sphere":
        panel.sphere_params_per_roi["roi"] = {
            "center": np.array([4.0, 0.0, 0.0]),
            "radius": 1.0,
        }
    else:
        panel.rectangle_params_per_roi["roi"] = {
            "voxel_min": np.array([4, 0, 0]),
            "voxel_max": np.array([4, 0, 0]),
        }
    window = SimpleNamespace(
        tractogram_data=[line],
        streamline_bboxes=np.array([[line.min(axis=0), line.max(axis=0)]]),
        roi_layers={
            "roi": {
                "data": np.zeros((9, 2, 2), dtype=np.uint8),
                "affine": np.eye(4),
                "inv_affine": np.eye(4),
            }
        },
        roi_intersection_cache={},
        vtk_panel=panel,
    )

    assert ROIManager(window).compute_roi_intersection("roi")

    assert window.roi_intersection_cache["roi"] == {0}
    if query_kind == "sphere":
        panel._find_streamlines_in_radius.assert_called_once()
    else:
        panel._find_streamlines_in_oriented_box.assert_called_once()


def test_analytic_query_kind_survives_undo_and_redo():
    lines = [
        np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        np.array([[0.0, 4.0, 0.0], [8.0, 4.0, 0.0]]),
    ]
    sphere_params = {"center": np.array([4.0, 0.0, 0.0]), "radius": 1.0}
    rectangle_params = {
        "start": np.array([4.0, 3.0, 0.0]),
        "end": np.array([4.0, 5.0, 0.0]),
        "view_type": "sagittal",
        "voxel_min": np.array([4, 3, 0]),
        "voxel_max": np.array([4, 5, 0]),
    }
    panel = SimpleNamespace(
        sphere_params_per_roi={},
        rectangle_params_per_roi={"roi": rectangle_params},
        _find_streamlines_in_radius=Mock(return_value={0}),
        _find_streamlines_in_oriented_box=Mock(return_value={1}),
        update_roi_layer=Mock(),
        update_status=Mock(),
    )
    data = np.ones((9, 9, 2), dtype=np.uint8)
    window = SimpleNamespace(
        tractogram_data=lines,
        streamline_bboxes=np.array(
            [[line.min(axis=0), line.max(axis=0)] for line in lines]
        ),
        roi_layers={
            "roi": {
                "data": data,
                "affine": np.eye(4),
                "inv_affine": np.eye(4),
            }
        },
        roi_intersection_cache={"roi": {1}},
        unified_undo_stack=[
            {
                "action_type": ActionType.ROI_MODIFICATION,
                "roi_name": "roi",
                "data_snapshot": np.zeros_like(data),
                "sphere_params": sphere_params,
                "rectangle_params": None,
            }
        ],
        unified_redo_stack=[],
        vtk_panel=panel,
        _update_action_states=Mock(),
    )
    roi_manager = ROIManager(window)
    roi_manager.update_roi_visual_selection = Mock()
    roi_manager.apply_logic_filters = Mock()
    window.roi_manager = roi_manager
    state_manager = StateManager(window)

    state_manager.perform_undo()
    assert window.roi_intersection_cache["roi"] == {0}
    assert "roi" in panel.sphere_params_per_roi
    assert "roi" not in panel.rectangle_params_per_roi

    state_manager.perform_redo()
    assert window.roi_intersection_cache["roi"] == {1}
    assert "roi" not in panel.sphere_params_per_roi
    assert "roi" in panel.rectangle_params_per_roi






def test_cli_sphere_uses_shared_world_space_rasterizer(monkeypatch):
    from tractedit_pkg import main_window

    data = np.zeros((9, 9, 9), dtype=np.uint8)
    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    rasterize = Mock()
    monkeypatch.setattr(main_window, "_rasterize_world_sphere", rasterize)
    window = SimpleNamespace(
        anatomical_image_data=data,
        current_drawing_roi="roi",
        vtk_panel=None,
        roi_layers={
            "roi": {
                "data": data,
                "affine": affine,
                "inv_affine": np.linalg.inv(affine),
            }
        },
        _trigger_new_roi=Mock(),
        _update_action_states=Mock(),
        _update_data_panel_display=Mock(),
    )

    main_window.MainWindow.load_initial_files(
        window, roi_in=[[8.0, 4.0, 4.0]], radius=[2.0]
    )

    rasterize.assert_called_once()
    args = rasterize.call_args.args
    np.testing.assert_allclose(args[1], [4.0, 4.0, 4.0])
    assert args[2] == 2.0
    np.testing.assert_array_equal(args[3], affine)


@pytest.mark.parametrize("center", [[1.5, 1.5, 1.5], [-2.0, 1.5, 1.5], [1., 1., 1.]])
def test_cli_sphere_registers_geometry_before_actor_creation(center):
    from tractedit_pkg import main_window

    data = np.zeros((4, 4, 4), dtype=np.uint8)
    center = np.asarray(center)
    radius = 0.2
    panel = Mock(sphere_params_per_roi={}, rectangle_params_per_roi={})
    actor_params = []

    def check_actor_geometry(name, mask, affine):
        actor_params.append(panel.sphere_params_per_roi.get(name))

    panel.add_roi_layer.side_effect = check_actor_geometry
    window = SimpleNamespace(
        anatomical_image_data=data, current_drawing_roi="roi", vtk_panel=panel,
        roi_layers={"roi": {"data": data, "affine": np.eye(4), "inv_affine": np.eye(4)}},
        _trigger_new_roi=Mock(), _update_action_states=Mock(),
        _update_data_panel_display=Mock(),
    )
    main_window.MainWindow.load_initial_files(window, roi_in=[center.tolist()], radius=[radius])
    panel.add_roi_layer.assert_called_once()
    assert actor_params[0] is not None
    np.testing.assert_array_equal(actor_params[0]["center"], center)
    assert actor_params[0]["radius"] == radius
    # The continuous sphere still intersects a segment with both endpoints outside.
    params = panel.sphere_params_per_roi["roi"]
    points = np.array([center - [0.5, 0, 0], center + [0.5, 0, 0]])
    hit = batch_check_sphere_intersection(
        points, np.array([0, 2], dtype=np.int64), params["center"], params["radius"]
    )
    assert hit[0]
    grid = np.indices(data.shape).reshape(3, -1).T
    expected = (np.linalg.norm(grid - center, axis=1) <= radius).reshape(data.shape)
    np.testing.assert_array_equal(data > 0, expected)


@pytest.mark.parametrize("radius", [0., -1., np.nan, np.inf])
def test_cli_sphere_rejects_invalid_radius_before_creating_roi(radius):
    from tractedit_pkg import main_window

    window = SimpleNamespace(
        anatomical_image_data=np.zeros((4, 4, 4)),
        _trigger_new_roi=Mock(), _update_action_states=Mock(),
        _update_data_panel_display=Mock(),
    )
    main_window.MainWindow.load_initial_files(window, roi_in=[[1.5, 1.5, 1.5]], radius=[radius])
    window._trigger_new_roi.assert_not_called()
