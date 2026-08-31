# -*- coding: utf-8 -*-
"""Regression tests for native-grid rendering of oblique images."""

import numpy as np
import pytest
import vtk
from fury import window

from tractedit_pkg.visualization.actors import (
    apply_slicer_affine,
    create_slicer_actor,
)
from tractedit_pkg.visualization.coordinates import (
    camera_frame_from_affine,
    translate_camera_to_plane,
)
from tractedit_pkg.visualization.scene_manager import SceneManager


def _vtk_matrix_to_numpy(matrix: vtk.vtkMatrix4x4) -> np.ndarray:
    return np.array(
        [[matrix.GetElement(row, col) for col in range(4)] for row in range(4)]
    )


@pytest.fixture(
    params=[
        np.array(
            [
                [0.70, 0.00, 0.00, -12.0],
                [0.00, 0.70, 0.00, -18.0],
                [0.00, 0.00, 0.70, 9.0],
                [0.00, 0.00, 0.00, 1.0],
            ]
        ),
        np.array(
            [
                [0.77, -0.16, 0.04, -33.0],
                [0.18, 0.69, -0.11, 14.0],
                [-0.02, 0.12, 1.08, 22.0],
                [0.00, 0.00, 0.00, 1.0],
            ]
        ),
        np.array(
            [
                [0.62, 0.05, -0.09, 38.0],
                [-0.08, 0.83, 0.13, -21.0],
                [0.04, -0.11, 1.16, 7.0],
                [0.00, 0.00, 0.00, 1.0],
            ]
        ),
    ],
    ids=["01-axis-aligned", "02-oblique", "03-oblique-shear"],
)
def image_affine(request) -> np.ndarray:
    return request.param


def test_native_slicer_preserves_voxels_and_uses_full_affine(image_affine):
    data = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)

    slicer = create_slicer_actor(data, image_affine, (0.0, float(data.max())))

    assert slicer is not None
    np.testing.assert_array_equal(slicer.resliced_array(), data)
    np.testing.assert_allclose(
        _vtk_matrix_to_numpy(slicer.GetUserMatrix()), image_affine, atol=1e-12
    )


def test_slicer_copy_shares_pipeline_and_restores_radiological_matrix(image_affine):
    data = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    base = create_slicer_actor(data, image_affine, (0.0, float(data.max())))
    copied = base.copy()

    apply_slicer_affine(copied, image_affine, radiological=True)

    assert (
        copied.GetMapper().GetInputAlgorithm()
        is base.GetMapper().GetInputAlgorithm()
    )
    radiological = np.diag([-1.0, 1.0, 1.0, 1.0]) @ image_affine
    np.testing.assert_allclose(
        _vtk_matrix_to_numpy(copied.GetUserMatrix()), radiological, atol=1e-12
    )
    np.testing.assert_allclose(
        _vtk_matrix_to_numpy(base.GetUserMatrix()), image_affine, atol=1e-12
    )


def test_axis_aligned_camera_frames_match_existing_views():
    affine = np.diag([0.7, 0.8, 1.2, 1.0])

    axial = camera_frame_from_affine(affine, 2, 1, 1, radiological=True)
    coronal = camera_frame_from_affine(affine, 1, -1, 2, radiological=True)
    sagittal = camera_frame_from_affine(affine, 0, 1, 2)
    three_d = camera_frame_from_affine(affine, 1, 1, 2)

    np.testing.assert_allclose(axial, ([0, 0, 1], [0, 1, 0]), atol=1e-12)
    np.testing.assert_allclose(coronal, ([0, -1, 0], [0, 0, 1]), atol=1e-12)
    np.testing.assert_allclose(sagittal, ([1, 0, 0], [0, 0, 1]), atol=1e-12)
    np.testing.assert_allclose(three_d, ([0, 1, 0], [0, 0, 1]), atol=1e-12)


def test_oblique_camera_frame_is_orthonormal_and_plane_aligned(image_affine):
    direction, view_up = camera_frame_from_affine(
        image_affine, 2, 1, 1, radiological=True
    )
    displayed_affine = np.diag([-1.0, 1.0, 1.0, 1.0]) @ image_affine
    expected_normal = np.cross(displayed_affine[:3, 0], displayed_affine[:3, 1])
    if np.dot(expected_normal, displayed_affine[:3, 2]) < 0:
        expected_normal *= -1
    expected_normal /= np.linalg.norm(expected_normal)

    np.testing.assert_allclose(direction, expected_normal, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(view_up), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(direction, view_up), 0.0, atol=1e-12)


def test_camera_translation_preserves_pan_zoom_and_distance():
    position = np.array([7.0, -5.0, 40.0])
    focal_point = np.array([4.0, 2.0, 3.0])
    plane_point = np.array([11.0, 13.0, 17.0])
    normal = np.array([0.2, -0.3, 0.9])
    normal /= np.linalg.norm(normal)

    new_position, new_focal = translate_camera_to_plane(
        position, focal_point, plane_point, normal
    )

    np.testing.assert_allclose(new_position - new_focal, position - focal_point)
    np.testing.assert_allclose(
        np.dot(plane_point - new_focal, normal), 0.0, atol=1e-12
    )
    shift = new_focal - focal_point
    np.testing.assert_allclose(
        shift, normal * np.dot(plane_point - focal_point, normal)
    )


@pytest.mark.parametrize(
    ("affine", "kwargs"),
    [
        (np.eye(3), {}),
        (np.full((4, 4), np.nan), {}),
        (np.eye(4), {"slice_axis": 3}),
        (np.eye(4), {"view_up_axis": 3}),
        (np.eye(4), {"slice_axis": 1, "view_up_axis": 1}),
        (np.eye(4), {"view_sign": 0}),
        (np.diag([1.0, 1.0, 0.0, 1.0]), {}),
    ],
)
def test_camera_frame_rejects_invalid_geometry(affine, kwargs):
    parameters = {
        "slice_axis": 2,
        "view_sign": 1,
        "view_up_axis": 1,
        **kwargs,
    }
    with pytest.raises(ValueError):
        camera_frame_from_affine(affine, **parameters)


def test_slicer_affine_rejects_invalid_matrix():
    data = np.zeros((2, 2, 2), dtype=np.float32)
    slicer = create_slicer_actor(data, np.eye(4), (0.0, 1.0))

    with pytest.raises(ValueError):
        apply_slicer_affine(slicer, np.eye(3))
    assert create_slicer_actor(data, np.eye(3), (0.0, 1.0)) is None


def test_camera_translation_rejects_zero_normal():
    with pytest.raises(ValueError):
        translate_camera_to_plane(np.ones(3), np.zeros(3), np.ones(3), np.zeros(3))


def test_scene_manager_camera_follows_oblique_slice_without_losing_state():
    data = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
    affine = np.array(
        [
            [0.77, -0.16, 0.04, -33.0],
            [0.18, 0.69, -0.11, 14.0],
            [-0.02, 0.12, 1.08, 22.0],
            [0.00, 0.00, 0.00, 1.00],
        ]
    )
    slicer = create_slicer_actor(
        data, affine, (0.0, float(data.max())), radiological=True
    )
    slicer.display_extent(0, 6, 0, 7, 3, 3)
    scene = window.Scene()
    scene.add(slicer)
    panel = type("Panel", (), {})()
    panel.main_window = type("MainWindow", (), {"anatomical_image_affine": affine})()
    panel.axial_scene = scene
    panel.coronal_scene = window.Scene()
    panel.sagittal_scene = window.Scene()
    panel.axial_slice_actor = slicer
    panel.axial_slice_actor_2d = slicer
    panel.coronal_slice_actor = None
    panel.coronal_slice_actor_2d = None
    panel.sagittal_slice_actor = None
    panel.sagittal_slice_actor_2d = None
    manager = SceneManager(panel)

    manager.update_axial_camera(reset_zoom_pan=True)

    camera = scene.GetActiveCamera()
    expected_direction, expected_up = camera_frame_from_affine(
        affine, 2, 1, 1, radiological=True
    )
    position = np.asarray(camera.GetPosition())
    focal = np.asarray(camera.GetFocalPoint())
    np.testing.assert_allclose(
        (position - focal) / camera.GetDistance(), expected_direction
    )
    np.testing.assert_allclose(camera.GetViewUp(), expected_up)

    in_plane_pan = expected_up * 2.5
    camera.SetPosition(*(position + in_plane_pan))
    camera.SetFocalPoint(*(focal + in_plane_pan))
    camera.SetParallelScale(17.25)
    old_position = np.asarray(camera.GetPosition())
    old_focal = np.asarray(camera.GetFocalPoint())
    old_distance = camera.GetDistance()
    slicer.display_extent(0, 6, 0, 7, 5, 5)

    manager.update_axial_camera(reset_zoom_pan=False)

    new_position = np.asarray(camera.GetPosition())
    new_focal = np.asarray(camera.GetFocalPoint())
    np.testing.assert_allclose(new_position - new_focal, old_position - old_focal)
    np.testing.assert_allclose(camera.GetDistance(), old_distance)
    np.testing.assert_allclose(camera.GetParallelScale(), 17.25)
    np.testing.assert_allclose(
        np.dot(np.asarray(slicer.GetCenter()) - new_focal, expected_direction),
        0.0,
        atol=1e-6,
    )
