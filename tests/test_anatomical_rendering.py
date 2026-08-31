# -*- coding: utf-8 -*-
"""Integration tests for shared native-grid anatomical slicers."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from fury import window

from tractedit_pkg.visualization.vtk_panel import VTKPanel


def _vtk_matrix_to_numpy(matrix) -> np.ndarray:
    return np.array(
        [[matrix.GetElement(row, col) for col in range(4)] for row in range(4)]
    )


def test_anatomical_views_share_pipeline_and_keep_oblique_affine():
    data = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
    affine = np.array(
        [
            [0.77, -0.16, 0.04, -33.0],
            [0.18, 0.69, -0.11, 14.0],
            [-0.02, 0.12, 1.08, 22.0],
            [0.00, 0.00, 0.00, 1.00],
        ]
    )
    panel = VTKPanel.__new__(VTKPanel)
    panel.main_window = SimpleNamespace(
        anatomical_image_data=data,
        anatomical_image_affine=affine,
        anatomical_mmap_image=None,
        image_opacity=0.35,
    )
    panel.scene = window.Scene()
    panel.axial_scene = window.Scene()
    panel.coronal_scene = window.Scene()
    panel.sagittal_scene = window.Scene()
    panel.axial_overlay_renderer = window.Scene()
    panel.coronal_overlay_renderer = window.Scene()
    panel.sagittal_overlay_renderer = window.Scene()
    panel.current_slice_indices = {"x": None, "y": None, "z": None}
    panel.image_extents = {"x": None, "y": None, "z": None}
    panel.mmap_image_extents = {"x": None, "y": None, "z": None}
    panel.roi_slice_actors = {}
    panel.scale_bar_manager = Mock()
    panel.clear_anatomical_slices = lambda reset_state=False: None
    panel._create_or_update_crosshairs = lambda: None
    panel._setup_ortho_cameras = lambda: None
    panel._initialize_3d_camera = lambda: None
    panel._render_all = lambda: None
    panel.update_status = lambda *_args: None

    panel.update_anatomical_slices()

    slicers = (
        panel.axial_slice_actor,
        panel.coronal_slice_actor,
        panel.sagittal_slice_actor,
        panel.axial_slice_actor_2d,
        panel.coronal_slice_actor_2d,
        panel.sagittal_slice_actor_2d,
    )
    algorithms = [slicer.GetMapper().GetInputAlgorithm() for slicer in slicers]
    assert all(algorithm is algorithms[0] for algorithm in algorithms[1:])
    np.testing.assert_array_equal(panel.axial_slice_actor.resliced_array(), data)

    mirror_x = np.diag([-1.0, 1.0, 1.0, 1.0])
    for slicer in (
        panel.axial_slice_actor,
        panel.coronal_slice_actor,
        panel.sagittal_slice_actor,
        panel.sagittal_slice_actor_2d,
    ):
        np.testing.assert_allclose(_vtk_matrix_to_numpy(slicer.GetUserMatrix()), affine)
    for slicer in (panel.axial_slice_actor_2d, panel.coronal_slice_actor_2d):
        np.testing.assert_allclose(
            _vtk_matrix_to_numpy(slicer.GetUserMatrix()), mirror_x @ affine
        )
        np.testing.assert_allclose(slicer.GetScale(), (1.0, 1.0, 1.0))

    for slicer in (
        panel.axial_slice_actor,
        panel.coronal_slice_actor,
        panel.sagittal_slice_actor,
    ):
        assert slicer.GetProperty().GetOpacity() == 0.35
    for slicer in (
        panel.axial_slice_actor_2d,
        panel.coronal_slice_actor_2d,
        panel.sagittal_slice_actor_2d,
    ):
        assert slicer.GetProperty().GetOpacity() == 1.0
