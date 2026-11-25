# -*- coding: utf-8 -*-
"""Interaction restores actor shading without opening a render context."""

from unittest.mock import Mock

import pytest
import vtk

from tractedit_pkg.visualization.vtk_panel import VTKPanel


@pytest.fixture
def panel():
    result = VTKPanel.__new__(VTKPanel)
    result._is_interacting = False
    result._interaction_streamline_property = None
    result._fast_update_rate = 30.0
    result._original_desired_update_rate = 0.0001
    result._mesh_render_settings = None
    result._set_mesh_interaction_preview = Mock()
    result._is_render_window_ready = Mock(return_value=True)
    result.render_window = Mock()
    result.streamlines_actor = vtk.vtkActor()
    return result


@pytest.mark.parametrize(
    "interpolation", [vtk.VTK_FLAT, vtk.VTK_GOURAUD, vtk.VTK_PHONG]
)
def test_interaction_restores_original_shading(panel, interpolation):
    prop = panel.streamlines_actor.GetProperty()
    prop.SetInterpolation(interpolation)
    for _ in range(3):
        panel._enable_fast_render()
        assert prop.GetInterpolation() == vtk.VTK_FLAT
        panel.render_window.SetMultiSamples.assert_called_with(0)
        panel._enable_fast_render()
        panel._disable_fast_render()
        assert prop.GetInterpolation() == interpolation
        assert panel._interaction_streamline_property is None
        panel.render_window.SetMultiSamples.assert_called_with(4)
        panel.render_window.SetDesiredUpdateRate.assert_called_with(0.0001)


def test_interaction_does_not_overwrite_replacement_shading(panel):
    original = panel.streamlines_actor.GetProperty()
    original.SetInterpolationToPhong()
    panel._enable_fast_render()
    panel.streamlines_actor = vtk.vtkActor()
    replacement = panel.streamlines_actor.GetProperty()
    replacement.SetInterpolationToFlat()
    panel._disable_fast_render()
    assert original.GetInterpolation() == vtk.VTK_PHONG
    assert replacement.GetInterpolation() == vtk.VTK_FLAT
