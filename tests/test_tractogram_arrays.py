"""Array inspection and component extraction preserve source correspondence."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from tractedit_pkg.tractogram_arrays import (
    DataArray,
    array_statistics,
    collect_arrays,
    colour_range,
    validate_scalar_display,
)


@pytest.mark.parametrize(
    "low,high,expected",
    [(-2, 5, (-2., 5.)), (3., 3., (2.5, 3.5)), (-1e308, 1e308, None),
     (2**53, 2**53 + 1, None), (1e308, 1e308, None), (np.nan, np.inf, None)],
)
def test_colour_range_representability(low, high, expected):
    assert colour_range(low, high) == expected


def test_streamline_components_follow_original_ids_without_mutating_source():
    values = np.arange(24, dtype=np.float64).reshape(6, 4)
    field = DataArray("streamline", "embeddings", values)
    assert field.components == 4
    assert field.dtype == np.dtype("float64")
    np.testing.assert_array_equal(field.gather([5, 1], 2), [22, 6])
    np.testing.assert_array_equal(values, np.arange(24).reshape(6, 4))


def test_point_components_preserve_variable_lengths_and_requested_order():
    values = nib.streamlines.ArraySequence(
        [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6]], dtype=np.float32),
            np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32),
        ]
    )
    field = DataArray("point", "metric", values)
    assert field.shape == (6, 2)
    np.testing.assert_array_equal(field.gather([2, 0], 1), [8, 10, 12, 2, 4])


@pytest.mark.parametrize("component", [-1, 4])
def test_component_outside_array_is_rejected(component):
    field = DataArray("streamline", "embeddings", np.zeros((2, 4)))
    with pytest.raises(ValueError, match="component"):
        field.gather([0], component)


def test_inventory_excludes_internal_cache_and_includes_group_data():
    embeddings = np.zeros((3, 32), dtype=np.float64)
    window = SimpleNamespace(
        scalar_data_per_point={},
        data_per_streamline={
            "embeddings": embeddings,
            "_tractedit_bboxes": np.zeros((3, 6), dtype=np.float32),
        },
        trx_file_reference=SimpleNamespace(
            data_per_group={"bundle": {"mean_fa": np.array([[0.7]])}},
        ),
    )
    fields = collect_arrays(window)
    assert [(field.association, field.name, field.group) for field in fields] == [
        ("streamline", "embeddings", ""),
        ("group", "mean_fa", "bundle"),
    ]
    assert fields[0].values is embeddings


def test_scalar_arrays_and_empty_point_arrays_have_one_component():
    field = DataArray("streamline", "weight", np.array([1, 2], dtype=np.int64))
    assert field.shape == (2, 1)
    np.testing.assert_array_equal(field.gather([1], 0), [2])
    empty = DataArray("point", "metric", nib.streamlines.ArraySequence())
    assert empty.shape == (0, 1)
    assert empty.gather([], 0).size == 0


def test_statistics_report_nonfinite_values_without_replacing_them():
    values = np.array([1.0, np.nan, np.inf, -np.inf, 1.0])
    field = DataArray("streamline", "weight", values)
    result = array_statistics(field, 0, range(5), lambda: False)
    assert result["count"] == 2 and result["missing"] == 3
    assert result["min"] == result["max"] == 1.0
    assert result["histogram"].sum() == 2
    assert np.isnan(values[1]) and np.isposinf(values[2])
    empty = array_statistics(field, 0, [1, 2, 3], lambda: False)
    assert empty == {"count": 0, "missing": 3, "histogram": None}


@pytest.mark.parametrize(
    "setting",
    [
        {"association": "group", "component": 0, "colormap": "viridis"},
        {"association": "point", "component": True, "colormap": "viridis"},
        {"association": "point", "component": -1, "colormap": "viridis"},
        {"association": "point", "component": 0, "colormap": "unknown"},
    ],
)
def test_invalid_colour_recipes_are_rejected(setting):
    with pytest.raises(ValueError, match="scalar component"):
        validate_scalar_display(setting)


def test_table_model_is_read_only_and_uses_source_indices():
    from PyQt6.QtCore import Qt
    from tractedit_pkg.ui.tractogram_data import ArrayTableModel

    field = DataArray("streamline", "identifier", np.array([2**53 + 1, 7, 9]))
    model = ArrayTableModel(field, [2, 0])
    assert model.data(model.index(1, 0)) == "0"
    assert model.data(model.index(1, 1)) == str(2**53 + 1)
    assert not model.flags(model.index(0, 1)) & Qt.ItemFlag.ItemIsEditable


@pytest.mark.parametrize(
    "item_type,name,column,opens",
    [
        ("tractogram_array", "embeddings", 0, True),
        ("tractogram_array", "missing", 0, False),
        ("tractogram_array", "embeddings", 1, False),
        ("bundle", "", 0, False),
        (None, "", 0, False),
    ],
)
def test_double_click_opens_only_known_array_rows(item_type, name, column, opens):
    from unittest.mock import Mock
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTreeWidgetItem
    from tractedit_pkg.ui.data_panel import DataPanelManager

    manager = DataPanelManager.__new__(DataPanelManager)
    manager.tractogram_data = Mock()
    manager.tractogram_data.fields = {("streamline", "", "embeddings"): object()}
    item = QTreeWidgetItem(["row"])
    data = (
        None
        if item_type is None
        else {
            "type": item_type,
            "association": "streamline",
            "group": "",
            "name": name,
        }
    )
    item.setData(0, Qt.ItemDataRole.UserRole, data)
    manager.on_data_panel_item_double_clicked(item, column)
    if opens:
        manager.tractogram_data.select.assert_called_once_with(data)
        manager.tractogram_data.inspect_values.assert_called_once_with()
    else:
        manager.tractogram_data.select.assert_not_called()
        manager.tractogram_data.inspect_values.assert_not_called()


def test_component_palettes_without_matplotlib(monkeypatch):
    import sys
    from hashlib import sha256
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkLookupTable
    from tractedit_pkg.tractogram_arrays import COLOR_MAPS, configure_component_lut

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    expected = {
        "cividis": "fa34e03df2aac0396eb738bdb39bdecb652c180a7bd1ad740248fef10e263fae",
        "viridis": "3f712e85e75bbbbe5003a91ff6f038f1287255270b6e22fd888e28499cb1cfd2",
        "RdBu": "b4a0e684132321913ba3ffb52f5b261896267c4441b911f52b98ccd28d3ccffb",
    }
    for name in COLOR_MAPS:
        lut = vtkLookupTable()
        configure_component_lut(lut, name)
        assert lut.GetNumberOfTableValues() == 256
        assert lut.GetNanColor() == (0.5, 0.5, 0.5, 1.0)
        assert sha256(vtk_to_numpy(lut.GetTable()).tobytes()).hexdigest() == expected[name]
