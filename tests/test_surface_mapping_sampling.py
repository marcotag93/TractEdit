# -*- coding: utf-8 -*-
"""Small deterministic contracts for mapping presentation samples."""

import numpy as np
import pytest

from tractedit_pkg.surface_mapping import (
    ENDPOINT_DTYPE,
    INTERSECTION_DTYPE,
    INTERVAL_DTYPE,
    SurfaceMappingCancelled,
    process,
)


@pytest.mark.parametrize("count", [0, 6, 7, 8])
@pytest.mark.parametrize(
    "dtype,field,classes",
    [(ENDPOINT_DTYPE, "status", (0, 2)), (INTERSECTION_DTYPE, "class", (0, 1, 3))],
)
def test_marker_sampling_limit_and_record_correspondence(
    monkeypatch, count, dtype, field, classes
):
    monkeypatch.setattr(process, "MARKER_LIMIT", 7)
    table = np.zeros(count, dtype=dtype)
    table["streamline_id"] = np.arange(count) // 2
    table[field] = np.resize(classes, count)
    table["position_ras"][:, 0] = np.arange(count)
    points, labels, associated, total = process._sample_points(
        table, field, classes, lambda: False
    )
    assert total == count
    assert points.shape == (min(count, 7), 3)
    rows = points[:, 0].astype(int)
    np.testing.assert_array_equal(labels, table[field][rows])
    np.testing.assert_array_equal(
        associated, np.unique(table["streamline_id"][table[field] == 0])
    )
    assert set(labels) == set(table[field])
    if count <= 7:
        np.testing.assert_array_equal(rows, np.arange(count))


def test_marker_sampling_excludes_nonmarker_records():
    table = np.zeros(5, dtype=ENDPOINT_DTYPE)
    table["streamline_id"] = np.arange(5)
    table["status"] = [1, 0, 3, 2, 1]
    table["position_ras"][:, 0] = np.arange(5)
    points, labels, associated, total = process._sample_points(
        table, "status", (0, 2), lambda: False
    )
    assert total == 2
    np.testing.assert_array_equal(points[:, 0], [1, 3])
    np.testing.assert_array_equal(labels, [0, 2])
    np.testing.assert_array_equal(associated, [1])


@pytest.mark.parametrize("empty", [None, np.empty(0, dtype=INTERVAL_DTYPE)])
def test_overlap_sampling_empty(empty):
    points, total = process._sample_overlaps(empty, [], lambda: False)
    assert points.shape == (0, 2, 3)
    assert total == 0


def test_overlap_sampling_clips_single_segment():
    line = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    intervals = np.zeros(1, dtype=INTERVAL_DTYPE)
    intervals["start_position_ras"] = [2, 0, 0]
    intervals["end_position_ras"] = [7, 0, 0]
    points, total = process._sample_overlaps(intervals, [line], lambda: False)
    assert total == 1
    np.testing.assert_array_equal(points, [[[2, 0, 0], [7, 0, 0]]])


def test_marker_sampling_honors_cancellation():
    with pytest.raises(SurfaceMappingCancelled):
        process._sample_points(
            np.zeros(1, dtype=ENDPOINT_DTYPE), "status", (0, 2), lambda: True
        )
