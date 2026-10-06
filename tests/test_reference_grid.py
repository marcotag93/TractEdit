"""Tests for normalized tractogram and NIfTI reference grids."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from tractedit_pkg.reference_grid import ReferenceGrid


def _oblique_affine() -> np.ndarray:
    return np.array(
        [
            [0.0, -3.0, 0.4, 10.0],
            [2.0, 0.0, 0.0, -5.0],
            [0.2, 0.5, 4.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_trx_header_is_normalized_with_affine_column_norms():
    affine = _oblique_affine()
    grid = ReferenceGrid.from_header(
        {
            "DIMENSIONS": np.array([7, 8, 9]),
            "VOXEL_TO_RASMM": affine,
        },
        provenance="trx",
    )

    assert grid is not None
    assert grid.shape == (7, 8, 9)
    np.testing.assert_array_equal(grid.affine, affine)
    np.testing.assert_allclose(
        grid.voxel_sizes,
        np.linalg.norm(affine[:3, :3], axis=0),
    )


def test_incomplete_header_does_not_create_a_reference_grid():
    assert (
        ReferenceGrid.from_header(
            {"DIMENSIONS": (7, 8, 9)},
            provenance="incomplete",
        )
        is None
    )


def test_string_header_serializes_to_both_format_conventions():
    affine = _oblique_affine()
    grid = ReferenceGrid.from_header(
        {
            "dimensions": "(7, 8, 9)",
            "voxel_to_rasmm": affine.tolist(),
        },
        provenance="trk",
    )

    assert grid is not None
    fields = grid.header_fields()
    reference = grid.trx_reference(nb_vertices=20, nb_streamlines=3)
    assert fields["dimensions"] == (7, 8, 9)
    assert fields["voxel_order"] == grid.voxel_order
    np.testing.assert_allclose(fields["voxel_sizes"], grid.voxel_sizes)
    np.testing.assert_array_equal(reference["DIMENSIONS"], [7, 8, 9])
    np.testing.assert_array_equal(reference["VOXEL_TO_RASMM"], affine)
    assert reference["NB_VERTICES"] == 20
    assert reference["NB_STREAMLINES"] == 3


def test_synthetic_grid_contains_negative_voxel_coordinates():
    points = np.array(
        [[-2.2, 3.0, 4.0], [1.1, 5.0, 7.0]],
        dtype=np.float64,
    )
    grid = ReferenceGrid.from_points(points, provenance="synthetic")
    voxel_points = nib.affines.apply_affine(
        np.linalg.inv(grid.affine),
        points,
    )

    assert np.all(voxel_points >= 0)
    assert np.all(voxel_points < np.asarray(grid.shape))
    np.testing.assert_array_equal(grid.affine[:3, 3], [-3.0, 0.0, 0.0])


def test_synthetic_grid_with_rotated_affine_contains_all_points():
    affine = _oblique_affine()
    voxel_points = np.array(
        [[-2.2, -1.0, 0.0], [1.1, 2.0, 3.0]],
        dtype=np.float64,
    )
    world_points = nib.affines.apply_affine(affine, voxel_points)

    grid = ReferenceGrid.from_points(
        world_points,
        affine=affine,
        provenance="synthetic-oblique",
    )
    normalized = nib.affines.apply_affine(
        np.linalg.inv(grid.affine),
        world_points,
    )

    assert np.all(normalized >= 0)
    assert np.all(normalized < np.asarray(grid.shape))


def test_oblique_grid_from_world_bounds_contains_all_corners():
    affine = _oblique_affine()
    minimum = np.array([-10.0, -8.0, -6.0])
    maximum = np.array([12.0, 14.0, 16.0])
    corners = (
        np.array(np.meshgrid(*zip(minimum, maximum), indexing="ij")).reshape(3, -1).T
    )

    grid = ReferenceGrid.from_bounds(
        minimum,
        maximum,
        affine=affine,
        provenance="synthetic-oblique-bounds",
    )
    normalized = nib.affines.apply_affine(np.linalg.inv(grid.affine), corners)

    assert np.all(normalized >= 0)
    assert np.all(normalized < np.asarray(grid.shape))


def test_nifti_output_uses_affine_zooms_and_reference_units():
    affine = _oblique_affine()
    source = nib.Nifti1Image(np.zeros((7, 8, 9), dtype=np.uint8), affine)
    source.header.set_xyzt_units("mm", "sec")
    grid = ReferenceGrid.from_nifti(source, provenance="anatomical")

    output = grid.create_nifti(np.zeros(grid.shape, dtype=np.float32))

    np.testing.assert_array_equal(output.affine, affine)
    np.testing.assert_allclose(output.header.get_zooms()[:3], grid.voxel_sizes)
    assert output.header.get_xyzt_units() == ("mm", "sec")


@pytest.mark.parametrize(
    ("affine", "shape"),
    [
        (np.eye(3), (7, 8, 9)),
        (np.full((4, 4), np.nan), (7, 8, 9)),
        (np.eye(4), (7, 0, 9)),
    ],
)
def test_invalid_reference_grid_is_rejected(affine, shape):
    with pytest.raises(ValueError):
        ReferenceGrid(affine=affine, shape=shape, provenance="invalid")


def test_nifti_data_must_match_reference_shape():
    grid = ReferenceGrid(
        affine=np.eye(4),
        shape=(7, 8, 9),
        provenance="shape-check",
    )

    with pytest.raises(ValueError, match="does not match"):
        grid.create_nifti(np.zeros((7, 8, 8), dtype=np.float32))


def test_header_preparation_uses_complete_fallback_grid():
    from tractedit_pkg import file_io

    grid = ReferenceGrid(
        affine=_oblique_affine(),
        shape=(7, 8, 9),
        provenance="anatomical",
    )

    trk_header = file_io._prepare_trk_header(
        {},
        3,
        reference_grid=grid,
    )
    trx_header = file_io._prepare_trx_header(
        {},
        3,
        reference_grid=grid,
    )

    for header in (trk_header, trx_header):
        assert tuple(header["dimensions"]) == grid.shape
        np.testing.assert_array_equal(header["voxel_to_rasmm"], grid.affine)
        np.testing.assert_allclose(header["voxel_sizes"], grid.voxel_sizes)


def test_trx_save_passes_metadata_reference_without_dummy_volume(
    monkeypatch,
    tmp_path,
):
    from tractedit_pkg import file_io

    captured = {}

    def from_lazy_tractogram(tractogram, reference):
        captured["reference"] = reference
        return SimpleNamespace(data_per_streamline={})

    monkeypatch.setattr(
        file_io.tbx.TrxFile,
        "from_lazy_tractogram",
        from_lazy_tractogram,
    )
    monkeypatch.setattr(
        file_io.tbx,
        "save",
        lambda _obj, path: Path(path).write_bytes(b"trx"),
    )
    affine = _oblique_affine()
    tractogram = nib.streamlines.Tractogram(
        [np.array([[0.0, 0.0, 0.0]], dtype=np.float32)],
        affine_to_rasmm=np.eye(4),
    )

    file_io._save_tractogram_file(
        tractogram,
        {
            "dimensions": (7, 8, 9),
            "voxel_to_rasmm": affine,
        },
        str(tmp_path / "output.trx"),
        ".trx",
    )

    assert not isinstance(captured["reference"], nib.Nifti1Image)
    np.testing.assert_array_equal(captured["reference"]["DIMENSIONS"], [7, 8, 9])
    np.testing.assert_array_equal(captured["reference"]["VOXEL_TO_RASMM"], affine)
