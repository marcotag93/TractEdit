"""Track density on a scanner-RAS reference grid."""

import nibabel as nib
import numpy as np

from ..reference_grid import ReferenceGrid


def density_grid_from_points(points, *, provenance):
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    lower -= 5.0
    upper += 5.0
    shape = tuple(np.ceil((upper - lower) / np.ones(3)).astype(int))
    affine = np.eye(4)
    affine[:3, 3] = lower
    return ReferenceGrid(affine=affine, shape=shape, provenance=provenance)


def density_image(points, reference_grid):
    shape = reference_grid.shape
    affine = reference_grid.affine
    scale = {"meter": 1000.0, "micron": 0.001}.get(reference_grid.xyzt_units[0])
    if scale is not None:
        affine = affine.copy()
        affine[:3] *= scale
    voxels = nib.affines.apply_affine(np.linalg.inv(affine), points)
    indices = np.rint(voxels).astype(int)
    valid = (
        (indices[:, 0] >= 0) & (indices[:, 0] < shape[0])
        & (indices[:, 1] >= 0) & (indices[:, 1] < shape[1])
        & (indices[:, 2] >= 0) & (indices[:, 2] < shape[2])
    )
    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(counts, tuple(indices[valid].T), 1)
    return reference_grid.create_nifti(counts.astype(np.float32)), counts
