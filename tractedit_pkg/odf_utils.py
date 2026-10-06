# tractedit_pkg/odf_utils.py

"""
ODF (Orientation Distribution Function) utilities for spherical harmonics
visualization and streamline-based mask generation.
"""

# ============================================================================
# Imports
# ============================================================================

import math
from numbers import Integral

import numpy as np
import vtk
from scipy.ndimage import binary_dilation
from scipy.special import sph_harm_y
from vtk.util import numpy_support


SUPPORTED_SH_BASES = frozenset({"mrtrix3", "tournier07"})


# ============================================================================
# Sphere Geometry
# ============================================================================


class SimpleSphere:
    """A minimal sphere class compatible with FURY actors."""

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces


def generate_symmetric_sphere(radius=1.0, subdivisions=3):
    """
    Generates an icosahedral sphere using VTK.
    subdivisions=3 -> ~642 vertices.
    """
    # Create Icosahedron
    source = vtk.vtkPlatonicSolidSource()
    source.SetSolidTypeToIcosahedron()

    # Subdivide
    subdivider = vtk.vtkLoopSubdivisionFilter()
    subdivider.SetInputConnection(source.GetOutputPort())
    subdivider.SetNumberOfSubdivisions(subdivisions)
    subdivider.Update()

    mesh = subdivider.GetOutput()

    # Extract Vertices
    v_data = mesh.GetPoints().GetData()
    vertices = numpy_support.vtk_to_numpy(v_data)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = (vertices / norms) * radius

    # Extract Faces
    polys_data = mesh.GetPolys().GetData()
    polys_raw = numpy_support.vtk_to_numpy(polys_data)
    faces = polys_raw.reshape(-1, 4)[:, 1:]

    return SimpleSphere(vertices, faces)


# ============================================================================
# Spherical Harmonics
# ============================================================================


def compute_sh_basis(
    vertices: np.ndarray,
    sh_order: int,
    basis_type: str = "tournier07",
) -> np.ndarray:
    """Compute the symmetric real MRtrix3 spherical harmonic basis."""
    if not isinstance(sh_order, Integral) or isinstance(sh_order, bool):
        raise ValueError("SH order must be a non-negative even integer.")
    sh_order = int(sh_order)
    if sh_order < 0 or sh_order % 2:
        raise ValueError("SH order must be a non-negative even integer.")

    if (
        not isinstance(basis_type, str)
        or basis_type.casefold() not in SUPPORTED_SH_BASES
    ):
        supported = ", ".join(sorted(SUPPORTED_SH_BASES))
        raise ValueError(
            f"Unsupported SH basis '{basis_type}'. Supported bases: {supported}."
        )

    x, y, z = vertices.T
    r = np.sqrt(x**2 + y**2 + z**2)
    polar = np.arccos(np.clip(z / r, -1.0, 1.0))
    azimuth = np.arctan2(y, x)

    n_coeffs = (sh_order + 1) * (sh_order + 2) // 2
    B = np.zeros((len(vertices), n_coeffs))

    idx = 0
    for degree in range(0, sh_order + 1, 2):
        for m in range(-degree, degree + 1):
            if m < 0:
                basis_val = np.sqrt(2) * sph_harm_y(degree, abs(m), polar, azimuth).imag
            elif m == 0:
                basis_val = sph_harm_y(degree, 0, polar, azimuth).real
            else:
                basis_val = np.sqrt(2) * sph_harm_y(degree, m, polar, azimuth).real

            B[:, idx] = basis_val
            idx += 1
    return B


def calculate_sh_order(n_coeffs: int) -> int:
    """Calculates SH order from number of coefficients."""
    if not isinstance(n_coeffs, Integral) or isinstance(n_coeffs, bool):
        raise ValueError(f"Invalid coefficient count ({n_coeffs}).")
    n_coeffs = int(n_coeffs)
    if n_coeffs < 1:
        raise ValueError(f"Invalid coefficient count ({n_coeffs}).")

    discriminant = 8 * n_coeffs + 1
    root = math.isqrt(discriminant)
    sh_order = (root - 3) // 2
    if (
        root * root != discriminant
        or sh_order < 0
        or sh_order % 2
        or (sh_order + 1) * (sh_order + 2) // 2 != n_coeffs
    ):
        raise ValueError(f"Invalid coefficient count ({n_coeffs}).")
    return sh_order


# ============================================================================
# Streamline Mask Generation
# ============================================================================


def _voxel_indices(streamline, inverse_affine, volume_shape):
    points = np.asarray(streamline)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.intp)
    voxel_coordinates = points @ inverse_affine[:3, :3].T
    voxel_coordinates += inverse_affine[:3, 3]
    indices = np.rint(voxel_coordinates).astype(np.intp)
    shape = np.asarray(volume_shape[:3], dtype=np.intp)
    valid = np.all((indices >= 0) & (indices < shape), axis=1)
    return indices[valid]


def create_cropped_tunnel_mask(streamlines, affine, volume_shape, dilation_iter=1):
    """Return the occupied mask crop and its origin in source voxels."""
    shape = np.asarray(volume_shape[:3], dtype=np.intp)
    if shape.shape != (3,) or np.any(shape < 0):
        raise ValueError("ODF volume shape must contain three non-negative values.")

    streamlines = tuple(streamlines)
    inverse_affine = np.linalg.inv(affine)
    lower = shape.copy()
    upper = np.zeros(3, dtype=np.intp)
    found = False

    for streamline in streamlines:
        indices = _voxel_indices(streamline, inverse_affine, shape)
        if indices.size == 0:
            continue
        lower = np.minimum(lower, indices.min(axis=0))
        upper = np.maximum(upper, indices.max(axis=0) + 1)
        found = True

    if not found:
        return np.zeros((0, 0, 0), dtype=bool), np.zeros(3, dtype=np.intp)

    dilation_iter = max(0, int(dilation_iter))
    lower = np.maximum(0, lower - dilation_iter)
    upper = np.minimum(shape, upper + dilation_iter)
    mask = np.zeros(tuple(upper - lower), dtype=bool)

    for streamline in streamlines:
        indices = _voxel_indices(streamline, inverse_affine, shape)
        if indices.size == 0:
            continue
        local = indices - lower
        mask[local[:, 0], local[:, 1], local[:, 2]] = True

    if dilation_iter:
        mask = binary_dilation(mask, iterations=dilation_iter)
    return mask, lower


def create_tunnel_mask(streamlines, affine, volume_shape, dilation_iter=1):
    """Generate a full-size tunnel mask without concatenating streamline points."""
    shape = tuple(volume_shape[:3])
    mask = np.zeros(shape, dtype=bool)
    cropped, origin = create_cropped_tunnel_mask(
        streamlines, affine, shape, dilation_iter=dilation_iter
    )
    if cropped.size == 0:
        return mask
    stop = origin + np.asarray(cropped.shape)
    mask[
        origin[0] : stop[0],
        origin[1] : stop[1],
        origin[2] : stop[2],
    ] = cropped
    return mask


def build_tunnel_odf_amplitudes(
    coefficients,
    streamlines,
    affine,
    basis,
    dilation_iter=1,
    projection_chunk_size=65536,
):
    """Project SH coefficients only inside the streamline tunnel crop."""
    coefficients = np.asarray(coefficients)
    basis = np.asarray(basis)
    if coefficients.ndim != 4 or basis.ndim != 2:
        raise ValueError("ODF coefficients and SH basis must be 4D and 2D arrays.")
    if coefficients.shape[-1] != basis.shape[-1]:
        raise ValueError("ODF coefficient count does not match the SH basis.")
    if projection_chunk_size < 1:
        raise ValueError("Projection chunk size must be positive.")

    mask, origin = create_cropped_tunnel_mask(
        streamlines,
        affine,
        coefficients.shape,
        dilation_iter=dilation_iter,
    )
    if mask.size == 0:
        return None, np.asarray(affine).copy()

    amplitudes = np.zeros(mask.shape + (basis.shape[0],), dtype=np.float32)
    occupied = np.flatnonzero(mask)
    flat_amplitudes = amplitudes.reshape(-1, basis.shape[0])

    for start in range(0, len(occupied), projection_chunk_size):
        selected = occupied[start : start + projection_chunk_size]
        local = np.unravel_index(selected, mask.shape)
        selected_coefficients = coefficients[
            local[0] + origin[0],
            local[1] + origin[1],
            local[2] + origin[2],
        ]
        flat_amplitudes[selected] = selected_coefficients @ basis.T

    translation = np.eye(4)
    translation[:3, 3] = origin
    cropped_affine = np.asarray(affine) @ translation
    return amplitudes, cropped_affine
