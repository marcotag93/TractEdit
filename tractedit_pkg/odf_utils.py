# tractedit_pkg/odf_utils.py

"""
ODF (Orientation Distribution Function) utilities for spherical harmonics
visualization and streamline-based mask generation.
"""

# ============================================================================
# Imports
# ============================================================================

import numpy as np
import math
import vtk
from vtk.util import numpy_support
from scipy.special import sph_harm
from scipy.ndimage import binary_dilation
import nibabel as nib


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


def compute_sh_basis(vertices, sh_order, basis_type='tournier07'):
    """
    Computes the Spherical Harmonic basis matrix B.
    """
    x, y, z = vertices.T
    r = np.sqrt(x**2 + y**2 + z**2)
    polar = np.arccos(np.clip(z / r, -1.0, 1.0)) 
    azimuth = np.arctan2(y, x)
    
    n_coeffs = int((sh_order + 1) * (sh_order + 2) / 2)
    B = np.zeros((len(vertices), n_coeffs))
    
    idx = 0
    for l in range(0, sh_order + 1, 2):  # Even orders only
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, azimuth, polar)
            
            # Tournier07 Real Basis
            if m < 0:
                basis_val = np.sqrt(2) * Y_lm.imag
            elif m == 0:
                basis_val = Y_lm.real
            else:
                basis_val = np.sqrt(2) * Y_lm.real
            
            B[:, idx] = basis_val
            idx += 1
    return B


def calculate_sh_order(n_coeffs):
    """Calculates SH order from number of coefficients."""
    l_max = (math.sqrt(8 * n_coeffs + 1) - 3) / 2
    if not l_max.is_integer():
        raise ValueError(f"Invalid coefficient count ({n_coeffs}).")
    return int(l_max)


# ============================================================================
# Streamline Mask Generation
# ============================================================================


def create_tunnel_mask(streamlines, affine, volume_shape, dilation_iter=1):
    """
    Generates a dilated binary mask of the voxels occupied by the streamlines.
    
    Args:
        streamlines: List/ArraySequence of streamlines in RASmm space.
        affine: 4x4 affine matrix of the ODF volume.
        volume_shape: Shape of the ODF volume (x, y, z).
        dilation_iter: Number of voxels to dilate the mask.
        
    Returns:
        A boolean numpy array of shape volume_shape (3D).
    """
    mask = np.zeros(volume_shape[:3], dtype=bool)
    inv_affine = np.linalg.inv(affine)
    
    # Transform all points to voxel coordinates
    all_points = np.concatenate(streamlines, axis=0)
    
    # Apply inverse affine
    # P_vox = P_world * R^T + T  (simplified: dot product with inv affine)
    # Nibabel affines: voxel_coord = inv_affine @ [x,y,z,1]
    # Efficient affine transform:
    # (N, 3) -> (N, 4)
    all_points_homog = np.hstack((all_points, np.ones((len(all_points), 1))))
    vox_coords = np.dot(all_points_homog, inv_affine.T)[:, :3]
    
    # Round to nearest integer
    vox_indices = np.rint(vox_coords).astype(int)
    
    # Filter bounds
    valid_mask = (
        (vox_indices[:, 0] >= 0) & (vox_indices[:, 0] < volume_shape[0]) &
        (vox_indices[:, 1] >= 0) & (vox_indices[:, 1] < volume_shape[1]) &
        (vox_indices[:, 2] >= 0) & (vox_indices[:, 2] < volume_shape[2])
    )
    valid_indices = vox_indices[valid_mask]
    
    # Mark voxels
    mask[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True
    
    # Dilate
    if dilation_iter > 0:
        mask = binary_dilation(mask, iterations=dilation_iter)
        
    return mask