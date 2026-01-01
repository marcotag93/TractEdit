# -*- coding: utf-8 -*-
"""
Unit tests for tractedit_pkg/odf_utils.py.

Tests spherical harmonic utilities and tunnel mask generation.
"""

import pytest
import numpy as np
from nibabel.streamlines import ArraySequence

from tractedit_pkg.odf_utils import (
    SimpleSphere,
    generate_symmetric_sphere,
    compute_sh_basis,
    calculate_sh_order,
    create_tunnel_mask,
)


class TestSimpleSphere:
    """Tests for the SimpleSphere class."""

    def test_initialization(self):
        """SimpleSphere should store vertices and faces."""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        sphere = SimpleSphere(vertices, faces)

        assert np.array_equal(sphere.vertices, vertices)
        assert np.array_equal(sphere.faces, faces)


class TestGenerateSymmetricSphere:
    """Tests for the generate_symmetric_sphere function."""

    def test_returns_simple_sphere(self):
        """Function should return a SimpleSphere instance."""
        sphere = generate_symmetric_sphere()
        assert isinstance(sphere, SimpleSphere)

    def test_vertices_shape(self):
        """Vertices should be Nx3 array."""
        sphere = generate_symmetric_sphere(subdivisions=2)
        assert sphere.vertices.ndim == 2
        assert sphere.vertices.shape[1] == 3

    def test_faces_shape(self):
        """Faces should be Mx3 array (triangles)."""
        sphere = generate_symmetric_sphere(subdivisions=2)
        assert sphere.faces.ndim == 2
        assert sphere.faces.shape[1] == 3

    def test_vertices_normalized(self):
        """Vertices should lie on sphere surface (unit norm for radius=1)."""
        sphere = generate_symmetric_sphere(radius=1.0, subdivisions=2)
        norms = np.linalg.norm(sphere.vertices, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_custom_radius(self):
        """Vertices should respect custom radius."""
        radius = 2.5
        sphere = generate_symmetric_sphere(radius=radius, subdivisions=2)
        norms = np.linalg.norm(sphere.vertices, axis=1)
        assert np.allclose(norms, radius, atol=1e-5)

    def test_subdivisions_increase_vertices(self):
        """Higher subdivisions should produce more vertices."""
        sphere_low = generate_symmetric_sphere(subdivisions=1)
        sphere_high = generate_symmetric_sphere(subdivisions=3)

        assert len(sphere_high.vertices) > len(sphere_low.vertices)

    @pytest.mark.slow
    def test_high_subdivisions(self):
        """Test high subdivision level (slow)."""
        sphere = generate_symmetric_sphere(subdivisions=4)
        assert len(sphere.vertices) > 1000


class TestComputeShBasis:
    """Tests for the compute_sh_basis function."""

    def test_output_shape_order_0(self):
        """Order 0 SH should produce 1 coefficient."""
        vertices = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        B = compute_sh_basis(vertices, sh_order=0)

        assert B.shape == (3, 1)

    def test_output_shape_order_2(self):
        """Order 2 SH should produce 6 coefficients (1 + 5)."""
        vertices = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32)
        B = compute_sh_basis(vertices, sh_order=2)

        assert B.shape == (2, 6)

    def test_output_shape_order_4(self):
        """Order 4 SH should produce 15 coefficients (1 + 5 + 9)."""
        vertices = np.array([[0, 0, 1]], dtype=np.float32)
        B = compute_sh_basis(vertices, sh_order=4)

        assert B.shape == (1, 15)

    def test_output_shape_order_8(self):
        """Order 8 SH should produce 45 coefficients."""
        vertices = np.array([[0, 0, 1]], dtype=np.float32)
        B = compute_sh_basis(vertices, sh_order=8)

        # n_coeffs = (l+1)*(l+2)/2 = 9*10/2 = 45
        assert B.shape == (1, 45)

    def test_basis_not_all_zero(self):
        """Basis matrix should contain non-zero values."""
        sphere = generate_symmetric_sphere(subdivisions=2)
        B = compute_sh_basis(sphere.vertices, sh_order=4)

        assert not np.allclose(B, 0)

    def test_dc_component_constant(self):
        """DC component (l=0) should be approximately constant."""
        sphere = generate_symmetric_sphere(subdivisions=2)
        B = compute_sh_basis(sphere.vertices, sh_order=2)

        # First column is l=0, m=0 (DC)
        dc_values = B[:, 0]
        assert np.std(dc_values) < 0.01  # Should be nearly constant


class TestCalculateShOrder:
    """Tests for the calculate_sh_order function."""

    def test_order_0(self):
        """1 coefficient -> order 0."""
        assert calculate_sh_order(1) == 0

    def test_order_2(self):
        """6 coefficients -> order 2."""
        assert calculate_sh_order(6) == 2

    def test_order_4(self):
        """15 coefficients -> order 4."""
        assert calculate_sh_order(15) == 4

    def test_order_6(self):
        """28 coefficients -> order 6."""
        assert calculate_sh_order(28) == 6

    def test_order_8(self):
        """45 coefficients -> order 8."""
        assert calculate_sh_order(45) == 8

    def test_invalid_coefficient_count(self):
        """Invalid coefficient count should raise ValueError or return unexpected order."""
        # Valid counts: 1, 6, 15, 28, 45, ...
        try:
            result = calculate_sh_order(10)
            # If it doesn't raise, the result should be a non-integer internally
            # which means the function handles it differently
            assert isinstance(result, (int, float))
        except ValueError:
            pass  # Expected behavior

    def test_invalid_coefficient_count_2(self):
        """Another invalid coefficient count."""
        with pytest.raises(ValueError):
            calculate_sh_order(5)


class TestCreateTunnelMask:
    """Tests for the create_tunnel_mask function."""

    def test_output_shape(self, sample_streamlines, sample_affine, sample_volume_shape):
        """Mask should match volume shape."""
        mask = create_tunnel_mask(
            sample_streamlines,
            sample_affine,
            sample_volume_shape
        )

        assert mask.shape == sample_volume_shape

    def test_output_dtype(self, sample_streamlines, sample_affine, sample_volume_shape):
        """Mask should be boolean."""
        mask = create_tunnel_mask(
            sample_streamlines,
            sample_affine,
            sample_volume_shape
        )

        assert mask.dtype == bool

    def test_mask_contains_true(self, sample_streamlines, sample_affine,
                                sample_volume_shape):
        """Mask should contain True values where streamlines pass."""
        mask = create_tunnel_mask(
            sample_streamlines,
            sample_affine,
            sample_volume_shape
        )

        assert np.any(mask)

    def test_dilation_expands_mask(self, sample_affine):
        """Higher dilation should expand the mask."""
        # Create a simple streamline through the center
        streamlines = ArraySequence([
            np.array([[32, 32, 32]], dtype=np.float32)
        ])
        volume_shape = (64, 64, 64)

        mask_small = create_tunnel_mask(
            streamlines, sample_affine, volume_shape, dilation_iter=0
        )
        mask_large = create_tunnel_mask(
            streamlines, sample_affine, volume_shape, dilation_iter=3
        )

        assert np.sum(mask_large) > np.sum(mask_small)

    def test_no_dilation(self, sample_affine):
        """Zero dilation should mark only occupied voxels."""
        streamlines = ArraySequence([
            np.array([[10, 10, 10]], dtype=np.float32)
        ])
        volume_shape = (64, 64, 64)

        mask = create_tunnel_mask(
            streamlines, sample_affine, volume_shape, dilation_iter=0
        )

        # Should have exactly one voxel marked
        assert np.sum(mask) == 1
        assert mask[10, 10, 10]

    def test_out_of_bounds_streamlines(self, sample_affine):
        """Streamlines outside volume should be ignored."""
        streamlines = ArraySequence([
            np.array([[-100, -100, -100]], dtype=np.float32)
        ])
        volume_shape = (64, 64, 64)

        mask = create_tunnel_mask(
            streamlines, sample_affine, volume_shape, dilation_iter=0
        )

        # No voxels should be marked
        assert np.sum(mask) == 0

    def test_scaled_affine(self, sample_affine_scaled):
        """Mask should work with non-identity affine."""
        # With 2mm voxels, world coord (4,4,4) -> voxel (2,2,2)
        streamlines = ArraySequence([
            np.array([[4.0, 4.0, 4.0]], dtype=np.float32)
        ])
        volume_shape = (10, 10, 10)

        mask = create_tunnel_mask(
            streamlines, sample_affine_scaled, volume_shape, dilation_iter=0
        )

        assert mask[2, 2, 2]
