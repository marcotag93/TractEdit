# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures for TractEdit tests.

Provides reusable test data and setup for streamline, NIfTI, and ODF testing.
"""

import os
import tempfile
import numpy as np
import pytest
import nibabel as nib
from nibabel.streamlines import ArraySequence


# ============================================================================
# Streamline Fixtures
# ============================================================================


@pytest.fixture
def sample_streamlines():
    """
    Creates a simple ArraySequence with 3 synthetic streamlines.

    Returns:
        ArraySequence: Three streamlines with varying point counts.
    """
    streamlines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]], dtype=np.float32),
        np.array(
            [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 2.0, 0.0], [10.0, 3.0, 0.0]],
            dtype=np.float32,
        ),
    ]
    return ArraySequence(streamlines)


@pytest.fixture
def sample_affine():
    """
    Creates a standard identity affine matrix.

    Returns:
        np.ndarray: 4x4 identity affine matrix with 1mm isotropic voxels.
    """
    return np.eye(4, dtype=np.float32)


@pytest.fixture
def sample_affine_scaled():
    """
    Creates an affine matrix with 2mm isotropic voxels.

    Returns:
        np.ndarray: 4x4 affine matrix with 2mm scaling.
    """
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = 2.0
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0
    return affine


# ============================================================================
# NIfTI Fixtures
# ============================================================================


@pytest.fixture
def temp_nifti_file():
    """
    Creates a temporary NIfTI file with synthetic 3D data.

    Yields:
        str: Path to the temporary NIfTI file.
    """
    # Create synthetic 3D volume
    data = np.random.rand(32, 32, 32).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_path = f.name

    nib.save(img, temp_path)

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


# ============================================================================
# ODF / Spherical Harmonics Fixtures
# ============================================================================


@pytest.fixture
def sample_volume_shape():
    """
    Creates a standard 3D volume shape for testing.

    Returns:
        tuple: Shape (64, 64, 64).
    """
    return (64, 64, 64)


# ============================================================================
# Markers for Test Categories
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gui: marks tests as requiring GUI")
    config.addinivalue_line("markers", "numba: marks tests using AOT-compiled kernels")
