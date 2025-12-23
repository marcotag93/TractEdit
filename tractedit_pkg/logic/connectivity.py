# -*- coding: utf-8 -*-

"""
Connectivity Manager for TractEdit application.

Handles structural connectivity matrix computation from streamlines
and FreeSurfer parcellation/segmentation volumes.

Features:
- Load FreeSurfer parcellation files (aparc+aseg, etc.)
- Compute endpoint-based connectivity matrices with Numba optimization
- Export matrices as CSV (with region names) or NPY format + PNG matrix image
- Parse FreeSurfer color LUT for region names
"""

from __future__ import annotations

import os
import time
import logging
from typing import TYPE_CHECKING, Optional, Dict, Tuple, List, Any

import numpy as np
from numba import njit, prange
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)
from PyQt6.QtCore import Qt
import nibabel as nib

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# Default FreeSurfer LUT location (can be overridden)
FREESURFER_LUT_PATHS = [
    os.path.join(os.environ.get("FREESURFER_HOME", ""), "FreeSurferColorLUT.txt"),
    os.path.expanduser("~/.freesurfer/FreeSurferColorLUT.txt"),
]

# Embedded FreeSurfer aparc+aseg labels (most common labels). This ensures labels are always available even without FreeSurfer installation
FREESURFER_BUILTIN_LABELS = {
    0: "Unknown",
    # Subcortical structures
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    72: "5th-Ventricle",
    77: "WM-hypointensities",
    78: "Left-WM-hypointensities",
    79: "Right-WM-hypointensities",
    80: "non-WM-hypointensities",
    81: "Left-non-WM-hypointensities",
    82: "Right-non-WM-hypointensities",
    85: "Optic-Chiasm",
    251: "CC_Posterior",
    252: "CC_Mid_Posterior",
    253: "CC_Central",
    254: "CC_Mid_Anterior",
    255: "CC_Anterior",
    # Desikan-Killiany atlas (aparc) - Left hemisphere (1000s)
    1000: "ctx-lh-unknown",
    1001: "ctx-lh-bankssts",
    1002: "ctx-lh-caudalanteriorcingulate",
    1003: "ctx-lh-caudalmiddlefrontal",
    1005: "ctx-lh-cuneus",
    1006: "ctx-lh-entorhinal",
    1007: "ctx-lh-fusiform",
    1008: "ctx-lh-inferiorparietal",
    1009: "ctx-lh-inferiortemporal",
    1010: "ctx-lh-isthmuscingulate",
    1011: "ctx-lh-lateraloccipital",
    1012: "ctx-lh-lateralorbitofrontal",
    1013: "ctx-lh-lingual",
    1014: "ctx-lh-medialorbitofrontal",
    1015: "ctx-lh-middletemporal",
    1016: "ctx-lh-parahippocampal",
    1017: "ctx-lh-paracentral",
    1018: "ctx-lh-parsopercularis",
    1019: "ctx-lh-parsorbitalis",
    1020: "ctx-lh-parstriangularis",
    1021: "ctx-lh-pericalcarine",
    1022: "ctx-lh-postcentral",
    1023: "ctx-lh-posteriorcingulate",
    1024: "ctx-lh-precentral",
    1025: "ctx-lh-precuneus",
    1026: "ctx-lh-rostralanteriorcingulate",
    1027: "ctx-lh-rostralmiddlefrontal",
    1028: "ctx-lh-superiorfrontal",
    1029: "ctx-lh-superiorparietal",
    1030: "ctx-lh-superiortemporal",
    1031: "ctx-lh-supramarginal",
    1032: "ctx-lh-frontalpole",
    1033: "ctx-lh-temporalpole",
    1034: "ctx-lh-transversetemporal",
    1035: "ctx-lh-insula",
    # Desikan-Killiany atlas (aparc) - Right hemisphere (2000s)
    2000: "ctx-rh-unknown",
    2001: "ctx-rh-bankssts",
    2002: "ctx-rh-caudalanteriorcingulate",
    2003: "ctx-rh-caudalmiddlefrontal",
    2005: "ctx-rh-cuneus",
    2006: "ctx-rh-entorhinal",
    2007: "ctx-rh-fusiform",
    2008: "ctx-rh-inferiorparietal",
    2009: "ctx-rh-inferiortemporal",
    2010: "ctx-rh-isthmuscingulate",
    2011: "ctx-rh-lateraloccipital",
    2012: "ctx-rh-lateralorbitofrontal",
    2013: "ctx-rh-lingual",
    2014: "ctx-rh-medialorbitofrontal",
    2015: "ctx-rh-middletemporal",
    2016: "ctx-rh-parahippocampal",
    2017: "ctx-rh-paracentral",
    2018: "ctx-rh-parsopercularis",
    2019: "ctx-rh-parsorbitalis",
    2020: "ctx-rh-parstriangularis",
    2021: "ctx-rh-pericalcarine",
    2022: "ctx-rh-postcentral",
    2023: "ctx-rh-posteriorcingulate",
    2024: "ctx-rh-precentral",
    2025: "ctx-rh-precuneus",
    2026: "ctx-rh-rostralanteriorcingulate",
    2027: "ctx-rh-rostralmiddlefrontal",
    2028: "ctx-rh-superiorfrontal",
    2029: "ctx-rh-superiorparietal",
    2030: "ctx-rh-superiortemporal",
    2031: "ctx-rh-supramarginal",
    2032: "ctx-rh-frontalpole",
    2033: "ctx-rh-temporalpole",
    2034: "ctx-rh-transversetemporal",
    2035: "ctx-rh-insula",
}


@njit(parallel=True, cache=True)
def _compute_endpoint_labels(
    start_points: np.ndarray,
    end_points: np.ndarray,
    inv_affine_3x3: np.ndarray,
    inv_affine_offset: np.ndarray,
    parcellation: np.ndarray,
    dims: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized computation of endpoint labels.

    Uses parallel loop over streamlines for maximum performance.

    Args:
        start_points: (N, 3) array of streamline start points in world coords.
        end_points: (N, 3) array of streamline end points in world coords.
        inv_affine_3x3: (3, 3) rotation/scale part of inverse affine.
        inv_affine_offset: (3,) translation part of inverse affine.
        parcellation: 3D label volume (int array).
        dims: (3,) volume dimensions.

    Returns:
        Tuple of (start_labels, end_labels), each (N,) int32 array.
    """
    n_streamlines = start_points.shape[0]
    start_labels = np.zeros(n_streamlines, dtype=np.int32)
    end_labels = np.zeros(n_streamlines, dtype=np.int32)

    for i in prange(n_streamlines):
        # Transform start point to voxel coordinates
        sx = (
            inv_affine_3x3[0, 0] * start_points[i, 0]
            + inv_affine_3x3[0, 1] * start_points[i, 1]
            + inv_affine_3x3[0, 2] * start_points[i, 2]
            + inv_affine_offset[0]
        )
        sy = (
            inv_affine_3x3[1, 0] * start_points[i, 0]
            + inv_affine_3x3[1, 1] * start_points[i, 1]
            + inv_affine_3x3[1, 2] * start_points[i, 2]
            + inv_affine_offset[1]
        )
        sz = (
            inv_affine_3x3[2, 0] * start_points[i, 0]
            + inv_affine_3x3[2, 1] * start_points[i, 1]
            + inv_affine_3x3[2, 2] * start_points[i, 2]
            + inv_affine_offset[2]
        )

        # Round to nearest voxel
        vx_s = int(np.round(sx))
        vy_s = int(np.round(sy))
        vz_s = int(np.round(sz))

        # Check bounds and get label
        if 0 <= vx_s < dims[0] and 0 <= vy_s < dims[1] and 0 <= vz_s < dims[2]:
            start_labels[i] = parcellation[vx_s, vy_s, vz_s]
        else:
            start_labels[i] = 0  # Unknown/outside

        # Transform end point to voxel coordinates
        ex = (
            inv_affine_3x3[0, 0] * end_points[i, 0]
            + inv_affine_3x3[0, 1] * end_points[i, 1]
            + inv_affine_3x3[0, 2] * end_points[i, 2]
            + inv_affine_offset[0]
        )
        ey = (
            inv_affine_3x3[1, 0] * end_points[i, 0]
            + inv_affine_3x3[1, 1] * end_points[i, 1]
            + inv_affine_3x3[1, 2] * end_points[i, 2]
            + inv_affine_offset[1]
        )
        ez = (
            inv_affine_3x3[2, 0] * end_points[i, 0]
            + inv_affine_3x3[2, 1] * end_points[i, 1]
            + inv_affine_3x3[2, 2] * end_points[i, 2]
            + inv_affine_offset[2]
        )

        # Round to nearest voxel
        vx_e = int(np.round(ex))
        vy_e = int(np.round(ey))
        vz_e = int(np.round(ez))

        # Check bounds and get label
        if 0 <= vx_e < dims[0] and 0 <= vy_e < dims[1] and 0 <= vz_e < dims[2]:
            end_labels[i] = parcellation[vx_e, vy_e, vz_e]
        else:
            end_labels[i] = 0  # Unknown/outside

    return start_labels, end_labels


def parse_freesurfer_lut(lut_path: Optional[str] = None) -> Dict[int, str]:
    """
    Parses a FreeSurfer color LUT file to extract label-to-name mapping.

    Priority: External LUT file > Built-in labels

    Args:
        lut_path: Path to FreeSurfer LUT file. If None, searches default paths.

    Returns:
        Dictionary mapping label IDs to region names.
    """
    # First, try to find and parse external LUT file (higher priority)
    if lut_path and os.path.isfile(lut_path):
        paths_to_try = [lut_path]
    else:
        paths_to_try = FREESURFER_LUT_PATHS

    lut_file = None
    for path in paths_to_try:
        if os.path.isfile(path):
            lut_file = path
            break

    # If external LUT found, parse it
    if lut_file is not None:
        label_names: Dict[int, str] = {0: "Unknown"}
        try:
            with open(lut_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            label_id = int(parts[0])
                            label_name = parts[1]
                            label_names[label_id] = label_name
                        except ValueError:
                            continue

            logger.info(
                f"Loaded {len(label_names)} labels from FreeSurfer LUT: {lut_file}"
            )
            return label_names
        except Exception as e:
            logger.warning(f"Failed to parse FreeSurfer LUT ({lut_file}): {e}")
            # Fall through to use built-in labels

    # Fallback to built-in labels
    logger.info("Using built-in FreeSurfer labels (no external LUT file found).")
    return FREESURFER_BUILTIN_LABELS.copy()


def _create_connectivity_visualization(
    matrix: np.ndarray,
    label_names: List[str],
    output_path: str,
    title: str = "Structural Connectivity Matrix",
) -> bool:
    """
    Creates a heatmap visualization of the connectivity matrix.

    Args:
        matrix: 2D numpy array of connectivity values.
        label_names: List of region names for axis labels.
        output_path: Path to save the PNG file.
        title: Title for the plot.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        logger.warning("Matplotlib not available. Skipping visualization.")
        return False

    try:
        n_regions = len(label_names)

        # Determine figure size based on number of regions
        if n_regions <= 20:
            figsize = (12, 10)
            fontsize = 8
            rotation = 45
        elif n_regions <= 50:
            figsize = (16, 14)
            fontsize = 6
            rotation = 90
        else:
            figsize = (20, 18)
            fontsize = 4
            rotation = 90

        fig, ax = plt.subplots(figsize=figsize)

        # Use log scale for better visualization of sparse matrices
        # Add 1 to avoid log(0), then use LogNorm
        matrix_plot = matrix.astype(float)
        matrix_plot[matrix_plot == 0] = np.nan  # Show zeros as white

        # Create heatmap
        im = ax.imshow(
            matrix_plot,
            cmap="YlOrRd",
            aspect="equal",
            interpolation="nearest",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Number of Streamlines", fontsize=10)

        # Set axis labels
        ax.set_xticks(np.arange(n_regions))
        ax.set_yticks(np.arange(n_regions))
        ax.set_xticklabels(
            label_names, fontsize=fontsize, rotation=rotation, ha="right"
        )
        ax.set_yticklabels(label_names, fontsize=fontsize)

        # Add title
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Add grid
        ax.set_xticks(np.arange(-0.5, n_regions, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_regions, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Saved connectivity visualization to {output_path}")
        return True

    except Exception as e:
        logger.warning(f"Failed to create connectivity visualization: {e}")
        return False


class ConnectivityManager:
    """
    Manages connectivity matrix computation for TractEdit.

    This class handles:
    - Loading FreeSurfer parcellation files
    - Computing endpoint-based structural connectivity matrices
    - Exporting matrices in various formats (CSV, NPY)
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the connectivity manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def load_parcellation(self, file_path: Optional[str] = None) -> bool:
        """
        Loads a FreeSurfer parcellation/segmentation NIfTI file.

        Args:
            file_path: Optional path to parcellation file. If None, opens dialog.

        Returns:
            True if loading was successful, False otherwise.
        """
        mw = self.mw

        # Get file path via dialog if not provided
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                mw,
                "Load FreeSurfer Parcellation",
                "",
                "NIfTI Files (*.nii *.nii.gz);;All Files (*)",
            )

        if not file_path:
            return False

        try:
            mw.vtk_panel.update_status("Loading parcellation...")
            QApplication.processEvents()

            # Load NIfTI
            img = nib.load(file_path)
            data = np.asarray(img.dataobj, dtype=np.int32)
            affine = img.affine.astype(np.float64)

            # Validate - should be 3D integer volume
            if data.ndim != 3:
                QMessageBox.warning(
                    mw,
                    "Parcellation Error",
                    "Parcellation file must be a 3D volume.",
                )
                return False

            # Store parcellation data
            mw.parcellation_data = data
            mw.parcellation_affine = affine
            mw.parcellation_path = file_path

            # Try to load label names from FreeSurfer LUT
            mw.parcellation_labels = parse_freesurfer_lut()

            # Add any labels found in parcellation but not in LUT
            unique_labels = np.unique(data)
            for label in unique_labels:
                if label not in mw.parcellation_labels:
                    mw.parcellation_labels[label] = f"Region_{label}"

            # Update UI
            n_regions = len(unique_labels)
            mw.vtk_panel.update_status(
                f"Parcellation loaded: {os.path.basename(file_path)} "
                f"({n_regions} regions)"
            )
            mw._update_action_states()
            mw._update_bundle_info_display()
            mw._update_data_panel_display()

            logger.info(f"Loaded parcellation: {file_path} with {n_regions} regions")
            return True

        except Exception as e:
            logger.error(f"Error loading parcellation: {e}", exc_info=True)
            QMessageBox.critical(
                mw, "Load Error", f"Could not load parcellation file:\n{e}"
            )
            return False

    def save_region_as_nifti(self, label: int) -> bool:
        """
        Saves a single parcellation region as a NIfTI file.

        Creates a binary mask where the selected region label is 1 and all
        other voxels are 0.

        Args:
            label: The parcellation region label ID to save.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None:
            QMessageBox.warning(mw, "No Parcellation", "No parcellation loaded.")
            return False

        region_name = mw.parcellation_labels.get(label, f"Region_{label}")

        # Create binary mask for this region
        region_mask = (mw.parcellation_data == label).astype(np.uint8)

        # Check if region has any voxels
        if not np.any(region_mask):
            QMessageBox.warning(
                mw, "Empty Region", f"Region '{region_name}' has no voxels."
            )
            return False

        # Suggest filename
        safe_name = region_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        default_name = f"{safe_name}.nii.gz"

        save_path, _ = QFileDialog.getSaveFileName(
            mw,
            f"Save Region: {region_name}",
            default_name,
            "NIfTI Files (*.nii *.nii.gz)",
        )

        if not save_path:
            return False

        try:
            nib.save(nib.Nifti1Image(region_mask, mw.parcellation_affine), save_path)
            mw.vtk_panel.update_status(f"Saved region '{region_name}' to: {save_path}")
            logger.info(f"Saved parcellation region {label} to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving region: {e}", exc_info=True)
            QMessageBox.critical(mw, "Save Error", f"Failed to save region:\n{e}")
            return False

    def remove_parcellation(self) -> None:
        """
        Completely removes all parcellation data and frees the cache.

        Clears all parcellation-related data structures and updates the UI.
        """
        mw = self.mw

        # Remove overlay actors from scene
        self._hide_parcellation_actors()

        # Clear cached actors to free memory
        if hasattr(mw, "parcellation_overlay_actor"):
            mw.parcellation_overlay_actor = None
        if hasattr(mw, "parcellation_region_actors"):
            mw.parcellation_region_actors = {}

        # Clear parcellation data
        mw.parcellation_data = None
        mw.parcellation_affine = None
        mw.parcellation_path = None
        mw.parcellation_labels = {}
        mw.parcellation_connected_labels = set()
        mw.parcellation_region_visibility = {}
        mw.parcellation_main_labels = set()
        mw.parcellation_label_colors = {}

        # Clear cache flags
        mw._parcellation_overlay_cached = False
        mw._parcellation_overlay_visible = False

        # Clear region filter data
        mw.parcellation_region_states = {}
        mw.parcellation_region_intersection_cache = {}
        mw.parcellation_start_labels = None
        mw.parcellation_end_labels = None
        mw.parcellation_visible_indices = None

        # Update menu action state
        if hasattr(mw, "view_parcellation_action"):
            mw.view_parcellation_action.setChecked(False)

        # Re-apply logic filters (removes any parcellation-based filtering)
        mw.roi_manager.apply_logic_filters()

        # Update UI
        mw._update_action_states()
        mw._update_data_panel_display()

        mw.vtk_panel.update_status("Parcellation removed and cache cleared.")
        logger.info("Parcellation removed and cache cleared")

    def compute_connectivity_matrix(self) -> Optional[Dict[str, Any]]:
        """
        Computes the structural connectivity matrix from visible streamlines.

        Uses streamline endpoints to determine connections between parcellation
        regions. The matrix is symmetric and includes self-connections.

        Returns:
            Dictionary with 'matrix', 'labels', 'label_names', 'counts' or None on error.
        """
        mw = self.mw

        # Validate data
        if mw.tractogram_data is None:
            QMessageBox.warning(
                mw, "No Streamlines", "Please load a streamline bundle first."
            )
            return None

        if mw.parcellation_data is None:
            QMessageBox.warning(
                mw, "No Parcellation", "Please load a parcellation file first."
            )
            return None

        if not mw.visible_indices:
            QMessageBox.warning(
                mw, "No Visible Streamlines", "No streamlines are currently visible."
            )
            return None

        try:
            # Setup progress dialog
            progress = QProgressDialog(
                "Computing connectivity matrix...", "Cancel", 0, 100, mw
            )
            progress.setWindowTitle("Connectivity Matrix")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QApplication.processEvents()

            # Extract visible streamlines endpoints
            visible_indices = list(mw.visible_indices)
            n_streamlines = len(visible_indices)

            progress.setLabelText(
                f"Extracting endpoints from {n_streamlines} streamlines..."
            )
            progress.setValue(10)
            QApplication.processEvents()

            if progress.wasCanceled():
                return None

            # Collect start and end points
            start_points = np.zeros((n_streamlines, 3), dtype=np.float64)
            end_points = np.zeros((n_streamlines, 3), dtype=np.float64)

            for i, idx in enumerate(visible_indices):
                sl = mw.tractogram_data[idx]
                if sl is not None and len(sl) >= 2:
                    start_points[i] = sl[0]
                    end_points[i] = sl[-1]

            progress.setLabelText("Computing endpoint labels with Numba...")
            progress.setValue(30)
            QApplication.processEvents()

            if progress.wasCanceled():
                return None

            # Compute inverse affine
            inv_affine = np.linalg.inv(mw.parcellation_affine)
            inv_affine_3x3 = inv_affine[:3, :3].astype(np.float64)
            inv_affine_offset = inv_affine[:3, 3].astype(np.float64)
            dims = np.array(mw.parcellation_data.shape, dtype=np.int64)

            # JIT compile on first run (may be a bit slower)
            start_labels, end_labels = _compute_endpoint_labels(
                start_points,
                end_points,
                inv_affine_3x3,
                inv_affine_offset,
                mw.parcellation_data,
                dims,
            )

            progress.setLabelText("Building connectivity matrix...")
            progress.setValue(70)
            QApplication.processEvents()

            if progress.wasCanceled():
                return None

            # Get unique labels (excluding 0 = Unknown unless specifically needed)
            all_labels = np.union1d(start_labels, end_labels)
            nonzero_labels = all_labels[all_labels > 0]

            if len(nonzero_labels) == 0:
                QMessageBox.warning(
                    mw,
                    "No Connections",
                    "No streamline endpoints fall within labeled regions.",
                )
                return None

            # Create label-to-index mapping for compact matrix
            label_to_idx = {label: idx for idx, label in enumerate(nonzero_labels)}
            n_labels = len(nonzero_labels)

            # Initialize connectivity matrix
            matrix = np.zeros((n_labels, n_labels), dtype=np.int32)

            # Populate matrix (vectorized)
            for i in range(n_streamlines):
                s_label = start_labels[i]
                e_label = end_labels[i]

                # Skip if either endpoint is in unknown region
                if s_label == 0 or e_label == 0:
                    continue

                s_idx = label_to_idx[s_label]
                e_idx = label_to_idx[e_label]

                # Increment symmetric matrix
                matrix[s_idx, e_idx] += 1
                if s_idx != e_idx:
                    matrix[e_idx, s_idx] += 1

            progress.setLabelText("Finalizing...")
            progress.setValue(90)
            QApplication.processEvents()

            # Get label names
            label_names = [
                mw.parcellation_labels.get(int(label), f"Region_{label}")
                for label in nonzero_labels
            ]

            # Compute connection counts per region
            connection_counts = np.sum(matrix, axis=1)

            progress.setValue(100)
            progress.close()

            result = {
                "matrix": matrix,
                "labels": nonzero_labels.tolist(),
                "label_names": label_names,
                "counts": connection_counts.tolist(),
                "n_streamlines": n_streamlines,
                "n_regions": n_labels,
            }

            logger.info(
                f"Computed connectivity matrix: {n_labels}x{n_labels} from {n_streamlines} streamlines"
            )

            return result

        except Exception as e:
            logger.error(f"Error computing connectivity matrix: {e}", exc_info=True)
            QMessageBox.critical(
                mw, "Computation Error", f"Failed to compute connectivity matrix:\n{e}"
            )
            return None

    def export_connectivity_matrix(
        self, result: Dict[str, Any], output_path: Optional[str] = None
    ) -> bool:
        """
        Exports the connectivity matrix to file.

        Args:
            result: Dictionary from compute_connectivity_matrix().
            output_path: Optional path. If None, opens save dialog.

        Returns:
            True if export was successful, False otherwise.
        """
        mw = self.mw

        if result is None:
            return False

        # Get output path via dialog
        if not output_path:
            output_path, selected_filter = QFileDialog.getSaveFileName(
                mw,
                "Save Connectivity Matrix",
                "",
                "CSV File (*.csv);;NumPy Array (*.npy);;All Files (*)",
            )

        if not output_path:
            return False

        try:
            matrix = result["matrix"]
            label_names = result["label_names"]

            # Determine format from extension
            ext = os.path.splitext(output_path)[1].lower()

            if ext == ".npy":
                # Save as NumPy binary
                np.save(output_path, matrix)

                # Also save metadata as JSON sidecar
                import json

                metadata_path = output_path.replace(".npy", "_labels.json")
                metadata = {
                    "labels": result["labels"],
                    "label_names": label_names,
                    "n_streamlines": result["n_streamlines"],
                    "n_regions": result["n_regions"],
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(
                    f"Saved connectivity matrix to {output_path} and {metadata_path}"
                )

            else:
                # Default to CSV format
                if not output_path.endswith(".csv"):
                    output_path += ".csv"

                # Write CSV with headers
                with open(output_path, "w", encoding="utf-8") as f:
                    # Header row
                    f.write("," + ",".join(label_names) + "\n")
                    # Data rows
                    for i, row in enumerate(matrix):
                        f.write(label_names[i] + "," + ",".join(map(str, row)) + "\n")

                logger.info(f"Saved connectivity matrix to {output_path}")

            # Generate PNG visualization with same base name
            base_path = os.path.splitext(output_path)[0]
            png_path = base_path + ".png"
            _create_connectivity_visualization(
                matrix,
                label_names,
                png_path,
                title=f"Connectivity Matrix ({result['n_regions']} regions)",
            )

            mw.vtk_panel.update_status(
                f"Connectivity matrix saved: {os.path.basename(output_path)} "
                f"({result['n_regions']} regions, {result['n_streamlines']} streamlines)"
            )

            return True

        except Exception as e:
            logger.error(f"Error exporting connectivity matrix: {e}", exc_info=True)
            QMessageBox.critical(
                mw, "Export Error", f"Failed to save connectivity matrix:\n{e}"
            )
            return False

    def compute_and_export(self) -> bool:
        """
        Convenience method to compute and export connectivity matrix in one step.

        Returns:
            True if successful, False otherwise.
        """
        result = self.compute_connectivity_matrix()
        if result is not None:
            return self.export_connectivity_matrix(result)
        return False

    def create_parcellation_overlay(self) -> bool:
        """
        Creates a 3D parcellation overlay with connected regions highlighted.

        Uses caching to avoid recomputation on toggle. Regions where streamlines
        terminate are colored distinctly.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None:
            QMessageBox.warning(
                mw, "No Parcellation", "Please load a parcellation file first."
            )
            return False

        if mw.tractogram_data is None:
            QMessageBox.warning(mw, "No Streamlines", "Please load streamlines first.")
            return False

        # If no visible streamlines (e.g. after ROI filtering), just show status - no popup
        if not mw.visible_indices:
            mw.vtk_panel.update_status(
                "No visible streamlines for parcellation overlay"
            )
            return False

        # Check if we have cached actors - just show them
        if (
            hasattr(mw, "_parcellation_overlay_cached")
            and mw._parcellation_overlay_cached
            and hasattr(mw, "parcellation_region_actors")
            and mw.parcellation_region_actors
        ):
            self._show_parcellation_actors()
            mw.vtk_panel.update_status(
                f"Parcellation overlay restored ({len(getattr(mw, 'parcellation_connected_labels', []))} regions)"
            )
            return True

        try:
            # Clean up any residual actors from previous computation to prevent crashes
            self._safe_cleanup_actors()

            TOTAL_STEPS = 5
            mw.vtk_panel.update_progress_bar(0, TOTAL_STEPS, visible=True)
            mw.vtk_panel.update_status(
                "Computing parcellation overlay (Step 1/5: Extracting endpoints)..."
            )
            QApplication.processEvents()

            # Vectorized extraction of endpoints
            visible_indices = np.array(list(mw.visible_indices), dtype=np.int64)
            n_streamlines = len(visible_indices)

            # Pre-allocate arrays
            start_points = np.zeros((n_streamlines, 3), dtype=np.float64)
            end_points = np.zeros((n_streamlines, 3), dtype=np.float64)

            # Vectorized extraction using numpy stacking. Get all streamlines as list first, then extract endpoints
            streamlines = [mw.tractogram_data[idx] for idx in visible_indices]

            for i, sl in enumerate(streamlines):
                if sl is not None and len(sl) >= 2:
                    start_points[i] = sl[0]
                    end_points[i] = sl[-1]

            # Numba optimized label extraction
            inv_affine = np.linalg.inv(mw.parcellation_affine)
            inv_affine_3x3 = inv_affine[:3, :3].astype(np.float64)
            inv_affine_offset = inv_affine[:3, 3].astype(np.float64)
            dims = np.array(mw.parcellation_data.shape, dtype=np.int64)

            start_labels, end_labels = _compute_endpoint_labels(
                start_points,
                end_points,
                inv_affine_3x3,
                inv_affine_offset,
                mw.parcellation_data,
                dims,
            )

            mw.vtk_panel.update_progress_bar(1, TOTAL_STEPS, visible=True)
            mw.vtk_panel.update_status(
                "Computing parcellation overlay (Step 2/5: Finding connected regions)..."
            )
            QApplication.processEvents()

            # Get unique connected labels (vectorized)
            all_endpoint_labels = np.concatenate([start_labels, end_labels])
            connected_labels = np.unique(all_endpoint_labels[all_endpoint_labels > 0])
            connected_set = set(connected_labels.tolist())

            logger.info(f"Found {len(connected_labels)} connected regions")
            mw.parcellation_connected_labels = connected_set

            # Store endpoint labels for region intersection computation (include/exclude filters)
            mw.parcellation_start_labels = start_labels
            mw.parcellation_end_labels = end_labels
            mw.parcellation_visible_indices = visible_indices.copy()

            # Vectorized color assignment. Generate colormap colors for all possible labels at once
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                cmap = plt.cm.get_cmap("tab20")
                use_cmap = True
            except ImportError:
                use_cmap = False

            # Create lookup table: label -> color index
            label_to_color_idx = {
                label: i % 20 for i, label in enumerate(connected_labels)
            }

            # Pre-compute all colors (vectorized)
            if use_cmap:
                color_array = np.array([cmap(i)[:3] for i in range(20)]) * 255
            else:
                color_array = np.zeros((20, 3))
                for i in range(20):
                    hue = (i * 0.618033988749895) % 1.0
                    color_array[i] = _hue_to_rgb(hue)
                color_array *= 255

            color_array = color_array.astype(np.uint8)

            # Store color mapping for actors
            label_colors = {}
            for label in connected_labels:
                idx = label_to_color_idx[label]
                label_colors[int(label)] = tuple(color_array[idx].tolist()) + (200,)

            # Store label colors on main window for on-demand actor creation
            mw.parcellation_label_colors = label_colors

            mw.vtk_panel.update_progress_bar(2, TOTAL_STEPS, visible=True)
            mw.vtk_panel.update_status(
                "Computing parcellation overlay (Step 3/5: Creating base contour)..."
            )
            QApplication.processEvents()

            # Create VTK actors
            from fury import actor

            # Only colored connected regions are shown.
            mw.parcellation_overlay_actor = None  # No background actor

            # Create colored contours for connected regions (limit for performance)
            mw.parcellation_region_actors = {}  # {label: actor}
            max_regions = min(len(connected_labels), 20)  # Reduced limit for stability

            # Compute all region sizes in ONE pass using bincount
            all_region_sizes = np.bincount(mw.parcellation_data.ravel())

            # Build region_sizes list from precomputed bincount
            region_sizes = [
                (label, all_region_sizes[label] if label < len(all_region_sizes) else 0)
                for label in connected_labels
            ]
            region_sizes.sort(key=lambda x: x[1], reverse=True)

            mw.vtk_panel.update_progress_bar(3, TOTAL_STEPS, visible=True)
            mw.vtk_panel.update_status(
                f"Computing parcellation overlay (Step 4/5: Creating {min(len(region_sizes), max_regions)} region contours)..."
            )
            QApplication.processEvents()

            for i, (label, size) in enumerate(region_sizes[:max_regions]):
                if size < 50:
                    continue

                color = label_colors[int(label)]
                rgb_color = (color[0] / 255, color[1] / 255, color[2] / 255)

                try:
                    # Create binary mask for this region
                    region_mask = (mw.parcellation_data == label).astype(np.uint8)

                    region_actor = actor.contour_from_roi(
                        region_mask,
                        affine=mw.parcellation_affine,
                        color=rgb_color,
                        opacity=0.85,  # High opacity for intersected regions
                    )
                    mw.parcellation_region_actors[int(label)] = region_actor
                except Exception as e:
                    logger.warning(f"Failed to create contour for label {label}: {e}")

            # Store set of labels with pre-created actors (main labels)
            mw.parcellation_main_labels = set(mw.parcellation_region_actors.keys())

            # Add all actors to scene, respecting visibility settings
            if mw.vtk_panel and mw.vtk_panel.scene:
                # Only add background actor if it exists
                if mw.parcellation_overlay_actor is not None:
                    mw.vtk_panel.scene.add(mw.parcellation_overlay_actor)

                # Get existing visibility settings (from previous session or hide all)
                region_visibility = getattr(mw, "parcellation_region_visibility", {})

                for label, region_actor in mw.parcellation_region_actors.items():
                    # Only add to scene if marked visible (default True for new regions)
                    if region_visibility.get(label, True):
                        mw.vtk_panel.scene.add(region_actor)
                mw.vtk_panel.render_window.Render()

            # Mark as cached
            mw._parcellation_overlay_cached = True
            mw._parcellation_overlay_visible = True

            mw.vtk_panel.update_progress_bar(TOTAL_STEPS, TOTAL_STEPS, visible=True)
            QApplication.processEvents()

            # Hide progress bar after short delay
            mw.vtk_panel.update_progress_bar(0, 0, visible=False)

            mw.vtk_panel.update_status(
                f"Parcellation overlay: {len(connected_labels)} connected regions "
                f"({len(mw.parcellation_region_actors)} displayed)"
            )

            # Update data panel to show connected regions
            try:
                mw._update_data_panel_display()
            except RuntimeError as e:
                if "has been deleted" in str(e):
                    logger.debug("Tree widget item deleted during update, retrying...")
                    # Small delay then retry
                    QApplication.processEvents()
                    try:
                        mw._update_data_panel_display()
                    except RuntimeError:
                        logger.warning("Could not update data panel display")
                else:
                    raise

            # Ensure VTK is ready before allowing interaction
            try:
                # Force multiple renders to ensure all actors are initialized
                mw.vtk_panel.render_window.Render()
                QApplication.processEvents()

                # Another render to ensure scene is stable
                mw.vtk_panel.render_window.Render()
                QApplication.processEvents()

                # Non-blocking delay for VTK state stabilization
                from PyQt6.QtCore import QTimer

                QTimer.singleShot(100, lambda: self._finalize_render(mw))
            except Exception:
                pass

            return True

        except Exception as e:
            mw.vtk_panel.update_progress_bar(0, 0, visible=False)
            logger.error(f"Error creating parcellation overlay: {e}", exc_info=True)
            QMessageBox.critical(
                mw, "Overlay Error", f"Failed to create parcellation overlay:\n{e}"
            )
            return False

    def _finalize_render(self, mw: "MainWindow") -> None:
        """
        Performs final render after non-blocking delay.

        Called by QTimer.singleShot to complete VTK state stabilization
        without blocking the UI thread.
        """
        try:
            if mw.vtk_panel and mw.vtk_panel.render_window:
                mw.vtk_panel.render_window.Render()
                QApplication.processEvents()
        except Exception:
            pass

    def _safe_cleanup_actors(self) -> None:
        """
        Safely cleans up any residual parcellation actors before recomputing.
        Prevents crashes from stale VTK state.
        """
        mw = self.mw

        if not mw.vtk_panel or not mw.vtk_panel.scene:
            return

        # Remove main parcellation actor from scene if it exists
        if hasattr(mw, "parcellation_overlay_actor") and mw.parcellation_overlay_actor:
            try:
                mw.vtk_panel.scene.rm(mw.parcellation_overlay_actor)
            except Exception:
                pass
            mw.parcellation_overlay_actor = None

        # Remove all region actors from scene
        if hasattr(mw, "parcellation_region_actors") and mw.parcellation_region_actors:
            for actor in list(mw.parcellation_region_actors.values()):
                try:
                    mw.vtk_panel.scene.rm(actor)
                except Exception:
                    pass
            mw.parcellation_region_actors = {}

        # Reset cache flags
        mw._parcellation_overlay_cached = False
        mw._parcellation_overlay_visible = False

        # Force VTK garbage collection
        try:
            mw.vtk_panel.render_window.Render()
        except Exception:
            pass

    def _show_parcellation_actors(self) -> None:
        """Shows cached parcellation actors (adds them back to scene)."""
        mw = self.mw
        if not mw.vtk_panel or not mw.vtk_panel.scene:
            return

        if hasattr(mw, "parcellation_overlay_actor") and mw.parcellation_overlay_actor:
            try:
                mw.vtk_panel.scene.add(mw.parcellation_overlay_actor)
            except Exception:
                pass

        # Only add region actors that are marked as visible
        if hasattr(mw, "parcellation_region_actors"):
            region_visibility = getattr(mw, "parcellation_region_visibility", {})
            for label, actor in mw.parcellation_region_actors.items():
                # Default to visible if not explicitly hidden
                if region_visibility.get(label, True):
                    try:
                        mw.vtk_panel.scene.add(actor)
                    except Exception:
                        pass

        mw.vtk_panel.render_window.Render()
        mw._parcellation_overlay_visible = True

    def _hide_parcellation_actors(self) -> None:
        """Hides parcellation actors (removes from scene but keeps cached)."""
        mw = self.mw
        if not mw.vtk_panel or not mw.vtk_panel.scene:
            return

        if hasattr(mw, "parcellation_overlay_actor") and mw.parcellation_overlay_actor:
            try:
                mw.vtk_panel.scene.rm(mw.parcellation_overlay_actor)
            except Exception:
                pass

        if hasattr(mw, "parcellation_region_actors"):
            for actor in mw.parcellation_region_actors.values():
                try:
                    mw.vtk_panel.scene.rm(actor)
                except Exception:
                    pass

        mw.vtk_panel.render_window.Render()
        mw._parcellation_overlay_visible = False

    def remove_parcellation_overlay(self) -> None:
        """
        Hides the parcellation overlay from the 3D scene.

        Keeps actors cached so the overlay can be quickly restored without
        recomputation. Use _clear_parcellation() to fully clear all data.
        """
        mw = self.mw

        # Hide actors from scene (keeping them cached)
        self._hide_parcellation_actors()

        mw._parcellation_overlay_visible = False

        mw.vtk_panel.update_status("Parcellation overlay hidden")

    def invalidate_parcellation_cache(self) -> None:
        """
        Invalidates the parcellation overlay cache.
        Call this when streamlines or parcellation data changes.
        """
        mw = self.mw

        # Remove actors from scene if visible
        self._hide_parcellation_actors()

        # Clear cached actors
        if hasattr(mw, "parcellation_overlay_actor"):
            mw.parcellation_overlay_actor = None
        if hasattr(mw, "parcellation_region_actors"):
            mw.parcellation_region_actors = {}

        mw._parcellation_overlay_cached = False
        mw._parcellation_overlay_visible = False

    def toggle_region_visibility(
        self, label: int, visible: bool, batch_mode: bool = False
    ) -> None:
        """
        Toggles visibility of an individual parcellation region.

        Supports on-demand actor creation for regions beyond the initial limit.

        Args:
            label: The parcellation label ID.
            visible: True to show, False to hide.
            batch_mode: If True, skip render and status update (for bulk operations).
        """
        ##TODO - this needs to be refactored a bit

        mw = self.mw

        if not hasattr(mw, "parcellation_region_actors"):
            mw.parcellation_region_actors = {}

        # Don't create actors if the main overlay is not visible
        overlay_visible = getattr(mw, "_parcellation_overlay_visible", False)

        # If toggling ON and no actor exists, create on-demand
        if visible and label not in mw.parcellation_region_actors:
            # Only create on-demand if the overlay is currently visible
            if not overlay_visible:
                logger.debug(
                    f"Skipping on-demand actor creation for {label} - overlay not visible"
                )
                return

            # Check if this is a connected region (has streamline endpoints)
            connected_labels = getattr(mw, "parcellation_connected_labels", set())
            if label in connected_labels:
                logger.info(f"Creating on-demand actor for region {label}")
                success = self._create_region_actor_on_demand(label)
                if not success:
                    logger.warning(f"Failed to create actor for region {label}")
                    return
            else:
                logger.debug(f"Region {label} is not a connected region")
                return

        # Now toggle visibility
        if label not in mw.parcellation_region_actors:
            logger.debug(f"Region {label} has no actor to toggle")
            return

        region_actor = mw.parcellation_region_actors[label]

        # Only manipulate scene if overlay is visible
        if overlay_visible and mw.vtk_panel and mw.vtk_panel.scene:
            try:
                if visible:
                    mw.vtk_panel.scene.add(region_actor)
                else:
                    mw.vtk_panel.scene.rm(region_actor)
                # Only render if not in batch mode (batch will render once at end)
                if not batch_mode:
                    mw.vtk_panel.render_window.Render()
            except Exception as e:
                logger.warning(f"Failed to toggle region {label} visibility: {e}")

        # Only update status if not in batch mode and overlay is visible
        if not batch_mode and overlay_visible:
            region_name = mw.parcellation_labels.get(label, f"Region_{label}")
            status = "shown" if visible else "hidden"
            mw.vtk_panel.update_status(f"{region_name} {status}")

    def _create_region_actor_on_demand(self, label: int) -> bool:
        """
        Creates a single region actor on-demand.

        Used for regions beyond the initial 20-region limit that are
        toggled ON by the user.

        Args:
            label: The parcellation label ID.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None:
            return False

        try:
            from fury import actor

            # Get color from stored label colors, or generate one
            label_colors = getattr(mw, "parcellation_label_colors", {})
            if label in label_colors:
                color = label_colors[label]
                rgb_color = (color[0] / 255, color[1] / 255, color[2] / 255)
            else:
                # Generate a color using golden ratio cycling
                connected_labels = list(
                    getattr(mw, "parcellation_connected_labels", set())
                )
                try:
                    idx = connected_labels.index(label)
                except ValueError:
                    idx = label  # Use label as index if not found
                hue = (idx * 0.618033988749895) % 1.0
                rgb = _hue_to_rgb(hue)
                rgb_color = tuple(rgb.tolist())

                # Store the color for future use
                if not hasattr(mw, "parcellation_label_colors"):
                    mw.parcellation_label_colors = {}
                mw.parcellation_label_colors[label] = (
                    int(rgb_color[0] * 255),
                    int(rgb_color[1] * 255),
                    int(rgb_color[2] * 255),
                    200,
                )

            # Create binary mask for this region
            region_mask = (mw.parcellation_data == label).astype(np.uint8)

            # Check region size
            size = np.sum(region_mask)
            if size < 50:
                logger.warning(f"Region {label} is too small ({size} voxels)")
                return False

            # Create contour actor
            region_actor = actor.contour_from_roi(
                region_mask,
                affine=mw.parcellation_affine,
                color=rgb_color,
                opacity=0.85,
            )

            # Store the actor
            mw.parcellation_region_actors[label] = region_actor

            logger.info(f"Created on-demand actor for region {label}")
            return True

        except Exception as e:
            logger.error(f"Error creating on-demand actor for region {label}: {e}")
            return False

    def compute_region_intersection(self, label: int) -> bool:
        """
        Computes which streamlines pass through a parcellation region.

        Uses cached endpoint labels from create_parcellation_overlay() for fast lookup.
        Stores result in parcellation_region_intersection_cache.

        Args:
            label: The parcellation region label ID.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None or mw.tractogram_data is None:
            return False

        # Check if we have cached endpoint labels
        start_labels = getattr(mw, "parcellation_start_labels", None)
        end_labels = getattr(mw, "parcellation_end_labels", None)
        visible_indices = getattr(mw, "parcellation_visible_indices", None)

        if start_labels is None or end_labels is None or visible_indices is None:
            logger.warning(
                "Endpoint labels not cached. Please enable parcellation overlay first."
            )
            return False

        # Initialize cache if not exists
        if not hasattr(mw, "parcellation_region_intersection_cache"):
            mw.parcellation_region_intersection_cache = {}

        try:
            # Find streamlines where either start or end is in this region (Vectorized)
            matches_start = start_labels == label
            matches_end = end_labels == label
            matches = matches_start | matches_end

            # Get indices of matching streamlines (relative to visible_indices)
            matching_relative_indices = np.where(matches)[0]

            # Convert to actual tractogram indices
            matching_absolute_indices = set(
                int(visible_indices[i]) for i in matching_relative_indices
            )

            mw.parcellation_region_intersection_cache[label] = matching_absolute_indices

            region_name = mw.parcellation_labels.get(label, f"Region_{label}")
            logger.info(
                f"Region {region_name}: {len(matching_absolute_indices)} streamlines"
            )

            return True

        except Exception as e:
            logger.error(f"Error computing region intersection for {label}: {e}")
            return False

    def recalculate_all_intersections(self) -> bool:
        """
        Recalculates all parcellation intersections based on current visible streamlines.

        This clears the endpoint labels cache, recomputes them from current visible
        streamlines, clears the intersection cache, and re-applies any active filters.

        Returns:
            True if successful, False otherwise.
        """
        mw = self.mw

        if mw.parcellation_data is None or mw.tractogram_data is None:
            mw.vtk_panel.update_status("No parcellation or streamlines loaded")
            return False

        if not mw.visible_indices:
            mw.vtk_panel.update_status("No visible streamlines for intersection")
            return False

        try:
            mw.vtk_panel.update_status("Recalculating parcellation intersections...")
            QApplication.processEvents()

            # Clear existing caches
            mw.parcellation_region_intersection_cache = {}

            # Recompute endpoint labels from current visible streamlines
            visible_indices = np.array(list(mw.visible_indices), dtype=np.int64)
            n_streamlines = len(visible_indices)

            # Pre-allocate arrays
            start_points = np.zeros((n_streamlines, 3), dtype=np.float64)
            end_points = np.zeros((n_streamlines, 3), dtype=np.float64)

            # Extract endpoints
            streamlines = [mw.tractogram_data[idx] for idx in visible_indices]

            for i, sl in enumerate(streamlines):
                if sl is not None and len(sl) >= 2:
                    start_points[i] = sl[0]
                    end_points[i] = sl[-1]

            # Numba optimized label extraction
            inv_affine = np.linalg.inv(mw.parcellation_affine)
            inv_affine_3x3 = inv_affine[:3, :3].astype(np.float64)
            inv_affine_offset = inv_affine[:3, 3].astype(np.float64)
            dims = np.array(mw.parcellation_data.shape, dtype=np.int64)

            start_labels, end_labels = _compute_endpoint_labels(
                start_points,
                end_points,
                inv_affine_3x3,
                inv_affine_offset,
                mw.parcellation_data,
                dims,
            )

            # Update cached endpoint labels
            mw.parcellation_start_labels = start_labels
            mw.parcellation_end_labels = end_labels
            mw.parcellation_visible_indices = visible_indices.copy()

            # Update connected labels set
            all_endpoint_labels = np.concatenate([start_labels, end_labels])
            connected_labels = np.unique(all_endpoint_labels[all_endpoint_labels > 0])
            mw.parcellation_connected_labels = set(connected_labels.tolist())

            # Clear all region filter states (reset include/exclude)
            mw.parcellation_region_states = {}

            # Re-apply logic filters to restore full streamline visibility
            mw.roi_manager.apply_logic_filters()

            # Invalidate existing actor cache and recreate the overlay
            self.invalidate_parcellation_cache()

            # Recreate the visual overlay with new actors
            self.create_parcellation_overlay()

            mw.vtk_panel.update_status(
                f"Parcellation intersections recalculated ({len(connected_labels)} regions, "
                f"{n_streamlines} streamlines)"
            )

            # Update data panel
            mw._update_data_panel_display()

            logger.info(
                f"Recalculated parcellation intersections: {len(connected_labels)} regions, "
                f"{n_streamlines} streamlines"
            )
            return True

        except Exception as e:
            logger.error(f"Error recalculating intersections: {e}", exc_info=True)
            mw.vtk_panel.update_status(f"Error recalculating intersections: {e}")
            return False

    def set_region_logic_mode(self, label: int, mode: str) -> None:
        """
        Sets the logic mode for a parcellation region.

        Modes: 'none', 'include', 'exclude'
        - Include: Applies to THIS specific region (cumulative with others)
        - Exclude: Applies to THIS specific region (can switch from include)
        - None: Clears filter for THIS specific region only

        Args:
            label: The parcellation region label ID.
            mode: The logic mode to set.
        """
        mw = self.mw

        # Only allow filter operations when parcellation overlay is visible
        overlay_visible = getattr(mw, "_parcellation_overlay_visible", False)
        if not overlay_visible:
            mw.vtk_panel.update_status(
                "Enable parcellation overlay first to use region filters"
            )
            return

        # Initialize state dict if not exists
        if not hasattr(mw, "parcellation_region_states"):
            mw.parcellation_region_states = {}

        # Initialize intersection cache if not exists
        if not hasattr(mw, "parcellation_region_intersection_cache"):
            mw.parcellation_region_intersection_cache = {}

        region_name = mw.parcellation_labels.get(label, f"Region_{label}")

        if mode == "none":
            # Clear filter for THIS specific region only
            if label in mw.parcellation_region_states:
                del mw.parcellation_region_states[label]
            mw.vtk_panel.update_status(f"{region_name}: filter removed")

        elif mode == "include":
            if label not in mw.parcellation_region_states:
                mw.parcellation_region_states[label] = {
                    "include": False,
                    "exclude": False,
                }

            # Set this region to include (clear exclude if set)
            mw.parcellation_region_states[label]["include"] = True
            mw.parcellation_region_states[label]["exclude"] = False

            # Compute intersection if not cached
            if label not in mw.parcellation_region_intersection_cache:
                success = self.compute_region_intersection(label)
                if not success:
                    mw.parcellation_region_states[label]["include"] = False
                    return

            mw.vtk_panel.update_status(f"{region_name}: set as INCLUDE")

        elif mode == "exclude":
            if label not in mw.parcellation_region_states:
                mw.parcellation_region_states[label] = {
                    "include": False,
                    "exclude": False,
                }

            # Set this region to exclude (clear include if set)
            mw.parcellation_region_states[label]["include"] = False
            mw.parcellation_region_states[label]["exclude"] = True

            # Compute intersection if not cached
            if label not in mw.parcellation_region_intersection_cache:
                success = self.compute_region_intersection(label)
                if not success:
                    mw.parcellation_region_states[label]["exclude"] = False
                    return

            mw.vtk_panel.update_status(f"{region_name}: set as EXCLUDE")

        # Apply filters
        mw.roi_manager.apply_logic_filters()

        # Update data panel with signals blocked to prevent cascading callbacks
        if mw.data_tree_widget:
            mw.data_tree_widget.blockSignals(True)
            mw._update_data_panel_display()
            mw.data_tree_widget.blockSignals(False)


def _hue_to_rgb(hue: float) -> np.ndarray:
    """Convert hue (0-1) to RGB (0-1). Simple HSV to RGB with S=V=1."""
    if hue < 1 / 6:
        return np.array([1.0, hue * 6, 0.0])
    elif hue < 2 / 6:
        return np.array([(2 / 6 - hue) * 6, 1.0, 0.0])
    elif hue < 3 / 6:
        return np.array([0.0, 1.0, (hue - 2 / 6) * 6])
    elif hue < 4 / 6:
        return np.array([0.0, (4 / 6 - hue) * 6, 1.0])
    elif hue < 5 / 6:
        return np.array([(hue - 4 / 6) * 6, 0.0, 1.0])
    else:
        return np.array([1.0, 0.0, (1.0 - hue) * 6])
