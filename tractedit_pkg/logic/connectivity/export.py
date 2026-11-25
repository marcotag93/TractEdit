"""Transactional parcellation and connectivity exports."""

from __future__ import annotations

import csv
import logging
import os
from typing import Optional, Dict, List, Any

import numpy as np
import nibabel as nib

from ...reference_grid import mask_image
from .. import connectivity

logger = logging.getLogger(__name__)


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

    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"Failed to create connectivity visualization: {e}")
        return False


class ConnectivityExportMixin:
    """Export operations sharing the connectivity manager owner."""

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
            connectivity.QMessageBox.warning(mw, "No Parcellation", "No parcellation loaded.")
            return False

        region_name = mw.parcellation_labels.get(label, f"Region_{label}")

        # Create binary mask for this region
        region_mask = (mw.parcellation_data == label).astype(np.uint8)

        # Check if region has any voxels
        if not np.any(region_mask):
            connectivity.QMessageBox.warning(
                mw, "Empty Region", f"Region '{region_name}' has no voxels."
            )
            return False

        # Suggest filename
        safe_name = region_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        default_name = f"{safe_name}.nii.gz"

        save_path, _ = connectivity.QFileDialog.getSaveFileName(
            mw,
            f"Save Region: {region_name}",
            default_name,
            "NIfTI Files (*.nii *.nii.gz)",
        )

        if not save_path:
            return False

        try:
            image = mask_image(
                region_mask, mw.parcellation_affine, getattr(mw, "parcellation_spatial", None)
            )
            connectivity.transactional_save(save_path, lambda path: nib.save(image, path))
            mw.vtk_panel.update_status(f"Saved region '{region_name}' to: {save_path}")
            logger.info(f"Saved parcellation region {label} to {save_path}")
            return True

        except (OSError, ValueError) as e:
            logger.error(f"Error saving region: {e}", exc_info=True)
            connectivity.QMessageBox.critical(mw, "Save Error", f"Failed to save region:\n{e}")
            return False

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
            output_path, selected_filter = connectivity.QFileDialog.getSaveFileName(
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

            if ext != ".npy" and ext != ".csv":
                output_path += ".csv"

            base_path = os.path.splitext(output_path)[0]
            png_path = base_path + ".png"
            destinations = [output_path, png_path]
            metadata_path = None
            if ext == ".npy":
                metadata_path = base_path + "_labels.json"
                destinations.append(metadata_path)

            from pathlib import Path

            destination_paths = [Path(path) for path in destinations]
            with connectivity.staged_output_set(destination_paths) as staged:
                output_key = Path(output_path)
                if ext == ".npy":
                    with staged[output_key].open("wb") as stream:
                        np.save(stream, matrix)

                    import json

                    metadata = {
                        "labels": result["labels"],
                        "label_names": label_names,
                        "n_streamlines": result["n_streamlines"],
                        "n_regions": result["n_regions"],
                    }
                    with open(
                        staged[Path(metadata_path)], "w", encoding="utf-8"
                    ) as file:
                        json.dump(metadata, file, indent=2)
                else:
                    with open(staged[output_key], "w", encoding="utf-8", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["", *label_names])
                        for name, row in zip(label_names, matrix):
                            writer.writerow([name, *row])

                if not connectivity._create_connectivity_visualization(
                    matrix,
                    label_names,
                    str(staged[Path(png_path)]),
                    title=f"Connectivity Matrix ({result['n_regions']} regions)",
                ):
                    raise OSError("Failed to create connectivity visualization.")

            logger.info("Saved connectivity output set to %s", output_path)

            mw.vtk_panel.update_status(
                f"Connectivity matrix saved: {os.path.basename(output_path)} "
                f"({result['n_regions']} regions, {result['n_streamlines']} streamlines)"
            )

            return True

        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Error exporting connectivity matrix: {e}", exc_info=True)
            connectivity.QMessageBox.critical(
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
