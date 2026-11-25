from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_contracts import AnatomicalImageLoadResult

import os
import numpy as np
import nibabel as nib
import logging
from scipy.ndimage import gaussian_filter
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication
from ..workers import CancellableWorkerThread
from ..file_dialogs import choose_input
from ..reference_grid import (
    mask_spatial_metadata,
    ReferenceGrid,
    validate_volume_geometry as _validate_volume_geometry,
)
from typing import Any, List, Optional, Tuple
from .. import file_io

logger = logging.getLogger(__name__)


MAX_VOXELS = 512**3  # ~134M voxels - target max for display

# LRU cache size for memory-mapped slices (number of slices to cache)
MMAP_SLICE_CACHE_SIZE = 64


class MemoryMappedImage:
    """
    Memory-mapped NIfTI image wrapper for efficient on-demand slice extraction.

    Uses nibabel's memory-mapping to avoid loading the entire volume into RAM.
    Provides cached full-resolution slices for 2D panel display.
    Data is kept in canonical RAS+ orientation (from as_closest_canonical).
    """

    def __init__(self, img: nib.Nifti1Image, *, source_shape=None, volume_index=0, rgb=False):
        """
        Initialize memory-mapped image wrapper.

        Args:
            img: NiBabel image object (will be accessed via memory-mapping).
        """
        self._img = img
        self._affine = img.affine.copy()
        self._shape = img.shape[:3]
        self.source_shape = tuple(source_shape if source_shape is not None else img.shape)
        self.volume_index = volume_index
        self.rgb = rgb

        # Create cached slice getter
        self._get_slice_cached = self._create_cached_slice_getter()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the 3D shape of the image."""
        return self._shape

    @property
    def affine(self) -> np.ndarray:
        """Return the affine matrix."""
        return self._affine

    def _create_cached_slice_getter(self):
        """Create an LRU-cached slice getter function."""
        from functools import lru_cache

        @lru_cache(maxsize=MMAP_SLICE_CACHE_SIZE)
        def get_slice_cached(axis: str, index: int) -> np.ndarray:
            """
            Extract a single slice from the memory-mapped image.

            Args:
                axis: 'x' (sagittal), 'y' (coronal), or 'z' (axial).
                index: Slice index along the specified axis.

            Returns:
                2D numpy array (float32) of the slice data.
            """
            # Use dataobj for memory-mapped access (no full load)
            dataobj = self._img.dataobj

            if axis == "x":
                # Sagittal slice
                slice_data = np.asarray(dataobj[index, :, :])
            elif axis == "y":
                # Coronal slice
                slice_data = np.asarray(dataobj[:, index, :])
            elif axis == "z":
                # Axial slice
                slice_data = np.asarray(dataobj[:, :, index])
            else:
                raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'.")

            # Ensure contiguous float32 for VTK
            return np.ascontiguousarray(slice_data, dtype=np.float32)

        return get_slice_cached

    def get_slice(self, axis: str, index: int) -> np.ndarray:
        """
        Get a cached full-resolution slice.

        Args:
            axis: 'x' (sagittal), 'y' (coronal), or 'z' (axial).
            index: Slice index along the specified axis.

        Returns:
            2D numpy array (float32) of the slice data.
        """
        # Clamp index to valid range
        axis_map = {"x": 0, "y": 1, "z": 2}
        max_idx = self._shape[axis_map[axis]] - 1
        index = max(0, min(index, max_idx))
        return self._get_slice_cached(axis, index)

    def get_value_range(self) -> Tuple[float, float]:
        """
        Get the min/max value range by sampling the image.

        Uses a sampling approach to avoid loading the full volume.
        """
        # Sample slices at 25%, 50%, 75% through each axis
        samples = []
        for axis in ["x", "y", "z"]:
            axis_map = {"x": 0, "y": 1, "z": 2}
            size = self._shape[axis_map[axis]]
            for pct in [0.25, 0.5, 0.75]:
                idx = int(size * pct)
                slice_data = self.get_slice(axis, idx)
                samples.append(slice_data)

        all_samples = np.concatenate([s.ravel() for s in samples])
        finite_samples = all_samples[np.isfinite(all_samples)]

        if finite_samples.size > 0:
            return float(np.min(finite_samples)), float(np.max(finite_samples))
        return 0.0, 1.0

    def clear_cache(self):
        """Clear the slice cache."""
        self._get_slice_cached.cache_clear()


def _canonicalize_image(
    img: nib.Nifti1Image, input_path: str, status_updater: callable
) -> nib.Nifti1Image:
    """
    Reorient an image to the closest canonical voxel order without resampling.

    NiBabel reorders/flips the data proxy and composes the affine accordingly,
    so every voxel retains its original scanner-RAS position. Oblique rotation
    and shear are deliberately preserved for the rendering transform.
    """
    from ..reference_grid import canonicalize_nifti

    original_orientation = nib.aff2axcodes(img.affine)
    canonical = canonicalize_nifti(img)
    if nib.aff2axcodes(canonical.affine) != original_orientation:
        logger.info(
            "Image %s reoriented from %s to RAS+ without resampling.",
            input_path,
            original_orientation,
        )
    return canonical


def _maybe_downsample_image(
    img: nib.Nifti1Image,
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Checks if an image is too large and downsamples if needed.

    Data is kept in its canonical RAS+ orientation (from as_closest_canonical).
    Radiological display convention is handled at the rendering level.

    Args:
        img: NiBabel image object.
        progress_callback: Optional callback(percent, message) for progress updates.

    Returns:
        Tuple of (image_data, image_affine, was_downsampled).
    """
    shape = np.array(img.shape[:3])
    total_voxels = shape[0] * shape[1] * shape[2]

    if total_voxels <= file_io.MAX_VOXELS:
        # Image is small enough, no downsampling needed
        if progress_callback:
            progress_callback(50, "Loading image data...")
        return img.get_fdata(dtype=np.float32), img.affine.copy(), False

    # Image is too large - calculate striding step
    if progress_callback:
        progress_callback(
            30, f"Image too large ({shape[0]}×{shape[1]}×{shape[2]}), downsampling..."
        )

    # Calculate step to reach ~256³ voxels
    target_size = int(file_io.MAX_VOXELS ** (1 / 3))  # ~256
    step = max(1, int(np.ceil(max(shape) / target_size)))

    if progress_callback:
        progress_callback(40, "Loading display samples...")

    original_affine = img.affine.copy()

    if progress_callback:
        progress_callback(60, f"Downsampling with step={step}...")

    source_proxy = img.dataobj
    resampled_data = np.ascontiguousarray(
        source_proxy[::step, ::step, ::step],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(75, "Applying anti-aliasing filter...")

    # Anti-aliasing: Apply mild Gaussian blur AFTER downsampling
    sigma = (0.8, 0.8, 0.8, 0.0) if resampled_data.ndim == 4 else 0.8
    resampled_data = gaussian_filter(resampled_data, sigma=sigma, mode="nearest")

    # Adjust affine: multiply the voxel-step columns by the step size
    new_affine = original_affine.copy()
    new_affine[:3, :3] *= step

    if progress_callback:
        progress_callback(90, "Done resampling...")

    new_shape = resampled_data.shape[:3]
    logger.info(
        f"Downsampled image from {tuple(shape)} to {new_shape} " f"(step: {step})"
    )

    return resampled_data, new_affine, True


class AnatomicalImageLoaderThread(CancellableWorkerThread):
    """
    Background thread to load anatomical images without freezing the GUI.
    """

    progress = pyqtSignal(int, str)  # Signal to update progress bar (percent, message)
    finished = pyqtSignal(dict)  # Signal when loading is done
    error = pyqtSignal(str)  # Signal if an error occurs
    done = pyqtSignal()

    def __init__(self, input_path: str, *, volume_index=0, rgb=False):
        super().__init__()
        self.input_path = input_path
        self.volume_index = volume_index
        self.rgb = rgb
        self._result: Optional["AnatomicalImageLoadResult"] = None

    def take_result(
        self,
        result: "AnatomicalImageLoadResult",
    ) -> Optional["AnatomicalImageLoadResult"]:
        """Transfer an image result to its GUI consumer."""
        if result is not self._result:
            return None
        self._result = None
        return result

    def discard_result(self, result: "AnatomicalImageLoadResult") -> None:
        """Release an image result rejected by its GUI consumer."""
        self.complete_result()
        owned_result = self.take_result(result)
        if owned_result is not None:
            mmap_image = owned_result.get("mmap_image")
            if mmap_image is not None:
                try:
                    mmap_image.clear_cache()
                except (OSError, RuntimeError, AttributeError):
                    logger.warning("Failed to clear a rejected image result.")

    def discard_pending_result(self) -> None:
        """Release an emitted image result that was never consumed."""
        self.complete_result()
        if self._result is not None:
            mmap_image = self._result.get("mmap_image")
            if mmap_image is not None:
                try:
                    mmap_image.clear_cache()
                except (OSError, RuntimeError, AttributeError):
                    logger.warning("Failed to clear a pending image result.")
            self._result = None

    def run(self):
        mmap_image = None
        published = False
        try:
            self.progress.emit(10, "Loading NIfTI header...")

            # Load the NIfTI file (lazy - data not loaded yet)
            img = nib.load(self.input_path)
            if self.is_cancelled:
                return

            from ..image_volume import select_image_volume

            source_shape = img.shape
            img = select_image_volume(img, self.volume_index, self.rgb)

            img = _canonicalize_image(
                img, self.input_path, lambda msg: self.progress.emit(15, msg)
            )
            if self.is_cancelled:
                return

            # Get file size estimate for progress feedback
            header = img.header
            shape = header.get_data_shape()
            if len(shape) >= 3:
                total_voxels = shape[0] * shape[1] * shape[2]
                size_mb = (total_voxels * 4) / (1024 * 1024)  # float32 = 4 bytes
                self.progress.emit(
                    20,
                    f"Checking {shape[0]}×{shape[1]}×{shape[2]} ({size_mb:.0f} MB)...",
                )
            else:
                self.progress.emit(20, "Loading image data...")

            # Use auto-downsampling for large images
            def progress_callback(percent, message):
                self.progress.emit(percent, message)

            image_data, image_affine, was_downsampled = _maybe_downsample_image(
                img, progress_callback
            )
            if self.is_cancelled:
                return
            reference_grid = ReferenceGrid.from_nifti(
                img,
                provenance=f"anatomical:{self.input_path}",
            )

            self.progress.emit(85, "Creating memory-mapped accessor...")

            # Create memory-mapped image for full-resolution 2D slicing
            mmap_image = MemoryMappedImage(
                img, source_shape=source_shape, volume_index=self.volume_index, rgb=self.rgb
            )
            if self.is_cancelled:
                return

            self.progress.emit(90, "Validating...")

            _validate_volume_geometry(
                image_data,
                image_affine,
                "Anatomical image",
                allow_rgb=self.rgb,
            )

            status_msg = "Done"
            if was_downsampled:
                status_msg = f"Done (downsampled to {image_data.shape[0]}×{image_data.shape[1]}×{image_data.shape[2]})"

            self.progress.emit(100, status_msg)
            if self.is_cancelled:
                return
            self._result = {
                "data": image_data,
                "affine": image_affine,
                "path": self.input_path,
                "was_downsampled": was_downsampled,
                "mmap_image": mmap_image,
                "reference_grid": reference_grid,
            }
            self._result_pending = True
            self.finished.emit(self._result)
            published = True

        except FileNotFoundError:
            self._result_pending = True
            self.error.emit(f"File not found: {self.input_path}")
        except nib.filebasedimages.ImageFileError as e:
            self._result_pending = True
            self.error.emit(f"Invalid NIfTI file: {e}")
        except (OSError, ValueError, MemoryError) as e:
            self._result_pending = True
            self.error.emit(f"Error loading image: {type(e).__name__}: {e}")
        finally:
            if not published and mmap_image is not None:
                try:
                    mmap_image.clear_cache()
                except (OSError, AttributeError):
                    logger.debug("Failed to clear an unpublished image accessor.")
            self.done.emit()


def _status_updater(main_window):
    return getattr(
        main_window.vtk_panel, "update_status", lambda msg: logger.info("Status: %s", msg)
    )


def _update_vtk_and_ui_after_load(main_window: Any, status_msg: str) -> None:
    """
    Updates VTK panel and main window UI elements after loading.

    Args:
        main_window: Reference to the main window.
        status_msg: Status message to display.
    """
    if main_window.vtk_panel:
        if main_window.vtk_panel.scene and main_window.anatomical_image_data is None:
            main_window.vtk_panel.scene.reset_camera()
            main_window.vtk_panel.scene.reset_clipping_range()
        elif not main_window.vtk_panel.scene:
            logger.warning("vtk_panel.scene not available for camera reset.")

        main_window.vtk_panel.update_status(status_msg, render=False)

        if not main_window.vtk_panel.render_window:
            raise RuntimeError("Render window unavailable during bundle load.")
        main_window.vtk_panel.render_window.Render()
    else:
        raise RuntimeError("VTK panel unavailable during bundle load.")

    # Ensure radio button reflects default state if no scalars loaded
    if hasattr(main_window, "color_orientation_action"):
        main_window.color_orientation_action.setChecked(True)

    main_window._update_action_states()
    main_window._update_bundle_info_display()
    main_window._update_data_panel_display()  # Update tree widget after load


# ROI Image Loading Function
class _LoadedROI(tuple):
    """Keep three-value unpacking while carrying the source coordinate systems."""

    def __new__(cls, data, image, path):
        roi = super().__new__(cls, (data, image.affine, path))
        roi.spatial = mask_spatial_metadata(image)
        return roi


def load_roi_images(
    main_window: Any,
    file_paths: Optional[List[str]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Loads multiple NIfTI ROI files (.nii, .nii.gz).
    Returns a list of tuples, where each tuple is: (image data, affine matrix, file path).

    Args:
        main_window: The instance of the main application window.
        file_paths: Optional list of file paths to load. If None, opens a dialog.

    Returns:
        List[Tuple[...]]: A list containing data for all successfully loaded ROIs.
    """
    if not hasattr(main_window, "vtk_panel") or not main_window.vtk_panel.scene:
        logger.error("Error: Scene not initialized in vtk_panel.")
        file_io.QMessageBox.critical(main_window, "Error", "VTK Scene not initialized.")
        return []

    file_filter = "NIfTI Image Files (*.nii *.nii.gz);;All Files (*.*)"
    start_dir = ""
    if main_window.anatomical_image_path:
        start_dir = os.path.dirname(main_window.anatomical_image_path)
    elif main_window.original_trk_path:
        start_dir = os.path.dirname(main_window.original_trk_path)

    # getOpenFileNames to allow multiple selection
    if file_paths:
        input_paths = file_paths
    else:
        input_paths, _ = choose_input(
            file_io.QFileDialog.getOpenFileNames,
            main_window, "Select Input ROI Image File(s)", start_dir, file_filter
        )

    if not input_paths:
        status_updater = _status_updater(main_window)
        status_updater("ROI image load cancelled.")
        return []

    status_updater = _status_updater(main_window)

    status_updater = getattr(main_window.vtk_panel, "update_progress_status", status_updater)
    loaded_rois = []

    # Iterate through all selected paths
    for input_path in input_paths:
        status_updater(f"Loading ROI: {os.path.basename(input_path)}...")
        file_io.QApplication.processEvents()

        try:
            img = nib.load(input_path)

            _validate_volume_geometry(img.dataobj, img.affine, "ROI")

            img = _canonicalize_image(img, input_path, status_updater)

            # Apply proxy scaling exactly once, retaining the effective dtype.
            # Own a writable array for editing; never round or narrow values.
            image_data = np.asarray(img.dataobj).copy()
            image_affine = img.affine

            _validate_volume_geometry(image_data, image_affine, "ROI")

            loaded_rois.append(_LoadedROI(image_data, img, input_path))
            status_updater(f"Successfully loaded ROI: {os.path.basename(input_path)}")

        except (OSError, ValueError, MemoryError) as e:
            error_msg = f"Error loading {os.path.basename(input_path)}:\n{type(e).__name__}: {e}"
            logger.error(error_msg)
            file_io.QMessageBox.warning(main_window, "Load Error", error_msg)

    return loaded_rois
