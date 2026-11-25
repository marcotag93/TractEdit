# -*- coding: utf-8 -*-

"""
Image and ROI loader manager for TractEdit.

Centralizes background anatomical image loading (CLI startup, interactive
replacement, and 4D volume/RGB switching), ROI NIfTI loading and alignment,
and anatomical/ROI clearing with rollback protection.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from nibabel.orientations import ornt_transform
from PyQt6.QtCore import Qt

from ..render_batch import batch_render, defer_render, request_render
from ..utils import signals_blocked

if TYPE_CHECKING:
    from ..data_contracts import AnatomicalImageLoadResult
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)

_ROI_ROLLBACK_ATTRS = (
    "roi_layers",
    "roi_visibility",
    "roi_opacities",
    "roi_states",
    "roi_intersection_cache",
    "roi_highlight_indices",
    "visible_indices",
    "selected_streamline_indices",
    "_inversion_active",
    "_inversion_keeper_indices",
    "unified_undo_stack",
    "unified_redo_stack",
    "_visibility_version",
    "_skip_user_disabled",
    "render_stride",
    "_last_visibility_version",
    "_last_render_stride",
    "_last_color_mode",
    "_last_active_scalar",
    "_last_tube_mode",
    "_last_bundle_opacity",
)
_DRAWING_STATE_ATTRS = (
    "is_drawing_mode",
    "is_eraser_mode",
    "is_sphere_mode",
    "is_rectangle_mode",
    "current_drawing_roi",
)
_DRAWING_ACTION_ATTRS = (
    "draw_mode_action",
    "erase_mode_action",
    "sphere_mode_action",
    "rectangle_mode_action",
)
_DRAWING_BUTTON_ATTRS = (
    "draw_mode_button",
    "erase_mode_button",
    "sphere_mode_button",
    "rectangle_mode_button",
)


def _mw_mod():
    """Return the main_window module so existing monkeypatch targets remain active."""
    from .. import main_window

    return main_window


class ImageLoaderManager:
    """Manages anatomical image and ROI loading/clearing for MainWindow."""

    def __init__(self, main_window: "MainWindow") -> None:
        self.mw = main_window

    def trigger_load_anatomical_image(self) -> None:
        """Prompts the user for a NIfTI anatomical image and loads it asynchronously."""
        mw = self.mw
        mw_mod = _mw_mod()

        saved_image_path = mw.anatomical_image_path
        saved_trk_path = mw.original_trk_path

        file_filter = "NIfTI Image Files (*.nii *.nii.gz);;All Files (*.*)"
        start_dir = ""
        if saved_image_path:
            start_dir = os.path.dirname(saved_image_path)
        elif saved_trk_path:
            start_dir = os.path.dirname(saved_trk_path)

        input_path, _ = mw_mod.choose_input(
            mw_mod.QFileDialog.getOpenFileName,
            mw,
            "Select Input Anatomical Image File",
            start_dir,
            file_filter,
        )

        if not input_path:
            if mw.vtk_panel:
                mw.vtk_panel.update_status("Anatomical image load cancelled.")
            return

        if mw.anatomical_image_data is not None:
            # Custom message box to enforce "Yes" on the Left and "No" on the Right
            msg_box = mw_mod.QMessageBox(mw)
            msg_box.setWindowTitle("Replace Image?")
            msg_box.setText("An anatomical image is already loaded.\nReplace it?")
            msg_box.setIcon(mw_mod.QMessageBox.Icon.Question)

            # Use ActionRole to prevent platform-specific reordering
            yes_btn = msg_box.addButton("Yes", mw_mod.QMessageBox.ButtonRole.ActionRole)
            no_btn = msg_box.addButton("No", mw_mod.QMessageBox.ButtonRole.ActionRole)

            msg_box.setDefaultButton(no_btn)
            msg_box.exec()

            if msg_box.clickedButton() != yes_btn:
                return

        self.load_anatomical_file(input_path, initial_load=False)

    def load_anatomical_file(
        self,
        anat_path: str,
        *,
        initial_load: bool = False,
        bundle_path: Optional[str] = None,
        roi_paths: Optional[List[str]] = None,
        roi_in: Optional[List[List[float]]] = None,
        radius: Optional[List[float]] = None,
        anatomical_display: Optional[dict] = None,
    ) -> None:
        """Load an anatomical NIfTI image on a background thread with rollback.

        Supports CLI startup chaining (``initial_load=True``), interactive image
        replacement (``initial_load=False``), and 4D volume/RGB refresh
        (``anatomical_display`` set).
        """
        mw = self.mw
        mw_mod = _mw_mod()
        interactive_replace = not initial_load and anatomical_display is None

        # Setup modal progress dialog
        progress = mw_mod.QProgressDialog("Initializing...", "Cancel", 0, 100, mw)
        progress.setWindowTitle("Loading Image")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(350)
        manager = getattr(mw, "theme_manager", None)
        if manager is not None:
            progress.setStyleSheet(manager.get_progress_dialog_style())
        else:
            from ..ui.theme_manager import ThemeManager

            progress.setStyleSheet(ThemeManager.dialog_style(True))
        progress.setValue(0)
        progress.show()

        # Pass kwargs only when selecting a 4D volume/RGB display mode
        if anatomical_display is None:
            loader_thread = mw_mod.file_io.AnatomicalImageLoaderThread(anat_path)
        else:
            loader_thread = mw_mod.file_io.AnatomicalImageLoaderThread(
                anat_path, **anatomical_display
            )
        loader_thread.progress_dialog = progress
        generation = mw_mod.file_io._register_background_worker(
            mw,
            "_image_loader_thread",
            "_image_load_generation",
            loader_thread,
        )
        completed = False

        def on_progress(val, msg):
            if not mw_mod.file_io._worker_is_current(
                mw,
                "_image_loader_thread",
                "_image_load_generation",
                loader_thread,
                generation,
            ):
                return
            progress.setValue(val)
            progress.setLabelText(msg)

        def on_error(msg):
            nonlocal completed
            if not mw_mod.file_io._worker_is_current(
                mw,
                "_image_loader_thread",
                "_image_load_generation",
                loader_thread,
                generation,
            ):
                progress.close()
                loader_thread.complete_result()
                mw_mod.file_io._release_finished_worker(
                    mw, "_image_loader_thread", loader_thread
                )
                return
            completed = True
            loader_thread.begin_result()
            progress.close()
            if initial_load:
                logger.error(f"Error loading initial anatomical image: {msg}")
            mw_mod.QMessageBox.critical(mw, "Load Error", f"Error loading image:\n{msg}")
            if mw.vtk_panel:
                mw.vtk_panel.update_status("Error loading image.")
            loader_thread.complete_result()
            mw_mod.file_io._release_finished_worker(
                mw, "_image_loader_thread", loader_thread
            )
            if initial_load and anatomical_display is None:
                mw.load_initial_files(
                    bundle_path=bundle_path,
                    roi_paths=roi_paths,
                    roi_in=roi_in,
                    radius=radius,
                )

        def on_finished(data: "AnatomicalImageLoadResult") -> None:
            nonlocal completed
            if completed or not mw_mod.file_io._worker_is_current(
                mw,
                "_image_loader_thread",
                "_image_load_generation",
                loader_thread,
                generation,
            ):
                progress.close()
                discard = getattr(loader_thread, "discard_result", None)
                if callable(discard):
                    discard(data)
                mw_mod.file_io._release_finished_worker(
                    mw, "_image_loader_thread", loader_thread
                )
                return

            take_result = getattr(loader_thread, "take_result", None)
            if callable(take_result):
                data = take_result(data)
                if data is None:
                    loader_thread.complete_result()
                    mw_mod.file_io._release_finished_worker(
                        mw, "_image_loader_thread", loader_thread
                    )
                    return

            completed = True
            loader_thread.begin_result()
            previous_image = {
                "anatomical_image_data": mw.anatomical_image_data,
                "anatomical_image_affine": mw.anatomical_image_affine,
                "anatomical_image_path": mw.anatomical_image_path,
                "anatomical_mmap_image": mw.anatomical_mmap_image,
                "anatomical_reference_grid": mw.anatomical_reference_grid,
                "image_is_visible": mw.image_is_visible,
            }

            roi_state = {}
            drawing_state = {}
            drawing_actions = {}
            drawing_buttons = {}
            sphere_radius_visible = None
            if interactive_replace:
                for name in _ROI_ROLLBACK_ATTRS:
                    if hasattr(mw, name):
                        value = getattr(mw, name)
                        roi_state[name] = (
                            value.copy() if isinstance(value, (dict, set, list)) else value
                        )
                drawing_state = {
                    name: getattr(mw, name)
                    for name in _DRAWING_STATE_ATTRS
                    if hasattr(mw, name)
                }
                drawing_actions = {
                    name: getattr(mw, name).isChecked()
                    for name in _DRAWING_ACTION_ATTRS
                    if hasattr(mw, name)
                }
                drawing_buttons = {
                    name: getattr(mw, name).styleSheet()
                    for name in _DRAWING_BUTTON_ATTRS
                    if hasattr(mw, name)
                }
                sphere_radius_visible = (
                    mw.sphere_radius_container.isVisible()
                    if hasattr(mw, "sphere_radius_container")
                    else None
                )

            roi_clear_started = False
            drawing_reset_started = False
            committed = False
            try:
                progress.setLabelText("Creating slicer actors...")
                progress.setValue(95)
                mw_mod.QApplication.processEvents()
                if loader_thread.is_cancelled:
                    raise RuntimeError("Image load cancelled.")

                if anatomical_display is not None:
                    old_data = previous_image["anatomical_image_data"]
                    if (
                        old_data is None
                        or data["data"].shape[:3] != old_data.shape[:3]
                        or not np.array_equal(
                            data["affine"], previous_image["anatomical_image_affine"]
                        )
                        or data["mmap_image"].source_shape
                        != previous_image["anatomical_mmap_image"].source_shape
                    ):
                        raise ValueError(
                            "The image geometry changed on disk. Load it as a new image."
                        )

                mw.anatomical_image_data = data["data"]
                mw.anatomical_image_affine = data["affine"]
                mw.anatomical_image_path = data["path"]
                mw.anatomical_mmap_image = data.get("mmap_image")
                mw.anatomical_reference_grid = data.get("reference_grid")
                mw.image_is_visible = (
                    previous_image["image_is_visible"]
                    if anatomical_display is not None
                    else True
                )

                if mw.vtk_panel:
                    with defer_render(mw.vtk_panel):
                        if anatomical_display is None:
                            mw.vtk_panel.update_anatomical_slices()
                        else:
                            mw.vtk_panel.update_anatomical_slices(preserve_view=True)
                            if mw.vtk_panel.axial_slice_actor is None:
                                raise RuntimeError(
                                    "Could not render the selected volume."
                                )
                            mw.vtk_panel.set_anatomical_slice_visibility(
                                mw.image_is_visible
                            )
                        mw_mod.QApplication.processEvents()
                        if interactive_replace and loader_thread.is_cancelled:
                            raise RuntimeError("Image load cancelled.")
                        if mw.vtk_panel.scene and anatomical_display is None:
                            mw.vtk_panel.scene.reset_camera()
                            mw.vtk_panel.scene.reset_clipping_range()
                        mw.vtk_panel.update_status(
                            f"Loaded: {os.path.basename(data['path'])}"
                        )
                        if mw.vtk_panel.render_window:
                            request_render(mw.vtk_panel, mw.vtk_panel.render_window)

                mw._update_bundle_info_display()
                mw._update_action_states()
                if anatomical_display is not None:
                    mw.data_panel_manager.refresh_image_volume_label()
                elif interactive_replace:
                    mw._update_data_panel_display()
                    if mw.roi_layers:
                        roi_clear_started = True
                        mw._trigger_clear_all_rois(notify=False)
                    drawing_reset_started = True
                    mw._reset_all_drawing_modes()

                if loader_thread.is_cancelled:
                    raise RuntimeError("Image load cancelled.")

                progress.close()
                committed = True

            except Exception as e:
                was_cancelled = loader_thread.is_cancelled
                for name, value in previous_image.items():
                    setattr(mw, name, value)
                if roi_clear_started:
                    for name, value in roi_state.items():
                        setattr(mw, name, value)
                if roi_clear_started or drawing_reset_started:
                    for name, value in drawing_state.items():
                        setattr(mw, name, value)
                    for name, checked in drawing_actions.items():
                        try:
                            action = getattr(mw, name)
                            with signals_blocked(action):
                                action.setChecked(checked)
                        except Exception:
                            logger.debug("Failed to restore drawing action %s.", name)
                    for name, style in drawing_buttons.items():
                        try:
                            getattr(mw, name).setStyleSheet(style)
                        except Exception:
                            logger.debug("Failed to restore drawing button %s.", name)
                    if sphere_radius_visible is not None:
                        try:
                            mw.sphere_radius_container.setVisible(sphere_radius_visible)
                        except Exception:
                            logger.debug("Failed to restore sphere radius control.")
                    if mw.vtk_panel:
                        try:
                            mw.vtk_panel.set_drawing_mode(
                                drawing_state.get("is_drawing_mode", False),
                                is_eraser=drawing_state.get("is_eraser_mode", False),
                                is_sphere=drawing_state.get("is_sphere_mode", False),
                                is_rectangle=drawing_state.get(
                                    "is_rectangle_mode", False
                                ),
                            )
                        except Exception:
                            logger.debug("Failed to restore drawing mode.")
                if mw.vtk_panel:
                    try:
                        if mw.anatomical_image_data is None:
                            if not interactive_replace:
                                mw.vtk_panel.clear_anatomical_slices()
                        else:
                            if anatomical_display is None:
                                mw.vtk_panel.update_anatomical_slices()
                            else:
                                mw.vtk_panel.update_anatomical_slices(
                                    preserve_view=True
                                )
                                mw.vtk_panel.set_anatomical_slice_visibility(
                                    mw.image_is_visible
                                )
                    except Exception:
                        logger.warning("Failed to restore previous anatomical actors.")
                if roi_clear_started and mw.vtk_panel:
                    try:
                        mw.vtk_panel.clear_all_roi_layers()
                        for name, layer in mw.roi_layers.items():
                            mw.vtk_panel.add_roi_layer(
                                name, layer["data"], layer["affine"], render=False
                            )
                    except Exception:
                        logger.warning("Failed to restore previous ROI actors.")

                refresh_callbacks = (
                    (
                        mw._update_bundle_info_display,
                        mw._update_action_states,
                        mw._update_data_panel_display,
                    )
                    if interactive_replace
                    else (
                        mw._update_bundle_info_display,
                        mw._update_action_states,
                    )
                )
                for update in refresh_callbacks:
                    try:
                        update()
                    except Exception:
                        logger.debug("Failed to refresh previous image UI.")
                if anatomical_display is not None:
                    mw.data_panel_manager.refresh_image_volume_label()
                logger.error(f"Error finalizing image load: {e}", exc_info=True)
                try:
                    progress.close()
                except Exception:
                    logger.debug("Failed to close rejected image progress dialog.")
                if not was_cancelled:
                    try:
                        mw_mod.QMessageBox.critical(
                            mw, "Load Error", f"Error finalizing load:\n{e}"
                        )
                    except Exception:
                        logger.debug("Failed to display image load error.")
            finally:
                try:
                    rejected_mmap = data.get("mmap_image")
                    if (
                        not committed
                        and rejected_mmap is not None
                        and rejected_mmap is not previous_image["anatomical_mmap_image"]
                    ):
                        try:
                            rejected_mmap.clear_cache()
                        except Exception:
                            logger.warning("Failed to clear rejected image cache.")
                    if committed:
                        old_mmap = previous_image["anatomical_mmap_image"]
                        if (
                            old_mmap is not None
                            and old_mmap is not mw.anatomical_mmap_image
                        ):
                            try:
                                old_mmap.clear_cache()
                            except Exception:
                                logger.warning("Failed to clear previous image cache.")
                finally:
                    loader_thread.complete_result()
                    mw_mod.file_io._release_finished_worker(
                        mw, "_image_loader_thread", loader_thread
                    )

            if initial_load and anatomical_display is None:
                mw.load_initial_files(
                    bundle_path=bundle_path,
                    roi_paths=roi_paths,
                    roi_in=roi_in,
                    radius=radius,
                )

        def on_done():
            mw_mod.file_io._release_finished_worker(
                mw, "_image_loader_thread", loader_thread
            )

        # Connect signals (QueuedConnection ensures VTK calls run on main thread)
        loader_thread.progress.connect(on_progress)
        loader_thread.error.connect(on_error)
        loader_thread.finished.connect(
            on_finished, type=Qt.ConnectionType.QueuedConnection
        )
        progress.canceled.connect(loader_thread.cancel)
        if hasattr(loader_thread, "done"):
            loader_thread.done.connect(on_done, type=Qt.ConnectionType.QueuedConnection)

        loader_thread.start()

    def clear_anatomical_image(self, notify: bool = True) -> None:
        """Clears the currently loaded anatomical image."""
        mw = self.mw
        if mw.anatomical_image_data is None:
            return

        mw.anatomical_image_data = None
        mw.anatomical_image_affine = None
        mw.anatomical_image_path = None
        mw.anatomical_reference_grid = None
        if mw.anatomical_mmap_image:
            mw.anatomical_mmap_image.clear_cache()
        mw.anatomical_mmap_image = None
        mw.image_is_visible = True  # Reset visibility flag

        if mw.vtk_panel:
            mw.vtk_panel.clear_anatomical_slices()
            if notify:
                mw.vtk_panel.update_status("Anatomical image cleared.")

            # If no bundle is loaded, reset camera
            if not mw.tractogram_data and mw.vtk_panel.scene:
                mw.vtk_panel.scene.reset_camera()

        mw._update_bundle_info_display()
        mw._update_action_states()

    def _register_loaded_rois(
        self,
        loaded_rois: List[Tuple[np.ndarray, np.ndarray, str]],
        *,
        interactive: bool,
    ) -> None:
        """Reorient and register loaded ROI volumes onto the main window."""
        mw = self.mw
        mw_mod = _mw_mod()
        main_affine = mw.anatomical_image_affine
        target_ornt = (
            mw_mod.nib.io_orientation(main_affine)
            if (interactive or mw.anatomical_image_path)
            else None
        )

        for loaded_roi in loaded_rois:
            roi_data, roi_affine, roi_path = loaded_roi
            spatial = getattr(loaded_roi, "spatial", None)
            if roi_path in mw.roi_layers:
                if interactive:
                    mw_mod.QMessageBox.warning(
                        mw,
                        "ROI Already Loaded",
                        f"The ROI from '{os.path.basename(roi_path)}' is already loaded.",
                    )
                else:
                    logger.warning(
                        f"ROI '{os.path.basename(roi_path)}' is already loaded."
                    )
                continue

            mw.roi_visibility[roi_path] = True
            mw.roi_opacities[roi_path] = 0.5  # Default ROI opacity

            if interactive and mw.vtk_panel:
                mw.vtk_panel.update_progress_status(
                    f"Processing {os.path.basename(roi_path)}..."
                )
                mw_mod.QApplication.processEvents()

            try:
                if not interactive and not mw.anatomical_image_path:
                    logger.warning(
                        f"Skipping ROI {roi_path}: No anatomical image loaded for reorientation."
                    )
                    continue

                # Reuse the canonicalized, scaled ROI returned by the loader.
                # Reorient array and affine together without another file read.
                current_ornt = mw_mod.nib.io_orientation(roi_affine)
                if not np.array_equal(current_ornt, target_ornt):
                    transform = ornt_transform(current_ornt, target_ornt)
                    original_shape = roi_data.shape
                    roi_data = mw_mod.nib.orientations.apply_orientation(
                        roi_data, transform
                    )
                    voxel_transform = mw_mod.nib.orientations.inv_ornt_aff(
                        transform, original_shape
                    )
                    roi_affine = roi_affine @ voxel_transform
                    if spatial is not None:
                        spatial = dict(spatial)
                        for form in ("sform", "qform"):
                            spatial[form] = spatial[form] @ voxel_transform

                inv_affine = np.linalg.inv(roi_affine)
                T_main_to_roi = np.dot(inv_affine, main_affine)

                mw.roi_layers[roi_path] = {
                    "data": roi_data,
                    "affine": roi_affine,
                    "path": roi_path,
                    "spatial": spatial,
                    "inv_affine": inv_affine,
                    "T_main_to_roi": T_main_to_roi,
                }
                mw.roi_visibility[roi_path] = True

                if mw.vtk_panel:
                    if interactive:
                        mw.vtk_panel.add_roi_layer(roi_path, roi_data, roi_affine)
                        mw.vtk_panel.update_status(
                            f"Aligned and added {os.path.basename(roi_path)}."
                        )
                    else:
                        with defer_render(mw.vtk_panel):
                            mw.vtk_panel.add_roi_layer(roi_path, roi_data, roi_affine)
                            mw.vtk_panel.update_status(
                                f"Aligned and added {os.path.basename(roi_path)}."
                            )

            except FileNotFoundError as e:
                if interactive:
                    logger.error(f"File not found during processing: {e}")
                else:
                    logger.error(f"Error processing ROI {roi_path}: {e}")
                    mw_mod.QMessageBox.warning(
                        mw,
                        "ROI Load Error",
                        f"Failed to load ROI: {os.path.basename(roi_path)}\n{e}",
                    )
                continue
            except np.linalg.LinAlgError as e:
                if interactive:
                    mw_mod.QMessageBox.critical(
                        mw,
                        "Load Error",
                        f"Could not invert affine matrix for {os.path.basename(roi_path)}.",
                    )
                    if roi_path in mw.roi_visibility:
                        del mw.roi_visibility[roi_path]
                else:
                    logger.error(f"Error processing ROI {roi_path}: {e}")
                    mw_mod.QMessageBox.warning(
                        mw,
                        "ROI Load Error",
                        f"Failed to load ROI: {os.path.basename(roi_path)}\n{e}",
                    )
                continue
            except (OSError, ValueError, TypeError) as e:
                if interactive:
                    raise
                logger.error(f"Error processing ROI {roi_path}: {e}")
                mw_mod.QMessageBox.warning(
                    mw,
                    "ROI Load Error",
                    f"Failed to load ROI: {os.path.basename(roi_path)}\n{e}",
                )
                continue

    def load_roi_files(self, roi_paths: List[str]) -> None:
        """Load initial ROI files specified via CLI arguments."""
        mw = self.mw
        mw_mod = _mw_mod()
        valid_rois = [p for p in roi_paths if os.path.exists(p)]
        if not valid_rois:
            logger.warning("No valid ROI paths found in arguments.")
            return

        logger.info(f"Loading initial ROIs: {valid_rois}")
        loaded_rois = mw_mod.file_io.load_roi_images(mw, file_paths=valid_rois)
        self._register_loaded_rois(loaded_rois, interactive=False)
        mw._update_action_states()

    @batch_render(lambda self: self.mw.vtk_panel)
    def trigger_load_roi(self) -> None:
        """Triggers interactive loading of ROI image layer(s)."""
        mw = self.mw
        mw_mod = _mw_mod()

        if mw.anatomical_image_data is None:
            mw_mod.QMessageBox.warning(
                mw,
                "Load Error",
                "Please load a main anatomical image before adding an ROI layer.",
            )
            return

        if not mw.anatomical_image_path:
            mw_mod.QMessageBox.warning(
                mw,
                "Load Error",
                "Cannot find the path for the loaded anatomical image. Cannot re-orient.",
            )
            return

        loaded_rois = mw_mod.file_io.load_roi_images(mw)
        if not loaded_rois:
            return

        self._register_loaded_rois(loaded_rois, interactive=True)

        mw._update_bundle_info_display()
        mw._update_action_states()
        mw._update_data_panel_display()
        if mw.vtk_panel:
            mw.vtk_panel.update_status("ROI loading complete.")

    def clear_all_rois(self, notify: bool = True) -> None:
        """Clears all loaded ROI image layers and resets logic filters."""
        mw = self.mw
        if not mw.roi_layers:
            return

        mw._reset_all_drawing_modes()

        mw.roi_layers.clear()
        mw.roi_visibility.clear()

        mw.roi_states.clear()
        mw.roi_intersection_cache.clear()
        mw.roi_highlight_indices.clear()

        roi_mgr = getattr(mw, "roi_manager", None)
        if roi_mgr is not None:
            roi_mgr.update_roi_visual_selection()
            roi_mgr._apply_filters_with_skip_protection()

        if mw.vtk_panel:
            mw.vtk_panel.clear_all_roi_layers()
            if notify:
                mw.vtk_panel.update_status("All ROI layers and filters cleared.")

        mw._update_data_panel_display()
        mw._update_bundle_info_display()
        mw._update_action_states()
