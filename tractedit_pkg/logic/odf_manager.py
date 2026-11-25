# -*- coding: utf-8 -*-

"""
ODF manager for TractEdit.

Handles loading spherical-harmonic ODF volumes, computing and toggling the ODF
tunnel glyph view, and resetting ODF state on removal or bundle closure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..render_batch import batch_render, request_render
from ..utils import signals_blocked

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


def _mw_mod():
    """Return the main_window module so existing monkeypatch targets remain active."""
    from .. import main_window

    return main_window


class ODFManager:
    """Manages ODF SH data and tunnel glyph rendering for MainWindow."""

    def __init__(self, main_window: "MainWindow") -> None:
        self.mw = main_window

    def reset_odf_state(self) -> None:
        """Clear all ODF data fields and reset the ODF tunnel menu action."""
        mw = self.mw
        mw.odf_data = None
        mw.odf_affine = None
        mw.odf_path = None
        mw.odf_sh_order = 0
        mw.odf_sphere = None
        mw.odf_basis_matrix = None
        mw.odf_tunnel_sphere = None
        mw.odf_tunnel_basis = None
        mw.odf_tunnel_is_visible = False
        mw._session_deferred_odf = False
        mw._session_deferred_odf_owner = None
        if getattr(mw, "view_odf_tunnel_action", None) is not None:
            with signals_blocked(mw.view_odf_tunnel_action):
                mw.view_odf_tunnel_action.setChecked(False)
                mw.view_odf_tunnel_action.setEnabled(False)

    def trigger_load_odf(self) -> None:
        """Loads a NIfTI file as ODF spherical harmonic coefficients."""
        mw = self.mw
        mw_mod = _mw_mod()

        file_path, _ = mw_mod.choose_input(
            mw_mod.QFileDialog.getOpenFileName,
            mw,
            "Load ODF (SH Coefficients)",
            "",
            "NIfTI Files (*.nii *.nii.gz)",
        )
        if not file_path:
            return

        try:
            mw.vtk_panel.update_status("Loading ODF data...")
            mw_mod.QApplication.processEvents()

            img = mw_mod.nib.load(file_path)
            data = img.get_fdata()
            affine = img.affine

            # Validate Shape
            if data.ndim != 4:
                mw_mod.QMessageBox.warning(
                    mw, "ODF Error", "File must be a 4D volume (SH coefficients)."
                )
                return

            n_coeffs = data.shape[-1]
            try:
                sh_order = mw_mod.odf_utils.calculate_sh_order(n_coeffs)
            except ValueError as e:
                mw_mod.QMessageBox.warning(mw, "ODF Error", str(e))
                return

            mw.odf_data = data
            mw._session_deferred_odf = False
            mw.odf_affine = affine
            mw.odf_path = file_path
            mw.odf_sh_order = sh_order

            # Pre-compute Sphere and Basis (Tournier07)
            mw.odf_sphere = mw_mod.odf_utils.generate_symmetric_sphere(
                radius=1.0, subdivisions=3
            )
            mw.vtk_panel.update_status("Computing SH Basis...")
            mw_mod.QApplication.processEvents()

            mw.odf_basis_matrix = mw_mod.odf_utils.compute_sh_basis(
                mw.odf_sphere.vertices, sh_order, basis_type="tournier07"
            )

            # Lower-resolution sphere for lightweight tunnel rendering
            # (subdivisions=2 -> 162 vertices vs 642, ~4x fewer polygons)
            mw.odf_tunnel_sphere = mw_mod.odf_utils.generate_symmetric_sphere(
                radius=1.0, subdivisions=2
            )
            mw.odf_tunnel_basis = mw_mod.odf_utils.compute_sh_basis(
                mw.odf_tunnel_sphere.vertices, sh_order, basis_type="tournier07"
            )

            mw.vtk_panel.update_status(
                f"ODF Loaded (Order {sh_order}). Ready for Tunnel View."
            )
            mw._update_action_states()
            mw._update_data_panel_display()

        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Error loading ODF: {e}", exc_info=True)
            mw_mod.QMessageBox.critical(
                mw, "Load Error", f"Could not load ODF file:\n{e}"
            )

    def toggle_odf_tunnel(self, checked: bool) -> None:
        """Computes the mask and updates the VTK actor with progress indication."""
        mw = self.mw
        mw_mod = _mw_mod()

        if checked and getattr(mw, "_session_deferred_odf", False):
            mw._toggle_odf_tunnel_visibility(True)
            return
        if not checked:
            mw.odf_tunnel_is_visible = False
            if mw.vtk_panel:
                mw.vtk_panel.remove_odf_actor()
            mw._update_data_panel_display()
            return

        if mw.odf_data is None or mw.tractogram_data is None:
            return

        # Apply Stride (Skip) to match visual representation
        sorted_indices = sorted(list(mw.visible_indices))

        # Apply the current render stride
        stride = mw.render_stride
        strided_indices = sorted_indices[::stride]

        # Retrieve only the subset of streamlines
        current_streamlines = [mw.tractogram_data[i] for i in strided_indices]

        # Check Limit against the STRIDED count
        if len(current_streamlines) > mw.MAX_ODF_STREAMLINES:
            mw_mod.QMessageBox.warning(
                mw,
                "Performance Warning",
                f"Too many streamlines selected ({len(current_streamlines)}).\n"
                f"Limit is {mw.MAX_ODF_STREAMLINES}. \n\n"
                f"Tip: Increase the 'Skip %' or use ROIs to reduce the count.",
            )
            mw.view_odf_tunnel_action.setChecked(False)
            return

        mw.vtk_panel.update_status("Computing Tunnel View...")

        # Initialize Progress Bar (4 steps total)
        # 1. Mask Creation, 2. Mask Application, 3. SH Projection, 4. Rendering
        TOTAL_STEPS = 4
        mw.vtk_panel.update_progress_bar(0, TOTAL_STEPS, visible=True)
        mw_mod.QApplication.processEvents()

        try:
            tunnel_sphere = mw.odf_tunnel_sphere
            tunnel_basis = mw.odf_tunnel_basis

            mw.vtk_panel.update_progress_bar(1, TOTAL_STEPS, visible=True)
            mw_mod.QApplication.processEvents()

            odf_amplitudes, cropped_affine = mw_mod.odf_utils.build_tunnel_odf_amplitudes(
                mw.odf_data,
                current_streamlines,
                mw.odf_affine,
                tunnel_basis,
                dilation_iter=1,
            )

            mw.vtk_panel.update_progress_bar(2, TOTAL_STEPS, visible=True)
            mw_mod.QApplication.processEvents()

            extent = None
            if odf_amplitudes is not None:
                shape = odf_amplitudes.shape
                extent = (0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1)

            # Update Progress -> 75%
            mw.vtk_panel.update_progress_bar(3, TOTAL_STEPS, visible=True)
            mw_mod.QApplication.processEvents()

            mw.vtk_panel.update_odf_actor(
                odf_amplitudes, tunnel_sphere, cropped_affine, extent=extent
            )

            # Update Progress -> 100% and Hide
            mw.vtk_panel.update_progress_bar(TOTAL_STEPS, TOTAL_STEPS, visible=True)
            mw_mod.QApplication.processEvents()

            # Mark tunnel as visible and update data panel
            mw._odf_tunnel_indices = np.asarray(strided_indices, dtype=np.int64)
            mw._odf_tunnel_source = (
                id(mw.odf_data),
                id(mw.tractogram_data),
                np.asarray(mw.odf_affine, dtype=np.float64).tobytes(),
            )
            mw.odf_tunnel_is_visible = True
            mw._update_data_panel_display()

        except MemoryError:
            logger.error("Insufficient memory for ODF Tunnel View", exc_info=True)
            mw.vtk_panel.update_status("Insufficient memory for Tunnel View.")
            mw.view_odf_tunnel_action.setChecked(False)
            mw.odf_tunnel_is_visible = False
        except (RuntimeError, ValueError, IndexError, TypeError) as e:
            logger.error(f"Error computing Tunnel View: {e}", exc_info=True)
            mw.vtk_panel.update_status("Error generating Tunnel View.")
            mw.view_odf_tunnel_action.setChecked(False)
            mw.odf_tunnel_is_visible = False
        finally:
            mw.vtk_panel.update_progress_bar(0, 0, visible=False)

    def toggle_odf_tunnel_visibility(self, visible: bool) -> None:
        """
        Toggles the visibility of the ODF tunnel actor without recomputing.

        Args:
            visible: True to show, False to hide the ODF tunnel.
        """
        mw = self.mw
        if visible and getattr(mw, "_session_deferred_odf", False):
            from .session_manager import materialize_session_odf

            if not materialize_session_odf(mw):
                with signals_blocked(mw.view_odf_tunnel_action):
                    mw.view_odf_tunnel_action.setChecked(False)
                mw._update_data_panel_display()
                return
        if not mw.vtk_panel or not mw.vtk_panel.odf_actor:
            return

        try:
            mw.vtk_panel.odf_actor.SetVisibility(visible)
            mw.odf_tunnel_is_visible = visible

            # Sync the menu action state
            with signals_blocked(mw.view_odf_tunnel_action):
                mw.view_odf_tunnel_action.setChecked(visible)

            status = "shown" if visible else "hidden"
            mw.vtk_panel.update_status(f"ODF Tunnel {status}", render=False)
            if mw.vtk_panel.render_window:
                request_render(mw.vtk_panel, mw.vtk_panel.render_window)

        except (ValueError, RuntimeError) as e:
            logger.error(f"Error toggling ODF tunnel visibility: {e}", exc_info=True)

    @batch_render(lambda self: self.mw.vtk_panel)
    def remove_odf_data(self) -> None:
        """
        Removes the ODF data and tunnel actor from the scene.

        Clears all ODF-related data and updates the UI accordingly.
        """
        mw = self.mw
        try:
            if mw.vtk_panel:
                mw.vtk_panel.remove_odf_actor()

            self.reset_odf_state()

            mw._update_data_panel_display()
            mw._update_action_states()

            if mw.vtk_panel:
                mw.vtk_panel.update_status("ODF data removed")

        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Error removing ODF data: {e}", exc_info=True)
