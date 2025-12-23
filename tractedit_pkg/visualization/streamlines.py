# -*- coding: utf-8 -*-

"""
Streamlines manager for TractEdit visualization.

Handles streamline visualization including actor creation, color mapping,
scalar bar management, and highlight actors for selected streamlines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import numpy as np
import vtk
from fury import actor, colormap
from vtk.util import numpy_support

from ..utils import ColorMode

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


class StreamlinesManager:
    """
    Manages streamline visualization operations.

    Handles actor creation, color mapping, scalar bar display,
    and highlight actors for selected streamlines.
    """

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        """
        Initialize the streamlines manager.

        Args:
            vtk_panel: Reference to the parent VTKPanel instance.
        """
        self.panel = vtk_panel

    def update_highlight(self) -> None:
        """Updates the actor for highlighted/selected streamlines."""
        if not self.panel.scene:
            return

        # Safely Remove Existing Highlight Actor
        actor_removed = False
        if self.panel.highlight_actor is not None:
            try:
                self.panel.scene.rm(self.panel.highlight_actor)
                actor_removed = True
            except (ValueError, AttributeError):
                actor_removed = True
            except Exception as e:
                logger.error(
                    f"  Error removing highlight actor: {e}. Proceeding cautiously."
                )
            finally:
                self.panel.highlight_actor = None

        # Check prerequisites
        if (
            not self.panel.main_window
            or not hasattr(self.panel.main_window, "selected_streamline_indices")
            or not self.panel.main_window.tractogram_data
        ):
            if self.panel.main_window:
                self.panel.main_window._update_action_states()
            return

        selected_indices: Set[int] = self.panel.main_window.selected_streamline_indices
        tractogram = self.panel.main_window.tractogram_data

        # Create new actor only if there's a valid selection
        if selected_indices:
            valid_indices = {
                idx for idx in selected_indices if 0 <= idx < len(tractogram)
            }

            # Create a concrete list of *non-empty* selected streamlines
            selected_sl_data_list = []
            for idx in valid_indices:
                try:
                    sl = tractogram[idx]
                    if sl is not None and len(sl) > 0:
                        selected_sl_data_list.append(sl.astype(np.float32, copy=False))
                except Exception:
                    pass  # Ignore if index fails

            if selected_sl_data_list:  # Check if the list is not empty
                try:
                    highlight_linewidth = 6
                    self.panel.highlight_actor = actor.line(
                        selected_sl_data_list,  # Pass the list
                        colors=(1, 1, 0),
                        linewidth=highlight_linewidth,
                        opacity=1.0,  # Fully opaque
                    )
                    self.panel.scene.add(self.panel.highlight_actor)

                    if (
                        self.panel.main_window
                        and not self.panel.main_window.bundle_is_visible
                    ):
                        self.panel.highlight_actor.SetVisibility(0)

                    if self.panel.highlight_actor:
                        try:
                            mapper = self.panel.highlight_actor.GetMapper()
                            if mapper:
                                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                                    -1, -15
                                )
                        except Exception as e:
                            logger.warning(
                                f"Warning: Could not apply mapper offset to highlight_actor: {e}"
                            )

                except Exception as e:
                    logger.error(f"Error creating highlight actor:", exc_info=True)
                    self.panel.highlight_actor = None

        # Update UI action states
        if self.panel.main_window:
            self.panel.main_window._update_action_states()

    def update_roi_highlight_actor(self) -> None:
        """
        Updates the Red highlight actor based on ROI 'Select' indices.
        """
        if not self.panel.scene:
            return

        # Remove existing
        if self.panel.roi_highlight_actor:
            try:
                self.panel.scene.rm(self.panel.roi_highlight_actor)
            except Exception:
                pass
            self.panel.roi_highlight_actor = None

        if not self.panel.main_window or not self.panel.main_window.tractogram_data:
            return

        indices = getattr(self.panel.main_window, "roi_highlight_indices", set())
        if not indices:
            return

        # Get actual streamline data
        tractogram = self.panel.main_window.tractogram_data
        streamlines_list = []
        for idx in indices:
            try:
                if idx < len(tractogram):
                    sl = tractogram[idx]
                    if len(sl) > 0:
                        streamlines_list.append(sl.astype(np.float32, copy=False))
            except Exception:
                pass

        if not streamlines_list:
            return

        try:
            # Create Red Actor (1, 0, 0)
            self.panel.roi_highlight_actor = actor.line(
                streamlines_list,
                colors=(1, 0, 0),
                linewidth=4,
                opacity=1.0,
                depth_cue=False,  # Make it pop out
            )
            self.panel.scene.add(self.panel.roi_highlight_actor)

            # Use mapper offset to ensure it draws ON TOP of the grey/yellow bundles
            mapper = self.panel.roi_highlight_actor.GetMapper()
            if mapper:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                    -2, -20
                )  # Stronger offset than yellow

            self.panel.render_window.Render()

        except Exception as e:
            logger.error(f"Error creating ROI highlight actor: {e}")

    def calculate_scalar_colors(
        self,
        streamlines_gen,
        scalar_gen,
        vmin: float,
        vmax: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates vertex colors based on a list of scalar arrays per streamline,
        using the provided vmin and vmax for the colormap range.
        Returns a single concatenated (TotalPoints, 3) numpy array for FURY.

        Args:
            streamlines_gen: Generator/iterable of streamline arrays.
            scalar_gen: Generator/iterable of scalar arrays (one per streamline).
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.

        Returns:
            Dictionary with 'colors', 'opacity', 'linewidth', 'lut' keys, or None.
        """

        # Create the LUT
        try:
            lut = vtk.vtkLookupTable()
            table_min = vmin - 0.5 if vmin == vmax else vmin
            table_max = vmax + 0.5 if vmin == vmax else vmax

            lut.SetTableRange(table_min, table_max)
            lut.SetHueRange(0.667, 0.0)  # Blue to Red (standard)
            lut.Build()
        except Exception as e:
            logger.error(f"Error creating scalar LUT: {e}. Defaulting to grey.")
            return None  # Fallback to grey

        # Build the list of color arrays (one per *non-empty* streamline)
        default_color_rgb = np.array([128, 128, 128], dtype=np.uint8)
        vertex_colors_list: List[np.ndarray] = []  # List to hold individual np.arrays
        rgb_output: List[float] = [0.0, 0.0, 0.0]

        for sl, sl_scalars in zip(streamlines_gen, scalar_gen):
            num_points = len(sl) if sl is not None else 0
            if num_points == 0:  # Skip empty streamlines
                continue

            sl_colors_rgb = np.empty((num_points, 3), dtype=np.uint8)

            # Check if this (non-empty) streamline has valid scalar data
            has_valid_scalar_for_this_sl = False
            if (
                sl_scalars is not None
                and hasattr(sl_scalars, "size")
                and len(sl_scalars) == num_points
            ):
                try:
                    for j in range(num_points):
                        lut.GetColor(sl_scalars[j], rgb_output)
                        sl_colors_rgb[j] = [int(c * 255) for c in rgb_output]
                    has_valid_scalar_for_this_sl = True
                except Exception:
                    has_valid_scalar_for_this_sl = False  # e.g., non-numeric data

            if not has_valid_scalar_for_this_sl:
                sl_colors_rgb[:] = default_color_rgb  # Fill with default color

            vertex_colors_list.append(sl_colors_rgb)

        if not vertex_colors_list:
            return None

        # Concatenate all color arrays into one big array
        try:
            concatenated_colors = np.concatenate(vertex_colors_list, axis=0)
        except ValueError as ve:
            logger.error(f"Failed to concatenate color arrays: {ve}")
            return None  # Fallback to grey

        return {
            "colors": concatenated_colors,
            "opacity": 0.8,
            "linewidth": 3,
            "lut": lut,
        }

    def update_scalar_bar(self, lut: vtk.vtkLookupTable, title: str) -> None:
        """Creates or updates the scalar bar actor with improved UX, title positioning, and precision."""
        if not self.panel.scene:
            return

        if self.panel.scalar_bar_actor is None:
            self.panel.scalar_bar_actor = vtk.vtkScalarBarActor()
            self.panel.scalar_bar_actor.SetOrientationToVertical()

            # Size and Positioning
            self.panel.scalar_bar_actor.SetWidth(0.04)
            self.panel.scalar_bar_actor.SetHeight(0.35)

            # Position: (X, Y) normalized viewport coordinates
            self.panel.scalar_bar_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            self.panel.scalar_bar_actor.SetPosition(0.94, 0.32)

            # Typography
            title_prop = vtk.vtkTextProperty()
            title_prop.SetColor(0.9, 0.9, 0.9)
            title_prop.SetFontSize(16)
            title_prop.SetFontFamilyToArial()
            title_prop.ItalicOn()
            title_prop.ShadowOn()
            title_prop.SetJustificationToRight()

            # Label Styling
            label_prop = vtk.vtkTextProperty()
            label_prop.SetColor(0.8, 0.8, 0.8)
            label_prop.SetFontSize(10)
            label_prop.SetFontFamilyToArial()
            label_prop.ShadowOff()

            self.panel.scalar_bar_actor.SetTitleTextProperty(title_prop)
            self.panel.scalar_bar_actor.SetLabelTextProperty(label_prop)

            # Formatting
            self.panel.scalar_bar_actor.SetNumberOfLabels(5)
            self.panel.scalar_bar_actor.SetLabelFormat("%-.4g")
            self.panel.scalar_bar_actor.UnconstrainedFontSizeOn()
            self.panel.scalar_bar_actor.SetTextPositionToPrecedeScalarBar()
            self.panel.scene.add(self.panel.scalar_bar_actor)

        # Update Data
        self.panel.scalar_bar_actor.SetLookupTable(lut)
        self.panel.scalar_bar_actor.SetTitle(title + "\n")
        self.panel.scalar_bar_actor.SetVisibility(1)
        self.panel.scalar_bar_actor.Modified()

    def get_streamline_actor_params(self) -> Dict[str, Any]:
        """
        Determines parameters for the main streamlines actor using STRIDE.
        Optimized to avoid double-copying data by using simple lists instead
        of rebuilding ArraySequence containers.

        Returns:
            Dictionary with streamline visualization parameters.
        """
        # Get opacity from MainWindow state
        current_opacity = getattr(self.panel.main_window, "bundle_opacity", 1.0)

        # Default parameters
        params: Dict[str, Any] = {
            "colors": (0.8, 0.8, 0.8),
            "opacity": current_opacity,
            "linewidth": 2,
        }

        if not self.panel.main_window or not self.panel.main_window.tractogram_data:
            return params

        tractogram = self.panel.main_window.tractogram_data

        # Get ALL Visible Indices
        visible_indices_list = sorted(list(self.panel.main_window.visible_indices))

        # Apply STRIDE (Skip Logic)
        stride = self.panel.main_window.render_stride
        subset_indices = visible_indices_list[::stride]  # Only pick 1 every N

        if not subset_indices:
            params["streamlines_list"] = []
            return params

        # Ensure float32 for VTK compatibility
        visible_streamlines_list = [
            tractogram[i].astype(np.float32, copy=False) for i in subset_indices
        ]

        # Coloring Logic
        current_mode = self.panel.main_window.current_color_mode

        if current_mode == ColorMode.ORIENTATION:
            params["colors"] = colormap.line_colors(visible_streamlines_list)
            params["opacity"] = current_opacity

        elif current_mode == ColorMode.SCALAR:
            active_scalar = self.panel.main_window.active_scalar_name
            scalar_seq = None
            is_per_streamline = False

            # Try generic dictionary
            if (
                self.panel.main_window.scalar_data_per_point
                and active_scalar in self.panel.main_window.scalar_data_per_point
            ):
                scalar_seq = self.panel.main_window.scalar_data_per_point[active_scalar]

            # Try TRX 'Data Per Vertex' (dpv)
            elif (
                hasattr(tractogram, "data_per_vertex")
                and active_scalar in tractogram.data_per_vertex
            ):
                scalar_seq = tractogram.data_per_vertex[active_scalar]

            # Try TRX 'Data Per Streamline' (dps)
            elif (
                hasattr(tractogram, "data_per_streamline")
                and active_scalar in tractogram.data_per_streamline
            ):
                scalar_seq = tractogram.data_per_streamline[active_scalar]
                is_per_streamline = True

            if scalar_seq is not None:
                try:
                    flat_scalars = None

                    if is_per_streamline:
                        # Extract values for the subset
                        subset_values = [scalar_seq[i] for i in subset_indices]

                        # Calculate lengths manually
                        lengths = [len(sl) for sl in visible_streamlines_list]

                        # Repeat value N times for each streamline
                        flat_scalars = np.repeat(subset_values, lengths)
                    else:
                        subset_arrays = [scalar_seq[i] for i in subset_indices]
                        flat_scalars = np.concatenate(subset_arrays)

                    # Vectorized Color Mapping
                    vmin = self.panel.main_window.scalar_min_val
                    vmax = self.panel.main_window.scalar_max_val

                    # Create LUT
                    lut = vtk.vtkLookupTable()
                    table_min = vmin - 0.5 if vmin == vmax else vmin
                    table_max = vmax + 0.5 if vmin == vmax else vmax
                    lut.SetTableRange(table_min, table_max)
                    lut.SetHueRange(0.667, 0.0)  # Blue to Red
                    lut.Build()

                    # Convert scalars to VTK array (explicit float32)
                    vtk_scalars = numpy_support.numpy_to_vtk(
                        flat_scalars.astype(np.float32),
                        deep=True,
                        array_type=vtk.VTK_FLOAT,
                    )

                    # Map Scalars (Returns vtkUnsignedCharArray)
                    vtk_colors = lut.MapScalars(
                        vtk_scalars, vtk.VTK_COLOR_MODE_DEFAULT, -1
                    )

                    # Convert back to Numpy and strip Alpha
                    colors_flat = numpy_support.vtk_to_numpy(vtk_colors)
                    colors_rgb = colors_flat[:, :3]

                    params["colors"] = colors_rgb
                    params["lut"] = lut
                    params["opacity"] = current_opacity

                except Exception as e:
                    logger.error(
                        f"Error during vectorized scalar calculation: {e}",
                        exc_info=True,
                    )
                    params["colors"] = (0.5, 0.5, 0.5)

        params["streamlines_list"] = visible_streamlines_list
        return params

    def update_main_streamlines_actor(self) -> None:
        """
        Recreates the main actor using the skipped/strided dataset.
        Handles both Line and Tube visualization.
        """
        try:
            if not self.panel.scene:
                return

            # Remove old actor
            if self.panel.streamlines_actor is not None:
                self.panel.scene.rm(self.panel.streamlines_actor)
                self.panel.streamlines_actor = None

            if not self.panel.main_window or not self.panel.main_window.tractogram_data:
                if self.panel.scalar_bar_actor:
                    self.panel.scalar_bar_actor.SetVisibility(0)
                self.panel.render_window.Render()
                return

            # Get Params (includes subsampled list)
            params = self.get_streamline_actor_params()
            sl_list = params.get("streamlines_list", [])

            current_mode = self.panel.main_window.current_color_mode
            active_scalar = self.panel.main_window.active_scalar_name

            if current_mode == ColorMode.SCALAR and "lut" in params and active_scalar:
                self.update_scalar_bar(params["lut"], active_scalar)
            else:
                if self.panel.scalar_bar_actor:
                    self.panel.scalar_bar_actor.SetVisibility(0)

            if not sl_list:
                self.panel.render_window.Render()
                return

            # Geometry Logic
            use_tubes = getattr(self.panel.main_window, "render_as_tubes", False)

            # Define visual properties
            colors = params.get("colors")
            opacity = params.get("opacity", 0.5)

            # Adaptive rendering for large datasets:
            # - Large tractograms (>threshold): Use thinner lines to reduce GPU load
            # - Small tractograms: Full quality with normal line width
            # Note: default LOD is disabled as it conflicts with DesiredUpdateRate-based
            #       fast render mode and causes visual artifacts
            LARGE_DATASET_THRESHOLD = 19000
            num_streamlines = len(sl_list)
            is_large_dataset = num_streamlines > LARGE_DATASET_THRESHOLD

            # Use thinner lines for large datasets to reduce GPU load
            if is_large_dataset:
                line_width = 1  # Thinner lines for performance
            else:
                line_width = params.get("linewidth", 2)  # Normal width

            try:
                if use_tubes:
                    # Tube Rendering
                    tube_radius = (
                        0.15 if not is_large_dataset else 0.10
                    )  # Thinner tubes for large datasets

                    self.panel.streamlines_actor = actor.streamtube(
                        sl_list,
                        colors=colors,
                        opacity=opacity,
                        linewidth=tube_radius,
                        lod=False,  # Explicitly disable LOD
                    )
                else:
                    # Line Rendering (Standard)
                    self.panel.streamlines_actor = actor.line(
                        sl_list,
                        colors=colors,
                        opacity=opacity,
                        linewidth=line_width,
                        lod=False,  # Explicitly disable LOD
                    )
            except Exception as e:
                logger.error(
                    f"Error creating streamline actor (Tubes={use_tubes}): {e}"
                )
                self.panel.update_status(f"Error rendering streamlines: {e}")
                return

            self.panel.scene.add(self.panel.streamlines_actor)

            # Apply visibility toggle
            if not self.panel.main_window.bundle_is_visible:
                self.panel.streamlines_actor.SetVisibility(0)

            self.update_highlight()
            self.panel.render_window.Render()

        except Exception as e:
            logger.error(f"Error in update_main_streamlines_actor: {e}", exc_info=True)
            self.panel.update_status("Error updating visualization.")
