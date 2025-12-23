# -*- coding: utf-8 -*-

"""
Selection manager for TractEdit visualization.

Handles streamline selection operations including sphere-based and
box-based streamline finding using vectorized bounding box checks
followed by precise geometric checks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Set

import numpy as np
import vtk
from numba import njit

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


@njit(nogil=True)
def _check_streamline_sphere_intersection(
    streamline: np.ndarray,
    center: np.ndarray,
    radius_sq: float,
) -> bool:
    """
    Numba-optimized check if a streamline intersects a sphere.

    Checks both vertex distances and segment-to-center distances.

    Args:
        streamline: (N, 3) array of streamline points.
        center: (3,) sphere center.
        radius_sq: Squared radius of sphere.

    Returns:
        True if streamline intersects sphere.
    """
    n_pts = streamline.shape[0]
    if n_pts == 0:
        return False

    # First check vertices (fast path)
    for i in range(n_pts):
        dx = streamline[i, 0] - center[0]
        dy = streamline[i, 1] - center[1]
        dz = streamline[i, 2] - center[2]
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < radius_sq:
            return True

    # Check segments
    for i in range(n_pts - 1):
        # Segment from p1 to p2
        p1x, p1y, p1z = streamline[i, 0], streamline[i, 1], streamline[i, 2]
        p2x, p2y, p2z = streamline[i + 1, 0], streamline[i + 1, 1], streamline[i + 1, 2]

        # Segment vector
        seg_x = p2x - p1x
        seg_y = p2y - p1y
        seg_z = p2z - p1z

        # Vector from p1 to center
        pc_x = center[0] - p1x
        pc_y = center[1] - p1y
        pc_z = center[2] - p1z

        # Segment length squared
        seg_len_sq = seg_x * seg_x + seg_y * seg_y + seg_z * seg_z

        if seg_len_sq == 0:
            continue

        # Project center onto segment line
        t = (pc_x * seg_x + pc_y * seg_y + pc_z * seg_z) / seg_len_sq

        # Clamp to segment
        if t < 0:
            t = 0.0
        elif t > 1:
            t = 1.0

        # Closest point on segment
        closest_x = p1x + t * seg_x
        closest_y = p1y + t * seg_y
        closest_z = p1z + t * seg_z

        # Distance to center
        dx = closest_x - center[0]
        dy = closest_y - center[1]
        dz = closest_z - center[2]
        dist_sq = dx * dx + dy * dy + dz * dz

        if dist_sq < radius_sq:
            return True

    return False


class SelectionManager:
    """
    Manages streamline selection operations.

    Provides sphere-based and box-based streamline finding with a two-phase
    approach: vectorized bounding box checks (broad phase) followed by
    precise geometric checks (narrow phase).
    """

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        """
        Initialize the selection manager.

        Args:
            vtk_panel: Reference to the parent VTKPanel instance.
        """
        self.panel = vtk_panel

    def find_streamlines_in_radius(
        self, center_point: np.ndarray, radius: float, check_all: bool = False
    ) -> Set[int]:
        """
        Uses vectorized bounding box checks (Broad Phase) followed by precise
        geometric checks (Narrow Phase) to find streamlines within a sphere.
        Searches ALL streamlines, not just the rendered subset.

        Args:
            center_point: Center of the sphere in world coordinates.
            radius: Radius of the search sphere.
            check_all: If True, check all streamlines; if False, only visible ones.

        Returns:
            Set of streamline indices within the sphere.
        """
        if (
            not self.panel.main_window
            or not self.panel.main_window.tractogram_data
            or self.panel.main_window.streamline_bboxes is None
        ):
            return set()

        tractogram = self.panel.main_window.tractogram_data
        bboxes = self.panel.main_window.streamline_bboxes

        # Vectorized Bounding Box Intersection ---
        sphere_min = center_point - radius
        sphere_max = center_point + radius

        # Vectorized AABB check:
        # Streamline is taken if:
        # (Streamline_Max >= Sphere_Min) AND (Streamline_Min <= Sphere_Max) on ALL axes.
        # bboxes shape is (N, 2, 3) -> [:, 0, :] is Min, [:, 1, :] is Max

        overlap_mask = np.all(bboxes[:, 1] >= sphere_min, axis=1) & np.all(
            bboxes[:, 0] <= sphere_max, axis=1
        )

        # Get indices of potentially intersecting streamlines
        candidate_indices = np.where(overlap_mask)[0]

        # Filter candidates based on current visibility logic (e.g. removed fibers)
        # Note: We ignore 'stride' here to ensure accuracy (select what is there, not just what is drawn)
        if check_all:
            valid_candidates = candidate_indices
        else:
            valid_candidates = [
                idx
                for idx in candidate_indices
                if idx in self.panel.main_window.visible_indices
            ]

        if len(valid_candidates) == 0:
            return set()

        # NARROW PHASE: Numba-optimized Geometric Distance Check
        indices_in_radius: Set[int] = set()
        radius_sq = radius * radius

        # Ensure center is contiguous float64 for Numba
        center_c = np.ascontiguousarray(center_point, dtype=np.float64)

        for idx in valid_candidates:
            try:
                sl = tractogram[idx]
                if sl is None or sl.size == 0:
                    continue

                # Use Numba-optimized check
                sl_c = np.ascontiguousarray(sl, dtype=np.float64)
                if _check_streamline_sphere_intersection(sl_c, center_c, radius_sq):
                    indices_in_radius.add(idx)

            except Exception:
                pass

        return indices_in_radius

    def find_streamlines_in_box(
        self, min_point: np.ndarray, max_point: np.ndarray, check_all: bool = False
    ) -> Set[int]:
        """
        Uses vectorized bounding box checks (Broad Phase) followed by precise
        point-in-box checks (Narrow Phase) to find streamlines within a box.

        Args:
            min_point: Minimum corner of the box in world coordinates.
            max_point: Maximum corner of the box in world coordinates.
            check_all: If True, check all streamlines; if False, only visible ones.

        Returns:
            Set of streamline indices within the box.
        """
        if (
            not self.panel.main_window
            or not self.panel.main_window.tractogram_data
            or self.panel.main_window.streamline_bboxes is None
        ):
            return set()

        tractogram = self.panel.main_window.tractogram_data
        bboxes = self.panel.main_window.streamline_bboxes

        # Vectorized Bounding Box Intersection
        # Streamline is taken if:
        # (Streamline_Max >= Box_Min) AND (Streamline_Min <= Box_Max) on ALL axes.
        overlap_mask = np.all(bboxes[:, 1] >= min_point, axis=1) & np.all(
            bboxes[:, 0] <= max_point, axis=1
        )

        # Get indices of potentially intersecting streamlines
        candidate_indices = np.where(overlap_mask)[0]

        if check_all:
            valid_candidates = candidate_indices
        else:
            valid_candidates = [
                idx
                for idx in candidate_indices
                if idx in self.panel.main_window.visible_indices
            ]

        if len(valid_candidates) == 0:
            return set()

        # Precise Point-in-Box Check
        indices_in_box: Set[int] = set()

        for idx in valid_candidates:
            try:
                sl = tractogram[idx]
                if sl is None or sl.size == 0:
                    continue

                # Check if any point is inside the box
                # (sl >= min_point) & (sl <= max_point)
                in_box = np.all((sl >= min_point) & (sl <= max_point), axis=1)
                if np.any(in_box):
                    indices_in_box.add(idx)

            except Exception:
                pass

        return indices_in_box

    def toggle_selection(self, indices_to_toggle: Set[int]) -> None:
        """
        Toggles the selection state for given indices and updates status/highlight.

        Args:
            indices_to_toggle: Set of streamline indices to toggle.
        """
        if not self.panel.main_window or not hasattr(
            self.panel.main_window, "selected_streamline_indices"
        ):
            return
        current_selection: Set[int] = self.panel.main_window.selected_streamline_indices
        if current_selection is None:
            self.panel.main_window.selected_streamline_indices = current_selection = (
                set()
            )

        added_count, removed_count = 0, 0
        for idx in indices_to_toggle:
            if idx in current_selection:
                current_selection.remove(idx)
                removed_count += 1
            else:
                current_selection.add(idx)
                added_count += 1

        if added_count > 0 or removed_count > 0:
            total_selected = len(current_selection)
            status_msg = (
                f"Radius Sel: Found {len(indices_to_toggle)}. "
                f"Added {added_count}, Removed {removed_count}. Total Sel: {total_selected}"
            )
            self.panel.update_status(status_msg)
            self.panel.update_highlight()
        elif indices_to_toggle:
            self.panel.update_status(
                f"Radius Sel: Found {len(indices_to_toggle)}. Selection unchanged."
            )

    def handle_streamline_selection(self) -> None:
        """Handles the logic for selecting streamlines triggered by the 's' key."""
        if (
            not self.panel.scene
            or not self.panel.main_window
            or not self.panel.main_window.tractogram_data
        ):
            self.panel.update_status(
                "Select ('s'): No streamlines loaded to select from."
            )
            self.panel.update_radius_actor(visible=False)
            return

        display_pos = self.panel.interactor.GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(
            display_pos[0],
            display_pos[1],
            0,
            self.panel.render_window.GetRenderers().GetFirstRenderer(),
        )

        picked_actor = picker.GetActor()
        click_pos_world = picker.GetPickPosition()

        if (
            not picked_actor
            or not click_pos_world
            or len(click_pos_world) != 3
            or picker.GetCellId() < 0
        ):
            self.panel.update_status(
                "Select ('s'): Please click directly on visible streamlines."
            )
            self.panel.update_radius_actor(visible=False)
            return

        p_center_arr = np.array(click_pos_world)
        radius = self.panel.main_window.selection_radius_3d
        self.panel.update_radius_actor(
            center_point=p_center_arr, radius=radius, visible=True
        )
        indices_in_radius = self.find_streamlines_in_radius(p_center_arr, radius)

        if not indices_in_radius:
            self.panel.update_status(
                "Radius Sel: No streamlines found within radius at click position."
            )
            self.toggle_selection(set())
        else:
            self.toggle_selection(indices_in_radius)
