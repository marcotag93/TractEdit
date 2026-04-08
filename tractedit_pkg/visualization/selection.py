# -*- coding: utf-8 -*-

"""
Selection manager for TractEdit visualization.

Handles streamline selection operations including sphere-based and
box-based streamline finding using vectorized bounding box checks
followed by precise geometric checks with parallel AOT-compiled processing.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Set

import numpy as np
import vtk

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


# ============================================================================
# AOT-compiled functions (imported from pre-compiled extension)
# ============================================================================

from tractedit_pkg._numba_aot import (  # noqa: E402
    check_streamline_sphere_intersection as _check_streamline_sphere_intersection,
)


# _batch_check_sphere_intersection — AOT chunk + ThreadPool wrapper
from tractedit_pkg._numba_aot._parallel_wrappers import (
    batch_check_sphere_intersection as _batch_check_sphere_intersection,
)

# _batch_check_box_intersection — AOT chunk + ThreadPool wrapper
from tractedit_pkg._numba_aot._parallel_wrappers import (
    batch_check_box_intersection as _batch_check_box_intersection,
)


from ..utils import MAX_HIGHLIGHT_STREAMLINES


# ============================================================================
# Helper Functions
# ============================================================================


def _prepare_batch_data_fast(tractogram, candidate_indices: np.ndarray) -> tuple:
    """
    Prepare flattened streamline data and offsets for batch AOT processing.

    Optimized version that leverages ArraySequence's internal _data and _offsets
    arrays for vectorized access when available.

    Args:
        tractogram: The tractogram data (ArraySequence or similar).
        candidate_indices: Array of streamline indices to process.

    Returns:
        Tuple of (streamline_data, streamline_offsets, valid_mask) where:
        - streamline_data: (N_total_points, 3) float64 array
        - streamline_offsets: (len(candidate_indices) + 1,) int64 array
        - valid_mask: Boolean array indicating which candidates had valid data
    """
    n_candidates = len(candidate_indices)

    if n_candidates == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.bool_),
        )

    # Check if we can use fast path with ArraySequence internals
    has_array_sequence = (
        hasattr(tractogram, "_data")
        and hasattr(tractogram, "_offsets")
        and hasattr(tractogram, "_lengths")
    )

    if has_array_sequence:
        # FAST PATH: Use internal arrays for vectorized access
        src_data = tractogram._data  # (N_total, 3) all points
        src_offsets = tractogram._offsets  # Start index of each streamline
        src_lengths = tractogram._lengths  # Length of each streamline

        # Filter candidates to valid range
        max_idx = len(src_lengths)
        valid_mask = candidate_indices < max_idx

        # Get lengths for valid candidates
        valid_candidates = candidate_indices[valid_mask]
        lengths = src_lengths[valid_candidates].astype(np.int64)

        # Mark zero-length streamlines as invalid
        zero_length_mask = lengths == 0
        if np.any(zero_length_mask):
            # Update valid_mask for zero-length streamlines
            temp_mask = valid_mask.copy()
            temp_mask[valid_mask] = ~zero_length_mask
            valid_mask = temp_mask
            lengths = lengths[~zero_length_mask]
            valid_candidates = candidate_indices[valid_mask]

        if len(valid_candidates) == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.zeros(n_candidates + 1, dtype=np.int64),
                np.zeros(n_candidates, dtype=np.bool_),
            )

        total_points = int(lengths.sum())

        # SMALL SET OPTIMIZATION: For very few candidates, skip AOT overhead
        # and use simple concatenation instead
        if len(valid_candidates) < 20:
            # Simple concatenation for small sets (faster than parallel copy)
            slices = []
            for idx in valid_candidates:
                start = src_offsets[idx]
                end = start + src_lengths[idx]
                slices.append(src_data[start:end])

            if slices:
                streamline_data = np.concatenate(slices, axis=0).astype(
                    np.float64, copy=False
                )
            else:
                streamline_data = np.empty((0, 3), dtype=np.float64)

            # Build simple offsets
            offsets = np.zeros(n_candidates + 1, dtype=np.int64)
            cumsum = 0
            valid_idx = 0
            for i in range(n_candidates):
                if valid_mask[i]:
                    cumsum += lengths[valid_idx]
                    valid_idx += 1
                offsets[i + 1] = cumsum

            full_valid_mask = np.zeros(n_candidates, dtype=np.bool_)
            valid_indices = np.where(valid_mask)[0]
            full_valid_mask[valid_indices] = True

            return streamline_data, offsets, full_valid_mask

        # Build output offsets
        out_offsets = np.zeros(n_candidates + 1, dtype=np.int64)
        valid_cumsum = np.cumsum(lengths)

        # Place cumsum values at valid positions
        valid_indices = np.where(valid_mask)[0]
        out_offsets[valid_indices + 1] = valid_cumsum

        # Forward fill to create proper offset array (vectorized)
        np.maximum.accumulate(out_offsets, out=out_offsets)

        # Copy only the needed streamlines into a packed float64 buffer.
        # Optimisation: merge *contiguous* source runs into large block copies
        src_starts = src_offsets[valid_candidates].astype(np.int64)
        src_ends = src_starts + lengths

        # Destination starts (packed, no gaps)
        dst_starts = np.empty(len(lengths), dtype=np.int64)
        dst_starts[0] = 0
        if len(lengths) > 1:
            np.cumsum(lengths[:-1], out=dst_starts[1:])

        # Detect contiguous runs: a break occurs when the next source
        # start doesn't immediately follow the current source end.
        if len(src_starts) > 1:
            breaks = np.where(src_starts[1:] != src_ends[:-1])[0]
        else:
            breaks = np.empty(0, dtype=np.int64)

        streamline_data = np.empty((total_points, 3), dtype=np.float64)

        # Run boundaries
        n_runs = len(breaks) + 1
        run_starts = np.empty(n_runs, dtype=np.int64)
        run_starts[0] = 0
        if len(breaks) > 0:
            run_starts[1:] = breaks + 1
        run_ends_idx = np.empty(n_runs, dtype=np.int64)
        if len(breaks) > 0:
            run_ends_idx[:-1] = breaks + 1
        run_ends_idx[-1] = len(valid_candidates)

        for r in range(n_runs):
            rs = int(run_starts[r])
            re = int(run_ends_idx[r])
            s_begin = int(src_starts[rs])
            s_end = int(src_ends[re - 1])
            d_begin = int(dst_starts[rs])
            d_end = int(dst_starts[re - 1]) + int(lengths[re - 1])
            streamline_data[d_begin:d_end] = src_data[s_begin:s_end]

        # Rebuild full valid_mask for all candidates
        full_valid_mask = np.zeros(n_candidates, dtype=np.bool_)
        full_valid_mask[valid_indices] = True

        return streamline_data, out_offsets, full_valid_mask

    else:
        # SLOW PATH: Fall back to individual indexing
        point_counts = np.zeros(n_candidates, dtype=np.int64)
        valid_mask = np.ones(n_candidates, dtype=np.bool_)

        for i, idx in enumerate(candidate_indices):
            try:
                sl = tractogram[idx]
                if sl is None or sl.size == 0:
                    valid_mask[i] = False
                    point_counts[i] = 0
                else:
                    point_counts[i] = len(sl)
            except (IndexError, ValueError, TypeError):
                logger.debug("Failed to extract streamline at index %d.", idx)
                valid_mask[i] = False
                point_counts[i] = 0

        total_points = int(point_counts.sum())

        if total_points == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.array([0], dtype=np.int64),
                valid_mask,
            )

        offsets = np.zeros(n_candidates + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(point_counts)

        streamline_data = np.empty((total_points, 3), dtype=np.float64)

        for i, idx in enumerate(candidate_indices):
            if not valid_mask[i]:
                continue
            start = offsets[i]
            end = offsets[i + 1]
            sl = tractogram[idx]
            streamline_data[start:end] = sl

        return streamline_data, offsets, valid_mask


# ============================================================================
# Selection Manager Class
# ============================================================================


class SelectionManager:
    """
    Manages streamline selection operations.

    Provides sphere-based and box-based streamline finding with a two-phase
    approach: vectorized bounding box checks (broad phase) followed by
    precise geometric checks (narrow phase) with parallel AOT-compiled processing.
    """

    def __init__(self, vtk_panel: "VTKPanel") -> None:
        """
        Initialize the selection manager.

        Args:
            vtk_panel: Reference to the parent VTKPanel instance.
        """
        self.panel = vtk_panel
        # Cached visible array for faster filtering
        self._cached_visible_array: np.ndarray = None
        self._cached_visibility_version: int = -1
        # Reusable cell picker — avoids recreating the VTK object (and its
        # internal cell locator structures) on every S key press.
        self._cell_picker: vtk.vtkCellPicker = vtk.vtkCellPicker()
        self._cell_picker.SetTolerance(0.005)

    def _filter_visible_candidates(
        self, candidate_indices: np.ndarray, check_all: bool
    ) -> np.ndarray:
        """
        Filter candidate indices to only include visible streamlines.

        Uses vectorized NumPy intersection for performance with large sets.

        Args:
            candidate_indices: Array of candidate streamline indices.
            check_all: If True, return all candidates without filtering.

        Returns:
            Array of valid candidate indices.
        """
        if check_all:
            return candidate_indices

        visible_indices = self.panel.main_window.visible_indices

        if len(candidate_indices) == 0 or len(visible_indices) == 0:
            return np.array([], dtype=np.int64)

        # For small candidate sets, simple loop may be faster
        if len(candidate_indices) < 100:
            return np.array(
                [idx for idx in candidate_indices if idx in visible_indices],
                dtype=np.int64,
            )

        # Use cached visible array
        visible_arr = self._get_cached_visible_array()

        return np.intersect1d(candidate_indices, visible_arr, assume_unique=False)

    def _get_cached_visible_array(self) -> np.ndarray:
        """
        Returns the cached visible indices as a sorted numpy array.

        Rebuilds the cache only if the visible set has changed.
        This avoids repeated np.fromiter() calls on every selection.

        Returns:
            Sorted numpy array of visible streamline indices.
        """
        mw = self.panel.main_window
        visible_indices = mw.visible_indices
        current_version = mw._visibility_version

        if len(visible_indices) == 0:
            return np.array([], dtype=np.int64)

        if (
            self._cached_visible_array is None
            or self._cached_visibility_version != current_version
        ):
            # Rebuild cache
            self._cached_visible_array = np.fromiter(
                visible_indices, dtype=np.int64, count=len(visible_indices)
            )
            self._cached_visible_array.sort()
            self._cached_visibility_version = current_version

        return self._cached_visible_array

    def invalidate_visible_cache(self) -> None:
        """Invalidate the cached visible array, forcing rebuild on next use."""
        self._cached_visible_array = None
        self._cached_visibility_version = -1

    def find_streamlines_in_radius(
        self, center_point: np.ndarray, radius: float, check_all: bool = False
    ) -> Set[int]:
        """
        Find streamlines within a sphere using optimized batch processing.

        Uses vectorized bounding box checks (Broad Phase) followed by parallel
        AOT-optimized geometric checks (Narrow Phase).

        When working with filtered tractograms, pre-filters to visible indices
        before the broad phase to avoid checking millions of irrelevant bboxes.

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
        total_streamlines = len(tractogram)

        # Sphere bounds for AABB check
        sphere_min = center_point - radius
        sphere_max = center_point + radius

        # OPTIMIZATION: Pre-filter by visibility when visible set is much smaller
        # than the full tractogram. This avoids checking millions of bboxes.
        if not check_all:
            visible_indices = self.panel.main_window.visible_indices
            visible_count = len(visible_indices)

            # If visible is <10% of total, pre-filter first (faster path)
            if visible_count < total_streamlines * 0.1 and visible_count > 0:
                # Use cached visible array (avoids repeated np.fromiter calls)
                visible_arr = self._get_cached_visible_array()

                # Check only visible bboxes
                visible_bboxes = bboxes[visible_arr]
                overlap_mask = np.all(
                    visible_bboxes[:, 1] >= sphere_min, axis=1
                ) & np.all(visible_bboxes[:, 0] <= sphere_max, axis=1)

                # Get candidates directly from visible set
                valid_candidates = visible_arr[overlap_mask]

                if len(valid_candidates) == 0:
                    return set()

                # Skip to NARROW PHASE (already filtered by visibility)
                radius_sq = radius * radius
                center_c = np.ascontiguousarray(center_point, dtype=np.float64)

                streamline_data, offsets, valid_mask = _prepare_batch_data_fast(
                    tractogram, valid_candidates
                )

                if streamline_data.shape[0] == 0:
                    return set()

                intersection_results = _batch_check_sphere_intersection(
                    streamline_data, offsets, center_c, radius_sq
                )

                final_mask = valid_mask & intersection_results
                return set(valid_candidates[final_mask].tolist())

        # STANDARD PATH: BROAD PHASE on all bboxes
        # Used when check_all=True or visible count is large (>10% of total)
        overlap_mask = np.all(bboxes[:, 1] >= sphere_min, axis=1) & np.all(
            bboxes[:, 0] <= sphere_max, axis=1
        )

        candidate_indices = np.where(overlap_mask)[0]

        # Filter by visibility using vectorized intersection
        valid_candidates = self._filter_visible_candidates(candidate_indices, check_all)

        if len(valid_candidates) == 0:
            return set()

        # NARROW PHASE: Parallel AOT Geometric Check
        radius_sq = radius * radius
        center_c = np.ascontiguousarray(center_point, dtype=np.float64)

        # Prepare batch data for parallel processing
        streamline_data, offsets, valid_mask = _prepare_batch_data_fast(
            tractogram, valid_candidates
        )

        if streamline_data.shape[0] == 0:
            return set()

        # Run parallel batch intersection check
        intersection_results = _batch_check_sphere_intersection(
            streamline_data, offsets, center_c, radius_sq
        )

        # Combine results: must be valid AND intersecting
        final_mask = valid_mask & intersection_results
        indices_in_radius = set(valid_candidates[final_mask].tolist())

        return indices_in_radius

    def find_streamlines_in_box(
        self, min_point: np.ndarray, max_point: np.ndarray, check_all: bool = False
    ) -> Set[int]:
        """
        Find streamlines within a box using optimized batch processing.

        Uses vectorized bounding box checks (Broad Phase) followed by parallel
        AOT-optimized point-in-box checks (Narrow Phase).

        When working with filtered tractograms, pre-filters to visible indices
        before the broad phase to avoid checking millions of irrelevant bboxes.

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
        total_streamlines = len(tractogram)

        # OPTIMIZATION: Pre-filter by visibility when visible set is much smaller
        # than the full tractogram. This avoids checking millions of bboxes.
        if not check_all:
            visible_indices = self.panel.main_window.visible_indices
            visible_count = len(visible_indices)

            # If visible is <10% of total, pre-filter first (faster path)
            if visible_count < total_streamlines * 0.1 and visible_count > 0:
                # Use cached visible array (avoids repeated np.fromiter calls)
                visible_arr = self._get_cached_visible_array()

                # Check only visible bboxes
                visible_bboxes = bboxes[visible_arr]
                overlap_mask = np.all(
                    visible_bboxes[:, 1] >= min_point, axis=1
                ) & np.all(visible_bboxes[:, 0] <= max_point, axis=1)

                # Get candidates directly from visible set
                valid_candidates = visible_arr[overlap_mask]

                if len(valid_candidates) == 0:
                    return set()

                # Skip to NARROW PHASE (already filtered by visibility)
                box_min_c = np.ascontiguousarray(min_point, dtype=np.float64)
                box_max_c = np.ascontiguousarray(max_point, dtype=np.float64)

                streamline_data, offsets, valid_mask = _prepare_batch_data_fast(
                    tractogram, valid_candidates
                )

                if streamline_data.shape[0] == 0:
                    return set()

                intersection_results = _batch_check_box_intersection(
                    streamline_data, offsets, box_min_c, box_max_c
                )

                final_mask = valid_mask & intersection_results
                return set(valid_candidates[final_mask].tolist())

        # STANDARD PATH: BROAD PHASE on all bboxes
        # Used when check_all=True or visible count is large (>10% of total)
        overlap_mask = np.all(bboxes[:, 1] >= min_point, axis=1) & np.all(
            bboxes[:, 0] <= max_point, axis=1
        )

        candidate_indices = np.where(overlap_mask)[0]

        # Filter by visibility using vectorized intersection
        valid_candidates = self._filter_visible_candidates(candidate_indices, check_all)

        if len(valid_candidates) == 0:
            return set()

        # NARROW PHASE: Parallel AOT Point-in-Box Check
        box_min_c = np.ascontiguousarray(min_point, dtype=np.float64)
        box_max_c = np.ascontiguousarray(max_point, dtype=np.float64)

        # Prepare batch data for parallel processing
        streamline_data, offsets, valid_mask = _prepare_batch_data_fast(
            tractogram, valid_candidates
        )

        if streamline_data.shape[0] == 0:
            return set()

        # Run parallel batch box check
        intersection_results = _batch_check_box_intersection(
            streamline_data, offsets, box_min_c, box_max_c
        )

        # Combine results: must be valid AND intersecting
        final_mask = valid_mask & intersection_results
        indices_in_box = set(valid_candidates[final_mask].tolist())

        return indices_in_box

    def apply_selection(
        self, indices_in_sphere: Set[int], deselect: bool = False
    ) -> None:
        """Apply a sphere selection result to the current selection set.

        Depending on the ``deselect`` flag, this method either adds streamlines
        to the selection (add-only mode) or removes them from it
        (deselect-in-sphere mode).  Neither path can accidentally perform the
        other's job:

        - ``deselect=False`` (default, ``S`` key): performs a pure set union.
          Streamlines already selected are silently ignored; the selection can
          only grow.
        - ``deselect=True`` (``Shift+S``): performs a pure set difference.
          Streamlines not currently selected are silently ignored; the
          selection can only shrink.

        Both branches use a single C-level ``set`` operation (``update`` /
        ``difference_update``), so there is no per-element Python loop and no
        performance regression relative to the previous toggle implementation.

        Args:
            indices_in_sphere: Set of streamline indices found within the
                selection sphere.
            deselect: When ``True``, remove ``indices_in_sphere`` from the
                current selection.  When ``False`` (default), add them.
        """
        if not self.panel.main_window or not hasattr(
            self.panel.main_window, "selected_streamline_indices"
        ):
            return

        current_selection: Set[int] = self.panel.main_window.selected_streamline_indices
        if current_selection is None:
            current_selection = set()
            self.panel.main_window.selected_streamline_indices = current_selection

        if deselect:
            removed = indices_in_sphere & current_selection
            current_selection.difference_update(removed)
            added_count = 0
            removed_count = len(removed)
        else:
            added = indices_in_sphere - current_selection
            current_selection.update(added)
            added_count = len(added)
            removed_count = 0

        total_selected = len(current_selection)

        if added_count > 0 or removed_count > 0:
            verb = "Deselected" if deselect else "Added"
            n = removed_count if deselect else added_count
            self.panel.update_status(
                f"Radius Sel: {verb} {n:,}. Total selected: {total_selected:,}"
            )
            self.panel.update_highlight()
        elif indices_in_sphere:
            action = "to deselect" if deselect else "to add"
            self.panel.update_status(
                f"Radius Sel: Found {len(indices_in_sphere):,} — "
                f"none {action}. Total selected: {total_selected:,}"
            )

    def invert_selection(self) -> None:
        """Invert the current selection against the set of visible streamlines.

        Selects all visible streamlines that are NOT currently selected.

        The inversion itself is always computed (trivial set difference).
        When the result exceeds ``MAX_HIGHLIGHT_STREAMLINES``, the yellow
        highlight is suppressed to avoid RAM exhaustion, but a cyan contour
        is always rendered on the (small) pre-inversion keeper set so the
        user retains clear spatial feedback.

        Pressing ``I`` a second time while inversion mode is active restores
        the original keeper selection and exits inversion mode.
        """
        if (
            not self.panel.main_window
            or not self.panel.main_window.tractogram_data
            or not hasattr(self.panel.main_window, "selected_streamline_indices")
        ):
            self.panel.update_status(
                "Inverse Sel: No streamlines loaded or selection state unavailable."
            )
            return

        visible_indices = self.panel.main_window.visible_indices
        if visible_indices is None or len(visible_indices) == 0:
            self.panel.update_status("Inverse Sel: No visible streamlines to select.")
            return

        mw = self.panel.main_window
        current_selection: Set[int] = mw.selected_streamline_indices
        if current_selection is None:
            current_selection = set()
            mw.selected_streamline_indices = current_selection

        # --- Toggle: if already inverted, revert to the keeper set ---
        if mw._inversion_active:
            restored = mw._inversion_keeper_indices.copy()
            current_selection.clear()
            current_selection.update(restored)

            mw._inversion_active = False
            mw._inversion_keeper_indices = set()

            self.panel.clear_invert_contour()
            self.panel.update_highlight()
            self.panel.update_status(
                f"Inverse Sel: Reverted to original "
                f"{len(current_selection):,} streamlines."
            )
            return

        # --- First inversion: save keepers, compute set difference ---
        if not current_selection:
            self.panel.update_status("Inverse Sel: No current selection to invert.")
            return

        # Snapshot the pre-inversion (keeper) set before mutating.
        keeper_indices: Set[int] = current_selection.copy()

        # Set difference is O(|visible|) and negligible compared to actor build.
        new_selection = visible_indices - current_selection

        # Update in-place to preserve any external references to the set object.
        current_selection.clear()
        current_selection.update(new_selection)

        count = len(current_selection)

        mw._inversion_active = True
        mw._inversion_keeper_indices = keeper_indices

        # Cyan contour on keepers is always cheap (keeper set is small).
        self.panel.update_invert_contour()

        if count > MAX_HIGHLIGHT_STREAMLINES:
            self.panel.clear_highlight()
            self.panel.update_status(
                f"Inverse Sel: {count:,} streamlines selected "
                f"(cyan contour shows {len(keeper_indices):,} keepers — "
                f"press D to delete, I to revert)."
            )
            logger.info(
                "Invert selection: %d streamlines selected; yellow highlight "
                "suppressed (exceeds MAX_HIGHLIGHT_STREAMLINES=%d). "
                "Cyan contour rendered for %d keepers.",
                count,
                MAX_HIGHLIGHT_STREAMLINES,
                len(keeper_indices),
            )
            mw._update_action_states()
        else:
            # Cyan contour on keepers is the sole visual indicator during
            # inversion — no yellow highlight regardless of bundle size.
            self.panel.clear_highlight()
            self.panel.update_status(
                f"Inverse Sel: Selected {count:,} streamlines "
                f"(cyan contour shows {len(keeper_indices):,} keepers)."
            )
            mw._update_action_states()

    def handle_streamline_selection(self, deselect: bool = False) -> None:
        """Handle the sphere-based streamline selection triggered by the ``S`` key.

        A new sphere selection always exits inversion mode: the cyan contour is
        removed and inversion state is cleared before the new selection is built.

        Args:
            deselect: When ``True`` (``Shift+S``), remove streamlines found
                within the sphere from the current selection.  When ``False``
                (default, ``S``), add them.  Corresponds to the two modes
                implemented in :meth:`apply_selection`.
        """
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

        # Exit inversion mode before starting a fresh sphere selection.
        # Crucially, ``selected_streamline_indices`` must be restored to the
        # pre-inversion keeper set BEFORE clearing ``_inversion_keeper_indices``
        mw = self.panel.main_window
        if mw._inversion_active:
            mw.selected_streamline_indices.clear()
            mw.selected_streamline_indices.update(mw._inversion_keeper_indices)
            mw._inversion_active = False
            mw._inversion_keeper_indices = set()
            self.panel.clear_invert_contour()

        display_pos = self.panel.interactor.GetEventPosition()

        # Reuse the cached vtkCellPicker to avoid recreating the internal
        # cell locator on every S press.
        picker = self._cell_picker
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
        else:
            self.apply_selection(indices_in_radius, deselect=deselect)
