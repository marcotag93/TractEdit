# -*- coding: utf-8 -*-

"""
Selection manager for TractEdit visualization.

Handles streamline selection operations including sphere-based and
box-based streamline finding using vectorized bounding box checks
followed by precise geometric checks with parallel Numba processing.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from itertools import islice
from typing import TYPE_CHECKING, List, Set

import numpy as np
import vtk
from numba import njit, prange

if TYPE_CHECKING:
    from .vtk_panel import VTKPanel

logger = logging.getLogger(__name__)


# ============================================================================
# Numba Optimized Functions
# ============================================================================


@njit(nogil=True, cache=True)
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


@njit(parallel=True, nogil=True, cache=True)
def _batch_check_sphere_intersection(
    streamline_data: np.ndarray,
    streamline_offsets: np.ndarray,
    center: np.ndarray,
    radius_sq: float,
) -> np.ndarray:
    """
    Batch check multiple streamlines against a sphere using parallel execution.

    Args:
        streamline_data: Flattened (N_total_points, 3) array of all streamline points.
        streamline_offsets: (N_streamlines + 1,) array of start indices for each
            streamline in streamline_data. The i-th streamline's points are at
            streamline_data[offsets[i]:offsets[i+1]].
        center: (3,) sphere center.
        radius_sq: Squared radius of sphere.

    Returns:
        Boolean array of length (N_streamlines,) indicating intersection.
    """
    n_streamlines = len(streamline_offsets) - 1
    results = np.zeros(n_streamlines, dtype=np.bool_)

    for i in prange(n_streamlines):
        start_idx = streamline_offsets[i]
        end_idx = streamline_offsets[i + 1]

        if end_idx <= start_idx:
            continue

        streamline = streamline_data[start_idx:end_idx]
        results[i] = _check_streamline_sphere_intersection(
            streamline, center, radius_sq
        )

    return results


@njit(parallel=True, nogil=True, cache=True)
def _batch_check_box_intersection(
    streamline_data: np.ndarray,
    streamline_offsets: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray:
    """
    Batch check multiple streamlines against a box using parallel execution.

    Args:
        streamline_data: Flattened (N_total_points, 3) array of all streamline points.
        streamline_offsets: (N_streamlines + 1,) array of start indices.
        box_min: (3,) minimum corner of box.
        box_max: (3,) maximum corner of box.

    Returns:
        Boolean array of length (N_streamlines,) indicating intersection.
    """
    n_streamlines = len(streamline_offsets) - 1
    results = np.zeros(n_streamlines, dtype=np.bool_)

    for i in prange(n_streamlines):
        start_idx = streamline_offsets[i]
        end_idx = streamline_offsets[i + 1]

        if end_idx <= start_idx:
            continue

        # Check if any point in the streamline is inside the box
        for j in range(start_idx, end_idx):
            x, y, z = (
                streamline_data[j, 0],
                streamline_data[j, 1],
                streamline_data[j, 2],
            )
            if (
                x >= box_min[0]
                and x <= box_max[0]
                and y >= box_min[1]
                and y <= box_max[1]
                and z >= box_min[2]
                and z <= box_max[2]
            ):
                results[i] = True
                break

    return results


@njit(parallel=True, nogil=True, cache=True)
def _copy_streamlines_parallel(
    src_data: np.ndarray,
    dst_data: np.ndarray,
    src_starts: np.ndarray,
    dst_starts: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """
    Parallel copy of streamline data from source to destination buffer.

    Uses Numba parallel execution to accelerate data preparation for
    batch geometric checks.

    Args:
        src_data: Source array containing all streamline points.
        dst_data: Pre-allocated destination array.
        src_starts: Start indices in source for each streamline.
        dst_starts: Start indices in destination for each streamline.
        lengths: Number of points in each streamline.
    """
    n = len(lengths)
    for i in prange(n):
        src_start = src_starts[i]
        dst_start = dst_starts[i]
        length = lengths[i]
        for j in range(length):
            dst_data[dst_start + j, 0] = src_data[src_start + j, 0]
            dst_data[dst_start + j, 1] = src_data[src_start + j, 1]
            dst_data[dst_start + j, 2] = src_data[src_start + j, 2]


def warmup_selection_numba_functions() -> None:
    """
    Pre-compiles selection-related Numba JIT functions with minimal dummy data.

    This should be called during application startup to avoid JIT compilation
    delay on first sphere/box selection operation.
    """
    # Minimal dummy data (5 streamlines, 3 points each)
    dummy_data = np.zeros((15, 3), dtype=np.float64)
    dummy_offsets = np.array([0, 3, 6, 9, 12, 15], dtype=np.int64)
    dummy_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dummy_box_min = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    dummy_box_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # Warmup sphere intersection check
    try:
        _batch_check_sphere_intersection(dummy_data, dummy_offsets, dummy_center, 1.0)
    except Exception:
        pass

    # Warmup box intersection check
    try:
        _batch_check_box_intersection(
            dummy_data, dummy_offsets, dummy_box_min, dummy_box_max
        )
    except Exception:
        pass

    # Warmup parallel copy
    try:
        dst_data = np.zeros((15, 3), dtype=np.float64)
        src_starts = np.array([0, 3, 6, 9, 12], dtype=np.int64)
        dst_starts = np.array([0, 3, 6, 9, 12], dtype=np.int64)
        lengths = np.array([3, 3, 3, 3, 3], dtype=np.int64)
        _copy_streamlines_parallel(
            dummy_data, dst_data, src_starts, dst_starts, lengths
        )
    except Exception:
        pass

    logger.debug("Selection Numba functions warmed up")


# ============================================================================
# Helper Functions
# ============================================================================


def _prepare_batch_data_fast(tractogram, candidate_indices: np.ndarray) -> tuple:
    """
    Prepare flattened streamline data and offsets for batch Numba processing.

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

        # SMALL SET OPTIMIZATION: For very few candidates, skip Numba overhead
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

        # LARGE SET PATH: Use Numba parallel copy
        # Build output offsets
        out_offsets = np.zeros(n_candidates + 1, dtype=np.int64)
        valid_cumsum = np.cumsum(lengths)

        # Place cumsum values at valid positions
        valid_indices = np.where(valid_mask)[0]
        out_offsets[valid_indices + 1] = valid_cumsum

        # Forward fill to create proper offset array
        for i in range(1, n_candidates + 1):
            if out_offsets[i] == 0 and i > 0:
                out_offsets[i] = out_offsets[i - 1]

        # Allocate output data
        streamline_data = np.empty((total_points, 3), dtype=np.float64)

        # Vectorized copy using source offsets
        src_starts = src_offsets[valid_candidates]
        dst_starts = np.zeros(len(valid_candidates), dtype=np.int64)
        if len(valid_cumsum) > 1:
            dst_starts[1:] = valid_cumsum[:-1]

        # Parallel copy using Numba
        _copy_streamlines_parallel(
            np.ascontiguousarray(src_data, dtype=np.float64),
            streamline_data,
            src_starts.astype(np.int64),
            dst_starts.astype(np.int64),
            lengths.astype(np.int64),
        )

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
            except Exception:
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
    precise geometric checks (narrow phase) with parallel Numba processing.
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
        self._cached_visible_count: int = 0
        self._cached_visible_hash: int = 0

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
        visible_indices = self.panel.main_window.visible_indices
        current_count = len(visible_indices)

        if current_count == 0:
            return np.array([], dtype=np.int64)

        # Use a quick hash based on a sample of indices to detect changes
        # Use islice for efficient sampling without creating intermediate list
        sample_hash = hash(frozenset(islice(visible_indices, 100)))

        if (
            self._cached_visible_array is None
            or self._cached_visible_count != current_count
            or self._cached_visible_hash != sample_hash
        ):
            # Rebuild cache
            self._cached_visible_array = np.fromiter(
                visible_indices, dtype=np.int64, count=current_count
            )
            self._cached_visible_array.sort()
            self._cached_visible_count = current_count
            self._cached_visible_hash = sample_hash

        return self._cached_visible_array

    def invalidate_visible_cache(self) -> None:
        """Invalidate the cached visible array, forcing rebuild on next use."""
        self._cached_visible_array = None
        self._cached_visible_count = 0
        self._cached_visible_hash = 0

    def find_streamlines_in_radius(
        self, center_point: np.ndarray, radius: float, check_all: bool = False
    ) -> Set[int]:
        """
        Find streamlines within a sphere using optimized batch processing.

        Uses vectorized bounding box checks (Broad Phase) followed by parallel
        Numba-optimized geometric checks (Narrow Phase).

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

        # --- STANDARD PATH: BROAD PHASE on all bboxes ---
        # Used when check_all=True or visible count is large (>10% of total)
        overlap_mask = np.all(bboxes[:, 1] >= sphere_min, axis=1) & np.all(
            bboxes[:, 0] <= sphere_max, axis=1
        )

        candidate_indices = np.where(overlap_mask)[0]

        # Filter by visibility using vectorized intersection
        valid_candidates = self._filter_visible_candidates(candidate_indices, check_all)

        if len(valid_candidates) == 0:
            return set()

        # --- NARROW PHASE: Parallel Numba Geometric Check ---
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
        Numba-optimized point-in-box checks (Narrow Phase).

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

        # --- STANDARD PATH: BROAD PHASE on all bboxes ---
        # Used when check_all=True or visible count is large (>10% of total)
        overlap_mask = np.all(bboxes[:, 1] >= min_point, axis=1) & np.all(
            bboxes[:, 0] <= max_point, axis=1
        )

        candidate_indices = np.where(overlap_mask)[0]

        # Filter by visibility using vectorized intersection
        valid_candidates = self._filter_visible_candidates(candidate_indices, check_all)

        if len(valid_candidates) == 0:
            return set()

        # --- NARROW PHASE: Parallel Numba Point-in-Box Check ---
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

    def toggle_selection(self, indices_to_toggle: Set[int]) -> None:
        """
        Toggles the selection state for given indices and updates status/highlight.

        Uses deferred updates with debouncing for rapid consecutive operations
        to prevent redundant actor rebuilds.

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

            # Always update highlight immediately for visual feedback
            self.panel.update_highlight()
        elif indices_to_toggle:
            self.panel.update_status(
                f"Radius Sel: Found {len(indices_to_toggle)}. Selection unchanged."
            )

    def invert_selection(self) -> None:
        """
        Inverts the current selection based on visible streamlines.
        Selects all visible streamlines that are NOT currently selected.
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

        current_selection: Set[int] = self.panel.main_window.selected_streamline_indices
        if current_selection is None:
            current_selection = set()
            self.panel.main_window.selected_streamline_indices = current_selection

        # Convert to numpy array for fast set difference if possible, or sets
        # Using sets for clarity and typical selection sizes
        visible_set = set(visible_indices)

        # New selection = Visible - Currently Selected
        # This will select everything visible that wasn't selected,
        # and unselect everything that was selected (effectively, though we rewrite the set).
        # Note: If there are selected items that are NOT visible (e.g. filtered out),
        # should they remain selected? "Inverse" typically implies "swap state".
        # If I am looking at a subset, "Inverse" usually means "Select the other part of this subset".
        # So we should probably clearer: new_selection = Visible - (Visible & Current)
        # effectively: new_selection = visible_set - current_selection

        new_selection = visible_set - current_selection

        # Update the main set
        # We replace the content of the set to maintain the reference if used elsewhere,
        # or just reassign. Reassigning is safer if we own it.
        # But 'selected_streamline_indices' is likely a set object on main_window.
        # Let's clear and update to be safe and keep the object reference if it matters.

        current_selection.clear()
        current_selection.update(new_selection)

        self.panel.update_status(
            f"Inverse Sel: Selected {len(current_selection)} streamlines."
        )
        self.panel.update_highlight()

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
