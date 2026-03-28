# -*- coding: utf-8 -*-

"""
Thread-based parallel dispatch for AOT-compiled chunk kernels.

AOT-compiled functions release the GIL, so ``ThreadPoolExecutor`` achieves
true parallelism without the overhead of process-based approaches.

The module provides:
- ``parallel_chunks``: distributes a chunk kernel across a reusable pool.
- Per-function wrappers that allocate output arrays, call
  ``parallel_chunks``, and return results.
"""

from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np

# Match the core count available to the process.
_N_WORKERS: int = os.cpu_count() or 4

# Module-level reusable thread pool (lazy-initialized).
_pool: ThreadPoolExecutor | None = None
_pool_lock: threading.Lock = threading.Lock()


def _get_pool(n_workers: int | None = None) -> ThreadPoolExecutor:
    """Return a reusable thread pool, creating one if needed.

    Thread-safe: uses a lock to prevent concurrent callers from
    creating duplicate pools (and leaking threads).
    """
    global _pool  # noqa: PLW0603
    workers = n_workers or _N_WORKERS
    with _pool_lock:
        if _pool is None or _pool._max_workers != workers:
            if _pool is not None:
                _pool.shutdown(wait=False)
            _pool = ThreadPoolExecutor(max_workers=workers)
        return _pool


def _shutdown_pool() -> None:
    """Shut down the module-level thread pool on interpreter exit."""
    global _pool  # noqa: PLW0603
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=True)
            _pool = None


atexit.register(_shutdown_pool)


def parallel_chunks(
    kernel_fn,
    n_items: int,
    *args,
    n_workers: int | None = None,
) -> None:
    """Dispatch an AOT chunk-processing kernel across threads.

    The *kernel_fn* must accept ``(*args, start_i, end_i)`` where
    ``start_i`` / ``end_i`` define the half-open range of items to process.
    The kernel must be GIL-free (compiled AOT with nogil semantics).

    Parameters
    ----------
    kernel_fn : callable
        AOT-compiled chunk kernel.
    n_items : int
        Total number of items to distribute.
    *args
        Positional arguments forwarded to *kernel_fn* before the range.
    n_workers : int | None
        Override the default thread count.
    """
    if n_items == 0:
        return

    workers = n_workers or _N_WORKERS
    chunk_size = max(1, (n_items + workers - 1) // workers)

    # Fast path: single chunk — skip pool overhead entirely.
    if n_items <= chunk_size or workers == 1:
        kernel_fn(*args, 0, n_items)
        return

    pool = _get_pool(workers)
    futures = []
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        futures.append(pool.submit(kernel_fn, *args, start, end))
    for future in futures:
        future.result()  # Propagate exceptions immediately.


def parallel_chunks_range(
    kernel_fn,
    start_offset: int,
    end_offset: int,
    *args,
    n_workers: int | None = None,
) -> None:
    """Dispatch an AOT chunk kernel over an explicit row range.

    Like :func:`parallel_chunks`, but the work is distributed over
    ``[start_offset, end_offset)`` instead of ``[0, n_items)``.  This
    allows callers to process a large array in macro-batches (for
    cancellation / progress) while still parallelising each batch.

    Parameters
    ----------
    kernel_fn : callable
        AOT-compiled chunk kernel accepting ``(*args, start_i, end_i)``.
    start_offset : int
        First row index (inclusive).
    end_offset : int
        Last row index (exclusive).
    *args
        Positional arguments forwarded to *kernel_fn* before the range.
    n_workers : int | None
        Override the default thread count.
    """
    n_items = end_offset - start_offset
    if n_items <= 0:
        return

    workers = n_workers or _N_WORKERS
    chunk_size = max(1, (n_items + workers - 1) // workers)

    # Fast path: single chunk — skip pool overhead.
    if n_items <= chunk_size or workers == 1:
        kernel_fn(*args, start_offset, end_offset)
        return

    pool = _get_pool(workers)
    futures = []
    for start in range(start_offset, end_offset, chunk_size):
        end = min(start + chunk_size, end_offset)
        futures.append(pool.submit(kernel_fn, *args, start, end))
    for future in futures:
        future.result()


# ====================================================================
# Phase 3 — Per-function wrappers
# ====================================================================


def compute_bboxes(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Compute bounding boxes for all streamlines (parallel).

    The AOT kernel expects ``(float32[:,::1], int64[::1], int64[::1])``.
    TRX files store lengths as ``uint32`` and data as ``memmap``, so we
    ensure correct dtypes and contiguity here.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2, 3)`` with ``[min_coords, max_coords]`` per streamline.
    """
    from . import compute_bboxes_chunk

    flat_data = np.ascontiguousarray(flat_data, dtype=np.float32)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    lengths = np.ascontiguousarray(lengths, dtype=np.int64)

    n = len(lengths)
    bboxes = np.zeros((n, 2, 3), dtype=np.float32)
    parallel_chunks(compute_bboxes_chunk, n, flat_data, offsets, lengths, bboxes)
    return bboxes


def batch_check_sphere_intersection(
    streamline_data: np.ndarray,
    streamline_offsets: np.ndarray,
    center: np.ndarray,
    radius_sq: float,
) -> np.ndarray:
    """Batch check streamline–sphere intersection (parallel).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)``.
    """
    from . import check_sphere_chunk

    n = len(streamline_offsets) - 1
    results = np.zeros(n, dtype=np.bool_)
    parallel_chunks(
        check_sphere_chunk,
        n,
        streamline_data,
        streamline_offsets,
        center,
        radius_sq,
        results,
    )
    return results


def batch_check_box_intersection(
    streamline_data: np.ndarray,
    streamline_offsets: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray:
    """Batch check streamline–box intersection (parallel).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)``.
    """
    from . import check_box_chunk

    n = len(streamline_offsets) - 1
    results = np.zeros(n, dtype=np.bool_)
    parallel_chunks(
        check_box_chunk,
        n,
        streamline_data,
        streamline_offsets,
        box_min,
        box_max,
        results,
    )
    return results


def copy_streamlines_parallel(
    src_data: np.ndarray,
    dst_data: np.ndarray,
    src_starts: np.ndarray,
    dst_starts: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """Copy streamline data from source to destination buffer (parallel)."""
    from . import copy_streamlines_chunk

    n = len(lengths)
    parallel_chunks(
        copy_streamlines_chunk,
        n,
        src_data,
        dst_data,
        src_starts,
        dst_starts,
        lengths,
    )


def resample_batch(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    nb_points: int,
) -> np.ndarray:
    """Resample all streamlines to *nb_points* (parallel).

    Returns
    -------
    np.ndarray
        Shape ``(N, nb_points, 3)``.
    """
    from . import resample_batch_chunk

    n = len(lengths)
    result = np.empty((n, nb_points, 3), dtype=np.float64)
    parallel_chunks(
        resample_batch_chunk,
        n,
        flat_data,
        offsets,
        lengths,
        nb_points,
        result,
    )
    return result


def compute_mdf_distance_matrix(
    resampled: np.ndarray,
) -> np.ndarray:
    """Compute full MDF distance matrix (parallel).

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape ``(N, N)``.
    """
    from . import compute_mdf_rows_chunk

    n = resampled.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    parallel_chunks(compute_mdf_rows_chunk, n, resampled, dist_matrix)
    return dist_matrix


def compute_distances_to_samples(
    resampled: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Compute MDF distances from all streamlines to sampled subset (parallel).

    Returns
    -------
    np.ndarray
        Shape ``(N, k)`` distance array.
    """
    from . import compute_distances_chunk

    n = resampled.shape[0]
    k = len(sample_indices)
    distances = np.zeros((n, k), dtype=np.float64)
    parallel_chunks(
        compute_distances_chunk,
        n,
        resampled,
        sample_indices,
        distances,
    )
    return distances


def compute_endpoint_labels(
    start_points: np.ndarray,
    end_points: np.ndarray,
    inv_affine_3x3: np.ndarray,
    inv_affine_offset: np.ndarray,
    parcellation: np.ndarray,
    dims: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute endpoint labels from parcellation volume (parallel).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(start_labels, end_labels)`` each of shape ``(N,)`` int32.
    """
    from . import compute_labels_chunk

    n = start_points.shape[0]
    start_labels = np.zeros(n, dtype=np.int32)
    end_labels = np.zeros(n, dtype=np.int32)
    parallel_chunks(
        compute_labels_chunk,
        n,
        start_points,
        end_points,
        inv_affine_3x3,
        inv_affine_offset,
        parcellation,
        dims,
        start_labels,
        end_labels,
    )
    return start_labels, end_labels
