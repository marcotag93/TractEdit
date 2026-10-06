# -*- coding: utf-8 -*-

"""
Thread-based parallel dispatch for AOT-compiled chunk kernels.

AOT-compiled functions release the GIL, so ``ThreadPoolExecutor`` achieves
true parallelism without the copying overhead of process-based approaches.

The module provides:
- ``parallel_chunks``: distributes a chunk kernel across a reusable pool.
- Per-function wrappers that allocate output arrays, call
  ``parallel_chunks``, and return results.
"""

from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Tuple

import numpy as np

_CPU_COUNT: int = max(1, os.cpu_count() or 4)
_MAX_WORKERS: int = 8
_N_WORKERS: int = min(_CPU_COUNT, _MAX_WORKERS)
_MIN_PARALLEL_ITEMS: int = 2_048
_MIN_PARALLEL_MDF_ROWS: int = 256

# Module-level reusable thread pool (lazy-initialized).
_pool: ThreadPoolExecutor | None = None
_pool_lock: threading.Lock = threading.Lock()
_pool_condition = threading.Condition(_pool_lock)
_active_dispatches = 0
_pool_stopping = False


def _get_pool(n_workers: int | None = None) -> ThreadPoolExecutor:
    """Lease the stable eight-thread pool until ``_release_pool`` is called."""
    global _pool, _active_dispatches  # noqa: PLW0603
    workers = _N_WORKERS if n_workers is None else n_workers
    if workers < 1:
        raise ValueError("Worker count must be positive.")
    with _pool_condition:
        if _pool_stopping:
            raise RuntimeError("AOT worker pool is shutting down.")
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS,
                thread_name_prefix="tractedit-aot",
            )
        _active_dispatches += 1
        return _pool


def _release_pool() -> None:
    global _active_dispatches  # noqa: PLW0603
    with _pool_condition:
        _active_dispatches -= 1
        _pool_condition.notify_all()


def _submit_and_drain(pool: ThreadPoolExecutor, calls) -> None:
    """Propagate the first error only after every accepted task has stopped."""
    futures = []
    try:
        for function, arguments in calls:
            futures.append(pool.submit(function, *arguments))
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for future in futures:
            if future in done:
                future.result()
        for future in pending:
            future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        wait(futures)
        raise


def _worker_count(
    n_items: int,
    n_workers: int | None,
    min_parallel_items: int,
) -> int:
    """Return a bounded worker count, using serial execution for small work."""
    requested = _N_WORKERS if n_workers is None else n_workers
    if requested < 1:
        raise ValueError("Worker count must be positive.")
    if n_items < min_parallel_items:
        return 1
    return min(requested, _MAX_WORKERS, n_items)


def _shutdown_pool() -> None:
    """Reject new dispatches, drain accepted work, then stop the pool."""
    global _pool, _pool_stopping  # noqa: PLW0603
    with _pool_condition:
        if _pool_stopping:
            while _pool_stopping:
                _pool_condition.wait()
            return
        _pool_stopping = True
        while _active_dispatches:
            _pool_condition.wait()
        pool = _pool
    try:
        if pool is not None:
            pool.shutdown(wait=True)
    finally:
        with _pool_condition:
            _pool = None
            _pool_stopping = False
            _pool_condition.notify_all()


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

    workers = _worker_count(n_items, n_workers, _MIN_PARALLEL_ITEMS)
    chunk_size = max(1, (n_items + workers - 1) // workers)

    # Fast path: single chunk — skip pool overhead entirely.
    if n_items <= chunk_size or workers == 1:
        kernel_fn(*args, 0, n_items)
        return

    pool = _get_pool(workers)
    try:
        _submit_and_drain(
            pool,
            (
                (kernel_fn, (*args, start, min(start + chunk_size, n_items)))
                for start in range(0, n_items, chunk_size)
            ),
        )
    finally:
        _release_pool()


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

    workers = _worker_count(n_items, n_workers, _MIN_PARALLEL_MDF_ROWS)
    chunk_size = max(1, (n_items + workers - 1) // workers)

    # Fast path: single chunk — skip pool overhead.
    if n_items <= chunk_size or workers == 1:
        kernel_fn(*args, start_offset, end_offset)
        return

    pool = _get_pool(workers)
    try:
        _submit_and_drain(
            pool,
            (
                (kernel_fn, (*args, start, min(start + chunk_size, end_offset)))
                for start in range(start_offset, end_offset, chunk_size)
            ),
        )
    finally:
        _release_pool()


def accumulate_mdf_totals_range(
    resampled: np.ndarray,
    start_offset: int,
    end_offset: int,
    n_workers: int | None = None,
) -> np.ndarray:
    """Accumulate exact MDF totals for an upper-triangle row range."""
    from . import accumulate_mdf_totals_chunk

    count = end_offset - start_offset
    totals = np.zeros(resampled.shape[0], dtype=np.float64)
    if count <= 0:
        return totals

    workers = _worker_count(count, n_workers, _MIN_PARALLEL_MDF_ROWS)
    chunk_size = max(1, (count + workers - 1) // workers)
    ranges = [
        (start, min(start + chunk_size, end_offset))
        for start in range(start_offset, end_offset, chunk_size)
    ]
    partial_totals = np.zeros((len(ranges), resampled.shape[0]), dtype=np.float64)

    if len(ranges) == 1:
        accumulate_mdf_totals_chunk(
            resampled, partial_totals[0], ranges[0][0], ranges[0][1]
        )
    else:
        pool = _get_pool(workers)
        try:
            _submit_and_drain(
                pool,
                (
                    (
                        accumulate_mdf_totals_chunk,
                        (resampled, partial_totals[index], start, end),
                    )
                    for index, (start, end) in enumerate(ranges)
                ),
            )
        finally:
            _release_pool()

    np.sum(partial_totals, axis=0, out=totals)
    return totals


# ====================================================================
# Phase 3 — Per-function wrappers
# ====================================================================


def _packed_arrays(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    data_dtype: np.dtype = np.dtype(np.float32),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_data = np.asarray(flat_data)
    offsets = np.asarray(offsets)
    lengths = np.asarray(lengths)
    if flat_data.ndim != 2 or flat_data.shape[1] != 3:
        raise ValueError("Streamline coordinates must have shape (N, 3).")
    if offsets.ndim != 1 or lengths.ndim != 1 or len(offsets) != len(lengths):
        raise ValueError("Streamline offsets and lengths must be aligned vectors.")
    if not np.issubdtype(offsets.dtype, np.integer) or not np.issubdtype(
        lengths.dtype, np.integer
    ):
        raise ValueError("Streamline offsets and lengths must be integers.")
    if np.any(offsets < 0) or np.any(offsets > len(flat_data)):
        raise ValueError("A streamline offset is outside the coordinate buffer.")
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if np.any(lengths < 0) or np.any(lengths > len(flat_data) - offsets):
        raise ValueError("A streamline length exceeds the coordinate buffer.")
    if len(offsets) > 1 and np.any(offsets[1:] < offsets[:-1] + lengths[:-1]):
        raise ValueError("Streamline offsets overlap or are not ordered.")
    return (
        np.ascontiguousarray(flat_data, dtype=data_dtype),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(lengths, dtype=np.int64),
    )


def compute_bboxes(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    _validated: bool = False,
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

    if _validated:
        flat_data = np.ascontiguousarray(flat_data, dtype=np.float32)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        lengths = np.ascontiguousarray(lengths, dtype=np.int64)
    else:
        flat_data, offsets, lengths = _packed_arrays(flat_data, offsets, lengths)

    n = len(lengths)
    bboxes = np.zeros((n, 2, 3), dtype=np.float32)
    parallel_chunks(compute_bboxes_chunk, n, flat_data, offsets, lengths, bboxes)
    return bboxes


def compute_bboxes_prevalidated(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Compute boxes after the ingestion boundary validated packed arrays."""
    return compute_bboxes(
        flat_data,
        offsets,
        lengths,
        _validated=True,
    )


def validate_streamlines(
    flat_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    _validated: bool = False,
) -> np.ndarray:
    """Return per-streamline coordinate status without recomputing boxes."""
    from . import validate_streamlines_chunk

    if _validated:
        flat_data = np.ascontiguousarray(flat_data, dtype=np.float32)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        lengths = np.ascontiguousarray(lengths, dtype=np.int64)
    else:
        flat_data, offsets, lengths = _packed_arrays(flat_data, offsets, lengths)
    status = np.zeros(len(lengths), dtype=np.uint8)
    parallel_chunks(
        validate_streamlines_chunk,
        len(lengths),
        flat_data,
        offsets,
        lengths,
        status,
    )
    return status


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


def batch_check_oriented_box_intersection(
    streamline_data: np.ndarray,
    streamline_offsets: np.ndarray,
    world_to_voxel_linear: np.ndarray,
    world_to_voxel_offset: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray:
    """Batch check streamline intersection with a voxel-oriented box."""
    from . import check_oriented_box_chunk

    n = len(streamline_offsets) - 1
    results = np.zeros(n, dtype=np.bool_)
    parallel_chunks(
        check_oriented_box_chunk,
        n,
        streamline_data,
        streamline_offsets,
        world_to_voxel_linear,
        world_to_voxel_offset,
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

    flat_data, offsets, lengths = _packed_arrays(
        flat_data,
        offsets,
        lengths,
        np.dtype(np.float64),
    )
    if np.any(lengths < 1):
        raise ValueError("Cannot resample an empty streamline.")
    if nb_points < 2:
        raise ValueError("Resampling requires at least two output points.")
    if not np.all(np.isfinite(flat_data)):
        raise ValueError("Streamline coordinates must be finite.")
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
