from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import trx.trx_file_memmap as tbx

import hashlib
import zlib
import numpy as np
import logging
from ..reference_grid import validate_affine as _validate_affine
from typing import Any, Dict, Optional, Tuple
from tractedit_pkg._numba_aot._parallel_wrappers import (
    compute_bboxes_prevalidated as _compute_bboxes_prevalidated,
    validate_streamlines as _validate_streamlines_numba,
)
from .. import file_io

logger = logging.getLogger(__name__)


def _packed_streamline_arrays(
    streamlines: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from ..input_validation import packed_streamline_arrays

    return packed_streamline_arrays(streamlines)


def _array_is_finite(array: np.ndarray, block_bytes: int = 4 * 1024 * 1024) -> bool:
    from ..input_validation import array_is_finite

    return array_is_finite(array, block_bytes)


def _validate_scalar_data(
    scalar_data: Dict[str, Any],
    streamline_lengths: np.ndarray,
) -> None:
    for name, sequence in scalar_data.items():
        if len(sequence) != len(streamline_lengths):
            raise ValueError(
                f"Scalar {name!r} has {len(sequence)} streamlines; "
                f"expected {len(streamline_lengths)}."
            )
        scalar_lengths = getattr(sequence, "_lengths", None)
        if scalar_lengths is not None and not np.array_equal(
            np.asarray(scalar_lengths, dtype=np.int64), streamline_lengths
        ):
            raise ValueError(f"Scalar {name!r} does not match streamline lengths.")
        values = getattr(sequence, "_data", None)
        if values is not None:
            if not _array_is_finite(np.asarray(values)):
                raise ValueError(f"Scalar {name!r} contains non-finite values.")
            continue
        for index, item in enumerate(sequence):
            array = np.asarray(item)
            if len(array) != streamline_lengths[index]:
                raise ValueError(
                    f"Scalar {name!r} does not match streamline {index}."
                )
            if not _array_is_finite(array):
                raise ValueError(f"Scalar {name!r} contains non-finite values.")


def _validate_and_compute_bboxes(
    streamlines: Any,
    affine: np.ndarray,
    scalar_data: Dict[str, Any],
    cached_bboxes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Validate one loaded tractogram and return authoritative bounding boxes."""
    _validate_affine(affine, "Tractogram")
    packed = _packed_streamline_arrays(streamlines)
    if packed is None:
        bboxes = []
        lengths = []
        for index, streamline in enumerate(streamlines):
            coordinates = np.asarray(streamline)
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError(
                    f"Streamline {index} must have shape (N, 3), "
                    f"got {coordinates.shape}."
                )
            if len(coordinates) == 0:
                raise ValueError(f"Tractogram contains an empty streamline at index {index}.")
            if len(coordinates) == 1:
                raise ValueError(
                    f"Tractogram contains a single-point streamline at index {index}."
                )
            if not _array_is_finite(coordinates):
                raise ValueError(
                    f"Tractogram contains non-finite coordinates at index {index}."
                )
            minimum = np.min(coordinates, axis=0)
            maximum = np.max(coordinates, axis=0)
            bboxes.append([minimum, maximum])
            lengths.append(len(coordinates))
        _validate_scalar_data(scalar_data, np.asarray(lengths, dtype=np.int64))
        return np.asarray(bboxes, dtype=np.float32)

    data, offsets, lengths = packed
    _validate_scalar_data(scalar_data, lengths)
    if cached_bboxes is None:
        bboxes = _compute_bboxes_prevalidated(
            data,
            offsets,
            lengths,
        )
        nonfinite = np.flatnonzero(~np.all(np.isfinite(bboxes), axis=(1, 2)))
        if nonfinite.size:
            raise ValueError(
                "Tractogram contains non-finite coordinates at index "
                f"{nonfinite[0]}."
            )
        return bboxes

    status = _validate_streamlines_numba(
        data,
        offsets,
        lengths,
        _validated=True,
    )
    nonfinite = np.flatnonzero(status == 1)
    if nonfinite.size:
        raise ValueError(
            f"Tractogram contains non-finite coordinates at index {nonfinite[0]}."
        )
    return cached_bboxes

_TRX_BBOX_KEY = "_tractedit_bboxes"
_TRX_BBOX_CACHE_VERSION = 1
_TRX_BBOX_CACHE_VERSION_KEY = "TRACTEDIT_BBOX_CACHE_VERSION"
_TRX_BBOX_CACHE_DIGEST_KEY = "TRACTEDIT_BBOX_CACHE_DIGEST"
_TRX_BBOX_GEOMETRY_DIGEST_KEY = "TRACTEDIT_BBOX_GEOMETRY_SAMPLE_DIGEST"
_TRX_BBOX_SAMPLE_STREAMLINES = 64
_TRX_BBOX_SAMPLE_POINTS = 16
_TRX_BBOX_HEADER_KEYS = (
    _TRX_BBOX_CACHE_VERSION_KEY,
    _TRX_BBOX_CACHE_DIGEST_KEY,
    _TRX_BBOX_GEOMETRY_DIGEST_KEY,
)


# ============================================================================
# TRX Bbox Caching Helpers (Phase 1)
# ============================================================================


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=limit, dtype=np.int64)


def _bbox_cache_digest(flat_bboxes: np.ndarray) -> str:
    canonical = np.ascontiguousarray(flat_bboxes, dtype="<f4")
    checksum = zlib.crc32(np.asarray(canonical.shape, dtype="<u8").tobytes())
    checksum = zlib.crc32(canonical, checksum)
    return f"{checksum:08x}"


def _bbox_geometry_sample_digest(
    streamlines: Any,
    n_streamlines: int,
) -> Optional[str]:
    if streamlines is None or len(streamlines) != n_streamlines:
        return None

    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([n_streamlines], dtype="<u8").tobytes())
    streamline_indices = _sample_indices(
        n_streamlines,
        _TRX_BBOX_SAMPLE_STREAMLINES,
    )
    for streamline_index in streamline_indices:
        points = np.asarray(streamlines[int(streamline_index)])
        if points.ndim != 2 or points.shape[1] != 3:
            return None
        point_indices = _sample_indices(len(points), _TRX_BBOX_SAMPLE_POINTS)
        digest.update(
            np.asarray(
                [int(streamline_index), len(points)],
                dtype="<u8",
            ).tobytes()
        )
        digest.update(np.asarray(point_indices, dtype="<u8").tobytes())
        digest.update(np.ascontiguousarray(points[point_indices], dtype="<f4"))
    return digest.hexdigest()


def _discard_cached_bboxes(trx_obj: "tbx.TrxFile") -> None:
    data_per_streamline = getattr(trx_obj, "data_per_streamline", None)
    if data_per_streamline is not None:
        data_per_streamline.pop(_TRX_BBOX_KEY, None)
    header = getattr(trx_obj, "header", None)
    if header is not None:
        for key in _TRX_BBOX_HEADER_KEYS:
            header.pop(key, None)


def _try_load_cached_bboxes(
    trx_obj: "tbx.TrxFile",
    n_streamlines: int,
    *,
    validate_geometry: bool = False,
) -> Optional[np.ndarray]:
    """
    Attempt to load cached bounding boxes from a TRX file's
    ``data_per_streamline``.

    The bboxes are stored as a flat ``(N, 6)`` float32 array
    (``[min_x, min_y, min_z, max_x, max_y, max_z]``) and are reshaped
    to the application-standard ``(N, 2, 3)`` layout on load.

    Args:
        trx_obj: The loaded TRX file object (memmap-backed).
        n_streamlines: Expected number of streamlines for shape validation.
        validate_geometry: Recompute every bounding box and require exact
            equality. The default keeps the bounded cache-hit path.

    Returns:
        Bounding boxes as ``np.ndarray`` of shape ``(N, 2, 3)``
        and dtype ``float32``, or ``None`` if the cache is absent,
        corrupted, or has the wrong shape.
    """
    try:
        dps = getattr(trx_obj, "data_per_streamline", None)
        if dps is None or _TRX_BBOX_KEY not in dps:
            return None

        header = getattr(trx_obj, "header", None)
        if (
            header is None
            or header.get(_TRX_BBOX_CACHE_VERSION_KEY) != _TRX_BBOX_CACHE_VERSION
        ):
            logger.warning(
                "TRX bbox cache: unsupported or missing schema, recomputing."
            )
            return None

        raw = dps[_TRX_BBOX_KEY]
        if not isinstance(raw, np.ndarray):
            logger.warning(
                "TRX bbox cache: unexpected type %s, recomputing.",
                type(raw).__name__,
            )
            return None

        if raw.shape != (n_streamlines, 6):
            logger.warning(
                "TRX bbox cache: shape mismatch (expected (%d, 6), got %s), "
                "recomputing.",
                n_streamlines,
                raw.shape,
            )
            return None

        if not np.issubdtype(raw.dtype, np.number):
            logger.warning("TRX bbox cache: nonnumeric data, recomputing.")
            return None

        flat = np.ascontiguousarray(raw, dtype=np.float32)
        cached = flat.reshape(-1, 2, 3)
        if not np.all(np.isfinite(cached)):
            logger.warning("TRX bbox cache: nonfinite bounds, recomputing.")
            return None
        if not np.all(cached[:, 0] <= cached[:, 1]):
            logger.warning("TRX bbox cache: inverted bounds, recomputing.")
            return None

        expected_cache_digest = header.get(_TRX_BBOX_CACHE_DIGEST_KEY)
        if (
            not isinstance(expected_cache_digest, str)
            or _bbox_cache_digest(flat) != expected_cache_digest
        ):
            logger.warning("TRX bbox cache: payload provenance mismatch, recomputing.")
            return None

        expected_geometry_digest = header.get(_TRX_BBOX_GEOMETRY_DIGEST_KEY)
        geometry_digest = _bbox_geometry_sample_digest(
            getattr(trx_obj, "streamlines", None),
            n_streamlines,
        )
        if (
            not isinstance(expected_geometry_digest, str)
            or geometry_digest != expected_geometry_digest
        ):
            logger.warning("TRX bbox cache: geometry provenance mismatch, recomputing.")
            return None

        if validate_geometry:
            streamlines = trx_obj.streamlines
            computed = file_io._compute_bboxes_numba(
                streamlines._data,
                streamlines._offsets,
                streamlines._lengths,
            )
            if not np.array_equal(cached, computed):
                logger.warning(
                    "TRX bbox cache: full geometry validation failed, recomputing."
                )
                return None

        logger.info(
            "TRX: loaded cached bounding boxes from "
            "data_per_streamline (%d streamlines, skipped computation).",
            n_streamlines,
        )
        return cached

    except (AttributeError, ValueError, KeyError, TypeError, IndexError) as exc:
        logger.warning("TRX bbox cache: failed to read (%s), recomputing.", exc)
        return None


def _embed_cached_bboxes(
    trx_obj: "tbx.TrxFile",
    bboxes: Optional[np.ndarray],
) -> None:
    """
    Embed precomputed bounding boxes into a TRX object before saving.

    The bboxes are flattened from the application-standard ``(N, 2, 3)``
    layout to ``(N, 6)`` float32 for TRX ``data_per_streamline`` storage.

    Args:
        trx_obj: The TRX object that will be saved.
        bboxes: Bounding boxes of shape ``(N, 2, 3)``.  If ``None`` or
            empty, no embedding is performed.
    """
    if bboxes is None or bboxes.size == 0:
        return

    try:
        streamlines = getattr(trx_obj, "streamlines", None)
        n_streamlines = len(streamlines) if streamlines is not None else -1
        if bboxes.shape != (n_streamlines, 2, 3):
            raise ValueError(f"expected ({n_streamlines}, 2, 3), got {bboxes.shape}")
        flat = np.ascontiguousarray(bboxes.reshape(-1, 6), dtype=np.float32)
        cached = flat.reshape(-1, 2, 3)
        if not np.all(np.isfinite(cached)) or not np.all(cached[:, 0] <= cached[:, 1]):
            raise ValueError("bounding boxes must be finite and ordered")
        geometry_digest = _bbox_geometry_sample_digest(streamlines, n_streamlines)
        if geometry_digest is None:
            raise ValueError("streamline geometry is unavailable")
        header = getattr(trx_obj, "header", None)
        if header is None:
            raise ValueError("TRX header is unavailable")

        trx_obj.data_per_streamline[_TRX_BBOX_KEY] = flat
        header[_TRX_BBOX_CACHE_VERSION_KEY] = _TRX_BBOX_CACHE_VERSION
        header[_TRX_BBOX_CACHE_DIGEST_KEY] = _bbox_cache_digest(flat)
        header[_TRX_BBOX_GEOMETRY_DIGEST_KEY] = geometry_digest
        logger.debug(
            "TRX: embedded cached bounding boxes (%d streamlines).",
            flat.shape[0],
        )
    except (AttributeError, ValueError, TypeError, IndexError) as exc:
        _discard_cached_bboxes(trx_obj)
        logger.warning(
            "TRX: failed to embed cached bboxes (%s). "
            "Saving continues without bbox cache.",
            exc,
        )


def _extract_visible_bboxes(main_window: Any) -> Optional[np.ndarray]:
    """
    Extract bounding boxes for the currently visible streamlines.

    Returns a contiguous ``(M, 2, 3)`` float32 array indexed by the
    sorted visible indices, or ``None`` if bboxes are not available.

    Args:
        main_window: The MainWindow instance holding application state.

    Returns:
        Bounding boxes array or ``None``.
    """
    if main_window.streamline_bboxes is None or not main_window.visible_indices:
        return None

    indices = sorted(main_window.visible_indices)
    return main_window.streamline_bboxes[indices]
