# -*- coding: utf-8 -*-

"""Typed mapping contracts shared across application managers."""

from typing import Any, NotRequired, TypedDict

import numpy as np

from .reference_grid import ReferenceGrid


class StreamlineLoadResult(TypedDict):
    """Payload emitted after a successful tractogram load."""

    path: str
    ext: str
    streamlines: Any
    header: Any
    affine: np.ndarray
    scalars: dict[str, Any]
    active_scalar: str | None
    data_per_streamline: dict[str, Any]
    bboxes: np.ndarray
    reference_grid: ReferenceGrid | None
    trx_obj: NotRequired[Any]


class AnatomicalImageLoadResult(TypedDict):
    """Payload emitted after a successful anatomical image load."""

    data: np.ndarray
    affine: np.ndarray
    path: str
    was_downsampled: bool
    mmap_image: Any
    reference_grid: ReferenceGrid


class RoiLayer(TypedDict):
    """Shared state stored for one ROI layer."""

    data: np.ndarray
    affine: np.ndarray
    inv_affine: np.ndarray
    T_main_to_roi: np.ndarray
    path: NotRequired[str]
    color: NotRequired[tuple[float, float, float]]
    display_name: NotRequired[str]
