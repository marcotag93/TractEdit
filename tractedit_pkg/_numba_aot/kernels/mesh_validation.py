# -*- coding: utf-8 -*-

"""Mesh point and triangle validation AOT kernels for TractEdit."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numba.pycc import CC


@njit
def _validate_points_impl(points, out_bounds, out_ok, start_i, end_i):
    if start_i >= end_i:
        out_ok[0] = 1
        return
    min_x = np.inf
    min_y = np.inf
    min_z = np.inf
    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf
    for i in range(start_i, end_i):
        x = np.float64(points[i, 0])
        y = np.float64(points[i, 1])
        z = np.float64(points[i, 2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            out_ok[0] = 0
            return
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
        if z < min_z:
            min_z = z
        if z > max_z:
            max_z = z
    out_bounds[0] = min_x
    out_bounds[1] = min_y
    out_bounds[2] = min_z
    out_bounds[3] = max_x
    out_bounds[4] = max_y
    out_bounds[5] = max_z
    out_ok[0] = 1


@njit
def _validate_triangles_impl(
    points, indices, allow_zero_area, status, start_face, end_face
):
    has_area = np.int64(0)
    for face in range(start_face, end_face):
        base = face * 3
        i0 = indices[base]
        i1 = indices[base + 1]
        i2 = indices[base + 2]
        if i0 == i1 or i1 == i2 or i2 == i0:
            status[0] = 1
            return
        x0 = np.float64(points[i0, 0])
        y0 = np.float64(points[i0, 1])
        z0 = np.float64(points[i0, 2])
        x1 = np.float64(points[i1, 0])
        y1 = np.float64(points[i1, 1])
        z1 = np.float64(points[i1, 2])
        x2 = np.float64(points[i2, 0])
        y2 = np.float64(points[i2, 1])
        z2 = np.float64(points[i2, 2])
        scale = abs(x0)
        for v in (abs(y0), abs(z0), abs(x1), abs(y1), abs(z1), abs(x2), abs(y2), abs(z2)):
            if v > scale:
                scale = v
        if scale == 0.0:
            if allow_zero_area == 0:
                status[0] = 2
                return
            continue
        _, exp = math.frexp(scale)
        neg_exp = -exp
        sx0 = math.ldexp(x0, neg_exp)
        sy0 = math.ldexp(y0, neg_exp)
        sz0 = math.ldexp(z0, neg_exp)
        e1x = math.ldexp(x1, neg_exp) - sx0
        e1y = math.ldexp(y1, neg_exp) - sy0
        e1z = math.ldexp(z1, neg_exp) - sz0
        e2x = math.ldexp(x2, neg_exp) - sx0
        e2y = math.ldexp(y2, neg_exp) - sy0
        e2z = math.ldexp(z2, neg_exp) - sz0
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        if cx != 0.0 or cy != 0.0 or cz != 0.0:
            has_area = 1
        elif allow_zero_area == 0:
            status[0] = 2
            return
    status[0] = 0
    status[1] = has_area


def register_mesh_validation_kernels(compiler: CC) -> None:
    """Register mesh point and triangle validation AOT kernels."""

    @compiler.export(
        "validate_points_f32_chunk",
        "void(float32[:,::1], float64[::1], uint8[::1], int64, int64)",
    )
    def validate_points_f32_chunk(points, out_bounds, out_ok, start_i, end_i):
        _validate_points_impl(points, out_bounds, out_ok, start_i, end_i)

    @compiler.export(
        "validate_points_f64_chunk",
        "void(float64[:,::1], float64[::1], uint8[::1], int64, int64)",
    )
    def validate_points_f64_chunk(points, out_bounds, out_ok, start_i, end_i):
        _validate_points_impl(points, out_bounds, out_ok, start_i, end_i)

    @compiler.export(
        "validate_triangles_f32_chunk",
        "void(float32[:,::1], int64[::1], uint8, int64[::1], int64, int64)",
    )
    def validate_triangles_f32_chunk(
        points, indices, allow_zero_area, status, start_face, end_face
    ):
        _validate_triangles_impl(
            points, indices, allow_zero_area, status, start_face, end_face
        )

    @compiler.export(
        "validate_triangles_f64_chunk",
        "void(float64[:,::1], int64[::1], uint8, int64[::1], int64, int64)",
    )
    def validate_triangles_f64_chunk(
        points, indices, allow_zero_area, status, start_face, end_face
    ):
        _validate_triangles_impl(
            points, indices, allow_zero_area, status, start_face, end_face
        )
