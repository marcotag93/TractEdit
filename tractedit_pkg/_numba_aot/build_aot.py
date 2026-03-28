# -*- coding: utf-8 -*-

"""
AOT compilation script for TractEdit Numba functions.

Compiles all performance-critical numerical kernels into a platform-specific
shared library (tractedit_numba.pyd on Windows, .so on Unix) using Numba's
Ahead-of-Time compilation via ``numba.pycc.CC``.

Usage:
    python tractedit_pkg/_numba_aot/build_aot.py

Output:
    tractedit_pkg/_numba_aot/tractedit_numba.<ext>

The compiled module is imported at runtime by ``__init__.py`` without
requiring Numba to be installed.

Notes:
    On Windows, if MSVC is not in PATH, the script will attempt to locate
    Visual Studio Build Tools and activate the environment automatically.
    You can also set ``DISTUTILS_USE_SDK=1`` and ``MSSdk=1`` manually after
    running ``vcvars64.bat``.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

import numpy as np  # noqa: F401 — required by Numba type resolution
from numba.pycc import CC

# ---------------------------------------------------------------------------
# Output directory: place the compiled extension next to this script.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Windows MSVC auto-detection
# ---------------------------------------------------------------------------


def _ensure_msvc_env() -> None:
    """Activate MSVC environment on Windows if not already available.

    Sets ``DISTUTILS_USE_SDK`` and ``MSSdk`` so that ``distutils``
    picks up the pre-configured SDK environment rather than trying
    (and failing) to auto-detect Visual Studio installations.
    """
    if sys.platform != "win32":
        return

    # Already configured?
    if os.environ.get("DISTUTILS_USE_SDK") == "1":
        return

    # Try to find vcvarsall.bat
    patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    vcvarsall = None
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            vcvarsall = matches[-1]  # Latest version
            break

    if vcvarsall is None:
        print(
            "WARNING: Could not find vcvarsall.bat. "
            "MSVC may not be detected by distutils.",
            file=sys.stderr,
        )
        return

    print(f"Activating MSVC environment via: {vcvarsall}")

    # Run vcvarsall.bat and capture the resulting environment.
    # We use cmd.exe explicitly with a fixed argument list to avoid
    # shell injection via the vcvarsall path.
    result = subprocess.run(
        ["cmd.exe", "/C", vcvarsall, "amd64", ">nul", "2>&1", "&&", "set"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(
            f"WARNING: vcvarsall.bat failed (exit {result.returncode})",
            file=sys.stderr,
        )
        return

    # Only import the environment variables required by distutils/MSVC.
    _MSVC_ENV_WHITELIST = {
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "PATH",
        "VSINSTALLDIR",
        "VCINSTALLDIR",
        "WINDOWSSDKDIR",
        "WINDOWSSDKVERSION",
        "UCRTVERSION",
        "VCTOOLSINSTALLDIR",
        "VCTOOLSREDISTDIR",
        "VSCMD_ARG_HOST_ARCH",
        "VSCMD_ARG_TGT_ARCH",
    }
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key in _MSVC_ENV_WHITELIST:
                os.environ[key] = value

    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"
    print("MSVC environment activated successfully.")


# ---------------------------------------------------------------------------
# Compiler setup
# ---------------------------------------------------------------------------


def _create_compiler() -> CC:
    """Create and configure the Numba AOT compiler."""
    compiler = CC("tractedit_numba")
    compiler.verbose = True

    # Use 'host' for local dev builds (fastest on the build machine).
    # CI/release builds should override to 'generic' for portability.
    compiler.target_cpu = os.environ.get("TRACTEDIT_AOT_TARGET_CPU", "host")

    # Place the compiled extension next to this script.
    compiler.output_dir = _SCRIPT_DIR

    return compiler


# ---------------------------------------------------------------------------
# AOT function definitions
# ---------------------------------------------------------------------------
# Functions are registered on a CC instance via ``_register_functions(cc)``.
# This keeps them in one place and avoids module-level side effects.


def _register_functions(compiler: CC) -> None:
    """Register all AOT-exported functions on the given CC instance."""

    # ===================================================================
    # Phase 2 — Serial functions
    # ===================================================================

    # -- 2A: Sphere-streamline intersection (from selection.py:36) ------

    @compiler.export(
        "check_streamline_sphere_intersection",
        "boolean(float64[:,::1], float64[::1], float64)",
    )
    def check_streamline_sphere_intersection(streamline, center, radius_sq):
        """Check if a streamline intersects a sphere.

        Tests vertex distances first (fast path), then segment-to-center
        distances for a precise check.
        """
        n_pts = streamline.shape[0]
        if n_pts == 0:
            return False

        # Fast path: check vertices
        for i in range(n_pts):
            dx = streamline[i, 0] - center[0]
            dy = streamline[i, 1] - center[1]
            dz = streamline[i, 2] - center[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < radius_sq:
                return True

        # Precise path: check segments
        for i in range(n_pts - 1):
            p1x = streamline[i, 0]
            p1y = streamline[i, 1]
            p1z = streamline[i, 2]
            p2x = streamline[i + 1, 0]
            p2y = streamline[i + 1, 1]
            p2z = streamline[i + 1, 2]

            seg_x = p2x - p1x
            seg_y = p2y - p1y
            seg_z = p2z - p1z

            pc_x = center[0] - p1x
            pc_y = center[1] - p1y
            pc_z = center[2] - p1z

            seg_len_sq = seg_x * seg_x + seg_y * seg_y + seg_z * seg_z
            if seg_len_sq == 0:
                continue

            t = (pc_x * seg_x + pc_y * seg_y + pc_z * seg_z) / seg_len_sq
            if t < 0:
                t = 0.0
            elif t > 1:
                t = 1.0

            closest_x = p1x + t * seg_x
            closest_y = p1y + t * seg_y
            closest_z = p1z + t * seg_z

            dx = closest_x - center[0]
            dy = closest_y - center[1]
            dz = closest_z - center[2]
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < radius_sq:
                return True

        return False

    # -- 2B: Streamline resampling (from file_io.py:960) ----------------

    @compiler.export(
        "resample_streamline",
        "float64[:,::1](float64[:,::1], int64)",
    )
    def resample_streamline(streamline, nb_points):
        """Resample a streamline to *nb_points* using linear interpolation."""
        n_pts = streamline.shape[0]

        if n_pts <= 1:
            result = np.empty((nb_points, 3), dtype=np.float64)
            for i in range(nb_points):
                result[i, 0] = streamline[0, 0]
                result[i, 1] = streamline[0, 1]
                result[i, 2] = streamline[0, 2]
            return result

        # Cumulative arc-length distances
        cum_dists = np.zeros(n_pts, dtype=np.float64)
        for i in range(1, n_pts):
            dx = streamline[i, 0] - streamline[i - 1, 0]
            dy = streamline[i, 1] - streamline[i - 1, 1]
            dz = streamline[i, 2] - streamline[i - 1, 2]
            cum_dists[i] = cum_dists[i - 1] + np.sqrt(dx * dx + dy * dy + dz * dz)

        total_length = cum_dists[-1]
        if total_length == 0:
            result = np.empty((nb_points, 3), dtype=np.float64)
            for i in range(nb_points):
                result[i, 0] = streamline[0, 0]
                result[i, 1] = streamline[0, 1]
                result[i, 2] = streamline[0, 2]
            return result

        # Interpolate at uniform arc-length intervals
        result = np.empty((nb_points, 3), dtype=np.float64)
        for i in range(nb_points):
            target_dist = total_length * i / (nb_points - 1)

            # Binary search: cum_dists is monotonically non-decreasing,
            # find the first index where cum_dists >= target_dist.
            lo = 0
            hi = n_pts - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum_dists[mid] < target_dist:
                    lo = mid + 1
                else:
                    hi = mid
            # seg_idx is the segment start: one before the found index
            seg_idx = lo - 1 if lo > 0 else 0

            seg_start = cum_dists[seg_idx]
            seg_end = cum_dists[seg_idx + 1] if seg_idx + 1 < n_pts else seg_start
            seg_len = seg_end - seg_start

            t = (target_dist - seg_start) / seg_len if seg_len > 0 else 0.0

            result[i, 0] = streamline[seg_idx, 0] + t * (
                streamline[seg_idx + 1, 0] - streamline[seg_idx, 0]
            )
            result[i, 1] = streamline[seg_idx, 1] + t * (
                streamline[seg_idx + 1, 1] - streamline[seg_idx, 1]
            )
            result[i, 2] = streamline[seg_idx, 2] + t * (
                streamline[seg_idx + 1, 2] - streamline[seg_idx, 2]
            )

        return result

    # -- 2C: ROI-streamline intersection (from roi_manager.py:45) -------

    @compiler.export(
        "check_streamline_roi_intersection",
        "boolean(float64[:,::1], float64[:,::1], float64[::1],"
        " uint8[:,:,::1], int64[::1])",
    )
    def check_streamline_roi_intersection(streamline, R, T, roi_data, dims):
        """Check if a streamline intersects an ROI volume.

        Transforms each point to voxel coordinates via the inverse affine
        (split into rotation *R* and translation *T*), then looks up the
        ROI volume.
        """
        n_pts = streamline.shape[0]

        for i in range(n_pts):
            vx = (
                streamline[i, 0] * R[0, 0]
                + streamline[i, 1] * R[1, 0]
                + streamline[i, 2] * R[2, 0]
                + T[0]
            )
            vy = (
                streamline[i, 0] * R[0, 1]
                + streamline[i, 1] * R[1, 1]
                + streamline[i, 2] * R[2, 1]
                + T[1]
            )
            vz = (
                streamline[i, 0] * R[0, 2]
                + streamline[i, 1] * R[1, 2]
                + streamline[i, 2] * R[2, 2]
                + T[2]
            )

            ix = int(np.round(vx))
            iy = int(np.round(vy))
            iz = int(np.round(vz))

            if (
                ix >= 0
                and ix < dims[0]
                and iy >= 0
                and iy < dims[1]
                and iz >= 0
                and iz < dims[2]
            ):
                if roi_data[ix, iy, iz] > 0:
                    return True

        return False

    # -- 2D: Centroid with MDF alignment (from file_io.py:1131) ---------
    #    Note: the original had parallel=True but the loop is sequential
    #    (accumulator dependency). Compiled as a serial AOT function.

    @compiler.export(
        "compute_centroid",
        "float64[:,::1](float64[:,:,::1])",
    )
    def compute_centroid(resampled):
        """Compute the mean streamline (centroid) with MDF alignment.

        Each streamline is compared to the reference (first streamline)
        in both direct and flipped orientation.  The closer alignment
        is accumulated to produce the average.
        """
        n = resampled.shape[0]
        nb_points = resampled.shape[1]

        if n == 0:
            return np.zeros((nb_points, 3), dtype=np.float64)

        ref = resampled[0]
        centroid = np.zeros((nb_points, 3), dtype=np.float64)

        # Seed with the reference streamline
        for k in range(nb_points):
            centroid[k, 0] = ref[k, 0]
            centroid[k, 1] = ref[k, 1]
            centroid[k, 2] = ref[k, 2]

        # Align and accumulate remaining streamlines
        for i in range(1, n):
            s = resampled[i]

            # Direct distance
            d_direct = 0.0
            for k in range(nb_points):
                dx = ref[k, 0] - s[k, 0]
                dy = ref[k, 1] - s[k, 1]
                dz = ref[k, 2] - s[k, 2]
                d_direct += np.sqrt(dx * dx + dy * dy + dz * dz)
            d_direct /= nb_points

            # Flipped distance
            d_flipped = 0.0
            for k in range(nb_points):
                flipped_k = nb_points - 1 - k
                dx = ref[k, 0] - s[flipped_k, 0]
                dy = ref[k, 1] - s[flipped_k, 1]
                dz = ref[k, 2] - s[flipped_k, 2]
                d_flipped += np.sqrt(dx * dx + dy * dy + dz * dz)
            d_flipped /= nb_points

            # Accumulate the better-aligned version
            if d_flipped < d_direct:
                for k in range(nb_points):
                    flipped_k = nb_points - 1 - k
                    centroid[k, 0] += s[flipped_k, 0]
                    centroid[k, 1] += s[flipped_k, 1]
                    centroid[k, 2] += s[flipped_k, 2]
            else:
                for k in range(nb_points):
                    centroid[k, 0] += s[k, 0]
                    centroid[k, 1] += s[k, 1]
                    centroid[k, 2] += s[k, 2]

        # Average
        for k in range(nb_points):
            centroid[k, 0] /= n
            centroid[k, 1] /= n
            centroid[k, 2] /= n

        return centroid

    # ===================================================================
    # Phase 3 — Chunk kernels (process start_i..end_i per thread)
    # ===================================================================

    # -- 3A: Bounding-box computation (from file_io.py:766) ------------

    @compiler.export(
        "compute_bboxes_chunk",
        "void(float32[:,::1], int64[::1], int64[::1],"
        " float32[:,:,::1], int64, int64)",
    )
    def compute_bboxes_chunk(flat_data, offsets, lengths, bboxes, start_i, end_i):
        """Compute bounding boxes for streamlines in [start_i, end_i)."""
        for i in range(start_i, end_i):
            start = offsets[i]
            length = lengths[i]

            if length == 0:
                continue

            first_idx = start
            min_x = flat_data[first_idx, 0]
            max_x = flat_data[first_idx, 0]
            min_y = flat_data[first_idx, 1]
            max_y = flat_data[first_idx, 1]
            min_z = flat_data[first_idx, 2]
            max_z = flat_data[first_idx, 2]

            for j in range(1, length):
                idx = start + j
                x = flat_data[idx, 0]
                y = flat_data[idx, 1]
                z = flat_data[idx, 2]

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

            bboxes[i, 0, 0] = min_x
            bboxes[i, 0, 1] = min_y
            bboxes[i, 0, 2] = min_z
            bboxes[i, 1, 0] = max_x
            bboxes[i, 1, 1] = max_y
            bboxes[i, 1, 2] = max_z

    # -- 3B: Batch sphere intersection (from selection.py:45) ----------
    #    Note: sphere check logic is inlined because AOT cannot resolve
    #    cross-references between exported functions.

    @compiler.export(
        "check_sphere_chunk",
        "void(float64[:,::1], int64[::1], float64[::1], float64,"
        " boolean[::1], int64, int64)",
    )
    def check_sphere_chunk(data, offsets, center, radius_sq, results, start_i, end_i):
        """Check sphere intersection for streamlines in [start_i, end_i)."""
        for i in range(start_i, end_i):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]
            n_pts = end_idx - start_idx

            if n_pts <= 0:
                continue

            found = False

            # Fast path: check vertices
            for p in range(start_idx, end_idx):
                dx = data[p, 0] - center[0]
                dy = data[p, 1] - center[1]
                dz = data[p, 2] - center[2]
                if dx * dx + dy * dy + dz * dz < radius_sq:
                    found = True
                    break

            # Precise path: check segments
            if not found:
                for p in range(start_idx, end_idx - 1):
                    p1x = data[p, 0]
                    p1y = data[p, 1]
                    p1z = data[p, 2]
                    p2x = data[p + 1, 0]
                    p2y = data[p + 1, 1]
                    p2z = data[p + 1, 2]

                    seg_x = p2x - p1x
                    seg_y = p2y - p1y
                    seg_z = p2z - p1z

                    pc_x = center[0] - p1x
                    pc_y = center[1] - p1y
                    pc_z = center[2] - p1z

                    seg_len_sq = seg_x * seg_x + seg_y * seg_y + seg_z * seg_z
                    if seg_len_sq == 0:
                        continue

                    t = (pc_x * seg_x + pc_y * seg_y + pc_z * seg_z) / seg_len_sq
                    if t < 0:
                        t = 0.0
                    elif t > 1:
                        t = 1.0

                    cx = p1x + t * seg_x - center[0]
                    cy = p1y + t * seg_y - center[1]
                    cz = p1z + t * seg_z - center[2]

                    if cx * cx + cy * cy + cz * cz < radius_sq:
                        found = True
                        break

            results[i] = found

    # -- 3C: Batch box intersection (from selection.py:84) -------------

    @compiler.export(
        "check_box_chunk",
        "void(float64[:,::1], int64[::1], float64[::1], float64[::1],"
        " boolean[::1], int64, int64)",
    )
    def check_box_chunk(data, offsets, box_min, box_max, results, start_i, end_i):
        """Check box intersection for streamlines in [start_i, end_i)."""
        for i in range(start_i, end_i):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]

            if end_idx <= start_idx:
                continue

            for j in range(start_idx, end_idx):
                x = data[j, 0]
                y = data[j, 1]
                z = data[j, 2]
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

    # -- 3D: Parallel streamline copy (from selection.py:134) ----------

    @compiler.export(
        "copy_streamlines_chunk",
        "void(float64[:,::1], float64[:,::1], int64[::1],"
        " int64[::1], int64[::1], int64, int64)",
    )
    def copy_streamlines_chunk(
        src_data, dst_data, src_starts, dst_starts, lengths, start_i, end_i
    ):
        """Copy streamline data for indices in [start_i, end_i)."""
        for i in range(start_i, end_i):
            src_start = src_starts[i]
            dst_start = dst_starts[i]
            length = lengths[i]
            for j in range(length):
                dst_data[dst_start + j, 0] = src_data[src_start + j, 0]
                dst_data[dst_start + j, 1] = src_data[src_start + j, 1]
                dst_data[dst_start + j, 2] = src_data[src_start + j, 2]

    # -- 3E: Batch resampling (from file_io.py:967) --------------------

    @compiler.export(
        "resample_batch_chunk",
        "void(float64[:,::1], int64[::1], int64[::1], int64,"
        " float64[:,:,::1], int64, int64)",
    )
    def resample_batch_chunk(
        flat_data, offsets, lengths, nb_points, result, start_i, end_i
    ):
        """Resample streamlines in [start_i, end_i) into *result*."""
        for i in range(start_i, end_i):
            start = offsets[i]
            length = lengths[i]

            if length <= 1:
                if length == 1:
                    for k in range(nb_points):
                        result[i, k, 0] = flat_data[start, 0]
                        result[i, k, 1] = flat_data[start, 1]
                        result[i, k, 2] = flat_data[start, 2]
                else:
                    for k in range(nb_points):
                        result[i, k, 0] = 0.0
                        result[i, k, 1] = 0.0
                        result[i, k, 2] = 0.0
                continue

            # Cumulative distances
            cum_dists = np.zeros(length, dtype=np.float64)
            for j in range(1, length):
                idx = start + j
                idx_prev = start + j - 1
                dx = flat_data[idx, 0] - flat_data[idx_prev, 0]
                dy = flat_data[idx, 1] - flat_data[idx_prev, 1]
                dz = flat_data[idx, 2] - flat_data[idx_prev, 2]
                cum_dists[j] = cum_dists[j - 1] + np.sqrt(dx * dx + dy * dy + dz * dz)

            total_length = cum_dists[length - 1]

            if total_length == 0.0:
                for k in range(nb_points):
                    result[i, k, 0] = flat_data[start, 0]
                    result[i, k, 1] = flat_data[start, 1]
                    result[i, k, 2] = flat_data[start, 2]
                continue

            for k in range(nb_points):
                target_dist = total_length * k / (nb_points - 1)

                # Binary search: cum_dists is monotonically non-decreasing,
                # find the first index where cum_dists >= target_dist.
                lo = 0
                hi = length - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cum_dists[mid] < target_dist:
                        lo = mid + 1
                    else:
                        hi = mid
                # seg_idx is the segment start: one before the found index
                seg_idx = lo - 1 if lo > 0 else 0

                seg_start_dist = cum_dists[seg_idx]
                seg_end_dist = (
                    cum_dists[seg_idx + 1] if seg_idx + 1 < length else seg_start_dist
                )
                seg_len = seg_end_dist - seg_start_dist

                if seg_len > 0:
                    t = (target_dist - seg_start_dist) / seg_len
                else:
                    t = 0.0

                p0 = start + seg_idx
                p1 = start + seg_idx + 1 if seg_idx + 1 < length else p0

                result[i, k, 0] = flat_data[p0, 0] + t * (
                    flat_data[p1, 0] - flat_data[p0, 0]
                )
                result[i, k, 1] = flat_data[p0, 1] + t * (
                    flat_data[p1, 1] - flat_data[p0, 1]
                )
                result[i, k, 2] = flat_data[p0, 2] + t * (
                    flat_data[p1, 2] - flat_data[p0, 2]
                )

    # -- 3F: MDF distance matrix rows (from file_io.py:1109) ----------

    @compiler.export(
        "compute_mdf_rows_chunk",
        "void(float64[:,:,::1], float64[:,::1], int64, int64)",
    )
    def compute_mdf_rows_chunk(resampled, dist_matrix, start_i, end_i):
        """Compute MDF distance matrix rows [start_i, end_i)."""
        n = resampled.shape[0]
        nb_points = resampled.shape[1]

        for i in range(start_i, end_i):
            for j in range(i + 1, n):
                d_direct = 0.0
                for k in range(nb_points):
                    dx = resampled[i, k, 0] - resampled[j, k, 0]
                    dy = resampled[i, k, 1] - resampled[j, k, 1]
                    dz = resampled[i, k, 2] - resampled[j, k, 2]
                    d_direct += np.sqrt(dx * dx + dy * dy + dz * dz)
                d_direct /= nb_points

                d_flipped = 0.0
                for k in range(nb_points):
                    flipped_k = nb_points - 1 - k
                    dx = resampled[i, k, 0] - resampled[j, flipped_k, 0]
                    dy = resampled[i, k, 1] - resampled[j, flipped_k, 1]
                    dz = resampled[i, k, 2] - resampled[j, flipped_k, 2]
                    d_flipped += np.sqrt(dx * dx + dy * dy + dz * dz)
                d_flipped /= nb_points

                dist = d_direct if d_direct < d_flipped else d_flipped
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    # -- 3G: Distances to samples (from file_io.py:1153) ---------------

    @compiler.export(
        "compute_distances_chunk",
        "void(float64[:,:,::1], int64[::1], float64[:,::1], int64, int64)",
    )
    def compute_distances_chunk(resampled, sample_indices, distances, start_i, end_i):
        """Compute MDF distances to sample streamlines for rows
        [start_i, end_i).
        """
        nb_points = resampled.shape[1]
        k = sample_indices.shape[0]

        for i in range(start_i, end_i):
            for j_idx in range(k):
                j = sample_indices[j_idx]

                if i == j:
                    continue

                d_direct = 0.0
                for pt in range(nb_points):
                    dx = resampled[i, pt, 0] - resampled[j, pt, 0]
                    dy = resampled[i, pt, 1] - resampled[j, pt, 1]
                    dz = resampled[i, pt, 2] - resampled[j, pt, 2]
                    d_direct += np.sqrt(dx * dx + dy * dy + dz * dz)
                d_direct /= nb_points

                d_flipped = 0.0
                for pt in range(nb_points):
                    flipped_pt = nb_points - 1 - pt
                    dx = resampled[i, pt, 0] - resampled[j, flipped_pt, 0]
                    dy = resampled[i, pt, 1] - resampled[j, flipped_pt, 1]
                    dz = resampled[i, pt, 2] - resampled[j, flipped_pt, 2]
                    d_flipped += np.sqrt(dx * dx + dy * dy + dz * dz)
                d_flipped /= nb_points

                if d_direct < d_flipped:
                    distances[i, j_idx] = d_direct
                else:
                    distances[i, j_idx] = d_flipped

    # -- 3H: Endpoint labeling (from connectivity.py:187) --------------

    @compiler.export(
        "compute_labels_chunk",
        "void(float64[:,::1], float64[:,::1], float64[:,::1],"
        " float64[::1], int32[:,:,::1], int64[::1],"
        " int32[::1], int32[::1], int64, int64)",
    )
    def compute_labels_chunk(
        start_points,
        end_points,
        inv_3x3,
        inv_offset,
        parcellation,
        dims,
        start_labels,
        end_labels,
        start_i,
        end_i,
    ):
        """Label endpoints for streamlines in [start_i, end_i)."""
        for i in range(start_i, end_i):
            # Transform start point to voxel coordinates
            sx = (
                inv_3x3[0, 0] * start_points[i, 0]
                + inv_3x3[0, 1] * start_points[i, 1]
                + inv_3x3[0, 2] * start_points[i, 2]
                + inv_offset[0]
            )
            sy = (
                inv_3x3[1, 0] * start_points[i, 0]
                + inv_3x3[1, 1] * start_points[i, 1]
                + inv_3x3[1, 2] * start_points[i, 2]
                + inv_offset[1]
            )
            sz = (
                inv_3x3[2, 0] * start_points[i, 0]
                + inv_3x3[2, 1] * start_points[i, 1]
                + inv_3x3[2, 2] * start_points[i, 2]
                + inv_offset[2]
            )

            vx_s = int(np.round(sx))
            vy_s = int(np.round(sy))
            vz_s = int(np.round(sz))

            if 0 <= vx_s < dims[0] and 0 <= vy_s < dims[1] and 0 <= vz_s < dims[2]:
                start_labels[i] = parcellation[vx_s, vy_s, vz_s]
            else:
                start_labels[i] = 0

            # Transform end point to voxel coordinates
            ex = (
                inv_3x3[0, 0] * end_points[i, 0]
                + inv_3x3[0, 1] * end_points[i, 1]
                + inv_3x3[0, 2] * end_points[i, 2]
                + inv_offset[0]
            )
            ey = (
                inv_3x3[1, 0] * end_points[i, 0]
                + inv_3x3[1, 1] * end_points[i, 1]
                + inv_3x3[1, 2] * end_points[i, 2]
                + inv_offset[1]
            )
            ez = (
                inv_3x3[2, 0] * end_points[i, 0]
                + inv_3x3[2, 1] * end_points[i, 1]
                + inv_3x3[2, 2] * end_points[i, 2]
                + inv_offset[2]
            )

            vx_e = int(np.round(ex))
            vy_e = int(np.round(ey))
            vz_e = int(np.round(ez))

            if 0 <= vx_e < dims[0] and 0 <= vy_e < dims[1] and 0 <= vz_e < dims[2]:
                end_labels[i] = parcellation[vx_e, vy_e, vz_e]
            else:
                end_labels[i] = 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Build the AOT extension module."""
    _ensure_msvc_env()

    compiler = _create_compiler()
    _register_functions(compiler)

    print(f"Compiling tractedit_numba -> {_SCRIPT_DIR}")
    compiler.compile()
    print("AOT compilation completed successfully.")


if __name__ == "__main__":
    main()
