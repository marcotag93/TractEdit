# -*- coding: utf-8 -*-

"""Wavefront OBJ byte-level scanning and parsing AOT kernels for TractEdit."""

from __future__ import annotations

import numpy as np
from numba import njit
from numba.pycc import CC


@njit
def _is_obj_ws(byte_val):
    return byte_val == 32 or byte_val == 9 or byte_val == 11 or byte_val == 12


@njit
def _match_obj_line_end(buf, pos, end):
    while pos < end and _is_obj_ws(buf[pos]):
        pos += 1
    if pos < end and buf[pos] == 13:
        pos += 1
    if pos < end and buf[pos] == 10:
        return pos + 1
    return -1


@njit
def _match_obj_decimal(buf, pos, end):
    if pos >= end or not _is_obj_ws(buf[pos]):
        return -1
    while pos < end and _is_obj_ws(buf[pos]):
        pos += 1
    if pos < end and (buf[pos] == 43 or buf[pos] == 45):
        pos += 1
    d_start = pos
    while pos < end and 48 <= buf[pos] <= 57:
        pos += 1
    n_int = pos - d_start
    if n_int < 1 or n_int > 9:
        return -1
    if pos < end and buf[pos] == 46:
        pos += 1
        f_start = pos
        while pos < end and 48 <= buf[pos] <= 57:
            pos += 1
        n_frac = pos - f_start
        if n_frac < 1 or n_frac > 12:
            return -1
    return pos


@njit
def _parse_obj_pos_int9(buf, pos, end):
    if pos >= end or buf[pos] < 49 or buf[pos] > 57:
        return -1, np.int64(0)
    val = np.int64(buf[pos] - 48)
    pos += 1
    digits = 1
    while pos < end and 48 <= buf[pos] <= 57:
        digits += 1
        if digits > 9:
            return -1, np.int64(0)
        val = val * 10 + (buf[pos] - 48)
        pos += 1
    return pos, val


@njit
def _parse_obj_f64_decimal(buf, pos, end):
    if pos >= end or not _is_obj_ws(buf[pos]):
        return -1, 0.0
    while pos < end and _is_obj_ws(buf[pos]):
        pos += 1
    if pos >= end:
        return -1, 0.0
    neg = False
    if buf[pos] == 45:
        neg = True
        pos += 1
    elif buf[pos] == 43:
        pos += 1
    mantissa = np.int64(0)
    sig_digits = 0
    has_digit = False
    while pos < end and 48 <= buf[pos] <= 57:
        has_digit = True
        digit = buf[pos] - 48
        if sig_digits > 0 or digit != 0:
            sig_digits += 1
            if sig_digits > 15:
                return -1, 0.0
            mantissa = mantissa * 10 + digit
        pos += 1
    frac_digits = 0
    if pos < end and buf[pos] == 46:
        pos += 1
        while pos < end and 48 <= buf[pos] <= 57:
            has_digit = True
            digit = buf[pos] - 48
            frac_digits += 1
            if frac_digits > 15:
                return -1, 0.0
            if sig_digits > 0 or digit != 0:
                sig_digits += 1
                if sig_digits > 15:
                    return -1, 0.0
                mantissa = mantissa * 10 + digit
            pos += 1
    if not has_digit:
        return -1, 0.0
    if pos < end and not (_is_obj_ws(buf[pos]) or buf[pos] == 13 or buf[pos] == 10):
        return -1, 0.0
    scale = 1.0
    for _ in range(frac_digits):
        scale *= 10.0
    val = np.float64(mantissa) / scale
    if neg:
        val = -val
    return pos, val


def register_obj_kernels(compiler: CC) -> None:
    """Register OBJ preflight, texture-block, and geometry-block AOT kernels."""

    @compiler.export(
        "scan_obj_preflight_block",
        "int64(uint8[::1], int64, int64, int64, int64[::1])",
    )
    def scan_obj_preflight_block(buf, start, end, max_line_bytes, out_state):
        """Scan common OBJ v/vt/vn and triangle f lines; stop at first fallback line."""
        for i in range(12):
            out_state[i] = 0
        pos = start
        v_count = np.int64(0)
        vt_count = np.int64(0)
        vn_count = np.int64(0)
        tex_tri = np.int64(0)
        simple_tri = np.int64(0)
        tex_no_n_tri = np.int64(0)
        norm_only_tri = np.int64(0)
        max_v = np.int64(0)
        max_vt = np.int64(0)
        max_vn = np.int64(0)
        diff_flag = np.int64(0)
        total_lines = np.int64(0)

        while pos < end:
            line_start = pos
            p = pos
            while p < end and _is_obj_ws(buf[p]):
                p += 1
            if p >= end:
                break
            b0 = buf[p]
            if b0 == 118:  # 'v'
                p += 1
                if p < end and buf[p] == 116:  # 'vt'
                    p = _match_obj_decimal(buf, p + 1, end)
                    if p < 0:
                        break
                    p = _match_obj_decimal(buf, p, end)
                    if p < 0:
                        break
                    next_pos = _match_obj_line_end(buf, p, end)
                    if next_pos < 0:
                        break
                    if (next_pos - 1) - line_start >= max_line_bytes:
                        return np.int64(-2)
                    vt_count += 1
                    total_lines += 1
                    pos = next_pos
                elif p < end and buf[p] == 110:  # 'vn'
                    p = _match_obj_decimal(buf, p + 1, end)
                    if p < 0:
                        break
                    p = _match_obj_decimal(buf, p, end)
                    if p < 0:
                        break
                    p = _match_obj_decimal(buf, p, end)
                    if p < 0:
                        break
                    next_pos = _match_obj_line_end(buf, p, end)
                    if next_pos < 0:
                        break
                    if (next_pos - 1) - line_start >= max_line_bytes:
                        return np.int64(-2)
                    vn_count += 1
                    total_lines += 1
                    pos = next_pos
                else:  # 'v' (3 or 6 decimal components)
                    p = _match_obj_decimal(buf, p, end)
                    if p < 0:
                        break
                    p = _match_obj_decimal(buf, p, end)
                    if p < 0:
                        break
                    p = _match_obj_decimal(buf, p, end)
                    if p < 0:
                        break
                    next_pos = _match_obj_line_end(buf, p, end)
                    if next_pos < 0:
                        p = _match_obj_decimal(buf, p, end)
                        if p < 0:
                            break
                        p = _match_obj_decimal(buf, p, end)
                        if p < 0:
                            break
                        p = _match_obj_decimal(buf, p, end)
                        if p < 0:
                            break
                        next_pos = _match_obj_line_end(buf, p, end)
                        if next_pos < 0:
                            break
                    if (next_pos - 1) - line_start >= max_line_bytes:
                        return np.int64(-2)
                    v_count += 1
                    total_lines += 1
                    pos = next_pos
            elif b0 == 102:  # 'f'
                p += 1
                if p >= end or not _is_obj_ws(buf[p]):
                    break
                while p < end and _is_obj_ws(buf[p]):
                    p += 1
                p, a = _parse_obj_pos_int9(buf, p, end)
                if p < 0:
                    break
                if p < end and buf[p] == 47:
                    if p + 1 < end and buf[p + 1] == 47:  # v//vn triangle
                        if tex_tri > 0 or simple_tri > 0 or tex_no_n_tri > 0:
                            break
                        p, na = _parse_obj_pos_int9(buf, p + 2, end)
                        if p < 0 or p >= end or not _is_obj_ws(buf[p]):
                            break
                        while p < end and _is_obj_ws(buf[p]):
                            p += 1
                        p, b = _parse_obj_pos_int9(buf, p, end)
                        if p < 0 or p + 1 >= end or buf[p] != 47 or buf[p + 1] != 47:
                            break
                        p, nb = _parse_obj_pos_int9(buf, p + 2, end)
                        if p < 0 or p >= end or not _is_obj_ws(buf[p]):
                            break
                        while p < end and _is_obj_ws(buf[p]):
                            p += 1
                        p, c = _parse_obj_pos_int9(buf, p, end)
                        if p < 0 or p + 1 >= end or buf[p] != 47 or buf[p + 1] != 47:
                            break
                        p, nc = _parse_obj_pos_int9(buf, p + 2, end)
                        if p < 0:
                            break
                        next_pos = _match_obj_line_end(buf, p, end)
                        if next_pos < 0:
                            break
                        if (next_pos - 1) - line_start >= max_line_bytes:
                            return np.int64(-2)
                        if a == b or b == c or a == c:
                            return np.int64(-1)
                        if a > max_v:
                            max_v = a
                        if b > max_v:
                            max_v = b
                        if c > max_v:
                            max_v = c
                        if na > max_vn:
                            max_vn = na
                        if nb > max_vn:
                            max_vn = nb
                        if nc > max_vn:
                            max_vn = nc
                        if a != na or b != nb or c != nc:
                            diff_flag = 1
                        norm_only_tri += 1
                        total_lines += 1
                        pos = next_pos
                    else:
                        p, ta = _parse_obj_pos_int9(buf, p + 1, end)
                        if p < 0 or p >= end:
                            break
                        if buf[p] == 47:  # v/vt/vn triangle
                            if simple_tri > 0 or tex_no_n_tri > 0 or norm_only_tri > 0:
                                break
                            p, na = _parse_obj_pos_int9(buf, p + 1, end)
                            if p < 0 or p >= end or not _is_obj_ws(buf[p]):
                                break
                            while p < end and _is_obj_ws(buf[p]):
                                p += 1
                            p, b = _parse_obj_pos_int9(buf, p, end)
                            if p < 0 or p >= end or buf[p] != 47:
                                break
                            p, tb = _parse_obj_pos_int9(buf, p + 1, end)
                            if p < 0 or p >= end or buf[p] != 47:
                                break
                            p, nb = _parse_obj_pos_int9(buf, p + 1, end)
                            if p < 0 or p >= end or not _is_obj_ws(buf[p]):
                                break
                            while p < end and _is_obj_ws(buf[p]):
                                p += 1
                            p, c = _parse_obj_pos_int9(buf, p, end)
                            if p < 0 or p >= end or buf[p] != 47:
                                break
                            p, tc = _parse_obj_pos_int9(buf, p + 1, end)
                            if p < 0 or p >= end or buf[p] != 47:
                                break
                            p, nc = _parse_obj_pos_int9(buf, p + 1, end)
                            if p < 0:
                                break
                            next_pos = _match_obj_line_end(buf, p, end)
                            if next_pos < 0:
                                break
                            if (next_pos - 1) - line_start >= max_line_bytes:
                                return np.int64(-2)
                            if a == b or b == c or a == c:
                                return np.int64(-1)
                            if a > max_v:
                                max_v = a
                            if b > max_v:
                                max_v = b
                            if c > max_v:
                                max_v = c
                            if ta > max_vt:
                                max_vt = ta
                            if tb > max_vt:
                                max_vt = tb
                            if tc > max_vt:
                                max_vt = tc
                            if na > max_vn:
                                max_vn = na
                            if nb > max_vn:
                                max_vn = nb
                            if nc > max_vn:
                                max_vn = nc
                            if (
                                a != ta
                                or a != na
                                or b != tb
                                or b != nb
                                or c != tc
                                or c != nc
                            ):
                                diff_flag = 1
                            tex_tri += 1
                            total_lines += 1
                            pos = next_pos
                        elif _is_obj_ws(buf[p]):  # v/vt triangle
                            if tex_tri > 0 or simple_tri > 0 or norm_only_tri > 0:
                                break
                            while p < end and _is_obj_ws(buf[p]):
                                p += 1
                            p, b = _parse_obj_pos_int9(buf, p, end)
                            if p < 0 or p >= end or buf[p] != 47:
                                break
                            p, tb = _parse_obj_pos_int9(buf, p + 1, end)
                            if p < 0 or p >= end or not _is_obj_ws(buf[p]):
                                break
                            while p < end and _is_obj_ws(buf[p]):
                                p += 1
                            p, c = _parse_obj_pos_int9(buf, p, end)
                            if p < 0 or p >= end or buf[p] != 47:
                                break
                            p, tc = _parse_obj_pos_int9(buf, p + 1, end)
                            if p < 0:
                                break
                            next_pos = _match_obj_line_end(buf, p, end)
                            if next_pos < 0:
                                break
                            if (next_pos - 1) - line_start >= max_line_bytes:
                                return np.int64(-2)
                            if a == b or b == c or a == c:
                                return np.int64(-1)
                            if a > max_v:
                                max_v = a
                            if b > max_v:
                                max_v = b
                            if c > max_v:
                                max_v = c
                            if ta > max_vt:
                                max_vt = ta
                            if tb > max_vt:
                                max_vt = tb
                            if tc > max_vt:
                                max_vt = tc
                            if a != ta or b != tb or c != tc:
                                diff_flag = 1
                            tex_no_n_tri += 1
                            total_lines += 1
                            pos = next_pos
                        else:
                            break
                else:  # simple v1 v2 v3 triangle
                    if tex_tri > 0 or tex_no_n_tri > 0 or norm_only_tri > 0:
                        break
                    if p >= end or not _is_obj_ws(buf[p]):
                        break
                    while p < end and _is_obj_ws(buf[p]):
                        p += 1
                    p, b = _parse_obj_pos_int9(buf, p, end)
                    if p < 0 or p >= end or not _is_obj_ws(buf[p]):
                        break
                    while p < end and _is_obj_ws(buf[p]):
                        p += 1
                    p, c = _parse_obj_pos_int9(buf, p, end)
                    if p < 0:
                        break
                    next_pos = _match_obj_line_end(buf, p, end)
                    if next_pos < 0:
                        break
                    if (next_pos - 1) - line_start >= max_line_bytes:
                        return np.int64(-2)
                    if a == b or b == c or a == c:
                        return np.int64(-1)
                    if a > max_v:
                        max_v = a
                    if b > max_v:
                        max_v = b
                    if c > max_v:
                        max_v = c
                    simple_tri += 1
                    total_lines += 1
                    pos = next_pos
            else:
                break

        out_state[0] = v_count
        out_state[1] = vt_count
        out_state[2] = vn_count
        out_state[3] = tex_tri
        out_state[4] = simple_tri
        out_state[5] = max_v
        out_state[6] = max_vt
        out_state[7] = max_vn
        out_state[8] = diff_flag
        out_state[9] = total_lines
        out_state[10] = tex_no_n_tri
        out_state[11] = norm_only_tri
        return pos

    @compiler.export(
        "scan_obj_texture_block",
        "int64(uint8[::1], int64, int64, int64, int64[::1])",
    )
    def scan_obj_texture_block(buf, start, end, max_line_bytes, out_state):
        """Skip non-asset/non-face OBJ lines and count uniform triangle UV runs."""
        out_state[0] = 0  # textured triangle count
        out_state[1] = 0  # untextured triangle count
        pos = start
        tex_tri = np.int64(0)
        untex_tri = np.int64(0)

        while pos < end:
            line_start = pos
            p = pos
            while p < end and _is_obj_ws(buf[p]):
                p += 1
            if p >= end:
                break
            b0 = buf[p]
            if b0 == 109:  # 'm' -> stop on mtllib
                if (
                    p + 6 <= end
                    and buf[p + 1] == 116
                    and buf[p + 2] == 108
                    and buf[p + 3] == 108
                    and buf[p + 4] == 105
                    and buf[p + 5] == 98
                    and (
                        p + 6 == end
                        or _is_obj_ws(buf[p + 6])
                        or buf[p + 6] == 13
                        or buf[p + 6] == 10
                    )
                ):
                    break
            elif b0 == 117:  # 'u' -> stop on usemtl
                if (
                    p + 6 <= end
                    and buf[p + 1] == 115
                    and buf[p + 2] == 101
                    and buf[p + 3] == 109
                    and buf[p + 4] == 116
                    and buf[p + 5] == 108
                    and (
                        p + 6 == end
                        or _is_obj_ws(buf[p + 6])
                        or buf[p + 6] == 13
                        or buf[p + 6] == 10
                    )
                ):
                    break
            elif b0 == 102 and p + 1 < end and _is_obj_ws(buf[p + 1]):  # 'f '
                q = p + 1
                corners = 0
                has_uv_all = 1
                has_uv_any = 0
                fast_ok = True
                while True:
                    while q < end and _is_obj_ws(buf[q]):
                        q += 1
                    if q >= end or buf[q] == 10 or buf[q] == 13:
                        break
                    if buf[q] == 35 or buf[q] == 92:  # '#' or '\'
                        fast_ok = False
                        break
                    corners += 1
                    if corners > 3:
                        fast_ok = False
                        break
                    t_start = q
                    slash1 = -1
                    slash2 = -1
                    while (
                        q < end
                        and not _is_obj_ws(buf[q])
                        and buf[q] != 10
                        and buf[q] != 13
                    ):
                        if buf[q] == 35 or buf[q] == 92:
                            fast_ok = False
                            break
                        if buf[q] == 47:
                            if slash1 < 0:
                                slash1 = q
                            elif slash2 < 0:
                                slash2 = q
                        q += 1
                    if not fast_ok:
                        break
                    if slash1 >= 0:
                        vt_end = slash2 if slash2 >= 0 else q
                        if vt_end > slash1 + 1:
                            has_uv_any = 1
                        else:
                            has_uv_all = 0
                    else:
                        has_uv_all = 0
                    if t_start == q:
                        fast_ok = False
                        break
                if not fast_ok or corners != 3:
                    break
                if q < end and buf[q] == 13:
                    q += 1
                if q >= end or buf[q] != 10:
                    break
                next_pos = q + 1
                if q - line_start >= max_line_bytes:
                    break
                if has_uv_all == 1:
                    if untex_tri > 0:
                        break
                    tex_tri += 1
                elif has_uv_any == 0:
                    if tex_tri > 0:
                        break
                    untex_tri += 1
                else:
                    break
                pos = next_pos
                continue

            # Skip any other record (v, vt, vn, #, o, g, s, blank) to '\n'.
            while p < end and buf[p] != 10:
                p += 1
            if p >= end:
                break
            if p - line_start >= max_line_bytes:
                break
            pos = p + 1

        out_state[0] = tex_tri
        out_state[1] = untex_tri
        return pos

    @compiler.export(
        "parse_obj_geom_block",
        "int64(uint8[::1], int64, int64, uint8,"
        " float64[:,::1], float64[:,::1], int64[:,::1], int64[:,::1], int64[::1])",
    )
    def parse_obj_geom_block(
        buf, start, end, with_uv, v_out, vt_out, fv_out, fvt_out, counts
    ):
        """Parse common OBJ v, vt and triangle f records into preallocated buffers."""
        v_cap = v_out.shape[0]
        vt_cap = vt_out.shape[0]
        f_cap = fv_out.shape[0]
        v_n = np.int64(0)
        vt_n = np.int64(0)
        f_n = np.int64(0)
        pos = start

        while pos < end:
            p = pos
            while p < end and _is_obj_ws(buf[p]):
                p += 1
            if p >= end:
                break
            b0 = buf[p]
            if b0 == 118:  # 'v'
                p1 = p + 1
                if p1 < end and buf[p1] == 116 and p1 + 1 < end and _is_obj_ws(buf[p1 + 1]):
                    if with_uv == 0:
                        while p < end and buf[p] != 10:
                            p += 1
                        if p >= end:
                            break
                        pos = p + 1
                        continue
                    if vt_n >= vt_cap:
                        break
                    q, u = _parse_obj_f64_decimal(buf, p1 + 1, end)
                    if q < 0:
                        break
                    q, v = _parse_obj_f64_decimal(buf, q, end)
                    if q < 0:
                        break
                    next_pos = _match_obj_line_end(buf, q, end)
                    if next_pos < 0:
                        break
                    vt_out[vt_n, 0] = u
                    vt_out[vt_n, 1] = v
                    vt_n += 1
                    pos = next_pos
                elif p1 < end and buf[p1] == 110 and p1 + 1 < end and _is_obj_ws(buf[p1 + 1]):
                    while p < end and buf[p] != 10:
                        p += 1
                    if p >= end:
                        break
                    pos = p + 1
                elif p1 < end and _is_obj_ws(buf[p1]):
                    if v_n >= v_cap:
                        break
                    q, x = _parse_obj_f64_decimal(buf, p1, end)
                    if q < 0:
                        break
                    q, y = _parse_obj_f64_decimal(buf, q, end)
                    if q < 0:
                        break
                    q, z = _parse_obj_f64_decimal(buf, q, end)
                    if q < 0:
                        break
                    next_pos = _match_obj_line_end(buf, q, end)
                    if next_pos < 0:
                        q2 = _match_obj_decimal(buf, q, end)
                        if q2 < 0:
                            break
                        q2 = _match_obj_decimal(buf, q2, end)
                        if q2 < 0:
                            break
                        q2 = _match_obj_decimal(buf, q2, end)
                        if q2 < 0:
                            break
                        next_pos = _match_obj_line_end(buf, q2, end)
                        if next_pos < 0:
                            break
                    v_out[v_n, 0] = x
                    v_out[v_n, 1] = y
                    v_out[v_n, 2] = z
                    v_n += 1
                    pos = next_pos
                else:
                    break
            elif b0 == 102 and p + 1 < end and _is_obj_ws(buf[p + 1]):  # 'f '
                if f_n >= f_cap:
                    break
                q = p + 1
                ok = True
                for corner in range(3):
                    if q >= end or not _is_obj_ws(buf[q]):
                        ok = False
                        break
                    while q < end and _is_obj_ws(buf[q]):
                        q += 1
                    q, vi = _parse_obj_pos_int9(buf, q, end)
                    if q < 0:
                        ok = False
                        break
                    fv_out[f_n, corner] = vi - 1
                    ti = np.int64(-1)
                    if q < end and buf[q] == 47:
                        q += 1
                        if q < end and buf[q] != 47 and not _is_obj_ws(buf[q]) and buf[q] != 10 and buf[q] != 13:
                            q, ti_val = _parse_obj_pos_int9(buf, q, end)
                            if q < 0:
                                ok = False
                                break
                            ti = ti_val - 1
                        if q < end and buf[q] == 47:
                            q += 1
                            if q < end and not _is_obj_ws(buf[q]) and buf[q] != 10 and buf[q] != 13:
                                q, _ = _parse_obj_pos_int9(buf, q, end)
                                if q < 0:
                                    ok = False
                                    break
                    if with_uv != 0:
                        fvt_out[f_n, corner] = ti
                if not ok:
                    break
                next_pos = _match_obj_line_end(buf, q, end)
                if next_pos < 0:
                    break
                f_n += 1
                pos = next_pos
            elif b0 == 35 or b0 == 13 or b0 == 10:  # '#' or empty
                while p < end and buf[p] != 10:
                    p += 1
                if p >= end:
                    break
                pos = p + 1
            elif b0 in (111, 103, 115, 109, 117):  # o, g, s, mtllib, usemtl
                while p < end and buf[p] != 10:
                    p += 1
                if p >= end:
                    break
                pos = p + 1
            else:
                break

        counts[0] = v_n
        counts[1] = vt_n
        counts[2] = f_n
        return pos
