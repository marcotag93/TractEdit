"""Source OBJ position indices shared by quantitative surface operations."""

from array import array

import numpy as np


def read_obj_faces_stream(stream, check, *, with_uv, update_digest=None):
    """Stream-parse OBJ geometry with AOT fast path and exact Python fallback."""
    from .._numba_aot import parse_obj_geom_block
    from ..mesh_io import MeshLoadError, _OBJ_LINE_BYTES

    cap = 32_768
    v_buf = np.empty((cap, 3), dtype=np.float64)
    vt_buf = np.empty((cap if with_uv else 1, 2), dtype=np.float64)
    fv_buf = np.empty((cap, 3), dtype=np.int64)
    fvt_buf = np.empty((cap if with_uv else 1, 3), dtype=np.int64)
    counts = np.zeros(3, dtype=np.int64)
    with_uv_flag = np.uint8(1 if with_uv else 0)

    v_parts: list[np.ndarray] = []
    vt_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    corner_parts: list[np.ndarray] = []
    offset_parts: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
    v_total = 0
    vt_total = 0
    idx_total = 0

    py_v = array("d")
    py_vt = array("d")
    py_idx = array("q")
    py_corners = array("q")
    py_offsets = array("q")

    def flush_py():
        if py_v:
            v_parts.append(np.array(py_v, dtype=np.float64, copy=True).reshape(-1, 3))
            del py_v[:]
        if py_vt:
            vt_parts.append(np.array(py_vt, dtype=np.float64, copy=True).reshape(-1, 2))
            del py_vt[:]
        if py_idx:
            idx_parts.append(np.array(py_idx, dtype=np.int64, copy=True))
            del py_idx[:]
        if py_corners:
            corner_parts.append(np.array(py_corners, dtype=np.int64, copy=True))
            del py_corners[:]
        if py_offsets:
            offset_parts.append(np.array(py_offsets, dtype=np.int64, copy=True))
            del py_offsets[:]

    def emit_face(face_fields):
        nonlocal idx_total
        for field in face_fields[1:]:
            parts = field.split(b"/")
            idx = int(parts[0])
            py_idx.append(idx - 1 if idx > 0 else v_total + idx)
            if with_uv:
                t_idx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                py_corners.append(t_idx - 1 if t_idx >= 0 else vt_total + t_idx)
        idx_total += len(face_fields) - 1
        py_offsets.append(idx_total)

    carry = b""
    in_f_continuation = False
    pending_f: list[bytes] = []
    py_lines = 0

    while True:
        check()
        incoming = stream.read(1024 * 1024)
        if update_digest is not None and incoming:
            update_digest(incoming)
        eof = not incoming
        block = carry + incoming
        if eof:
            boundary = len(block)
            carry = b""
        else:
            boundary = block.rfind(b"\n") + 1
            carry = block[boundary:]
            if len(carry) > _OBJ_LINE_BYTES:
                raise MeshLoadError("OBJ exceeds the native line size/count limit")
            if not boundary:
                continue
        if boundary:
            line_start = 0
            while boundary - line_start > _OBJ_LINE_BYTES:
                newline = block.rfind(b"\n", line_start, line_start + _OBJ_LINE_BYTES)
                if newline < 0:
                    raise MeshLoadError("OBJ exceeds the native line size/count limit")
                line_start = newline + 1
            view = np.frombuffer(block, dtype=np.uint8, count=boundary)
            pos = 0
            while pos < boundary:
                if not in_f_continuation:
                    next_pos = int(
                        parse_obj_geom_block(
                            view, pos, boundary, with_uv_flag,
                            v_buf, vt_buf, fv_buf, fvt_buf, counts,
                        )
                    )
                    if next_pos > pos:
                        flush_py()
                        vn = int(counts[0])
                        vtn = int(counts[1])
                        fn = int(counts[2])
                        if vn:
                            v_parts.append(v_buf[:vn].copy())
                            v_total += vn
                        if with_uv and vtn:
                            vt_parts.append(vt_buf[:vtn].copy())
                            vt_total += vtn
                        if fn:
                            idx_parts.append(fv_buf[:fn].ravel().copy())
                            if with_uv:
                                corner_parts.append(fvt_buf[:fn].ravel().copy())
                            offset_parts.append(
                                idx_total
                                + np.arange(1, fn + 1, dtype=np.int64) * 3
                            )
                            idx_total += fn * 3
                        pos = next_pos
                        if pos < boundary and (vn == cap or vtn == cap or fn == cap):
                            check()
                            continue
                if pos >= boundary:
                    break
                nl = block.find(b"\n", pos, boundary)
                line_end = boundary if nl < 0 else nl + 1
                raw = block[pos:line_end]
                pos = line_end
                py_lines += 1
                if py_lines % 8192 == 0:
                    check()
                fields = raw.partition(b"#")[0].split()
                if in_f_continuation:
                    pending_f[-1:] = fields
                    if pending_f and pending_f[-1] == b"\\":
                        continue
                    in_f_continuation = False
                    emit_face(pending_f)
                    pending_f = []
                    continue
                if not fields:
                    continue
                if fields[0] == b"v":
                    py_v.extend(map(float, fields[1:4]))
                    v_total += 1
                elif with_uv and fields[0] == b"vt":
                    py_vt.extend(map(float, fields[1:3]))
                    vt_total += 1
                elif fields[0] == b"f":
                    if fields[-1] == b"\\":
                        in_f_continuation = True
                        pending_f = fields
                    else:
                        emit_face(fields)
        if eof:
            break

    if in_f_continuation:
        pending_f[-1:] = []
        emit_face(pending_f)
    flush_py()
    return (
        np.concatenate(v_parts, axis=0) if v_parts else np.empty((0, 3), dtype=np.float64),
        np.concatenate(offset_parts),
        np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int64),
        np.concatenate(vt_parts, axis=0) if vt_parts else np.empty((0, 2), dtype=np.float64),
        np.concatenate(corner_parts) if corner_parts else np.empty(0, dtype=np.int64),
    )


def read_obj_faces(lines, check, *, with_uv):
    if hasattr(lines, "read"):
        return read_obj_faces_stream(lines, check, with_uv=with_uv)
    vertices, indices, uv, corners = (array(t) for t in ("d", "q", "d", "q"))
    offsets = array("q", [0])
    lines = iter(lines)
    for number, raw in enumerate(lines):
        if number % 8192 == 0:
            check()
        fields = raw.partition(b"#")[0].split()
        if not fields:
            continue
        if fields[0] == b"v":
            vertices.extend(map(float, fields[1:4]))
        elif with_uv and fields[0] == b"vt":
            uv.extend(map(float, fields[1:3]))
        elif fields[0] == b"f":
            while fields[-1] == b"\\":
                fields[-1:] = next(lines, b"").partition(b"#")[0].split()
            for field in fields[1:]:
                parts = field.split(b"/")
                idx = int(parts[0])
                indices.append(idx - 1 if idx > 0 else len(vertices) // 3 + idx)
                if with_uv:
                    idx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                    corners.append(idx - 1 if idx >= 0 else len(uv) // 2 + idx)
            offsets.append(len(indices))
    return (
        np.asarray(vertices).reshape(-1, 3),
        np.asarray(offsets),
        np.asarray(indices),
        np.asarray(uv).reshape(-1, 2),
        np.asarray(corners),
    )


def verify_obj_faces(mesh, vertices, offsets, indices, check):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    cells = mesh.GetPolys()
    display = vtk_to_numpy(mesh.GetPoints().GetData())
    if not np.array_equal(offsets, vtk_to_numpy(cells.GetOffsetsArray())):
        raise ValueError("Cannot verify displayed OBJ face order")
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    for start in range(0, len(indices), 98304):
        check()
        block = slice(start, start + 98304)
        ids = indices[block]
        if (
            np.any(ids < 0) or np.any(ids >= len(vertices))
            or not np.array_equal(
                vertices[ids].astype(display.dtype), display[connectivity[block]]
            )
        ):
            raise ValueError("Cannot verify displayed OBJ corner correspondence")
