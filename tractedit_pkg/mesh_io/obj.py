import re
import numpy as np
from collections.abc import Callable
from pathlib import Path
from ..data_contracts import ObjTextureIndex
from .validation import MeshLoadError, _check_cancel
from .. import mesh_io


_OBJ_LINE_BYTES = 256 * 1024 - 1
_OBJ_BLOCK_BYTES = 1024 * 1024
_OBJ_INT_MAX = 2_147_483_647
_OBJ_MAX_MATERIALS = 4096
_OBJ_MATERIAL_BYTES = 256 * 1024 * 1024
_OBJ_INDEX_UV_BYTES = 16 * 1024 * 1024
_OBJ_INDEX_REFERENCE_BYTES = 256 * 1024
_OBJ_FLOAT_MAX = float(np.finfo(np.float32).max)
# Half the smallest float32 subnormal: nonzero values at/below this tie round
# to zero and can set failbit in MSVC's extraction, leaving later xyz stale.
_OBJ_FLOAT_ZERO_LIMIT = 2.0 ** -150
_OBJ_NUMBER = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_OBJ_INTEGER = re.compile(rb"[+-]?[0-9]+")
_OBJ_CONTROL_BYTES = re.compile(rb"[\x00-\x08\x0e-\x1f]")
# These fixed decimals lie safely within float32, including nonzero magnitudes.
# Other spellings retain the complete numeric checks below.
_OBJ_DECIMAL_COMPONENT = rb"[ \t\v\f]+[+-]?[0-9]{1,9}(?:\.[0-9]{1,12})?"
_OBJ_DECIMAL_RECORD = re.compile(
    rb"[ \t\v\f]*(?:(v)" + _OBJ_DECIMAL_COMPONENT * 3
    + rb"(?:" + _OBJ_DECIMAL_COMPONENT * 3 + rb")?"
    + rb"|(vt)" + _OBJ_DECIMAL_COMPONENT * 2
    + rb"|(vn)" + _OBJ_DECIMAL_COMPONENT * 3 + rb")[ \t\v\f]*\r?\n?"
)
# Positive indices up to nine digits cannot overflow the native signed C int.
_OBJ_TEXTURED_TRIANGLE = re.compile(
    rb"[ \t\v\f]*f" + (
        rb"[ \t\v\f]+([1-9][0-9]{0,8})/([1-9][0-9]{0,8})/([1-9][0-9]{0,8})"
    ) * 3 + rb"[ \t\v\f]*\r?\n?"
)
# Complete LF-terminated records need no backtracking between repetitions.
_OBJ_BLOCK_RECORDS = tuple(
    re.compile(
        rb"(?:[ \t\v\f]*" + command + _OBJ_DECIMAL_COMPONENT * width
        + (rb"(?:" + _OBJ_DECIMAL_COMPONENT * 3 + rb")?" if command == b"v" else b"")
        + rb"[ \t\v\f]*\r?\n)++"
    )
    for command, width in ((b"v", 3), (b"vt", 2), (b"vn", 3))
) + (
    re.compile(
        rb"(?:" + _OBJ_TEXTURED_TRIANGLE.pattern.removesuffix(rb"\r?\n?")
        + rb"\r?\n)++"
    ),
    re.compile(rb"(?:[ \t\v\f]*f(?:[ \t\v\f]+[1-9][0-9]{0,8}){3}[ \t\v\f]*\r?\n)++"),
)


def _obj_index(token: bytes) -> int:
    if not _OBJ_INTEGER.fullmatch(token):
        raise MeshLoadError("Invalid OBJ face index syntax")
    digits = token.lstrip(b"+-").lstrip(b"0")
    if len(digits) > 10:
        raise MeshLoadError("OBJ face index exceeds the native C int range")
    value = int(digits or b"0") * (-1 if token.startswith(b"-") else 1)
    if value == 0 or not -_OBJ_INT_MAX <= value <= _OBJ_INT_MAX:
        raise MeshLoadError("OBJ face index is zero or outside the native C int range")
    return value


def _scan_obj_file(
    path: Path, cancel_check: Callable[[], bool] | None,
    update_digest: Callable[[bytes], None] | None = None,
    *, collect_texture_index: bool = False,
) -> tuple[int, ObjTextureIndex | None]:
    """Validate the surface subset of vtkOBJReader 9.2.6 without storing geometry.

    Uses the AOT preflight scanner on large blocks with an exact physical-line
    fallback. Positive indices may refer forward; negatives use current table
    sizes. When requested, captures lightweight OBJ texture metadata without
    reading any external MTL or image files.
    """
    from .._numba_aot import scan_obj_preflight_block

    counts = [0, 0, 0]  # v, vt, vn
    positive_maxima = [0, 0, 0]
    face_count = 0
    face_width = 0
    face_style = None
    face_normals = None
    face_first = face_previous = None
    continuing = False
    duplicates = False
    misaligned_attributes = False
    face_textures = False
    materials = set()
    face_references = 0
    line_number = 0

    current_material = ""
    reference_lines: list[bytes] = []
    reference_bytes = 0
    tex_layout: dict[str, dict[str, int | bool]] = {}
    tex_total_faces = 0
    tex_coverage: bytearray | None = None
    index_usable = collect_texture_index

    def discard_index() -> None:
        nonlocal index_usable, tex_coverage
        index_usable = False
        tex_coverage = None
        reference_lines.clear()
        tex_layout.clear()

    def record_tex_faces(n_faces: int, n_corners: int, has_uv: bool) -> None:
        nonlocal tex_total_faces, tex_coverage, index_usable
        if not index_usable:
            return
        summary = tex_layout.setdefault(
            current_material,
            {"face_count": 0, "corner_count": 0, "complete_uv": True},
        )
        summary["face_count"] += n_faces
        summary["corner_count"] += n_corners
        summary["complete_uv"] &= has_uv
        if tex_coverage is None:
            if not has_uv:
                if tex_total_faces + n_faces > _OBJ_INDEX_UV_BYTES:
                    discard_index()
                    return
                tex_coverage = bytearray(b"\x01") * tex_total_faces
                tex_coverage.extend(b"\x00" * n_faces)
        else:
            if len(tex_coverage) + n_faces > _OBJ_INDEX_UV_BYTES:
                discard_index()
                return
            tex_coverage.extend((b"\x01" if has_uv else b"\x00") * n_faces)
        tex_total_faces += n_faces

    def process_line(line: bytes) -> None:
        nonlocal face_count, face_width, face_style, face_normals
        nonlocal face_first, face_previous, continuing, duplicates
        nonlocal misaligned_attributes, face_textures, face_references
        nonlocal line_number, current_material, reference_bytes
        _check_cancel(cancel_check)
        line_number += 1
        if len(line) > mesh_io._OBJ_LINE_BYTES or line_number > _OBJ_INT_MAX:
            raise MeshLoadError("OBJ exceeds the native line size/count limit")
        # NUL defeats strlen; DOS EOF can truncate the native Windows text
        # stream. Restrict the lexical subset to ASCII/C whitespace.
        if (
            not line.isascii()
            or _OBJ_CONTROL_BYTES.search(line)
            or b"\r" in line.rstrip(b"\r\n")
        ):
            raise MeshLoadError("OBJ contains unsupported text/control bytes")
        if not continuing:
            numeric = _OBJ_DECIMAL_RECORD.fullmatch(line)
            if numeric is not None:
                counts[numeric.lastindex - 1] += 1
                return
            triangle = _OBJ_TEXTURED_TRIANGLE.fullmatch(line)
            if triangle is not None:
                a, ta, na, b, tb, nb, c, tc, nc = map(int, triangle.groups())
                positive_maxima[0] = max(positive_maxima[0], a, b, c)
                positive_maxima[1] = max(positive_maxima[1], ta, tb, tc)
                positive_maxima[2] = max(positive_maxima[2], na, nb, nc)
                different = (
                    a != ta or a != na or b != tb or b != nb or c != tc or c != nc
                )
                duplicates |= different
                misaligned_attributes |= different
                if a == b or b == c or a == c:
                    raise MeshLoadError("Degenerate OBJ face: repeated vertex")
                if face_normals is False:
                    raise MeshLoadError("OBJ cannot mix faces with and without normals")
                face_normals = face_textures = True
                face_count += 1
                face_references += 3
                if collect_texture_index:
                    record_tex_faces(1, 3, True)
                return
        fields = line.split()
        if continuing:
            command = b"f"
        else:
            if not fields or fields[0].startswith(b"#"):
                return
            command, fields = fields[0], fields[1:]
        if command in (b"v", b"vt", b"vn"):
            # Numeric extraction accepts comments only after all components.
            for index, token in enumerate(fields):
                if token.startswith(b"#"):
                    fields = fields[:index]
                    break
            width = 2 if command == b"vt" else 3
            # VTK reads XYZ only; accept the XYZ RGB vertex extension while
            # retaining the mesh layer's uniform colour and numeric checks.
            if len(fields) != width and not (command == b"v" and len(fields) == 6):
                components = "3 (XYZ) or 6 (XYZ RGB)" if command == b"v" else str(width)
                raise MeshLoadError(
                    f"OBJ {command.decode()} requires {components} components"
                )
            for token in fields:
                if not _OBJ_NUMBER.fullmatch(token):
                    raise MeshLoadError("Invalid OBJ numeric component syntax")
                magnitude = abs(float(token))
                if not magnitude <= _OBJ_FLOAT_MAX:
                    raise MeshLoadError("OBJ numeric component exceeds float32")
                if magnitude <= _OBJ_FLOAT_ZERO_LIMIT and (
                    token.lower().split(b"e", 1)[0].strip(b"+-.0")
                ):
                    raise MeshLoadError("OBJ nonzero numeric component underflows float32")
            counts[(b"v", b"vt", b"vn").index(command)] += 1
        elif command == b"f":
            if not continuing:
                face_width = 0
                face_style = None
                face_first = face_previous = None
            continuing = bool(fields and fields[-1] == b"\\")
            if continuing:
                # VTK compares the literal suffix to backslash + LF. CRLF
                # continuations behave differently on Windows and Unix.
                if not line.endswith(b"\\\n"):
                    raise MeshLoadError(
                        "OBJ face continuation requires an exact backslash-LF suffix"
                    )
                fields = fields[:-1]
            for token_number, token in enumerate(fields):
                if token_number % 1024 == 0:
                    _check_cancel(cancel_check)
                parts = token.split(b"/")
                if not (
                    len(parts) == 1
                    or len(parts) == 2 and all(parts)
                    or len(parts) == 3 and parts[0] and parts[2]
                ):
                    raise MeshLoadError("Unsupported OBJ face reference form")
                style = (len(parts) > 1 and bool(parts[1]), len(parts) == 3)
                if face_style is not None and style != face_style:
                    raise MeshLoadError("OBJ face references must use a consistent form")
                face_style = style
                values = []
                resolved = []
                for table, part in enumerate(parts):
                    if not part:
                        values.append(None)
                        resolved.append(None)
                        continue
                    value = _obj_index(part)
                    if value < 0 and -value > counts[table]:
                        raise MeshLoadError(
                            "OBJ relative face index is outside its current table"
                        )
                    positive_maxima[table] = max(positive_maxima[table], value)
                    values.append(value)
                    resolved.append(
                        counts[table] + value if value < 0 else value - 1
                    )
                for table in range(1, len(parts)):
                    if values[table] is not None:
                        duplicates |= values[table] != values[0]
                        misaligned_attributes |= resolved[table] != resolved[0]
                if face_previous == resolved[0]:
                    raise MeshLoadError("Degenerate OBJ face: repeated adjacent vertex")
                if face_first is None:
                    face_first = resolved[0]
                face_previous = resolved[0]
                face_width += 1
                if face_width > _OBJ_INT_MAX:
                    raise MeshLoadError("OBJ face exceeds the native vertex count range")
            if not continuing:
                if face_width < 3 or face_first == face_previous:
                    raise MeshLoadError(
                        "Degenerate OBJ face: fewer than three vertices or repeated closure"
                    )
                if face_normals is not None and face_normals != face_style[1]:
                    raise MeshLoadError("OBJ cannot mix faces with and without normals")
                face_normals = face_style[1]
                face_textures |= face_style[0]
                face_count += 1
                face_references += face_width
                if collect_texture_index:
                    record_tex_faces(1, face_width, face_style[0])
        elif command == b"usemtl":
            if not fields or len(fields[0]) > 99 or (
                len(fields) > 1 and not fields[1].startswith(b"#")
            ):
                raise MeshLoadError("OBJ usemtl requires one name of at most 99 bytes")
            if b'"' in fields[0]:
                raise MeshLoadError("Unsupported character in OBJ material name")
            known_material = fields[0] in materials
            materials.add(fields[0])
            if len(materials) > mesh_io._OBJ_MAX_MATERIALS:
                raise MeshLoadError("OBJ material count exceeds the resource limit")
            if collect_texture_index:
                current_material = fields[0].decode("ascii")
                if not fields[0].startswith(b"#"):
                    if known_material:
                        return
                    line = b"usemtl " + fields[0]
        elif command not in (b"mtllib", b"o", b"g", b"s"):
            raise MeshLoadError("Unsupported OBJ record; export only polygon surfaces")
        if index_usable and command in (b"usemtl", b"mtllib"):
            reference_bytes += len(line) + 64
            if reference_bytes > mesh_io._OBJ_INDEX_REFERENCE_BYTES:
                discard_index()
            else:
                reference_lines.append(line.removesuffix(b"\n"))

    patterns = mesh_io._OBJ_BLOCK_RECORDS
    if patterns is not _OBJ_BLOCK_RECORDS:
        native_scan = scan_obj_preflight_block

        def scan_obj_preflight_block(view, start, end, max_line_bytes, state):
            # Profile substituted patterns on complete records; native validation
            # still checks every byte in the matched run.
            for pattern in patterns:
                match = pattern.match(view, start, end)
                if match is not None and match.end() > start:
                    end = match.end()
                    break
            return native_scan(view, start, end, max_line_bytes, state)

    out_state = np.zeros(12, dtype=np.int64)
    carry = b""
    with path.open("rb") as stream:
        while True:
            _check_cancel(cancel_check)
            incoming = stream.read(mesh_io._OBJ_BLOCK_BYTES)
            if update_digest is not None:
                update_digest(incoming)
            if not incoming:
                if carry:
                    process_line(carry)
                break
            block = carry + incoming
            boundary = block.rfind(b"\n") + 1
            carry = block[boundary:]
            if len(carry) > mesh_io._OBJ_LINE_BYTES:
                raise MeshLoadError("OBJ exceeds the native line size/count limit")
            if not boundary:
                continue
            if boundary < mesh_io._OBJ_BLOCK_BYTES // 2:
                lines = block[:boundary].split(b"\n")[:-1]
                if any(len(raw_line) >= mesh_io._OBJ_LINE_BYTES for raw_line in lines):
                    raise MeshLoadError("OBJ exceeds the native line size/count limit")
                for raw_line in lines:
                    process_line(raw_line + b"\n")
                continue
            view = np.frombuffer(block, dtype=np.uint8, count=boundary)
            pos = 0
            while pos < boundary:
                _check_cancel(cancel_check)
                if not continuing:
                    next_pos = int(
                        scan_obj_preflight_block(
                            view, pos, boundary, mesh_io._OBJ_LINE_BYTES, out_state
                        )
                    )
                    if next_pos == -1:
                        raise MeshLoadError("Degenerate OBJ face: repeated vertex")
                    if next_pos == -2:
                        raise MeshLoadError(
                            "OBJ exceeds the native line size/count limit"
                        )
                    if next_pos > pos:
                        line_number += int(out_state[9])
                        if line_number > _OBJ_INT_MAX:
                            raise MeshLoadError(
                                "OBJ exceeds the native line size/count limit"
                            )
                        counts[0] += int(out_state[0])
                        counts[1] += int(out_state[1])
                        counts[2] += int(out_state[2])
                        if out_state[5] > positive_maxima[0]:
                            positive_maxima[0] = int(out_state[5])
                        if out_state[6] > positive_maxima[1]:
                            positive_maxima[1] = int(out_state[6])
                        if out_state[7] > positive_maxima[2]:
                            positive_maxima[2] = int(out_state[7])
                        if out_state[8]:
                            duplicates = True
                            misaligned_attributes = True
                        tex_tri = int(out_state[3])
                        simple_tri = int(out_state[4])
                        tex_no_n_tri = int(out_state[10])
                        norm_only_tri = int(out_state[11])
                        run_faces = tex_tri + simple_tri + tex_no_n_tri + norm_only_tri
                        if run_faces:
                            has_normals = bool(tex_tri or norm_only_tri)
                            has_uv = bool(tex_tri or tex_no_n_tri)
                            if face_normals is not None and face_normals != has_normals:
                                raise MeshLoadError(
                                    "OBJ cannot mix faces with and without normals"
                                )
                            face_normals = has_normals
                            face_textures |= has_uv
                            face_count += run_faces
                            face_references += 3 * run_faces
                            if collect_texture_index:
                                record_tex_faces(run_faces, 3 * run_faces, has_uv)
                        pos = next_pos
                        continue
                nl = block.index(b"\n", pos)
                if nl - pos >= mesh_io._OBJ_LINE_BYTES:
                    raise MeshLoadError("OBJ exceeds the native line size/count limit")
                process_line(block[pos : nl + 1])
                pos = nl + 1

    if continuing or not counts[0] or not face_count:
        raise MeshLoadError("OBJ requires complete polygon faces and vertices")
    if any(maximum > count for maximum, count in zip(positive_maxima, counts)):
        raise MeshLoadError(
            "OBJ positive face index is outside its vertex/texture/normal table"
        )
    if duplicates:
        if counts[2] and not face_normals:
            raise MeshLoadError("OBJ unused normals would cause VTK to discard faces")
    elif misaligned_attributes or (
        counts[2] and counts[2] != counts[0]
        or face_textures and counts[1] != counts[0]
    ):
        raise MeshLoadError(
            "OBJ attributes do not match vertices on the native direct-copy path"
        )
    # Each material adds UV tables, including temporary duplicated tuples.
    texture_tuples = counts[1] + (2 * face_references if duplicates else 0)
    material_bytes = max(0, len(materials) - 1) * texture_tuples * 8
    if material_bytes > mesh_io._OBJ_MATERIAL_BYTES:
        raise MeshLoadError(
            "OBJ materials exceed the additional texture memory budget (256 MiB); "
            "reduce unused materials or export the surface as VTP"
        )
    texture_index: ObjTextureIndex | None = None
    if collect_texture_index and index_usable:
        texture_index = {
            "reference_lines": tuple(reference_lines),
            "face_layout": tex_layout,
            "face_uv_bytes": bytes(tex_coverage) if tex_coverage is not None else None,
        }
    return face_count, texture_index


def _check_obj_file(
    path: Path, cancel_check: Callable[[], bool] | None,
    update_digest: Callable[[bytes], None] | None = None,
) -> int:
    """Validate the surface subset of vtkOBJReader 9.2.6 without storing geometry."""
    face_count, _ = mesh_io._scan_obj_file(
        path, cancel_check, update_digest, collect_texture_index=False
    )
    return face_count


_DEFAULT_CHECK_OBJ_FILE = _check_obj_file
