import os
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from .validation import MeshLoadError, _check_cancel
from .. import mesh_io


_PLY_HEADER_BYTES = 1_048_576
_PLY_HEADER_LINES = 8_192
_PLY_LINE_BYTES = 4_096
_PLY_RECORD_BYTES = 4_096
_PLY_BLOCK_BYTES = 4 * 1024 * 1024


_PLY_TYPES = {
    name: np.dtype(dtype)
    for names, dtype in (
        (("char", "int8"), "i1"),
        (("uchar", "uint8"), "u1"),
        (("short", "int16"), "i2"),
        (("ushort", "uint16"), "u2"),
        (("int", "int32"), "i4"),
        (("uint", "uint32"), "u4"),
        (("float", "float32"), "f4"),
        (("double", "float64"), "f8"),
    )
    for name in names
}


@dataclass
class _PlyProperty:
    name: bytes
    dtype: np.dtype
    count_dtype: np.dtype | None = None


@dataclass
class _PlyElement:
    name: bytes
    count: int
    properties: list[_PlyProperty] = field(default_factory=list)


def _check_ply_header(
    stream: BinaryIO, cancel_check: Callable[[], bool] | None,
    update_digest: Callable[[bytes], None] | None = None,
) -> tuple[bytes, list[_PlyElement]]:
    """Bound the native PLY header scan and require its geometry declarations."""
    byte_count = 0
    element = None
    elements = []
    encoding = b""
    header_lines = mesh_io._PLY_HEADER_LINES
    line_bytes = mesh_io._PLY_LINE_BYTES
    header_bytes = mesh_io._PLY_HEADER_BYTES
    for line_number in range(header_lines):
        _check_cancel(cancel_check)
        line = stream.readline(line_bytes + 1)
        if update_digest is not None:
            update_digest(line)
        byte_count += len(line)
        if len(line) > line_bytes or byte_count > header_bytes:
            raise MeshLoadError("PLY header exceeds the byte/line size limit")
        if not line.endswith(b"\n"):
            raise MeshLoadError("Truncated PLY header: missing end_header")
        fields = line.split()
        if line_number == 0:
            if fields != [b"ply"]:
                raise MeshLoadError("Invalid PLY signature")
        elif line_number == 1:
            if fields not in (
                [b"format", b"ascii", b"1.0"],
                [b"format", b"binary_little_endian", b"1.0"],
                [b"format", b"binary_big_endian", b"1.0"],
            ):
                raise MeshLoadError("Unsupported PLY format declaration")
            encoding = fields[1]
        elif fields and fields[0] == b"element":
            if len(fields) != 3:
                raise MeshLoadError("Invalid PLY element declaration")
            try:
                count = int(fields[2])
            except ValueError as error:
                raise MeshLoadError("Invalid PLY element count") from error
            if not 0 <= count <= np.iinfo(np.int32).max:
                raise MeshLoadError("Invalid PLY element count")
            if any(previous.name == fields[1] for previous in elements):
                raise MeshLoadError("Duplicate PLY element")
            if fields[1] not in (b"vertex", b"face") and count:
                raise MeshLoadError("VTK 9.2.6 supports only PLY vertex/face elements")
            element = _PlyElement(fields[1], count)
            elements.append(element)
        elif fields and fields[0] == b"property":
            if element is None:
                raise MeshLoadError("PLY property has no element")
            try:
                if len(fields) == 3:
                    prop = _PlyProperty(
                        fields[2], _PLY_TYPES[fields[1].decode("ascii")]
                    )
                elif len(fields) == 5 and fields[1] == b"list":
                    prop = _PlyProperty(
                        fields[4],
                        _PLY_TYPES[fields[3].decode("ascii")],
                        _PLY_TYPES[fields[2].decode("ascii")],
                    )
                    if prop.count_dtype.kind not in "iu":
                        raise MeshLoadError("PLY list count must have an integer type")
                else:
                    raise MeshLoadError("Invalid PLY property declaration")
            except (KeyError, UnicodeDecodeError) as error:
                raise MeshLoadError("Unsupported PLY property type") from error
            list_property = element.name == b"face" and prop.name in (
                b"vertex_indices",
                b"texcoord",
            )
            scalar_property = prop.name in {
                b"x",
                b"y",
                b"z",
                b"nx",
                b"ny",
                b"nz",
                b"u",
                b"v",
                b"texture_u",
                b"texture_v",
                b"red",
                b"green",
                b"blue",
                b"alpha",
                b"diffuse_red",
                b"diffuse_green",
                b"diffuse_blue",
                b"intensity",
            }
            if (list_property and prop.count_dtype is None) or (
                scalar_property and prop.count_dtype is not None
            ):
                raise MeshLoadError("Invalid PLY scalar/list property declaration")
            if any(previous.name == prop.name for previous in element.properties):
                raise MeshLoadError("Duplicate PLY property")
            element.properties.append(prop)
        elif fields == [b"end_header"]:
            vertices = next((item for item in elements if item.name == b"vertex"), None)
            faces = next((item for item in elements if item.name == b"face"), None)
            if (
                vertices is None
                or not vertices.count
                or not {b"x", b"y", b"z"}
                <= {
                    prop.name
                    for prop in vertices.properties
                    if prop.count_dtype is None
                }
            ):
                raise MeshLoadError("PLY surface requires points with x, y, z")
            if (
                faces is None
                or not faces.count
                or not any(
                    prop.name == b"vertex_indices"
                    and prop.count_dtype is not None
                    and prop.dtype.kind in "iu"
                    for prop in faces.properties
                )
            ):
                raise MeshLoadError(
                    "PLY surface requires polygon faces with integer indices"
                )
            if elements.index(vertices) > elements.index(faces):
                raise MeshLoadError("PLY faces must follow the vertex element")
            return encoding, elements
    raise MeshLoadError("PLY header exceeds the line count limit")


def _check_ply_list_count(prop: _PlyProperty, count: int) -> None:
    if count < 0:
        raise MeshLoadError("Invalid PLY list count")
    if prop.name in (b"vertex_indices", b"texcoord") and count > 255:
        raise MeshLoadError("VTK 9.2.6 PLY face lists are limited to 255 values")
    if prop.name == b"vertex_indices" and count < 3:
        raise MeshLoadError("Degenerate PLY polygon: fewer than three vertices")


def _hash_stream_bytes(
    stream: BinaryIO, size: int, cancel_check: Callable[[], bool] | None,
    update_digest: Callable[[bytes], None],
) -> None:
    remaining = size
    while remaining:
        _check_cancel(cancel_check)
        block = stream.read(min(remaining, mesh_io._PLY_BLOCK_BYTES))
        if not block:
            raise MeshLoadError("Truncated PLY binary records")
        update_digest(block)
        remaining -= len(block)


def _check_ply_binary_faces(
    stream: BinaryIO, element: _PlyElement, point_count: int, endian: str,
    cancel_check: Callable[[], bool] | None,
    update_digest: Callable[[bytes], None] | None = None,
) -> int:
    """Check homogeneous prefixes, leaving varying lists to the general scan."""
    remaining = element.count
    while remaining:
        _check_cancel(cancel_check)
        start = stream.tell()
        fields = []
        list_checks = []
        vi_field = None
        record_bytes = 0
        for index, prop in enumerate(element.properties):
            if prop.count_dtype is not None:
                count_dtype = prop.count_dtype.newbyteorder(endian)
                payload = stream.read(count_dtype.itemsize)
                if len(payload) != count_dtype.itemsize:
                    raise MeshLoadError("Truncated PLY binary list count")
                count = int(np.frombuffer(payload, dtype=count_dtype)[0])
                _check_ply_list_count(prop, count)
                if count == 0:
                    stream.seek(start)
                    return element.count - remaining
                record_bytes += count_dtype.itemsize + count * prop.dtype.itemsize
                if record_bytes > mesh_io._PLY_BLOCK_BYTES:
                    stream.seek(start)
                    return element.count - remaining
                fields.append((f"c{index}", count_dtype))
                fields.append((f"d{index}", prop.dtype.newbyteorder(endian), (count,)))
                list_checks.append((f"c{index}", count))
                if prop.name == b"vertex_indices":
                    vi_field = f"d{index}"
                stream.seek(count * prop.dtype.itemsize, os.SEEK_CUR)
            else:
                record_bytes += prop.dtype.itemsize
                if record_bytes > mesh_io._PLY_BLOCK_BYTES:
                    stream.seek(start)
                    return element.count - remaining
                fields.append((f"s{index}", prop.dtype.newbyteorder(endian)))
                stream.seek(prop.dtype.itemsize, os.SEEK_CUR)
        record_dtype = np.dtype(fields)
        stream.seek(start)
        batch_size = min(
            remaining, mesh_io._VALIDATION_BLOCK_SIZE,
            max(1, mesh_io._PLY_BLOCK_BYTES // record_dtype.itemsize),
        )
        payload = stream.read(batch_size * record_dtype.itemsize)
        complete = len(payload) // record_dtype.itemsize
        if not complete:
            raise MeshLoadError("Truncated PLY binary record")
        records = np.frombuffer(payload, dtype=record_dtype, count=complete)
        if len(list_checks) == 1:
            field_name, expected = list_checks[0]
            changed = np.flatnonzero(records[field_name] != expected)
        else:
            mismatch = np.zeros(complete, dtype=bool)
            for field_name, expected in list_checks:
                mismatch |= records[field_name] != expected
            changed = np.flatnonzero(mismatch)
        accepted = int(changed[0]) if changed.size else complete
        indices = records[vi_field][:accepted]
        if np.any(indices < 0) or np.any(indices >= point_count):
            raise MeshLoadError("Invalid PLY polygon point index")
        accepted_bytes = accepted * record_dtype.itemsize
        if update_digest is not None and accepted_bytes:
            update_digest(memoryview(payload)[:accepted_bytes])
        remaining -= accepted
        stream.seek(start + accepted_bytes)
        if changed.size:
            return element.count - remaining
    return element.count


def _check_ply_file(
    path: Path, cancel_check: Callable[[], bool] | None, digest=None,
) -> bool:
    """Preflight records that the native reader can crash on or silently truncate.

    This scan builds no geometry. VTK remains the sole geometry/attribute reader;
    SPACE comes from GetComments(), not from a second metadata parser. Return
    whether a source coordinate or corner UV can exceed display precision.
    """
    with path.open("rb") as stream:
        encoding, elements = _check_ply_header(
            stream, cancel_check, digest.update if digest is not None else None
        )
        needs_digest = any(
            (element.name == b"face" and prop.name == b"texcoord") or (
                element.name == b"vertex" and prop.name in (b"x", b"y", b"z")
                and (prop.dtype.itemsize > 4 or (prop.dtype.itemsize == 4 and prop.dtype.kind in "iu"))
            )
            for element in elements for prop in element.properties
        )
        update = digest.update if digest is not None and needs_digest else None
        point_count = next(item.count for item in elements if item.name == b"vertex")
        file_size = os.fstat(stream.fileno()).st_size
        endian = ">" if encoding == b"binary_big_endian" else "<"
        for element in elements:
            if encoding != b"ascii" and all(
                prop.count_dtype is None for prop in element.properties
            ):
                _check_cancel(cancel_check)
                size = element.count * sum(
                    prop.dtype.itemsize for prop in element.properties
                )
                if size > file_size - stream.tell():
                    raise MeshLoadError("Truncated PLY binary records")
                if update is not None:
                    _hash_stream_bytes(stream, size, cancel_check, update)
                else:
                    stream.seek(size, os.SEEK_CUR)
                continue
            first_record = 0
            if encoding != b"ascii" and element.name == b"face":
                first_record = _check_ply_binary_faces(
                    stream, element, point_count, endian, cancel_check, update
                )
            for record in range(first_record, element.count):
                if record % mesh_io._VALIDATION_BLOCK_SIZE == 0:
                    _check_cancel(cancel_check)
                if encoding == b"ascii":
                    record_bytes = mesh_io._PLY_RECORD_BYTES
                    line = stream.readline(record_bytes + 1)
                    if not line.endswith(b"\n") or len(line) > record_bytes:
                        raise MeshLoadError(
                            "Truncated/oversized PLY ASCII record; newline required"
                        )
                    if update is not None:
                        update(line)
                    fields = line.split()
                    cursor = 0
                    try:
                        for prop in element.properties:
                            count = 1
                            if prop.count_dtype is not None:
                                count = int(fields[cursor])
                                cursor += 1
                                _check_ply_list_count(prop, count)
                            if count > len(fields) - cursor:
                                raise MeshLoadError("Truncated PLY ASCII record")
                            convert = int if prop.dtype.kind in "iu" else float
                            for index in range(cursor, cursor + count):
                                value = convert(fields[index])
                                if (
                                    prop.name == b"vertex_indices"
                                    and not 0 <= value < point_count
                                ):
                                    raise MeshLoadError(
                                        "Invalid PLY polygon point index"
                                    )
                            cursor += count
                    except (IndexError, ValueError, OverflowError) as error:
                        if isinstance(error, MeshLoadError):
                            raise
                        raise MeshLoadError("Invalid PLY ASCII record") from error
                    if cursor != len(fields):
                        raise MeshLoadError("Invalid PLY ASCII record length")
                else:
                    for prop in element.properties:
                        count = 1
                        if prop.count_dtype is not None:
                            payload = stream.read(prop.count_dtype.itemsize)
                            if len(payload) != prop.count_dtype.itemsize:
                                raise MeshLoadError("Truncated PLY binary list count")
                            if update is not None:
                                update(payload)
                            count = int(
                                np.frombuffer(
                                    payload, dtype=prop.count_dtype.newbyteorder(endian)
                                )[0]
                            )
                            _check_ply_list_count(prop, count)
                        size = count * prop.dtype.itemsize
                        if size > file_size - stream.tell():
                            raise MeshLoadError("Truncated PLY binary record")
                        if prop.name == b"vertex_indices":
                            raw = stream.read(size)
                            if update is not None:
                                update(raw)
                            indices = np.frombuffer(
                                raw, dtype=prop.dtype.newbyteorder(endian)
                            )
                            if np.any(indices < 0) or np.any(indices >= point_count):
                                raise MeshLoadError("Invalid PLY polygon point index")
                        elif update is not None:
                            _hash_stream_bytes(stream, size, cancel_check, update)
                        else:
                            stream.seek(size, os.SEEK_CUR)
        if update is not None:
            while tail := stream.read(mesh_io._PLY_BLOCK_BYTES):
                _check_cancel(cancel_check)
                update(tail)
        return needs_digest


_DEFAULT_CHECK_PLY_FILE = _check_ply_file
