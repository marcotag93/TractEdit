import hashlib
import re
import numpy as np
from dataclasses import fields
from pathlib import Path
from ..data_contracts import MeshTextureFaceSummary
from .assets import (
    _BLOCK_BYTES,
    _TOCTOU_FULL_HASH_LIMIT,
    MeshTextureError,
    _check_cancel,
    _local_reference,
    _fingerprint,
)
from .mtl import _tokens, _material_name, _empty_material
from .. import mesh_texture


_OBJ_ASSET_RECORD = re.compile(
    rb"(?m)^[ \t\v\f]*(?:mtllib|usemtl)(?:[ \t\v\f][^\n]*|\r?$)"
)


def _obj_asset_records(stream, cancelled):
    from ..mesh_io import _OBJ_LINE_BYTES

    carry = b""
    while True:
        _check_cancel(cancelled)
        incoming = stream.read(_BLOCK_BYTES)
        block = carry + incoming
        boundary = block.rfind(b"\n") + 1 if incoming else len(block)
        carry = block[boundary:]
        if len(carry) > _OBJ_LINE_BYTES:
            raise MeshTextureError(
                "obj_line_limit", "OBJ exceeds its native line limit"
            )
        for match in _OBJ_ASSET_RECORD.finditer(block, 0, boundary):
            raw = match.group()
            if len(raw) > _OBJ_LINE_BYTES:
                raise MeshTextureError(
                    "obj_line_limit", "OBJ exceeds its native line limit"
                )
            yield raw
        if not incoming:
            return


def _parse_obj_reference_record(raw, budgets, libraries, materials, diagnostics):
    command = raw.lstrip().split(None, 1)
    if not command or command[0] not in (b"mtllib", b"usemtl"):
        return
    try:
        text = raw.decode("ascii")
        tokens = _tokens(text, material_record=True)[1:]
        if command[0] == b"usemtl":
            name = _material_name(tokens)
            materials.setdefault(name, _empty_material(name))
            if len(materials) > budgets.max_materials:
                raise MeshTextureError(
                    "material_limit",
                    "OBJ material count exceeds the texture limit",
                )
        elif tokens:
            libraries.extend(_local_reference(token[0]) for token in tokens)
            if len(libraries) > budgets.mtl_records:
                raise MeshTextureError(
                    "mtl_record_limit", "Too many material library references"
                )
        else:
            raise MeshTextureError(
                "missing_library", "mtllib requires a library path"
            )
    except (MeshTextureError, UnicodeDecodeError) as error:
        if isinstance(error, MeshTextureError) and error.code.endswith("limit"):
            raise
        diagnostics.add(getattr(error, "code", "invalid_obj_text"), str(error))


def _read_obj_references(path, budgets, diagnostics, cancelled):
    libraries, materials = [], {}
    with Path(path).open("rb") as stream:
        for raw in _obj_asset_records(stream, cancelled):
            _check_cancel(cancelled)
            _parse_obj_reference_record(raw, budgets, libraries, materials, diagnostics)
    return libraries, materials


def _scan_obj_stream(
    path, budgets, cancelled, *,
    collect_references: bool = False,
    diagnostics=None,
    compute_source: bool = False,
):
    """Single-pass OBJ scanner combining SHA-256, asset directives and face UV layout."""
    from .._numba_aot import scan_obj_texture_block
    from ..mesh_io import _OBJ_LINE_BYTES
    from ..session.io import source_stamp

    resolved = Path(path).resolve(strict=True)
    before_stamp = None
    digest = None
    if compute_source:
        try:
            before_stamp = source_stamp(resolved)
        except FileNotFoundError as error:
            raise MeshTextureError("missing_asset", f"Asset not found: {path}") from error
        except PermissionError as error:
            raise MeshTextureError("asset_permission", f"Cannot read asset: {path}") from error
        except OSError as error:
            raise MeshTextureError("asset_io", f"Cannot read asset: {path}") from error
        digest = hashlib.sha256()

    libraries, materials = [], {}
    layout: dict[str, MeshTextureFaceSummary] = {}
    material = ""
    total_faces = 0
    coverage: bytearray | None = None
    continuing, width, complete = False, 0, True
    max_cov_bytes = min(
        budgets.temporary_cpu_bytes,
        budgets.layer_cache_bytes,
        budgets.global_cache_bytes,
    )
    out_state = np.zeros(2, dtype=np.int64)

    def add_faces(face_count: int, corners: int, has_uv: bool) -> None:
        nonlocal total_faces, coverage
        summary = layout.setdefault(
            material, {"face_count": 0, "corner_count": 0, "complete_uv": True}
        )
        summary["face_count"] += face_count
        summary["corner_count"] += corners
        summary["complete_uv"] &= has_uv
        if coverage is None:
            if not has_uv:
                if 2 * (total_faces + face_count) > max_cov_bytes:
                    raise MeshTextureError(
                        "texture_budget", "UV coverage exceeds the budget"
                    )
                coverage = bytearray(b"\x01") * total_faces
                coverage.extend(b"\x00" * face_count)
        else:
            if 2 * (len(coverage) + face_count) > max_cov_bytes:
                raise MeshTextureError(
                    "texture_budget", "UV coverage exceeds the budget"
                )
            coverage.extend((b"\x01" if has_uv else b"\x00") * face_count)
        total_faces += face_count
        if len(layout) > budgets.max_materials + 1:
            raise MeshTextureError("material_limit", "Too many face materials")

    def process_line(raw: bytes) -> None:
        nonlocal material, continuing, width, complete
        if len(raw) > _OBJ_LINE_BYTES:
            raise MeshTextureError("obj_line_limit", "OBJ exceeds its native line limit")
        raw_no_nl = raw[:-1] if raw.endswith(b"\n") else raw
        stripped = raw_no_nl.lstrip()
        head = stripped.split(None, 1)[0] if stripped else b""
        if collect_references and head in (b"mtllib", b"usemtl"):
            _parse_obj_reference_record(
                raw_no_nl, budgets, libraries, materials, diagnostics
            )
        fields = raw.split(b"#", 1)[0].split()
        if not fields and not continuing:
            return
        if not continuing:
            command, fields = fields[0], fields[1:]
            if command == b"usemtl":
                parts = raw.split(None, 2)
                if len(parts) >= 2:
                    if parts[1].isascii():
                        material = parts[1].decode("ascii")
                    elif not collect_references:
                        material = parts[1].decode("ascii")
                return
            if command != b"f":
                return
            width, complete = 0, True
        continuing = bool(fields and fields[-1] == b"\\")
        if continuing:
            fields = fields[:-1]
        width += len(fields)
        complete &= all(
            len(parts := corner.split(b"/")) > 1 and bool(parts[1])
            for corner in fields
        )
        if continuing:
            return
        add_faces(1, width, complete)

    carry = b""
    total_bytes = 0
    try:
        with resolved.open("rb") as stream:
            while True:
                _check_cancel(cancelled)
                incoming = stream.read(_BLOCK_BYTES)
                if incoming:
                    total_bytes += len(incoming)
                    if digest is not None:
                        digest.update(incoming)
                eof = not incoming
                block = carry + incoming
                if eof:
                    boundary = len(block)
                    carry = b""
                else:
                    boundary = block.rfind(b"\n") + 1
                    carry = block[boundary:]
                    if len(carry) > _OBJ_LINE_BYTES:
                        raise MeshTextureError(
                            "obj_line_limit", "OBJ exceeds its native line limit"
                        )
                    if not boundary:
                        continue
                if boundary:
                    view = np.frombuffer(block, dtype=np.uint8, count=boundary)
                    pos = 0
                    while pos < boundary:
                        _check_cancel(cancelled)
                        if not continuing:
                            next_pos = int(
                                scan_obj_texture_block(
                                    view, pos, boundary, _OBJ_LINE_BYTES, out_state
                                )
                            )
                            if next_pos > pos:
                                tex_tri = int(out_state[0])
                                untex_tri = int(out_state[1])
                                if tex_tri:
                                    add_faces(tex_tri, 3 * tex_tri, True)
                                elif untex_tri:
                                    add_faces(untex_tri, 3 * untex_tri, False)
                                pos = next_pos
                                continue
                        nl = block.find(b"\n", pos, boundary)
                        line_end = boundary if nl < 0 else nl + 1
                        process_line(block[pos:line_end])
                        pos = line_end
                if eof:
                    break
    except FileNotFoundError as error:
        raise MeshTextureError("missing_asset", f"Asset not found: {path}") from error
    except PermissionError as error:
        raise MeshTextureError("asset_permission", f"Cannot read asset: {path}") from error
    except OSError as error:
        raise MeshTextureError("asset_io", f"Cannot read asset: {path}") from error

    if continuing:
        raise MeshTextureError("invalid_obj_text", "Incomplete OBJ face")
    face_uv = None
    if not all(summary["complete_uv"] for summary in layout.values()):
        face_uv = np.frombuffer(coverage, dtype=np.uint8).copy()
        face_uv.flags.writeable = False

    source = None
    if compute_source:
        if source_stamp(resolved) != before_stamp:
            raise MeshTextureError("asset_changed", f"Source changed while verifying: {path}")
        source = {
            "path": str(resolved),
            "size": total_bytes,
            "sha256": digest.hexdigest(),
        }
    return source, before_stamp, libraries, materials, layout, face_uv


def _read_obj_face_layout(
    path, budgets, cancelled,
) -> tuple[dict[str, MeshTextureFaceSummary], np.ndarray | None]:
    _, _, _, _, layout, face_uv = mesh_texture._scan_obj_stream(
        path, budgets, cancelled, collect_references=False, compute_source=False,
    )
    return layout, face_uv


_DEFAULT_READ_OBJ_REFERENCES = _read_obj_references
_DEFAULT_READ_OBJ_FACE_LAYOUT = _read_obj_face_layout


def _validate_obj_index(path: Path, obj_index, cancelled):
    """Return ``(source, before_stamp)`` when ``obj_index`` matches the current file."""
    from ..session.io import source_stamp

    if not isinstance(obj_index, dict):
        return None
    try:
        if Path(obj_index["path"]).resolve(strict=True) != path:
            return None
        current_stamp = source_stamp(path)
        if (
            current_stamp != obj_index["stamp"]
            or current_stamp[2] != obj_index["size"]
        ):
            return None
        source = {
            "path": str(path),
            "size": obj_index["size"],
            "sha256": obj_index["sha256"],
        }
        if current_stamp[2] <= _TOCTOU_FULL_HASH_LIMIT:
            if _fingerprint(path, cancelled) != source:
                return None
        return source, current_stamp
    except (KeyError, TypeError, OSError, MeshTextureError):
        return None


def _layout_from_obj_index(index_data, budgets):
    raw_layout = index_data["face_layout"]
    if len(raw_layout) > budgets.max_materials + 1:
        raise MeshTextureError("material_limit", "Too many face materials")
    face_layout = {name: dict(summary) for name, summary in raw_layout.items()}
    face_uv = None
    uv_bytes = index_data["face_uv_bytes"]
    if uv_bytes is not None:
        max_cov_bytes = min(
            budgets.temporary_cpu_bytes,
            budgets.layer_cache_bytes,
            budgets.global_cache_bytes,
        )
        if 2 * len(uv_bytes) > max_cov_bytes:
            raise MeshTextureError("texture_budget", "UV coverage exceeds the budget")
        face_uv = np.frombuffer(uv_bytes, dtype=np.uint8).copy()
        face_uv.flags.writeable = False
    return face_layout, face_uv


def _verify_obj_unchanged(path: Path, source, before_stamp, cancelled, message: str):
    from ..session.io import source_stamp

    try:
        changed = before_stamp is None or source_stamp(path) != before_stamp
    except OSError:
        changed = True
    if changed or (
        source["size"] <= _TOCTOU_FULL_HASH_LIMIT
        and _fingerprint(path, cancelled) != source
    ):
        raise MeshTextureError("asset_changed", message)
