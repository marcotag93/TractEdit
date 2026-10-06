# -*- coding: utf-8 -*-
"""Versioned, non-executable session archives and source verification."""

from __future__ import annotations

import hashlib
from datetime import datetime
import json
import math
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, TypedDict
import zipfile

import numpy as np

from . import __version__
from .transactional_io import staged_new_output

FORMAT = "tractedit-session"
SCHEMA_VERSION = 2
BLOCK_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARRAY_BYTES = 2 * 1024**3
MAX_PAYLOAD_BYTES = 8 * 1024**3
MAX_MEMBERS = 10000
SESSION_EXTENSION = ".tractedit-session"
TIMESTAMP_PATTERN = r"\d{8}_\d{6}_\d{6}(?:_\d+)?"


def timestamped_path(path: str) -> Path:
    """Choose a new local-time session name without accumulating timestamps."""
    base = Path(path)
    if base.name.endswith(SESSION_EXTENSION):
        stem = base.name[: -len(SESSION_EXTENSION)]
        stem = re.sub(rf"_{TIMESTAMP_PATTERN}$", "", stem)
    else:
        stem = re.sub(
            rf"{re.escape(SESSION_EXTENSION)}-{TIMESTAMP_PATTERN}$", "", base.name
        )
    stem = stem or "session"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = base.with_name(f"{stem}_{stamp}{SESSION_EXTENSION}")
    sequence = 1
    while candidate.exists():
        candidate = base.with_name(f"{stem}_{stamp}_{sequence}{SESSION_EXTENSION}")
        sequence += 1
    return candidate


class SessionError(ValueError):
    """An invalid or unreproducible session."""


class SessionCancelled(Exception):
    """Cooperative cancellation before publication."""


class SourceResolutionError(SessionError):
    """Sources needing user relocation, with the already decoded document."""

    def __init__(self, document, issues):
        self.document = document
        self.issues = issues
        super().__init__("Cannot verify session sources: " + "; ".join(issues.values()))


class VerifiedDocument(dict):
    """Ephemeral verification receipts; never serialized as session settings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relative_sources = {}
        self.verified_sources = {}


def source_stamp(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def checked_fingerprint(path, cancelled=lambda: False, progress=lambda text: None):
    before = source_stamp(path)
    record = fingerprint(path, cancelled, progress)
    if source_stamp(path) != before:
        raise SessionError(f"Source changed while verifying: {path}")
    return record, before


class SourceRecord(TypedDict):
    path: str
    size: int
    sha256: str


class SessionDocument(TypedDict):
    sources: dict[str, SourceRecord]
    state: dict[str, Any]
    views: dict[str, Any]


def checkpoint(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise SessionCancelled()


def fingerprint(
    path: str,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[str], None] = lambda message: None,
) -> SourceRecord:
    """Hash a stable regular file with bounded temporary memory."""
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise SessionError(f"Source is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        before = os.fstat(stream.fileno())
        count = 0
        while True:
            checkpoint(cancelled)
            block = stream.read(BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            count += len(block)
            progress(f"Verifying {source.name}: {count}/{before.st_size} bytes")
        after = os.fstat(stream.fileno())
    current = source.stat()

    def identity(stat):
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    if identity(before) != identity(after) or identity(after) != identity(current):
        raise SessionError(f"Source changed while reading: {source}")
    return {"path": str(source), "size": count, "sha256": digest.hexdigest()}


def validate_sources(sources):
    if not isinstance(sources, dict) or set(sources) - {
        "tractogram",
        "anatomy",
        "parcellation",
        "odf",
    }:
        raise SessionError("Invalid source table")
    for record in sources.values():
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise SessionError("Invalid source record")
        if (
            not isinstance(record["path"], str)
            or not (
                PurePosixPath(record["path"]).is_absolute()
                or PureWindowsPath(record["path"]).is_absolute()
            )
            or type(record["size"]) is not int
            or record["size"] < 0
            or not isinstance(record["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise SessionError("Session sources must use absolute paths")


def verify_sources(sources, cancelled=lambda: False, progress=lambda message: None):
    validate_sources(sources)
    for record in sources.values():
        actual = fingerprint(record["path"], cancelled, progress)
        if actual != record:
            raise SessionError(f"Source has changed: {record['path']}")


def resolve_session_sources(
    document,
    session_path,
    source_paths=None,
    cancelled=lambda: False,
    progress=lambda text: None,
):
    """Find identical sources at original, relative or user-selected paths."""
    validate_sources(document["sources"])
    overrides = source_paths or {}
    if set(overrides) - set(document["sources"]):
        raise SessionError("Unknown source role in relocation")
    directory = Path(session_path).resolve().parent
    relative = getattr(document, "relative_sources", {})
    receipts = getattr(document, "verified_sources", {})
    resolved, verified, issues = {}, {}, {}
    for role, record in document["sources"].items():
        original = PureWindowsPath(record["path"])
        windows_path = original.is_absolute()
        if not windows_path:
            original = PurePosixPath(record["path"])
        candidates = [Path(overrides[role])] if role in overrides else []
        if role not in overrides:
            # Never interpret another OS's absolute path against the local cwd/drive.
            if windows_path == (os.name == "nt"):
                candidates.append(Path(record["path"]))
            if role in relative:
                candidates.append(directory / relative[role])
            candidates.append(directory / original.name)
        reason = "missing or changed"
        for candidate in dict.fromkeys(candidates):
            checkpoint(cancelled)
            try:
                candidate = candidate.resolve(strict=True)
                receipt = receipts.get(role)
                if (
                    receipt
                    and receipt[0] == record
                    and receipt[1] == source_stamp(candidate)
                    and str(candidate) == record["path"]
                ):
                    actual, stamp = receipt
                else:
                    actual, stamp = checked_fingerprint(candidate, cancelled, progress)
                if (actual["size"], actual["sha256"]) != (
                    record["size"],
                    record["sha256"],
                ):
                    reason = "content has changed (SHA-256 mismatch)"
                    continue
                resolved[role] = actual
                verified[role] = (actual.copy(), stamp)
                break
            except OSError as error:
                if reason != "content has changed (SHA-256 mismatch)":
                    reason = str(error)
        else:
            issues[role] = f"{role}: {record['path']} — {reason}"
    document["sources"] = {**document["sources"], **resolved}
    if isinstance(document, VerifiedDocument):
        document.verified_sources = verified
    if issues:
        raise SourceResolutionError(document, issues)
    return document


def _numeric(array: np.ndarray) -> None:
    if array.dtype.kind not in "biufc" or array.dtype.hasobject or array.ndim > 8:
        raise SessionError(
            "Only numeric arrays of at most eight dimensions are allowed"
        )


class _CheckedWriter:
    def __init__(self, stream, cancelled):
        self.stream = stream
        self.cancelled = cancelled

    def write(self, data):
        view = memoryview(data).cast("B")
        for start in range(0, len(view), BLOCK_BYTES):
            checkpoint(self.cancelled)
            self.stream.write(view[start : start + BLOCK_BYTES])
        return len(view)


def write_session(path, document, cancelled=lambda: False, progress=lambda text: None):
    """Publish one complete archive, leaving an existing destination on failure."""
    destination = Path(path).resolve()
    validate_sources(document["sources"])
    for record in document["sources"].values():
        source = Path(record["path"])
        if destination == source or (
            destination.exists() and os.path.samefile(destination, source)
        ):
            raise SessionError("A session cannot overwrite a source file")
    with staged_new_output(destination) as staged:
        with zipfile.ZipFile(
            staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            counter = 0
            payload_bytes = 0

            def encode(value):
                nonlocal counter, payload_bytes
                checkpoint(cancelled)
                if isinstance(value, np.ndarray):
                    _numeric(value)
                    if value.nbytes > MAX_ARRAY_BYTES or counter >= MAX_MEMBERS - 1:
                        raise SessionError("Session array or member limit exceeded")
                    payload_bytes += value.nbytes + 256
                    if payload_bytes > MAX_PAYLOAD_BYTES:
                        raise SessionError("Session payload limit exceeded")
                    name = f"arrays/{counter}.npy"
                    counter += 1
                    progress(f"Writing {name}")
                    with archive.open(name, "w", force_zip64=True) as stream:
                        np.lib.format.write_array(
                            _CheckedWriter(stream, cancelled), value, allow_pickle=False
                        )
                    return {"array": name}
                if isinstance(value, np.generic):
                    return encode(value.item())
                if isinstance(value, dict):
                    return {
                        "mapping": [[encode(k), encode(v)] for k, v in value.items()]
                    }
                if isinstance(value, (set, tuple)):
                    kind = "set" if isinstance(value, set) else "tuple"
                    if kind == "set" and all(
                        type(v) is int or isinstance(v, np.integer) for v in value
                    ):
                        try:
                            packed = np.fromiter(
                                value, dtype=np.int64, count=len(value)
                            )
                        except OverflowError:
                            pass
                        else:
                            return {"intset": encode(packed)}
                    return {kind: [encode(v) for v in value]}
                if isinstance(value, list):
                    return [encode(v) for v in value]
                if value is None or type(value) in (str, int, bool):
                    return value
                if type(value) is float and math.isfinite(value):
                    return value
                raise SessionError(f"Unsupported session value: {type(value).__name__}")

            manifest = {
                "format": FORMAT,
                "schema": SCHEMA_VERSION,
                "application_version": __version__,
                "conventions": "RASmm; ROI>0; MRtrix3/Tournier; native-reference-grid",
                "sources": document["sources"],
                "payload": encode(
                    {k: v for k, v in document.items() if k != "sources"}
                ),
            }
            relative_sources = {}
            for role, record in document["sources"].items():
                if PureWindowsPath(record["path"]).is_absolute() != (os.name == "nt"):
                    continue
                try:
                    relative_sources[role] = Path(
                        os.path.relpath(record["path"], destination.parent)
                    ).as_posix()
                except ValueError:  # Different Windows drives have no relative path.
                    pass
            manifest["relative_sources"] = relative_sources
            data = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode(
                "utf8"
            )
            if len(data) > MAX_MANIFEST_BYTES:
                raise SessionError("Session manifest is too large")
            if (
                sum(entry.file_size for entry in archive.infolist()) + len(data)
                > MAX_PAYLOAD_BYTES
            ):
                raise SessionError("Session payload limit exceeded")
            archive.writestr("manifest.json", data)
        checkpoint(cancelled)


def read_session(
    path,
    cancelled=lambda: False,
    progress=lambda text: None,
    *,
    source_paths=None,
    verify=True,
):
    """Validate an archive before decoding bounded, non-pickled arrays."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        physical_size = Path(path).stat().st_size
        if (
            len(entries) > MAX_MEMBERS
            or sum(entry.file_size for entry in entries) > MAX_PAYLOAD_BYTES
        ):
            raise SessionError("Session payload or member limit exceeded")
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise SessionError("Duplicate members or missing manifest")
        for entry in entries:
            name = entry.filename
            valid_array = (
                name.startswith("arrays/")
                and name.endswith(".npy")
                and name[7:-4].isdigit()
            )
            if name != "manifest.json" and not valid_array:
                raise SessionError(f"Invalid archive member: {name}")
            if (
                entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or entry.flag_bits & 1
            ):
                raise SessionError("Unsupported compression or encryption")
            if valid_array and entry.file_size > MAX_ARRAY_BYTES + 65536:
                raise SessionError("Session array limit exceeded")
            if (
                entry.compress_type == zipfile.ZIP_STORED
                and entry.compress_size != entry.file_size
            ):
                raise SessionError("Invalid stored member size")
        if archive.getinfo("manifest.json").file_size > MAX_MANIFEST_BYTES:
            raise SessionError("Session manifest is too large")

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise SessionError("Duplicate JSON key")
                result[key] = value
            return result

        manifest = json.loads(archive.read("manifest.json"), object_pairs_hook=pairs)
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != FORMAT
            or type(manifest.get("schema")) is not int
            or manifest["schema"] not in (1, SCHEMA_VERSION)
        ):
            raise SessionError("Unsupported session format or schema version")
        if manifest["schema"] == 1 and (
            any(entry.compress_type != zipfile.ZIP_STORED for entry in entries)
            or sum(entry.file_size for entry in entries) > physical_size
        ):
            raise SessionError("Invalid legacy stored archive")
        application_version = manifest.get("application_version")
        if not isinstance(application_version, str) or not application_version:
            raise SessionError("Invalid application version metadata")
        validate_sources(manifest.get("sources"))
        relative = manifest.get("relative_sources", {})
        if not isinstance(relative, dict) or set(relative) - set(manifest["sources"]):
            raise SessionError("Invalid relative source table")
        for value in relative.values():
            if (
                not isinstance(value, str)
                or not value
                or Path(value).is_absolute()
                or PureWindowsPath(value).drive
                or PureWindowsPath(value).root
            ):
                raise SessionError("Invalid relative source path")
        used = {"manifest.json"}

        def decode(value, depth=0):
            checkpoint(cancelled)
            if depth > 40:
                raise SessionError("Session nesting limit exceeded")

            def descend(v):
                return decode(v, depth + 1)

            if isinstance(value, list):
                return [descend(v) for v in value]
            if not isinstance(value, dict):
                if isinstance(value, float) and not math.isfinite(value):
                    raise SessionError("Nonfinite session setting")
                return value
            if len(value) != 1:
                raise SessionError("Invalid value encoding")
            kind, body = next(iter(value.items()))
            if kind == "array":
                if body not in names or body in used or body == "manifest.json":
                    raise SessionError("Invalid array reference")
                used.add(body)
                progress(f"Reading {body}")
                with archive.open(body) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                            stream
                        )
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                            stream
                        )
                    else:
                        raise SessionError("Unsupported NumPy header")
                    if dtype.kind not in "biufc" or dtype.hasobject or len(shape) > 8:
                        raise SessionError("Unsafe array dtype or dimensions")
                    size = math.prod(shape) * dtype.itemsize
                    if size != archive.getinfo(body).file_size - stream.tell():
                        raise SessionError("Array shape does not match its payload")
                    if size > MAX_ARRAY_BYTES:
                        raise SessionError("Session array limit exceeded")
                    result = np.empty(shape, dtype=dtype, order="F" if fortran else "C")
                    target = memoryview(result.ravel(order="K")).cast("B")
                    for start in range(0, size, BLOCK_BYTES):
                        checkpoint(cancelled)
                        block = stream.read(min(BLOCK_BYTES, size - start))
                        if len(block) != min(BLOCK_BYTES, size - start):
                            raise SessionError("Truncated array")
                        target[start : start + len(block)] = block
                    stream.read(1)
                    return result
            if kind == "mapping":
                result = {}
                for key, item in body:
                    key = descend(key)
                    if key in result:
                        raise SessionError("Duplicate mapping key")
                    result[key] = descend(item)
                return result
            if kind == "tuple":
                return tuple(descend(v) for v in body)
            if kind == "set":
                return set(descend(v) for v in body)
            if kind == "intset":
                array = descend(body)
                if (
                    not isinstance(array, np.ndarray)
                    or array.ndim != 1
                    or array.dtype != np.int64
                ):
                    raise SessionError("Invalid index set")
                return set(array.tolist())
            raise SessionError("Unknown value encoding")

        document = decode(manifest["payload"])
        if set(names) != used or not isinstance(document, dict):
            raise SessionError("Unreferenced archive members or invalid document")
        document = VerifiedDocument(document)
        document["sources"] = manifest["sources"]
        document.relative_sources = relative
        if verify:
            resolve_session_sources(document, path, source_paths, cancelled, progress)
        return document
