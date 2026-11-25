# -*- coding: utf-8 -*-
"""Bounded ordered XYZ tables in scanner-RAS millimetres."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
import math
import json
import os
from pathlib import Path
import re
from typing import Literal, NotRequired, TypedDict

import numpy as np

from .polylines import (
    DEFAULT_POLYLINE_COLOR, MAX_POLYLINE_POINTS, validated_color,
    validated_name, validated_points, validated_style,
)
from ..transactional_io import staged_output

MAX_CSV_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
MAX_METADATA_BYTES = 16 * 1024
BLOCK_ROWS = 4096
SIGNATURE = "# tractedit_polyline"
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
INTEGER = re.compile(r"[+-]?[0-9]+\Z")
CancelCheck = Callable[[], bool] | None


class PolylineCSVData(TypedDict):
    auto_color: NotRequired[bool]
    points: np.ndarray
    name: str
    color: tuple[float, float, float]
    line_width: float
    point_size: float
    show_points: bool
    visible: bool


class PolylineCSVCancelled(InterruptedError):
    """The table operation was cancelled before publication."""


def _checkpoint(cancel_check: CancelCheck) -> None:
    if cancel_check is not None and cancel_check():
        raise PolylineCSVCancelled("Polyline CSV operation cancelled")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate metadata key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON value")


def _metadata(payload: str) -> dict:
    if len(payload.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("Metadata byte limit exceeded")
    value = json.loads(
        payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant,
    )
    if not isinstance(value, dict) or set(value) != {
        "version", "space", "units", "name", "color", "line_width",
        "point_size", "show_points", "visible",
    }:
        raise ValueError("Invalid polyline metadata fields")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unsupported polyline metadata version")
    if value["space"] != "RAS" or value["units"] != "mm":
        raise ValueError("Polyline metadata requires RAS space and mm units")
    if type(value["visible"]) is not bool:
        raise ValueError("Polyline visibility must be boolean")
    if not isinstance(value["color"], list) or any(
        type(component) not in (int, float) for component in value["color"]
    ):
        raise ValueError("Polyline color must contain numeric RGB values")
    return {
        "name": validated_name(value["name"]),
        "color": validated_color(value["color"]),
        "visible": value["visible"],
        **validated_style(
            value["line_width"], value["point_size"], value["show_points"],
        ),
    }


def _cells(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return line.split()
    cells = next(csv.reader([line], delimiter=delimiter, strict=True))
    return [cell.strip() for cell in cells]


def load_polyline_csv(
    path: str | os.PathLike[str], *, delimiter: str = ",", indexed: bool | None = None,
    point_count_header: bool = False, space: Literal["RAS", "LPS"] | None = None,
    ignore_extra_columns: bool = False, cancel_check: CancelCheck = None,
    mark_default_color: bool = False,
) -> PolylineCSVData:
    """Read one ordered trace; unannotated tables require an explicit space."""
    if delimiter not in (",", ";", "\t", "whitespace"):
        raise ValueError("Unsupported polyline delimiter")
    if (
        indexed is not None and type(indexed) is not bool
        or type(point_count_header) is not bool
        or type(ignore_extra_columns) is not bool
    ):
        raise ValueError("Polyline layout options must be boolean")
    if space is not None and space not in ("RAS", "LPS"):
        raise ValueError("Unsupported polyline coordinate space")
    _checkpoint(cancel_check)
    path = Path(path)
    blocks = []
    block = np.empty((min(BLOCK_ROWS, MAX_POLYLINE_POINTS), 3), dtype=np.float64)
    used = total = scanned = line_number = 0
    metadata = None
    count = count_line = previous_index = None
    previous_point = None
    header_seen = False
    index_header = ["index", "x", "y", "z"]
    expected_header = index_header if indexed else ["x", "y", "z"]
    with path.open("rb") as stream:
        while True:
            if line_number % BLOCK_ROWS == 0:
                _checkpoint(cancel_check)
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            line_number += 1
            scanned += len(raw)
            try:
                if scanned > MAX_CSV_BYTES:
                    raise ValueError("File byte limit exceeded")
                if len(raw) > MAX_LINE_BYTES:
                    raise ValueError("Line byte limit exceeded")
                text = raw.decode("utf-8-sig" if line_number == 1 else "utf-8").rstrip("\r\n")
                if delimiter != "\t" or text.lstrip().startswith(("#", "//")):
                    text = text.strip()
                if not text.strip():
                    continue
                if text.startswith(SIGNATURE):
                    if (
                        metadata is not None or total or header_seen
                        or count is not None
                    ):
                        raise ValueError("Metadata must occur once before the table")
                    metadata = _metadata(text[len(SIGNATURE):].strip())
                    if space == "LPS":
                        raise ValueError("Coordinate space conflicts with RAS metadata")
                    continue
                comment = text.startswith(("#", "//"))
                if comment:
                    text = text[1 if text.startswith("#") else 2:].strip()
                    if total or header_seen:
                        continue
                    try:
                        cells = _cells(text, delimiter)
                    except csv.Error:
                        continue
                else:
                    cells = _cells(text, delimiter)
                header_cells = (
                    [cell.lower() for cell in cells]
                    if not total or (cells and cells[0].isalpha()) else []
                )
                if ignore_extra_columns:
                    columns = len(expected_header)
                    if indexed is None and header_cells[:4] == index_header:
                        columns = len(index_header)
                    cells = cells[:columns]
                    header_cells = header_cells[:columns]
                if comment and header_cells != expected_header and not (
                    indexed is None and header_cells == index_header
                ):
                    continue
                if indexed is None and header_cells == index_header:
                    indexed = True
                    expected_header = index_header
                if header_cells == expected_header:
                    if total or header_seen:
                        raise ValueError("Unexpected or repeated header")
                    header_seen = True
                    indexed = bool(indexed)
                    continue
                if point_count_header and count is None and not total:
                    if (
                        header_seen or len(cells) != 1
                        or not INTEGER.fullmatch(cells[0])
                    ):
                        raise ValueError("Expected an initial integer point count")
                    count = int(cells[0])
                    count_line = line_number
                    if not 2 <= count <= MAX_POLYLINE_POINTS:
                        raise ValueError("Point count limit exceeded")
                    continue
                if point_count_header and count is None:
                    raise ValueError("Missing initial point count")
                if indexed is None:
                    indexed = False
                if len(cells) != len(expected_header) or any(
                    not cell for cell in cells
                ):
                    raise ValueError("Expected XYZ or explicitly indexed XYZ columns")
                if indexed:
                    if not INTEGER.fullmatch(cells[0]):
                        raise ValueError("Index must be an integer")
                    index = int(cells.pop(0))
                    if previous_index is not None and index != previous_index + 1:
                        raise ValueError("Indices must be contiguous and increasing")
                    previous_index = index
                if any(not NUMBER.fullmatch(cell) for cell in cells):
                    raise ValueError("Coordinates must be finite decimal numbers")
                point = tuple(map(float, cells))
                if not all(map(math.isfinite, point)):
                    raise ValueError("Coordinates must be finite")
                if point == previous_point:
                    raise ValueError("Adjacent vertices must be distinct")
                if total >= MAX_POLYLINE_POINTS:
                    raise ValueError("Vertex limit exceeded")
                if used == len(block):
                    blocks.append(block)
                    block = np.empty((min(BLOCK_ROWS, MAX_POLYLINE_POINTS - total), 3))
                    used = 0
                block[used] = point
                used += 1
                total += 1
                previous_point = point
            except (ValueError, TypeError, csv.Error, RecursionError) as error:
                raise ValueError(f"Line {line_number}: {error}") from error
    _checkpoint(cancel_check)
    if count is not None and count != total:
        raise ValueError(f"Line {count_line}: Point count does not match table")
    if total < 2:
        raise ValueError(
            f"Line {max(1, line_number)}: A polyline needs at least two vertices"
        )
    if metadata is None:
        if space is None:
            raise ValueError("An explicit RAS or LPS coordinate space is required")
        metadata = {
            "name": validated_name(path.stem), "color": DEFAULT_POLYLINE_COLOR,
            "visible": True, **validated_style(),
        }
        if mark_default_color:
            metadata["auto_color"] = True
    blocks.append(block[:used])
    points = validated_points(np.concatenate(blocks))
    if space == "LPS":
        points[:, :2] *= -1
    return {"points": points, **metadata}


def save_polyline_csv(
    path: str | os.PathLike[str], points: np.ndarray,
    color: Sequence[float] = DEFAULT_POLYLINE_COLOR, *,
    profile: Literal["xyz", "tractedit"] = "xyz", header: bool = True,
    name: str | None = None, line_width: float = 3.0, point_size: float = 7.0,
    show_points: bool = True, visible: bool = True,
    protected_paths: Sequence[str | os.PathLike[str]] = (),
    cancel_check: CancelCheck = None,
) -> None:
    """Snapshot canonical data and atomically publish one UTF-8 LF table."""
    if profile not in ("xyz", "tractedit") or type(header) is not bool:
        raise ValueError("Invalid CSV profile or header option")
    _checkpoint(cancel_check)
    from .polylines import protected_export_path

    destination = protected_export_path(path, protected_paths)
    values = validated_points(points).copy()
    metadata = {
        "version": 1, "space": "RAS", "units": "mm",
        "name": validated_name(destination.stem if name is None else name),
        "color": validated_color(color),
        **validated_style(line_width, point_size, show_points), "visible": visible,
    }
    payload = json.dumps(
        metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    )
    _metadata(payload)
    with staged_output(destination) as staged:
        with staged.open("w", encoding="utf-8", newline="\n") as stream:
            if profile == "tractedit":
                stream.write(f"{SIGNATURE} {payload}\n")
            if header:
                stream.write("x,y,z\n")
            for start in range(0, len(values), BLOCK_ROWS):
                _checkpoint(cancel_check)
                stream.writelines(
                    ",".join(format(value, ".17g") for value in point) + "\n"
                    for point in values[start:start + BLOCK_ROWS]
                )
        _checkpoint(cancel_check)
        protected_export_path(destination, protected_paths)
