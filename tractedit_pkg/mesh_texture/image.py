import os
import struct
import zlib
import numpy as np
from collections.abc import Callable
from pathlib import Path
from ..data_contracts import (
    MeshTextureAsset,
    MeshTextureCosts,
    MeshTextureImage,
    MeshTextureMap,
)
from .assets import (
    MeshTextureBudgets,
    DEFAULT_BUDGETS,
    _BLOCK_BYTES,
    _IMAGE_BLOCK_BYTES,
    MeshTextureError,
    _check_cancel,
    _fingerprint,
)
from .. import mesh_texture


def image_cost(width: int, height: int, channels: int) -> MeshTextureCosts:
    if any(
        type(value) is not int or value <= 0 for value in (width, height, channels)
    ) or channels not in (3, 4):
        raise MeshTextureError(
            "invalid_image_dimensions",
            "Positive dimensions and uint8 RGB/RGBA channels required",
        )
    size = width * height * channels
    return {
        "decoded_cpu_bytes": size,
        "temporary_cpu_bytes": 2 * size,
        "resident_estimate_bytes": (width * height * 16 + 2) // 3,
    }


def _check_image_budget(
    width,
    height,
    channels,
    gl_maximum,
    budgets,
    retained_bytes=0,
    resident_bytes=0,
    extra_temporary_bytes=0,
):
    if type(gl_maximum) is not int or gl_maximum <= 0:
        raise MeshTextureError(
            "invalid_gl_limit",
            "An observed positive GL texture dimension limit is required",
        )
    if max(width, height) > min(budgets.max_dimension, gl_maximum):
        raise MeshTextureError(
            "image_dimension_limit", "Image dimensions exceed the texture limit"
        )
    costs = image_cost(width, height, channels)
    costs["temporary_cpu_bytes"] += extra_temporary_bytes
    if (
        retained_bytes + costs["decoded_cpu_bytes"]
        > min(budgets.decoded_cpu_bytes, budgets.layer_cache_bytes)
        or costs["temporary_cpu_bytes"] > budgets.temporary_cpu_bytes
        or resident_bytes + costs["resident_estimate_bytes"]
        > budgets.resident_estimate_bytes
    ):
        raise MeshTextureError(
            "texture_budget", "Image payload exceeds the texture budget"
        )
    return costs


def _read_exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise MeshTextureError("invalid_image", "Truncated image")
    return data


class _PngPixels:
    def __init__(self, width, height, channels, interlace):
        self.decoder = zlib.decompressobj()
        passes = (
            ((0, 0, 1, 1),)
            if not interlace
            else (
                (0, 0, 8, 8),
                (4, 0, 8, 8),
                (0, 4, 4, 8),
                (2, 0, 4, 4),
                (0, 2, 2, 4),
                (1, 0, 2, 2),
                (0, 1, 1, 2),
            )
        )
        self.rows = iter(
            row_width * channels
            for x, y, dx, dy in passes
            if (row_width := max(0, (width - x + dx - 1) // dx))
            for _ in range(max(0, (height - y + dy - 1) // dy))
        )
        self.remaining = 0

    def feed(self, compressed, cancelled):
        while compressed:
            _check_cancel(cancelled)
            try:
                data = self.decoder.decompress(compressed, _IMAGE_BLOCK_BYTES)
            except zlib.error as error:
                raise MeshTextureError(
                    "invalid_image", "Invalid PNG compressed pixel stream"
                ) from error
            if self.decoder.unused_data:
                raise MeshTextureError(
                    "invalid_image", "PNG has extra compressed pixel data"
                )
            compressed = self.decoder.unconsumed_tail
            position = 0
            while position < len(data):
                if not self.remaining:
                    row_bytes = next(self.rows, None)
                    if row_bytes is None or data[position] > 4:
                        raise MeshTextureError(
                            "invalid_image", "Invalid PNG pixel count or filter"
                        )
                    self.remaining = row_bytes
                    position += 1
                amount = min(self.remaining, len(data) - position)
                self.remaining -= amount
                position += amount

    def finish(self):
        if not self.decoder.eof or self.remaining or next(self.rows, None) is not None:
            raise MeshTextureError("invalid_image", "Truncated PNG pixel stream")


def _png_header(stream, cancelled, check_dimensions):
    if _read_exact(stream, 8) != b"\x89PNG\r\n\x1a\n":
        raise MeshTextureError("invalid_image", "Invalid PNG signature")
    dimensions = None
    has_data = False
    metadata_bytes = 0
    pixels = None
    ended_data = False
    while True:
        _check_cancel(cancelled)
        length, kind = struct.unpack(">I4s", _read_exact(stream, 8))
        if dimensions is None and (kind != b"IHDR" or length != 13):
            raise MeshTextureError("invalid_image", "PNG requires a first IHDR chunk")
        if kind == b"IHDR":
            if dimensions is not None or length != 13:
                raise MeshTextureError("invalid_image", "Invalid PNG IHDR")
            header = _read_exact(stream, 13)
            width, height, depth, color, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", header)
            )
            if depth != 8 or color not in (2, 6):
                raise MeshTextureError(
                    "unsupported_image", "PNG must be 8-bit RGB or RGBA"
                )
            if (
                not width
                or not height
                or compression
                or filtering
                or interlace not in (0, 1)
            ):
                raise MeshTextureError(
                    "invalid_image", "Invalid PNG dimensions or encoding"
                )
            dimensions = (width, height, 3 if color == 2 else 4)
            check_dimensions(dimensions)
            pixels = _PngPixels(*dimensions, interlace)
            crc = zlib.crc32(header, zlib.crc32(kind))
        else:
            if kind == b"IDAT" and ended_data:
                raise MeshTextureError(
                    "invalid_image", "PNG IDAT chunks must be consecutive"
                )
            if has_data and kind != b"IDAT":
                ended_data = True
            if kind in (b"iCCP", b"tRNS", b"zTXt", b"iTXt"):
                raise MeshTextureError(
                    "unsupported_image",
                    "PNG profiles, compressed text and implicit channel conversions are unsupported",
                )
            if kind == b"tEXt":
                metadata_bytes += length
            crc = zlib.crc32(kind)
            remaining = length
            while remaining:
                _check_cancel(cancelled)
                block = _read_exact(stream, min(remaining, _IMAGE_BLOCK_BYTES))
                remaining -= len(block)
                crc = zlib.crc32(block, crc)
                if kind == b"IDAT":
                    pixels.feed(block, cancelled)
            has_data |= kind == b"IDAT"
        expected = struct.unpack(">I", _read_exact(stream, 4))[0]
        if crc & 0xFFFFFFFF != expected:
            raise MeshTextureError("invalid_image", "PNG chunk checksum mismatch")
        if kind == b"IEND":
            if length or not has_data or stream.read(1):
                raise MeshTextureError(
                    "invalid_image", "Invalid PNG end or missing pixels"
                )
            width, height, channels = dimensions
            pixels.finish()
            overhead = (
                height * struct.calcsize("P")
                + 2 * width * channels
                + _BLOCK_BYTES
                + 4 * metadata_bytes
            )
            return dimensions, overhead


def _jpeg_header(stream, cancelled):
    if _read_exact(stream, 2) != b"\xff\xd8":
        raise MeshTextureError("invalid_image", "Invalid JPEG signature")
    dimensions = None
    coefficients = 0
    while True:
        _check_cancel(cancelled)
        if _read_exact(stream, 1) != b"\xff":
            raise MeshTextureError("invalid_image", "Invalid JPEG marker")
        marker = _read_exact(stream, 1)[0]
        while marker == 0xFF:
            marker = _read_exact(stream, 1)[0]
        if marker in (0, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            raise MeshTextureError("invalid_image", "Unexpected JPEG marker")
        length = struct.unpack(">H", _read_exact(stream, 2))[0]
        if length < 2:
            raise MeshTextureError("invalid_image", "Invalid JPEG segment size")
        payload = _read_exact(stream, length - 2)
        if marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
            raise MeshTextureError(
                "unsupported_image", "JPEG color profiles are unsupported"
            )
        if marker in (0xC0, 0xC1, 0xC2):
            if len(payload) < 6 or len(payload) != 6 + 3 * payload[5]:
                raise MeshTextureError("invalid_image", "Invalid JPEG frame header")
            depth, height, width, channels = struct.unpack(">BHHB", payload[:6])
            if depth != 8 or channels != 3:
                raise MeshTextureError("unsupported_image", "JPEG must be 8-bit RGB")
            dimensions = (width, height, channels)
            if marker == 0xC2:
                sampling = [
                    (payload[7 + 3 * index] >> 4, payload[7 + 3 * index] & 15)
                    for index in range(channels)
                ]
                if any(not 1 <= value <= 4 for pair in sampling for value in pair):
                    raise MeshTextureError(
                        "invalid_image", "Invalid JPEG component sampling"
                    )
                horizontal = max(pair[0] for pair in sampling)
                vertical = max(pair[1] for pair in sampling)
                coefficients = sum(
                    ((width * h + horizontal * 8 - 1) // (horizontal * 8))
                    * ((height * v + vertical * 8 - 1) // (vertical * 8))
                    * 128
                    for h, v in sampling
                )
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            raise MeshTextureError(
                "unsupported_image", "Unsupported JPEG frame encoding"
            )
        elif marker == 0xDA:
            if dimensions is None:
                raise MeshTextureError(
                    "invalid_image", "JPEG scan lacks a supported frame"
                )
            stream.seek(-2, os.SEEK_END)
            if _read_exact(stream, 2) != b"\xff\xd9":
                raise MeshTextureError("invalid_image", "JPEG end marker missing")
            width, height, channels = dimensions
            rows = min(height, 4096)
            native_extra = (
                width * rows * channels
                + rows * struct.calcsize("P")
                + coefficients
                + _BLOCK_BYTES
            )
            overhead = max(_BLOCK_BYTES, native_extra - width * height * channels)
            return dimensions, overhead


def _inspect_texture_image(asset, cancelled, gl_maximum=16384, budgets=DEFAULT_BUDGETS):
    path = Path(asset["path"])
    with path.open("rb") as stream:
        if path.suffix.lower() == ".png":
            return _png_header(
                stream,
                cancelled,
                lambda dimensions: _check_image_budget(
                    *dimensions, gl_maximum, budgets
                ),
            )
        if path.suffix.lower() in (".jpg", ".jpeg"):
            return _jpeg_header(stream, cancelled)
    raise MeshTextureError(
        "unsupported_image", "Only PNG RGB/RGBA and JPEG images are supported"
    )


def plan_texture_image(
    asset: MeshTextureAsset,
    *,
    gl_maximum: int,
    budgets: MeshTextureBudgets = DEFAULT_BUDGETS,
    retained_bytes: int = 0,
    resident_bytes: int = 0,
    cancelled=None,
) -> MeshTextureCosts:
    dimensions, overhead = mesh_texture._inspect_texture_image(asset, cancelled, gl_maximum, budgets)
    return _check_image_budget(
        *dimensions, gl_maximum, budgets, retained_bytes, resident_bytes, overhead
    )


def _decode_image(asset, dimensions, cancelled):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOImage import vtkJPEGReader, vtkPNGReader

    reader = (
        vtkPNGReader()
        if Path(asset["path"]).suffix.lower() == ".png"
        else vtkJPEGReader()
    )
    failed = []
    observers = []
    try:
        for emitter in (reader, reader.GetExecutive()):
            for event in ("ErrorEvent", "WarningEvent"):
                observers.append(
                    (
                        emitter,
                        emitter.AddObserver(
                            event,
                            lambda *args: failed.append(True) if not failed else None,
                        ),
                    )
                )
        reader.SetFileName(asset["path"])
        _check_cancel(cancelled)
        status = reader.GetExecutive().Update()
        _check_cancel(cancelled)
        if failed or not status or reader.GetErrorCode():
            raise MeshTextureError(
                "invalid_image", "Native decoder rejected this image"
            )
        output = reader.GetOutput()
        width, height, channels = dimensions
        scalars = output.GetPointData().GetScalars()
        if (
            output.GetDimensions() != (width, height, 1)
            or scalars is None
            or scalars.GetNumberOfComponents() != channels
        ):
            raise MeshTextureError(
                "invalid_image", "Decoder output differs from the image header"
            )
        view = vtk_to_numpy(scalars)
        if view.dtype != np.uint8 or view.size != width * height * channels:
            raise MeshTextureError(
                "unsupported_image", "Decoder must return uint8 RGB/RGBA"
            )
        pixels = np.empty((height, width, channels), dtype=np.uint8)
        rows = max(1, _BLOCK_BYTES // (width * channels))
        view = view.reshape(height, width, channels)
        for start in range(0, height, rows):
            _check_cancel(cancelled)
            pixels[start : start + rows] = view[start : start + rows]
        return pixels
    finally:
        for emitter, observer in observers:
            emitter.RemoveObserver(observer)


def decode_texture_image(
    asset: MeshTextureAsset,
    *,
    gl_maximum: int,
    budgets: MeshTextureBudgets = DEFAULT_BUDGETS,
    retained_bytes: int = 0,
    resident_bytes: int = 0,
    cache: "MeshTextureCache | None" = None,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
    _verified_stamp=None,
) -> MeshTextureImage:
    """Validate before native allocation; return owned pixels without a second flip."""
    from ..session.io import source_stamp

    try:
        _check_cancel(cancelled)
        before = source_stamp(asset["path"])
        if (
            before != _verified_stamp
            and _fingerprint(asset["path"], cancelled) != asset
        ):
            raise MeshTextureError("asset_changed", "Image bytes changed before decode")
        if progress is not None:
            progress("image_read")
        dimensions, overhead = mesh_texture._inspect_texture_image(
            asset, cancelled, gl_maximum, budgets
        )
        costs = _check_image_budget(
            *dimensions, gl_maximum, budgets, retained_bytes, resident_bytes, overhead
        )
        if cache is not None:
            cache.reserve_preparation(retained_bytes + costs["decoded_cpu_bytes"])
        if progress is not None:
            progress("decode")
        pixels = mesh_texture._decode_image(asset, dimensions, cancelled)
        if before != source_stamp(asset["path"]):
            raise MeshTextureError("asset_changed", "Image changed during decode")
        has_transparency = False
        if dimensions[2] == 4:
            for start in range(0, dimensions[1], 256):
                _check_cancel(cancelled)
                has_transparency |= bool(
                    np.any(pixels[start : start + 256, :, 3] != 255)
                )
        pixels.flags.writeable = False
        return {
            "asset": asset,
            "pixels": pixels,
            "has_transparency": has_transparency,
            "resident_estimate_bytes": costs["resident_estimate_bytes"],
            "temporary_cpu_bytes": costs["temporary_cpu_bytes"],
        }
    except FileNotFoundError as error:
        raise MeshTextureError("missing_asset", "Image file is missing") from error
    except PermissionError as error:
        raise MeshTextureError(
            "asset_permission", "Image file cannot be read"
        ) from error
    except (OSError, RuntimeError) as error:
        raise MeshTextureError("invalid_image", "Image could not be decoded") from error
    except MemoryError as error:
        raise MeshTextureError(
            "texture_memory", "Insufficient memory for image preparation"
        ) from error


def transform_texture_uv(
    source_uv: np.ndarray,
    diffuse_map: MeshTextureMap,
    *,
    budgets: MeshTextureBudgets = DEFAULT_BUDGETS,
    retained_bytes: int = 0,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Transform explicit UV tuples; negative values are valid, never sentinels."""
    if (
        source_uv.ndim != 2
        or source_uv.shape[1] != 2
        or source_uv.dtype not in (np.dtype("float32"), np.dtype("float64"))
    ):
        raise MeshTextureError(
            "invalid_uv", "UV data must contain two floating components"
        )
    if (
        retained_bytes + source_uv.nbytes
        > min(budgets.layer_cache_bytes, budgets.decoded_cpu_bytes)
        or 2 * source_uv.nbytes > budgets.temporary_cpu_bytes
    ):
        raise MeshTextureError(
            "texture_budget", "Derived UV buffer exceeds the texture budget"
        )
    result = np.empty_like(source_uv, order="C")
    scale, offset = diffuse_map["scale"][:2], diffuse_map["offset"][:2]
    for start in range(0, len(source_uv), 16384):
        _check_cancel(cancelled)
        source = source_uv[start : start + 16384]
        target = result[start : start + 16384]
        if not np.isfinite(source).all():
            raise MeshTextureError("invalid_uv", "UV coordinates must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            np.multiply(source, scale, out=target)
            np.add(target, offset, out=target)
        if not np.isfinite(target).all():
            raise MeshTextureError("invalid_uv", "UV transform overflowed")
    result.flags.writeable = False
    return result
