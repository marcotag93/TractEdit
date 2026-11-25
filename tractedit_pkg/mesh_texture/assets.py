import os
import re
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path, PureWindowsPath
from ..data_contracts import MeshTextureAsset


@dataclass(frozen=True)
class MeshTextureBudgets:
    max_materials: int = 4096
    max_images: int = 64
    max_dimension: int = 16384
    decoded_cpu_bytes: int = 512 * 1024**2
    temporary_cpu_bytes: int = 512 * 1024**2
    resident_estimate_bytes: int = 768 * 1024**2
    layer_cache_bytes: int = 512 * 1024**2
    global_cache_bytes: int = 1024 * 1024**2
    mtl_bytes: int = 16 * 1024**2
    mtl_line_bytes: int = 65536
    mtl_records: int = 100000
    diagnostic_count: int = 32
    diagnostic_bytes: int = 2048

    def __post_init__(self):
        if any(
            type(getattr(self, item.name)) is not int or getattr(self, item.name) <= 0
            for item in fields(self)
        ):
            raise ValueError("Texture budgets must be positive integers")


DEFAULT_BUDGETS = MeshTextureBudgets()
_BLOCK_BYTES = 1024 * 1024
_IMAGE_BLOCK_BYTES = 65536
_TOCTOU_FULL_HASH_LIMIT = 4 * 1024 * 1024
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


AssetResolver = Callable[[str, str], str | os.PathLike | None]


class MeshTextureError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class MeshTextureCancelled(Exception):
    """No CPU preparation is published after cancellation."""


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise MeshTextureCancelled("Texture preparation cancelled")


def _local_reference(reference: str) -> str:
    if (
        not isinstance(reference, str)
        or not reference
        or any(ord(character) < 32 for character in reference)
    ):
        raise MeshTextureError("invalid_asset_path", "Empty or invalid asset path")
    windows = PureWindowsPath(reference)
    if _SCHEME.match(reference) and not windows.drive:
        raise MeshTextureError(
            "nonlocal_asset", "Only local image and MTL files are supported"
        )
    # A URL may also look like a Windows drive to pathlib.
    if "://" in reference or reference.lower().startswith("file:"):
        raise MeshTextureError(
            "nonlocal_asset", "Only local image and MTL files are supported"
        )
    return reference.replace("\\", "/")


def _fingerprint(path, cancelled, verified=None) -> MeshTextureAsset:
    from ..session.io import (
        SessionCancelled, SessionError, checked_fingerprint, fingerprint,
    )

    try:
        check = cancelled or (lambda: False)
        if verified is None:
            return fingerprint(str(path), check)
        asset, stamp = checked_fingerprint(str(path), check)
        verified[(asset["path"], asset["sha256"])] = stamp
        return asset
    except SessionCancelled as error:
        raise MeshTextureCancelled("Texture preparation cancelled") from error
    except SessionError as error:
        raise MeshTextureError("asset_changed", str(error)) from error
    except FileNotFoundError as error:
        raise MeshTextureError("missing_asset", f"Asset not found: {path}") from error
    except PermissionError as error:
        raise MeshTextureError(
            "asset_permission", f"Cannot read asset: {path}"
        ) from error
    except OSError as error:
        raise MeshTextureError("asset_io", f"Cannot read asset: {path}") from error


def resolve_texture_asset(
    reference: str,
    base_directory: str | os.PathLike,
    *,
    replacement: str | os.PathLike | None = None,
    expected: MeshTextureAsset | None = None,
    obj_directory: Path | None = None,
    max_bytes: int | None = None,
    cancelled: Callable[[], bool] | None = None,
    _verified=None,
) -> MeshTextureAsset:
    """Resolve one exact local reference; explicit relocation verifies its digest."""
    normalized = _local_reference(reference)
    windows = PureWindowsPath(reference)
    if replacement is None:
        if windows.drive or windows.root or normalized.startswith("/"):
            raise MeshTextureError(
                "asset_resolution_required",
                "Choose a local replacement for this absolute asset",
            )
        try:
            base = Path(base_directory).resolve()
            path = (base / normalized).resolve()
        except (OSError, RuntimeError) as error:
            raise MeshTextureError("asset_io", f"Cannot resolve asset: {reference}") from error
        if not (path.is_relative_to(base) or (
            obj_directory is not None and path.is_relative_to(obj_directory.resolve())
        )):
            raise MeshTextureError(
                "asset_resolution_required", "Choose a replacement for this external asset"
            )
    else:
        path = Path(replacement)
        if not path.is_absolute() or (
            os.name != "nt" and PureWindowsPath(str(replacement)).drive
        ):
            raise MeshTextureError(
                "asset_resolution_required",
                "A replacement must be an absolute local path",
            )
    if max_bytes is not None:
        try:
            if path.stat().st_size > max_bytes:
                raise MeshTextureError(
                    "mtl_byte_limit", "MTL data exceeds its per-layer byte limit"
                )
        except FileNotFoundError as error:
            raise MeshTextureError(
                "missing_asset", f"Asset not found: {path}"
            ) from error
        except PermissionError as error:
            raise MeshTextureError(
                "asset_permission", f"Cannot read asset: {path}"
            ) from error
        except OSError as error:
            raise MeshTextureError("asset_io", f"Cannot read asset: {path}") from error
    result = _fingerprint(path, cancelled, _verified)
    if expected is not None and (result["size"], result["sha256"]) != (
        expected["size"],
        expected["sha256"],
    ):
        raise MeshTextureError(
            "asset_changed", "Relocated asset does not match size and SHA-256"
        )
    return result
