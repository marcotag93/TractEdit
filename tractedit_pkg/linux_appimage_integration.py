# -*- coding: utf-8 -*-
"""Linux AppImage desktop and MIME integration for the current user."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


DESKTOP_FILENAME = "tractedit-appimage.desktop"
SYSTEM_DESKTOP_PATH = Path("/usr/share/applications/tractedit.desktop")
MIME_FILENAME = "tractedit-mime.xml"
APP_ICON_NAME = "tractedit-appimage"
_MIME_ICONS = Path("icons/hicolor/256x256/mimetypes")
PAYLOAD_HASH_KEY = "X-TractEdit-Payload-SHA256"
MANIFEST_FILENAME = ".tractedit-appimage-files.json"
LOCK_FILENAME = ".tractedit-appimage-integration.lock"
SUBPROCESS_TIMEOUT_S = 30
REQUIRED_HOST_TOOLS = ("update-mime-database", "update-desktop-database")
EXEC_PREFIX = '/bin/sh -c "exec \\\\"\\\\$@\\\\"" sh '
DEFAULT_MIME_ICON_NAMES = (
    "application-x-trk.png",
    "application-x-trx.png",
    "application-x-tck.png",
    "application-x-vtk.png",
    "application-x-vtp.png",
    "application-x-nifti.png",
    "application-x-tractedit-session.png",
)


class IntegrationError(RuntimeError):
    """Raised when user-scope AppImage integration fails."""


class IntegrationState(str, Enum):
    """Observed state of user-scope AppImage desktop integration."""

    UNREGISTERED = "unregistered"
    REGISTERED_CURRENT = "registered_current"
    REGISTERED_OUTDATED = "registered_outdated"
    REGISTERED_OTHER_VALID = "registered_other_valid"
    REGISTERED_BROKEN = "registered_broken"
    PARTIAL = "partial"


@dataclass(frozen=True)
class AppImageContext:
    """Validated runtime paths for an executing AppImage."""

    image: Path
    appdir: Path
    desktop_template: Path
    mime_package: Path
    app_icon: Path
    mime_icons_dir: Path

    def mime_icons(self) -> list[Path]:
        files = (p for p in self.mime_icons_dir.iterdir() if p.is_file())
        return sorted(files, key=lambda p: p.name)


@dataclass(frozen=True)
class IntegrationStatus:
    """Read-only snapshot of desktop integration for the current user."""

    state: IntegrationState
    data_home: Path
    desktop_path: Path
    registered_exec: Path | None
    payload_sha256: str
    system_deb_present: bool
    external_integrator_present: bool
    is_ephemeral_location: bool
    missing_tools: tuple[str, ...]
    removable: bool


@dataclass(frozen=True)
class OperationResult:
    """Outcome of a registration or removal action."""

    ok: bool
    message: str
    warnings: tuple[str, ...] = ()
    status: IntegrationStatus | None = None


def resolve_xdg_data_home(env: Mapping[str, str] | None = None) -> Path:
    """Return XDG_DATA_HOME when absolute, otherwise ~/.local/share."""
    mapping = os.environ if env is None else env
    raw = mapping.get("XDG_DATA_HOME", "")
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
    return Path.home() / ".local" / "share"


def _is_executable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name == "nt":
        return True
    return os.access(path, os.X_OK)


def detect_appimage_context(
    env: Mapping[str, str] | None = None,
    *,
    platform: str | None = None,
) -> AppImageContext | None:
    """Return AppImage asset paths when running inside a real Linux AppImage."""
    active_platform = sys.platform if platform is None else platform
    if not active_platform.startswith("linux"):
        return None
    mapping = os.environ if env is None else env
    raw_image = mapping.get("APPIMAGE", "")
    raw_appdir = mapping.get("APPDIR", "")
    if not raw_image or not raw_appdir:
        return None

    image = Path(raw_image)
    appdir = Path(raw_appdir)
    if not _is_executable_file(image) or not appdir.is_dir():
        return None
    if getattr(sys, "frozen", False) and not _same_path(
        Path(sys.executable), appdir / "usr/bin/tractedit"
    ):
        return None

    desktop_template = appdir / "tractedit.desktop"
    mime_package = appdir / "usr/share/mime/packages" / MIME_FILENAME
    app_icon = appdir / "tractedit.png"
    mime_icons_dir = appdir / "usr/share/icons/hicolor/256x256/mimetypes"
    if not (
        desktop_template.is_file()
        and mime_package.is_file()
        and app_icon.is_file()
        and mime_icons_dir.is_dir()
    ):
        return None

    return AppImageContext(
        image=image.resolve(),
        appdir=appdir,
        desktop_template=desktop_template,
        mime_package=mime_package,
        app_icon=app_icon,
        mime_icons_dir=mime_icons_dir,
    )


def _download_dir(env: Mapping[str, str]) -> Path:
    config_raw = env.get("XDG_CONFIG_HOME", "").strip()
    config_home = (
        Path(config_raw)
        if config_raw and Path(config_raw).is_absolute()
        else Path.home() / ".config"
    )
    user_dirs = config_home / "user-dirs.dirs"
    try:
        for line in user_dirs.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("XDG_DOWNLOAD_DIR="):
                continue
            value = stripped.split("=", 1)[1].strip().strip('"')
            value = value.replace("$HOME", str(Path.home()))
            candidate = Path(value)
            if candidate.is_absolute():
                return candidate
    except (OSError, UnicodeError):
        pass
    return Path.home() / "Downloads"


def is_ephemeral_location(
    image: Path,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return True when the AppImage resides in Downloads or a temp directory."""
    mapping = os.environ if env is None else env
    resolved = image.resolve()
    ephemeral_roots = [_download_dir(mapping), Path("/tmp"), Path("/var/tmp")]
    tmpdir = mapping.get("TMPDIR", "").strip()
    if tmpdir and Path(tmpdir).is_absolute():
        ephemeral_roots.append(Path(tmpdir))
    for root in ephemeral_roots:
        try:
            if resolved.is_relative_to(root.resolve()):
                return True
        except OSError:
            continue
    return False


def escape_exec_arg(path: str) -> str:
    """Escape a path for a double-quoted argument in a desktop Exec value."""
    escaped = path.replace("%", "%%")
    for char in ("\\", '"', "`", "$"):
        escaped = escaped.replace(char, "\\" + char)
    escaped = escaped.replace("\\", "\\\\")
    return escaped.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def escape_desktop_string(value: str) -> str:
    """Escape a string value per the freedesktop Desktop Entry Specification."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    if escaped.startswith(" "):
        escaped = "\\s" + escaped[1:]
    return escaped[:-1] + "\\s" if escaped.endswith(" ") else escaped


def unescape_desktop_string(value: str) -> str:
    """Decode a freedesktop Desktop Entry string value."""
    result: list[str] = []
    index = 0
    length = len(value)
    mapping = {"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}
    while index < length:
        char = value[index]
        if char == "\\" and index + 1 < length:
            nxt = value[index + 1]
            if nxt in mapping:
                result.append(mapping[nxt])
                index += 2
                continue
        result.append(char)
        index += 1
    return "".join(result)


def parse_desktop_entry(text: str) -> dict[str, str]:
    """Parse key/value pairs from the [Desktop Entry] section."""
    entries: dict[str, str] = {}
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == "[Desktop Entry]"
            continue
        if in_section and "=" in line:
            key, value = line.split("=", 1)
            entries[key.strip()] = value.strip()
    return entries


def _parse_exec_first_arg(decoded_exec: str) -> str | None:
    text = decoded_exec.lstrip()
    if not text:
        return None
    if text[0] != '"':
        return text.split(None, 1)[0]
    chars: list[str] = []
    index = 1
    length = len(text)
    while index < length:
        char = text[index]
        if (
            char == "\\"
            and index + 1 < length
            and text[index + 1] in ("\\", '"', "`", "$")
        ):
            chars.append(text[index + 1])
            index += 2
            continue
        if char == '"':
            return "".join(chars)
        chars.append(char)
        index += 1
    return None


def extract_registered_executable(entries: Mapping[str, str]) -> Path | None:
    """Extract the target executable path from parsed desktop entry keys."""
    raw_try = entries.get("TryExec", "").strip()
    if raw_try:
        decoded = unescape_desktop_string(raw_try)
        if decoded:
            return Path(decoded)

    raw_exec = entries.get("Exec", "").strip()
    if not raw_exec:
        return None
    decoded_exec = unescape_desktop_string(raw_exec.removeprefix(EXEC_PREFIX))
    first = _parse_exec_first_arg(decoded_exec)
    if not first:
        return None
    return Path(first.replace("%%", "%"))


def compute_payload_sha256(context: AppImageContext) -> str:
    """Digest the packaged desktop assets and MIME icon metadata."""
    digest = hashlib.sha256()
    for label, path in (
        (b"desktop", context.desktop_template),
        (b"mime", context.mime_package),
        (b"app_icon", context.app_icon),
    ):
        digest.update(label + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for icon in context.mime_icons():
        digest.update(icon.name.encode("utf-8") + b"\0")
        digest.update(str(icon.stat().st_size).encode("ascii") + b"\0")
    return digest.hexdigest()


def render_desktop_entry(context: AppImageContext, payload_sha256: str) -> str:
    """Build the user-scope .desktop file contents for the current AppImage."""
    resolved = str(context.image.resolve())
    exec_arg = escape_exec_arg(resolved)
    try_exec = escape_desktop_string(resolved)
    template = context.desktop_template.read_text(encoding="utf-8")
    lines: list[str] = []
    exec_replaced = False
    for line in template.splitlines():
        if line == "Exec=tractedit %f":
            prefix = EXEC_PREFIX if "%" in resolved or "=" in resolved else ""
            lines.append(f'Exec={prefix}"{exec_arg}" %f')
            lines.append(f"TryExec={try_exec}")
            exec_replaced = True
        elif line == "Icon=tractedit":
            lines.append(f"Icon={APP_ICON_NAME}")
        elif not line.startswith(f"{PAYLOAD_HASH_KEY}="):
            lines.append(line)
    if not exec_replaced:
        raise IntegrationError("Desktop template is missing 'Exec=tractedit %f'.")
    lines.append(f"{PAYLOAD_HASH_KEY}={payload_sha256}")
    return "\n".join(lines) + "\n"


def missing_host_tools() -> tuple[str, ...]:
    """Return any required freedesktop host utilities absent from PATH."""
    return tuple(tool for tool in REQUIRED_HOST_TOOLS if not shutil.which(tool))


def detect_system_deb(system_desktop: Path = SYSTEM_DESKTOP_PATH) -> bool:
    """Return True when a system-installed TractEdit package is active."""
    if not system_desktop.is_file():
        return False
    if _is_executable_file(Path("/usr/bin/tractedit")):
        return True
    command = shutil.which("tractedit")
    appdir = os.environ.get("APPDIR")
    return bool(command and not (
        appdir and Path(command).resolve().is_relative_to(Path(appdir).resolve())
    ))


def detect_external_integrator(applications_dir: Path) -> bool:
    """Return True if another user .desktop entry already integrates TractEdit."""
    if not applications_dir.is_dir():
        return False
    for path in applications_dir.glob("*.desktop"):
        if path.name == DESKTOP_FILENAME or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            entries = parse_desktop_entry(text)
        except OSError:
            continue
        is_tractedit = (
            entries.get("Name") == "TractEdit"
            or entries.get("StartupWMClass") == "tractedit"
        )
        if not is_tractedit:
            continue
        target = extract_registered_executable(entries)
        if target is not None and (
            _is_executable_file(target) if target.is_absolute()
            else shutil.which(str(target)) is not None
        ):
            return True
    return False


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


class _InstalledFiles:
    def __init__(self, data: Path) -> None:
        self.desktop = data / "applications" / DESKTOP_FILENAME
        self.mime = data / "mime/packages" / MIME_FILENAME
        self.icon = data / "icons/hicolor/256x256/apps" / f"{APP_ICON_NAME}.png"
        self.mime_icons = data / _MIME_ICONS

    def paths(self, icon_names: Iterable[str]) -> Iterator[Path]:
        yield from (self.desktop, self.mime, self.icon)
        for name in icon_names:
            yield self.mime_icons / name


def inspect_integration_status(
    context: AppImageContext,
    data_home: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    system_desktop: Path | None = None,
) -> IntegrationStatus:
    """Inspect user-scope AppImage integration without modifying disk state."""
    data = resolve_xdg_data_home(env) if data_home is None else Path(data_home)
    files = _InstalledFiles(data)
    desktop_path = files.desktop

    payload_sha256 = compute_payload_sha256(context)
    tools_missing = missing_host_tools()
    deb_present = detect_system_deb(system_desktop or SYSTEM_DESKTOP_PATH)
    external_present = detect_external_integrator(desktop_path.parent)
    ephemeral = is_ephemeral_location(context.image, env)

    has_any_artifact = desktop_path.is_file() or (data / MANIFEST_FILENAME).is_file()
    if not has_any_artifact:
        state = IntegrationState.UNREGISTERED
        registered_exec: Path | None = None
    elif not desktop_path.is_file():
        state = IntegrationState.PARTIAL
        registered_exec = None
    else:
        try:
            entries = parse_desktop_entry(desktop_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            entries = {}
        registered_exec = extract_registered_executable(entries)
        recorded_hash = entries.get(PAYLOAD_HASH_KEY, "")
        if registered_exec is None:
            state = IntegrationState.PARTIAL
        elif _same_path(registered_exec, context.image):
            expected = files.paths(icon.name for icon in context.mime_icons())
            files_present = all(path.is_file() for path in expected)
            mime_cache_ok = (data / "mime/mime.cache").is_file()
            mimeinfo_path = desktop_path.with_name("mimeinfo.cache")
            try:
                mimeinfo_ok = (
                    mimeinfo_path.is_file()
                    and DESKTOP_FILENAME in mimeinfo_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError):
                mimeinfo_ok = False
            try:
                mime_matches = (
                    files.mime.is_file()
                    and files.mime.read_bytes() == context.mime_package.read_bytes()
                )
            except OSError:
                mime_matches = False

            if not (files_present and mime_cache_ok and mimeinfo_ok):
                state = IntegrationState.PARTIAL
            elif (
                recorded_hash != payload_sha256 or not mime_matches
                or entries.get("Exec") != parse_desktop_entry(
                    render_desktop_entry(context, payload_sha256)
                )["Exec"]
            ):
                state = IntegrationState.REGISTERED_OUTDATED
            else:
                state = IntegrationState.REGISTERED_CURRENT
        elif _is_executable_file(registered_exec):
            state = IntegrationState.REGISTERED_OTHER_VALID
        else:
            state = IntegrationState.REGISTERED_BROKEN

    return IntegrationStatus(
        state=state,
        data_home=data,
        desktop_path=desktop_path,
        registered_exec=registered_exec,
        payload_sha256=payload_sha256,
        system_deb_present=deb_present,
        external_integrator_present=external_present,
        is_ephemeral_location=ephemeral,
        missing_tools=tools_missing,
        removable=(data / MANIFEST_FILENAME).is_file()
        or _legacy_desktop(context, desktop_path),
    )


@contextmanager
def _user_lock(data_home: Path) -> Iterator[None]:
    if not data_home.is_absolute() or any(
        data_home.resolve().is_relative_to(root)
        for root in (Path("/usr/share"), Path("/usr/local/share"))
    ):
        raise IntegrationError("Expected a user data directory.")
    data_home.mkdir(parents=True, exist_ok=True)
    lock_path = data_home / LOCK_FILENAME
    if lock_path.is_symlink():
        raise IntegrationError(f"Unexpected lock symlink: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(lock_path, flags, 0o600), "r+") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_sibling_temp(destination: Path, payload: bytes, prefix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=prefix,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _run_host_command(command: Sequence[str]) -> None:
    child_env = os.environ.copy()
    if getattr(sys, "frozen", False):
        appdir = child_env.get("APPDIR")
        child_env["LD_LIBRARY_PATH"] = child_env.pop("LD_LIBRARY_PATH_ORIG", "")
        for key in ("LD_LIBRARY_PATH", "PATH", "XDG_DATA_DIRS"):
            paths = [
                path for path in child_env.get(key, "").split(os.pathsep)
                if path and not (appdir and Path(path).is_relative_to(appdir))
            ]
            if paths:
                child_env[key] = os.pathsep.join(paths)
            else:
                child_env.pop(key, None)
    try:
        subprocess.run(
            list(command),
            env=child_env,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise IntegrationError(f"Required tool not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise IntegrationError(f"Command timed out: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise IntegrationError(
            f"{command[0]} failed (exit {exc.returncode}){suffix}"
        ) from exc


def _file_digest(path: Path) -> str | None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise IntegrationError(f"Unexpected integration target: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(data: Path) -> dict[str, str]:
    path = data / MANIFEST_FILENAME
    if path.is_symlink():
        raise IntegrationError(f"Unexpected manifest symlink: {path}")
    if not path.exists():
        return {}
    if not path.is_file():
        raise IntegrationError(f"Invalid integration manifest: {path}")
    try:
        record = json.loads(path.read_bytes())
        files = record["files"]
        if (
            record["version"] == 1 and isinstance(files, dict)
            and all(
                isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)
                for value in files.values()
            )
        ):
            return files
    except (ValueError, KeyError, TypeError):
        pass
    raise IntegrationError(f"Invalid integration manifest: {path}")


def _check_parents(data: Path, paths: Iterable[Path]) -> None:
    root = data.resolve()
    for path in paths:
        if path.parent.resolve() != root / path.parent.relative_to(data):
            raise IntegrationError(f"Unexpected integration directory: {path.parent}")


def _legacy_desktop(context: AppImageContext, path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError:
        return False
    entries = parse_desktop_entry(text)
    image = extract_registered_executable(entries)
    if image is None or not image.is_absolute():
        return False
    digest = entries.get(PAYLOAD_HASH_KEY, "")
    expected = render_desktop_entry(replace(context, image=image), digest)
    expected = expected.replace("Exec=" + EXEC_PREFIX, "Exec=")
    if digest and text == expected:
        return True
    legacy = [
        line for line in expected.splitlines()
        if not line.startswith(("TryExec=", PAYLOAD_HASH_KEY + "="))
    ]
    return text == "\n".join(legacy) + "\n"


def _commit_files(
    data: Path, changes: dict[Path, bytes | None], *, removing: bool = False
) -> list[str]:
    mime = data / "mime"
    theme = data / "icons/hicolor"
    applications = data / "applications"
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    attempted: list[list[str]] = []
    warnings: list[str] = []
    try:
        for dest, payload in changes.items():
            if payload is not None:
                staged[dest] = _write_sibling_temp(dest, payload, ".tractedit-stage-")
            if dest.exists():
                backups[dest] = _write_sibling_temp(dest, b"", ".tractedit-backup-")
                shutil.copy2(dest, backups[dest])
        for dest, payload in changes.items():
            if payload is None:
                dest.unlink()
            else:
                os.replace(staged[dest], dest)
            committed.append(dest)

        if mime.is_dir() and (not removing or shutil.which("update-mime-database")):
            attempted.append(["update-mime-database", str(mime)])
            _run_host_command(attempted[-1])
        if theme.is_dir():
            try:
                os.utime(theme)
            except OSError as exc:
                warnings.append(f"Icon theme refresh skipped: {exc}")
            if (theme / "icon-theme.cache").is_file() and shutil.which(
                "gtk-update-icon-cache"
            ):
                attempted.append(["gtk-update-icon-cache", "-f", "-t", str(theme)])
                try:
                    _run_host_command(attempted[-1])
                except IntegrationError as exc:
                    warnings.append(str(exc))
        if applications.is_dir() and (
            not removing or shutil.which("update-desktop-database")
        ):
            attempted.append(["update-desktop-database", str(applications)])
            _run_host_command(attempted[-1])
    except BaseException as exc:
        errors = []
        for dest in reversed(committed):
            backup = backups.pop(dest, None)
            try:
                if backup is None:
                    dest.unlink(missing_ok=True)
                else:
                    os.replace(backup, dest)
            except OSError as error:
                errors.append(f"Restore {dest} from {backup}: {error}")
        if committed and theme.is_dir():
            try:
                os.utime(theme)
            except OSError as error:
                errors.append(str(error))
        for command in attempted:
            try:
                _run_host_command(command)
            except (OSError, IntegrationError) as error:
                errors.append(str(error))
        if errors:
            raise IntegrationError(f"{exc}. Recovery: {'; '.join(errors)}") from exc
        raise
    finally:
        for temp_path in (*staged.values(), *backups.values()):
            temp_path.unlink(missing_ok=True)
    return warnings


def register_appimage(
    context: AppImageContext,
    data_home: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> OperationResult:
    """Atomically install or update this AppImage's Open With integration."""
    missing = missing_host_tools()
    if missing:
        tools = ", ".join(missing)
        raise IntegrationError(
            f"Missing required host tools: {tools}. "
            "Install shared-mime-info and desktop-file-utils."
        )

    data = resolve_xdg_data_home(env) if data_home is None else Path(data_home)
    files = _InstalledFiles(data)

    payload_sha256 = compute_payload_sha256(context)
    desktop_bytes = render_desktop_entry(context, payload_sha256).encode("utf-8")
    targets = {
        files.mime: context.mime_package.read_bytes(),
        files.icon: context.app_icon.read_bytes(),
        files.desktop: desktop_bytes,
    }
    for icon in context.mime_icons():
        targets[files.mime_icons / icon.name] = icon.read_bytes()

    with _user_lock(data):
        _check_parents(data, targets)
        manifest = _read_manifest(data)
        owned = dict(manifest)
        changes = {}
        for dest, payload in targets.items():
            key = dest.relative_to(data).as_posix()
            owned.pop(key, None)
            if dest.parent == files.mime_icons and dest.is_symlink():
                continue
            digest = _file_digest(dest)
            new_digest = hashlib.sha256(payload).hexdigest()
            if digest is not None and manifest.get(key) != digest:
                legacy = not manifest and dest == files.desktop and _legacy_desktop(
                    context, dest
                )
                if not legacy and (
                    digest == new_digest or dest.parent == files.mime_icons
                ):
                    continue
                if not legacy:
                    raise IntegrationError(f"Preserving existing file: {dest}")
            owned[key] = new_digest
            if digest != new_digest:
                changes[dest] = payload
        changes[data / MANIFEST_FILENAME] = json.dumps(
            {"version": 1, "files": owned}, sort_keys=True
        ).encode("utf-8")
        warnings = _commit_files(data, changes)

    try:
        status = inspect_integration_status(context, data, env=env)
    except (OSError, ValueError, RuntimeError) as exc:
        status = None
        warnings.append(f"Installed, but status could not be read: {exc}")
    return OperationResult(
        ok=True,
        message="TractEdit added to 'Open With'.",
        warnings=tuple(warnings),
        status=status,
    )


def unregister_appimage(
    data_home: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    mime_icon_names: Iterable[str] | None = None,
    context: AppImageContext | None = None,
) -> OperationResult:
    """Remove user-scope AppImage desktop, MIME and icon artifacts."""
    data = resolve_xdg_data_home(env) if data_home is None else Path(data_home)
    with _user_lock(data):
        manifest = _read_manifest(data)
        icon_names = {
            Path(key).name for key in manifest if Path(key).parent == _MIME_ICONS
        }
        if mime_icon_names is not None:
            icon_names.update(mime_icon_names)
        if any(Path(n).name != n or n in ("", ".", "..") for n in icon_names):
            raise IntegrationError("Invalid MIME icon name.")
        files = _InstalledFiles(data)
        candidates = list(files.paths(sorted(icon_names)))
        _check_parents(data, candidates)
        changes = {}
        warnings = []
        for path in candidates:
            key = path.relative_to(data).as_posix()
            if key in manifest:
                if path.is_symlink() or (path.exists() and not path.is_file()):
                    warnings.append(f"Preserved modified file: {path}")
                    continue
                digest = _file_digest(path)
                if digest == manifest[key]:
                    changes[path] = None
                elif digest is not None:
                    warnings.append(f"Preserved modified file: {path}")
            elif not manifest and path == files.desktop and context is not None:
                if _legacy_desktop(context, path):
                    changes[path] = None
        removed = bool(changes)
        if (data / MANIFEST_FILENAME).exists():
            changes[data / MANIFEST_FILENAME] = None
        if changes:
            warnings.extend(_commit_files(data, changes, removing=True))

    message = (
        "TractEdit removed from 'Open With'."
        if removed
        else "No user AppImage 'Open With' entry was installed."
    )
    return OperationResult(ok=True, message=message, warnings=tuple(warnings))


def register(image: Path, appdir: Path, data: Path) -> OperationResult:
    """Compatibility entry point used by CLI and desktop-association tests."""
    image_path = Path(image)
    appdir_path = Path(appdir)
    context = AppImageContext(
        image=image_path.resolve(),
        appdir=appdir_path,
        desktop_template=appdir_path / "tractedit.desktop",
        mime_package=appdir_path / "usr/share/mime/packages" / MIME_FILENAME,
        app_icon=appdir_path / "tractedit.png",
        mime_icons_dir=appdir_path / "usr/share/icons/hicolor/256x256/mimetypes",
    )
    return register_appimage(context, Path(data))


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Execute AppImage registration from AppRun or the bundled binary."""
    args = list(sys.argv[1:] if argv is None else argv)
    raw_image = os.environ.get("APPIMAGE", "")
    raw_appdir = args[0] if args else os.environ.get("APPDIR", "")
    if not raw_image or not raw_appdir:
        print(
            "Error: --register-file-types must be run from a TractEdit AppImage.",
            file=sys.stderr,
        )
        return 1
    try:
        context = detect_appimage_context(
            {"APPIMAGE": raw_image, "APPDIR": raw_appdir}
        )
        if context is None:
            raise IntegrationError("Invalid TractEdit AppImage or missing assets.")
        result = register_appimage(context)
    except (IntegrationError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
