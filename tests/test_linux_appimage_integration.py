"""Unit tests for Linux AppImage desktop and MIME integration."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tractedit_pkg import linux_appimage_integration as integration


ASSETS_DIR = Path(__file__).resolve().parents[1] / ".github/scripts/linux"
LOGO_PATH = Path(__file__).resolve().parents[1] / "tractedit_pkg/assets/tractedit.png"


def _build_appdir(root: Path) -> tuple[Path, Path]:
    appdir = root / "appdir"
    packages = appdir / "usr/share/mime/packages"
    mime_icons = appdir / "usr/share/icons/hicolor/256x256/mimetypes"
    packages.mkdir(parents=True)
    mime_icons.mkdir(parents=True)

    desktop_bytes = (ASSETS_DIR / "tractedit.desktop").read_bytes()
    mime_bytes = (ASSETS_DIR / "tractedit-mime.xml").read_bytes()
    logo_bytes = LOGO_PATH.read_bytes()

    (appdir / "tractedit.desktop").write_bytes(desktop_bytes)
    (packages / "tractedit-mime.xml").write_bytes(mime_bytes)
    (appdir / "tractedit.png").write_bytes(logo_bytes)
    for icon_name in integration.DEFAULT_MIME_ICON_NAMES:
        (mime_icons / icon_name).write_bytes(logo_bytes)

    image = root / "TractEdit 100% $test.AppImage"
    image.write_bytes(b"#!/bin/sh\n")
    image.chmod(0o755)
    return image, appdir


def _fake_host_updates(data_home: Path):
    def fake_run(command, **kwargs):
        tool = command[0]
        target = Path(command[-1])
        if tool == "update-mime-database":
            target.mkdir(parents=True, exist_ok=True)
            (target / "mime.cache").write_bytes(b"cache")
        elif tool == "update-desktop-database":
            desktop = target / integration.DESKTOP_FILENAME
            content = (
                f"[MIME Cache]\napplication/x-nifti={integration.DESKTOP_FILENAME};\n"
                if desktop.is_file()
                else "[MIME Cache]\n"
            )
            (target / "mimeinfo.cache").write_text(content, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    return fake_run


def test_xdg_data_home_ignores_relative_paths(tmp_path):
    assert integration.resolve_xdg_data_home({"XDG_DATA_HOME": "relative/path"}) == (
        Path.home() / ".local/share"
    )
    assert (
        integration.resolve_xdg_data_home({"XDG_DATA_HOME": str(tmp_path)})
        == tmp_path
    )


def test_detect_appimage_context_validates_platform_and_assets(tmp_path):
    image, appdir = _build_appdir(tmp_path)
    env = {"APPIMAGE": str(image), "APPDIR": str(appdir)}

    assert integration.detect_appimage_context(env, platform="win32") is None
    assert integration.detect_appimage_context(env, platform="darwin") is None

    ctx = integration.detect_appimage_context(env, platform="linux")
    assert ctx is not None
    assert ctx.image == image.resolve()
    assert ctx.appdir == appdir

    (appdir / "tractedit.desktop").unlink()
    assert integration.detect_appimage_context(env, platform="linux") is None


def test_escaping_and_roundtrip_extraction(tmp_path):
    image, appdir = _build_appdir(tmp_path)
    ctx = integration.detect_appimage_context(
        {"APPIMAGE": str(image), "APPDIR": str(appdir)}, platform="linux"
    )
    assert ctx is not None
    digest = integration.compute_payload_sha256(ctx)
    rendered = integration.render_desktop_entry(ctx, digest)

    entries = integration.parse_desktop_entry(rendered)
    assert entries[integration.PAYLOAD_HASH_KEY] == digest
    assert "TryExec=" in rendered
    assert integration.extract_registered_executable(entries) == image.resolve()

    legacy_entries = {k: v for k, v in entries.items() if k != "TryExec"}
    assert integration.extract_registered_executable(legacy_entries) == image.resolve()


@pytest.mark.parametrize("suffix", ['"', "$", "`", "\\", "\n", "\t", " ", "%f", "è"])
def test_desktop_paths_preserve_literal_characters(suffix):
    path = "/home/user/TractEdit " + suffix
    argument = integration.escape_exec_arg(path)
    entry = f'[Desktop Entry]\nExec="{argument}" %f\n'
    assert integration.extract_registered_executable(
        integration.parse_desktop_entry(entry)
    ) == Path(path)
    entry += "TryExec=" + integration.escape_desktop_string(path) + "\n"
    assert integration.extract_registered_executable(
        integration.parse_desktop_entry(entry)
    ) == Path(path)


def test_preflight_missing_tools_leaves_filesystem_untouched(tmp_path, monkeypatch):
    image, appdir = _build_appdir(tmp_path)
    ctx = integration.detect_appimage_context(
        {"APPIMAGE": str(image), "APPDIR": str(appdir)}, platform="linux"
    )
    assert ctx is not None
    data = tmp_path / "xdg-data"
    monkeypatch.setattr(integration.shutil, "which", lambda name: None)

    with pytest.raises(
        integration.IntegrationError, match="Missing required host tools"
    ):
        integration.register_appimage(ctx, data)

    assert not data.exists()


def test_uninstall_preserves_user_icons(tmp_path, monkeypatch):
    image, appdir = _build_appdir(tmp_path)
    data = tmp_path / "data"
    icons = data / "icons/hicolor/256x256/mimetypes"
    icons.mkdir(parents=True)
    foreign = icons / "application-x-trk.png"
    foreign.write_bytes(b"user icon")
    monkeypatch.setattr(integration.shutil, "which", lambda name: name)
    monkeypatch.setattr(integration.subprocess, "run", _fake_host_updates(data))

    integration.register(image, appdir, data)
    modified = icons / "application-x-nifti.png"
    modified.write_bytes(b"replacement")
    integration.unregister_appimage(data)

    assert foreign.read_bytes() == b"user icon"
    assert modified.read_bytes() == b"replacement"
    assert set(icons.iterdir()) == {foreign, modified}


def test_register_status_transitions_and_unregister(tmp_path, monkeypatch):
    image, appdir = _build_appdir(tmp_path)
    ctx = integration.detect_appimage_context(
        {"APPIMAGE": str(image), "APPDIR": str(appdir)}, platform="linux"
    )
    assert ctx is not None
    data = tmp_path / "xdg-data"
    monkeypatch.setattr(integration.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(integration.subprocess, "run", _fake_host_updates(data))

    initial = integration.inspect_integration_status(
        ctx, data, system_desktop=tmp_path / "missing.desktop"
    )
    assert initial.state == integration.IntegrationState.UNREGISTERED

    result = integration.register_appimage(ctx, data)
    assert result.ok
    assert result.status is not None
    assert result.status.state == integration.IntegrationState.REGISTERED_CURRENT

    # Updating MIME XML at the same path marks the installation outdated.
    (appdir / "usr/share/mime/packages/tractedit-mime.xml").write_text(
        (appdir / "usr/share/mime/packages/tractedit-mime.xml").read_text(
            encoding="utf-8"
        )
        + "\n",
        encoding="utf-8",
    )
    outdated = integration.inspect_integration_status(
        ctx, data, system_desktop=tmp_path / "missing.desktop"
    )
    assert outdated.state == integration.IntegrationState.REGISTERED_OUTDATED

    # Second AppImage sees REGISTERED_OTHER_VALID, then REGISTERED_BROKEN if removed.
    second = tmp_path / "TractEdit-next.AppImage"
    second.write_bytes(b"#!/bin/sh\n")
    second.chmod(0o755)
    second_ctx = integration.detect_appimage_context(
        {"APPIMAGE": str(second), "APPDIR": str(appdir)}, platform="linux"
    )
    assert second_ctx is not None
    other = integration.inspect_integration_status(
        second_ctx, data, system_desktop=tmp_path / "missing.desktop"
    )
    assert other.state == integration.IntegrationState.REGISTERED_OTHER_VALID

    image.unlink()
    broken = integration.inspect_integration_status(
        second_ctx, data, system_desktop=tmp_path / "missing.desktop"
    )
    assert broken.state == integration.IntegrationState.REGISTERED_BROKEN

    removed = integration.unregister_appimage(data)
    assert removed.ok
    after = integration.inspect_integration_status(
        second_ctx, data, system_desktop=tmp_path / "missing.desktop"
    )
    assert after.state == integration.IntegrationState.UNREGISTERED


def test_rollback_restores_previous_state_on_desktop_db_failure(tmp_path, monkeypatch):
    image, appdir = _build_appdir(tmp_path)
    ctx = integration.detect_appimage_context(
        {"APPIMAGE": str(image), "APPDIR": str(appdir)}, platform="linux"
    )
    assert ctx is not None
    data = tmp_path / "xdg-data"
    monkeypatch.setattr(integration.shutil, "which", lambda name: f"/usr/bin/{name}")

    calls: list[str] = []

    def failing_run(command, **kwargs):
        calls.append(command[0])
        if command[0] == "update-desktop-database":
            raise subprocess.CalledProcessError(1, command, stderr="cache error")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(integration.subprocess, "run", failing_run)

    with pytest.raises(integration.IntegrationError, match="update-desktop-database"):
        integration.register_appimage(ctx, data)

    assert calls == [
        "update-mime-database",
        "update-desktop-database",
        "update-mime-database",
        "update-desktop-database",
    ]
    assert not (data / "applications" / integration.DESKTOP_FILENAME).exists()
    assert not (data / "mime/packages" / integration.MIME_FILENAME).exists()


def test_gtk_icon_cache_failure_is_non_fatal(tmp_path, monkeypatch):
    image, appdir = _build_appdir(tmp_path)
    ctx = integration.detect_appimage_context(
        {"APPIMAGE": str(image), "APPDIR": str(appdir)}, platform="linux"
    )
    assert ctx is not None
    data = tmp_path / "xdg-data"
    theme = data / "icons/hicolor"
    theme.mkdir(parents=True)
    (theme / "icon-theme.cache").write_bytes(b"cache")
    monkeypatch.setattr(integration.shutil, "which", lambda name: f"/usr/bin/{name}")

    base_run = _fake_host_updates(data)

    def run_with_icon_failure(command, **kwargs):
        if command[0] == "gtk-update-icon-cache":
            raise subprocess.CalledProcessError(1, command, stderr="icon fail")
        return base_run(command, **kwargs)

    monkeypatch.setattr(integration.subprocess, "run", run_with_icon_failure)

    result = integration.register_appimage(ctx, data)
    assert result.ok
    assert result.warnings
    assert (data / "applications" / integration.DESKTOP_FILENAME).is_file()
