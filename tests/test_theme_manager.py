"""Check that both effective palettes generate complete Qt stylesheet rules."""

import re

import pytest

from tractedit_pkg.ui.theme_manager import ThemeManager, ThemeMode


@pytest.mark.parametrize("mode", [ThemeMode.LIGHT, ThemeMode.DARK])
def test_generated_stylesheets_have_complete_rules(mode, monkeypatch):
    manager = ThemeManager.__new__(ThemeManager)
    manager._current_mode = mode
    dark = manager.is_dark_theme()
    monkeypatch.setattr(manager, "_system_is_dark", lambda: dark)
    styles = (
        manager._application_stylesheet(dark), manager._get_system_menu_stylesheet(),
        manager.tree_style(dark), manager.get_button_style_default(),
        manager.get_button_style_active(), manager.get_slider_style(),
        manager.get_spinbox_style(), manager.get_progress_bar_style(),
        manager.get_progress_dialog_style(), manager.initial_slider_style(),
        manager.initial_spinbox_style(), manager.brush_value_style(),
    )
    rule = r"[^{}]+\{(?:\s*[\w-]+:\s*[^{};]+;)+\s*\}"
    for stylesheet in styles:
        assert re.fullmatch(rf"(?:{rule}\s*)+", stylesheet), stylesheet


@pytest.mark.parametrize("is_dark", [False, True])
def test_macos_stylesheet_covers_toolbar_and_preserves_dock_buttons(is_dark):
    rule = r"[^{}]+\{(?:\s*[\w-]+:\s*[^{};]+;)+\s*\}"
    mac_style = ThemeManager._application_stylesheet(is_dark, is_macos=True)
    other_style = ThemeManager._application_stylesheet(is_dark, is_macos=False)
    assert re.fullmatch(rf"(?:{rule}\s*)+", mac_style), mac_style
    assert "QToolBar {" in mac_style and "QToolBar QLabel {" in mac_style
    assert "QStatusBar {" in mac_style and "titlebar-close-icon: none" not in mac_style
    assert "QToolBar {" not in other_style
    if is_dark:
        assert "titlebar-close-icon: none" in other_style
