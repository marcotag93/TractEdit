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
