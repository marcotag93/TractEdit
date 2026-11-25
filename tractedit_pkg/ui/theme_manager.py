# -*- coding: utf-8 -*-

"""
Theme manager for TractEdit UI.

Handles application-wide theme switching between Light, Dark, and System modes.
Provides centralized style definitions for consistent theming across all components.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import QSettings

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# Theme Mode Enum
# ============================================================================


class ThemeMode(Enum):
    """Enumeration of available theme modes."""

    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


# ============================================================================
# Theme Manager Class
# ============================================================================


_THEME_VALUES = {
    "palette_text": ("#dcdcdc", "#1e1e1e"),
    "palette_base": ("#232323", "#ffffff"),
    "bright": ("#ffffff", "#ff0000"),
    "window": ("#2d2d2d", "#f0f0f0"),
    "text": ("#dddddd", "#1e1e1e"),
    "menubar_item_rule": (
        "",
        "QMenuBar::item { background-color: transparent; color: #1e1e1e; }",
    ),
    "raised": ("#3d3d3d", "#e0e0e0"),
    "menubar_selected_text": ("", "color: #1e1e1e;"),
    "base": ("#2d2d2d", "#ffffff"),
    "border": ("#555555", "#c0c0c0"),
    "menu_item_rule": (
        "",
        "QMenu::item { color: #1e1e1e; padding: 4px 20px 4px 20px; }",
    ),
    "highlight": ("#2a82da", "#0078d7"),
    "menu_selected_text": ("", "color: #ffffff;"),
    "menu_disabled_rules": (
        "",
        (
            "QMenu::item:disabled { color: #a0a0a0; } QMenu::separator { height: 1px; "
            "background-color: #d0d0d0; margin: 4px 10px; }"
        ),
    ),
    "scrollbar_hover": ("#666666", "#a0a0a0"),
    "dock_icons_rule": (
        "QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; }",
        "",
    ),
    "tree_colors": ("background-color: #2d2d2d; color: #dddddd;", ""),
    "selection": ("#2a82da", "#4a6984"),
    "tree_background_color": ("rgba(255, 255, 255, 0.05)", "rgba(0, 0, 0, 0.03)"),
    "selection_hover": ("#3a92ea", "#557ba0"),
    "button": ("#3d3d3d", "#f0f0f0"),
    "button_hover": ("#4d4d4d", "#e8e8e8"),
    "button_hover_border": ("#666666", "#b0b0b0"),
    "button_checked": ("#2d2d2d", "#d0d0d0"),
    "button_checked_border": ("#444444", "#a0a0a0"),
    "groove_border": ("#555555", "#bbb"),
    "groove": ("#2d2d2d", "#e0e0e0"),
    "label": ("#dddddd", "#333333"),
    "spinbox_button_rules": (
        (
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { "
            "background-color: #3d3d3d; border: 1px solid #555555; } "
            "QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { "
            "background-color: #4d4d4d; }"
        ),
        "",
    ),
    "progress_border": ("#555", "#c0c0c0"),
    "progress_background": ("#333", "#f0f0f0"),
    "progress_text": ("white", "#333333"),
    "dialog": ("#2b2b2b", "#f5f5f5"),
    "dialog_track": ("#333", "#e0e0e0"),
    "dialog_button": ("#444", "#e0e0e0"),
    "dialog_button_text": ("#ddd", "#333333"),
    "dialog_button_hover": ("#555", "#d0d0d0"),
}
_DARK_TOKENS = {name: dark for name, (dark, _) in _THEME_VALUES.items()}
_LIGHT_TOKENS = {name: light for name, (_, light) in _THEME_VALUES.items()}


def _tokens(is_dark):
    return _DARK_TOKENS if is_dark else _LIGHT_TOKENS


_MENU_QSS = """
    QMenuBar {{ background-color: {window}; color: {text}; }}
    {menubar_item_rule}
    QMenuBar::item:selected {{ background-color: {raised}; {menubar_selected_text} }}
    QMenu {{ background-color: {base}; color: {text}; border: 1px solid {border}; }}
    {menu_item_rule}
    QMenu::item:selected {{ background-color: {highlight}; {menu_selected_text} }}
    {menu_disabled_rules}
"""

_TOOLTIP_QSS = "QToolTip {{ color: {text}; background-color: {base}; border: 1px solid {border}; }}"

_WIDGETS_QSS = """
    QScrollBar:vertical {{ background: {window}; width: 12px; }}
    QScrollBar::handle:vertical {{ background: {border}; min-height: 20px; border-radius: 4px; }}
    QScrollBar::handle:vertical:hover {{ background: {scrollbar_hover}; }}
    QScrollBar:horizontal {{ background: {window}; height: 12px; }}
    QScrollBar::handle:horizontal {{ background: {border}; min-width: 20px; border-radius: 4px; }}
    QScrollBar::handle:horizontal:hover {{ background: {scrollbar_hover}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ background: none; border: none; }}
    {dock_icons_rule}
    QDockWidget::title {{ background-color: {raised}; padding: 4px; }}
"""

_TREE_QSS = """
    QTreeWidget {{ outline: 0; {tree_colors} }}
    QTreeWidget::item {{ padding: 4px; border-radius: 4px; }}
    QTreeWidget::item:selected {{ background-color: {selection}; color: white; border: none; }}
    QTreeWidget::item:selected:!active {{ background-color: {selection}; color: white; border: none; }}
    QTreeWidget::item:hover {{ background-color: {tree_background_color}; border: none; }}
    QTreeWidget::item:selected:hover {{ background-color: {selection_hover}; color: white; }}
"""

_BUTTON_QSS = """
    QToolButton {{ background-color: {button}; border: 1px solid {border}; border-radius: 5px; padding: 2px; margin-left: 5px; }}
    QToolButton:hover {{ background-color: {button_hover}; border: 1px solid {button_hover_border}; }}
    QToolButton:checked {{ background-color: {button_checked}; border: 1px inset {button_checked_border}; padding: 3px 1px 1px 3px; }}
"""

_SLIDER_QSS = """
    QSlider::groove:horizontal {{ border: 1px solid {groove_border}; height: 6px; background: {groove}; margin: 0px; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: #05B8CC; border: 1px solid #049dad; width: 14px; margin: -5px 0; border-radius: 7px; }}
    QSlider::handle:horizontal:hover {{ background: #06d4eb; }}
"""

_SPINBOX_QSS = """
    QDoubleSpinBox {{ background-color: {base}; color: {label}; border: 1px solid {border}; border-radius: 3px; padding: 2px; }}
    {spinbox_button_rules}
"""

_BAR_QSS = """
    QProgressBar {{ border: 1px solid {progress_border}; border-radius: 4px; background-color: {progress_background}; color: {progress_text}; text-align: center; font-size: 10px; }}
    QProgressBar::chunk {{ background-color: #05B8CC; }}
"""

_DIALOG_QSS = """
    QProgressDialog {{ background-color: {dialog}; color: {label}; }}
    QLabel {{ color: {label}; font-size: 12px; font-weight: bold; margin-bottom: 5px; }}
    QProgressBar {{ border: 1px solid {progress_border}; border-radius: 4px; background-color: {dialog_track}; color: {progress_text}; text-align: center; font-size: 12px; height: 25px; }}
    QProgressBar::chunk {{ background-color: #05B8CC; border-radius: 3px; }}
    QPushButton {{ background-color: {dialog_button}; color: {dialog_button_text}; border: 1px solid {progress_border}; border-radius: 4px; padding: 4px 12px; }}
    QPushButton:hover {{ background-color: {dialog_button_hover}; }}
"""


class ThemeManager:
    """
    Manages application-wide theming for TractEdit.

    This class handles:
    - Switching between Light, Dark, and System themes
    - Providing consistent style definitions for all UI components
    - Persisting theme preferences across sessions
    """

    # Settings key for persistence
    SETTINGS_KEY = "appearance/theme"

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the theme manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window
        self._current_mode = ThemeMode.SYSTEM
        self._settings = QSettings("TractEdit", "TractEdit")

        # Load saved preference
        saved_theme = self._settings.value(self.SETTINGS_KEY, ThemeMode.DARK.value)
        try:
            self._current_mode = ThemeMode(saved_theme)
        except ValueError:
            self._current_mode = ThemeMode.SYSTEM

    @staticmethod
    def _calculate_luminance(color: QColor) -> float:
        """
        Calculates the perceived luminance of a color using ITU-R BT.601 weights.

        Args:
            color: The QColor to calculate luminance for.

        Returns:
            Luminance value between 0 (black) and 255 (white).
        """
        return 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()

    @property
    def current_mode(self) -> ThemeMode:
        """Returns the current theme mode."""
        return self._current_mode

    def is_dark_theme(self) -> bool:
        """
        Determines if the current effective theme is dark.

        Returns:
            True if dark theme is active, False otherwise.
        """
        if self._current_mode == ThemeMode.DARK:
            return True
        elif self._current_mode == ThemeMode.LIGHT:
            return False
        return self._system_is_dark()

    def _system_is_dark(self) -> bool:
        app = QApplication.instance()
        if app:
            color = app.palette().color(QPalette.ColorRole.Window)
            return self._calculate_luminance(color) < 128
        return False

    def set_theme(self, mode: ThemeMode) -> None:
        """
        Sets the application theme.

        Args:
            mode: The theme mode to apply.
        """
        self._current_mode = mode
        self._settings.setValue(self.SETTINGS_KEY, mode.value)
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Applies the current theme to the application."""
        app = QApplication.instance()
        if not app:
            return

        if self._current_mode == ThemeMode.SYSTEM:
            # Reset to system default palette
            app.setPalette(app.style().standardPalette())
            # On Windows, menus don't properly inherit palette colors,
            # so we need to apply minimal menu styling based on detected theme
            app.setStyleSheet(self._get_system_menu_stylesheet())
        elif self._current_mode == ThemeMode.DARK:
            self._apply_dark_palette(app)
        else:
            self._apply_light_palette(app)

        # Update all themed components
        self._update_themed_components()

    def _apply_dark_palette(self, app: QApplication) -> None:
        self._apply_palette(app, True)

    def _apply_light_palette(self, app: QApplication) -> None:
        self._apply_palette(app, False)

    def _apply_palette(self, app, is_dark):
        tokens = _tokens(is_dark)
        palette = QPalette()
        role = QPalette.ColorRole
        colors = {
            role.Window: tokens["window"], role.WindowText: tokens["palette_text"],
            role.Base: tokens["palette_base"], role.AlternateBase: tokens["window"],
            role.ToolTipBase: tokens["base"], role.ToolTipText: tokens["palette_text"],
            role.Text: tokens["palette_text"], role.Button: tokens["window"],
            role.ButtonText: tokens["palette_text"], role.BrightText: tokens["bright"],
            role.Link: tokens["highlight"], role.Highlight: tokens["highlight"],
            role.HighlightedText: "#ffffff",
        }
        for color_role, color in colors.items():
            palette.setColor(color_role, QColor(color))
        for color_role in (role.WindowText, role.Text, role.ButtonText, role.HighlightedText):
            palette.setColor(QPalette.ColorGroup.Disabled, color_role, QColor("#7f7f7f"))
        app.setPalette(palette)
        app.setStyleSheet(self._application_stylesheet(is_dark))

    @staticmethod
    def _application_stylesheet(is_dark: bool) -> str:
        return (_TOOLTIP_QSS + _MENU_QSS + _WIDGETS_QSS).format(**_tokens(is_dark))

    def _get_system_menu_stylesheet(self) -> str:
        return _MENU_QSS.format(**_tokens(self._system_is_dark()))

    def _update_themed_components(self) -> None:
        """Updates all components that need theme-specific styling."""
        mw = self.mw

        # Update drawing mode button styles
        if hasattr(mw, "drawing_modes_manager"):
            self._update_drawing_mode_buttons()

        # Update toolbar component styles
        self._update_toolbar_styles()

        # Update data panel styles
        self._update_data_panel_styles()

        # Update VTK panel overlay styles
        if mw.vtk_panel:
            self._update_vtk_panel_styles()

    def _update_drawing_mode_buttons(self) -> None:
        """Updates drawing mode button styles and icons based on current theme."""
        mw = self.mw

        toolbar = getattr(mw, "toolbars_manager", None)
        for mode, icon, flag in (
            ("draw", "_create_pencil_icon", "is_drawing_mode"),
            ("erase", "_create_eraser_icon", "is_eraser_mode"),
            ("sphere", "_create_sphere_icon", "is_sphere_mode"),
            ("rectangle", "_create_rectangle_icon", "is_rectangle_mode"),
        ):
            button = getattr(mw, f"{mode}_mode_button", None)
            if button is None:
                continue
            action = getattr(mw, f"{mode}_mode_action", None)
            if toolbar is not None and action is not None:
                action.setIcon(getattr(toolbar, icon)())
            if getattr(mw, flag, False):
                button.setStyleSheet(self.get_button_style_active())
            else:
                button.setStyleSheet(self.get_button_style_default())

    def _update_toolbar_styles(self) -> None:
        """Updates toolbar component styles based on current theme."""
        mw = self.mw
        is_dark = self.is_dark_theme()

        label_color = _tokens(is_dark)["label"]

        # Update "Brush:" text label
        if hasattr(mw, "brush_label"):
            mw.brush_label.setStyleSheet(f"font-weight: bold; color: {label_color};")

        # Update brush size value label (keeps cyan color)
        if hasattr(mw, "brush_size_label"):
            mw.brush_size_label.setStyleSheet(
                "font-weight: bold; color: #00BCD4; font-size: 13px; min-width: 15px;"
            )

        # Update sphere radius label
        if hasattr(mw, "sphere_radius_container"):
            for child in mw.sphere_radius_container.children():
                if hasattr(child, "setStyleSheet") and hasattr(child, "text"):
                    if callable(getattr(child, "text", None)):
                        try:
                            if child.text() == "Radius:":
                                child.setStyleSheet(
                                    f"font-weight: bold; color: {label_color};"
                                )
                        except (AttributeError, RuntimeError):
                            logger.debug("Failed to update sphere radius label style.")

        # Update slider style
        if hasattr(mw, "brush_size_slider"):
            mw.brush_size_slider.setStyleSheet(self.get_slider_style())

        # Update spinbox style
        if hasattr(mw, "sphere_radius_spinbox"):
            mw.sphere_radius_spinbox.setStyleSheet(self.get_spinbox_style())

    def _update_data_panel_styles(self) -> None:
        """Updates data panel styles based on current theme."""
        mw = self.mw
        if hasattr(mw, "data_tree_widget"):
            mw.data_tree_widget.setStyleSheet(self.tree_style(self.is_dark_theme()))

    @staticmethod
    def tree_style(is_dark: bool) -> str:
        return _TREE_QSS.format(**_tokens(is_dark))

    def _update_vtk_panel_styles(self) -> None:
        """Updates VTK panel overlay styles based on current theme."""
        mw = self.mw
        if not mw.vtk_panel:
            return

        if hasattr(mw.vtk_panel, "overlay_progress_bar"):
            mw.vtk_panel.overlay_progress_bar.setStyleSheet(
                self.get_progress_bar_style()
            )

    def get_button_style_default(self) -> str:
        """Returns the default button style for the current theme."""
        return self.button_style(self.is_dark_theme())

    @staticmethod
    def button_style(is_dark: bool) -> str:
        return _BUTTON_QSS.format(**_tokens(is_dark))

    def get_button_style_active(self) -> str:
        """Returns the active button style for the current theme."""
        # Active style is the same for both themes (cyan accent)
        return """
            QToolButton {
                background-color: rgb(0, 188, 212);
                border: 1px solid #00ACC1;
                border-radius: 5px;
                padding: 2px;
                margin-left: 5px;
            }
            QToolButton:hover {
                background-color: rgb(0, 172, 193);
                border: 1px solid #0097A7;
            }
            QToolButton:checked {
                background-color: rgb(0, 151, 167);
                border: 1px inset #00838F;
                padding: 3px 1px 1px 3px;
            }
        """

    def get_slider_style(self) -> str:
        """Returns the slider style for the current theme."""
        return _SLIDER_QSS.format(**_tokens(self.is_dark_theme()))

    def get_spinbox_style(self) -> str:
        """Returns the spinbox style for the current theme."""
        return _SPINBOX_QSS.format(**_tokens(self.is_dark_theme()))

    def get_progress_bar_style(self) -> str:
        """Returns the progress bar style for the current theme."""
        return self.bar_style(self.is_dark_theme())

    @staticmethod
    def bar_style(is_dark: bool) -> str:
        return _BAR_QSS.format(**_tokens(is_dark))

    def get_progress_dialog_style(self) -> str:
        """Returns the progress dialog style for the current theme."""
        return self.dialog_style(self.is_dark_theme())

    @staticmethod
    def dialog_style(is_dark: bool) -> str:
        return _DIALOG_QSS.format(**_tokens(is_dark))

    @staticmethod
    def initial_slider_style() -> str:
        return """
            QSlider::groove:horizontal { border: 1px solid #c0c0c0; height: 6px; background: #e0e0e0; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00BCD4; border: 1px solid #00ACC1; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }
            QSlider::handle:horizontal:hover { background: #00ACC1; }
        """

    @staticmethod
    def initial_spinbox_style() -> str:
        return """
            QDoubleSpinBox { border: 1px solid #00BCD4; border-radius: 4px; padding: 2px 5px; background: white; font-weight: bold; color: #00838F; }
            QDoubleSpinBox:focus { border: 2px solid #00ACC1; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 18px; border: none; background: #e0f7fa; }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #b2ebf2; }
        """

    @staticmethod
    def brush_value_style() -> str:
        return """
            QLabel { font-weight: bold; color: #00BCD4; font-size: 13px; min-width: 15px; }
        """

    def initialize_theme(self) -> None:
        """Initializes the theme on application startup."""
        self._apply_theme()
