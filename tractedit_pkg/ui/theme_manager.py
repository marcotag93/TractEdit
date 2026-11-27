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

import ctypes
import logging
import sys
import weakref
from enum import Enum
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import QSettings, Qt

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)

_OBJC_LIB = None
_MSG_PTR_0 = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
_MSG_PTR_1_PTR = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
)
_MSG_PTR_1_CSTR = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
)
_MSG_PTR_1_SIZE = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
)
_MSG_VOID_0 = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
_MSG_VOID_1_PTR = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
)
_MSG_SIZE_0 = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p)
_MSG_CSTR_0 = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)


def _get_objc_lib():
    global _OBJC_LIB
    if _OBJC_LIB is None:
        lib = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        _OBJC_LIB = lib
    return _OBJC_LIB


def _is_macos_cocoa(app: Optional[QApplication]) -> bool:
    if sys.platform != "darwin" or app is None:
        return False
    platform_name = getattr(app, "platformName", None)
    return callable(platform_name) and platform_name() == "cocoa"


def _set_macos_appearance(app: Optional[QApplication], mode: "ThemeMode") -> bool:
    """Sync NSApp and open NSWindow appearances with the selected theme."""
    if not _is_macos_cocoa(app):
        return False
    pool = None
    try:
        lib = _get_objc_lib()
        msg_ptr_0 = ctypes.cast(lib.objc_msgSend, _MSG_PTR_0)
        msg_ptr_1_ptr = ctypes.cast(lib.objc_msgSend, _MSG_PTR_1_PTR)
        msg_ptr_1_cstr = ctypes.cast(lib.objc_msgSend, _MSG_PTR_1_CSTR)
        msg_ptr_1_size = ctypes.cast(lib.objc_msgSend, _MSG_PTR_1_SIZE)
        msg_void_0 = ctypes.cast(lib.objc_msgSend, _MSG_VOID_0)
        msg_void_1_ptr = ctypes.cast(lib.objc_msgSend, _MSG_VOID_1_PTR)
        msg_size_0 = ctypes.cast(lib.objc_msgSend, _MSG_SIZE_0)

        pool = msg_ptr_0(
            lib.objc_getClass(b"NSAutoreleasePool"),
            lib.sel_registerName(b"new"),
        )
        ns_app = msg_ptr_0(
            lib.objc_getClass(b"NSApplication"),
            lib.sel_registerName(b"sharedApplication"),
        )
        if not ns_app:
            return False

        if mode == ThemeMode.SYSTEM:
            appearance = None
        else:
            name = (
                b"NSAppearanceNameDarkAqua"
                if mode == ThemeMode.DARK
                else b"NSAppearanceNameAqua"
            )
            ns_name = msg_ptr_1_cstr(
                lib.objc_getClass(b"NSString"),
                lib.sel_registerName(b"stringWithUTF8String:"),
                name,
            )
            if not ns_name:
                return False
            appearance = msg_ptr_1_ptr(
                lib.objc_getClass(b"NSAppearance"),
                lib.sel_registerName(b"appearanceNamed:"),
                ns_name,
            )
            if not appearance:
                return False

        sel_set_appearance = lib.sel_registerName(b"setAppearance:")
        msg_void_1_ptr(ns_app, sel_set_appearance, appearance)
        windows = msg_ptr_0(ns_app, lib.sel_registerName(b"windows"))
        if windows:
            count = msg_size_0(windows, lib.sel_registerName(b"count"))
            sel_object_at = lib.sel_registerName(b"objectAtIndex:")
            for index in range(count):
                window = msg_ptr_1_size(windows, sel_object_at, index)
                if window:
                    msg_void_1_ptr(window, sel_set_appearance, appearance)
        return True
    except Exception as exc:
        logger.debug("Failed to update macOS appearance: %s", exc)
        return False
    finally:
        if pool:
            try:
                lib = _get_objc_lib()
                ctypes.cast(lib.objc_msgSend, _MSG_VOID_0)(
                    pool, lib.sel_registerName(b"drain")
                )
            except Exception:
                pass


def _macos_system_is_dark(app: Optional[QApplication]) -> Optional[bool]:
    """Read the live macOS system appearance polarity via AppKit."""
    if not _is_macos_cocoa(app):
        return None
    pool = None
    try:
        lib = _get_objc_lib()
        msg_ptr_0 = ctypes.cast(lib.objc_msgSend, _MSG_PTR_0)
        msg_ptr_1_ptr = ctypes.cast(lib.objc_msgSend, _MSG_PTR_1_PTR)
        msg_ptr_1_cstr = ctypes.cast(lib.objc_msgSend, _MSG_PTR_1_CSTR)
        msg_cstr_0 = ctypes.cast(lib.objc_msgSend, _MSG_CSTR_0)

        pool = msg_ptr_0(
            lib.objc_getClass(b"NSAutoreleasePool"),
            lib.sel_registerName(b"new"),
        )
        ns_app = msg_ptr_0(
            lib.objc_getClass(b"NSApplication"),
            lib.sel_registerName(b"sharedApplication"),
        )
        if not ns_app:
            return None

        sel_utf8 = lib.sel_registerName(b"UTF8String")
        explicit = msg_ptr_0(ns_app, lib.sel_registerName(b"appearance"))
        if not explicit:
            effective = msg_ptr_0(
                ns_app, lib.sel_registerName(b"effectiveAppearance")
            )
            if not effective:
                return None
            name_obj = msg_ptr_0(effective, lib.sel_registerName(b"name"))
            raw = msg_cstr_0(name_obj, sel_utf8) if name_obj else None
            return b"Dark" in raw if raw else None

        defaults = msg_ptr_0(
            lib.objc_getClass(b"NSUserDefaults"),
            lib.sel_registerName(b"standardUserDefaults"),
        )
        if not defaults:
            return None
        key = msg_ptr_1_cstr(
            lib.objc_getClass(b"NSString"),
            lib.sel_registerName(b"stringWithUTF8String:"),
            b"AppleInterfaceStyle",
        )
        style = msg_ptr_1_ptr(
            defaults, lib.sel_registerName(b"stringForKey:"), key
        )
        if not style:
            return False
        raw = msg_cstr_0(style, sel_utf8)
        return bool(raw and b"Dark" in raw)
    except Exception as exc:
        logger.debug("Failed to read macOS system appearance: %s", exc)
        return None
    finally:
        if pool:
            try:
                lib = _get_objc_lib()
                ctypes.cast(lib.objc_msgSend, _MSG_VOID_0)(
                    pool, lib.sel_registerName(b"drain")
                )
            except Exception:
                pass


def get_macos_appearance_state(
    app: Optional[QApplication] = None,
) -> Optional[dict[str, Optional[str]]]:
    """Return explicit and effective NSApp appearance names on macOS Cocoa."""
    app = app if app is not None else QApplication.instance()
    if not _is_macos_cocoa(app):
        return None
    pool = None
    try:
        lib = _get_objc_lib()
        msg_ptr_0 = ctypes.cast(lib.objc_msgSend, _MSG_PTR_0)
        msg_cstr_0 = ctypes.cast(lib.objc_msgSend, _MSG_CSTR_0)
        pool = msg_ptr_0(
            lib.objc_getClass(b"NSAutoreleasePool"),
            lib.sel_registerName(b"new"),
        )
        ns_app = msg_ptr_0(
            lib.objc_getClass(b"NSApplication"),
            lib.sel_registerName(b"sharedApplication"),
        )
        if not ns_app:
            return None
        sel_name = lib.sel_registerName(b"name")
        sel_utf8 = lib.sel_registerName(b"UTF8String")

        def _name(appearance_ptr):
            if not appearance_ptr:
                return None
            name_obj = msg_ptr_0(appearance_ptr, sel_name)
            raw = msg_cstr_0(name_obj, sel_utf8) if name_obj else None
            return raw.decode("utf-8", errors="replace") if raw else None

        explicit = msg_ptr_0(ns_app, lib.sel_registerName(b"appearance"))
        effective = msg_ptr_0(ns_app, lib.sel_registerName(b"effectiveAppearance"))
        return {"explicit": _name(explicit), "effective": _name(effective)}
    except Exception as exc:
        logger.debug("Failed to query macOS appearance state: %s", exc)
        return None
    finally:
        if pool:
            try:
                lib = _get_objc_lib()
                ctypes.cast(lib.objc_msgSend, _MSG_VOID_0)(
                    pool, lib.sel_registerName(b"drain")
                )
            except Exception:
                pass


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

_MACOS_WIDGETS_QSS = """
    QToolBar {{ background-color: {window}; color: {text}; border: none; border-bottom: 1px solid {border}; }}
    QToolBar QLabel {{ color: {text}; background-color: transparent; }}
    QDockWidget::title {{ background-color: {raised}; color: {text}; padding: 4px; }}
    QStatusBar {{ background-color: {window}; color: {text}; }}
    QStatusBar QLabel {{ color: {text}; background-color: transparent; }}
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
            if sys.platform == "darwin":
                macos_dark = _macos_system_is_dark(app)
                if macos_dark is not None:
                    return macos_dark
                hints = getattr(app, "styleHints", lambda: None)()
                scheme_fn = getattr(hints, "colorScheme", None)
                if callable(scheme_fn):
                    scheme = scheme_fn()
                    if scheme == Qt.ColorScheme.Dark:
                        return True
                    if scheme == Qt.ColorScheme.Light:
                        return False
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

        self._applying_theme = True
        try:
            self._connect_system_theme_changes(app)
            if sys.platform == "darwin":
                _set_macos_appearance(app, self._current_mode)

            if self._current_mode == ThemeMode.SYSTEM:
                if sys.platform == "darwin":
                    self._apply_palette(app, self._system_is_dark())
                else:
                    # Reset to system default palette
                    app.setPalette(app.style().standardPalette())
                    # On Windows, menus don't properly inherit palette colors,
                    # so we need to apply minimal menu styling based on detected theme
                    app.setStyleSheet(self._get_system_menu_stylesheet())
            elif self._current_mode == ThemeMode.DARK:
                self._apply_dark_palette(app)
            else:
                self._apply_light_palette(app)

            self._sync_theme_actions()
            # Update all themed components
            self._update_themed_components()
            if sys.platform == "darwin":
                self._refresh_macos_widgets()
        finally:
            self._applying_theme = False

    def _sync_theme_actions(self) -> None:
        mw = getattr(self, "mw", None)
        if mw is None:
            return
        action_map = {
            ThemeMode.LIGHT: getattr(mw, "theme_light_action", None),
            ThemeMode.DARK: getattr(mw, "theme_dark_action", None),
            ThemeMode.SYSTEM: getattr(mw, "theme_system_action", None),
        }
        target = action_map.get(self._current_mode)
        if target is None or target.isChecked():
            return
        group = getattr(mw, "theme_action_group", None)
        blockables = [obj for obj in (group, target) if obj is not None and hasattr(obj, "blockSignals")]
        previous = [(obj, obj.blockSignals(True)) for obj in blockables]
        try:
            target.setChecked(True)
        finally:
            for obj, was_blocked in reversed(previous):
                obj.blockSignals(was_blocked)

    def _connect_system_theme_changes(self, app: QApplication) -> None:
        if sys.platform != "darwin" or getattr(self, "_system_theme_connected", False):
            return
        hints = getattr(app, "styleHints", lambda: None)()
        signal = getattr(hints, "colorSchemeChanged", None)
        if signal is None:
            return
        self_ref = weakref.ref(self)

        def _on_scheme_changed(_scheme=None) -> None:
            manager = self_ref()
            if (
                manager is None
                or manager._current_mode != ThemeMode.SYSTEM
                or getattr(manager, "_applying_theme", False)
            ):
                return
            manager._apply_theme()

        try:
            signal.connect(_on_scheme_changed)
        except Exception:
            return
        self._system_theme_connected = True
        self._system_theme_slot = _on_scheme_changed
        mw = getattr(self, "mw", None)
        destroyed = getattr(mw, "destroyed", None)
        if destroyed is not None:

            def _disconnect(*_args) -> None:
                try:
                    signal.disconnect(_on_scheme_changed)
                except (RuntimeError, TypeError):
                    pass

            destroyed.connect(_disconnect)

    def _refresh_macos_widgets(self) -> None:
        """Repolish non-VTK chrome widgets on macOS after appearance updates."""
        mw = getattr(self, "mw", None)
        if mw is None:
            return
        containers = (
            getattr(mw, "main_toolbar", None),
            getattr(mw, "scalar_toolbar", None),
            getattr(mw, "data_dock_widget", None),
            getattr(mw, "status_bar", None),
        )
        seen: set[int] = set()
        targets: list[QWidget] = []
        for container in containers:
            if container is None:
                continue
            for widget in (container, *container.findChildren(QWidget)):
                ident = id(widget)
                if ident not in seen:
                    seen.add(ident)
                    targets.append(widget)
        for widget in targets:
            try:
                style = widget.style()
                if style is not None:
                    style.unpolish(widget)
                    style.polish(widget)
                widget.update()
            except RuntimeError:
                continue
        try:
            mw.update()
        except RuntimeError:
            pass

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
    def _application_stylesheet(
        is_dark: bool, is_macos: Optional[bool] = None
    ) -> str:
        if is_macos is None:
            is_macos = sys.platform == "darwin"
        tokens = _tokens(is_dark)
        if not is_macos:
            return (_TOOLTIP_QSS + _MENU_QSS + _WIDGETS_QSS).format(**tokens)
        mac_tokens = {**tokens, "dock_icons_rule": ""}
        return (
            _TOOLTIP_QSS + _MENU_QSS + _WIDGETS_QSS + _MACOS_WIDGETS_QSS
        ).format(**mac_tokens)

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
