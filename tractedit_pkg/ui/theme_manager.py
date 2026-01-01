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
from typing import TYPE_CHECKING, Dict, Any

from PyQt6.QtWidgets import QApplication, QStyleFactory
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
        saved_theme = self._settings.value(self.SETTINGS_KEY, ThemeMode.SYSTEM.value)
        try:
            self._current_mode = ThemeMode(saved_theme)
        except ValueError:
            self._current_mode = ThemeMode.SYSTEM

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
        else:
            # System mode - detect from palette
            app = QApplication.instance()
            if app:
                palette = app.palette()
                bg_color = palette.color(QPalette.ColorRole.Window)
                # If background luminance is low, it's a dark theme
                luminance = (
                    0.299 * bg_color.red()
                    + 0.587 * bg_color.green()
                    + 0.114 * bg_color.blue()
                )
                return luminance < 128
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
        """Applies a dark color palette to the application."""
        palette = QPalette()

        # Base colors
        dark_bg = QColor(45, 45, 45)
        darker_bg = QColor(35, 35, 35)
        light_text = QColor(220, 220, 220)
        disabled_text = QColor(127, 127, 127)
        highlight = QColor(42, 130, 218)
        highlight_text = QColor(255, 255, 255)

        # Window and base
        palette.setColor(QPalette.ColorRole.Window, dark_bg)
        palette.setColor(QPalette.ColorRole.WindowText, light_text)
        palette.setColor(QPalette.ColorRole.Base, darker_bg)
        palette.setColor(QPalette.ColorRole.AlternateBase, dark_bg)
        palette.setColor(QPalette.ColorRole.ToolTipBase, dark_bg)
        palette.setColor(QPalette.ColorRole.ToolTipText, light_text)
        palette.setColor(QPalette.ColorRole.Text, light_text)
        palette.setColor(QPalette.ColorRole.Button, dark_bg)
        palette.setColor(QPalette.ColorRole.ButtonText, light_text)
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Link, highlight)
        palette.setColor(QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, highlight_text)

        # Disabled colors
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.HighlightedText,
            disabled_text,
        )

        app.setPalette(palette)
        app.setStyleSheet(self._get_dark_stylesheet())

    def _apply_light_palette(self, app: QApplication) -> None:
        """Applies a light color palette to the application."""
        palette = QPalette()

        # Base colors
        light_bg = QColor(240, 240, 240)
        white_bg = QColor(255, 255, 255)
        dark_text = QColor(30, 30, 30)
        disabled_text = QColor(127, 127, 127)
        highlight = QColor(0, 120, 215)
        highlight_text = QColor(255, 255, 255)

        # Window and base
        palette.setColor(QPalette.ColorRole.Window, light_bg)
        palette.setColor(QPalette.ColorRole.WindowText, dark_text)
        palette.setColor(QPalette.ColorRole.Base, white_bg)
        palette.setColor(QPalette.ColorRole.AlternateBase, light_bg)
        palette.setColor(QPalette.ColorRole.ToolTipBase, white_bg)
        palette.setColor(QPalette.ColorRole.ToolTipText, dark_text)
        palette.setColor(QPalette.ColorRole.Text, dark_text)
        palette.setColor(QPalette.ColorRole.Button, light_bg)
        palette.setColor(QPalette.ColorRole.ButtonText, dark_text)
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, highlight)
        palette.setColor(QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, highlight_text)

        # Disabled colors
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.HighlightedText,
            disabled_text,
        )

        app.setPalette(palette)
        app.setStyleSheet(self._get_light_stylesheet())

    def _get_dark_stylesheet(self) -> str:
        """Returns the dark theme stylesheet."""
        return """
            QToolTip {
                color: #dddddd;
                background-color: #2d2d2d;
                border: 1px solid #555555;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #dddddd;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #dddddd;
                border: 1px solid #555555;
            }
            QMenu::item:selected {
                background-color: #2a82da;
            }
            QScrollBar:vertical {
                background: #2d2d2d;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #555555;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666666;
            }
            QScrollBar:horizontal {
                background: #2d2d2d;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background: #555555;
                min-width: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #666666;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
                border: none;
            }
            QDockWidget {
                titlebar-close-icon: none;
                titlebar-normal-icon: none;
            }
            QDockWidget::title {
                background-color: #3d3d3d;
                padding: 4px;
            }
        """

    def _get_light_stylesheet(self) -> str:
        """Returns the light theme stylesheet."""
        return """
            QToolTip {
                color: #1e1e1e;
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
            }
            QMenuBar {
                background-color: #f0f0f0;
                color: #1e1e1e;
            }
            QMenuBar::item {
                background-color: transparent;
                color: #1e1e1e;
            }
            QMenuBar::item:selected {
                background-color: #e0e0e0;
                color: #1e1e1e;
            }
            QMenu {
                background-color: #ffffff;
                color: #1e1e1e;
                border: 1px solid #c0c0c0;
            }
            QMenu::item {
                color: #1e1e1e;
                padding: 4px 20px 4px 20px;
            }
            QMenu::item:selected {
                background-color: #0078d7;
                color: #ffffff;
            }
            QMenu::item:disabled {
                color: #a0a0a0;
            }
            QMenu::separator {
                height: 1px;
                background-color: #d0d0d0;
                margin: 4px 10px;
            }
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar:horizontal {
                background: #f0f0f0;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background: #c0c0c0;
                min-width: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
                border: none;
            }
            QDockWidget::title {
                background-color: #e0e0e0;
                padding: 4px;
            }
        """

    def _get_system_menu_stylesheet(self) -> str:
        """
        Returns menu-only stylesheet for System theme mode.

        On Windows, Qt menus don't properly inherit QPalette colors,
        so we detect the system theme and apply appropriate menu styling.
        This preserves the native look while ensuring menu text is visible.
        """
        # Detect if system theme is dark or light
        app = QApplication.instance()
        if app:
            palette = app.palette()
            bg_color = palette.color(QPalette.ColorRole.Window)
            # Calculate luminance to determine if dark or light
            luminance = (
                0.299 * bg_color.red()
                + 0.587 * bg_color.green()
                + 0.114 * bg_color.blue()
            )
            is_system_dark = luminance < 128
        else:
            is_system_dark = False

        if is_system_dark:
            # Dark system theme - light text on dark background
            return """
                QMenuBar {
                    background-color: #2d2d2d;
                    color: #dddddd;
                }
                QMenuBar::item:selected {
                    background-color: #3d3d3d;
                }
                QMenu {
                    background-color: #2d2d2d;
                    color: #dddddd;
                    border: 1px solid #555555;
                }
                QMenu::item:selected {
                    background-color: #2a82da;
                }
            """
        else:
            # Light system theme - dark text on light background
            return """
                QMenuBar {
                    background-color: #f0f0f0;
                    color: #1e1e1e;
                }
                QMenuBar::item {
                    background-color: transparent;
                    color: #1e1e1e;
                }
                QMenuBar::item:selected {
                    background-color: #e0e0e0;
                    color: #1e1e1e;
                }
                QMenu {
                    background-color: #ffffff;
                    color: #1e1e1e;
                    border: 1px solid #c0c0c0;
                }
                QMenu::item {
                    color: #1e1e1e;
                    padding: 4px 20px 4px 20px;
                }
                QMenu::item:selected {
                    background-color: #0078d7;
                    color: #ffffff;
                }
                QMenu::item:disabled {
                    color: #a0a0a0;
                }
                QMenu::separator {
                    height: 1px;
                    background-color: #d0d0d0;
                    margin: 4px 10px;
                }
            """

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

        # Refresh icons from toolbar manager (they are theme-aware)
        if hasattr(mw, "toolbars_manager"):
            tm = mw.toolbars_manager

            if hasattr(mw, "draw_mode_button") and hasattr(mw, "draw_mode_action"):
                mw.draw_mode_action.setIcon(tm._create_pencil_icon())

            if hasattr(mw, "erase_mode_button") and hasattr(mw, "erase_mode_action"):
                mw.erase_mode_action.setIcon(tm._create_eraser_icon())

            if hasattr(mw, "sphere_mode_button") and hasattr(mw, "sphere_mode_action"):
                mw.sphere_mode_action.setIcon(tm._create_sphere_icon())

            if hasattr(mw, "rectangle_mode_button") and hasattr(
                mw, "rectangle_mode_action"
            ):
                mw.rectangle_mode_action.setIcon(tm._create_rectangle_icon())

        # Get current button states and re-apply styles
        if hasattr(mw, "draw_mode_button"):
            if mw.is_drawing_mode:
                mw.draw_mode_button.setStyleSheet(self.get_button_style_active())
            else:
                mw.draw_mode_button.setStyleSheet(self.get_button_style_default())

        if hasattr(mw, "erase_mode_button"):
            if mw.is_eraser_mode:
                mw.erase_mode_button.setStyleSheet(self.get_button_style_active())
            else:
                mw.erase_mode_button.setStyleSheet(self.get_button_style_default())

        if hasattr(mw, "sphere_mode_button"):
            if getattr(mw, "is_sphere_mode", False):
                mw.sphere_mode_button.setStyleSheet(self.get_button_style_active())
            else:
                mw.sphere_mode_button.setStyleSheet(self.get_button_style_default())

        if hasattr(mw, "rectangle_mode_button"):
            if getattr(mw, "is_rectangle_mode", False):
                mw.rectangle_mode_button.setStyleSheet(self.get_button_style_active())
            else:
                mw.rectangle_mode_button.setStyleSheet(self.get_button_style_default())

    def _update_toolbar_styles(self) -> None:
        """Updates toolbar component styles based on current theme."""
        mw = self.mw
        is_dark = self.is_dark_theme()

        label_color = "#dddddd" if is_dark else "#333333"

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
                        except Exception:
                            pass

        # Update slider style
        if hasattr(mw, "brush_size_slider"):
            mw.brush_size_slider.setStyleSheet(self.get_slider_style())

        # Update spinbox style
        if hasattr(mw, "sphere_radius_spinbox"):
            mw.sphere_radius_spinbox.setStyleSheet(self.get_spinbox_style())

    def _update_data_panel_styles(self) -> None:
        """Updates data panel styles based on current theme."""
        mw = self.mw
        is_dark = self.is_dark_theme()

        if hasattr(mw, "data_tree_widget"):
            if is_dark:
                mw.data_tree_widget.setStyleSheet(
                    """
                    QTreeWidget {
                        outline: 0;
                        background-color: #2d2d2d;
                        color: #dddddd;
                    }
                    QTreeWidget::item {
                        padding: 4px;
                        border-radius: 4px;
                    }
                    QTreeWidget::item:selected {
                        background-color: #2a82da;
                        color: white;
                        border: none;
                    }
                    QTreeWidget::item:selected:!active {
                        background-color: #2a82da;
                        color: white;
                        border: none;
                    }
                    QTreeWidget::item:hover {
                        background-color: rgba(255, 255, 255, 0.05);
                        border: none;
                    }
                    QTreeWidget::item:selected:hover {
                        background-color: #3a92ea;
                        color: white;
                    }
                """
                )
            else:
                mw.data_tree_widget.setStyleSheet(
                    """
                    QTreeWidget {
                        outline: 0;
                    }
                    QTreeWidget::item {
                        padding: 4px;
                        border-radius: 4px;
                    }
                    QTreeWidget::item:selected {
                        background-color: #4a6984;
                        color: white;
                        border: none;
                    }
                    QTreeWidget::item:selected:!active {
                        background-color: #4a6984;
                        color: white;
                        border: none;
                    }
                    QTreeWidget::item:hover {
                        background-color: rgba(0, 0, 0, 0.03);
                        border: none;
                    }
                    QTreeWidget::item:selected:hover {
                        background-color: #557ba0;
                        color: white;
                    }
                """
                )

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
        if self.is_dark_theme():
            return """
                QToolButton {
                    background-color: #3d3d3d;
                    border: 1px solid #555555;
                    border-radius: 5px;
                    padding: 2px;
                    margin-left: 5px;
                }
                QToolButton:hover {
                    background-color: #4d4d4d;
                    border: 1px solid #666666;
                }
                QToolButton:checked {
                    background-color: #2d2d2d;
                    border: 1px inset #444444;
                    padding: 3px 1px 1px 3px;
                }
            """
        else:
            return """
                QToolButton {
                    background-color: #f0f0f0;
                    border: 1px solid #c0c0c0;
                    border-radius: 5px;
                    padding: 2px;
                    margin-left: 5px;
                }
                QToolButton:hover {
                    background-color: #e8e8e8;
                    border: 1px solid #b0b0b0;
                }
                QToolButton:checked {
                    background-color: #d0d0d0;
                    border: 1px inset #a0a0a0;
                    padding: 3px 1px 1px 3px;
                }
            """

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
        if self.is_dark_theme():
            return """
                QSlider::groove:horizontal {
                    border: 1px solid #555555;
                    height: 6px;
                    background: #2d2d2d;
                    margin: 0px;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #05B8CC;
                    border: 1px solid #049dad;
                    width: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }
                QSlider::handle:horizontal:hover {
                    background: #06d4eb;
                }
            """
        else:
            return """
                QSlider::groove:horizontal {
                    border: 1px solid #bbb;
                    height: 6px;
                    background: #e0e0e0;
                    margin: 0px;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #05B8CC;
                    border: 1px solid #049dad;
                    width: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }
                QSlider::handle:horizontal:hover {
                    background: #06d4eb;
                }
            """

    def get_spinbox_style(self) -> str:
        """Returns the spinbox style for the current theme."""
        if self.is_dark_theme():
            return """
                QDoubleSpinBox {
                    background-color: #2d2d2d;
                    color: #dddddd;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 2px;
                }
                QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                    background-color: #3d3d3d;
                    border: 1px solid #555555;
                }
                QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                    background-color: #4d4d4d;
                }
            """
        else:
            return """
                QDoubleSpinBox {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #c0c0c0;
                    border-radius: 3px;
                    padding: 2px;
                }
            """

    def get_progress_bar_style(self) -> str:
        """Returns the progress bar style for the current theme."""
        if self.is_dark_theme():
            return """
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 4px;
                    background-color: #333;
                    color: white;
                    text-align: center;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #05B8CC;
                }
            """
        else:
            return """
                QProgressBar {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: #f0f0f0;
                    color: #333333;
                    text-align: center;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #05B8CC;
                }
            """

    def get_progress_dialog_style(self) -> str:
        """Returns the progress dialog style for the current theme."""
        if self.is_dark_theme():
            return """
                QProgressDialog {
                    background-color: #2b2b2b;
                    color: #dddddd;
                }
                QLabel {
                    color: #dddddd;
                    font-size: 12px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 4px;
                    background-color: #333;
                    color: white;
                    text-align: center;
                    font-size: 12px;
                    height: 25px;
                }
                QProgressBar::chunk {
                    background-color: #05B8CC;
                    border-radius: 3px;
                }
                QPushButton {
                    background-color: #444;
                    color: #ddd;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 4px 12px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
            """
        else:
            return """
                QProgressDialog {
                    background-color: #f5f5f5;
                    color: #333333;
                }
                QLabel {
                    color: #333333;
                    font-size: 12px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                QProgressBar {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: #e0e0e0;
                    color: #333333;
                    text-align: center;
                    font-size: 12px;
                    height: 25px;
                }
                QProgressBar::chunk {
                    background-color: #05B8CC;
                    border-radius: 3px;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    color: #333333;
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    padding: 4px 12px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """

    def initialize_theme(self) -> None:
        """Initializes the theme on application startup."""
        self._apply_theme()
