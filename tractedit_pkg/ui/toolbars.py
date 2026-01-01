# -*- coding: utf-8 -*-

"""
Toolbars manager for TractEdit UI.

Handles creation of toolbars, status bar, and central widget setup.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QToolBar,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QCheckBox,
    QSpinBox,
    QSlider,
    QToolButton,
    QDoubleSpinBox,
    QSpacerItem,
    QSizePolicy,
    QStatusBar,
    QLineEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon, QPixmap, QPainter, QImage

from ..utils import get_asset_path

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Constant for slider precision
SLIDER_PRECISION = 1000


# ============================================================================
# Toolbars Manager Class
# ============================================================================


class ToolbarsManager:
    """
    Manages toolbar and status bar creation for the main window.

    This class handles:
    - Creating the main toolbar with skip/density controls
    - Creating the scalar range toolbar
    - Setting up the status bar
    - Setting up the central widget with VTK panel
    """

    def __init__(self, main_window: "MainWindow") -> None:
        """
        Initialize the toolbars manager.

        Args:
            main_window: Reference to the parent MainWindow instance.
        """
        self.mw = main_window

    def _get_button_style_default(self) -> str:
        """Returns the default button style from theme manager."""
        if hasattr(self.mw, "theme_manager"):
            return self.mw.theme_manager.get_button_style_default()
        # Fallback light style if theme manager not available
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

    def _get_label_color(self) -> str:
        """Returns the label color based on current theme."""
        if hasattr(self.mw, "theme_manager") and self.mw.theme_manager.is_dark_theme():
            return "#dddddd"
        return "#333333"

    def create_main_toolbar(self) -> None:
        """Creates the main toolbar with skip, opacity, and drawing controls."""
        mw = self.mw

        mw.main_toolbar = QToolBar("Main Tools", mw)
        mw.addToolBar(Qt.ToolBarArea.TopToolBarArea, mw.main_toolbar)

        # Container for Toolbar Widgets
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Skip / Density Control
        mw.skip_checkbox = QCheckBox()
        mw.skip_checkbox.setChecked(False)
        mw.skip_checkbox.toggled.connect(mw._on_skip_toggled)
        layout.addWidget(mw.skip_checkbox)

        layout.addWidget(QLabel("Skip %: "))
        mw.skip_spinbox = QSpinBox()
        mw.skip_spinbox.setRange(0, 99)
        mw.skip_spinbox.setValue(0)
        mw.skip_spinbox.setToolTip(
            "Percentage of streamlines to skip for rendering (0 = Show All, 99 = Show 1%)"
        )
        mw.skip_spinbox.setEnabled(False)
        mw.skip_spinbox.editingFinished.connect(mw._on_skip_changed)
        layout.addWidget(mw.skip_spinbox)

        # Spacer
        layout.addSpacing(20)
        layout.addWidget(QLabel("|"))
        layout.addSpacing(20)

        # Opacity Control
        mw.opacity_label = QLabel("Opacity: ")
        layout.addWidget(mw.opacity_label)

        mw.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        mw.opacity_slider.setRange(0, 100)
        mw.opacity_slider.setValue(100)
        mw.opacity_slider.setFixedWidth(120)
        mw.opacity_slider.setEnabled(False)
        mw.opacity_slider.valueChanged.connect(mw._on_opacity_slider_changed)
        layout.addWidget(mw.opacity_slider)

        # Drawing Mode Buttons
        layout.addSpacing(10)
        self._add_drawing_mode_buttons(layout)

        # Sphere Radius Control (visible only when sphere mode is active)
        self._add_sphere_radius_control(layout)

        # Brush Size Control
        layout.addSpacing(15)
        layout.addWidget(QLabel("|"))
        layout.addSpacing(15)

        mw.brush_label = QLabel("Brush:")
        mw.brush_label.setStyleSheet("font-weight: bold; color: #dddddd;")
        layout.addWidget(mw.brush_label)

        # Brush size value label
        mw.brush_size_label = QLabel(f"{mw.draw_brush_size}")
        mw.brush_size_label.setStyleSheet(
            """
            QLabel {
                font-weight: bold;
                color: #00BCD4;
                font-size: 13px;
                min-width: 15px;
            }
            """
        )
        layout.addWidget(mw.brush_size_label)

        # Brush size slider
        mw.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        mw.brush_size_slider.setRange(1, 10)
        mw.brush_size_slider.setValue(mw.draw_brush_size)
        mw.brush_size_slider.setToolTip("Brush size: 1-10 voxels")
        mw.brush_size_slider.setFixedWidth(100)
        mw.brush_size_slider.valueChanged.connect(mw._on_brush_size_changed)
        mw.brush_size_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #c0c0c0;
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00BCD4;
                border: 1px solid #00ACC1;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #00ACC1;
            }
            """
        )
        layout.addWidget(mw.brush_size_slider)

        # Add stretch to push items to the left
        layout.addStretch()

        # Add container to toolbar
        mw.main_toolbar.addWidget(container)

    def _add_drawing_mode_buttons(self, layout: QHBoxLayout) -> None:
        """Adds drawing mode buttons to the toolbar layout."""
        mw = self.mw

        # Get theme-aware button style
        button_style = self._get_button_style_default()

        # Draw Mode Button
        mw.draw_mode_button = QToolButton()
        pencil_icon = self._create_pencil_icon()
        mw.draw_mode_action.setIcon(pencil_icon)
        mw.draw_mode_button.setDefaultAction(mw.draw_mode_action)
        mw.draw_mode_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        mw.draw_mode_button.setStyleSheet(button_style)
        layout.addWidget(mw.draw_mode_button)

        # Erase Mode Button
        mw.erase_mode_button = QToolButton()
        eraser_icon = self._create_eraser_icon()
        mw.erase_mode_action.setIcon(eraser_icon)
        mw.erase_mode_button.setDefaultAction(mw.erase_mode_action)
        mw.erase_mode_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        mw.erase_mode_button.setStyleSheet(button_style)
        layout.addWidget(mw.erase_mode_button)

        # Sphere Mode Button
        mw.sphere_mode_button = QToolButton()
        sphere_icon = self._create_sphere_icon()
        mw.sphere_mode_action.setIcon(sphere_icon)
        mw.sphere_mode_button.setDefaultAction(mw.sphere_mode_action)
        mw.sphere_mode_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        mw.sphere_mode_button.setStyleSheet(button_style)
        layout.addWidget(mw.sphere_mode_button)

        # Rectangle Mode Button
        mw.rectangle_mode_button = QToolButton()
        rectangle_icon = self._create_rectangle_icon()
        mw.rectangle_mode_action.setIcon(rectangle_icon)
        mw.rectangle_mode_button.setDefaultAction(mw.rectangle_mode_action)
        mw.rectangle_mode_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        mw.rectangle_mode_button.setStyleSheet(button_style)
        layout.addWidget(mw.rectangle_mode_button)

    def _add_sphere_radius_control(self, layout: QHBoxLayout) -> None:
        """
        Adds the sphere radius control to the toolbar.

        Creates a container with a label and spinbox for precise radius input.
        The container is initially hidden and shown only when sphere mode is active.
        """
        mw = self.mw

        # Container widget for sphere radius controls
        mw.sphere_radius_container = QWidget()
        radius_layout = QHBoxLayout(mw.sphere_radius_container)
        radius_layout.setContentsMargins(0, 0, 0, 0)
        radius_layout.setSpacing(5)

        # Separator
        separator = QLabel("|")
        separator.setStyleSheet("color: #888;")
        radius_layout.addWidget(separator)
        radius_layout.addSpacing(10)

        # Label
        radius_label = QLabel("Radius:")
        radius_label.setStyleSheet("font-weight: bold; color: #333;")
        radius_layout.addWidget(radius_label)

        # Spinbox for radius value
        mw.sphere_radius_spinbox = QDoubleSpinBox()
        mw.sphere_radius_spinbox.setRange(0.5, 100.0)
        mw.sphere_radius_spinbox.setSingleStep(0.5)
        mw.sphere_radius_spinbox.setValue(5.0)
        mw.sphere_radius_spinbox.setSuffix(" mm")
        mw.sphere_radius_spinbox.setDecimals(1)
        mw.sphere_radius_spinbox.setToolTip(
            "Set sphere radius in mm.\n"
            "Ctrl+Scroll over 2D views also adjusts radius."
        )
        mw.sphere_radius_spinbox.setFixedWidth(90)
        mw.sphere_radius_spinbox.setStyleSheet(
            """
            QDoubleSpinBox {
                border: 1px solid #00BCD4;
                border-radius: 4px;
                padding: 2px 5px;
                background: white;
                font-weight: bold;
                color: #00838F;
            }
            QDoubleSpinBox:focus {
                border: 2px solid #00ACC1;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 18px;
                border: none;
                background: #e0f7fa;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #b2ebf2;
            }
            """
        )
        # Preview on value change, apply on editing finished
        mw.sphere_radius_spinbox.valueChanged.connect(mw._on_sphere_radius_preview)
        mw.sphere_radius_spinbox.editingFinished.connect(mw._on_sphere_radius_changed)
        radius_layout.addWidget(mw.sphere_radius_spinbox)

        # Add container to main layout
        layout.addWidget(mw.sphere_radius_container)
        mw.sphere_radius_container.setVisible(False)

    def _is_dark_theme(self) -> bool:
        """Checks if the current theme is dark."""
        if hasattr(self.mw, "theme_manager"):
            return self.mw.theme_manager.is_dark_theme()
        return False

    def _create_themed_icon(self, asset_name: str) -> QIcon:
        """
        Creates a theme-aware icon from the given asset.

        For dark themes, the icon is inverted to ensure visibility.
        For light themes, the original icon is used.

        Args:
            asset_name: Name of the icon file in the assets folder.

        Returns:
            QIcon: Themed icon suitable for the current theme.
        """
        icon_path = get_asset_path(asset_name)
        pixmap = QPixmap(icon_path)

        if pixmap.isNull():
            logger.warning(f"Could not load icon from {icon_path}")
            return QIcon()

        if self._is_dark_theme():
            # Convert to QImage and use Qt's optimized inversion
            # InvertRgb mode inverts only RGB values, preserving alpha channel
            image = pixmap.toImage()
            image.invertPixels(QImage.InvertMode.InvertRgb)
            pixmap = QPixmap.fromImage(image)

        return QIcon(pixmap)

    def _create_pencil_icon(self) -> QIcon:
        """Loads the pencil icon from the assets folder with theme awareness."""
        return self._create_themed_icon("pencil.png")

    def _create_eraser_icon(self) -> QIcon:
        """Loads the eraser icon from the assets folder with theme awareness."""
        return self._create_themed_icon("eraser.png")

    def _create_sphere_icon(self) -> QIcon:
        """Loads the sphere icon from the assets folder with theme awareness."""
        return self._create_themed_icon("sphere.png")

    def _create_rectangle_icon(self) -> QIcon:
        """Loads the rectangle icon from the assets folder with theme awareness."""
        return self._create_themed_icon("rectangle.png")

    def create_scalar_toolbar(self) -> None:
        """Creates the toolbar for scalar range adjustment with sliders."""
        mw = self.mw

        mw.scalar_toolbar = QToolBar("Scalar Range", mw)
        mw.scalar_toolbar.setObjectName("ScalarToolbar")

        # Spinboxes for precise input/display
        mw.scalar_min_spinbox = QDoubleSpinBox(mw)
        mw.scalar_min_spinbox.setDecimals(3)
        mw.scalar_min_spinbox.setSingleStep(0.1)
        mw.scalar_min_spinbox.setRange(-1e9, 1e9)
        mw.scalar_min_spinbox.setToolTip("Min scalar value")

        mw.scalar_max_spinbox = QDoubleSpinBox(mw)
        mw.scalar_max_spinbox.setDecimals(3)
        mw.scalar_max_spinbox.setSingleStep(0.1)
        mw.scalar_max_spinbox.setRange(-1e9, 1e9)
        mw.scalar_max_spinbox.setToolTip("Max scalar value")

        # Sliders for interactive dragging
        mw.scalar_min_slider = QSlider(Qt.Orientation.Horizontal, mw)
        mw.scalar_min_slider.setRange(0, SLIDER_PRECISION)
        mw.scalar_min_slider.setToolTip("Drag to adjust min scalar value")

        mw.scalar_max_slider = QSlider(Qt.Orientation.Horizontal, mw)
        mw.scalar_max_slider.setRange(0, SLIDER_PRECISION)
        mw.scalar_max_slider.setValue(SLIDER_PRECISION)
        mw.scalar_max_slider.setToolTip("Drag to adjust max scalar value")

        # Reset Button
        mw.scalar_reset_button = QAction("Reset", mw)
        mw.scalar_reset_button.setStatusTip("Reset scalar range to data min/max")

        # Layout
        toolbar_widget = QWidget(mw)
        layout = QHBoxLayout(toolbar_widget)
        layout.setContentsMargins(5, 0, 5, 0)

        layout.addWidget(QLabel(" Min: "))
        layout.addWidget(mw.scalar_min_spinbox, 1)
        layout.addWidget(mw.scalar_min_slider, 3)

        layout.addSpacerItem(
            QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Ignored)
        )

        layout.addWidget(QLabel(" Max: "))
        layout.addWidget(mw.scalar_max_spinbox, 1)
        layout.addWidget(mw.scalar_max_slider, 3)

        mw.scalar_toolbar.addWidget(toolbar_widget)
        mw.scalar_toolbar.addSeparator()
        mw.scalar_toolbar.addAction(mw.scalar_reset_button)

        mw.addToolBar(Qt.ToolBarArea.TopToolBarArea, mw.scalar_toolbar)
        mw.scalar_toolbar.setVisible(False)

        # Connect Signals
        mw.scalar_min_slider.valueChanged.connect(mw._slider_value_changed)
        mw.scalar_max_slider.valueChanged.connect(mw._slider_value_changed)
        mw.scalar_min_slider.sliderReleased.connect(mw._trigger_vtk_update)
        mw.scalar_max_slider.sliderReleased.connect(mw._trigger_vtk_update)
        mw.scalar_min_spinbox.editingFinished.connect(mw._spinbox_value_changed)
        mw.scalar_max_spinbox.editingFinished.connect(mw._spinbox_value_changed)
        mw.scalar_reset_button.triggered.connect(mw._reset_scalar_range)

    def setup_status_bar(self) -> None:
        """
        Creates and configures the status bar with permanent widgets for
        bundle/image info and interactive RAS coordinate display.
        """
        mw = self.mw

        # Create Container Widget
        mw.permanent_status_widget = QWidget(mw)
        layout = QHBoxLayout(mw.permanent_status_widget)
        layout.setContentsMargins(0, 0, 5, 0)
        layout.setSpacing(10)

        # Data Info Label
        mw.data_info_label = QLabel(" No data loaded ")
        mw.data_info_label.setStyleSheet("border: 1px solid grey; padding: 2px;")
        layout.addWidget(mw.data_info_label, 1)

        # RAS Coordinate Display
        mw.ras_label = QLabel("RAS: ", mw)
        mw.ras_label.setToolTip(
            "Current RAS coordinates. Enter values (e.g., '10.5, -5, 20') and press Enter."
        )
        layout.addWidget(mw.ras_label, 0)

        mw.ras_coordinate_input = QLineEdit("--, --, --", mw)
        mw.ras_coordinate_input.setToolTip(
            "Current RAS coordinates. Enter values (e.g., '10.5, -5, 20') and press Enter."
        )
        mw.ras_coordinate_input.setMinimumWidth(150)
        mw.ras_coordinate_input.setMaximumWidth(180)
        mw.ras_coordinate_input.setStyleSheet("border: 1px solid grey; padding: 2px;")
        layout.addWidget(mw.ras_coordinate_input, 0)

        # Add Container to Status Bar
        mw.status_bar = mw.statusBar()
        mw.status_bar.addPermanentWidget(mw.permanent_status_widget)

        # Connect Signal for Manual Entry
        mw.ras_coordinate_input.returnPressed.connect(mw._on_ras_coordinate_entered)

    def setup_central_widget(self) -> None:
        """Sets up the main central widget which will contain the VTK panel."""
        from ..visualization import VTKPanel

        mw = self.mw

        mw.central_widget = QWidget()
        mw.setCentralWidget(mw.central_widget)
        mw.vtk_panel = VTKPanel(parent_widget=mw.central_widget, main_window_ref=mw)
