# -*- coding: utf-8 -*-

"""
Utility functions, constants, and enums for the TractEdit application.
"""

# ============================================================================
# Imports
# ============================================================================

import os
import ast
import logging
import numpy as np
import vtk
import enum
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Any, Union, List, Tuple, Generator

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

MAX_STACK_LEVELS: int = 20
DEFAULT_SELECTION_RADIUS: float = 3.5
MIN_SELECTION_RADIUS: float = 0.5
RADIUS_INCREMENT: float = 0.5
SLIDER_PRECISION: int = 1000  # Use 1000 steps for the slider

# Maximum number of streamlines that can be simultaneously highlighted. 
# Operations that would exceed this limit are
# rejected early with an informative status message instead of attempting the allocation.
MAX_HIGHLIGHT_STREAMLINES: int = 200_000

TARGET_RENDER_COUNT: int = 20_000
AUTO_SKIP_THRESHOLD: int = 150_000

# Predefined colors for ROIs (distinct palette)
ROI_COLORS = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (0.0, 1.0, 1.0),  # Cyan
    (1.0, 0.0, 1.0),  # Magenta
    (1.0, 0.5, 0.0),  # Orange
    (0.5, 0.0, 1.0),  # Purple
    (0.0, 1.0, 0.5),  # Spring Green
    (1.0, 0.0, 0.5),  # Rose
]


# ============================================================================
# Enums
# ============================================================================


class ColorMode(enum.Enum):
    """Enum defining the streamline coloring modes."""

    DEFAULT: int = 0
    ORIENTATION: int = 1
    SCALAR: int = 2


# ============================================================================
# Utility Functions
# ============================================================================


@contextmanager
def signals_blocked(*widgets: Any) -> Generator[None, None, None]:
    """Context manager that blocks Qt signals for one or more widgets.

    Guarantees that ``blockSignals(False)`` is called even if the body
    raises an exception, preventing permanently muted widgets.

    Args:
        *widgets: One or more Qt widgets whose signals should be blocked.

    Example::

        with signals_blocked(spinbox, slider):
            spinbox.setValue(42)
            slider.setValue(50)
    """
    for w in widgets:
        w.blockSignals(True)
    try:
        yield
    finally:
        for w in widgets:
            w.blockSignals(False)


def get_formatted_datetime() -> str:
    """
    Gets the current local date, time, and timezone formatted for the status bar.
    Uses the system's configured local timezone.
    """
    try:
        now_aware = datetime.now().astimezone()
        return now_aware.strftime("%d/%m/%Y %H:%M:%S %Z")
    except (OSError, ValueError, OverflowError) as e:
        logger.info(f"Could not get local aware time: {e}. Using system naive time.")
        now_naive = datetime.now()
        return now_naive.strftime("%Y-%m-%d %H:%M:%S")


def get_asset_path(asset_name: str) -> str:
    """Gets the absolute path to an asset in the 'assets' directory."""
    script_dir = os.path.dirname(__file__)
    asset_path = os.path.join(script_dir, "assets", asset_name)
    return asset_path


def format_tuple(data: Any, precision: int = 2) -> str:
    """Formats a tuple of numbers into a string '(num1, num2, ...)'."""
    if isinstance(data, (list, tuple)):
        try:
            return f"({', '.join(f'{x:.{precision}f}' for x in data)})"
        except (TypeError, ValueError):
            return str(data)  # Fallback
    return str(data)
