# -*- coding: utf-8 -*-

"""
Utility functions, constants, and enums for the TractEdit application.
"""

import os
import ast
import logging 
import numpy as np
import vtk
import pytz
import enum
from datetime import datetime
from typing import Optional, Any, Union, List, Tuple

logger = logging.getLogger(__name__)

# --- Constants ---
MAX_STACK_LEVELS: int = 20
DEFAULT_SELECTION_RADIUS: float = 3.5
MIN_SELECTION_RADIUS: float = 0.5
RADIUS_INCREMENT: float = 0.5

# --- Coloring Mode Enum ---
class ColorMode(enum.Enum):
    """Enum defining the streamline coloring modes."""
    DEFAULT: int = 0
    ORIENTATION: int = 1
    SCALAR: int = 2

def get_formatted_datetime() -> str:
    """
    Gets the current local date, time, and timezone formatted for the status bar.
    Uses the system's configured local timezone.
    """
    try:
        # Get the current time using the system's local timezone
        now_aware = datetime.now().astimezone()
        return now_aware.strftime("%d/%m/%Y %H:%M:%S %Z")
    except Exception as e:
        logger.info(f"Could not get local aware time: {e}. Using system naive time.")
        now_naive = datetime.now()
        return now_naive.strftime("%Y-%m-%d %H:%M:%S")

def get_asset_path(asset_name: str) -> str:
    """Gets the absolute path to an asset in the 'assets' directory."""
    script_dir = os.path.dirname(__file__) 
    asset_path = os.path.join(script_dir, 'assets', asset_name)
    return asset_path

def format_tuple(data: Any, precision: int = 2) -> str:
    """Formats a tuple of numbers into a string '(num1, num2, ...)'."""
    if isinstance(data, (list, tuple)):
        try:
            return f"({', '.join(f'{x:.{precision}f}' for x in data)})"
        except (TypeError, ValueError):
            return str(data) # Fallback
    return str(data)