# -*- coding: utf-8 -*-

"""
Utility functions, constants, and enums for the TractEdit application.
"""

# ============================================================================
# Imports
# ============================================================================

import os
import logging
import vtk
import enum
from contextlib import ExitStack, contextmanager
from datetime import datetime
from typing import Any, List, Generator

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

    Restore each previous signal state, including in nested scopes.

    Args:
        *widgets: One or more Qt widgets whose signals should be blocked.

    Example::

        with signals_blocked(spinbox, slider):
            spinbox.setValue(42)
            slider.setValue(50)
    """
    with ExitStack() as stack:
        for widget in widgets:
            previous = widget.blockSignals(True)
            stack.callback(widget.blockSignals, previous)
        yield


def iter_tree_items(tree, *, mesh_only: bool = False):
    pending = [tree.topLevelItem(i) for i in reversed(range(tree.topLevelItemCount()))]
    while pending:
        item = pending.pop()
        if mesh_only and item.parent() is None and item.text(0) != "Meshes":
            continue
        yield item
        pending.extend(item.child(i) for i in reversed(range(item.childCount())))


@contextmanager
def ui_tree_transaction(mw, *, include_camera=False, rollback=None, refresh_selection=True):
    """Restore the tree and its controls if model or scene publication fails."""
    from PyQt6.QtGui import QAction

    tree, slider = mw.data_tree_widget, mw.opacity_slider
    roots = [tree.topLevelItem(i).clone() for i in range(tree.topLevelItemCount())]
    current = tree.currentItem()
    states = [
        (item.isExpanded(), item.isSelected(), item is current)
        for item in iter_tree_items(tree)
    ]
    scroll = (tree.verticalScrollBar().value(), tree.horizontalScrollBar().value())
    opacity = (slider.isEnabled(), slider.value())
    actions = [(action, action.isEnabled()) for action in mw.findChildren(QAction)]
    pending = mw._data_panel_update_pending
    previous_active = getattr(mw, "_mesh_commit_active", False)
    if include_camera:
        camera = mw.vtk_panel.scene.GetActiveCamera()
        previous_camera = vtk.vtkCamera()
        previous_camera.DeepCopy(camera)
    mw._mesh_commit_active = True
    try:
        yield
    except Exception:
        if rollback is not None:
            rollback()
        if include_camera:
            camera.DeepCopy(previous_camera)
        with signals_blocked(tree, slider):
            tree.clear()
            tree.addTopLevelItems(roots)
            mw.surface_mapping_manager._refresh_tree()
            restored = list(iter_tree_items(tree))
            for item, (_, _, was_current) in zip(restored, states):
                if was_current:
                    tree.setCurrentItem(item)
                    break
            for item, (expanded, selected, _) in zip(restored, states):
                item.setSelected(selected)
                item.setExpanded(expanded)
            tree.verticalScrollBar().setValue(scroll[0])
            tree.horizontalScrollBar().setValue(scroll[1])
            slider.setEnabled(opacity[0])
            slider.setValue(opacity[1])
        for action, enabled in actions:
            action.setEnabled(enabled)
        try:
            if refresh_selection:
                mw.data_panel_manager.on_data_item_selected()
        finally:
            mw._data_panel_update_pending = pending
            if pending and mw._data_panel_debounce_timer is not None:
                mw._data_panel_debounce_timer.start(100)
        raise
    finally:
        mw._mesh_commit_active = previous_active


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


def write_vtk_polydata(writer: Any, poly_data: Any, output_path: str) -> None:
    """Write VTK polydata and raise if the writer does not complete."""
    errors: List[str] = []
    observer = writer.AddObserver(
        vtk.vtkCommand.ErrorEvent,
        lambda _caller, event: errors.append(event),
    )
    try:
        writer.SetFileName(output_path)
        writer.SetInputData(poly_data)
        result = writer.Write()
        error_code = writer.GetErrorCode()
    finally:
        writer.RemoveObserver(observer)

    if result == 0 or error_code != vtk.vtkErrorCode.NoError or errors:
        if error_code != vtk.vtkErrorCode.NoError:
            detail = vtk.vtkErrorCode.GetStringFromErrorCode(error_code)
        elif result == 0:
            detail = "writer returned failure"
        else:
            detail = "writer emitted an error event"
        raise OSError(f"VTK writer failed for '{output_path}': {detail}")
    if not os.path.isfile(output_path):
        raise OSError(f"VTK writer did not create '{output_path}'")


def format_tuple(data: Any, precision: int = 2) -> str:
    """Formats a tuple of numbers into a string '(num1, num2, ...)'."""
    if isinstance(data, (list, tuple)):
        try:
            return f"({', '.join(f'{x:.{precision}f}' for x in data)})"
        except (TypeError, ValueError):
            return str(data)  # Fallback
    return str(data)
