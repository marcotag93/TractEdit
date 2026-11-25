# -*- coding: utf-8 -*-
"""Session recipes reference verified inputs and never persist result tables."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..logic.surface_mapping_manager import SurfaceMappingManager

LEGACY_ALGORITHM_VERSION = "surface-mapping-v1"

DISPLAY_FIELDS = (
    "visible",
    "markers_enabled",
    "qc_enabled",
    "map_enabled",
    "density_enabled",
    "endpoint_enabled",
    "intersections_enabled",
)


def capture_recipe(manager: SurfaceMappingManager | None) -> dict[str, Any] | None:
    if (
        manager is None
        or manager.parameters is None
        or manager.mesh_id not in manager.mw.mesh_layers
    ):
        return None
    if manager.bundle is not None:
        ids = manager.bundle.result.request.streamline_ids
    elif manager.recipe_ids is not None:
        ids = manager.recipe_ids
    else:
        visible = manager.mw.visible_indices
        source = (
            visible
            if manager.parameters.scope == "visible"
            else manager.mw.selected_streamline_indices
        )
        ids = np.fromiter(
            sorted(index for index in source if index in visible), dtype=np.int64
        )
    if not len(ids):
        return None
    from ..surface_mapping import ALGORITHM_VERSION

    display = manager.recipe_display.copy()
    if manager.presentation is not None:
        display = {
            name: bool(getattr(manager.presentation, name)) for name in DISPLAY_FIELDS
        }
    return {
        "mesh_id": manager.mesh_id,
        "algorithm_version": ALGORITHM_VERSION,
        "parameters": asdict(manager.parameters),
        "streamline_ids": ids,
        "display": display,
    }


def validate_recipe(recipe: dict[str, Any] | None, state: dict[str, Any]) -> None:
    from .io import SessionError

    if recipe is None:
        return
    from ..surface_mapping import ALGORITHM_VERSION, SurfaceMappingParameters

    if not isinstance(recipe, dict) or set(recipe) != {
        "mesh_id",
        "algorithm_version",
        "parameters",
        "streamline_ids",
        "display",
    }:
        raise SessionError("Invalid surface mapping recipe fields")
    if not isinstance(recipe["mesh_id"], str) or recipe["mesh_id"] not in {
        layer["id"] for layer in state.get("mesh_layers", [])
    }:
        raise SessionError("Surface mapping recipe references an unknown mesh")
    if recipe["algorithm_version"] not in (LEGACY_ALGORITHM_VERSION, ALGORITHM_VERSION):
        raise SessionError("Unsupported surface mapping recipe algorithm")
    try:
        if not isinstance(recipe["parameters"], dict):
            raise ValueError("Invalid parameters")
        parameters = recipe["parameters"].copy()
        # Older sessions included a descriptive field with no computational effect.
        parameters.pop("surface_type", None)
        if set(parameters) != {
            "mode",
            "scope",
            "max_distance_mm",
        }:
            raise ValueError("Invalid parameters")
        SurfaceMappingParameters(**parameters)
    except (TypeError, ValueError) as error:
        raise SessionError("Invalid surface mapping recipe parameters") from error
    ids = recipe["streamline_ids"]
    if (
        type(state.get("streamline_count")) is not int
        or state["streamline_count"] < 0
        or not isinstance(ids, np.ndarray)
        or ids.dtype != np.int64
        or ids.ndim != 1
        or not len(ids)
        or np.any(ids < 0)
        or np.any(ids >= state["streamline_count"])
        or np.any(ids[1:] <= ids[:-1])
    ):
        raise SessionError("Invalid surface mapping recipe scope IDs")
    display = recipe["display"]
    if (
        not isinstance(display, dict)
        or set(display) not in (
            set(DISPLAY_FIELDS),
            set(DISPLAY_FIELDS) - {"endpoint_enabled", "intersections_enabled"},
        )
        or any(type(value) is not bool for value in display.values())
    ):
        raise SessionError("Invalid surface mapping recipe presentation")


def restore_recipe(manager: SurfaceMappingManager, recipe: dict[str, Any] | None) -> None:
    if recipe is None:
        return
    from ..surface_mapping import SurfaceMappingParameters

    parameters = recipe["parameters"].copy()
    parameters.pop("surface_type", None)
    manager.mesh_id = recipe["mesh_id"]
    manager.parameters = SurfaceMappingParameters(**parameters)
    manager.recipe_ids = recipe["streamline_ids"].copy()
    manager.recipe_ids.flags.writeable = False
    manager.recipe_display = recipe["display"].copy()
    # Surface-map controls have been removed; ignore legacy map visibility.
    manager.recipe_display["map_enabled"] = False
    manager.recipe_display.setdefault(
        "endpoint_enabled", "endpoint_proximity" in manager.parameters.modes
    )
    manager.recipe_display.setdefault(
        "intersections_enabled", "surface_intersections" in manager.parameters.modes
    )
    if not manager.recipe_display["markers_enabled"]:
        manager.recipe_display["endpoint_enabled"] = False
        manager.recipe_display["intersections_enabled"] = False
    manager.recipe_display["markers_enabled"] = True
    manager.stale = True
    manager.state = "idle"
    manager.message = f"Recompute required: restored recipe ({len(manager.recipe_ids):,} saved scope IDs)."
    if recipe["algorithm_version"] == LEGACY_ALGORITHM_VERSION:
        manager.message += " The previous algorithm used display vertex IDs for OBJ meshes."
    manager._refresh_tree()
    manager.changed.emit()
