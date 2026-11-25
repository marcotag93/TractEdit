# -*- coding: utf-8 -*-
"""Layer-owned mesh preferences, independent of VTK handles and source geometry."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import Literal

from ..data_contracts import (
    MeshComponent,
    MeshComponentStyle,
    MeshLayer,
    MeshMainPresentation,
    MeshPresentation,
)


DEFAULT_MESH_COLOR = (0.8, 0.8, 0.8)
DEFAULT_MESH_OPACITY = 1.0
DEFAULT_MESH_EDGE_COLOR = (0.1, 0.45, 0.8)
DEFAULT_MESH_VERTEX_COLOR = (0.85, 0.4, 0.1)
DEFAULT_MESH_LINE_WIDTH = 1.0
DEFAULT_MESH_POINT_SIZE = 4.0
MESH_LINE_WIDTH_RANGE = (1.0, 10.0)
MESH_POINT_SIZE_RANGE = (1.0, 20.0)
MESH_NORMAL_LENGTH_RANGE = (0.1, 100.0)
MESH_NORMAL_DENSITY_RANGE = (0.1, 100.0)
MAIN_MESH_COMPONENTS = ("surface", "edges", "vertices")
ADVANCED_MESH_COMPONENTS = ("boundary_edges", "surface_normals", "bounding_box")
MESH_INFORMATION_COMPONENTS = (
    "information_vertices", "information_polygons", "information_source",
    "information_display", "information_bounds", "information_texture",
)
MESH_TREE_COMPONENTS = (
    *MAIN_MESH_COMPONENTS, "texture", "advanced", *ADVANCED_MESH_COMPONENTS,
    "information", *MESH_INFORMATION_COMPONENTS,
)
MESH_COMPONENT_LABELS = {
    "surface": "Surface", "edges": "Edges", "vertices": "Vertices",
    "bounding_box": "Bounding box", "boundary_edges": "Boundary edges",
    "surface_normals": "Surface normals",
}


def default_mesh_presentation() -> MeshPresentation:
    """Return independent preferences; widths and point sizes are screen pixels."""
    return {
        "surface": {"enabled": True},
        "edges": {
            "enabled": False,
            "color": DEFAULT_MESH_EDGE_COLOR,
            "line_width": DEFAULT_MESH_LINE_WIDTH,
        },
        "vertices": {
            "enabled": False,
            "color": DEFAULT_MESH_VERTEX_COLOR,
            "point_size": DEFAULT_MESH_POINT_SIZE,
        },
        "bounding_box": {
            "enabled": False, "color": (0.6, 0.45, 0.85), "line_width": 1.0,
        },
        "boundary_edges": {
            "enabled": False, "color": (0.85, 0.25, 0.4), "line_width": 2.0,
        },
        "surface_normals": {
            "enabled": False, "color": (0.15, 0.65, 0.5),
            "length_mm": 2.0, "density": 1.0,
        },
    }


def validate_mesh_component(component: MeshComponent) -> MeshComponent:
    if not isinstance(component, str) or component not in (
        "surface",
        "edges",
        "vertices",
        *ADVANCED_MESH_COMPONENTS,
    ):
        raise ValueError("Unknown mesh component")
    return component


def validate_mesh_boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("Mesh visibility must be a boolean")
    return value


def _mesh_number(value: object, name: str, bounds: tuple[float, float]) -> float:
    message = f"Mesh {name} must be finite and in [{bounds[0]:g}, {bounds[1]:g}]"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(message) from error
    if not isfinite(number) or not bounds[0] <= number <= bounds[1]:
        raise ValueError(message)
    return number


def _mesh_color(value: object) -> tuple[float, float, float]:
    message = "Mesh color must contain three finite values in [0, 1]"
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(message)
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError(message) from error
    if len(values) != 3:
        raise ValueError(message)
    return tuple(_mesh_number(channel, "color", (0.0, 1.0)) for channel in values)


def validate_mesh_component_style(
    component: MeshComponent,
    style: Mapping[str, object],
) -> MeshComponentStyle:
    """Validate every field before a caller changes any presentation state."""
    validate_mesh_component(component)
    if not isinstance(style, Mapping):
        raise ValueError("Mesh style must be a mapping")
    fields = {
        "surface": ("color", "opacity"),
        "edges": ("color", "line_width"),
        "vertices": ("color", "point_size"),
        "bounding_box": ("color", "line_width"),
        "boundary_edges": ("color", "line_width"),
        "surface_normals": ("color", "length_mm", "density"),
    }[component]
    if set(style) - set(fields):
        raise ValueError("Unsupported style property for mesh component")
    result: MeshComponentStyle = {}
    for field, value in style.items():
        if field == "color":
            result["color"] = _mesh_color(value)
        elif field == "opacity":
            result["opacity"] = _mesh_number(value, "opacity", (0.0, 1.0))
        elif field == "line_width":
            result["line_width"] = _mesh_number(
                value, "line width", MESH_LINE_WIDTH_RANGE
            )
        elif field == "point_size":
            result["point_size"] = _mesh_number(
                value, "point size", MESH_POINT_SIZE_RANGE
            )
        elif field == "length_mm":
            result[field] = _mesh_number(
                value, "normal length", MESH_NORMAL_LENGTH_RANGE,
            )
        elif field == "density":
            result[field] = _mesh_number(
                value, "normal density", MESH_NORMAL_DENSITY_RANGE,
            )
    return result


def mesh_component_visibility(
    layer: MeshLayer,
    component: MeshComponent | Literal["texture"],
) -> bool:
    """Resolve requested visibility; rendering consumes these preferences later."""
    if component == "texture":
        return bool(
            layer["visible"]
            and layer["presentation"]["surface"]["enabled"]
            and layer.get("texture", {}).get("enabled", False)
        )
    validate_mesh_component(component)
    return layer["visible"] and layer["presentation"][component]["enabled"]


def validate_mesh_presentation(
    value: object, *, advanced: bool | None = None,
) -> MeshMainPresentation | MeshPresentation:
    """Accept exact schema 5 or schema 6 preferences, without derived data."""
    defaults = default_mesh_presentation()
    if advanced is None:
        advanced = isinstance(value, dict) and set(value) == set(defaults)
    if not advanced:
        defaults = {k: v for k, v in defaults.items() if k in MAIN_MESH_COMPONENTS}
    if not isinstance(value, dict) or set(value) != set(defaults):
        raise ValueError("Invalid mesh presentation components")
    result = {}
    for component, fields in defaults.items():
        settings = value[component]
        if not isinstance(settings, dict) or set(settings) != set(fields):
            raise ValueError("Invalid mesh component fields")
        result[component] = {
            "enabled": validate_mesh_boolean(settings["enabled"]),
            **validate_mesh_component_style(
                component, {k: v for k, v in settings.items() if k != "enabled"},
            ),
        }
    return result
