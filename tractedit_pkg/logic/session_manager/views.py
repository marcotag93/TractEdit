# -*- coding: utf-8 -*-
"""Session view capture, validation, camera and tree restoration."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter

from ...mesh.presentation import MESH_TREE_COMPONENTS
from ...session.io import SessionError
from ...session.mesh import MAX_MESH_LAYERS, _valid_flag_tuple

SCENES = ("scene", "axial_scene", "coronal_scene", "sagittal_scene")
CAMERA_PROPERTIES = (
    "Position",
    "FocalPoint",
    "ViewUp",
    "ClippingRange",
    "ParallelProjection",
    "ParallelScale",
    "ViewAngle",
    "WindowCenter",
    "ViewShear",
)
SLICE_ACTORS = (
    "axial_slice_actor",
    "coronal_slice_actor",
    "sagittal_slice_actor",
    "axial_slice_actor_2d",
    "coronal_slice_actor_2d",
    "sagittal_slice_actor_2d",
)


class SessionViews(TypedDict):
    cameras: dict[str, dict[str, Any]]
    slices: dict[str, int | None]
    crosshairs: dict[str, bool]
    slice_actors: dict[str, dict[str, Any]]
    roi_actors: dict[str, dict[str, dict[str, Any]]]
    radius: dict[str, Any] | None
    regions: dict[int, dict[str, Any]]
    geometry: str
    layout: str
    splitters: list[list[int]]
    tree: dict[tuple[int, ...], tuple[bool, bool]]
    tree_current: tuple[int, ...] | None
    skip_checked: bool
    skip_value: int
    sphere_radius: float
    opacity_control: tuple[bool, int]
    mesh_tree: NotRequired[dict[str, Any]]
    standalone_polyline_tree: NotRequired[dict[str, Any]]


def _presentation(actor):
    return {
        "visible": bool(actor.GetVisibility()),
        "opacity": actor.GetProperty().GetOpacity(),
    }


def _tree_items(tree):
    def visit(item, path):
        for index in range(item.childCount()):
            child = item.child(index)
            key = path + (index,)
            yield key, child
            yield from visit(child, key)

    yield from visit(tree.invisibleRootItem(), ())


def capture_views(mw) -> SessionViews:
    panel = mw.vtk_panel
    camera_states = {}
    for name in SCENES:
        camera = getattr(panel, name).GetActiveCamera()
        camera_states[name] = {
            key: getattr(camera, "Get" + key)() for key in CAMERA_PROPERTIES
        }
    radius = panel.radius_actor
    regions = dict(getattr(mw, "_session_deferred_regions", {}))
    for label, actor in mw.parcellation_region_actors.items():
        regions[label] = {
            "color": actor.GetProperty().GetColor(),
            "opacity": actor.GetProperty().GetOpacity(),
            "visible": bool(actor.GetVisibility()),
            "attached": bool(panel.scene.HasViewProp(actor)),
        }
    crosshairs = {
        name: bool(getattr(panel, name) and getattr(panel, name).GetVisibility())
        for name in (
            "axial_crosshair_actor",
            "coronal_crosshair_actor",
            "sagittal_crosshair_actor",
        )
    }
    mesh_tree = {
        "items": {}, "current": None, "expanded": True,
        "components": {
            (mesh_id, component): (False, False)
            for mesh_id in mw.mesh_layers for component in MESH_TREE_COMPONENTS
        },
        "current_component": None,
    }
    for mesh_id, layer in mw.mesh_layers.items():
        if layer.get("strip_count"):
            mesh_tree["components"][mesh_id, "information_strips"] = (False, False)
    standalone_tree = {
        "items": {
            record["id"]: (False, False)
            for record in getattr(mw, "standalone_polylines", [])
        },
        "current": None, "expanded": True, "selected": False, "current_group": False,
    }
    mesh_group = next((
        mw.data_tree_widget.topLevelItem(i)
        for i in range(mw.data_tree_widget.topLevelItemCount())
        if mw.data_tree_widget.topLevelItem(i).text(0) == "Meshes"
    ), None)
    if mesh_group is not None:
        mesh_tree["expanded"] = mesh_group.isExpanded()
    for _, item in _tree_items(mw.data_tree_widget):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, dict) and data.get("type") == "mesh":
            mesh_tree["items"][data["id"]] = (item.isExpanded(), item.isSelected())
            if item is mw.data_tree_widget.currentItem():
                mesh_tree["current"] = data["id"]
        elif isinstance(data, dict) and data.get("type") in (
            "mesh_polyline", "mesh_polylines", "mesh_polygon", "mesh_polygons",
        ):
            key = (data["id"], data["component"])
            annotations = mesh_tree.setdefault("annotations", {})
            annotations[key] = (
                item.isExpanded(), item.isSelected(),
                item is mw.data_tree_widget.currentItem(),
            )
        elif isinstance(data, dict) and data.get("type") == "standalone_polylines":
            standalone_tree["expanded"] = item.isExpanded()
            standalone_tree["selected"] = item.isSelected()
            standalone_tree["current_group"] = (
                item is mw.data_tree_widget.currentItem()
            )
        elif isinstance(data, dict) and data.get("type") == "standalone_polyline":
            standalone_tree["items"][data["id"]] = (
                item.isExpanded(), item.isSelected(),
            )
            if item is mw.data_tree_widget.currentItem():
                standalone_tree["current"] = data["id"]
    manager = mw.data_panel_manager
    mesh_tree["components"].update({
        key: flags for key, flags in manager.component_states().items()
        if key[0] in mw.mesh_layers and (key[1] != "appearance" or not flags[0])
    })
    inspector = manager.mesh_inspector
    for _, item in _tree_items(inspector):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and "component" in data:
            key = (data["id"], data["component"])
            if (key[1] != "appearance" or not item.isExpanded()
                    or item.isSelected() or item is inspector.currentItem()):
                mesh_tree["components"][key] = (item.isExpanded(), item.isSelected())
            if item is inspector.currentItem():
                mesh_tree["current_component"] = key
                mesh_tree["current"] = None
    result = {
        "cameras": camera_states,
        "slices": dict(panel.current_slice_indices),
        "crosshairs": crosshairs,
        "slice_actors": {
            name: _presentation(getattr(panel, name))
            for name in SLICE_ACTORS
            if getattr(panel, name) is not None
        },
        "roi_actors": {
            key: {
                name: _presentation(actor)
                for name, actor in actors.items()
                if hasattr(actor, "GetProperty") and hasattr(actor, "GetVisibility")
            }
            for key, actors in panel.roi_slice_actors.items()
        },
        "radius": (
            None
            if radius is None
            else {
                "center": radius.GetPosition(),
                "radius": panel.current_radius_actor_radius,
                "visible": bool(radius.GetVisibility()),
            }
        ),
        "regions": regions,
        "geometry": bytes(mw.saveGeometry()).hex(),
        "layout": bytes(mw.saveState()).hex(),
        "splitters": [
            splitter.sizes() for splitter in mw.central_widget.findChildren(QSplitter)
        ],
        "tree": {
            key: (item.isExpanded(), item.isSelected())
            for key, item in _tree_items(mw.data_tree_widget)
        },
        "tree_current": next(
            (
                key
                for key, item in _tree_items(mw.data_tree_widget)
                if item is mw.data_tree_widget.currentItem()
            ),
            None,
        ),
        "skip_checked": mw.skip_checkbox.isChecked(),
        "skip_value": mw.skip_spinbox.value(),
        "sphere_radius": mw.sphere_radius_spinbox.value(),
        "opacity_control": (mw.opacity_slider.isEnabled(), mw.opacity_slider.value()),
        "mesh_tree": mesh_tree,
    }
    if getattr(mw, "standalone_polylines", None):
        result["standalone_polyline_tree"] = standalone_tree
    return result


def validate_views(views):
    if not isinstance(views, dict) or set(views) - {
        "mesh_tree", "standalone_polyline_tree",
    } != {
        "cameras",
        "slices",
        "crosshairs",
        "radius",
        "regions",
        "geometry",
        "layout",
        "splitters",
        "tree",
        "skip_checked",
        "skip_value",
        "sphere_radius",
        "slice_actors",
        "roi_actors",
        "tree_current",
        "opacity_control",
    }:
        raise SessionError("Invalid view fields")
    if set(views["cameras"]) != set(SCENES):
        raise SessionError("Four cameras are required")
    lengths = {
        "Position": 3,
        "FocalPoint": 3,
        "ViewUp": 3,
        "ClippingRange": 2,
        "WindowCenter": 2,
        "ViewShear": 3,
    }
    for camera in views["cameras"].values():
        if set(camera) != set(CAMERA_PROPERTIES):
            raise SessionError("Invalid camera properties")
        for key, value in camera.items():
            array = np.asarray(value)
            expected = (lengths[key],) if key in lengths else ()
            if (
                array.shape != expected
                or array.dtype.kind not in "biuf"
                or not np.isfinite(array).all()
            ):
                raise SessionError("Invalid camera values")
        if camera["ParallelScale"] <= 0 or not 0 < camera["ViewAngle"] < 180:
            raise SessionError("Invalid camera scale")
        if (
            np.linalg.norm(np.asarray(camera["Position"]) - camera["FocalPoint"]) == 0
            or np.linalg.norm(camera["ViewUp"]) == 0
        ):
            raise SessionError("Degenerate camera")
    if set(views["slices"]) != {"x", "y", "z"}:
        raise SessionError("Invalid slice axes")
    for value in views["slices"].values():
        if value is not None and (type(value) is not int or value < 0):
            raise SessionError("Invalid slice index")
    if type(views["skip_value"]) is not int or not 0 <= views["skip_value"] <= 100:
        raise SessionError("Invalid skip percentage")
    if (
        type(views["skip_checked"]) is not bool
        or not isinstance(views["sphere_radius"], (int, float))
        or not np.isfinite(views["sphere_radius"])
        or views["sphere_radius"] <= 0
    ):
        raise SessionError("Invalid toolbar settings")
    if len(views["splitters"]) > 8:
        raise SessionError("Too many splitters")
    for sizes in views["splitters"]:
        if (
            not isinstance(sizes, list)
            or len(sizes) > 8
            or any(type(size) is not int or not 0 <= size <= 100000 for size in sizes)
        ):
            raise SessionError("Invalid splitter dimensions")
    for key, flags in views["tree"].items():
        if (
            not isinstance(key, tuple)
            or len(key) > MAX_MESH_LAYERS + 16
            or any(type(i) is not int or i < 0 for i in key)
            or not _valid_flag_tuple(flags)
        ):
            raise SessionError("Invalid data panel state")
    if views["tree_current"] is not None and views["tree_current"] not in views["tree"]:
        raise SessionError("Unknown active data panel item")
    opacity = views["opacity_control"]
    if (
        not isinstance(opacity, tuple)
        or len(opacity) != 2
        or type(opacity[0]) is not bool
        or type(opacity[1]) is not int
        or not 0 <= opacity[1] <= 100
    ):
        raise SessionError("Invalid opacity control")
    for name in ("geometry", "layout"):
        if not isinstance(views[name], str) or len(views[name]) > 131072:
            raise SessionError("Invalid window layout")
        bytes.fromhex(views[name])
    for params in views["regions"].values():
        color = np.asarray(params["color"])
        if (
            color.shape != (3,)
            or not np.isfinite(color).all()
            or np.any((color < 0) | (color > 1))
            or not 0 <= params["opacity"] <= 1
        ):
            raise SessionError("Invalid region presentation")
    if set(views["slice_actors"]) - set(SLICE_ACTORS):
        raise SessionError("Unknown anatomical actor")
    for presentation in [views["slice_actors"], *views["roi_actors"].values()]:
        for params in presentation.values():
            if (
                set(params) != {"visible", "opacity"}
                or type(params["visible"]) is not bool
                or not isinstance(params["opacity"], (int, float))
                or not 0 <= params["opacity"] <= 1
            ):
                raise SessionError("Invalid actor presentation")
    radius = views["radius"]
    if radius is not None:
        center = np.asarray(radius["center"])
        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(radius["radius"])
            or radius["radius"] <= 0
        ):
            raise SessionError("Invalid selection sphere")


def restore_mesh_tree(mw, views):
    """Apply group expansion after Qt scrolls to the restored current item."""
    from ...utils import signals_blocked

    tree = views.get("mesh_tree", {})
    standalone_tree = views.get("standalone_polyline_tree", {})
    items = []
    components = tree.get("components", {})
    mesh_group = next((
        mw.data_tree_widget.topLevelItem(i)
        for i in range(mw.data_tree_widget.topLevelItemCount())
        if mw.data_tree_widget.topLevelItem(i).text(0) == "Meshes"
    ), None)
    component = tree.get("current_component")
    owner = component[0] if component else tree.get("current")
    annotations = tree.get("annotations", {})
    with signals_blocked(mw.data_tree_widget):
        legacy_paths = {}
        for path, item in _tree_items(mw.data_tree_widget):
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(data, dict):
                continue
            if data.get("type") == "mesh":
                flags = tree.get("items", {}).get(data["id"])
                current = data["id"] == owner
                if not annotations:
                    # The old tree placed seven feature rows before the child meshes.
                    if item.parent() is mesh_group:
                        if (
                            path + (item.childCount(),) in views["tree"]
                            or item.childCount() <= 4 and path + (4,) in views["tree"]
                        ):
                            legacy_paths[data["id"]] = path
                    else:
                        parent_id = mw.mesh_layers[data["id"]]["parent_id"]
                        if parent_id in legacy_paths:
                            legacy_paths[data["id"]] = legacy_paths[parent_id] + (
                                7 + path[-1],
                            )
                if current and component and components[component][1]:
                    flags = (flags[0], True)
            elif data.get("type") in (
                "mesh_polyline", "mesh_polylines", "mesh_polygon", "mesh_polygons",
            ):
                key = (data["id"], data["component"])
                if key in annotations:
                    expanded, selected, current = annotations[key]
                    flags = (expanded, selected)
                elif (
                    data.get("type") == "mesh_polygons"
                    and (data["id"], "polylines") in annotations
                ):
                    expanded, selected, current = annotations[(data["id"], "polylines")]
                    has_plain = any(
                        "region" not in line
                        for line in mw.mesh_layers[data["id"]].get("polylines", ())
                    )
                    flags = (expanded, False if has_plain else selected)
                    current = False if has_plain else current
                else:
                    if data["id"] in legacy_paths:
                        path = legacy_paths[data["id"]] + (4,)
                        if data["type"] in ("mesh_polyline", "mesh_polygon"):
                            path += (item.parent().indexOfChild(item),)
                    flags = views["tree"].get(path)
                    current = views["tree_current"] == path
            elif data.get("type") == "standalone_polyline":
                flags = standalone_tree.get("items", {}).get(data["id"])
                current = data["id"] == standalone_tree.get("current")
            elif data.get("type") == "standalone_polylines":
                flags = (
                    standalone_tree.get("expanded", True),
                    standalone_tree.get("selected", False),
                )
                current = standalone_tree.get("current_group", False)
            else:
                continue
            if flags is not None:
                items.append((item, flags))
            if current:
                mw.data_tree_widget.setCurrentItem(item)
        for item, (_, selected) in items:
            item.setSelected(selected)
        for item, (expanded, _) in items:
            item.setExpanded(expanded)
        if mesh_group is not None:
            mesh_group.setExpanded(tree.get("expanded", True))
    manager = mw.data_panel_manager
    manager.on_data_item_selected()
    manager._component_states = dict(components)
    inspector = manager.mesh_inspector
    with signals_blocked(inspector):
        restored = []
        for _, item in _tree_items(inspector):
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and "component" in data:
                key = (data["id"], data["component"])
                if key == component:
                    inspector.setCurrentItem(item)
                if key in components:
                    restored.append((item, components[key]))
        for item, (expanded, selected) in restored:
            item.setExpanded(expanded)
            item.setSelected(selected)
    manager.update_mesh_properties()


def restore_cameras(mw, views):
    for name, settings in views["cameras"].items():
        camera = getattr(mw.vtk_panel, name).GetActiveCamera()
        for key, value in settings.items():
            (
                getattr(camera, "Set" + key)(*value)
                if isinstance(value, (list, tuple))
                else getattr(camera, "Set" + key)(value)
            )
