# -*- coding: utf-8 -*-
"""Session orchestration with isolated scene preparation and atomic publication."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, TypedDict

import numpy as np
from PyQt6.QtCore import QByteArray, QEventLoop, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QSplitter,
)

from ..session_io import (
    SessionCancelled,
    SessionError,
    checkpoint,
    checked_fingerprint,
    source_stamp,
    resolve_session_sources,
    SourceResolutionError,
    read_session,
    timestamped_path,
    write_session,
)
from ..session_state import (
    SOURCE_FIELDS,
    STATE_FIELDS,
    associate_sources,
    capture_state,
    dispose_sources,
    load_source,
    validate_state,
)

logger = logging.getLogger(__name__)
SAVE_FILE_FILTER = "TractEdit sessions (*.tractedit-session)"
OPEN_FILE_FILTER = "TractEdit sessions (*.tractedit-session *.tractedit-session-*)"
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


def _presentation(actor):
    return {
        "visible": bool(actor.GetVisibility()),
        "opacity": actor.GetProperty().GetOpacity(),
    }


class SessionWorker(QThread):
    """One cancellable operation; no Qt widgets or VTK work in run()."""

    progress = pyqtSignal(str)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.result = None
        self.error = None
        self.cancelled = False
        self.loader = None
        self._last_progress = 0.0

    def cancel(self):
        self.cancelled = True
        if self.loader is not None:
            self.loader.cancel()

    def set_loader(self, loader):
        self.loader = loader
        if loader is not None and self.cancelled:
            loader.cancel()

    def report(self, message):
        now = time.monotonic()
        if now - self._last_progress >= 0.05:
            self.progress.emit(message)
            self._last_progress = now

    def run(self):
        try:
            self.result = self.operation(self)
        except Exception as error:
            self.error = error


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
    return {
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
    }


def validate_views(views):
    if not isinstance(views, dict) or set(views) != {
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
            or len(key) > 16
            or any(type(i) is not int or i < 0 for i in key)
            or not isinstance(flags, tuple)
            or len(flags) != 2
            or any(type(flag) is not bool for flag in flags)
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


def prepare_session(document, worker):
    """Load and verify every dependency, disposing partial results on failure."""
    if not isinstance(document, dict) or set(document) != {"sources", "state", "views"}:
        raise SessionError("Invalid session document")
    validate_state(document["state"])
    validate_views(document["views"])
    sources = {}

    def cancelled():
        return worker.cancelled

    try:
        for role, record in document["sources"].items():
            receipt = getattr(document, "verified_sources", {}).get(role)
            stamp = source_stamp(record["path"])
            if receipt != (record, stamp):
                actual, stamp = checked_fingerprint(
                    record["path"], cancelled, worker.report
                )
                if actual != record:
                    raise SessionError(f"Source has changed: {record['path']}")
            sources[role] = load_source(
                role, record["path"], cancelled, worker.report, worker.set_loader
            )
            if source_stamp(record["path"]) != stamp:
                raise SessionError(f"Source changed during restore: {record['path']}")
        count = (
            len(sources["tractogram"]["streamlines"]) if "tractogram" in sources else 0
        )
        if count != document["state"]["streamline_count"]:
            raise SessionError("Tractogram count does not match the session")
        if document["state"]["roi_layers"] and "anatomy" not in sources:
            raise SessionError("ROI editing requires its anatomical reference")
        if "anatomy" in sources:
            shape = sources["anatomy"]["data"].shape
            for axis, size in zip("xyz", shape):
                index = document["views"]["slices"][axis]
                if index is None or index >= size:
                    raise SessionError("Slice is outside the anatomical preview")
        scalar = document["state"]["active_scalar_name"]
        if scalar is not None and scalar not in sources.get("tractogram", {}).get(
            "scalars", {}
        ):
            raise SessionError("Unknown active scalar")
        if document["state"]["odf_tunnel_indices"] is not None and "odf" not in sources:
            raise SessionError("Missing ODF source")
        if document["views"]["regions"] and "parcellation" not in sources:
            raise SessionError("Missing parcellation source")
        if set(document["views"]["roi_actors"]) - set(document["state"]["roi_layers"]):
            raise SessionError("Unknown ROI presentation")
        if "odf" in sources:
            from .. import odf_utils

            odf = sources["odf"]
            order = odf_utils.calculate_sh_order(odf["data"].shape[-1])
            odf["order"] = order
            for name, subdivisions in (("sphere", 3), ("tunnel_sphere", 2)):
                odf[name] = odf_utils.generate_symmetric_sphere(
                    radius=1.0, subdivisions=subdivisions
                )
                odf[name + "_basis"] = odf_utils.compute_sh_basis(
                    odf[name].vertices, order, basis_type="tournier07"
                )
            indices = document["state"]["odf_tunnel_indices"]
            if indices is not None and document["state"]["odf_tunnel_is_visible"]:
                worker.report("Reconstructing ODF tunnel...")
                streamlines = [
                    sources["tractogram"]["streamlines"][int(i)] for i in indices
                ]
                odf["tunnel"] = odf_utils.build_tunnel_odf_amplitudes(
                    odf["data"],
                    streamlines,
                    odf["affine"],
                    odf["tunnel_sphere_basis"],
                    dilation_iter=1,
                )
        checkpoint(cancelled)
        return document, sources
    except Exception:
        dispose_sources(sources)
        raise


def populate_window(
    mw, document, sources, cancelled=lambda: False, progress=lambda text: None
):
    """Build a hidden candidate window; the previous scene remains untouched."""
    from ..utils import ColorMode, signals_blocked
    from .state_manager import ActionType

    state, views = document["state"], document["views"]
    checkpoint(cancelled)
    for name in STATE_FIELDS:
        setattr(mw, name, state[name])
    if "tractogram" in sources:
        data = sources["tractogram"]
        mapping = {
            "tractogram_data": "streamlines",
            "streamline_bboxes": "bboxes",
            "original_trk_header": "header",
            "original_trk_affine": "affine",
            "original_trk_path": "path",
            "original_file_extension": "ext",
            "tractogram_reference_grid": "reference_grid",
            "scalar_data_per_point": "scalars",
            "data_per_streamline": "data_per_streamline",
        }
        for attribute, key in mapping.items():
            setattr(mw, attribute, data.get(key))
        mw.trx_file_reference = data.pop("trx_obj", None)
    if "anatomy" in sources:
        data = sources["anatomy"]
        mw.anatomical_image_data = data["data"]
        mw.anatomical_image_affine = data["affine"]
        mw.anatomical_image_path = data["path"]
        mw.anatomical_reference_grid = data["reference_grid"]
        mw.anatomical_mmap_image = data.pop("mmap_image", None)
    for role in ("parcellation", "odf"):
        if role in sources:
            for key in ("data", "affine", "path"):
                setattr(mw, role + "_" + key, sources[role][key])
    mw.current_color_mode = ColorMode[state["current_color_mode"]]
    for name, actions in state["history"].items():
        setattr(
            mw,
            name,
            [
                dict(action, action_type=ActionType[action["action_type"]])
                for action in actions
            ],
        )
    panel = mw.vtk_panel
    panel.sphere_params_per_roi = state["sphere_params"]
    panel.rectangle_params_per_roi = state["rectangle_params"]
    for key, layer in mw.roi_layers.items():
        layer["session_id"] = state["roi_ids"][key]
        layer["inv_affine"] = np.linalg.inv(layer["affine"])
        layer["T_main_to_roi"] = layer["inv_affine"] @ mw.anatomical_image_affine
    if mw.anatomical_image_data is not None:
        progress("Building anatomical views...")
        checkpoint(cancelled)
        panel.current_slice_indices = dict(views["slices"])
        panel.update_anatomical_slices()
        if panel.axial_slice_actor is None or panel.axial_slice_actor_2d is None:
            raise SessionError("Could not build anatomical slice actors")
        for key, layer in mw.roi_layers.items():
            progress(f"Building ROI: {key}")
            checkpoint(cancelled)
            panel.add_roi_layer(key, layer["data"], layer["affine"], render=False)
            if key not in panel.roi_slice_actors:
                raise SessionError(f"Could not build ROI actors: {key}")
    for name, presentation in views["slice_actors"].items():
        actor = getattr(panel, name)
        if actor is None:
            raise SessionError("Missing anatomical actor")
        actor.SetVisibility(presentation["visible"])
        actor.GetProperty().SetOpacity(presentation["opacity"])
    for key, actors in views["roi_actors"].items():
        for name, presentation in actors.items():
            actor = panel.roi_slice_actors.get(key, {}).get(name)
            if actor is None:
                raise SessionError(f"Missing ROI actor: {key}/{name}")
            actor.SetVisibility(presentation["visible"])
            actor.GetProperty().SetOpacity(presentation["opacity"])
    for key in mw.roi_layers:
        checkpoint(cancelled)
        if (
            mw.tractogram_data is not None
            and not mw.roi_manager.compute_roi_intersection(key)
        ):
            raise SessionError(f"Could not restore ROI filtering: {key}")
    mw._session_render_override = (
        mw._visibility_version + 1,
        mw.render_stride,
        state["rendered_indices"],
    )
    mw.roi_manager.apply_logic_filters()
    if mw.visible_indices != state["visible_indices"]:
        raise SessionError("Restored filters do not reproduce the saved visible fibers")
    if mw.visible_indices and panel.streamlines_actor is None:
        raise SessionError("Could not build streamline actor")
    panel.update_highlight()
    if mw._inversion_active:
        panel.streamlines_manager.update_invert_contour()
    mw.roi_manager.update_roi_visual_selection()
    if "odf" in sources:
        checkpoint(cancelled)
        odf = sources["odf"]
        mw.odf_sh_order = odf["order"]
        mw.odf_sphere = odf["sphere"]
        mw.odf_basis_matrix = odf["sphere_basis"]
        mw.odf_tunnel_sphere = odf["tunnel_sphere"]
        mw.odf_tunnel_basis = odf["tunnel_sphere_basis"]
        mw._odf_tunnel_indices = state["odf_tunnel_indices"]
        mw._odf_tunnel_source = (
            id(mw.odf_data),
            id(mw.tractogram_data),
            np.asarray(mw.odf_affine, dtype=np.float64).tobytes(),
        )
        mw._session_deferred_odf = (
            mw._odf_tunnel_indices is not None and not mw.odf_tunnel_is_visible
        )
        mw._session_deferred_odf_owner = deferred_odf_owner(mw)
        if mw._odf_tunnel_indices is not None and not mw._session_deferred_odf:
            amplitudes, affine = odf.pop("tunnel")
            extent = (
                None
                if amplitudes is None
                else tuple(v for n in amplitudes.shape[:3] for v in (0, n - 1))
            )
            panel.update_odf_actor(
                amplitudes, mw.odf_tunnel_sphere, affine, extent=extent
            )
            if amplitudes is not None and panel.odf_actor is None:
                raise SessionError("Could not build ODF actor")
            if panel.odf_actor is not None:
                panel.odf_actor.SetVisibility(mw.odf_tunnel_is_visible)
    if views["regions"]:
        from fury import actor

        mw._session_deferred_regions = {}
        for label, presentation in views["regions"].items():
            if mw.tractogram_data is not None and not (
                presentation["visible"] and presentation["attached"]
            ):
                mw._session_deferred_regions[label] = dict(presentation)
                continue
            progress(f"Building parcellation region: {label}")
            checkpoint(cancelled)
            region = actor.contour_from_roi(
                (mw.parcellation_data == label).astype(np.uint8),
                affine=mw.parcellation_affine,
                color=presentation["color"],
                opacity=presentation["opacity"],
            )
            if region is None:
                raise SessionError(f"Could not build region {label}")
            mw.parcellation_region_actors[label] = region
            region.SetVisibility(presentation["visible"])
            if presentation["attached"]:
                panel.scene.add(region)
        mw._parcellation_overlay_cached = True
        mw._parcellation_overlay_cache_key = (
            (mw.connectivity_manager._endpoint_cache_owner(), mw._visibility_version)
            if mw.tractogram_data is not None
            else None
        )
    radius = views["radius"]
    if radius is not None:
        panel._ensure_radius_actor_exists(radius["radius"], radius["center"])
        panel.radius_actor.SetVisibility(radius["visible"])
    for name, visible in views["crosshairs"].items():
        actor = getattr(panel, name, None)
        if actor is not None:
            actor.SetVisibility(visible)
    with signals_blocked(
        mw.skip_checkbox,
        mw.skip_spinbox,
        mw.brush_size_slider,
        mw.sphere_radius_spinbox,
    ):
        mw.skip_checkbox.setChecked(views["skip_checked"])
        mw.skip_spinbox.setValue(views["skip_value"])
        mw.skip_spinbox.setEnabled(views["skip_checked"])
        mw.brush_size_slider.setValue(mw.draw_brush_size)
        mw.sphere_radius_spinbox.setValue(views["sphere_radius"])
    panel.brush_size = mw.draw_brush_size
    mw.drawing_modes_manager.reset_all_drawing_modes()
    for field, method, action in (
        ("is_drawing_mode", "toggle_draw_mode", "draw_mode_action"),
        ("is_eraser_mode", "toggle_erase_mode", "erase_mode_action"),
        ("is_sphere_mode", "toggle_sphere_mode", "sphere_mode_action"),
        ("is_rectangle_mode", "toggle_rectangle_mode", "rectangle_mode_action"),
    ):
        if state[field]:
            getattr(mw.drawing_modes_manager, method)(True)
            with signals_blocked(getattr(mw, action)):
                getattr(mw, action).setChecked(True)
    checked_actions = {
        "geo_lines_action": not mw.render_as_tubes,
        "geo_tubes_action": mw.render_as_tubes,
        "color_default_action": mw.current_color_mode == ColorMode.DEFAULT,
        "color_orientation_action": mw.current_color_mode == ColorMode.ORIENTATION,
        "color_scalar_action": mw.current_color_mode == ColorMode.SCALAR,
        "view_odf_tunnel_action": mw.odf_tunnel_is_visible,
        "view_parcellation_action": mw._parcellation_overlay_visible,
        "auto_fill_action": mw.auto_fill_voxels,
    }
    for name, checked in checked_actions.items():
        with signals_blocked(getattr(mw, name)):
            getattr(mw, name).setChecked(checked)
    mw._update_scalar_range_widgets()
    mw._update_action_states()
    mw._update_bundle_info_display()
    mw._data_panel_update_pending = True
    mw._perform_data_panel_update()
    if getattr(mw, "_data_panel_debounce_timer", None) is not None:
        mw._data_panel_debounce_timer.stop()
    mw._data_panel_update_pending = False
    with signals_blocked(mw.data_tree_widget):
        for key, item in _tree_items(mw.data_tree_widget):
            if key == views["tree_current"]:
                mw.data_tree_widget.setCurrentItem(item)
            if key in views["tree"]:
                expanded, selected = views["tree"][key]
                item.setExpanded(expanded)
                item.setSelected(selected)
    with signals_blocked(mw.opacity_slider, mw.sphere_radius_spinbox):
        mw.opacity_slider.setEnabled(views["opacity_control"][0])
        mw.opacity_slider.setValue(views["opacity_control"][1])
        mw.sphere_radius_spinbox.setValue(views["sphere_radius"])
    mw.current_drawing_roi = state["current_drawing_roi"]
    mw.restoreGeometry(QByteArray.fromHex(views["geometry"].encode("ascii")))
    mw.restoreState(QByteArray.fromHex(views["layout"].encode("ascii")))
    for splitter, sizes in zip(
        mw.central_widget.findChildren(QSplitter), views["splitters"]
    ):
        splitter.setSizes(sizes)
    restore_cameras(mw, views)
    checkpoint(cancelled)


def restore_cameras(mw, views):
    for name, settings in views["cameras"].items():
        camera = getattr(mw.vtk_panel, name).GetActiveCamera()
        for key, value in settings.items():
            (
                getattr(camera, "Set" + key)(*value)
                if isinstance(value, (list, tuple))
                else getattr(camera, "Set" + key)(value)
            )


def deferred_odf_owner(mw):
    return (
        id(mw.odf_data),
        id(mw.tractogram_data),
        np.asarray(mw.odf_affine, dtype=np.float64).tobytes(),
        getattr(mw, "_tractogram_data_version", 0),
    )


def materialize_session_odf(mw):
    """Restore the saved hidden tunnel, using its original streamline IDs."""
    if not getattr(mw, "_session_deferred_odf", False):
        return True
    try:
        if mw._session_deferred_odf_owner != deferred_odf_owner(mw):
            raise SessionError("ODF inputs changed; recreate the tunnel")
        from .. import odf_utils

        def operation(worker):
            lines = [mw.tractogram_data[int(i)] for i in mw._odf_tunnel_indices]
            checkpoint(lambda: worker.cancelled)
            result = odf_utils.build_tunnel_odf_amplitudes(
                mw.odf_data,
                lines,
                mw.odf_affine,
                mw.odf_tunnel_basis,
                dilation_iter=1,
            )
            checkpoint(lambda: worker.cancelled)
            return result

        amplitudes, affine = SessionManager(mw)._run(
            "Restoring ODF tunnel...", operation, discard_result=lambda result: None
        )
        extent = (
            None
            if amplitudes is None
            else tuple(v for n in amplitudes.shape[:3] for v in (0, n - 1))
        )
        mw.vtk_panel.update_odf_actor(
            amplitudes, mw.odf_tunnel_sphere, affine, extent=extent
        )
        if amplitudes is not None and mw.vtk_panel.odf_actor is None:
            raise SessionError("Could not build ODF actor")
        mw._session_deferred_odf = False
        return True
    except SessionCancelled:
        return False
    except Exception as error:
        logger.exception("Deferred ODF restore failed")
        QMessageBox.critical(mw, "Session Error", str(error))
        return False


def materialize_session_regions(mw, labels):
    """Build requested saved meshes on the Qt thread; publish only on success."""
    pending = getattr(mw, "_session_deferred_regions", {})
    labels = [label for label in labels if label in pending]
    if not labels:
        return True
    if not mw.connectivity_manager.is_parcellation_overlay_cache_current():
        mw._session_deferred_regions = {}
        return False
    from fury import actor

    staged = {}
    try:
        for label in labels:
            presentation = pending[label]
            region = actor.contour_from_roi(
                (mw.parcellation_data == label).astype(np.uint8),
                affine=mw.parcellation_affine,
                color=presentation["color"],
                opacity=presentation["opacity"],
            )
            if region is None:
                raise SessionError(f"Could not build region {label}")
            region.SetVisibility(True)
            staged[label] = region
        mw.parcellation_region_actors.update(staged)
        for label in labels:
            pending.pop(label)
        return True
    except Exception as error:
        logger.exception("Deferred parcellation restore failed")
        QMessageBox.critical(mw, "Session Error", str(error))
        return False


class SessionManager:
    """Own dialog flow; scene data and the active session path live on MainWindow."""

    def __init__(self, main_window):
        self.mw = main_window

    def _idle(self):
        mw = self.mw
        if getattr(mw, "_session_busy", False) or any(
            worker.isRunning()
            or getattr(worker, "has_pending_result", False)
            or getattr(worker, "is_consuming_result", False)
            for worker in getattr(mw, "_background_workers", [])
        ):
            raise SessionError("Wait for the current operation to finish")
        if mw.vtk_panel.is_drawing_active:
            raise SessionError(
                "Finish the current ROI drawing before saving or opening a session"
            )

    def _run(self, title, operation, discard_result=None):
        mw = self.mw
        worker = SessionWorker(operation)
        dialog = QProgressDialog(title, "Cancel", 0, 0, mw)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.canceled.connect(worker.cancel)
        worker.progress.connect(dialog.setLabelText, Qt.ConnectionType.QueuedConnection)
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        mw._session_busy = True
        enabled = mw.isEnabled()
        mw.setEnabled(False)
        try:
            dialog.setEnabled(True)
            dialog.show()
            worker.start()
            loop.exec()
            worker.wait()
            if worker.error is not None:
                raise worker.error
            if worker.cancelled and discard_result is not None:
                discard_result(worker.result)
                raise SessionCancelled()
            return worker.result
        finally:
            dialog.close()
            dialog.deleteLater()
            mw.setEnabled(enabled)
            mw._session_busy = False
            worker.deleteLater()

    def save(self):
        try:
            self._idle()
            base = getattr(self.mw, "session_path", None)
            suggestion = timestamped_path(base or "session.tractedit-session")
            chosen, _ = QFileDialog.getSaveFileName(
                self.mw,
                "Save Session",
                str(suggestion),
                SAVE_FILE_FILTER,
            )
            if not chosen:
                return False
            selected = Path(chosen)
            destination = (
                selected
                if selected.name == suggestion.name and not selected.exists()
                else timestamped_path(chosen)
            )
            state, views = capture_state(self.mw), capture_views(self.mw)
            validate_state(state)
            validate_views(views)

            def operation(worker):
                def cancelled():
                    return worker.cancelled

                sources = associate_sources(
                    self.mw, cancelled, worker.report, worker.set_loader
                )
                document = {"sources": sources, "state": state, "views": views}
                write_session(destination, document, cancelled, worker.report)
                return str(destination)

            self.mw.session_path = self._run("Saving session...", operation)
            for key, identity in state["roi_ids"].items():
                self.mw.roi_layers[key]["session_id"] = identity
            self.mw.vtk_panel.update_status(
                f"Session saved: {Path(self.mw.session_path).name}"
            )
            return True
        except SessionCancelled:
            return False
        except Exception as error:
            logger.exception("Session save failed")
            QMessageBox.critical(self.mw, "Session Error", str(error))
            return False

    def open(self):
        candidate = None
        scene_dialog = None
        sources = {}
        try:
            self._idle()
            path, _ = QFileDialog.getOpenFileName(
                self.mw, "Open Session", "", OPEN_FILE_FILTER
            )
            if not path:
                return False
            if (
                any(
                    getattr(self.mw, field) is not None
                    for _, field in SOURCE_FIELDS.values()
                )
                or self.mw.roi_layers
            ):
                choice = QMessageBox.question(
                    self.mw,
                    "Replace Current Session",
                    "Save the current session before opening another?",
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Discard
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if choice == QMessageBox.StandardButton.Cancel or (
                    choice == QMessageBox.StandardButton.Save and not self.save()
                ):
                    return False

            def operation(worker):
                current = (
                    document
                    if document is not None
                    else read_session(
                        path, lambda: worker.cancelled, worker.report, verify=False
                    )
                )
                resolve_session_sources(
                    current, path, replacements, lambda: worker.cancelled, worker.report
                )
                return prepare_session(current, worker)

            document = None
            replacements = {}
            while True:
                try:
                    document, sources = self._run(
                        "Opening session...",
                        operation,
                        discard_result=lambda result: dispose_sources(result[1]),
                    )
                    break
                except SourceResolutionError as error:
                    document = error.document
                    for role, reason in error.issues.items():
                        replacement, _ = QFileDialog.getOpenFileName(
                            self.mw,
                            f"Locate {role} source — identical content required\n{reason}",
                            str(Path(path).resolve().parent),
                            "All files (*)",
                        )
                        if not replacement:
                            return False
                        replacements[role] = replacement
            from ..main_window import MainWindow

            self.mw._session_busy = True
            self.mw.setEnabled(False)
            scene_dialog = QProgressDialog(
                "Preparing scene...", "Cancel", 0, 0, self.mw
            )
            scene_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            scene_dialog.setAutoClose(False)
            scene_dialog.setAutoReset(False)
            scene_dialog.setMinimumDuration(0)
            scene_dialog.setEnabled(True)
            scene_dialog.show()

            def scene_progress(message):
                scene_dialog.setLabelText(message)
                QApplication.processEvents()

            candidate = MainWindow()
            candidate._session_busy = True
            populate_window(
                candidate,
                document,
                sources,
                cancelled=scene_dialog.wasCanceled,
                progress=scene_progress,
            )
            candidate.session_path = str(Path(path).resolve())
            candidate.show()
            restore_cameras(candidate, document["views"])
            candidate.vtk_panel._render_all()
            checkpoint(scene_dialog.wasCanceled)
            scene_dialog.close()
            candidate._session_busy = False
            QApplication.instance()._tractedit_session_window = candidate
            candidate.vtk_panel.update_status(f"Session restored: {Path(path).name}")
            candidate = None
            self.mw.hide()
            try:
                self.mw._cleanup_resources()
                self.mw._cleanup_vtk()
            except Exception:
                logger.exception("Could not fully retire the previous session window")
            self.mw.deleteLater()
            return True
        except SessionCancelled:
            return False
        except Exception as error:
            logger.exception("Session restore failed")
            if scene_dialog is not None:
                scene_dialog.close()
            QMessageBox.critical(self.mw, "Session Error", str(error))
            return False
        finally:
            if scene_dialog is not None:
                scene_dialog.close()
                scene_dialog.deleteLater()
            if candidate is not None:
                candidate.hide()
                candidate._cleanup_resources()
                candidate._cleanup_vtk()
                candidate.deleteLater()
            dispose_sources(sources)
            self.mw._session_busy = False
            self.mw.setEnabled(True)
