# -*- coding: utf-8 -*-
"""Explicit session state contracts and scientific source preparation."""

from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path
from typing import Any, NotRequired, TypedDict
import uuid

import numpy as np

from .io import SessionError, checkpoint, checked_fingerprint, source_stamp
from .mesh import (
    MESH_FIELDS,
    mesh_geometry_values,
    mesh_attribute_values,
    mesh_source_id,
    validate_mesh_records,
    validate_polyline_containers,
)

STATE_FIELDS = (
    "manual_visible_indices",
    "visible_indices",
    "selected_streamline_indices",
    "_inversion_active",
    "_inversion_keeper_indices",
    "selection_radius_3d",
    "render_stride",
    "_skip_user_disabled",
    "bundle_opacity",
    "image_opacity",
    "roi_layers",
    "roi_opacities",
    "roi_visibility",
    "roi_states",
    "bundle_is_visible",
    "image_is_visible",
    "render_as_tubes",
    "active_scalar_name",
    "scalar_min_val",
    "scalar_max_val",
    "scalar_data_min",
    "scalar_data_max",
    "scalar_range_initialized",
    "is_drawing_mode",
    "is_eraser_mode",
    "is_sphere_mode",
    "is_rectangle_mode",
    "current_drawing_roi",
    "manual_roi_counter",
    "draw_brush_size",
    "auto_fill_voxels",
    "parcellation_labels",
    "parcellation_region_visibility",
    "parcellation_main_labels",
    "parcellation_label_colors",
    "parcellation_region_states",
    "parcellation_connected_labels",
    "_parcellation_overlay_visible",
    "odf_tunnel_is_visible",
)
SOURCE_FIELDS = {
    "tractogram": ("original_trk_path", "tractogram_data"),
    "anatomy": ("anatomical_image_path", "anatomical_image_data"),
    "parcellation": ("parcellation_path", "parcellation_data"),
    "odf": ("odf_path", "odf_data"),
}


class SessionRoi(TypedDict):
    data: np.ndarray
    affine: np.ndarray
    color: NotRequired[tuple[float, float, float]]
    display_name: NotRequired[str]
    session_id: NotRequired[str]


class SessionHistoryAction(TypedDict):
    action_type: str
    roi_name: NotRequired[str]
    deleted_indices: NotRequired[set[int]]
    voxel_indices: NotRequired[np.ndarray]
    voxel_values: NotRequired[np.ndarray]
    data_snapshot: NotRequired[np.ndarray]
    sphere_params: NotRequired[dict[str, Any] | None]
    rectangle_params: NotRequired[dict[str, Any] | None]


def _validate_parameters(spheres, rectangles):
    for params in spheres:
        if not isinstance(params, dict):
            raise SessionError("Invalid sphere parameters")
        center = np.asarray(params.get("center"))
        radius = params.get("radius")
        if (
            center.shape != (3,)
            or center.dtype.kind not in "iuf"
            or not np.isfinite(center).all()
            or not isinstance(radius, (int, float))
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise SessionError("Invalid analytic sphere")
    for params in rectangles:
        if not isinstance(params, dict):
            raise SessionError("Invalid rectangle parameters")
        for name, shape in (
            ("start", (3,)),
            ("end", (3,)),
            ("corners", (8, 3)),
            ("voxel_min", (3,)),
            ("voxel_max", (3,)),
        ):
            if name not in ("start", "end") and name not in params:
                continue
            array = np.asarray(params.get(name))
            valid_shape = array.shape == shape or (
                name == "corners" and array.shape == (4, 3)
            )
            if (
                not valid_shape
                or array.dtype.kind not in "iuf"
                or not np.isfinite(array).all()
            ):
                raise SessionError("Invalid analytic rectangle")
        if params.get("view_type", "axial") not in ("axial", "coronal", "sagittal"):
            raise SessionError("Invalid rectangle view")


def capture_state(mw) -> dict[str, Any]:
    """Capture borrowed arrays while the caller prevents scene mutations."""
    state = {name: getattr(mw, name) for name in STATE_FIELDS}
    if getattr(mw, "scalar_display", None) is not None:
        state["scalar_display"] = dict(mw.scalar_display)
    selection = anatomical_display_selection(mw)
    if selection is not None:
        state["anatomical_display"] = selection
    state["current_color_mode"] = mw.current_color_mode.name
    state["history"] = {
        name: [
            dict(action, action_type=action["action_type"].name)
            for action in getattr(mw, name)
            if action["action_type"].name != "POLYLINE_EDIT"
        ]
        for name in ("unified_undo_stack", "unified_redo_stack")
    }
    panel = mw.vtk_panel
    state["sphere_params"] = panel.sphere_params_per_roi
    state["rectangle_params"] = panel.rectangle_params_per_roi
    state["odf_tunnel_indices"] = (
        getattr(mw, "_odf_tunnel_indices", None)
        if panel.odf_actor is not None
        or mw.odf_tunnel_is_visible
        or getattr(mw, "_session_deferred_odf", False)
        else None
    )
    if panel.odf_actor is not None and state["odf_tunnel_indices"] is None:
        raise SessionError("Rebuild the ODF tunnel once before saving this session")
    if state["odf_tunnel_indices"] is not None:
        owner = (
            id(mw.odf_data),
            id(mw.tractogram_data),
            np.asarray(mw.odf_affine, dtype=np.float64).tobytes(),
        )
        if getattr(mw, "_odf_tunnel_source", None) != owner:
            raise SessionError(
                "ODF inputs have changed; rebuild the tunnel before saving"
            )
    state["rendered_indices"] = (
        getattr(mw, "_rendered_streamline_indices", np.empty(0, dtype=np.int64))
        if mw.tractogram_data is not None
        else np.empty(0, dtype=np.int64)
    )
    ids = {}
    for key, layer in mw.roi_layers.items():
        ids[key] = layer.get("session_id", str(uuid.uuid4()))
    state["roi_ids"] = ids
    state["streamline_count"] = (
        len(mw.tractogram_data) if mw.tractogram_data is not None else 0
    )
    state["mesh_layers"] = [
        {**{field: layer[field] for field in MESH_FIELDS + ("texture", "polylines")
            if field in layer},
         "parent_id": layer.get("parent_id"),
         "presentation": {key: dict(value)
                          for key, value in layer["presentation"].items()}}
        for layer in getattr(mw, "mesh_layers", {}).values()
    ]
    if getattr(mw, "standalone_polylines", None):
        state["standalone_polylines"] = [
            dict(record) for record in mw.standalone_polylines
        ]
    from .surface_mapping import capture_recipe

    recipe = capture_recipe(getattr(mw, "surface_mapping_manager", None))
    if recipe is not None:
        state["surface_mapping_recipe"] = recipe
    return state


def _indices(values, count, name):
    if isinstance(values, set):
        valid = all(
            (type(v) is int or isinstance(v, np.integer)) and 0 <= v < count
            for v in values
        )
    elif isinstance(values, np.ndarray):
        valid = values.ndim == 1 and values.dtype.kind in "iu"
        valid = valid and (
            values.size == 0 or (values.min() >= 0 and values.max() < count)
        )
    else:
        valid = False
    if not valid:
        raise SessionError(f"Invalid streamline indices: {name}")


def validate_state(state):
    """Reject invalid references before any data reaches native consumers."""
    from ..reference_grid import validate_volume_geometry
    from ..utils import ColorMode

    required = set(STATE_FIELDS) | {
        "current_color_mode",
        "history",
        "sphere_params",
        "rectangle_params",
        "odf_tunnel_indices",
        "rendered_indices",
        "roi_ids",
        "streamline_count",
    }
    if not isinstance(state, dict) or set(state) - {
        "mesh_layers", "surface_mapping_recipe", "standalone_polylines",
        "anatomical_display", "scalar_display",
    } != required:
        raise SessionError("Invalid session state fields")
    if "scalar_display" in state:
        from ..tractogram_arrays import validate_scalar_display

        try:
            validate_scalar_display(state["scalar_display"])
        except ValueError as error:
            raise SessionError(str(error)) from error
    if "anatomical_display" in state:
        from ..image_volume import validate_display_selection

        try:
            validate_display_selection(state["anatomical_display"])
        except ValueError as error:
            raise SessionError(str(error)) from error
    validate_mesh_records(state.get("mesh_layers", []))
    validate_polyline_containers(state)
    for name in (
        "_inversion_active",
        "_skip_user_disabled",
        "bundle_is_visible",
        "image_is_visible",
        "render_as_tubes",
        "scalar_range_initialized",
        "is_drawing_mode",
        "is_eraser_mode",
        "is_sphere_mode",
        "is_rectangle_mode",
        "auto_fill_voxels",
        "_parcellation_overlay_visible",
        "odf_tunnel_is_visible",
    ):
        if type(state[name]) is not bool:
            raise SessionError(f"Invalid boolean setting: {name}")
    for name in (
        "scalar_min_val",
        "scalar_max_val",
        "scalar_data_min",
        "scalar_data_max",
    ):
        if not isinstance(state[name], (int, float)) or not np.isfinite(state[name]):
            raise SessionError("Invalid scalar range")
    if state["scalar_min_val"] > state["scalar_max_val"]:
        raise SessionError("Reversed scalar range")
    for name in ("draw_brush_size", "manual_roi_counter"):
        if type(state[name]) is not int or state[name] < 0:
            raise SessionError(f"Invalid integer setting: {name}")
    count = state["streamline_count"]
    if type(count) is not int or count < 0:
        raise SessionError("Invalid streamline count")
    from .surface_mapping import validate_recipe

    validate_recipe(state.get("surface_mapping_recipe"), state)
    for name in (
        "manual_visible_indices",
        "visible_indices",
        "selected_streamline_indices",
        "_inversion_keeper_indices",
        "rendered_indices",
    ):
        _indices(state[name], count, name)
    for name in (
        "manual_visible_indices",
        "visible_indices",
        "selected_streamline_indices",
        "_inversion_keeper_indices",
    ):
        if not isinstance(state[name], set):
            raise SessionError("Expected an index set")
    if not isinstance(state["rendered_indices"], np.ndarray):
        raise SessionError("Expected ordered rendering indices")
    if not state["visible_indices"] <= state["manual_visible_indices"]:
        raise SessionError("Visible fibers must survive manual deletions")
    if not set(state["rendered_indices"].tolist()) <= state["visible_indices"]:
        raise SessionError("Rendered fibers are not visible")
    if state["odf_tunnel_indices"] is not None:
        _indices(state["odf_tunnel_indices"], count, "ODF tunnel")
        if len(state["odf_tunnel_indices"]) > 26000:
            raise SessionError("ODF tunnel exceeds the application limit")
    if state["odf_tunnel_is_visible"] and state["odf_tunnel_indices"] is None:
        raise SessionError("Visible ODF tunnel has no construction inputs")
    if state["current_color_mode"] not in ColorMode.__members__:
        raise SessionError("Unknown color mode")
    if type(state["render_stride"]) is not int or state["render_stride"] < 1:
        raise SessionError("Invalid render stride")
    for name in ("selection_radius_3d", "draw_brush_size"):
        if not isinstance(state[name], (int, float)) or state[name] <= 0:
            raise SessionError(f"Invalid {name}")
    for name in ("bundle_opacity", "image_opacity"):
        if not isinstance(state[name], (int, float)) or not 0 <= state[name] <= 1:
            raise SessionError(f"Invalid {name}")
    layers = state["roi_layers"]
    if not isinstance(layers, dict) or any(
        not isinstance(k, str) or not k.strip() for k in layers
    ):
        raise SessionError("Invalid ROI names")
    if set(state["roi_ids"]) != set(layers) or len(
        set(state["roi_ids"].values())
    ) != len(layers):
        raise SessionError("Invalid ROI identities")
    for name in (
        "roi_opacities",
        "roi_visibility",
        "roi_states",
        "sphere_params",
        "rectangle_params",
    ):
        if not isinstance(state[name], dict) or set(state[name]) - set(layers):
            raise SessionError(f"Unknown ROI in {name}")
    if (
        state["current_drawing_roi"] is not None
        and state["current_drawing_roi"] not in layers
    ):
        raise SessionError("Unknown active ROI")
    for modes_by_layer in (state["roi_states"], state["parcellation_region_states"]):
        if not isinstance(modes_by_layer, dict):
            raise SessionError("Invalid layer filters")
        for modes in modes_by_layer.values():
            if (
                not isinstance(modes, dict)
                or set(modes) - {"include", "exclude", "select"}
                or any(type(value) is not bool for value in modes.values())
            ):
                raise SessionError("Invalid layer filter flags")
    for flags in (state["roi_visibility"], state["parcellation_region_visibility"]):
        if not isinstance(flags, dict) or any(
            type(value) is not bool for value in flags.values()
        ):
            raise SessionError("Invalid layer visibility")
    for name in (
        "parcellation_labels",
        "parcellation_label_colors",
        "parcellation_region_states",
        "parcellation_region_visibility",
    ):
        if not isinstance(state[name], dict) or any(
            not isinstance(label, (int, np.integer)) or not -(2**31) <= label < 2**31
            for label in state[name]
        ):
            raise SessionError("Invalid parcellation labels")
    modes = ("is_drawing_mode", "is_eraser_mode", "is_sphere_mode", "is_rectangle_mode")
    if sum(bool(state[name]) for name in modes) > 1:
        raise SessionError("Conflicting editing modes")
    for key, layer in layers.items():
        if not isinstance(layer, dict) or not {"data", "affine"} <= set(layer):
            raise SessionError("Invalid ROI layer")
        validate_volume_geometry(layer["data"], layer["affine"], "Session ROI")
        if any(size == 0 for size in layer["data"].shape):
            raise SessionError("Empty ROI grid")
        color = np.asarray(layer.get("color", (1, 0, 0)))
        if (
            color.shape != (3,)
            or not np.isfinite(color).all()
            or np.any((color < 0) | (color > 1))
        ):
            raise SessionError("Invalid ROI color")
        opacity = state["roi_opacities"].get(key, 1)
        if not isinstance(opacity, (int, float)) or not 0 <= opacity <= 1:
            raise SessionError("Invalid ROI opacity")
    _validate_parameters(
        state["sphere_params"].values(), state["rectangle_params"].values()
    )
    history = state["history"]
    if set(history) != {"unified_undo_stack", "unified_redo_stack"}:
        raise SessionError("Invalid history stacks")
    for actions in history.values():
        if not isinstance(actions, list) or len(actions) > 20:
            raise SessionError("Invalid history length")
        for action in actions:
            if action.get("action_type") == "STREAMLINE_DELETION":
                _indices(action.get("deleted_indices"), count, "history")
            elif action.get("action_type") == "ROI_MODIFICATION":
                key = action.get("roi_name")
                if key not in layers:
                    raise SessionError("History refers to an unknown ROI")
                data = layers[key]["data"]
                sphere, rectangle = (
                    action.get("sphere_params"),
                    action.get("rectangle_params"),
                )
                _validate_parameters(
                    [] if sphere is None else [sphere],
                    [] if rectangle is None else [rectangle],
                )
                if "data_snapshot" in action:
                    snapshot = action["data_snapshot"]
                    if (
                        not isinstance(snapshot, np.ndarray)
                        or snapshot.shape != data.shape
                        or snapshot.dtype != data.dtype
                    ):
                        raise SessionError("Invalid ROI history snapshot")
                else:
                    indices, values = (
                        action.get("voxel_indices"),
                        action.get("voxel_values"),
                    )
                    _indices(indices, data.size, "ROI history")
                    if (
                        not isinstance(indices, np.ndarray)
                        or action.get("data_shape", data.shape) != data.shape
                    ):
                        raise SessionError("Invalid ROI patch shape")
                    if (
                        not isinstance(values, np.ndarray)
                        or values.shape != indices.shape
                        or values.dtype != data.dtype
                    ):
                        raise SessionError("Invalid ROI history patch")
            else:
                raise SessionError("Unknown history action")


def _same(first, second, cancelled):
    checkpoint(cancelled)
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        first, second = np.asarray(first), np.asarray(second)
        if first.shape != second.shape or first.dtype != second.dtype:
            return False
        iterator = np.nditer(
            [first, second],
            flags=["external_loop", "buffered", "zerosize_ok"],
            order="C",
            buffersize=262144,
        )
        for left, right in iterator:
            checkpoint(cancelled)
            if left.tobytes() != right.tobytes():
                return False
        return True
    if hasattr(first, "_data") and hasattr(first, "_lengths"):
        if not hasattr(second, "_data") or not _same(
            first._lengths, second._lengths, cancelled
        ):
            return False
        if (
            _same(first._offsets, second._offsets, cancelled)
            and first._data.shape == second._data.shape
        ):
            return _same(first._data, second._data, cancelled)
        return all(_same(a, b, cancelled) for a, b in zip(first, second))
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(_same(first[k], second[k], cancelled) for k in first)
        )
    if hasattr(first, "__dataclass_fields__"):
        return type(first) is type(second) and all(
            _same(getattr(first, f.name), getattr(second, f.name), cancelled)
            for f in fields(first)
            if f.name != "provenance"
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_same(a, b, cancelled) for a, b in zip(first, second))
        )
    return first == second


def anatomical_display_selection(mw):
    image = getattr(mw, "anatomical_mmap_image", None)
    if image is not None and len(getattr(image, "source_shape", ())) == 4:
        return {"volume_index": image.volume_index, "rgb": image.rgb}
    return None


def load_source(
    role, path, cancelled, progress, set_loader=lambda loader: None,
    *, anatomical_display=None,
):
    """Run existing scientific loaders without touching the live window."""
    if mesh_source_id(role) is not None:
        from ..mesh_io import MeshLoadCancelled, load_mesh
        from .io import SessionCancelled

        progress(f"Loading surface: {Path(path).name}")
        try:
            return load_mesh(path, cancel_check=cancelled)
        except MeshLoadCancelled as error:
            raise SessionCancelled() from error
    if role not in SOURCE_FIELDS:
        raise SessionError(f"Unknown source role: {role}")
    from PyQt6.QtCore import Qt
    from .. import file_io

    if role in ("tractogram", "anatomy"):
        cls = (
            file_io.StreamlineLoaderThread
            if role == "tractogram"
            else file_io.AnatomicalImageLoaderThread
        )
        loader = cls(path, **(anatomical_display or {})) if role == "anatomy" else cls(path)
        result, errors = [], []
        loader.finished.connect(result.append, Qt.ConnectionType.DirectConnection)
        loader.error.connect(errors.append, Qt.ConnectionType.DirectConnection)
        loader.progress.connect(
            lambda percent, text: progress(text), Qt.ConnectionType.DirectConnection
        )
        set_loader(loader)
        try:
            checkpoint(cancelled)
            loader.run()
            checkpoint(cancelled)
            if errors or not result:
                raise SessionError(errors[0] if errors else f"Could not load {path}")
            payload = result[0]
            if role == "tractogram":
                loader.take_trx_owner(payload)
            else:
                loader.take_result(payload)
            return payload
        finally:
            loader.discard_pending_result()
            set_loader(None)
    import nibabel as nib
    from ..reference_grid import validate_volume_geometry, validate_affine

    image = nib.load(path)
    if role == "parcellation":
        from ..logic.connectivity import _read_atlas_c_int32

        validate_volume_geometry(image.dataobj, image.affine, "Parcellation")
        data = _read_atlas_c_int32(image.dataobj)
    else:
        from ..odf_utils import calculate_sh_order

        validate_affine(image.affine, "ODF")
        if len(image.shape) != 4:
            raise SessionError("ODF must have four dimensions")
        calculate_sh_order(image.shape[-1])
        data = image.get_fdata()
    checkpoint(cancelled)
    result = {"data": data, "affine": image.affine, "path": path}
    if role == "parcellation":
        from ..reference_grid import mask_spatial_metadata

        result["spatial"] = mask_spatial_metadata(image)
    return result


def dispose_sources(sources):
    for role, result in sources.items():
        owner = result.get("trx_obj")
        if owner is not None:
            owner.close()
            result["trx_obj"] = None
        mmap = result.get("mmap_image")
        if mmap is not None:
            mmap.clear_cache()
            result["mmap_image"] = None
        if mesh_source_id(role) is not None:
            result.pop("polydata", None)
            result.pop("texture_preparation", None)


def associate_sources(mw, cancelled, progress, set_loader=lambda loader: None):
    """Prove external files still reproduce the live immutable scientific inputs."""
    records = {}
    cache = getattr(mw, "_session_source_associations", {})
    for identity, layer in getattr(mw, "mesh_layers", {}).items():
        if identity != layer["id"]:
            raise SessionError("Mesh registry UUID does not match its layer")
        role, path = "mesh:" + identity, layer["path"]
        record, stamp = checked_fingerprint(path, cancelled, progress)
        if "texture" in layer and record["sha256"] != layer.get("source_sha256"):
            raise SessionError("OBJ changed since texture preparation; reload the mesh")
        if layer.get("source_sha256", record["sha256"]) != record["sha256"] or any(
            line["region"]["digest"] != record["sha256"]
            for line in layer.get("polylines", ()) if "region" in line
        ):
            raise SessionError("Mesh source changed since loading")
        signature = _content_digest(_source_values(mw, role), cancelled)
        if cache.get(role) != (record, signature):
            result = load_source(role, path, cancelled, progress, set_loader)
            try:
                expected_values = [
                    mesh_geometry_values(result["polydata"]),
                    result["space_metadata"],
                    mesh_attribute_values(result["polydata"]),
                ]
                if not _same(
                    _source_values(mw, role),
                    expected_values,
                    cancelled,
                ):
                    raise SessionError(
                        f"The {role} source no longer reproduces the loaded data: "
                        f"{path}"
                    )
            finally:
                dispose_sources({role: result})
        if source_stamp(path) != stamp:
            raise SessionError(f"Source changed during session save: {path}")
        records[role] = record
        cache[role] = (record.copy(), signature)
        if "texture" in layer:
            from .mesh_texture import verify_texture_assets

            records.update(verify_texture_assets(layer, cancelled, progress))
    for role, (path_name, data_name) in SOURCE_FIELDS.items():
        if getattr(mw, data_name) is None:
            continue
        path = getattr(mw, path_name)
        if not path:
            raise SessionError(f"No source file for {role}")
        record, stamp = checked_fingerprint(path, cancelled, progress)
        progress(f"Checking loaded {role} data...")
        signature = _content_digest(_source_values(mw, role), cancelled)
        if cache.get(role) == (record, signature):
            if source_stamp(path) != stamp:
                raise SessionError(f"Source changed during session save: {path}")
            records[role] = record
            continue
        options = {}
        if role == "anatomy":
            selection = anatomical_display_selection(mw)
            if selection is not None:
                options["anatomical_display"] = selection
        result = load_source(
            role, str(Path(path).resolve()), cancelled, progress, set_loader, **options
        )
        try:
            pairs = [
                (
                    getattr(mw, data_name),
                    result["streamlines" if role == "tractogram" else "data"],
                )
            ]
            if role == "tractogram":
                pairs.extend(
                    [
                        (mw.scalar_data_per_point or {}, result.get("scalars") or {}),
                        (
                            mw.data_per_streamline or {},
                            result.get("data_per_streamline") or {},
                        ),
                        (mw.original_trk_header, result.get("header")),
                        (mw.original_trk_affine, result.get("affine")),
                        (mw.tractogram_reference_grid, result.get("reference_grid")),
                    ]
                )
                owner = mw.trx_file_reference
                loaded_owner = result.get("trx_obj")
                for field in ("groups", "data_per_group"):
                    pairs.append(
                        (
                            getattr(owner, field, {}) or {},
                            getattr(loaded_owner, field, {}) or {},
                        )
                    )
            else:
                affine_name = (
                    "anatomical_image_affine" if role == "anatomy" else role + "_affine"
                )
                pairs.append((getattr(mw, affine_name), result["affine"]))
                if role == "anatomy":
                    pairs.append(
                        (mw.anatomical_reference_grid, result["reference_grid"])
                    )
            if not all(_same(a, b, cancelled) for a, b in pairs):
                raise SessionError(
                    f"The {role} source no longer reproduces the loaded data: {path}"
                )
            if source_stamp(path) != stamp:
                raise SessionError(f"Source changed during session save: {path}")
            records[role] = record
            cache[role] = (record.copy(), signature)
        finally:
            dispose_sources({role: result})
    mw._session_source_associations = cache
    return records


def _source_values(mw, role):
    """Exactly the immutable input values compared by associate_sources."""
    identity = mesh_source_id(role)
    if identity is not None:
        layer = mw.mesh_layers[identity]
        values = [
            mesh_geometry_values(layer["polydata"]),
            layer["coordinate_provenance"]["metadata"],
            mesh_attribute_values(layer["polydata"]),
        ]
        return values
    values = [getattr(mw, SOURCE_FIELDS[role][1])]
    if role == "tractogram":
        values.extend(
            [
                mw.scalar_data_per_point or {},
                mw.data_per_streamline or {},
                mw.original_trk_header,
                mw.original_trk_affine,
                mw.tractogram_reference_grid,
                getattr(mw.trx_file_reference, "groups", {}) or {},
                getattr(mw.trx_file_reference, "data_per_group", {}) or {},
            ]
        )
    else:
        values.append(
            getattr(
                mw, "anatomical_image_affine" if role == "anatomy" else role + "_affine"
            )
        )
        if role == "anatomy":
            values.append(mw.anatomical_reference_grid)
            selection = anatomical_display_selection(mw)
            if selection is not None:
                values.append(selection)
    return values


def _content_digest(value, cancelled):
    """Bounded content proof, including in-place edits without version increments."""
    digest = hashlib.sha256()

    def token(value):
        data = repr(value).encode("utf8")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

    def visit(item):
        checkpoint(cancelled)
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise SessionError("Object arrays cannot identify scientific sources")
            token(("array", item.shape, item.dtype.descr))
            for chunk in np.nditer(
                item,
                flags=["external_loop", "buffered", "zerosize_ok"],
                order="C",
                buffersize=262144,
            ):
                checkpoint(cancelled)
                digest.update(chunk.tobytes())
        elif hasattr(item, "_data") and hasattr(item, "_lengths"):
            token("ArraySequence")
            visit(item._lengths)
            visit(item._offsets)
            visit(item._data)
        elif isinstance(item, dict):
            token(("dict", len(item)))
            for key, entry in item.items():
                token(key)
                visit(entry)
        elif hasattr(item, "__dataclass_fields__"):
            token(type(item).__qualname__)
            for field in fields(item):
                if field.name != "provenance":
                    token(field.name)
                    visit(getattr(item, field.name))
        elif isinstance(item, (list, tuple)):
            token((type(item).__name__, len(item)))
            for entry in item:
                visit(entry)
        else:
            token((type(item).__name__, item))

    visit(value)
    return digest.hexdigest()
