# -*- coding: utf-8 -*-
"""Session source preparation, candidate window population and deferred restores."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QByteArray
from PyQt6.QtWidgets import QSplitter

from .. import session_manager
from ...render_batch import batch_render
from ...session.io import (
    SessionCancelled,
    SessionError,
    checked_fingerprint,
    checkpoint,
    source_stamp,
    validate_sources,
)
from ...session.mesh import restored_mesh_layer, validate_mesh_document
from ...session.mesh_texture import texture_asset_role, texture_references
from ...session.state import STATE_FIELDS

logger = logging.getLogger(__name__)


def _update_odf_tunnel_actor(mw, amplitudes, affine):
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


def _build_region_contour(mw, label, presentation, bounds):
    from fury import actor

    if bounds is None:
        bounds = session_manager.region_contour_bounds(
            mw.parcellation_data, mw.parcellation_affine
        )
    mask, placement = session_manager.region_contour_input(
        mw.parcellation_data, mw.parcellation_affine, label, bounds,
    )
    region = actor.contour_from_roi(
        mask,
        affine=placement,
        color=presentation["color"],
        opacity=presentation["opacity"],
    )
    if region is None:
        raise SessionError(f"Could not build region {label}")
    return region, bounds


def prepare_session(document, worker):
    """Load and verify every dependency, disposing partial results on failure."""
    if not isinstance(document, dict) or set(document) != {"sources", "state", "views"}:
        raise SessionError("Invalid session document")
    session_manager.validate_state(document["state"])
    session_manager.validate_views(document["views"])
    validate_sources(document["sources"])
    validate_mesh_document(document)
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
            if not texture_asset_role(role):
                options = {}
                if role == "anatomy" and "anatomical_display" in document["state"]:
                    options["anatomical_display"] = document["state"]["anatomical_display"]
                sources[role] = session_manager.load_source(
                    role, record["path"], cancelled, worker.report,
                    worker.set_loader, **options,
                )
            if source_stamp(record["path"]) != stamp:
                raise SessionError(f"Source changed during restore: {record['path']}")
        for record in document["state"].get("mesh_layers", []):
            result = sources["mesh:" + record["id"]]
            layer = restored_mesh_layer(result, record)
            from ...session.mesh import mesh_attribute_values, mesh_geometry_values
            from ...session.state import _content_digest

            source_record = document["sources"]["mesh:" + record["id"]]
            if "verified_geometry_sha256" in result:
                if result["verified_geometry_sha256"] != source_record["sha256"]:
                    raise SessionError(f"Mesh source changed during restore: {result['path']}")
                result["_session_association"] = (
                    source_record.copy(),
                    _content_digest(
                        [
                            mesh_geometry_values(result["polydata"]),
                            result["space_metadata"],
                            mesh_attribute_values(result["polydata"]),
                        ],
                        cancelled,
                    ),
                )
            components = document["views"].get("mesh_tree", {}).get("components", {})
            if (layer["id"], "information") in components and (
                ((layer["id"], "information_strips") in components)
                != bool(layer["strip_count"])
            ):
                raise SessionError("Mesh information tree does not match its source")
            regions = [line["region"] for line in record.get("polylines", ()) if "region" in line]
            if regions:
                from ...mesh.polygon import load_surface

                def check():
                    checkpoint(cancelled)

                surface = load_surface(layer, check)
                for region in regions:
                    surface.verify_region(region, check)
                result["territory_surface"] = surface
            if layer["visible"]:
                for component in ("boundary_edges", "surface_normals"):
                    if layer["presentation"][component]["enabled"]:
                        from ...mesh_io import MeshLoadCancelled
                        from ...mesh.overlays import prepare_display_overlay

                        worker.report(f"Preparing {component.replace('_', ' ')}: "
                                      f"{record['display_name']}")
                        try:
                            data = prepare_display_overlay(
                                layer["polydata"], component, cancelled,
                            )
                        except MeshLoadCancelled as error:
                            raise SessionCancelled() from error
                        result.setdefault("overlay_preparation", {})[component] = data
        from ...mesh_texture import MeshTextureCache, MeshTextureCancelled

        texture_cache = MeshTextureCache()
        for record in document["state"].get("mesh_layers", []):
            configuration = record.get("texture")
            if (
                configuration is None or not configuration["enabled"]
                or not record["visible"]
                or not record.get("presentation", {}).get("surface", {}).get(
                    "enabled", True,
                )
            ):
                continue
            from dataclasses import replace
            from ...mesh.process import prepare_textures_in_process

            result = sources["mesh:" + record["id"]]
            assets = {role: document["sources"][role]
                      for role in texture_references(configuration)}
            worker.report(f"Preparing texture: {record['display_name']}")
            budgets = replace(
                texture_cache.budgets,
                global_cache_bytes=max(1, texture_cache.budgets.global_cache_bytes
                                       - texture_cache.nbytes),
            )
            obj_index = None
            if (
                "obj_texture_index" in result
                and "source_stamp" in result
                and "source_sha256" in result
            ):
                obj_index = {
                    "path": result["path"],
                    "stamp": result["source_stamp"],
                    "sha256": result["source_sha256"],
                    "size": result["source_stamp"][2],
                    "index": result["obj_texture_index"],
                }
            try:
                prepared = prepare_textures_in_process(
                    Path(result["path"]), gl_maximum=budgets.max_dimension,
                    budgets=budgets, cancel_check=cancelled,
                    saved=(configuration, assets),
                    **({"obj_index": obj_index} if obj_index is not None else {}),
                )
            except MeshTextureCancelled as error:
                raise SessionCancelled() from error
            texture_cache.store(record["id"], prepared, active=True)
            result["texture_preparation"] = prepared
        count = (
            len(sources["tractogram"]["streamlines"]) if "tractogram" in sources else 0
        )
        if count != document["state"]["streamline_count"]:
            raise SessionError("Tractogram count does not match the session")
        if document["state"]["roi_layers"] and "anatomy" not in sources:
            raise SessionError("ROI editing requires its anatomical reference")
        if "anatomical_display" in document["state"] and "anatomy" not in sources:
            raise SessionError("Volume selection requires its anatomical source")
        if "anatomy" in sources:
            shape = sources["anatomy"]["data"].shape
            for axis, size in zip("xyz", shape):
                index = document["views"]["slices"][axis]
                if index is None or index >= size:
                    raise SessionError("Slice is outside the anatomical preview")
        scalar = document["state"]["active_scalar_name"]
        display = document["state"].get("scalar_display")
        scalar_source = "data_per_streamline" if (
            display is not None and display["association"] == "streamline"
        ) else "scalars"
        fields = sources.get("tractogram", {}).get(scalar_source, {})
        if scalar is not None and scalar not in fields:
            raise SessionError("Unknown active scalar")
        if display is not None:
            from ...tractogram_arrays import DataArray

            if scalar is None or display["component"] >= DataArray(
                display["association"], scalar, fields[scalar]
            ).components:
                raise SessionError("Unknown scalar component")
        if document["state"]["odf_tunnel_indices"] is not None and "odf" not in sources:
            raise SessionError("Missing ODF source")
        if document["views"]["regions"] and "parcellation" not in sources:
            raise SessionError("Missing parcellation source")
        if set(document["views"]["roi_actors"]) - set(document["state"]["roi_layers"]):
            raise SessionError("Unknown ROI presentation")
        if "odf" in sources:
            from ... import odf_utils

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
        session_manager.dispose_sources(sources)
        raise


@batch_render(lambda mw: mw.vtk_panel)
def populate_window(
    mw, document, sources, cancelled=lambda: False, progress=lambda text: None
):
    """Build a hidden candidate window; the previous scene remains untouched."""
    from copy import deepcopy
    from ...utils import ColorMode, signals_blocked
    from ..state_manager import ActionType

    state, views = document["state"], document["views"]
    checkpoint(cancelled)
    for name in STATE_FIELDS:
        setattr(mw, name, state[name])
    mw.scalar_display = state.get("scalar_display")
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
    if "parcellation" in sources:
        mw._parcellation_label_count = None
        mw.parcellation_spatial = sources["parcellation"].get("spatial")
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
    mw.standalone_polylines = deepcopy(state.get("standalone_polylines", []))
    mw.mesh_polyline_manager.restore(None)
    for record in state.get("mesh_layers", []):
        progress(f"Building surface: {record['display_name']}")
        checkpoint(cancelled)
        result = sources["mesh:" + record["id"]]
        layer = restored_mesh_layer(result, record)
        mw.mesh_layers[layer["id"]] = layer
        association = result.pop("_session_association", None)
        if association is not None:
            mw._session_source_associations["mesh:" + layer["id"]] = association
        panel.add_mesh_layer(layer, render=False)
        surface = result.pop("territory_surface", None)
        if surface is None:
            mw.mesh_polyline_manager.restore(layer["id"])
        else:
            mw.mesh_polyline_manager.polygon_editor().cache[layer["id"]] = surface
            mw.mesh_polyline_manager.restore(layer["id"], verified_surface=surface)
        for component, data in result.pop("overlay_preparation", {}).items():
            checkpoint(cancelled)
            panel.publish_mesh_overlay(layer, component, data, render=False)
        if "texture" in record:
            layer["texture_assets"] = {
                role: document["sources"][role].copy()
                for role in texture_references(record["texture"])
            }
            mw.mesh_manager.restore_texture(
                layer, result.pop("texture_preparation", None),
            )
        result.pop("polydata")
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
            _update_odf_tunnel_actor(mw, amplitudes, affine)
            if panel.odf_actor is not None:
                panel.odf_actor.SetVisibility(mw.odf_tunnel_is_visible)
    if views["regions"]:
        mw._session_deferred_regions = {}
        bounds = None
        for label, presentation in views["regions"].items():
            if mw.tractogram_data is not None and not (
                presentation["visible"] and presentation["attached"]
            ):
                mw._session_deferred_regions[label] = dict(presentation)
                continue
            progress(f"Building parcellation region: {label}")
            checkpoint(cancelled)
            region, bounds = _build_region_contour(mw, label, presentation, bounds)
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
    mw.scalar_manager.update_scalar_range_widgets()
    mw._update_action_states()
    mw._update_bundle_info_display()
    mw._data_panel_update_pending = True
    mw._perform_data_panel_update()
    if getattr(mw, "_data_panel_debounce_timer", None) is not None:
        mw._data_panel_debounce_timer.stop()
    mw._data_panel_update_pending = False
    with signals_blocked(mw.data_tree_widget):
        for key, item in session_manager._tree_items(mw.data_tree_widget):
            if key == views["tree_current"]:
                mw.data_tree_widget.setCurrentItem(item)
            if key in views["tree"]:
                expanded, selected = views["tree"][key]
                item.setExpanded(expanded)
                item.setSelected(selected)
    session_manager.restore_mesh_tree(mw, views)
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
    session_manager.restore_cameras(mw, views)
    from ...session.surface_mapping import restore_recipe

    restore_recipe(mw.surface_mapping_manager, state.get("surface_mapping_recipe"))
    checkpoint(cancelled)


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
        from ... import odf_utils

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

        amplitudes, affine = session_manager.SessionManager(mw)._run(
            "Restoring ODF tunnel...", operation, discard_result=lambda result: None
        )
        _update_odf_tunnel_actor(mw, amplitudes, affine)
        mw._session_deferred_odf = False
        return True
    except SessionCancelled:
        return False
    except Exception as error:
        logger.exception("Deferred ODF restore failed")
        session_manager.QMessageBox.critical(mw, "Session Error", str(error))
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

    staged = {}
    bounds = None
    try:
        for label in labels:
            region, bounds = _build_region_contour(mw, label, pending[label], bounds)
            region.SetVisibility(True)
            staged[label] = region
        mw.parcellation_region_actors.update(staged)
        for label in labels:
            pending.pop(label)
        return True
    except Exception as error:
        logger.exception("Deferred parcellation restore failed")
        session_manager.QMessageBox.critical(mw, "Session Error", str(error))
        return False
