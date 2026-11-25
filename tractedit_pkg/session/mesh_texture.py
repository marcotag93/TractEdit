"""Frozen diffuse presentation and external asset references for schema 4."""

from copy import deepcopy
import os
from pathlib import Path
from uuid import UUID, NAMESPACE_URL, uuid5

import numpy as np

from ..data_contracts import (
    MeshTextureAsset, MeshTextureConfiguration, MeshTexturePreparation,
)

MAX_TEXTURE_ASSETS = 256 * (4096 + 64)


def texture_asset_role(role: str) -> bool:
    if not isinstance(role, str) or ":" not in role:
        return False
    prefix, identity = role.split(":", 1)
    try:
        return prefix in ("mesh_mtl", "mesh_image") and str(UUID(identity)) == identity
    except ValueError:
        return False


def texture_references(configuration: MeshTextureConfiguration) -> set[str]:
    roles = set(configuration["libraries"])
    for material in configuration["materials"].values():
        roles.update(material[field] for field in ("library", "image")
                     if material[field] is not None)
    return roles


def texture_configuration(
    preparation: MeshTexturePreparation, enabled: bool, previous=None,
) -> tuple[MeshTextureConfiguration, dict[str, MeshTextureAsset]]:
    assets = {}
    identities = {
        (role.split(":", 1)[0], os.path.normcase(asset["path"]),
         asset["size"], asset["sha256"]): role
        for role, asset in (previous or {}).items()
    }

    def reference(kind, asset):
        if asset is None:
            return None
        identity = (kind, os.path.normcase(asset["path"]), asset["size"], asset["sha256"])
        role = identities.get(identity)
        if role is None:
            role = kind + ":" + str(uuid5(NAMESPACE_URL, repr(identity[1:])))
            identities[identity] = role
        assets[role] = asset.copy()
        return role

    libraries = tuple(dict.fromkeys(
        reference("mesh_mtl", asset) for asset in preparation["libraries"]
    ))
    materials = {}
    for name, material in preparation["materials"].items():
        image = preparation["images"].get(material["image_key"])
        materials[name] = {
            "library": reference("mesh_mtl", material["library"]),
            "image": reference("mesh_image", image["asset"]) if image else None,
            "diffuse_map": deepcopy(material["diffuse_map"]),
            "diffuse_color": material["diffuse_color"],
            "unavailable_reason": material["unavailable_reason"],
        }
        for field in ("image_origin", "library_origin"):
            if field in material:
                materials[name][field] = material[field]
    return {
        "enabled": enabled, "libraries": libraries, "materials": materials,
        "face_layout": deepcopy(preparation["face_layout"]),
    }, assets


def validate_texture_configuration(configuration) -> None:
    from .io import SessionError

    def invalid():
        raise SessionError("Invalid schema 4 texture presentation")

    def role(value, kind):
        return texture_asset_role(value) and value.startswith(kind + ":")

    def vector(value, low=None, high=None):
        return (
            isinstance(value, tuple) and len(value) == 3
            and all(type(v) in (float, int) and np.isfinite(v)
                    and (low is None or v >= low) and (high is None or v <= high)
                    for v in value)
        )

    if (
        not isinstance(configuration, dict)
        or set(configuration) != {"enabled", "libraries", "materials", "face_layout"}
        or type(configuration["enabled"]) is not bool
    ):
        invalid()
    libraries = configuration["libraries"]
    if (not isinstance(libraries, tuple) or len(libraries) > 100000
            or any(not role(v, "mesh_mtl") for v in libraries)
            or len(set(libraries)) != len(libraries)):
        invalid()
    materials = configuration["materials"]
    if not isinstance(materials, dict) or not 0 < len(materials) <= 4096:
        invalid()
    for name, material in materials.items():
        if (not isinstance(name, str) or not name or len(name) > 99
                or not name.isascii() or any(c.isspace() for c in name)
                or not isinstance(material, dict)
                or set(material) - {"image_origin", "library_origin"}
                != {"library", "image", "diffuse_map",
                    "diffuse_color", "unavailable_reason"}):
            invalid()
        for field in ("image_origin", "library_origin"):
            if field in material and material[field] not in ("automatic", "manual"):
                invalid()
        library, image = material["library"], material["image"]
        if (library is not None and library not in libraries
                or image is not None and not role(image, "mesh_image")
                or not vector(material["diffuse_color"], 0, 1)):
            invalid()
        reason = material["unavailable_reason"]
        if reason is not None and (not isinstance(reason, str) or len(reason) > 2048):
            invalid()
        mapping = material["diffuse_map"]
        if mapping is not None and (
            not isinstance(mapping, dict)
            or set(mapping) != {"path", "scale", "offset", "clamp"}
            or not isinstance(mapping["path"], str)
            or not mapping["path"] or len(mapping["path"]) > 65536
            or not vector(mapping["scale"]) or mapping["scale"][2] != 1
            or not vector(mapping["offset"]) or mapping["offset"][2] != 0
            or type(mapping["clamp"]) is not bool
        ):
            invalid()
        if image is not None and (
            mapping is None or reason is not None
            or library is None and material.get("image_origin") != "manual"
        ):
            invalid()
    images = {m["image"] for m in materials.values() if m["image"] is not None}
    if not 0 < len(images) <= 64:
        invalid()
    layout = configuration["face_layout"]
    if not isinstance(layout, dict) or not 0 < len(layout) <= 4097:
        invalid()
    for name, summary in layout.items():
        if (not isinstance(name, str) or name and name not in materials
                or not isinstance(summary, dict)
                or set(summary) != {"face_count", "corner_count", "complete_uv"}
                or type(summary["face_count"]) is not int
                or not 0 < summary["face_count"] <= 2**31 - 1
                or type(summary["corner_count"]) is not int
                or not 3 * summary["face_count"] <= summary["corner_count"] <= 2**31 - 1
                or type(summary["complete_uv"]) is not bool):
            invalid()
    if set(materials) != set(layout) - {""}:
        invalid()


def verify_texture_assets(layer, cancelled, progress):
    from .io import checked_fingerprint, SessionError

    assets = layer.get("texture_assets", {})
    if set(assets) != texture_references(layer["texture"]):
        raise SessionError("Texture asset references do not match the loaded presentation")
    for expected in assets.values():
        actual, _ = checked_fingerprint(expected["path"], cancelled, progress)
        if actual != expected:
            raise SessionError(f"Texture asset changed since preparation: {expected['path']}")
    return assets


def prepare_saved_textures(
    path, configuration, assets, *, gl_maximum, budgets, cancelled=None, progress=None,
    image_overrides=None, refresh=False, _obj_index=None,
) -> MeshTexturePreparation:
    """Restore frozen assets or explicitly update them without rebinding images."""
    from .. import mesh_texture as _mt
    from ..mesh_texture import (
        MeshTextureCache, MeshTextureError, _fingerprint, _read_obj_face_layout,
        decode_texture_image, _check_cancel, resolve_texture_asset, replacement_texture_map,
        _Diagnostics, _read_mtl, _layout_from_obj_index, _scan_obj_stream,
        _validate_obj_index, _verify_obj_unchanged,
    )
    from .io import source_stamp

    validate_texture_configuration(configuration)
    if image_overrides or refresh:
        configuration = deepcopy(configuration)
        assets = deepcopy(assets)
        counts = [0, 0, 0]
        for role in configuration["libraries"]:
            previous = assets[role]
            current = _fingerprint(previous["path"], cancelled)
            if current == previous:
                continue
            materials, declared = {}, set()
            _read_mtl(
                current, materials, declared, counts, budgets,
                _Diagnostics(budgets), cancelled,
            )
            for name, saved in configuration["materials"].items():
                if saved["library"] != role:
                    continue
                material = materials.get(name)
                if material is None or material["unavailable_reason"] is not None:
                    raise MeshTextureError(
                        "invalid_material", f"Changed MTL has no valid material: {name}",
                    )
                saved["diffuse_color"] = material["diffuse_color"]
                if material["diffuse_map"] is not None:
                    saved["diffuse_map"] = deepcopy(material["diffuse_map"])
            assets[role] = current
        for name, replacement in (image_overrides or {}).items():
            if (
                name not in configuration["materials"]
                or not configuration["face_layout"][name]["complete_uv"]
            ):
                raise MeshTextureError(
                    "texture_geometry",
                    "Choose a material with complete texture coordinates",
                )
            asset = resolve_texture_asset(
                str(replacement), ".", replacement=replacement, cancelled=cancelled,
            )
            role = "mesh_image:" + str(uuid5(NAMESPACE_URL, repr(asset)))
            assets[role] = asset
            material = configuration["materials"][name]
            material["image"] = role
            material["image_origin"] = "manual"
            material["unavailable_reason"] = None
            if material["diffuse_map"] is None:
                material["diffuse_map"] = replacement_texture_map(replacement)
        references = texture_references(configuration)
        assets = {
            role: asset for role, asset in assets.items()
            if role in references
        }
        validate_texture_configuration(configuration)
    resolved_path = Path(path).resolve(strict=True)
    default_layout = _mt._read_obj_face_layout is _mt._DEFAULT_READ_OBJ_FACE_LAYOUT
    indexed = (
        _validate_obj_index(resolved_path, _obj_index, cancelled)
        if default_layout and _obj_index is not None
        else None
    )
    if indexed is not None:
        source, before_stamp = indexed
        layout, coverage = _layout_from_obj_index(_obj_index["index"], budgets)
    elif default_layout:
        source, before_stamp, _, _, layout, coverage = _scan_obj_stream(
            resolved_path, budgets, cancelled,
            collect_references=False, compute_source=True,
        )
    else:
        before_stamp = source_stamp(resolved_path)
        source = _fingerprint(resolved_path, cancelled)
        layout, coverage = _read_obj_face_layout(resolved_path, budgets, cancelled)
    verified_images = {}
    for role, expected in assets.items():
        actual = _fingerprint(
            expected["path"], cancelled,
            verified_images if role.startswith("mesh_image:") else None,
        )
        if refresh and role.startswith("mesh_image:"):
            assets[role] = actual
        elif actual != expected:
            raise MeshTextureError("asset_changed", "Saved texture asset changed")
    if layout != configuration["face_layout"]:
        raise MeshTextureError("face_layout", "Saved OBJ material/UV associations changed")
    materials, images, samplers = {}, {}, set()
    diagnostics = _Diagnostics(budgets)
    cache = MeshTextureCache(budgets)
    costs = dict(decoded_cpu_bytes=0, temporary_cpu_bytes=0, resident_estimate_bytes=0)
    for name, saved in configuration["materials"].items():
        _check_cancel(cancelled)
        library = assets[saved["library"]] if saved["library"] else None
        image_asset = assets[saved["image"]] if saved["image"] else None
        reason = saved["unavailable_reason"]
        if refresh and image_asset is None and library and saved["diffuse_map"]:
            try:
                image_asset = resolve_texture_asset(
                    saved["diffuse_map"]["path"], Path(library["path"]).parent,
                    obj_directory=Path(path).parent, cancelled=cancelled,
                    _verified=verified_images,
                )
                role = "mesh_image:" + str(uuid5(NAMESPACE_URL, repr(image_asset)))
                assets[role] = image_asset
                reason = None
            except MeshTextureError as error:
                reason = error.code
                diagnostics.add(error.code, str(error))
        key = os.path.normcase(image_asset["path"]) if image_asset else None
        if key is not None:
            if key not in images:
                try:
                    image = decode_texture_image(
                        image_asset, gl_maximum=gl_maximum,
                        budgets=budgets, cache=cache,
                        retained_bytes=costs["decoded_cpu_bytes"],
                        resident_bytes=costs["resident_estimate_bytes"],
                        cancelled=cancelled, progress=progress,
                        _verified_stamp=verified_images.get(
                            (image_asset["path"], image_asset["sha256"])
                        ),
                    )
                except MeshTextureError as error:
                    if saved["image"] is not None or not refresh or error.code in (
                        "texture_budget", "texture_memory", "image_limit",
                        "global_cache_budget",
                    ):
                        raise
                    reason, key = error.code, None
                    diagnostics.add(error.code, str(error))
                else:
                    images[key] = image
                    costs["decoded_cpu_bytes"] += image["pixels"].nbytes
                    costs["temporary_cpu_bytes"] = max(
                        costs["temporary_cpu_bytes"], image["temporary_cpu_bytes"],
                    )
        if key is not None:
            sampler = (key, saved["diffuse_map"]["clamp"])
            if sampler not in samplers:
                costs["resident_estimate_bytes"] += images[key]["resident_estimate_bytes"]
                samplers.add(sampler)
                if costs["resident_estimate_bytes"] > budgets.resident_estimate_bytes:
                    raise MeshTextureError("texture_budget", "Saved sampler textures exceed budget")
        materials[name] = {
            "name": name, "library": library, "image_key": key,
            "diffuse_map": deepcopy(saved["diffuse_map"]),
            "diffuse_color": saved["diffuse_color"],
            "unavailable_reason": reason,
        }
        for field in ("image_origin", "library_origin"):
            if field in saved:
                materials[name][field] = saved[field]
    _verify_obj_unchanged(
        resolved_path, source, before_stamp, cancelled,
        "OBJ changed during restore",
    )
    for expected in assets.values():
        if _fingerprint(expected["path"], cancelled) != expected:
            raise MeshTextureError("asset_changed", "Saved texture asset changed during restore")
    result = {
        "source": source, "materials": materials, "images": images,
        "libraries": tuple(assets[role] for role in configuration["libraries"]),
        "diagnostics": tuple(diagnostics.items), "costs": costs, "face_layout": layout,
    }
    if coverage is not None:
        result["face_uv"] = coverage
        if costs["decoded_cpu_bytes"] + coverage.nbytes > min(
            budgets.layer_cache_bytes, budgets.global_cache_bytes,
        ):
            raise MeshTextureError("texture_budget", "Saved UV coverage exceeds cache budget")
    return result
