import os
from collections.abc import Callable
from pathlib import Path
from ..data_contracts import (
    MeshLayer,
    MeshTextureAssetRequest,
    MeshTextureCosts,
    MeshTexturePreparation,
)
from .assets import (
    MeshTextureBudgets,
    DEFAULT_BUDGETS,
    AssetResolver,
    MeshTextureError,
    _check_cancel,
    _fingerprint,
    resolve_texture_asset,
)
from .mtl import replacement_texture_map, _Diagnostics
from .obj import (
    _parse_obj_reference_record,
    _DEFAULT_READ_OBJ_REFERENCES,
    _DEFAULT_READ_OBJ_FACE_LAYOUT,
    _validate_obj_index,
    _layout_from_obj_index,
    _verify_obj_unchanged,
)
from .. import mesh_texture


def _same_stem_asset(path, suffixes):
    candidates = {
        os.path.normcase(str(candidate.resolve())): candidate
        for suffix in suffixes
        if (candidate := path.with_suffix(suffix)).is_file()
        and candidate.resolve().is_relative_to(path.parent.resolve())
    }
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _compatible_library(path, used, budgets, cancelled):
    candidate = _same_stem_asset(path, (".mtl",))
    if candidate is None:
        return None
    try:
        asset = resolve_texture_asset(
            candidate.name, candidate.parent, max_bytes=budgets.mtl_bytes,
            cancelled=cancelled,
        )
        materials, declared = {}, set()
        mesh_texture._read_mtl(
            asset, materials, declared, [0, 0, 0], budgets,
            _Diagnostics(budgets), cancelled,
        )
        if used <= declared and all(
            materials[name]["unavailable_reason"] is None for name in used
        ):
            return candidate
    except (MeshTextureError, OSError):
        pass
    return None


def _resolve(
    reference, base, resolver, cancelled, max_bytes=None, fallback=None,
    obj_directory=None, verified=None,
):
    try:
        return resolve_texture_asset(
            reference, base, cancelled=cancelled, max_bytes=max_bytes,
            obj_directory=obj_directory, _verified=verified,
        )
    except MeshTextureError as error:
        if error.code == "missing_asset" and fallback is not None:
            replacement = fallback()
            if replacement is not None:
                return resolve_texture_asset(
                    reference, base, replacement=replacement,
                    cancelled=cancelled, max_bytes=max_bytes, _verified=verified,
                )
        if resolver is None or error.code not in (
            "asset_resolution_required",
            "missing_asset",
        ):
            raise
        replacement = resolver(reference, str(base))
        if replacement is None:
            raise
        return resolve_texture_asset(
            reference,
            base,
            replacement=replacement,
            cancelled=cancelled,
            max_bytes=max_bytes,
            _verified=verified,
        )


def _resolve_once(
    reference, base, resolver, cancelled, assets, max_bytes=None, fallback=None,
    obj_directory=None, verified=None,
):
    key = (str(base), reference)
    if key not in assets:
        assets[key] = _resolve(
            reference, base, resolver, cancelled, max_bytes, fallback, obj_directory,
            verified,
        )
    return assets[key]


def prepare_obj_textures(
    obj_path: str | os.PathLike,
    *,
    gl_maximum: int,
    budgets: MeshTextureBudgets = DEFAULT_BUDGETS,
    resolver: AssetResolver | None = None,
    cache: "MeshTextureCache | None" = None,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
    image_overrides: dict[str, str] | None = None,
    _obj_index: dict | None = None,
) -> MeshTexturePreparation:
    """Prepare used materials only; the caller retains geometry and uniform fallback."""
    from ..session.io import source_stamp

    path = Path(obj_path).resolve(strict=True)
    if path.suffix.lower() != ".obj":
        raise MeshTextureError(
            "unsupported_mesh", "Diffuse texture preparation is OBJ-only"
        )
    diagnostics = _Diagnostics(budgets)
    default_scanners = (
        mesh_texture._read_obj_references is _DEFAULT_READ_OBJ_REFERENCES
        and mesh_texture._read_obj_face_layout is _DEFAULT_READ_OBJ_FACE_LAYOUT
    )
    indexed = (
        _validate_obj_index(path, _obj_index, cancelled)
        if default_scanners and _obj_index is not None
        else None
    )
    if indexed is not None:
        source, before_stamp = indexed
        index_data = _obj_index["index"]
        if progress is not None:
            progress("obj_references")
        references, materials = [], {}
        for raw in index_data["reference_lines"]:
            _check_cancel(cancelled)
            _parse_obj_reference_record(
                raw, budgets, references, materials, diagnostics
            )
        if progress is not None:
            progress("face_layout")
        face_layout, face_uv = _layout_from_obj_index(index_data, budgets)
    elif default_scanners:
        if progress is not None:
            progress("obj_references")
        (
            source, before_stamp, references, materials, face_layout, face_uv
        ) = mesh_texture._scan_obj_stream(
            path, budgets, cancelled,
            collect_references=True, diagnostics=diagnostics, compute_source=True,
        )
        if progress is not None:
            progress("face_layout")
    else:
        before_stamp = source_stamp(path)
        source = _fingerprint(path, cancelled)
        if progress is not None:
            progress("obj_references")
        references, materials = mesh_texture._read_obj_references(
            path, budgets, diagnostics, cancelled
        )
        if progress is not None:
            progress("face_layout")
        face_layout, face_uv = mesh_texture._read_obj_face_layout(path, budgets, cancelled)
    used = set(face_layout) - {""}
    image_overrides = image_overrides or {}
    if any(
        name not in used or not face_layout[name]["complete_uv"]
        for name in image_overrides
    ):
        raise MeshTextureError(
            "texture_geometry", "Choose a material with complete texture coordinates",
        )
    libraries, declared, counts = {}, set(), [0, 0, 0]
    assets = {}
    unresolved: list[MeshTextureAssetRequest] = []
    origins = {}

    def automatic_asset(kind, candidate):
        if candidate is not None:
            origins[(kind, str(candidate.resolve()))] = "automatic"
        return candidate

    def request_asset(reference, base, kind):
        replacement = resolver(reference, base)
        if replacement is not None:
            origins[(kind, str(Path(replacement).resolve()))] = "manual"
        request = {"base": base, "reference": reference, "kind": kind}
        if replacement is None and request not in unresolved and len(unresolved) < 32:
            unresolved.append(request)
        return replacement

    library_resolver = (
        (lambda reference, base: request_asset(reference, base, "mtl"))
        if resolver is not None and not used <= image_overrides.keys() else None
    )
    image_resolver = (
        (lambda reference, base: request_asset(reference, base, "image"))
        if resolver is not None else None
    )
    for reference in dict.fromkeys(references):
        _check_cancel(cancelled)
        asset = None
        try:
            asset = mesh_texture._resolve_once(
                reference, path.parent, library_resolver, cancelled, assets,
                budgets.mtl_bytes,
                fallback=(lambda: automatic_asset(
                    "mtl", _compatible_library(path, used, budgets, cancelled),
                )) if len(set(references)) == 1 else None,
            )
            key = os.path.normcase(asset["path"])
            if key in libraries:
                continue
            if counts[0] + asset["size"] > budgets.mtl_bytes:
                raise MeshTextureError(
                    "mtl_byte_limit", "MTL data exceeds its per-layer byte limit"
                )
            libraries[key] = asset
            if progress is not None:
                progress("mtl")
            mesh_texture._read_mtl(
                asset, materials, declared, counts, budgets, diagnostics, cancelled
            )
            if _fingerprint(asset["path"], cancelled) != asset:
                raise MeshTextureError("asset_changed", "MTL changed while parsing")
        except (MeshTextureError, OSError) as error:
            if isinstance(error, MeshTextureError) and error.code.endswith("limit"):
                raise
            # A damaged library must not leave its partial materials usable.
            for material in materials.values():
                if (
                    material["library"] is not None
                    and asset is not None
                    and material["library"]["path"] == asset["path"]
                ):
                    material["unavailable_reason"] = getattr(error, "code", "asset_io")
            if not used <= image_overrides.keys():
                diagnostics.add(getattr(error, "code", "asset_io"), str(error))
    materials = {name: material for name, material in materials.items() if name in used}
    diffuse_references = {
        (material["library"]["path"], material["diffuse_map"]["path"])
        for material in materials.values()
        if material["library"] is not None and material["diffuse_map"] is not None
        and material["unavailable_reason"] is None
    }
    images, samplers = {}, set()
    verified_images = {}
    failed_assets = {}
    costs: MeshTextureCosts = {
        "decoded_cpu_bytes": 0,
        "temporary_cpu_bytes": 0,
        "resident_estimate_bytes": 0,
    }
    for material in materials.values():
        _check_cancel(cancelled)
        key = None
        name = material["name"]
        library = material["library"]
        if library is not None:
            origin = origins.get(("mtl", library["path"]))
            if origin is not None:
                material["library_origin"] = origin
        if material["unavailable_reason"] is not None:
            if name in image_overrides:
                raise MeshTextureError(
                    "invalid_material", f"Cannot replace invalid material: {name}",
                )
            continue
        diffuse_map = material["diffuse_map"]
        if name in image_overrides and diffuse_map is None:
            diffuse_map = replacement_texture_map(image_overrides[name])
            material["diffuse_map"] = diffuse_map
        if diffuse_map is None:
            material["unavailable_reason"] = (
                "no_diffuse_map" if material["library"] else "missing_material"
            )
            diagnostics.add(
                material["unavailable_reason"],
                f"No diffuse image for material {material['name']}",
            )
            continue
        try:
            base = Path(library["path"]).parent if library else path.parent
            if name in image_overrides:
                asset = resolve_texture_asset(
                    diffuse_map["path"], base, replacement=image_overrides[name],
                    cancelled=cancelled, _verified=verified_images,
                )
                material["image_origin"] = "manual"
            else:
                asset = mesh_texture._resolve_once(
                    diffuse_map["path"], base, image_resolver, cancelled, assets,
                    obj_directory=path.parent, verified=verified_images,
                    fallback=(lambda: automatic_asset(
                        "image", _same_stem_asset(path, (".png", ".jpg", ".jpeg")),
                    )) if len(diffuse_references) == 1 else None,
                )
                origin = origins.get(("image", asset["path"]))
                if origin is not None:
                    material["image_origin"] = origin
            key = os.path.normcase(asset["path"])
            if key in failed_assets:
                raise failed_assets[key]
            if key not in images:
                if len(images) >= budgets.max_images:
                    raise MeshTextureError(
                        "image_limit", "Too many unique diffuse images"
                    )
                image = mesh_texture.decode_texture_image(
                    asset,
                    gl_maximum=gl_maximum,
                    budgets=budgets,
                    retained_bytes=costs["decoded_cpu_bytes"],
                    resident_bytes=costs["resident_estimate_bytes"],
                    cache=cache,
                    cancelled=cancelled,
                    progress=progress,
                    _verified_stamp=verified_images.get(
                        (asset["path"], asset["sha256"])
                    ),
                )
                images[key] = image
                size = image["pixels"].nbytes
                costs["decoded_cpu_bytes"] += size
                costs["temporary_cpu_bytes"] = max(
                    costs["temporary_cpu_bytes"], image["temporary_cpu_bytes"]
                )
            elif images[key]["asset"] != asset:
                raise MeshTextureError(
                    "asset_changed", "Shared image changed during preparation"
                )
            sampler = (key, diffuse_map["clamp"])
            if sampler not in samplers:
                resident = (
                    costs["resident_estimate_bytes"]
                    + images[key]["resident_estimate_bytes"]
                )
                if resident > budgets.resident_estimate_bytes:
                    raise MeshTextureError(
                        "texture_budget",
                        "Sampler textures exceed the resident estimate budget",
                    )
                costs["resident_estimate_bytes"] = resident
                samplers.add(sampler)
            material["image_key"] = key
        except MeshTextureError as error:
            if name in image_overrides or error.code in (
                "texture_budget",
                "texture_memory",
                "image_limit",
                "global_cache_budget",
            ):
                raise
            if key is not None:
                failed_assets[key] = error
            material["unavailable_reason"] = error.code
            diagnostics.add(error.code, f"{material['name']}: {error}")
    _check_cancel(cancelled)
    _verify_obj_unchanged(
        path, source, before_stamp, cancelled,
        "OBJ changed during texture preparation",
    )
    result = {
        "source": source,
        "materials": materials,
        "libraries": tuple(libraries.values()),
        "images": images,
        "diagnostics": tuple(diagnostics.items),
        "costs": costs,
        "face_layout": face_layout,
    }
    if unresolved:
        result["unresolved_assets"] = tuple(unresolved)
    if face_uv is not None:
        result["face_uv"] = face_uv
        if costs["decoded_cpu_bytes"] + face_uv.nbytes > min(
            budgets.layer_cache_bytes, budgets.global_cache_bytes,
        ):
            raise MeshTextureError("texture_budget", "UV coverage exceeds the cache budget")
    return result


def validate_texture_source(
    layer: MeshLayer, preparation: MeshTexturePreparation
) -> None:
    expected = layer.get("source_sha256")
    if expected is None or preparation.get("source", {}).get("sha256") != expected:
        raise MeshTextureError(
            "asset_changed", "OBJ changed since loading; reload the mesh before texturing"
        )
