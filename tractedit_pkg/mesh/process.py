"""Isolate native OBJ reads and transfer owned arrays without geometry pickling."""

import json
import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from ..process_isolation import run_isolated_process, start_process
from ..data_contracts import MeshLoadResult, MeshTexturePreparation
from ..mesh_io import MeshLoadError, _check_cancel, _read_mesh

if TYPE_CHECKING:
    from vtkmodules.vtkCommonCore import vtkAbstractArray
    from ..mesh_texture import MeshTextureBudgets


_TRANSFER_BYTES = 4 * 1024 * 1024


def _write_result(result: MeshLoadResult, directory: Path) -> None:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    arrays = []

    def save(array: "vtkAbstractArray") -> int:
        index = len(arrays)
        record = {
            "name": array.GetName(),
            "type": array.GetDataType(),
            "components": array.GetNumberOfComponents(),
        }
        if array.IsA("vtkStringArray"):
            record["strings"] = [
                array.GetValue(i) for i in range(array.GetNumberOfValues())
            ]
        else:
            np.save(directory / f"{index}.npy", vtk_to_numpy(array), allow_pickle=False)
        arrays.append(record)
        return index

    data = result["polydata"]
    document = {
        "result": {
            key: value for key, value in result.items()
            if key not in ("polydata", "obj_texture_index")
        },
        "points": save(data.GetPoints().GetData()),
        "cells": [
            (save(cells.GetOffsetsArray()), save(cells.GetConnectivityArray()))
            for cells in (
                data.GetPolys(), data.GetStrips(), data.GetLines(), data.GetVerts()
            )
        ],
        "attributes": [],
        "arrays": arrays,
    }
    for attributes in (data.GetPointData(), data.GetCellData(), data.GetFieldData()):
        active = {}
        if attributes.IsA("vtkDataSetAttributes"):
            for role in range(attributes.NUM_ATTRIBUTES):
                array = attributes.GetAbstractAttribute(role)
                if array is not None:
                    active[role] = array.GetName()
        document["attributes"].append({
            "arrays": [save(attributes.GetAbstractArray(i))
                       for i in range(attributes.GetNumberOfArrays())],
            "active": active,
        })
    (directory / "result.json").write_text(json.dumps(document), encoding="utf-8")


def _read_obj(source: str, face_count: int, directory: str, channel=None) -> None:
    folder = Path(directory)
    try:
        def progress(value, message):
            if channel is not None:
                channel.send((value, message))

        result = _read_mesh(
            Path(source), face_count,
            **({"progress": progress} if channel is not None else {}),
        )
        progress(80, "Transferring surface data...")
        _write_result(result, folder)
    except Exception as error:
        (folder / "error.json").write_text(
            json.dumps(str(error)[:4096]), encoding="utf-8"
        )
    finally:
        if channel is not None:
            channel.close()


def _restore_result(
    directory: Path, cancel_check: Callable[[], bool]
) -> MeshLoadResult:
    from vtkmodules.util.numpy_support import create_vtk_array, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints, vtkStringArray
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    document = json.loads((directory / "result.json").read_text(encoding="utf-8"))

    def restore(index: int) -> "vtkAbstractArray":
        _check_cancel(cancel_check)
        record = document["arrays"][index]
        if "strings" in record:
            array = vtkStringArray()
            array.SetNumberOfComponents(record["components"])
            for value in record["strings"]:
                array.InsertNextValue(value)
        else:
            source = np.load(
                directory / f"{index}.npy", mmap_mode="r", allow_pickle=False
            )
            try:
                array = create_vtk_array(record["type"])
                array.SetNumberOfComponents(record["components"])
                array.SetNumberOfTuples(source.shape[0])
                step = max(1, _TRANSFER_BYTES // source.dtype.itemsize)
                target = vtk_to_numpy(array).reshape(-1)
                values = source.reshape(-1)
                for start in range(0, source.size, step):
                    _check_cancel(cancel_check)
                    target[start:start + step] = values[start:start + step]
            finally:
                source._mmap.close()
        if record["name"] is not None:
            array.SetName(record["name"])
        return array

    data = vtkPolyData()
    points = vtkPoints()
    points.SetData(restore(document["points"]))
    data.SetPoints(points)
    for setter, (offsets, indices) in zip(
        (data.SetPolys, data.SetStrips, data.SetLines, data.SetVerts), document["cells"]
    ):
        cells = vtkCellArray()
        cells.SetData(restore(offsets), restore(indices))
        setter(cells)
    for attributes, record in zip(
        (data.GetPointData(), data.GetCellData(), data.GetFieldData()),
        document["attributes"],
    ):
        for index in record["arrays"]:
            attributes.AddArray(restore(index))
        for role, name in record["active"].items():
            attributes.SetActiveAttribute(name, int(role))
    result = document["result"]
    result["source_bounds"] = tuple(result["source_bounds"])
    result["space_metadata"]["declarations"] = tuple(
        result["space_metadata"]["declarations"]
    )
    result["polydata"] = data
    _check_cancel(cancel_check)
    return result


def load_obj_in_process(
    source: Path, face_count: int, cancel_check: Callable[[], bool],
    progress: Callable[[int, str], None] | None = None,
) -> MeshLoadResult:
    with TemporaryDirectory(prefix="tractedit-obj-") as directory:
        exitcode = run_isolated_process(
            _read_obj, (str(source), face_count, directory),
            cancel_check=lambda: _check_cancel(cancel_check),
            channel=progress is not None,
            progress=lambda message: progress(*message), start=start_process,
        )
        if exitcode:
            raise MeshLoadError(
                f"OBJ reader process exited with code {exitcode}"
            )
        folder = Path(directory)
        error = folder / "error.json"
        if error.exists():
            raise MeshLoadError(json.loads(error.read_text(encoding="utf-8")))
        if progress is not None:
            progress(85, "Preparing surface arrays...")
        return _restore_result(folder, cancel_check)


def _overlay_record(polydata) -> MeshLoadResult:
    from .model import mesh_space_metadata

    return {
        "path": "", "polydata": polydata,
        "point_count": polydata.GetNumberOfPoints(),
        "polygon_count": polydata.GetNumberOfPolys(),
        "strip_count": polydata.GetNumberOfStrips(),
        "source_bounds": polydata.GetBounds(),
        "space_metadata": mesh_space_metadata("vtp_field_data", None),
    }


def _prepare_overlay(directory: str, component: str) -> None:
    from .overlays import prepare_mesh_overlay

    folder = Path(directory)
    try:
        source = _restore_result(folder / "input", lambda: False)["polydata"]
        result = prepare_mesh_overlay(source, component)
        _write_result(_overlay_record(result.polydata), folder / "output")
        (folder / "faces.json").write_text(json.dumps(result.face_count), "utf-8")
    except Exception as error:
        (folder / "error.json").write_text(
            json.dumps(str(error)[:4096] or "Insufficient memory for mesh overlay"),
            "utf-8",
        )


def prepare_overlay_in_process(source, component: str, cancel_check: Callable[[], bool]):
    """Isolate native filters that retain the GIL; transfer only owned CPU arrays."""
    from .overlays import MeshOverlayData

    with TemporaryDirectory(prefix="tractedit-overlay-") as directory:
        folder = Path(directory)
        (folder / "input").mkdir()
        (folder / "output").mkdir()
        _check_cancel(cancel_check)
        _write_result(_overlay_record(source), folder / "input")
        exitcode = run_isolated_process(
            _prepare_overlay, (directory, component),
            cancel_check=lambda: _check_cancel(cancel_check), start=start_process,
        )
        if exitcode:
            raise MeshLoadError(
                f"Mesh overlay process exited with code {exitcode}"
            )
        error = folder / "error.json"
        if error.exists():
            raise MeshLoadError(json.loads(error.read_text("utf-8")))
        result = _restore_result(folder / "output", cancel_check)
        faces = json.loads((folder / "faces.json").read_text("utf-8"))
        return MeshOverlayData(result["polydata"], faces)


def _texture_message(message: str, limit: int = 2048) -> str:
    return str(message).encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _write_texture_result(result: MeshTexturePreparation, directory: Path) -> None:
    document = {**result, "images": {}}
    if "face_uv" in result:
        np.save(directory / "face-uv.npy", result["face_uv"], allow_pickle=False)
        document["face_uv"] = result["face_uv"].size
    for index, (key, image) in enumerate(result["images"].items()):
        np.save(directory / f"image-{index}.npy", image["pixels"], allow_pickle=False)
        document["images"][key] = {
            name: value for name, value in image.items() if name != "pixels"
        }
    (directory / "texture.json").write_text(json.dumps(document), encoding="utf-8")


def _prepare_textures(
    source, gl_maximum, budgets, replacements, directory, channel, saved=None,
    image_overrides=None, refresh=False, obj_index=None,
):
    from ..mesh_texture import MeshTextureCache, MeshTextureError, prepare_obj_textures

    folder = Path(directory)
    started = time.perf_counter()

    def report(phase):
        channel.send((phase, time.perf_counter() - started))

    def resolve(reference, base):
        return replacements.get((base, reference))

    try:
        report("child_ready")
        if saved is None:
            result = prepare_obj_textures(
                source, gl_maximum=gl_maximum, budgets=budgets,
                resolver=resolve, cache=MeshTextureCache(budgets), progress=report,
                image_overrides=image_overrides, _obj_index=obj_index,
            )
        else:
            from ..session.mesh_texture import prepare_saved_textures

            result = prepare_saved_textures(
                source, saved[0], saved[1], gl_maximum=gl_maximum,
                budgets=budgets, progress=report,
                image_overrides=image_overrides, refresh=refresh,
                _obj_index=obj_index,
            )
        report("write")
        _write_texture_result(result, folder)
        report("written")
    except Exception as error:
        code = error.code if isinstance(error, MeshTextureError) else (
            "texture_memory" if isinstance(error, MemoryError) else "texture_worker"
        )
        (folder / "texture-error.json").write_text(
            json.dumps({"code": code, "message": _texture_message(error)}),
            encoding="utf-8",
        )
    finally:
        channel.close()


def _restore_texture_result(
    directory: Path, cancelled: Callable[[], bool], budgets: "MeshTextureBudgets",
    gl_maximum: int,
) -> MeshTexturePreparation:
    from ..mesh_texture import MeshTextureError, _check_cancel, image_cost

    def invalid():
        raise MeshTextureError("texture_transfer", "Invalid texture worker result")

    def asset(value):
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or type(value.get("size")) is not int or value["size"] < 0
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in value["sha256"])
        ):
            invalid()

    _check_cancel(cancelled)
    metadata = directory / "texture.json"
    try:
        if metadata.stat().st_size > 16 * 1024**2:
            invalid()
        document = json.loads(metadata.read_text(encoding="utf-8"))
        if "unresolved_assets" in document:
            requests = document["unresolved_assets"]
            if not isinstance(requests, list) or len(requests) > 32:
                invalid()
            for request in requests:
                if (
                    not isinstance(request, dict)
                    or set(request) != {"base", "reference", "kind"}
                    or request["kind"] not in ("mtl", "image")
                    or any(not isinstance(v, str) or not v or len(v) > 65536
                           for v in request.values())
                    or not Path(request["base"]).is_absolute()
                ):
                    invalid()
            document["unresolved_assets"] = tuple(requests)
        asset(document["source"])
        materials, images = document["materials"], document["images"]
        layout = document.get("face_layout", {})
        if not isinstance(layout, dict) or len(layout) > budgets.max_materials + 1:
            invalid()
        for name, summary in layout.items():
            if (
                not isinstance(name, str) or (name and name not in materials)
                or not isinstance(summary, dict)
                or type(summary.get("face_count")) is not int
                or not 0 < summary["face_count"] <= 2**31 - 1
                or type(summary.get("corner_count")) is not int
                or not 3 * summary["face_count"] <= summary["corner_count"]
                <= 2**31 - 1
                or type(summary.get("complete_uv")) is not bool
            ):
                invalid()
        coverage_bytes = 0
        if "face_uv" in document:
            count = document["face_uv"]
            if (
                type(count) is not int
                or count != sum(item["face_count"] for item in layout.values())
                or not 0 < count <= min(
                    budgets.layer_cache_bytes, budgets.global_cache_bytes,
                    budgets.temporary_cpu_bytes,
                )
            ):
                invalid()
            source = np.load(directory / "face-uv.npy", mmap_mode="r", allow_pickle=False)
            try:
                if source.dtype != np.uint8 or source.shape != (count,):
                    invalid()
                coverage = np.empty(count, dtype=np.uint8)
                for start in range(0, count, _TRANSFER_BYTES):
                    _check_cancel(cancelled)
                    block = source[start:start + _TRANSFER_BYTES]
                    if np.any(block > 1):
                        invalid()
                    coverage[start:start + _TRANSFER_BYTES] = block
                coverage.flags.writeable = False
                document["face_uv"] = coverage
                coverage_bytes = count
            finally:
                source._mmap.close()
        if (
            not isinstance(materials, dict) or len(materials) > budgets.max_materials
            or not isinstance(images, dict) or len(images) > budgets.max_images
            or len(document["diagnostics"]) > budgets.diagnostic_count
            or len(document["libraries"]) > budgets.mtl_records
        ):
            invalid()
        for library in document["libraries"]:
            asset(library)
        for diagnostic in document["diagnostics"]:
            if any(
                not isinstance(diagnostic[name], str)
                or len(diagnostic[name].encode("utf-8")) > budgets.diagnostic_bytes
                for name in ("code", "message")
            ):
                invalid()
        total, samplers = 0, set()
        for index, (key, image) in enumerate(images.items()):
            _check_cancel(cancelled)
            asset(image["asset"])
            if key != os.path.normcase(image["asset"]["path"]):
                invalid()
            source = np.load(
                directory / f"image-{index}.npy", mmap_mode="r", allow_pickle=False
            )
            try:
                if (
                    source.dtype != np.uint8 or source.ndim != 3
                    or not source.flags.c_contiguous or source.shape[2] not in (3, 4)
                    or not 0 < source.shape[0] <= min(gl_maximum, budgets.max_dimension)
                    or not 0 < source.shape[1] <= min(gl_maximum, budgets.max_dimension)
                ):
                    invalid()
                total += source.nbytes
                if (
                    total > budgets.decoded_cpu_bytes
                    or total + coverage_bytes > min(
                        budgets.layer_cache_bytes, budgets.global_cache_bytes,
                    )
                ):
                    invalid()
                costs = image_cost(source.shape[1], source.shape[0], source.shape[2])
                if (
                    type(image["has_transparency"]) is not bool
                    or image["resident_estimate_bytes"]
                    != costs["resident_estimate_bytes"]
                    or type(image["temporary_cpu_bytes"]) is not int
                    or not costs["temporary_cpu_bytes"] <= image["temporary_cpu_bytes"]
                    <= budgets.temporary_cpu_bytes
                ):
                    invalid()
                pixels = np.empty(source.shape, dtype=np.uint8)
                rows = max(1, _TRANSFER_BYTES // (source.shape[1] * source.shape[2]))
                transparent = False
                for start in range(0, source.shape[0], rows):
                    _check_cancel(cancelled)
                    pixels[start:start + rows] = source[start:start + rows]
                    if source.shape[2] == 4:
                        transparent |= bool(
                            np.any(pixels[start:start + rows, :, 3] != 255)
                        )
                if transparent != image["has_transparency"]:
                    invalid()
                pixels.flags.writeable = False
                image["pixels"] = pixels
            finally:
                source._mmap.close()
        for name, material in materials.items():
            if not isinstance(name, str) or material["name"] != name:
                invalid()
            for field in ("image_origin", "library_origin"):
                if field in material and material[field] not in ("automatic", "manual"):
                    invalid()
            if material["library"] is not None:
                asset(material["library"])
                if material["library"] not in document["libraries"]:
                    invalid()
            color = material["diffuse_color"]
            if len(color) != 3 or not all(
                type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1
                for value in color
            ):
                invalid()
            material["diffuse_color"] = tuple(color)
            diffuse = material["diffuse_map"]
            if diffuse is not None:
                if (
                    not isinstance(diffuse["path"], str)
                    or type(diffuse["clamp"]) is not bool
                ):
                    invalid()
                for field in ("scale", "offset"):
                    values = diffuse[field]
                    if len(values) != 3 or not all(
                        type(value) in (int, float) and np.isfinite(value)
                        for value in values
                    ):
                        invalid()
                    diffuse[field] = tuple(values)
            if material["unavailable_reason"] is not None and not isinstance(
                material["unavailable_reason"], str
            ):
                invalid()
            key = material["image_key"]
            if key is not None:
                if (
                    key not in images or diffuse is None
                    or material["unavailable_reason"]
                ):
                    invalid()
                samplers.add((key, diffuse["clamp"]))
        resident = sum(images[key]["resident_estimate_bytes"] for key, _ in samplers)
        temporary = max(
            (image["temporary_cpu_bytes"] for image in images.values()), default=0
        )
        if document["costs"] != {
            "decoded_cpu_bytes": total, "temporary_cpu_bytes": temporary,
            "resident_estimate_bytes": resident,
        } or resident > budgets.resident_estimate_bytes:
            invalid()
        document["libraries"] = tuple(document["libraries"])
        document["diagnostics"] = tuple(document["diagnostics"])
        _check_cancel(cancelled)
        return document
    except MemoryError as error:
        raise MeshTextureError(
            "texture_memory", "Insufficient memory for texture transfer"
        ) from error
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
        if isinstance(error, MeshTextureError):
            raise
        raise MeshTextureError(
            "texture_transfer", "Invalid texture worker result"
        ) from error


def prepare_textures_in_process(
    source: Path, *, gl_maximum: int, cancel_check: Callable[[], bool],
    budgets: "MeshTextureBudgets | None" = None,
    replacements: dict[tuple[str, str], str] | None = None,
    report: Callable[[str, float], None] | None = None,
    saved: tuple | None = None,
    image_overrides: dict[str, str] | None = None,
    refresh: bool = False,
    obj_index: dict | None = None,
) -> MeshTexturePreparation:
    """Transfer only owned CPU pixels/descriptors; terminate unresponsive decoders."""
    from ..mesh_texture import DEFAULT_BUDGETS, MeshTextureError, _check_cancel

    budgets = budgets or DEFAULT_BUDGETS
    if type(gl_maximum) is not int or gl_maximum <= 0:
        raise MeshTextureError("invalid_gl_limit", "Texture dimension limit is invalid")
    if obj_index is not None:
        data = obj_index["index"]
        payload_bytes = len(data["face_uv_bytes"] or b"")
        payload_bytes += sum(len(line) + 64 for line in data["reference_lines"])
        payload_bytes += sum(len(name) + 128 for name in data["face_layout"])
        if payload_bytes > 65536:
            obj_index = None
    with TemporaryDirectory(prefix="tractedit-texture-") as directory:
        started = time.perf_counter()
        exitcode = run_isolated_process(
            _prepare_textures, (str(source), gl_maximum, budgets, replacements or {}, directory),
            cancel_check=lambda: _check_cancel(cancel_check), channel=True,
            progress=(lambda message: report(*message)) if report is not None else None,
            start=start_process, kwargs={
                **({"saved": saved} if saved is not None else {}),
                **({"image_overrides": image_overrides} if image_overrides else {}),
                **({"refresh": True} if refresh else {}),
                **({"obj_index": obj_index} if obj_index is not None else {}),
            },
        )
        if report is not None:
            report("process_wait", time.perf_counter() - started)
        if exitcode:
            raise MeshTextureError(
                "texture_process",
                f"Texture process exited with code {exitcode}",
            )
        folder = Path(directory)
        error = folder / "texture-error.json"
        if error.exists():
            if error.stat().st_size > 8192:
                raise MeshTextureError("texture_transfer", "Invalid worker error")
            record = json.loads(error.read_text(encoding="utf-8"))
            raise MeshTextureError(
                _texture_message(record["code"], 128),
                _texture_message(record["message"]),
            )
        transfer = time.perf_counter()
        result = _restore_texture_result(folder, cancel_check, budgets, gl_maximum)
        if report is not None:
            report("restore", time.perf_counter() - transfer)
        return result
