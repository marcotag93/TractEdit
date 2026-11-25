# -*- coding: utf-8 -*-
"""Texture preparation, verified replacement and GUI publication coordination."""

import logging
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, Qt

from ...file_dialogs import choose_input
from ... import file_io
from ...data_contracts import MeshLayer, MeshTexturePreparation
from .. import mesh_manager
from .overlay_controller import MeshLoaderThread

if TYPE_CHECKING:
    from ...mesh_texture import MeshTextureBudgets
    from .manager import MeshManager

logger = logging.getLogger(__name__)


class MeshTextureLoaderThread(MeshLoaderThread):
    """One isolated CPU preparation; Qt consumes its owned result once."""

    def __init__(
        self, layer: MeshLayer, gl_maximum: int, budgets: "MeshTextureBudgets",
        replacements: dict[tuple[str, str], str],
    ) -> None:
        super().__init__(layer["path"])
        self.layer = layer
        self.gl_maximum = gl_maximum
        self.budgets = budgets
        self.replacements = replacements
        self.saved = None
        self.image_overrides: dict[str, str] = {}
        self.refresh = False
        self._texture_result: MeshTexturePreparation | None = None

    def take_texture_result(self) -> MeshTexturePreparation | None:
        self.is_consuming_result = True
        result, self._texture_result = self._texture_result, None
        return result

    def complete_result(self) -> None:
        self._texture_result = None
        super().complete_result()

    def run(self) -> None:
        from ...mesh.process import prepare_textures_in_process, _texture_message
        from ...mesh_texture import MeshTextureCancelled

        try:
            obj_index = None
            if (
                "obj_texture_index" in self.layer
                and "source_stamp" in self.layer
                and "source_sha256" in self.layer
            ):
                obj_index = {
                    "path": self.layer["path"],
                    "stamp": self.layer["source_stamp"],
                    "sha256": self.layer["source_sha256"],
                    "size": self.layer["source_stamp"][2],
                    "index": self.layer["obj_texture_index"],
                }
            result = prepare_textures_in_process(
                Path(self.path), gl_maximum=self.gl_maximum, budgets=self.budgets,
                replacements=self.replacements, cancel_check=self._cancelled.is_set,
                report=lambda phase, elapsed: self.progress.emit(0, phase),
                **({"saved": self.saved} if self.saved is not None else {}),
                **({"image_overrides": self.image_overrides}
                   if self.image_overrides else {}),
                **({"refresh": True} if self.refresh else {}),
                **({"obj_index": obj_index} if obj_index is not None else {}),
            )
            if self.is_cancelled:
                return
            self._texture_result = result
            self._result_pending = True
            self.result_ready.emit()
        except MeshTextureCancelled:
            pass
        except Exception as error:
            if not self.is_cancelled:
                message = "Insufficient memory for texture preparation" if isinstance(
                    error, MemoryError
                ) else str(error)
                self._result_pending = True
                self.error.emit(_texture_message(message))


class MeshTextureController:
    """Coordinate texture preparation while the manager owns Qt slots and state."""

    def __init__(self, manager: "MeshManager") -> None:
        self.manager = manager

    def texture_control_state(self, mesh_id: str) -> tuple[bool, bool, str, str]:
        """Describe the checkbox from published presentation and worker state."""
        manager = self.manager
        layer = manager.mw.mesh_layers.get(mesh_id)
        if layer is None or Path(layer["path"]).suffix.lower() != ".obj":
            return (
                False, False, "Texture (unavailable)",
                "Diffuse textures are supported for OBJ meshes only.",
            )
        state = manager.texture_status(mesh_id)
        checked = layer.get("texture", {}).get(
            "enabled", manager.mw.vtk_panel.mesh_texture_enabled(mesh_id),
        )
        message = {
            "unloaded": "Texture OFF. Not examined. Show the OBJ diffuse texture.",
            "loading": "Preparing texture. Click again to cancel.",
            "ready": (
                "Texture ON." if checked else "Texture OFF. Prepared assets are cached."
            ),
            "unavailable": (
                "No usable texture. Click to retry after correcting the assets."
            ),
        }[state]
        if state == "unloaded" and "texture" in layer:
            message = (
                "Saved Texture ON. Prepared when the surface becomes visible."
                if checked else
                "Texture OFF. Configured assets will be verified before display."
            )
        detail = manager._texture_messages.get(mesh_id)
        if detail:
            message += "\n" + detail
        configuration = layer.get("texture", {})
        assets = layer.get("texture_assets", {})
        for name, material in islice(configuration.get("materials", {}).items(), 16):
            for field, label in (("library", "MTL"), ("image", "Image")):
                asset = assets.get(material[field])
                if asset is not None:
                    origin = material.get(field + "_origin", "reference")
                    message += f"\n{name} — {label} ({origin}): {asset['path']}"
        surface_enabled = layer["presentation"]["surface"]["enabled"]
        if not surface_enabled:
            message = "Enable Surface to use its texture.\n" + message
        text = {
            "loading": "Texture (loading...)", "unavailable": "Texture (unavailable)",
        }.get(state, "Texture")
        return surface_enabled, checked, text, message

    def _texture_material_names(self, layer: MeshLayer) -> list[str]:
        manager = self.manager
        source = layer["polydata"]
        names = source.GetFieldData().GetAbstractArray("MaterialNames")
        if names is None or not names.IsA("vtkStringArray"):
            return []
        layout = layer.get("texture", {}).get("face_layout")
        result = []
        for index in range(names.GetNumberOfValues()):
            name = names.GetValue(index)
            uv = source.GetPointData().GetArray(name)
            if (
                uv is not None and uv.GetNumberOfComponents() == 2
                and uv.GetNumberOfTuples() == source.GetNumberOfPoints()
                and (layout is None or layout.get(name, {}).get("complete_uv", False))
            ):
                result.append(name)
        return result

    def can_load_texture(self, mesh_id: str | None = None) -> bool:
        manager = self.manager
        if (
            not manager._can_mutate() or manager.texture_busy
            or manager.mw._mesh_loader_thread is not None
        ):
            return False
        layers = (
            [manager.mw.mesh_layers[mesh_id]] if mesh_id in manager.mw.mesh_layers
            else manager.mw.mesh_layers.values() if mesh_id is None else []
        )
        return any(
            Path(layer["path"]).suffix.lower() == ".obj"
            and layer["presentation"]["surface"]["enabled"]
            and bool(manager._texture_material_names(layer))
            for layer in layers
        )

    def load_texture(self, mesh_id: str | None = None) -> None:
        manager = self.manager
        if not manager.can_load_texture(mesh_id):
            return
        if mesh_id is None:
            items = manager.mw.data_tree_widget.selectedItems()
            data = items[0].data(0, Qt.ItemDataRole.UserRole) if items else None
            selected = data.get("id") if isinstance(data, dict) else None
            if selected is not None and manager.can_load_texture(selected):
                mesh_id = selected
            else:
                choices = {
                    f"{layer['display_name']} [{identity[:8]}]": identity
                    for identity, layer in manager.mw.mesh_layers.items()
                    if manager.can_load_texture(identity)
                }
                if len(choices) == 1:
                    mesh_id = next(iter(choices.values()))
                else:
                    choice, accepted = mesh_manager.QInputDialog.getItem(
                        manager.mw, "Load/Replace Texture", "Mesh:",
                        list(choices), 0, False,
                    )
                    if not accepted:
                        return
                    mesh_id = choices[choice]
        if not manager.can_load_texture(mesh_id):
            return
        layer = manager.mw.mesh_layers[mesh_id]
        names = manager._texture_material_names(layer)
        material = names[0]
        if len(names) > 1:
            material, accepted = mesh_manager.QInputDialog.getItem(
                manager.mw, "Load/Replace Texture", "Material:", names, 0, False,
            )
            if not accepted:
                return
        path, _ = choose_input(
            mesh_manager.QFileDialog.getOpenFileName,
            manager.mw, "Load/Replace Texture", str(Path(layer["path"]).parent),
            "Texture images (*.png *.jpg *.jpeg);;All files (*)",
        )
        if (
            not path or not manager.can_load_texture(mesh_id)
            or manager.mw.mesh_layers.get(mesh_id) is not layer
        ):
            return
        if Path(path).suffix.lower() not in (".png", ".jpg", ".jpeg"):
            manager._report_error("Choose a PNG or JPEG texture image.")
            return
        manager.request_texture(
            mesh_id, True, retry=True,
            image_overrides={material: str(Path(path).resolve())},
        )

    def request_texture(
        self, mesh_id: str, enabled: bool, *, retry: bool = False,
        image_overrides: dict[str, str] | None = None,
    ) -> None:
        manager = self.manager
        layer = manager.mw.mesh_layers.get(mesh_id)
        if (
            not manager._can_mutate() or layer is None
            or enabled and not layer["presentation"]["surface"]["enabled"]
        ):
            manager.mw.data_panel_manager.update_mesh_items(mesh_id)
            return
        try:
            if not enabled or manager.texture_status(mesh_id) == "ready" and not retry:
                manager.set_texture_enabled(mesh_id, enabled)
            else:
                maximum = manager.mw.vtk_panel.maximum_texture_size()
                if manager.prepare_texture(
                    mesh_id, gl_maximum=maximum,
                    retry=retry,
                    image_overrides=image_overrides,
                ):
                    manager._texture_requested.add(mesh_id)
                    manager._show_texture_progress(manager.mw._mesh_texture_thread)
        except Exception as error:
            manager._texture_requested.discard(mesh_id)
            manager._texture_messages[mesh_id] = str(error)
            manager._report_error(str(error))
        finally:
            manager._update_actions()

    def _show_texture_progress(self, worker) -> None:
        manager = self.manager
        if worker is None or worker.is_cancelled or worker.layer["id"] not in manager._texture_requested:
            return
        if worker.progress_dialog is not None:
            return
        dialog = mesh_manager.QProgressDialog("Preparing texture...", "Cancel", 0, 0, manager.mw)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        worker.progress_dialog = dialog
        dialog.canceled.connect(lambda: manager.cancel_texture(worker.layer["id"]))
        dialog.show()

    @property
    def texture_busy(self) -> bool:
        manager = self.manager
        return (
            manager.mw._mesh_texture_thread is not None
            or manager._pending_texture is not None
        )

    def texture_status(self, mesh_id: str) -> str:
        """CPU readiness only; texture rendering is a separate publication step."""
        manager = self.manager
        state = manager._texture_states.get(mesh_id, "unloaded")
        if state == "ready" and (
            manager._texture_cache is None or manager._texture_cache.get(mesh_id) is None
        ):
            return "unloaded"
        return state

    def set_texture_enabled(
        self, mesh_id: str, enabled: bool, *, render: bool = True,
    ) -> bool:
        """Switch prepared presentation and persist only a successful transition."""
        manager = self.manager
        if QThread.currentThread() is not manager.thread():
            raise RuntimeError("Texture rendering requires the Qt thread")
        layer = manager.mw.mesh_layers.get(mesh_id)
        if not manager._can_mutate() or layer is None:
            return False
        if enabled and not layer["presentation"]["surface"]["enabled"]:
            return False
        if not enabled:
            manager.cancel_texture(mesh_id)
        result = manager._texture_cache.get(mesh_id) if manager._texture_cache else None
        if enabled and (manager.texture_busy or result is None):
            return False
        previous_bytes = (
            manager._texture_cache.presentation_bytes(mesh_id) if result else 0
        )
        manager.mw._mesh_commit_active = True
        try:
            if enabled:
                from ...visualization.mesh_texture import derived_uv_bytes

                manager._texture_cache.set_presentation_bytes(
                    mesh_id, derived_uv_bytes(layer, result)
                )
            manager.mw.vtk_panel.set_mesh_texture(
                layer, result, enabled, render=render,
                budgets=manager._texture_cache.budgets if manager._texture_cache else None,
            )
            if result is not None:
                manager._texture_cache.set_active(mesh_id, bool(enabled))
            if "texture" in layer:
                layer["texture"]["enabled"] = bool(enabled)
        except Exception:
            if result is not None:
                manager._texture_cache.set_presentation_bytes(mesh_id, previous_bytes)
            raise
        finally:
            manager.mw._mesh_commit_active = False
            manager._update_actions()
        return True

    def restore_texture(self, layer, preparation) -> None:
        """Populate a private session candidate, including its pinned CPU ownership."""
        manager = self.manager
        if preparation is None:
            return
        from ...mesh_texture import MeshTextureCache, validate_texture_source
        from ...visualization.mesh_texture import derived_uv_bytes

        validate_texture_source(layer, preparation)
        if manager._texture_cache is None:
            manager._texture_cache = MeshTextureCache()
        key = layer["id"]
        manager._texture_cache.store(key, preparation, active=True)
        manager._texture_cache.set_presentation_bytes(
            key, derived_uv_bytes(layer, preparation),
        )
        manager.mw.vtk_panel.set_mesh_texture(
            layer, preparation, True, render=False, budgets=manager._texture_cache.budgets,
        )
        manager._texture_states[key] = "ready"

    def _resume_visible_texture(self, layer: MeshLayer) -> None:
        manager = self.manager
        if (
            layer.get("texture", {}).get("enabled", False)
            and layer["visible"] and layer["presentation"]["surface"]["enabled"]
            and not manager.mw.vtk_panel.mesh_texture_enabled(layer["id"])
            and manager.texture_status(layer["id"]) != "loading"
        ):
            manager.request_texture(layer["id"], True)

    def _texture_current(self, worker) -> bool:
        manager = self.manager
        return (
            isinstance(worker, mesh_manager.MeshTextureLoaderThread)
            and manager._can_mutate()
            and manager.mw.mesh_layers.get(worker.layer["id"]) is worker.layer
            and file_io._worker_is_current(
                manager.mw, "_mesh_texture_thread", "_mesh_texture_generation",
                worker, worker.generation,
            )
        )

    def prepare_texture(
        self, mesh_id: str, *, gl_maximum: int, retry: bool = False,
        replacements: dict[tuple[str, str], str] | None = None,
        image_overrides: dict[str, str] | None = None,
    ) -> bool:
        """Internal opt-in CPU request; the caller supplies a GUI-observed GL limit."""
        manager = self.manager
        from ...mesh_texture import MeshTextureCache

        if QThread.currentThread() is not manager.thread():
            raise RuntimeError("Texture requests require the Qt thread")
        layer = manager.mw.mesh_layers.get(mesh_id)
        if (
            not manager._can_mutate()
            or layer is None
            or Path(layer["path"]).suffix.lower() != ".obj"
        ):
            return False
        if type(gl_maximum) is not int or gl_maximum <= 0:
            raise ValueError("Texture dimension limit must be a positive integer")
        if not retry and manager.texture_status(mesh_id) == "ready":
            return True
        if manager._texture_cache is None:
            manager._texture_cache = MeshTextureCache()
        manager.cancel_texture()
        manager._pending_texture = (
            layer, gl_maximum, dict(replacements or {}), retry,
            dict(image_overrides or {}),
        )
        manager._texture_states[mesh_id] = "loading"
        manager._start_pending_texture()
        return True

    def _start_pending_texture(self) -> None:
        manager = self.manager
        if manager.mw._mesh_texture_thread is not None or manager._pending_texture is None:
            return
        layer, gl_maximum, replacements, retry, image_overrides = manager._pending_texture
        manager._pending_texture = None
        mesh_id = layer["id"]
        if not manager._can_mutate() or manager.mw.mesh_layers.get(mesh_id) is not layer:
            manager._texture_states.pop(mesh_id, None)
            return
        cache = manager._texture_cache
        previous = cache.get(mesh_id)
        if previous is not None:
            cache.set_active(mesh_id, True)
        try:
            budgets = cache.budgets
            capacity = max(1, cache.preparation_capacity())
            for evicted_id in cache.reserve_preparation(capacity):
                manager.mw.vtk_panel.discard_mesh_texture(evicted_id)
            available = budgets.global_cache_bytes - cache.nbytes
            worker = mesh_manager.MeshTextureLoaderThread(
                layer, gl_maximum,
                replace(budgets, global_cache_bytes=available), replacements,
            )
            worker.image_overrides = image_overrides
            if "texture" in layer:
                worker.saved = (layer["texture"], layer["texture_assets"])
                worker.refresh = retry and not image_overrides
            worker.generation = file_io._register_background_worker(
                manager.mw, "_mesh_texture_thread", "_mesh_texture_generation", worker
            )
            worker.result_ready.connect(
                manager._on_texture_result, Qt.ConnectionType.QueuedConnection
            )
            worker.error.connect(
                manager._on_texture_error, Qt.ConnectionType.QueuedConnection
            )
            worker.finished.connect(
                manager._on_texture_finished, Qt.ConnectionType.QueuedConnection
            )
            manager._update_actions()
            manager._show_texture_progress(worker)
            worker.start()
        except Exception as error:
            if previous is not None:
                cache.set_active(
                    mesh_id, manager.mw.vtk_panel.mesh_texture_enabled(mesh_id)
                )
            manager._texture_states[mesh_id] = (
                "ready" if previous is not None else "unavailable"
            )
            worker = manager.mw._mesh_texture_thread
            if worker is not None and not worker.isRunning():
                file_io._release_finished_worker(
                    manager.mw, "_mesh_texture_thread", worker
                )
            manager._report_error(str(error))
            manager._update_actions()

    def _publish_texture(self, worker, result: MeshTexturePreparation) -> None:
        """Commit a complete CPU preparation without changing the rendered scene."""
        manager = self.manager
        if QThread.currentThread() is not manager.thread():
            raise RuntimeError("Texture publication requires the Qt thread")
        if not manager._texture_current(worker):
            return
        from ...mesh_texture import validate_texture_source

        validate_texture_source(worker.layer, result)
        mesh_id = worker.layer["id"]
        from ...session.mesh_texture import texture_configuration

        enabled = (
            manager.mw.vtk_panel.mesh_texture_enabled(mesh_id)
            or mesh_id in manager._texture_requested
        )
        configuration, assets = texture_configuration(
            result, enabled,
            worker.layer.get("texture_assets"),
        )
        if not any(
            material["image_key"] is not None
            for material in result["materials"].values()
        ):
            from ...mesh_texture import MeshTextureError
            from ...mesh.process import _texture_message

            detail = "; ".join(item["message"] for item in result["diagnostics"])
            raise MeshTextureError(
                "texture_unavailable",
                _texture_message(f"No usable diffuse image was found. {detail}"),
            )
        previous_state = manager._texture_states.get(mesh_id, "unloaded")
        manager._texture_states[mesh_id] = "ready"
        manager.mw._mesh_commit_active = True
        try:
            with manager._texture_cache.transaction():
                evicted = manager._texture_cache.store(mesh_id, result, active=enabled)
                if enabled:
                    from ...visualization.mesh_texture import derived_uv_bytes

                    manager._texture_cache.set_presentation_bytes(
                        mesh_id, derived_uv_bytes(worker.layer, result)
                    )
                    manager.mw.vtk_panel.set_mesh_texture(
                        worker.layer, result, True, gl_maximum=worker.gl_maximum,
                        budgets=manager._texture_cache.budgets,
                    )
                else:
                    manager.mw.vtk_panel.discard_mesh_texture(mesh_id)
                    manager._texture_cache.set_presentation_bytes(mesh_id, 0)
            for evicted_id in evicted:
                manager.mw.vtk_panel.discard_mesh_texture(evicted_id)
            worker.layer["texture"] = configuration
            worker.layer["texture_assets"] = assets
        except Exception:
            manager._texture_states[mesh_id] = previous_state
            raise
        finally:
            manager.mw._mesh_commit_active = False

    def _texture_failed(self, worker, error=None) -> None:
        manager = self.manager
        mesh_id = worker.layer["id"]
        manager._texture_requested.discard(mesh_id)
        manager._close_progress(worker)
        previous = manager._texture_cache.get(mesh_id)
        manager._texture_states[mesh_id] = "ready" if previous is not None else (
            "unavailable" if error is not None else "unloaded"
        )
        if error is not None:
            if previous is None and "texture" in worker.layer:
                worker.layer["texture"]["enabled"] = False
            manager._texture_messages[mesh_id] = str(error)
            manager._report_error(error)

    def _release_texture_worker(self, worker) -> None:
        manager = self.manager
        if not worker.finished_delivered or worker.has_pending_result:
            return
        manager._close_progress(worker)
        mesh_id = worker.layer["id"]
        if (
            manager._texture_cache is not None
            and manager._texture_cache.get(mesh_id) is not None
        ):
            manager._texture_cache.set_active(
                mesh_id, manager.mw.vtk_panel.mesh_texture_enabled(mesh_id)
            )
        file_io._release_finished_worker(manager.mw, "_mesh_texture_thread", worker)
        manager._start_pending_texture()

    def _on_texture_result(self) -> None:
        manager = self.manager
        worker = manager.sender()
        if not isinstance(worker, mesh_manager.MeshTextureLoaderThread):
            return
        result = worker.take_texture_result()
        try:
            if result is not None and manager._texture_current(worker):
                if manager._resolve_texture_requests(worker, result):
                    return
                manager._publish_texture(worker, result)
                key = worker.layer["id"]
                public_request = key in manager._texture_requested
                manager._texture_requested.discard(key)
                detail = "; ".join(
                    item["message"] for item in result["diagnostics"]
                    if item["code"] != "ignored_mtl_record"
                )[:4096]
                manager._texture_messages[key] = detail
                if public_request and detail:
                    mesh_manager.QMessageBox.warning(
                        manager.mw, "Mesh Texture", "Texture loaded with warnings:\n" + detail,
                    )
        except Exception as error:
            logger.exception("Texture CPU publication failed")
            manager._texture_failed(worker, str(error))
        finally:
            worker.complete_result()
            manager._release_texture_worker(worker)
            manager._update_actions()

    def _resolve_texture_requests(self, worker, result) -> bool:
        manager = self.manager
        if worker.layer["id"] not in manager._texture_requested:
            return False
        replacements = worker.replacements.copy()
        for request in result.get("unresolved_assets", ()):
            manager._close_progress(worker)
            is_image = request["kind"] == "image"
            title = "Locate texture image" if is_image else "Locate MTL library"
            filters = (
                "Texture images (*.png *.jpg *.jpeg);;All files (*)" if is_image
                else "Material libraries (*.mtl);;All files (*)"
            )
            while manager._texture_current(worker):
                path, _ = choose_input(
                    mesh_manager.QFileDialog.getOpenFileName,
                    manager.mw, f"{title}: {request['reference']}", request["base"], filters,
                )
                if not manager._texture_current(worker):
                    return True
                if not path:
                    break
                suffix = Path(path).suffix.lower()
                if is_image and suffix not in (".png", ".jpg", ".jpeg"):
                    message = (
                        f"Choose the PNG or JPEG image referenced by map_Kd: "
                        f"{request['reference']}. An MTL file describes materials; "
                        "it is not an image."
                    )
                elif not is_image and suffix in (".png", ".jpg", ".jpeg"):
                    message = "Choose the MTL material library, then its image when prompted."
                else:
                    break
                mesh_manager.QMessageBox.warning(manager.mw, "Mesh Texture", message)
            else:
                return True
            if path:
                replacements[(request["base"], request["reference"])] = path
        if replacements != worker.replacements:
            manager._pending_texture = (
                worker.layer, worker.gl_maximum, replacements, True,
                worker.image_overrides,
            )
            return True
        return False

    def _on_texture_error(self, error: str) -> None:
        manager = self.manager
        worker = manager.sender()
        if not isinstance(worker, mesh_manager.MeshTextureLoaderThread):
            return
        try:
            if manager._texture_current(worker):
                manager._texture_failed(worker, error)
        finally:
            worker.complete_result()
            manager._release_texture_worker(worker)
            manager._update_actions()

    def _on_texture_finished(self) -> None:
        manager = self.manager
        worker = manager.sender()
        if not isinstance(worker, mesh_manager.MeshTextureLoaderThread):
            return
        worker.wait()
        worker.finished_delivered = True
        if not worker.has_pending_result:
            if manager._texture_states.get(worker.layer["id"]) == "loading" and (
                manager._pending_texture is None
                or manager._pending_texture[0] is not worker.layer
            ):
                manager._texture_failed(worker)
            manager._release_texture_worker(worker)
        manager._update_actions()

    def cancel_texture(
        self, mesh_id: str | None = None, *, update_actions: bool = True,
    ) -> None:
        manager = self.manager
        if mesh_id is None:
            manager._texture_requested.clear()
        else:
            manager._texture_requested.discard(mesh_id)
        pending = manager._pending_texture
        if pending is not None and (mesh_id is None or pending[0]["id"] == mesh_id):
            pending_id = pending[0]["id"]
            cached = (
                manager._texture_cache.get(pending_id) if manager._texture_cache else None
            )
            manager._texture_states[pending_id] = (
                "ready" if cached is not None else "unloaded"
            )
            manager._pending_texture = None
        worker = manager.mw._mesh_texture_thread
        if worker is not None and (mesh_id is None or worker.layer["id"] == mesh_id):
            manager.mw._mesh_texture_generation += 1
            worker.cancel()
            manager._texture_failed(worker)
            manager._close_progress(worker)
        if update_actions:
            manager._update_actions()

    def close_textures(self) -> None:
        manager = self.manager
        manager.cancel_texture()
        worker = manager.mw._mesh_texture_thread
        if worker is not None:
            worker.wait()
            worker.discard_pending_result()
            file_io._release_finished_worker(manager.mw, "_mesh_texture_thread", worker)
        if manager._texture_cache is not None:
            for mesh_id in tuple(manager.mw.mesh_layers):
                manager.mw.vtk_panel.discard_mesh_texture(mesh_id)
            manager._texture_cache.clear()
        manager._texture_states.clear()
        manager._texture_messages.clear()
