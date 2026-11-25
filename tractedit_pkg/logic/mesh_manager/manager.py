# -*- coding: utf-8 -*-
"""Asynchronous surface loading and Qt-thread publication of mesh layers."""

import logging
import os
from typing import TYPE_CHECKING, Literal, Unpack

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QDialog

from ...render_batch import request_render
from ...file_dialogs import choose_input
from ... import file_io
from ...data_contracts import (
    MeshCoordinateChoice, MeshLayer, MeshLoadResult, MeshTexturePreparation,
    MeshComponent, MeshComponentStyle,
)
from ...mesh.model import (
    MAX_MESH_LAYERS,
    create_mesh_layer, find_mesh_layer, mesh_subtree_postorder,
    validate_mesh_forest, validate_mesh_parent,
)
from ...mesh.presentation import (
    validate_mesh_boolean,
    validate_mesh_component,
    validate_mesh_component_style,
)
from ...ui.data_panel.tree import _tree_item_key
from ...utils import iter_tree_items as _tree_items, signals_blocked, ui_tree_transaction
from .. import mesh_manager
from .overlay_controller import MeshOverlayController
from .texture_controller import MeshTextureController

if TYPE_CHECKING:
    from ...main_window import MainWindow
    from ...mesh_texture import MeshTextureCache

logger = logging.getLogger(__name__)


def _texture_controller(manager) -> MeshTextureController:
    controller = getattr(manager, "_textures", None)
    return controller if controller is not None else MeshTextureController(manager)


class MeshManager(QObject):
    """Keep surface presentation separate from scientific editing and history."""

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self.mw = main_window
        self._texture_cache: "MeshTextureCache | None" = None
        self._texture_states: dict[
            str, Literal["unloaded", "loading", "ready", "unavailable"]
        ] = {}
        self._texture_requested: set[str] = set()
        self._texture_messages: dict[str, str] = {}
        self._pending_texture: tuple[
            MeshLayer, int, dict[tuple[str, str], str], bool, dict[str, str]
        ] | None = None
        self._pending_overlays: dict[tuple[str, str], tuple[MeshLayer, object]] = {}
        self._textures = MeshTextureController(self)
        self._overlays = MeshOverlayController(self)

    @property
    def overlays_busy(self) -> bool:
        return self._overlays.overlays_busy

    def overlay_pending(self, mesh_id: str, component: str) -> bool:
        return self._overlays.overlay_pending(mesh_id, component)

    def _queue_overlay(self, layer: MeshLayer, component: MeshComponent) -> None:
        return self._overlays._queue_overlay(layer, component)

    def _start_pending_overlay(self) -> None:
        return self._overlays._start_pending_overlay()

    def _overlay_current(self, worker) -> bool:
        return self._overlays._overlay_current(worker)

    @pyqtSlot()
    def _on_overlay_result(self) -> None:
        return self._overlays._on_overlay_result()

    @pyqtSlot(str)
    def _on_overlay_error(self, message: str) -> None:
        return self._overlays._on_overlay_error(message)

    @pyqtSlot()
    def _on_overlay_finished(self) -> None:
        return self._overlays._on_overlay_finished()

    def _release_overlay_worker(self, worker) -> None:
        return self._overlays._release_overlay_worker(worker)

    def _complete_overlay_request(self, worker) -> None:
        return self._overlays._complete_overlay_request(worker)

    def cancel_overlays(
        self, mesh_id: str | None = None, component: str | None = None,
        *, update_actions: bool = True,
    ) -> None:
        return self._overlays.cancel_overlays(
            mesh_id, component, update_actions=update_actions,
        )

    def _resume_visible_overlays(self, layer: MeshLayer) -> None:
        return self._overlays._resume_visible_overlays(layer)

    def texture_control_state(self, mesh_id: str) -> tuple[bool, bool, str, str]:
        """Describe the checkbox from published presentation and worker state."""
        return self._textures.texture_control_state(mesh_id)

    def _texture_material_names(self, layer: MeshLayer) -> list[str]:
        return self._textures._texture_material_names(layer)

    def can_load_texture(self, mesh_id: str | None = None) -> bool:
        return self._textures.can_load_texture(mesh_id)

    def load_texture(self, mesh_id: str | None = None) -> None:
        return self._textures.load_texture(mesh_id)

    def request_texture(
        self, mesh_id: str, enabled: bool, *, retry: bool = False,
        image_overrides: dict[str, str] | None = None,
    ) -> None:
        return self._textures.request_texture(
            mesh_id, enabled, retry=retry, image_overrides=image_overrides,
        )

    def _show_texture_progress(self, worker) -> None:
        return self._textures._show_texture_progress(worker)

    @property
    def texture_busy(self) -> bool:
        return self._textures.texture_busy

    def texture_status(self, mesh_id: str) -> str:
        """CPU readiness only; texture rendering is a separate publication step."""
        return self._textures.texture_status(mesh_id)

    def set_texture_enabled(
        self, mesh_id: str, enabled: bool, *, render: bool = True,
    ) -> bool:
        """Switch prepared presentation and persist only a successful transition."""
        return self._textures.set_texture_enabled(mesh_id, enabled, render=render)

    def restore_texture(self, layer, preparation) -> None:
        """Populate a private session candidate, including its pinned CPU ownership."""
        return self._textures.restore_texture(layer, preparation)

    def _resume_visible_texture(self, layer: MeshLayer) -> None:
        return self._textures._resume_visible_texture(layer)

    def _texture_current(self, worker) -> bool:
        return self._textures._texture_current(worker)

    def prepare_texture(
        self, mesh_id: str, *, gl_maximum: int, retry: bool = False,
        replacements: dict[tuple[str, str], str] | None = None,
        image_overrides: dict[str, str] | None = None,
    ) -> bool:
        """Internal opt-in CPU request; the caller supplies a GUI-observed GL limit."""
        return self._textures.prepare_texture(
            mesh_id, gl_maximum=gl_maximum, retry=retry,
            replacements=replacements, image_overrides=image_overrides,
        )

    def _start_pending_texture(self) -> None:
        return _texture_controller(self)._start_pending_texture()

    def _publish_texture(self, worker, result: MeshTexturePreparation) -> None:
        """Commit a complete CPU preparation without changing the rendered scene."""
        return _texture_controller(self)._publish_texture(worker, result)

    def _texture_failed(self, worker, error=None) -> None:
        return self._textures._texture_failed(worker, error)

    def _release_texture_worker(self, worker) -> None:
        return self._textures._release_texture_worker(worker)

    @pyqtSlot()
    def _on_texture_result(self) -> None:
        return self._textures._on_texture_result()

    def _resolve_texture_requests(self, worker, result) -> bool:
        return self._textures._resolve_texture_requests(worker, result)

    @pyqtSlot(str)
    def _on_texture_error(self, error: str) -> None:
        return self._textures._on_texture_error(error)

    @pyqtSlot()
    def _on_texture_finished(self) -> None:
        return self._textures._on_texture_finished()

    def cancel_texture(
        self, mesh_id: str | None = None, *, update_actions: bool = True,
    ) -> None:
        return self._textures.cancel_texture(mesh_id, update_actions=update_actions)

    def close_textures(self) -> None:
        return self._textures.close_textures()

    def _can_mutate(self) -> bool:
        return not (
            self.mw._session_busy
            or self.mw._shutdown_requested
            or getattr(self.mw, "_mesh_commit_active", False)
        )

    def _current(self, worker: mesh_manager.MeshLoaderThread) -> bool:
        return (
            isinstance(worker, mesh_manager.MeshLoaderThread)
            and self._can_mutate()
            and file_io._worker_is_current(
                self.mw,
                "_mesh_loader_thread",
                "_mesh_load_generation",
                worker,
                worker.generation,
            )
        )

    def _validate_load_parent(self, parent_id: str | None) -> None:
        validate_mesh_forest(self.mw.mesh_layers)
        if parent_id is not None and (
            not isinstance(parent_id, str) or not parent_id
            or parent_id not in self.mw.mesh_layers
        ):
            raise ValueError("The requested parent mesh no longer exists")

    def _select_duplicate(self, mesh_id: str, parent_id: str | None) -> None:
        self.select(mesh_id)
        if parent_id is not None:
            mesh_manager.QMessageBox.information(
                self.mw, "Mesh already loaded",
                "This mesh is already loaded. "
                "Drag it onto another mesh in the Data Panel to change its parent.",
            )

    def trigger_load(self, *, parent_id: str | None = None) -> None:
        if not self._can_mutate() or self.mw._mesh_loader_thread is not None:
            return
        try:
            self._validate_load_parent(parent_id)
        except ValueError as error:
            self._report_error(str(error))
            return
        destination = (
            self.mw.mesh_layers[parent_id]["display_name"] if parent_id else "Top level"
        )
        if parent_id and sum(
            layer["display_name"] == destination for layer in self.mw.mesh_layers.values()
        ) > 1:
            destination = f"{destination} [{parent_id}]"
        paths, _ = choose_input(
            mesh_manager.QFileDialog.getOpenFileNames,
            self.mw, f"Load Mesh (destination: {destination})", "", "Meshes (*.ply *.vtp *.obj)"
        )
        self._load_selected(iter(paths), parent_id)

    def _load_selected(self, paths, parent_id) -> None:
        for path in paths:
            try:
                if not self._load_selected_path(path, parent_id):
                    return
            except ValueError as error:
                self._report_error(str(error))
                return
            worker = self.mw._mesh_loader_thread
            if worker is not None:
                worker.remaining_paths = paths
                return

    def _load_selected_path(self, path, parent_id) -> bool:
        if not self._can_mutate() or self.mw._mesh_loader_thread is not None:
            return False
        self._validate_load_parent(parent_id)
        if not path:
            return False
        normalized = os.path.normcase(os.path.realpath(path))
        existing = next(
            (
                layer
                for layer in self.mw.mesh_layers.values()
                if os.path.normcase(os.path.realpath(layer["path"])) == normalized
            ),
            None,
        )
        choice = None
        if existing is not None:
            dialog = mesh_manager.MeshImportDialog(
                existing["coordinate_provenance"]["metadata"], self.mw
            )
            dialog.coordinate_combo.setCurrentIndex(
                dialog.coordinate_combo.findData(
                    existing["coordinate_provenance"]["choice"]
                )
            )
            try:
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return False
                choice = dialog.coordinate_combo.currentData()
            finally:
                dialog.deleteLater()
        return self.load(path, choice, parent_id=parent_id)

    def load(
        self, path: str, choice: MeshCoordinateChoice | None = None, *,
        parent_id: str | None = None,
    ) -> bool:
        """Start one surface read; None requests the coordinate dialog on delivery."""
        mw = self.mw
        if not self._can_mutate() or mw._mesh_loader_thread is not None:
            return False
        self._validate_load_parent(parent_id)
        if choice is not None:
            existing = find_mesh_layer(mw.mesh_layers, path, choice)
            if existing is not None:
                self._select_duplicate(existing["id"], parent_id)
                return True
        if len(mw.mesh_layers) >= MAX_MESH_LAYERS:
            raise ValueError(f"Mesh layer limit reached ({MAX_MESH_LAYERS})")
        worker = mesh_manager.MeshLoaderThread(path)
        worker.choice = choice
        worker.parent_id = parent_id
        dialog = mesh_manager.QProgressDialog("Loading mesh...", "Cancel", 0, 100, mw)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        worker.progress_dialog = dialog
        worker.generation = file_io._register_background_worker(
            mw, "_mesh_loader_thread", "_mesh_load_generation", worker
        )
        dialog.canceled.connect(worker.cancel)
        worker.progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        worker.result_ready.connect(self._on_result, Qt.ConnectionType.QueuedConnection)
        worker.error.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._on_finished, Qt.ConnectionType.QueuedConnection)
        try:
            mw._update_action_states()
            dialog.show()
            worker.start()
        except Exception:
            worker.cancel()
            self._close_progress(worker)
            file_io._release_finished_worker(mw, "_mesh_loader_thread", worker)
            mw._update_action_states()
            raise
        return True

    def _close_progress(self, worker: mesh_manager.MeshLoaderThread) -> None:
        try:
            if worker.progress_dialog is not None:
                try:
                    worker.progress_dialog.canceled.disconnect()
                except (TypeError, RuntimeError):
                    pass
                worker.progress_dialog.close()
                worker.progress_dialog.deleteLater()
                worker.progress_dialog = None
        except Exception:
            logger.warning("Could not close mesh progress dialog", exc_info=True)

    def _report_error(self, error: str) -> None:
        try:
            mesh_manager.QMessageBox.critical(self.mw, "Mesh Error", error)
        except Exception:
            logger.warning("Could not display mesh error", exc_info=True)

    def _update_actions(self) -> None:
        try:
            self.mw._update_action_states()
        except Exception:
            logger.warning("Could not update mesh action states", exc_info=True)

    def _release_result_worker(self, worker: mesh_manager.MeshLoaderThread) -> None:
        if not worker.finished_delivered or worker.has_pending_result:
            return
        file_io._release_finished_worker(self.mw, "_mesh_loader_thread", worker)
        paths, worker.remaining_paths = worker.remaining_paths, None
        if (paths is not None and worker.published and not worker.is_cancelled
                and self.mw._mesh_load_generation == worker.generation):
            self._load_selected(paths, worker.parent_id)

    @pyqtSlot(int, str)
    def _on_progress(self, value: int, message: str) -> None:
        worker = self.sender()
        if not self._current(worker):
            return
        try:
            worker.progress_dialog.setLabelText(message)
            worker.progress_dialog.setValue(value)
        except Exception:
            logger.warning("Could not update mesh progress dialog", exc_info=True)

    @pyqtSlot()
    def _on_result(self) -> None:
        worker = self.sender()
        if not isinstance(worker, mesh_manager.MeshLoaderThread):
            return
        result = worker.take_result()
        association = worker.take_source_association()
        try:
            if result is None or not self._current(worker):
                return
            choice = worker.choice
            if choice is None:
                self._close_progress(worker)
                dialog = mesh_manager.MeshImportDialog(result["space_metadata"], self.mw)
                try:
                    if dialog.exec() != QDialog.DialogCode.Accepted:
                        return
                    choice = dialog.coordinate_combo.currentData()
                finally:
                    dialog.deleteLater()
            if self._current(worker):
                if worker.progress_dialog is not None:
                    worker.progress_dialog.setLabelText("Displaying geometry...")
                    worker.progress_dialog.setValue(95)
                    worker.progress_dialog.repaint()
                worker.published = self.publish(
                    result, choice, parent_id=worker.parent_id,
                    source_association=association,
                ) is not None
                if worker.progress_dialog is not None:
                    worker.progress_dialog.setValue(100)
        except Exception as error:
            logger.exception("Mesh publication failed")
            self._report_error(str(error))
        finally:
            worker.complete_result()
            self._close_progress(worker)
            self._release_result_worker(worker)
            self._update_actions()

    @pyqtSlot(str)
    def _on_error(self, error: str) -> None:
        worker = self.sender()
        if not isinstance(worker, mesh_manager.MeshLoaderThread):
            return
        current = self._current(worker)
        try:
            self._close_progress(worker)
            if current:
                self._report_error(error)
        finally:
            worker.complete_result()
            self._release_result_worker(worker)
            self._update_actions()

    @pyqtSlot()
    def _on_finished(self) -> None:
        worker = self.sender()
        if not isinstance(worker, mesh_manager.MeshLoaderThread):
            return
        worker.wait()
        worker.finished_delivered = True
        if not worker.has_pending_result:
            self._close_progress(worker)
        self._release_result_worker(worker)
        self._update_actions()

    def _refresh_tree(self) -> None:
        mw = self.mw
        if mw._data_panel_debounce_timer is not None:
            mw._data_panel_debounce_timer.stop()
        mw._data_panel_update_pending = True
        mw._perform_data_panel_update(refresh_selection=True)
        mw._update_action_states()

    def select(self, mesh_id: str) -> None:
        if mesh_id not in self.mw.mesh_layers:
            return
        tree = self.mw.data_tree_widget
        identity = {"type": "mesh", "id": mesh_id}

        def find_item():
            return next((
                item for item in _tree_items(tree, mesh_only=True)
                if item.data(0, Qt.ItemDataRole.UserRole) == identity
            ), None)

        item = find_item()
        if item is None:
            self._refresh_tree()
            item = find_item()
        if item is not None:
            tree.setCurrentItem(item)
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
            self.mw.data_panel_manager.on_data_item_selected()

    def publish(
        self, result: MeshLoadResult, choice: MeshCoordinateChoice, *,
        parent_id: str | None = None,
        source_association: tuple[dict, str] | None = None,
    ) -> str | None:
        """Publish actor, layer and UI together; restore the scene on failure."""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Mesh publication requires the Qt thread")
        if not self._can_mutate():
            return None
        mw = self.mw
        self._validate_load_parent(parent_id)
        existing = find_mesh_layer(mw.mesh_layers, result["path"], choice)
        if existing is not None:
            self._select_duplicate(existing["id"], parent_id)
            return existing["id"]
        layer = create_mesh_layer(
            result, choice, layers=mw.mesh_layers, parent_id=parent_id,
        )
        panel = mw.vtk_panel

        def rollback():
            mw.mesh_layers.pop(layer["id"], None)
            try:
                panel.remove_mesh_layer(layer["id"], render=False)
            except Exception:
                logger.exception("Could not retire failed mesh actor")

        try:
            with ui_tree_transaction(mw, include_camera=True, rollback=rollback):
                panel.add_mesh_layer(layer, render=False)
                mw.mesh_layers[layer["id"]] = layer
                self.select(layer["id"])
                panel.render_window.Render()
        except Exception:
            try:
                panel.render_window.Render()
            except Exception:
                logger.warning("Could not render restored mesh scene", exc_info=True)
            raise
        if source_association is not None:
            cache = getattr(mw, "_session_source_associations", None)
            if cache is not None:
                cache[f"mesh:{layer['id']}"] = source_association
        self._update_actions()
        return layer["id"]

    def set_parent(self, mesh_id: str, parent_id: str | None) -> None:
        """Move an organizational branch without touching its scene resources."""
        if not self._can_mutate():
            return
        mw = self.mw
        validate_mesh_parent(mw.mesh_layers, mesh_id, parent_id)
        layer = mw.mesh_layers[mesh_id]
        previous = layer.get("parent_id")
        if previous == parent_id:
            return
        had_parent = "parent_id" in layer
        tree = mw.data_tree_widget
        current = tree.currentItem()
        expansion = {
            _tree_item_key(item): (item.isExpanded(), item.isSelected())
            for item in _tree_items(tree)
        }
        current_key = _tree_item_key(current) if current is not None else None
        scroll = (tree.verticalScrollBar().value(), tree.horizontalScrollBar().value())

        def rollback():
            if had_parent:
                layer["parent_id"] = previous
            else:
                layer.pop("parent_id", None)

        try:
            with ui_tree_transaction(mw, rollback=rollback):
                layer["parent_id"] = parent_id
                self._refresh_tree()
                with signals_blocked(tree):
                    for item in _tree_items(tree):
                        if _tree_item_key(item) == current_key:
                            tree.setCurrentItem(item)
                    for item in _tree_items(tree):
                        key = _tree_item_key(item)
                        if key in expansion:
                            item.setSelected(expansion[key][1])
                    for item in _tree_items(tree):
                        key = _tree_item_key(item)
                        if key in expansion:
                            item.setExpanded(expansion[key][0])
        finally:
            tree.doItemsLayout()
            tree.verticalScrollBar().setValue(scroll[0])
            tree.horizontalScrollBar().setValue(scroll[1])
        self._update_actions()

    def cancel_load(self) -> None:
        self.mw._mesh_load_generation += 1
        worker = self.mw._mesh_loader_thread
        if worker is not None:
            worker.cancel()
            self._close_progress(worker)

    def remove(self, mesh_id: str, *, refresh: bool = True) -> None:
        if not self._can_mutate() or mesh_id not in self.mw.mesh_layers:
            return
        self._remove_meshes(
            mesh_subtree_postorder(self.mw.mesh_layers, mesh_id), refresh,
        )

    def _remove_single_mesh(self, mesh_id: str) -> None:
        self.mw.mesh_polyline_manager.remove_mesh(mesh_id, update_actions=False)
        self.mw.surface_mapping_manager.remove(mesh_id, refresh=False)
        self.mw.vtk_panel.remove_mesh_layer(mesh_id, render=False)
        self.cancel_overlays(mesh_id, update_actions=False)
        self.cancel_texture(mesh_id, update_actions=False)
        self._texture_states.pop(mesh_id, None)
        self._texture_messages.pop(mesh_id, None)
        if self._texture_cache is not None:
            self._texture_cache.remove(mesh_id)
        del self.mw.mesh_layers[mesh_id]
        self.mw._session_source_associations.pop(f"mesh:{mesh_id}", None)

    def _remove_meshes(self, mesh_ids: list[str], refresh: bool) -> None:
        worker = self.mw._mesh_loader_thread
        if worker is not None and worker.parent_id in mesh_ids:
            self.cancel_load()
        self.mw._mesh_commit_active = True
        try:
            for mesh_id in mesh_ids:
                self._remove_single_mesh(mesh_id)
        finally:
            self.mw._mesh_commit_active = False
        if refresh:
            self._refresh_tree()
            request_render(self.mw.vtk_panel, self.mw.vtk_panel.render_window)

    def clear(self) -> None:
        if not self._can_mutate():
            return
        mesh_ids = mesh_subtree_postorder(self.mw.mesh_layers)
        self.cancel_load()
        self.cancel_texture(update_actions=False)
        self.cancel_overlays(update_actions=False)
        self._remove_meshes(mesh_ids, True)

    def set_opacity(self, mesh_id: str, value: float) -> None:
        self.set_component_style(mesh_id, "surface", opacity=value)

    def set_visible(self, mesh_id: str, visible: bool) -> None:
        if not self._can_mutate() or mesh_id not in self.mw.mesh_layers:
            return
        visible = validate_mesh_boolean(visible)
        layer = self.mw.mesh_layers[mesh_id]
        previous = layer["visible"]
        if visible == previous:
            return
        self.mw._mesh_commit_active = True
        try:
            self.mw.vtk_panel.set_mesh_layer_visibility(
                mesh_id, visible, layer=layer,
                render=not (
                    layer.get("polylines")
                    or self.mw.mesh_polyline_mesh_id == mesh_id
                ),
            )
            layer["visible"] = visible
        except Exception:
            try:
                self.mw.vtk_panel.set_mesh_layer_visibility(
                    mesh_id, previous, layer=layer, render=False,
                )
            except Exception:
                logger.exception("Could not restore mesh visibility")
            raise
        finally:
            self.mw._mesh_commit_active = False
        if visible:
            self._resume_visible_texture(layer)
            self._resume_visible_overlays(layer)
        else:
            self.cancel_overlays(mesh_id)
        self.mw.mesh_polyline_manager.sync_visibility(mesh_id)
        self.mw.data_panel_manager.update_mesh_items(mesh_id)

    def set_component_visible(
        self, mesh_id: str, component: MeshComponent, visible: bool,
    ) -> None:
        """Apply a component preference without changing the file visibility flag."""
        if not self._can_mutate() or mesh_id not in self.mw.mesh_layers:
            return
        validate_mesh_component(component)
        visible = validate_mesh_boolean(visible)
        layer = self.mw.mesh_layers[mesh_id]
        if component in ("boundary_edges", "surface_normals"):
            if not visible:
                self.cancel_overlays(mesh_id, component)
            elif (
                layer["visible"]
                and self.mw.vtk_panel.mesh_overlay_data(layer, component) is None
            ):
                self._queue_overlay(layer, component)
                return
        preference = layer["presentation"][component]
        previous = preference["enabled"]
        if previous == visible:
            return
        self.mw._mesh_commit_active = True
        try:
            preference["enabled"] = visible
            rendered = self.mw.vtk_panel.set_mesh_layer_visibility(
                mesh_id, layer["visible"], layer=layer, render=layer["visible"],
            )
        except Exception:
            preference["enabled"] = previous
            raise
        finally:
            self.mw._mesh_commit_active = False
        if component == "surface":
            self.mw.mesh_polyline_manager.sync_visibility(mesh_id, render=not rendered)
            if not visible:
                self.cancel_texture(mesh_id)
            else:
                self._resume_visible_texture(layer)
            self._update_actions()
        else:
            self.mw.data_panel_manager.update_mesh_items(mesh_id)

    def set_component_style(
        self, mesh_id: str, component: MeshComponent,
        **style: Unpack[MeshComponentStyle],
    ) -> None:
        """Configure an inactive component or atomically update its resident actor."""
        if not self._can_mutate() or mesh_id not in self.mw.mesh_layers:
            return
        values = validate_mesh_component_style(component, style)
        layer = self.mw.mesh_layers[mesh_id]
        target = layer if component == "surface" else layer["presentation"][component]
        changes = {
            field: value for field, value in values.items() if target[field] != value
        }
        if not changes:
            return
        if component == "surface":
            panel = self.mw.vtk_panel
            setters = {"color": panel.set_mesh_color, "opacity": panel.set_mesh_opacity}
            previous = {field: target[field] for field in changes}
            self.mw._mesh_commit_active = True
            try:
                for field, value in changes.items():
                    setters[field](mesh_id, value, render=False)
                target.update(changes)
                panel.render_window.Render()
            except Exception:
                target.update(previous)
                for field in changes:
                    try:
                        setters[field](mesh_id, target[field], render=False)
                    except Exception:
                        logger.exception("Could not restore mesh %s", field)
                raise
            finally:
                self.mw._mesh_commit_active = False
        else:
            self.mw._mesh_commit_active = True
            try:
                self.mw.vtk_panel.set_mesh_component_style(mesh_id, component, changes)
            finally:
                self.mw._mesh_commit_active = False
            target.update(changes)
        if component == "surface":
            self.mw.mesh_polyline_manager.sync_visibility(mesh_id, render=False)
        self.mw.data_panel_manager.update_mesh_properties()

    def change_color(self, mesh_id: str, component: MeshComponent = "surface") -> None:
        if not self._can_mutate() or mesh_id not in self.mw.mesh_layers:
            return
        validate_mesh_component(component)
        layer = self.mw.mesh_layers[mesh_id]
        style = layer if component == "surface" else layer["presentation"][component]
        color = mesh_manager.QColorDialog.getColor(
            QColor.fromRgbF(*style["color"]),
            self.mw,
            "Mesh Color" if component == "surface" else (
                f"{component.replace('_', ' ').capitalize()} Color "
                f"— {layer['display_name']}"
            ),
        )
        if color.isValid() and self.mw.mesh_layers.get(mesh_id) is layer:
            try:
                self.set_component_style(
                    mesh_id, component,
                    color=(color.redF(), color.greenF(), color.blueF()),
                )
            except Exception as error:
                logger.exception("Could not update mesh color")
                self._report_error(str(error))

    def set_color(self, mesh_id: str, color: tuple[float, float, float]) -> None:
        self.set_component_style(mesh_id, "surface", color=color)

    def fit(self, mesh_id: str) -> None:
        if self._can_mutate() and mesh_id in self.mw.mesh_layers:
            self.mw.vtk_panel.fit_mesh_in_view(mesh_id)
