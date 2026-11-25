# -*- coding: utf-8 -*-
"""Detached mesh workers and derived-overlay coordination."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, Qt, pyqtSignal

from ...workers import CancellableWorkerThread
from ... import file_io
from ...data_contracts import (
    MeshCoordinateChoice, MeshLayer, MeshLoadResult, MeshComponent,
)
from ...mesh_io import MeshLoadCancelled
from .. import mesh_manager

if TYPE_CHECKING:
    from .manager import MeshManager

logger = logging.getLogger(__name__)


class MeshLoaderThread(CancellableWorkerThread):
    """Transfer a detached CPU result once, retaining it until Qt consumes it."""

    progress = pyqtSignal(int, str)
    result_ready = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.choice: MeshCoordinateChoice | None = None
        self.parent_id: str | None = None
        self.remaining_paths = None
        self.published = False
        self.generation = 0
        self.progress_dialog = None
        self._result: MeshLoadResult | None = None
        self._source_association: tuple[dict, str] | None = None
        self.finished_delivered = False

    def take_result(self) -> MeshLoadResult | None:
        self.is_consuming_result = True
        result, self._result = self._result, None
        return result

    def take_source_association(self) -> tuple[dict, str] | None:
        association, self._source_association = self._source_association, None
        return association

    def complete_result(self) -> None:
        self._result = None
        self._source_association = None
        self._result_pending = False
        self.is_consuming_result = False

    def discard_pending_result(self) -> None:
        self.complete_result()

    def run(self) -> None:
        try:
            self.progress.emit(0, "Reading and validating surface...")
            result = mesh_manager.load_mesh(
                self.path, cancel_check=self._cancelled.is_set,
                progress=self.progress.emit,
            )
            if self.is_cancelled:
                return
            self.progress.emit(90, "Preparing surface presentation...")
            if (
                "verified_geometry_sha256" in result
                and result["verified_geometry_sha256"] == result.get("source_sha256")
            ):
                from ...session.io import SessionCancelled, source_stamp
                from ...session.mesh import mesh_attribute_values, mesh_geometry_values
                from ...session.state import _content_digest

                try:
                    signature = _content_digest(
                        [
                            mesh_geometry_values(result["polydata"]),
                            result["space_metadata"],
                            mesh_attribute_values(result["polydata"]),
                        ],
                        self._cancelled.is_set,
                    )
                    if source_stamp(result["path"]) == result["source_stamp"]:
                        record = {
                            "path": str(Path(result["path"]).resolve(strict=True)),
                            "size": result["source_stamp"][2],
                            "sha256": result["source_sha256"],
                        }
                        self._source_association = (record, signature)
                except SessionCancelled:
                    return
                except Exception:
                    self._source_association = None
            if self.is_cancelled:
                return
            self._result = result
            self._result_pending = True
            self.result_ready.emit()
        except MeshLoadCancelled:
            pass
        except Exception as error:
            if not self.is_cancelled:
                self._result_pending = True
                self.error.emit(str(error))


class MeshOverlayLoaderThread(MeshLoaderThread):
    """Prepare one isolated derived mesh, without creating rendering objects."""

    def __init__(self, layer: MeshLayer, component: MeshComponent) -> None:
        from ...mesh.overlays import mesh_geometry_key, overlay_source_snapshot

        super().__init__(layer["path"])
        self.layer = layer
        self.component = component
        self.geometry_key = mesh_geometry_key(layer)
        self.source = overlay_source_snapshot(layer["polydata"])
        self._overlay_result = None

    def take_overlay_result(self):
        self.is_consuming_result = True
        result, self._overlay_result = self._overlay_result, None
        return result

    def complete_result(self) -> None:
        self._overlay_result = None
        self.source = None
        super().complete_result()

    def run(self) -> None:
        from ...mesh.overlays import prepare_display_overlay

        try:
            result = prepare_display_overlay(
                self.source, self.component, self._cancelled.is_set,
            )
            if not self.is_cancelled:
                self._overlay_result = result
                self._result_pending = True
                self.result_ready.emit()
        except MeshLoadCancelled:
            pass
        except Exception as error:
            if not self.is_cancelled:
                self._result_pending = True
                self.error.emit(str(error) or "Insufficient memory for mesh overlay")


class MeshOverlayController:
    """Coordinate derived overlays while the manager owns Qt slots and state."""

    def __init__(self, manager: "MeshManager") -> None:
        self.manager = manager

    @property
    def overlays_busy(self) -> bool:
        manager = self.manager
        return bool(manager._pending_overlays or manager.mw._mesh_overlay_thread is not None)

    def overlay_pending(self, mesh_id: str, component: str) -> bool:
        manager = self.manager
        return (mesh_id, component) in manager._pending_overlays

    def _queue_overlay(self, layer: MeshLayer, component: MeshComponent) -> None:
        manager = self.manager
        key = (layer["id"], component)
        if key in manager._pending_overlays:
            return
        manager._pending_overlays[key] = (layer, object())
        try:
            manager._start_pending_overlay()
        except Exception:
            manager._pending_overlays.pop(key, None)
            raise
        finally:
            manager._update_actions()

    def _start_pending_overlay(self) -> None:
        manager = self.manager
        if (
            not manager._can_mutate() or manager.mw._mesh_overlay_thread is not None
            or not manager._pending_overlays
        ):
            return
        request, (layer, request_token) = next(iter(manager._pending_overlays.items()))
        key, component = request
        worker = mesh_manager.MeshOverlayLoaderThread(layer, component)
        worker.request_token = request_token
        worker.generation = file_io._register_background_worker(
            manager.mw, "_mesh_overlay_thread", "_mesh_overlay_generation", worker,
        )
        worker.result_ready.connect(
            manager._on_overlay_result, Qt.ConnectionType.QueuedConnection,
        )
        worker.error.connect(manager._on_overlay_error, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(
            manager._on_overlay_finished, Qt.ConnectionType.QueuedConnection,
        )
        dialog = mesh_manager.QProgressDialog(
            f"Preparing {component.replace('_', ' ')}...", "Cancel", 0, 0, manager.mw,
        )
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        worker.progress_dialog = dialog
        dialog.canceled.connect(lambda: manager.cancel_overlays(key, component))
        try:
            dialog.show()
            worker.start()
        except Exception:
            manager._pending_overlays.pop((key, component), None)
            worker.cancel()
            manager._close_progress(worker)
            file_io._release_finished_worker(manager.mw, "_mesh_overlay_thread", worker)
            raise

    def _overlay_current(self, worker) -> bool:
        manager = self.manager
        from ...mesh.overlays import mesh_geometry_key

        return (
            isinstance(worker, mesh_manager.MeshOverlayLoaderThread) and manager._can_mutate()
            and manager.mw.mesh_layers.get(worker.layer["id"]) is worker.layer
            and manager._pending_overlays.get(
                (worker.layer["id"], worker.component), (None, None),
            )[1] is worker.request_token
            and worker.geometry_key == mesh_geometry_key(worker.layer)
            and file_io._worker_is_current(
                manager.mw, "_mesh_overlay_thread", "_mesh_overlay_generation",
                worker, worker.generation,
            )
        )

    def _on_overlay_result(self) -> None:
        manager = self.manager
        worker = manager.sender()
        if not isinstance(worker, mesh_manager.MeshOverlayLoaderThread):
            return
        result = worker.take_overlay_result()
        try:
            if result is None or not manager._overlay_current(worker):
                return
            layer, component = worker.layer, worker.component
            preference = layer["presentation"][component]
            previous = preference["enabled"]
            manager.mw._mesh_commit_active = True
            try:
                preference["enabled"] = True
                manager.mw.vtk_panel.publish_mesh_overlay(layer, component, result)
            except Exception:
                preference["enabled"] = previous
                raise
            finally:
                manager.mw._mesh_commit_active = False
        except Exception as error:
            logger.exception("Could not publish mesh overlay")
            manager._report_error(str(error))
        finally:
            manager._complete_overlay_request(worker)
            worker.complete_result()
            manager._close_progress(worker)
            if worker.finished_delivered:
                manager._release_overlay_worker(worker)
            manager._update_actions()

    def _on_overlay_error(self, message: str) -> None:
        manager = self.manager
        worker = manager.sender()
        if isinstance(worker, mesh_manager.MeshOverlayLoaderThread):
            try:
                current = manager._overlay_current(worker)
                manager._complete_overlay_request(worker)
                manager._close_progress(worker)
                if current:
                    manager._report_error(message)
            finally:
                worker.complete_result()
                if worker.finished_delivered:
                    manager._release_overlay_worker(worker)
                manager._update_actions()

    def _on_overlay_finished(self) -> None:
        manager = self.manager
        worker = manager.sender()
        if not isinstance(worker, mesh_manager.MeshOverlayLoaderThread):
            return
        worker.wait()
        worker.finished_delivered = True
        if not worker.has_pending_result:
            manager._complete_overlay_request(worker)
            worker.complete_result()
            manager._close_progress(worker)
            manager._release_overlay_worker(worker)
        manager._update_actions()

    def _release_overlay_worker(self, worker) -> None:
        manager = self.manager
        file_io._release_finished_worker(manager.mw, "_mesh_overlay_thread", worker)
        try:
            manager._start_pending_overlay()
        except Exception as error:
            manager._pending_overlays.clear()
            manager._report_error(str(error))
        manager._update_actions()

    def _complete_overlay_request(self, worker) -> None:
        manager = self.manager
        key = (worker.layer["id"], worker.component)
        pending = manager._pending_overlays.get(key)
        if pending is not None and pending[1] is worker.request_token:
            del manager._pending_overlays[key]

    def cancel_overlays(
        self, mesh_id: str | None = None, component: str | None = None,
        *, update_actions: bool = True,
    ) -> None:
        manager = self.manager
        for key in tuple(manager._pending_overlays):
            if (
                (mesh_id is None or key[0] == mesh_id)
                and (component is None or key[1] == component)
            ):
                del manager._pending_overlays[key]
        worker = manager.mw._mesh_overlay_thread
        if worker is not None and (
            mesh_id is None or worker.layer["id"] == mesh_id
        ) and (component is None or worker.component == component):
            manager.mw._mesh_overlay_generation += 1
            worker.cancel()
            manager._close_progress(worker)
        if update_actions:
            manager._update_actions()

    def _resume_visible_overlays(self, layer: MeshLayer) -> None:
        manager = self.manager
        if layer["visible"]:
            for component in ("boundary_edges", "surface_normals"):
                if (
                    layer["presentation"][component]["enabled"]
                    and manager.mw.vtk_panel.mesh_overlay_data(layer, component) is None
                    and not manager.overlay_pending(layer["id"], component)
                ):
                    manager._queue_overlay(layer, component)
