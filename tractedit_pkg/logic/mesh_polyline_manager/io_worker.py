# -*- coding: utf-8 -*-
"""Polyline workers, import/export and transactional batch publication."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QDialog

from ... import file_io
from ...file_dialogs import choose_input
from ..mesh_manager import MeshLoaderThread
from ...mesh.polylines import (
    POLYLINE_STYLE_FIELDS,
    validate_polylines,
)
from ...utils import ui_tree_transaction
from .. import mesh_polyline_manager

if TYPE_CHECKING:
    from ...data_contracts import MeshLayer, MeshPolyline
    from ...mesh.polyline_csv import PolylineCSVData

logger = logging.getLogger(__name__)


class PolylineCSVThread(MeshLoaderThread):
    """Read or write a detached coordinate table without rendering objects."""

    def __init__(
        self,
        path: str | Path,
        mesh_id: str | None,
        target: MeshLayer | list[MeshPolyline],
        *,
        record: PolylineCSVData | None = None,
        options: dict | None = None,
        extra_paths: tuple[str, ...] = (),
    ) -> None:
        super().__init__(str(path))
        self.mesh_id = mesh_id
        self.target = target
        self.record = record
        self.options = options or {}
        self.extra_paths = extra_paths
        self._csv_result: PolylineCSVData | list[PolylineCSVData] | bool | None = None

    def take_csv_result(self) -> PolylineCSVData | list[PolylineCSVData] | bool | None:
        self.is_consuming_result = True
        result, self._csv_result = self._csv_result, None
        return result

    def complete_result(self) -> None:
        self._csv_result = None
        super().complete_result()

    def run(self) -> None:
        from ...mesh.polyline_csv import (
            PolylineCSVCancelled,
            load_polyline_csv,
            save_polyline_csv,
        )

        try:
            if self.record is None:
                result = []
                total = 0
                paths = (self.path, *self.extra_paths)
                vtp_space = self.options.get("space", "from_file")
                if any(Path(path).suffix.lower() != ".vtp" for path in paths):
                    vtp_space = "from_file"
                for path in paths:
                    if self.is_cancelled:
                        return
                    if Path(path).suffix.lower() == ".vtp":
                        batch = mesh_polyline_manager.load_polyline_vtp(
                            path, space=vtp_space,
                            cancel_check=self._cancelled.is_set, mark_default_color=True,
                        )
                    else:
                        batch = [load_polyline_csv(
                            path, **self.options, cancel_check=self._cancelled.is_set,
                            mark_default_color=True,
                        )]
                    total += sum(len(item["points"]) for item in batch)
                    if len(result) + len(batch) > mesh_polyline_manager.MAX_POLYLINES or total > mesh_polyline_manager.MAX_POLYLINE_POINTS:
                        raise ValueError("Polyline container capacity exceeded")
                    result.extend(batch)
            elif Path(self.path).suffix.lower() == ".csv":
                save_polyline_csv(
                    self.path,
                    **self.record,
                    **self.options,
                    cancel_check=self._cancelled.is_set,
                )
                result = True
            else:
                mesh_polyline_manager.save_polyline(
                    self.path, self.record["points"], self.record["color"],
                    **self.options, cancel_check=self._cancelled.is_set,
                )
                result = True
            if not self.is_cancelled:
                self._csv_result = result
                self._result_pending = True
                self.result_ready.emit()
        except PolylineCSVCancelled:
            pass
        except Exception as error:
            if not self.is_cancelled:
                message = f"{path}: {error}" if self.extra_paths else str(error)
                self._result_pending = True
                self.error.emit(message)


class PolylineIOMixin:
    """Annotation import, export and transactional batch publication."""

    @property
    def io_busy(self) -> bool:
        return self.mw._polyline_io_thread is not None or bool(self._polygon and self._polygon.busy)

    def _target(self, mesh_id: str | None) -> MeshLayer | list[MeshPolyline] | None:
        return (
            self.mw.standalone_polylines
            if mesh_id is None
            else self.mw.mesh_layers.get(mesh_id)
        )

    def can_import(self, mesh_id: str | None = None) -> bool:
        return (
            self._target(mesh_id) is not None
            and not self.active
            and not self.io_busy
            and self.mw.mesh_manager._can_mutate()
        )

    def trigger_import(self, mesh_id: str | None = None) -> None:
        from ...ui.polyline_csv_dialog import PolylineImportDialog

        if not self.can_import(mesh_id):
            return
        self._request_generation += 1
        request = self._request_generation
        target = self._target(mesh_id)
        selected, _ = choose_input(
            mesh_polyline_manager.QFileDialog.getOpenFileNames, self.mw, "Import Polyline", "",
            "Polylines (*.csv *.txt *.vtp);;All files (*)",
        )
        if not selected:
            return
        paths = tuple(selected)
        dialog = PolylineImportDialog(self.mw, paths)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            if (
                request == self._request_generation
                and self._target(mesh_id) is target
                and self.can_import(mesh_id)
            ):
                self.load_csv(paths[0], mesh_id, extra_paths=paths[1:], **dialog.options())
        finally:
            dialog.deleteLater()

    def load_csv(self, path: str, mesh_id: str | None = None, *, extra_paths=(), **options) -> bool:
        if not self.can_import(mesh_id):
            return False
        self._request_generation += 1
        try:
            self.check_capacity(mesh_id, 2)
            worker = mesh_polyline_manager.PolylineCSVThread(
                path, mesh_id, self._target(mesh_id), options=options, extra_paths=extra_paths
            )
            self._start_io(worker, "Reading polyline coordinates...")
        except Exception as error:
            logger.exception("Could not start polyline import")
            mesh_polyline_manager.QMessageBox.critical(self.mw, "Polyline Import Error", str(error))
            return False
        return True

    def _start_io(self, worker: mesh_polyline_manager.PolylineCSVThread, message: str) -> None:
        mw = self.mw
        dialog = mesh_polyline_manager.QProgressDialog(message, "Cancel", 0, 0, mw)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        worker.progress_dialog = dialog
        worker.generation = file_io._register_background_worker(
            mw,
            "_polyline_io_thread",
            "_polyline_io_generation",
            worker,
        )
        dialog.canceled.connect(worker.cancel)
        worker.result_ready.connect(
            self._on_csv_result, Qt.ConnectionType.QueuedConnection
        )
        worker.error.connect(self._on_csv_error, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(
            self._on_csv_finished, Qt.ConnectionType.QueuedConnection
        )
        try:
            mw._update_action_states()
            dialog.show()
            worker.start()
        except Exception:
            worker.cancel()
            mw.mesh_manager._close_progress(worker)
            file_io._release_finished_worker(mw, "_polyline_io_thread", worker)
            raise

    def _io_current(self, worker) -> bool:
        return (
            isinstance(worker, mesh_polyline_manager.PolylineCSVThread)
            and file_io._worker_is_current(
                self.mw,
                "_polyline_io_thread",
                "_polyline_io_generation",
                worker,
                worker.generation,
            )
            and self._target(worker.mesh_id) is worker.target
            and self.mw.mesh_manager._can_mutate()
            and not self.active
        )

    def _on_csv_result(self) -> None:
        worker = self.sender()
        if not isinstance(worker, mesh_polyline_manager.PolylineCSVThread):
            return
        result = worker.take_csv_result()
        try:
            if result is None or not self._io_current(worker):
                return
            if isinstance(result, list):
                self.publish_polylines(result, worker.mesh_id)
            elif worker.record is None:
                self.publish_csv(result, worker.mesh_id)
            else:
                self.mw.vtk_panel.update_status(
                    f"Polyline exported: {Path(worker.path).name}",
                )
        except Exception as error:
            logger.exception("Could not publish polyline CSV")
            mesh_polyline_manager.QMessageBox.critical(self.mw, "Polyline I/O Error", str(error))
        finally:
            worker.complete_result()
            self.mw.mesh_manager._close_progress(worker)
            if worker.finished_delivered:
                file_io._release_finished_worker(self.mw, "_polyline_io_thread", worker)
            self.mw._update_action_states()

    def _on_csv_error(self, error: str) -> None:
        worker = self.sender()
        if not isinstance(worker, mesh_polyline_manager.PolylineCSVThread):
            return
        current = self._io_current(worker)
        worker.complete_result()
        self.mw.mesh_manager._close_progress(worker)
        if worker.finished_delivered:
            file_io._release_finished_worker(self.mw, "_polyline_io_thread", worker)
        if current:
            mesh_polyline_manager.QMessageBox.critical(self.mw, "Polyline I/O Error", error)

    def _on_csv_finished(self) -> None:
        worker = self.sender()
        if not isinstance(worker, mesh_polyline_manager.PolylineCSVThread):
            return
        worker.wait()
        worker.finished_delivered = True
        if not worker.has_pending_result:
            self.mw.mesh_manager._close_progress(worker)
        file_io._release_finished_worker(self.mw, "_polyline_io_thread", worker)

    def cancel_io(self, mesh_id: str | None = None, *, all_targets=False) -> None:
        self._request_generation += 1
        worker = self.mw._polyline_io_thread
        if worker is not None and (all_targets or worker.mesh_id == mesh_id):
            self.mw._polyline_io_generation += 1
            worker.cancel()
            self.mw.mesh_manager._close_progress(worker)

    def publish_csv(
        self,
        data: PolylineCSVData,
        mesh_id: str | None = None,
    ) -> str | None:
        """Commit one validated annotation and its tree/actors as a transaction."""
        identities = self.publish_polylines([data], mesh_id)
        return identities[0] if identities else None

    def publish_polylines(
        self, data: list[PolylineCSVData], mesh_id: str | None = None,
        *, confirming_polygon=False,
    ) -> list[str]:
        """Publish a complete annotation batch or retain the previous scene."""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("Polyline publication requires the Qt thread")
        mw = self.mw
        if (
            (self.active and not confirming_polygon)
            or not mw.mesh_manager._can_mutate()
            or self._target(mesh_id) is None
        ):
            return []
        records = self.container(mesh_id)
        if not data or len(records) + len(data) > mesh_polyline_manager.MAX_POLYLINES:
            raise ValueError("Polyline container capacity exceeded")
        self.check_capacity(mesh_id, sum(len(item["points"]) for item in data))
        is_polygon = "region" in data[0]
        names = {
            record["name"] for record in records if ("region" in record) == is_polygon
        }
        incoming = []
        colors = self._available_colors(data)
        for item in data:
            name, number = item["name"], 2
            while name in names:
                suffix = f" ({number})"
                name = item["name"][: 4096 - len(suffix)] + suffix
                number += 1
            names.add(name)
            incoming.append({**item, "id": str(uuid4()), "name": name})
            if incoming[-1].pop("auto_color", False):
                incoming[-1]["color"] = next(colors)
        validate_polylines(incoming)
        identities = {record["id"] for record in incoming}
        if any(record["id"] in identities for _, record in self.iter_records()):
            raise ValueError("Duplicate polyline identity")
        prepared = [
            (record, mesh_polyline_manager.create_mesh_polyline_actor(
                record["points"], record["color"],
                allow_duplicate_adjacent="region" in record,
                **{key: record[key] for key in POLYLINE_STYLE_FIELDS},
            )) for record in incoming
        ]
        panel = mw.vtk_panel
        empty = not (
            mw.mesh_layers
            or mw.standalone_polylines
            or mw.roi_layers
            or mw.tractogram_data is not None
            or mw.anatomical_image_data is not None
            or mw.odf_data is not None
            or mw.parcellation_data is not None
        )

        def rollback():
            records[:] = [item for item in records if item["id"] not in identities]
            if mesh_id is not None and not records:
                mw.mesh_layers[mesh_id].pop("polylines", None)
            for record, actor in prepared:
                self.actors.pop(record["id"], None)
                self._release_actor(actor)

        try:
            with ui_tree_transaction(
                mw, include_camera=True, rollback=rollback, refresh_selection=False,
            ):
                for record, actor in prepared:
                    self._publish_actor(mesh_id, record, actor=actor)
                if mesh_id is not None:
                    mw.mesh_layers[mesh_id].setdefault("polylines", records)
                records.extend(incoming)
                mw.mesh_manager._refresh_tree()
                if empty:
                    self._fit_bounds(np.column_stack((
                        np.min([r["points"].min(axis=0) for r in incoming], axis=0),
                        np.max([r["points"].max(axis=0) for r in incoming], axis=0),
                    )))
                else:
                    panel.scene.ResetCameraClippingRange()
                panel.render_window.Render()
                mw._mesh_commit_active = False
                mw.data_panel_manager.on_data_item_selected()
                mw._update_action_states()
        except Exception:
            try:
                mw.surface_mapping_manager.refresh_panel()
                mw.data_panel_manager.on_data_item_selected()
            except Exception:
                logger.warning(
                    "Could not refresh restored polyline properties", exc_info=True
                )
            raise
        self._remember(mesh_id, "add", incoming, len(records) - len(incoming))
        return [record["id"] for record in incoming]

    def export_sources(self):
        from ...session.state import SOURCE_FIELDS

        sources = [
            getattr(self.mw, field)
            for field, _ in SOURCE_FIELDS.values()
            if getattr(self.mw, field, None)
        ]
        for layer in self.mw.mesh_layers.values():
            sources.append(layer["path"])
            sources.extend(
                asset["path"] for asset in layer.get("texture_assets", {}).values()
            )
        sources.extend(
            layer.get("path") or layer.get("source_path")
            for layer in self.mw.roi_layers.values()
            if layer.get("path") or layer.get("source_path")
        )
        if self.mw.session_path:
            sources.append(self.mw.session_path)
        return sources

    def export(self, mesh_id: str | None, identity: str) -> None:
        record = self._record(mesh_id, identity)
        if record is None or not self.mw.mesh_manager._can_mutate() or self.io_busy:
            return
        if "region" in record:
            self.polygon_editor().export(self.mw.mesh_layers[mesh_id], record)
            return
        path, selected = mesh_polyline_manager.QFileDialog.getSaveFileName(
            self.mw,
            "Export Polyline",
            record["name"] + ".vtp",
            "VTK XML PolyData (*.vtp);;VTK PolyData (*.vtk);;"
            "Wavefront OBJ (*.obj);;PLY Edges (*.ply);;"
            "TrackVis - geometry only (*.trk);;TRX - geometry only (*.trx);;"
            "CSV XYZ - coordinates only (*.csv);;"
            "CSV TractEdit - coordinates and presentation (*.csv)",
        )
        if (
            not path
            or not self.mw.mesh_manager._can_mutate()
            or self.io_busy
            or self._record(mesh_id, identity) is not record
        ):
            return
        if not Path(path).suffix:
            path += next(
                (
                    suffix
                    for suffix in (
                        ".vtp", ".vtk", ".obj", ".ply", ".csv", ".trk", ".trx",
                    )
                    if "*" + suffix in selected
                ),
                ".vtp",
            )
            if (
                Path(path).exists()
                and mesh_polyline_manager.QMessageBox.question(
                    self.mw,
                    "Replace Polyline File?",
                    f"Replace {Path(path).name}?",
                    mesh_polyline_manager.QMessageBox.StandardButton.Yes | mesh_polyline_manager.QMessageBox.StandardButton.No,
                    mesh_polyline_manager.QMessageBox.StandardButton.No,
                )
                != mesh_polyline_manager.QMessageBox.StandardButton.Yes
            ):
                return
        if (
            not self.mw.mesh_manager._can_mutate()
            or self.io_busy
            or self._record(mesh_id, identity) is not record
        ):
            return
        sources = self.export_sources()
        try:
            snapshot = {key: value for key, value in record.items() if key != "id"}
            snapshot["points"] = record["points"].copy()
            options = {"protected_paths": sources}
            if Path(path).suffix.lower() == ".csv":
                options["profile"] = "tractedit" if "CSV TractEdit" in selected else "xyz"
            else:
                options["reference_grid"] = (
                    getattr(self.mw, "anatomical_reference_grid", None)
                    or getattr(self.mw, "tractogram_reference_grid", None)
                )
            worker = mesh_polyline_manager.PolylineCSVThread(
                path, mesh_id, self._target(mesh_id), record=snapshot, options=options,
            )
            worker.record_id = identity
            self._start_io(worker, "Writing polyline coordinates...")
        except Exception as error:
            logger.exception("Could not export mesh polyline")
            mesh_polyline_manager.QMessageBox.critical(self.mw, "Polyline Export Error", str(error))

    def _publish_actor(
        self, mesh_id: str | None, record: MeshPolyline, *, actor=None
    ) -> None:
        actor = (
            actor
            if actor is not None
            else mesh_polyline_manager.create_mesh_polyline_actor(
                record["points"],
                record["color"],
                allow_duplicate_adjacent="region" in record,
                **{key: record[key] for key in POLYLINE_STYLE_FIELDS},
            )
        )
        visible = record["visible"] and (
            mesh_id is None or self.mw.mesh_layers[mesh_id]["visible"]
        )
        if "region" in record:
            from ...mesh.polygon_io import ras_points

            layer = self.mw.mesh_layers[mesh_id]
            surface = self.polygon_editor().surface(layer)
            region = record["region"]
            loop = region["loop"]
            if (region["faces"].max() >= len(surface.faces)
                    or loop.max() >= len(surface.vertices)
                    or not np.array_equal(record["points"], ras_points(surface, loop, layer))):
                raise ValueError("Saved surface polygon disagrees with original triangles")
        actor.SetVisibility(visible)
        actor.polyline_markers.SetVisibility(visible and record["show_points"])
        try:
            self.mw.vtk_panel.scene.AddActor(actor)
            self.mw.vtk_panel.scene.AddActor(actor.polyline_markers)
            if "region" in record:
                layer = self.mw.mesh_layers[mesh_id]
                actor.patch = self.polygon_editor().actor(layer, record["region"], record["color"])
                actor.patch.SetVisibility(visible)
                self.mw.vtk_panel.scene.AddActor(actor.patch)
        except Exception:
            self._release_actor(actor)
            raise
        self.actors[record["id"]] = actor

    def restore(self, mesh_id: str | None, *, verified_surface=None) -> None:
        records = self.container(mesh_id)
        validate_polylines(records)
        published = []
        try:
            for record in records:
                if "region" in record and verified_surface is None:
                    editor = self.polygon_editor()
                    surface = editor.surface(self.mw.mesh_layers[mesh_id])
                    editor.work(
                        "Checking saved surface territory...",
                        lambda check: surface.verify_region(record["region"], check),
                    )
                if record["id"] in self.actors:
                    raise ValueError("Duplicate polyline identity")
                self._publish_actor(mesh_id, record)
                published.append(record["id"])
        except Exception:
            for identity in published:
                self._release_actor(self.actors.pop(identity))
            raise
