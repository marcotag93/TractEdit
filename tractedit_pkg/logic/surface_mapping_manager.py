# -*- coding: utf-8 -*-
"""GUI ownership, cancellation and atomic publication of surface analyses."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from PyQt6.QtCore import QEventLoop, QObject, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QTreeWidgetItem

from ..workers import CancellableWorkerThread

if TYPE_CHECKING:
    from ..surface_mapping import SurfaceMappingParameters
    from ..ui.surface_mapping_panel import SurfaceMappingPanel

logger = logging.getLogger(__name__)

_PROGRESS_PHASES = (
    "staging",
    "spawn",
    "geometry",
    "triangulation",
    "geometry_validation",
    "locator",
    "query",
    "aggregation",
    "presentation",
    "restore",
)
_DEFAULT_DISPLAY = {
    "visible": True, "markers_enabled": True, "qc_enabled": False,
    "map_enabled": False, "density_enabled": False,
    "endpoint_enabled": True, "intersections_enabled": False,
}


class SurfaceMappingWorker(CancellableWorkerThread):
    result_ready = pyqtSignal()
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, mw, layer, parameters, token, selection_version, scope_ids=None):
        from ..surface_mapping import surface_geometry_key

        super().__init__()
        self.layer = {
            "id": layer["id"],
            "polydata": layer["polydata"],
            "path": layer.get("path", ""),
            "source_sha256": layer.get("source_sha256"),
            "source_to_ras": np.array(
                layer["source_to_ras"], dtype=np.float64, copy=True
            ),
        }
        self.geometry_key = surface_geometry_key(layer)
        self.streamlines = mw.tractogram_data
        self.tractogram_version = getattr(mw, "_tractogram_data_version", 0)
        self.visibility_version = mw._visibility_version
        self.selection_version = selection_version
        self.live_ids = ()
        if scope_ids is None:
            visible = mw.visible_indices
            if parameters.scope == "selected_visible":
                self.live_ids = mw.selected_streamline_indices.intersection(visible)
            elif self.streamlines is not None and len(visible) == len(self.streamlines):
                self.live_ids = range(len(self.streamlines))
            else:
                self.live_ids = frozenset(visible)
        self.parameters, self.token = parameters, token
        self.scope_ids = scope_ids
        self.request = None
        self.display_preferences = None
        self.trx_owner = mw.trx_file_reference
        self._bundle = None

    @property
    def has_pending_result(self):
        return self._bundle is not None

    def take_result(self):
        bundle, self._bundle = self._bundle, None
        return bundle

    def discard_pending_result(self):
        bundle = self.take_result()
        if bundle is not None:
            bundle.close()

    def run(self):
        from ..surface_mapping import (
            SurfaceMappingCancelled,
            SurfaceMappingRequest,
            _check_cancel,
        )
        from ..surface_mapping.process import run_surface_mapping

        try:

            def frozen_ids():
                for index, value in enumerate(self.live_ids):
                    if index % 4096 == 0:
                        _check_cancel(self._cancelled.is_set)
                    yield value

            ids = (
                self.scope_ids if self.scope_ids is not None
                else np.fromiter(frozen_ids(), dtype=np.int64)
            )
            _check_cancel(self._cancelled.is_set)
            request = SurfaceMappingRequest(
                self.layer["id"],
                self.geometry_key,
                str(id(self.streamlines)),
                self.tractogram_version,
                self.visibility_version,
                ids,
                self.parameters,
                (
                    self.selection_version
                    if self.parameters.scope == "selected_visible"
                    else None
                ),
                self.token,
            )
            self.request = request
            bundle = run_surface_mapping(
                self.layer,
                request,
                self.streamlines,
                self._cancelled.is_set,
                self.progress.emit,
            )
            self._bundle = bundle
            if self.is_cancelled:
                self.discard_pending_result()
            else:
                self.result_ready.emit()
        except SurfaceMappingCancelled:
            self.discard_pending_result()
        except Exception as error:
            self.discard_pending_result()
            if not self.is_cancelled:
                logger.exception("Surface mapping failed")
                self.failed.emit(str(error)[:4096] or type(error).__name__)
        finally:
            self.streamlines = self.live_ids = None


class SurfaceMappingManager(QObject):
    changed = pyqtSignal()

    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = "idle"
        self.message = "Choose mapping parameters."
        self.mesh_id = None
        self.parameters = None
        self.bundle = None
        self.presentation = None
        self.stale = False
        self.recipe_ids = None
        self.recipe_display = _DEFAULT_DISPLAY.copy()
        self._exporting = False
        self._worker = None
        self._token = None
        self._scope_visibility_version = None
        self._progress_step = None
        self._progress_maximum = 0
        self._panel = None
        self._closed = False
        self.selection_version = 0
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.check_current)

    @property
    def busy(self):
        return self._worker is not None

    @property
    def current(self):
        return bool(
            self.bundle is not None
            and not self.stale
            and self._owns(self.bundle.result.request)
            and (
                self.bundle.companion is None
                or self._owns(self.bundle.companion.result.request)
            )
        )

    def _incompatible(self):
        return bool(
            self._closed
            or getattr(self.mw, "_session_busy", False)
            or getattr(self.mw, "_shutdown_requested", False)
            or getattr(self.mw, "_loader_thread", None) is not None
        )

    def unavailable_reason(self, mesh_id):
        if self._incompatible() or getattr(self.mw, "_mesh_commit_active", False):
            return "Wait for data replacement or session restore to finish."
        if self.busy:
            return "A surface mapping computation is already active."
        layer = self.mw.mesh_layers.get(mesh_id)
        if (
            layer is None
            or not layer["polydata"].GetNumberOfPoints()
            or not (
                layer["polydata"].GetNumberOfPolys()
                + layer["polydata"].GetNumberOfStrips()
            )
        ):
            return "Choose a valid surface mesh."
        matrix = np.asarray(layer["source_to_ras"])
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0, 0, 0, 1])
            or np.linalg.slogdet(matrix[:3, :3])[0] == 0
        ):
            return "The surface-to-RAS transform is invalid."
        lines = self.mw.tractogram_data
        if lines is None or not len(lines):
            return "Load a nonempty tractogram first."
        return ""

    def panel_available(self, mesh_id: str | None) -> bool:
        return (
            mesh_id in self.mw.mesh_layers
            and not self._closed
            and not getattr(self.mw, "_shutdown_requested", False)
        )

    @property
    def result_actions_enabled(self) -> bool:
        return (
            self.current and not self.busy and not self._exporting
            and not self._incompatible()
        )

    def _geometry_matches(self, key):
        from ..surface_mapping import surface_geometry_key

        layer = self.mw.mesh_layers.get(key.mesh_id)
        if layer is None:
            return False
        try:
            return key == surface_geometry_key(layer)
        except (ValueError, RuntimeError, AttributeError):
            return False

    def _owns(self, request):
        return bool(
            not getattr(self.mw, "_shutdown_requested", False)
            and request.request_id == self._token
            and request.tractogram_id == str(id(self.mw.tractogram_data))
            and request.tractogram_version
            == getattr(self.mw, "_tractogram_data_version", 0)
            and self._scope_visibility_version == self.mw._visibility_version
            and (
                request.parameters.scope != "selected_visible"
                or request.selection_version == self.selection_version
            )
            and self._geometry_matches(request.geometry_key)
        )

    def start(self, mesh_id: str, parameters: SurfaceMappingParameters) -> bool:
        from ..surface_mapping import SurfaceMappingParameters

        reason = self.unavailable_reason(mesh_id)
        if reason:
            self.message = reason
            self.changed.emit()
            return False
        if not isinstance(parameters, SurfaceMappingParameters):
            raise TypeError("Expected validated surface mapping parameters")
        saved_ids = (
            self.recipe_ids
            if self.mesh_id == mesh_id and self.parameters is not None
            and self.parameters.scope == parameters.scope
            and (
                "endpoint_proximity" not in parameters.modes
                or self.parameters.max_distance_mm == parameters.max_distance_mm
            ) else None
        )
        if saved_ids is None and (
            not self.mw.visible_indices or (
                parameters.scope == "selected_visible"
                and self.mw.selected_streamline_indices.isdisjoint(self.mw.visible_indices)
            )
        ):
            self.message = "The selected scope is empty."
            self.changed.emit()
            return False
        display = None
        if self.mesh_id == mesh_id and self.parameters is not None:
            display = self.recipe_display.copy()
            if self.presentation is not None:
                display.update(
                    (name, bool(getattr(self.presentation, name))) for name in display
                )
            for mode, name in (
                ("endpoint_proximity", "endpoint_enabled"),
                ("surface_intersections", "intersections_enabled"),
            ):
                if mode not in self.parameters.modes:
                    display.pop(name, None)
                if mode not in parameters.modes:
                    display[name] = False
        else:
            self.recipe_display = _DEFAULT_DISPLAY.copy()
        self._clear_result()
        self.mesh_id, self.parameters = mesh_id, parameters
        self._token = str(uuid4())
        self._scope_visibility_version = self.mw._visibility_version
        self.stale = False
        self.state, self.message = "preparing", "Preparing surface mapping snapshot..."
        worker = SurfaceMappingWorker(
            self.mw,
            self.mw.mesh_layers[mesh_id],
            parameters,
            self._token,
            self.selection_version,
            saved_ids,
        )
        worker.display_preferences = display
        self._worker = worker
        self.mw._background_workers.append(worker)
        worker.result_ready.connect(self._on_result, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        worker.progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._on_finished, Qt.ConnectionType.QueuedConnection)
        self._progress_step = 0
        self._progress_maximum = len(_PROGRESS_PHASES) * len(parameters.modes) + 1
        self.mw.vtk_panel.update_progress_bar(0, self._progress_maximum)
        self._timer.start()
        self.changed.emit()
        worker.start()
        return True

    @pyqtSlot(str)
    def _on_progress(self, phase):
        if self.sender() is not self._worker or self._worker.is_cancelled:
            return
        if phase in _PROGRESS_PHASES:
            self._progress_step += 1
            self.mw.vtk_panel.update_progress_bar(
                self._progress_step, self._progress_maximum
            )
        self.state = "preparing" if phase == "staging" else "running"
        self.message = f"Surface mapping: {phase}..."
        self.changed.emit()

    @pyqtSlot()
    def _on_result(self):
        worker = self.sender()
        bundle = worker.take_result()
        if bundle is None:
            return
        if (
            worker is not self._worker
            or worker.is_cancelled
            or not self._owns(bundle.result.request)
            or (
                bundle.companion is not None
                and not self._owns(bundle.companion.result.request)
            )
        ):
            bundle.close()
            return
        from ..visualization.surface_mapping import SurfaceMappingPresentation

        candidate = None
        try:
            candidate = SurfaceMappingPresentation(self.mw.vtk_panel, bundle)
            if worker.display_preferences is not None:
                for name, value in worker.display_preferences.items():
                    if name == "density_enabled":
                        candidate.set_density(value)
                    else:
                        setattr(candidate, name, value)
            candidate.publish()
            self.bundle, self.presentation = bundle, candidate
            self.recipe_display.update(
                (name, bool(getattr(candidate, name))) for name in self.recipe_display
            )
            self.recipe_ids = None
            self.state = "ready"
            self.message = self._summary()
            self.mw.vtk_panel.update_progress_bar(
                self._progress_maximum, self._progress_maximum
            )
            self._refresh_tree()
            self.mw.vtk_panel.render_window.Render()
            self.changed.emit()
        except Exception as error:
            if candidate is not None:
                candidate.close()
            self.bundle = self.presentation = None
            bundle.close()
            self.state, self.message = "failed", f"Surface mapping failed: {error}"
            logger.exception("Could not publish surface mapping")
            self.changed.emit()

    @pyqtSlot(str)
    def _on_error(self, message):
        if self.sender() is self._worker and not self._worker.is_cancelled:
            self.state, self.message = "failed", f"Surface mapping failed: {message}"
            self.changed.emit()

    @pyqtSlot()
    def _on_finished(self):
        from ..file_io import _release_deferred_trx_owners

        worker = self.sender()
        worker.discard_pending_result()
        if worker in self.mw._background_workers:
            self.mw._background_workers.remove(worker)
        if worker is self._worker:
            self._hide_progress_bar()
            self._worker = None
            if worker.layer["id"] not in self.mw.mesh_layers:
                self.state, self.message = "idle", ""
            elif worker.is_cancelled:
                self.state = "cancelled"
            elif self.state not in ("ready", "failed"):
                self.state, self.message = (
                    "failed",
                    "Surface mapping returned no current result.",
                )
        _release_deferred_trx_owners(self.mw)
        worker.deleteLater()
        if self.bundle is None:
            self._timer.stop()
        self.changed.emit()

    def _hide_progress_bar(self) -> None:
        if self._progress_step is not None:
            self._progress_step = None
            self.mw.vtk_panel.update_progress_bar(0, 0, visible=False)

    def _summary(self):
        d = self.bundle.result.diagnostics
        bundles = [self.bundle]
        if self.bundle.companion is not None:
            bundles.append(self.bundle.companion)
        if self.bundle.result.request.parameters.mode == "endpoint_proximity":
            text = (
                f"{d.assigned_endpoints:,}/{d.evaluated_endpoints:,} endpoints assigned; "
                f"{d.beyond_distance_endpoints:,} beyond distance; "
                f"{d.ambiguous_endpoints:,} ambiguous; {d.invalid_endpoints:,} invalid."
            )
        else:
            text = (
                f"{d.crossing_events:,} crossing events / {d.crossing_streamlines:,} unique streamlines; "
                f"{d.touch_events:,} touches; {d.coplanar_intervals:,} coplanar intervals; "
                f"{d.ambiguous_events:,} ambiguous; {d.invalid_events:,} invalid."
            )
        if self.bundle.companion is not None:
            intersections = self.bundle.companion.result.diagnostics
            text += (
                f"\n{intersections.crossing_events:,} crossing events / "
                f"{intersections.crossing_streamlines:,} unique streamlines; "
                f"{intersections.touch_events:,} touches; "
                f"{intersections.coplanar_intervals:,} coplanar intervals; "
                f"{intersections.ambiguous_events:,} ambiguous; "
                f"{intersections.invalid_events:,} invalid."
            )
        return (
            f"{d.requested_streamlines:,} requested streamlines. {text}\n"
            f"Markers: {sum(len(bundle.marker_points) for bundle in bundles):,}/"
            f"{sum(bundle.marker_count for bundle in bundles):,}; "
            f"interval segments: "
            f"{sum(len(bundle.overlap_points) for bundle in bundles):,}/"
            f"{sum(bundle.overlap_segment_count for bundle in bundles):,} "
            f"displayed ({sum(bundle.overlap_count for bundle in bundles):,} "
            "QC intervals). "
            "Orange: assigned endpoints. Cyan: proven crossings. "
            "Magenta: QC (excluded from quantitative map). "
            "Endpoint radius / crossing half-width: 0.12 mm; "
            "overlap radius: 0.06 mm (presentation only).\n"
            + self._geometry_summary()
        )

    def _geometry_summary(self):
        geometry = self.bundle.geometry.diagnostics
        return (
            f"Surface QC: {geometry.valid_triangles:,}/{geometry.derived_triangles:,} valid triangles; "
            f"{geometry.degenerate_triangles:,} degenerate; {geometry.boundary_edges:,} boundary edges; "
            f"{geometry.nonmanifold_edges:,} nonmanifold edges; {geometry.duplicate_faces:,} duplicate faces; "
            f"{geometry.nonplanar_polygons:,} nonplanar polygons."
        )

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._token = None
            self._hide_progress_bar()
            self.state, self.message = "cancelled", "Surface mapping cancelled."
            self.changed.emit()

    def invalidate(self):
        if self._worker is None and self.bundle is None and self.recipe_ids is None:
            self.refresh_panel()
            return
        self.recipe_ids = None
        self.cancel()
        self._clear_result()
        self.stale = True
        self.message = "Recompute required: the data, surface or scope changed."
        self._refresh_tree()
        self.changed.emit()

    def selection_changed(self):
        self.selection_version += 1
        if self.parameters is not None and self.parameters.scope == "selected_visible":
            self.invalidate()
        else:
            self.changed.emit()

    def visibility_changed(self):
        """Accept a filter revision only when it preserves the frozen scope."""
        request = (
            self.bundle.result.request if self.bundle is not None
            else self._worker.request if self._worker is not None else None
        )
        ids = request.streamline_ids if request is not None else self.recipe_ids
        if ids is None:
            self.invalidate()
            return
        visible = self.mw.visible_indices
        scope = request.parameters.scope if request is not None else self.parameters.scope
        if scope == "visible":
            count = len(visible)
        elif request is None:
            count = len(ids)
        else:
            count = sum(index in visible for index in self.mw.selected_streamline_indices)
        if len(ids) != count or not visible.issuperset(ids):
            self.invalidate()
            return
        self._scope_visibility_version = self.mw._visibility_version
        self.check_current()
        self.refresh_panel()

    def check_current(self):
        if self.bundle is None and self._worker is None:
            return
        # A pending tractogram load can still roll back to this result.
        if getattr(self.mw, "_loader_thread", None) is not None:
            return
        if self.bundle is not None and not self._owns(self.bundle.result.request):
            self.invalidate()
            self.mw.vtk_panel.render_window.Render()
        elif self._worker is not None and not self._worker.is_cancelled:
            worker = self._worker
            layer = self.mw.mesh_layers.get(self.mesh_id)
            if (
                self._incompatible()
                or layer is None
                or worker.tractogram_version
                != getattr(self.mw, "_tractogram_data_version", 0)
                or self._scope_visibility_version != self.mw._visibility_version
                or (
                    worker.streamlines is not None
                    and worker.streamlines is not self.mw.tractogram_data
                )
                or not self._geometry_matches(worker.geometry_key)
            ):
                self.invalidate()

    def _clear_result(self):
        if self.presentation is not None:
            self.presentation.close()
            self.presentation = None
        if self.bundle is not None:
            self.bundle.close()
            self.bundle = None

    def remove(self, mesh_id=None, *, refresh=True):
        if self.mesh_id is None and not self.busy and self.bundle is None:
            return
        if mesh_id is not None and mesh_id != self.mesh_id:
            return
        self.cancel()
        self._clear_result()
        self.stale = False
        self._token = self.mesh_id = None
        self.parameters = None
        self.recipe_ids = None
        if not self.busy:
            self.state = "idle"
            self._timer.stop()
        if refresh:
            self._refresh_tree()
        self.changed.emit()

    def stop_for_restore(self):
        self.cancel()
        if self._worker is not None and self._worker.isRunning():
            loop = QEventLoop()
            worker = self._worker
            worker.finished.connect(loop.quit)
            if worker.isRunning():
                loop.exec()

    def close(self):
        self._closed = True
        if self._panel is not None:
            self._panel.shutdown()
        self._timer.stop()
        self.remove()
        if self._worker is not None:
            self._worker.wait()
            self._worker.discard_pending_result()

    def set_display(self, name, enabled):
        if name not in (
            "visible",
            "markers_enabled",
            "qc_enabled",
            "map_enabled",
            "density_enabled",
        ):
            raise ValueError("Unknown surface mapping display option")
        self.check_current()
        if not self.current:
            return
        if name == "density_enabled":
            self.presentation.set_density(enabled)
        else:
            setattr(self.presentation, name, bool(enabled))
        self.recipe_display[name] = bool(enabled)
        self.presentation.render()
        self._refresh_tree()
        self.changed.emit()

    def set_method_enabled(self, mode: str, enabled: bool) -> bool:
        self.check_current()
        if mode not in ("endpoint_proximity", "surface_intersections"):
            return False
        name = (
            "endpoint_enabled" if mode == "endpoint_proximity"
            else "intersections_enabled"
        )
        if self.recipe_ids is not None and not self.busy:
            self.recipe_display[name] = bool(enabled)
            self.changed.emit()
            return True
        if not self.current or mode not in self.parameters.modes:
            return False
        setattr(self.presentation, name, bool(enabled))
        self.recipe_display[name] = bool(enabled)
        self.presentation.render()
        self.changed.emit()
        return True

    def set_surface_map(self, choice: str) -> None:
        """Apply the three UI choices in one presentation transaction."""
        if choice not in ("off", "contributions", "density"):
            raise ValueError("Unknown surface map choice")
        self.check_current()
        if not self.current:
            return
        presentation = self.presentation
        if choice != "off":
            if presentation.map_actor is None:
                return
            density = choice == "density"
            if presentation.density_enabled != density:
                presentation.set_density(density)
        presentation.map_enabled = choice != "off"
        self.recipe_display.update(
            map_enabled=presentation.map_enabled,
            density_enabled=presentation.density_enabled,
        )
        presentation.render()
        self._refresh_tree()
        self.changed.emit()

    def refresh_panel(self) -> None:
        if self._panel is not None:
            self._panel.refresh()

    def open_panel(self, mesh_id: str) -> SurfaceMappingPanel | None:
        if not self.panel_available(mesh_id):
            return None
        if self._panel is None:
            from ..ui.surface_mapping_panel import SurfaceMappingPanel

            self._panel = SurfaceMappingPanel(self.mw)
        target = mesh_id
        if self.busy and self._worker.layer["id"] in self.mw.mesh_layers:
            target = self._worker.layer["id"]
        manager = self.mw.data_panel_manager
        if manager.inspected_mesh_id != target:
            self.mw.mesh_manager.select(target)
        tree = manager.mesh_inspector
        blocked = tree.blockSignals(True)
        try:
            self._panel.set_target(target)
            self._refresh_tree()
            root = self._panel.root_item
            if root is not None:
                parent = root
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                tree.scrollToItem(root)
        finally:
            tree.blockSignals(blocked)
        self.mw.data_dock_widget.show()
        self._panel.refresh()
        return self._panel

    def add_menu_actions(self, menu, mesh_id, *, mapping_node=False):
        if not mapping_node:
            action = menu.addAction("Tractography mapping…")
            action.setEnabled(self.panel_available(mesh_id))
            action.triggered.connect(lambda: self.open_panel(mesh_id))
            return
        if mesh_id == self.mesh_id:
            self.check_current()
        token = self._token

        def run_action(operation):
            self.check_current()
            if (
                mesh_id == self.mesh_id and self._token == token
                and self.result_actions_enabled
            ):
                operation()

        def can_remove():
            return (
                mesh_id == self.mesh_id and self._token == token
                and (self.bundle is not None or self.stale)
                and not self.busy and not self._exporting and not self._incompatible()
            )

        def remove_mapping():
            if can_remove():
                self.remove(mesh_id)

        export_action = menu.addAction("Export mapping…")
        export_action.setEnabled(
            mesh_id == self.mesh_id and self.result_actions_enabled
        )
        export_action.triggered.connect(lambda: run_action(self.open_export_dialog))
        remove_action = menu.addAction("Remove mapping", remove_mapping)
        remove_action.setEnabled(can_remove())

    def export_to(self, prefix, *, scientific=True, figure_options=None, overwrite=False):
        from ..surface_mapping.export import stage_surface_mapping
        from ..surface_mapping import SurfaceMappingCancelled
        from .session_manager import SessionManager
        from ..session.io import SessionCancelled

        self.check_current()
        if not self.current or self.busy or self._exporting or self._incompatible():
            raise ValueError("A complete, current surface mapping is required for export")
        bundle = self.bundle
        self._exporting = True
        transaction = None
        try:
            with bundle.lease():
                from ..visualization.surface_mapping_export import (
                    capture_figure, presentation_metadata,
                )

                presentation = presentation_metadata(self.presentation)
                figure = (
                    capture_figure(self.mw.vtk_panel, self.presentation, figure_options)
                    if figure_options is not None else None
                )
                sources = self._export_sources()
                streamlines = self.mw.tractogram_data

                def operation(worker):
                    return stage_surface_mapping(
                        prefix, bundle, sources, scientific=scientific, figure=figure,
                        presentation=presentation, overwrite=overwrite,
                        cancel_check=lambda: worker.cancelled,
                        streamlines=streamlines,
                    )

                transaction = SessionManager(self.mw)._run(
                    "Exporting surface mapping...", operation,
                    discard_result=lambda result: (
                        result.close() if result is not None else None
                    ),
                    trx_owner=self.mw.trx_file_reference,
                )
                transaction.cancel_check = None
                return transaction.commit(lambda: self.bundle is bundle and self.current)
        except (SessionCancelled, SurfaceMappingCancelled):
            return None
        finally:
            if transaction is not None:
                transaction.close()
            self._exporting = False

    def _export_sources(self):
        from ..session.state import SOURCE_FIELDS

        sources = {
            role: getattr(self.mw, field)
            for role, (field, _) in SOURCE_FIELDS.items()
            if getattr(self.mw, field, None)
        }
        for identity, layer in self.mw.mesh_layers.items():
            sources["mesh:" + identity] = layer["path"]
            sources.update(layer.get("texture_assets", {}))
        for name, layer in self.mw.roi_layers.items():
            path = layer.get("path") or layer.get("source_path")
            if path:
                sources["roi:" + name] = path
        return sources

    def open_export_dialog(self):
        from ..ui.surface_mapping_export_dialog import SurfaceMappingExportDialog

        if (
            self.current and not self.busy and not self._exporting
            and not self._incompatible()
        ):
            dialog = SurfaceMappingExportDialog(self)
            try:
                dialog.exec()
            finally:
                self.changed.disconnect(dialog.refresh)
                dialog.deleteLater()

    def sync_tree(self, parent, mesh_id):
        item = next(
            (
                parent.child(i)
                for i in range(parent.childCount())
                if (parent.child(i).data(0, Qt.ItemDataRole.UserRole) or {}).get("type")
                == "surface_mapping"
            ),
            None,
        )
        if item is None:
            item = QTreeWidgetItem(parent)
            item.setData(
                0, Qt.ItemDataRole.UserRole, {"type": "surface_mapping", "id": mesh_id}
            )
        item.setText(0, "Tractography mapping")
        item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        current = mesh_id == self.mesh_id and self.current
        flags = item.flags() | Qt.ItemFlag.ItemIsEnabled
        if current:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        else:
            flags &= ~Qt.ItemFlag.ItemIsUserCheckable
        item.setFlags(flags)
        item.setData(
            0, Qt.ItemDataRole.CheckStateRole, (
                Qt.CheckState.Checked
                if current and self.presentation.visible
                else Qt.CheckState.Unchecked
            ) if current else None,
        )
        if (
            parent.treeWidget() is not None
            and self._panel is not None and self._panel.mesh_id == mesh_id
        ):
            self._panel.attach(item)

    def _refresh_tree(self):
        manager = getattr(self.mw, "data_panel_manager", None)
        mesh_id = getattr(manager, "inspected_mesh_id", None)
        if mesh_id is None or mesh_id not in self.mw.mesh_layers:
            return
        tree = manager.mesh_inspector
        blocked = tree.blockSignals(True)
        try:
            self.sync_tree(tree.invisibleRootItem(), mesh_id)
        finally:
            tree.blockSignals(blocked)
