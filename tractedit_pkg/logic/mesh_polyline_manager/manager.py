# -*- coding: utf-8 -*-
"""Surface mouse tracing and mesh-associated or standalone annotation lifecycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import vtk
from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QInputDialog,
    QMenu,
    QTreeWidgetItem,
)

from ...render_batch import request_render
from ...mesh.polylines import (
    DEFAULT_POLYLINE_COLOR,
    validated_color,
    validated_name,
    validated_style,
    validate_polylines,
)
from ...utils import MAX_STACK_LEVELS, ROI_COLORS
from .. import mesh_polyline_manager
from ..state_manager import ActionType
from .io_worker import PolylineIOMixin

if TYPE_CHECKING:
    from ...data_contracts import MeshLayer, MeshPolyline
    from ...main_window import MainWindow
    from ...mesh.polyline_csv import PolylineCSVData

logger = logging.getLogger(__name__)


class MeshPolylineManager(QObject, PolylineIOMixin):
    def __init__(self, main_window: MainWindow) -> None:
        super().__init__(main_window)
        self.mw = main_window
        self._polygon = None
        self.actors: dict[str, vtk.vtkActor] = {}
        self._picker = None
        self._pick_actor = None
        self._preview = None
        self._preview_markers = None
        self._draft_style = validated_style()
        self._floating = None
        self._floating_source = None
        self._position = None
        self._camera_drag = False
        self._draft_redo = []
        self._right_press = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._hover)
        self._style_timer = QTimer(self)
        self._style_timer.setInterval(33)
        self._style_timer.setSingleShot(True)
        self._style_timer.timeout.connect(self.flush_style_render)
        self._style_dirty = False
        self._request_generation = 0

    @pyqtSlot()
    def _on_csv_result(self) -> None:
        return PolylineIOMixin._on_csv_result(self)

    @pyqtSlot(str)
    def _on_csv_error(self, error: str) -> None:
        return PolylineIOMixin._on_csv_error(self, error)

    @pyqtSlot()
    def _on_csv_finished(self) -> None:
        return PolylineIOMixin._on_csv_finished(self)

    def _remember(self, mesh_id, operation, records, index=0, names=None):
        stack = self.mw.unified_undo_stack
        stack.append({
            "action_type": ActionType.POLYLINE_EDIT, "mesh_id": mesh_id,
            "operation": operation, "records": records, "index": index, "names": names,
        })
        del stack[:-MAX_STACK_LEVELS]
        self.mw.unified_redo_stack.clear()
        self.mw._update_action_states()

    def can_replay_history(self) -> bool:
        return (
            not self.active and not self.io_busy and self.mw.mesh_manager._can_mutate()
        )

    def replay_history(self, *, undo: bool) -> None:
        mw = self.mw
        if not self.can_replay_history():
            return
        source, destination = mw.unified_undo_stack, mw.unified_redo_stack
        if not undo:
            source, destination = destination, source
        action = source[-1]
        mesh_id, batch = action["mesh_id"], action["records"]
        records = self.container(mesh_id)
        if action["operation"] == "rename":
            batch[0]["name"] = action["names"][0 if undo else 1]
        elif (action["operation"] == "remove") == undo:
            added = []
            try:
                for record in batch:
                    self._publish_actor(mesh_id, record)
                    added.append(record["id"])
            except Exception:
                for identity in added:
                    self._release_actor(self.actors.pop(identity))
                raise
            if mesh_id is not None:
                mw.mesh_layers[mesh_id].setdefault("polylines", records)
            index = action["index"]
            records[index:index] = batch
        else:
            ids = {record["id"] for record in batch}
            records[:] = [record for record in records if record["id"] not in ids]
            for identity in ids:
                self._release_actor(self.actors.pop(identity))
            if mesh_id is not None and not records:
                mw.mesh_layers[mesh_id].pop("polylines", None)
        destination.append(source.pop())
        mw.mesh_manager._refresh_tree()
        mw.vtk_panel.render_window.Render()

    def _forget_history(self, mesh_id):
        for stack in (self.mw.unified_undo_stack, self.mw.unified_redo_stack):
            stack[:] = [
                action for action in stack
                if action.get("action_type") != ActionType.POLYLINE_EDIT
                or action["mesh_id"] != mesh_id
            ]

    def _fit_bounds(self, bounds) -> None:
        padding = max(float(np.max(bounds[:, 1] - bounds[:, 0])) * 0.05, 1.0)
        bounds += [-padding, padding]
        self.mw.vtk_panel.scene.ResetCamera(bounds.ravel())
        self.mw.vtk_panel.scene.ResetCameraClippingRange()

    def fit(self, mesh_id: str | None, identity: str) -> None:
        record = self._record(mesh_id, identity)
        if record is not None and self.mw.mesh_manager._can_mutate():
            actor = self.actors.get(identity)
            if actor is not None and hasattr(actor, "patch"):
                bounds = np.asarray(actor.patch.GetBounds(), dtype=np.float64).reshape(3, 2)
            else:
                points = record["points"]
                bounds = np.column_stack((points.min(axis=0), points.max(axis=0)))
            self._fit_bounds(bounds)
            self.mw.vtk_panel.render_window.Render()

    @property
    def active(self) -> bool:
        return self.mw.mesh_polyline_mesh_id is not None

    def container(self, mesh_id: str | None) -> list[MeshPolyline]:
        if mesh_id is None:
            return self.mw.standalone_polylines
        return self.mw.mesh_layers[mesh_id].get("polylines", [])

    def iter_records(self):
        yield from ((None, record) for record in self.mw.standalone_polylines)
        for mesh_id, layer in self.mw.mesh_layers.items():
            yield from ((mesh_id, record) for record in layer.get("polylines", []))

    def _available_colors(self, incoming=()):
        def rgb_key(color):
            return tuple(round(255 * component) for component in color)

        used = {rgb_key(record["color"]) for _, record in self.iter_records()}
        used.update(rgb_key(item["color"]) for item in incoming if not item.get("auto_color"))
        for color in (DEFAULT_POLYLINE_COLOR, *ROI_COLORS):
            key = rgb_key(color)
            if key not in used:
                used.add(key)
                yield color
        for index in range(1, 1 << 24):
            value = (index * 0x9E3779) & 0xFFFFFF
            key = (value >> 16, (value >> 8) & 255, value & 255)
            if max(key) >= 160 and max(key) - min(key) >= 80 and key not in used:
                yield tuple(component / 255 for component in key)
        while True:
            yield DEFAULT_POLYLINE_COLOR

    def find(self, identity: str) -> tuple[str | None, MeshPolyline] | None:
        return next(
            ((mesh_id, record) for mesh_id, record in self.iter_records()
             if record["id"] == identity), None,
        )

    def check_capacity(self, mesh_id: str | None, point_count: int) -> None:
        records = self.container(mesh_id)
        if len(records) >= mesh_polyline_manager.MAX_POLYLINES or (
            sum(len(record["points"]) for record in records) + point_count
            > mesh_polyline_manager.MAX_POLYLINE_POINTS
        ):
            raise ValueError("Polyline container capacity exceeded")

    def available(self, mesh_id: str) -> bool:
        layer = self.mw.mesh_layers.get(mesh_id)
        return bool(
            layer is not None
            and layer["visible"]
            and layer["opacity"] > 0
            and layer["presentation"]["surface"]["enabled"]
            and self.mw.mesh_manager._can_mutate()
        )

    def start(self, mesh_id: str) -> bool:
        if not self.available(mesh_id) or self.io_busy:
            return False
        self.cancel(render=False)
        self._draft_style = validated_style()
        mw, panel = self.mw, self.mw.vtk_panel
        layer = mw.mesh_layers[mesh_id]
        try:
            locator = vtk.vtkStaticCellLocator()
            locator.SetDataSet(layer["polydata"])
            locator.BuildLocator()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(layer["polydata"])
            self._pick_actor = vtk.vtkActor()
            self._pick_actor.SetMapper(mapper)
            self._pick_actor.SetUserMatrix(panel.mesh_actors[mesh_id].GetMatrix())
            self._picker = vtk.vtkCellPicker()
            self._picker.SetTolerance(0.0005)
            self._picker.PickFromListOn()
            self._picker.AddPickList(self._pick_actor)
            self._picker.AddLocator(locator)
            for action in (
                mw.draw_mode_action,
                mw.erase_mode_action,
                mw.sphere_mode_action,
                mw.rectangle_mode_action,
            ):
                action.setChecked(False)
            self._preview = vtk.vtkActor()
            preview_mapper = vtk.vtkPolyDataMapper()
            preview_mapper.SetInputData(vtk.vtkPolyData())
            preview_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            preview_mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)
            self._preview.SetMapper(preview_mapper)
            self._preview.PickableOff()
            self._draft_color = next(self._available_colors())
            self._preview.GetProperty().SetColor(*self._draft_color)
            self._preview.GetProperty().SetLineWidth(3)
            self._preview.GetProperty().SetPointSize(7)
            self._preview.GetProperty().LightingOff()
            panel.scene.AddActor(self._preview)
            self._preview_markers = vtk.vtkActor()
            self._preview_markers.ShallowCopy(self._preview)
            marker_mapper = vtk.vtkPolyDataMapper()
            marker_mapper.SetInputData(vtk.vtkPolyData())
            marker_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            marker_mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-2)
            self._preview_markers.SetMapper(marker_mapper)
            panel.scene.AddActor(self._preview_markers)
            self._floating_source = vtk.vtkLineSource()
            floating_mapper = vtk.vtkPolyDataMapper()
            floating_mapper.SetInputConnection(self._floating_source.GetOutputPort())
            floating_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            floating_mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)
            self._floating = vtk.vtkActor()
            self._floating.ShallowCopy(self._preview)
            self._floating.SetMapper(floating_mapper)
            self._floating.VisibilityOff()
            panel.scene.AddActor(self._floating)
            mw.mesh_polyline_mesh_id = mesh_id
            mw.mesh_polyline_points = []
            widget = panel.vtk_widget
            self._previous_tracking = widget.hasMouseTracking()
            self._previous_cursor = widget.cursor()
            widget.setMouseTracking(True)
            widget.setCursor(Qt.CursorShape.CrossCursor)
            widget.setFocus()
            panel.update_status(
                "Polyline: left click adds a vertex; right click or Enter/Space "
                "saves and exits; Esc cancels; Backspace removes a vertex. "
                "Alt+left drag rotates; middle drag pans; wheel zooms."
            )
            mw._update_action_states()
            mw.data_panel_manager.on_data_item_selected()
            return True
        except Exception as error:
            logger.exception("Could not start mesh polyline drawing")
            self.cancel()
            mesh_polyline_manager.QMessageBox.critical(mw, "Polyline Error", str(error))
            return False

    def polygon_editor(self):
        if self._polygon is None:
            from ..surface_polygon_editor import SurfacePolygonEditor

            self._polygon = SurfacePolygonEditor(self)
        return self._polygon

    def _display_position(self, position: QPoint):
        widget = self.mw.vtk_panel.vtk_widget
        width, height = self.mw.vtk_panel.render_window.GetSize()
        x = position.x() * width / max(1, widget.width())
        y = (widget.height() - position.y() - 1) * height / max(1, widget.height())
        return x, y

    def _surface_menu(self, position: QPoint) -> None:
        if self.active or self.io_busy or not self.mw.mesh_manager._can_mutate():
            return
        panel = self.mw.vtk_panel
        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        targets = {}
        for key, actor in panel.mesh_actors.items():
            if self.available(key):
                proxy = vtk.vtkActor()
                presentation = panel._mesh_texture_presentations.get(key)
                proxy.SetMapper(presentation.uniform_mapper if presentation else actor.GetMapper())
                proxy.SetUserMatrix(actor.GetMatrix())
                picker.AddPickList(proxy)
                targets[proxy] = key
        if not targets or not picker.Pick(
            *self._display_position(position), 0, panel.scene
        ):
            return
        key = targets.get(picker.GetActor())
        if key is not None:
            menu = QMenu(self.mw)
            try:
                draw = menu.addAction("Draw Polyline...")
                polygon = menu.addAction("Draw Surface Polygon...")
                selected = menu.exec(panel.vtk_widget.mapToGlobal(position))
                if selected is polygon:
                    self.polygon_editor().start(key)
                elif selected is draw:
                    self.start(key)
            finally:
                menu.deleteLater()

    def pick(self, position: QPoint) -> np.ndarray | None:
        if not self.active or not self.available(self.mw.mesh_polyline_mesh_id):
            return None
        x, y = self._display_position(position)
        if not self._picker.Pick(x, y, 0, self.mw.vtk_panel.scene):
            return None
        if self._picker.GetCellId() < 0:
            return None
        point = np.array(self._picker.GetPickPosition(), dtype=np.float64)
        return point if np.isfinite(point).all() else None

    def _track_context_click(self, event) -> None:
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton:
                self._right_press = event.position().toPoint()
        elif self._right_press is not None and kind in (
            QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease,
        ):
            position = event.position().toPoint()
            distance = (position - self._right_press).manhattanLength()
            if distance >= QApplication.startDragDistance():
                self._right_press = None
            elif kind == QEvent.Type.MouseButtonRelease:
                self._right_press = None
                if event.button() == Qt.MouseButton.RightButton and self.mw.mesh_layers:
                    QTimer.singleShot(0, lambda: self._surface_menu(position))

    def eventFilter(self, watched, event) -> bool:
        if self._polygon and self._polygon.busy:
            kind = event.type()
            if kind in (
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonRelease,
                QEvent.Type.MouseButtonDblClick,
                QEvent.Type.MouseMove,
                QEvent.Type.KeyPress,
                QEvent.Type.KeyRelease,
                QEvent.Type.ShortcutOverride,
            ):
                if kind == QEvent.Type.ShortcutOverride:
                    event.accept()
                elif (
                    kind == QEvent.Type.KeyPress
                    and event.key() == Qt.Key.Key_Escape
                    and self._polygon._active_job is not None
                ):
                    self._polygon._active_job.cancel()
                return True
            return False
        if not self.active:
            self._track_context_click(event)
            return False
        event_type = event.type()
        try:
            if event_type == QEvent.Type.ShortcutOverride:
                event.accept()
                return True
            if event_type == QEvent.Type.KeyPress:
                if event.key() in (
                    Qt.Key.Key_Return,
                    Qt.Key.Key_Enter,
                    Qt.Key.Key_Space,
                ):
                    self.finish()
                elif event.key() == Qt.Key.Key_Escape:
                    self.cancel()
                elif event.matches(QKeySequence.StandardKey.Redo) or (
                    event.key() == Qt.Key.Key_Y
                    and event.modifiers() == Qt.KeyboardModifier.ControlModifier
                ) or (
                    event.key() == Qt.Key.Key_Z and event.modifiers() ==
                    (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
                ):
                    self.step_draft(undo=False)
                elif (event.matches(QKeySequence.StandardKey.Undo)
                      or event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete)):
                    self.step_draft(undo=True)
                return True
            if event_type == QEvent.Type.KeyRelease:
                return True
            if event_type == QEvent.Type.MouseButtonPress:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and event.modifiers() & Qt.KeyboardModifier.AltModifier
                ):
                    self._camera_drag = True
                    self._floating.VisibilityOff()
                    self._timer.stop()
                    return False
                if event.button() == Qt.MouseButton.LeftButton:
                    point = self.pick(event.position().toPoint())
                    if point is not None:
                        if self._polygon and self._polygon.layer is not None:
                            self._polygon.click(point, self._picker.GetCellId())
                        else:
                            self.add_point(point)
                    return True
                if event.button() == Qt.MouseButton.RightButton:
                    self.finish()
                    return True
            if (
                event_type == QEvent.Type.MouseButtonRelease
                and self._camera_drag
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._camera_drag = False
                return False
            if event_type in (
                QEvent.Type.MouseButtonRelease,
                QEvent.Type.MouseButtonDblClick,
            ) and event.button() in (
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.RightButton,
            ):
                return True
            if event_type == QEvent.Type.MouseMove and not event.buttons():
                self._position = event.position().toPoint()
                if self._polygon and self._polygon.layer is not None:
                    return True
                if self.mw.mesh_polyline_points and not self._timer.isActive():
                    self._timer.start()
                return True
            if event_type == QEvent.Type.Leave:
                self._timer.stop()
                self._update_floating(None)
        except Exception as error:
            logger.exception("Mesh polyline interaction failed")
            self.cancel()
            self.mw.vtk_panel.update_status(f"Polyline cancelled: {error}")
            return True
        return False

    def step_draft(self, *, undo: bool) -> None:
        points = self.mw.mesh_polyline_points
        if self._polygon and self._polygon.closed:
            if not undo:
                return
            self._polygon.closure_redo.append(self._polygon.region)
            if self._polygon.fill is not None:
                self.mw.vtk_panel.scene.RemoveActor(self._polygon.fill)
                self.mw.vtk_panel._release_mesh_component(self._polygon.fill)
                self._polygon.fill = None
            self._polygon.region = None
            self._polygon.closed = False
            self._update_preview()
            return
        if (
            not undo and not self._draft_redo
            and self._polygon and self._polygon.closure_redo
        ):
            self._polygon.restore_closure()
            self._update_preview()
            return
        source, dest = (points, self._draft_redo) if undo else (self._draft_redo, points)
        if self.active and source:
            dest.append(source.pop())
            self._update_preview()

    @property
    def can_redo_point(self) -> bool:
        return bool(self._draft_redo or (self._polygon and self._polygon.closure_redo))

    def add_point(self, point) -> None:
        if not self.active:
            return
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            return
        points = self.mw.mesh_polyline_points
        if points and np.array_equal(points[-1], point):
            return
        layer = self.mw.mesh_layers[self.mw.mesh_polyline_mesh_id]
        total = sum(len(record["points"]) for record in layer.get("polylines", []))
        if total + len(points) >= mesh_polyline_manager.MAX_POLYLINE_POINTS:
            self.mw.vtk_panel.update_status("Polyline vertex limit reached.")
            return
        points.append(point.copy())
        self._draft_redo.clear()
        self._timer.stop()
        self._update_preview()

    def _hover(self) -> None:
        if self.active and self._position is not None:
            try:
                self._update_floating(self.pick(self._position))
            except Exception as error:
                logger.exception("Mesh polyline preview failed")
                self.cancel()
                self.mw.vtk_panel.update_status(f"Polyline cancelled: {error}")

    def _update_floating(self, point) -> None:
        if self._floating is None:
            return
        points = self.mw.mesh_polyline_points
        visible = point is not None and bool(points)
        if not visible and not self._floating.GetVisibility():
            return
        if visible:
            self._floating_source.SetPoint1(*points[-1])
            self._floating_source.SetPoint2(*point)
        self._floating.SetVisibility(visible)
        self.mw.vtk_panel.render_window.Render()

    def _update_preview(self) -> None:
        if self._preview is None:
            return
        points = self.mw.mesh_polyline_points
        self.mw.undo_action.setEnabled(bool(points))
        self.mw.redo_action.setEnabled(self.can_redo_point)
        self._floating.VisibilityOff()
        if self._polygon and self._polygon.layer is not None:
            from ...mesh.polygon_io import ras_points

            editor = self._polygon
            ids = np.asarray(editor.route(points, editor.closed), dtype=np.int64)
            points = ras_points(editor.cache[editor.layer["id"]], ids, editor.layer)
        line_data, marker_data = mesh_polyline_manager.polyline_display_data(points)
        self._preview.GetMapper().SetInputData(line_data)
        self._preview_markers.GetMapper().SetInputData(marker_data)
        self._preview_markers.SetVisibility(self._draft_style["show_points"])
        request_render(self.mw.vtk_panel, self.mw.vtk_panel.render_window)

    def confirm(self, *, continue_drawing: bool = True) -> bool:
        if not self.active or not self.mw.mesh_manager._can_mutate():
            return False
        points = self.mw.mesh_polyline_points
        if len(points) < 2:
            self.mw.vtk_panel.update_status("Pick at least two mesh points to confirm.")
            return False
        mesh_id = self.mw.mesh_polyline_mesh_id
        layer = self.mw.mesh_layers[mesh_id]
        records = layer.get("polylines", [])
        if len(records) >= mesh_polyline_manager.MAX_POLYLINES:
            raise ValueError("Polyline count limit reached")
        names = {record["name"] for record in records}
        number = 1
        while f"Polyline {number}" in names:
            number += 1
        record = {
            "id": str(uuid4()),
            "name": f"Polyline {number}",
            "points": np.array(points, dtype=np.float64),
            "color": self._draft_color,
            "visible": True,
            **self._draft_style,
        }
        self.check_capacity(mesh_id, len(record["points"]))
        if self.find(record["id"]) is not None:
            raise ValueError("Duplicate polyline identity")
        validate_polylines([record])
        self._publish_actor(mesh_id, record)
        layer.setdefault("polylines", []).append(record)
        try:
            self.mw.mesh_manager._refresh_tree()
        except Exception:
            layer["polylines"][:] = [item for item in records if item is not record]
            if not layer["polylines"]:
                del layer["polylines"]
            self._release_actor(self.actors.pop(record["id"]))
            raise
        self.mw.mesh_polyline_points = []
        self._draft_redo.clear()
        self._remember(mesh_id, "add", [record], len(layer["polylines"]) - 1)
        self._timer.stop()
        if continue_drawing:
            self._update_preview()
        return True

    def finish(self) -> None:
        if self._polygon and self._polygon.layer is not None:
            self._polygon.finish()
            return
        if self.active:
            if len(self.mw.mesh_polyline_points) >= 2:
                self.confirm(continue_drawing=False)
            self.cancel()

    def cancel(self, *, render: bool = True, update_actions: bool = True) -> None:
        self._timer.stop()
        if self._preview is not None:
            self._style_timer.stop()
            self._style_dirty = False
        panel = self.mw.vtk_panel
        was_active = self.active
        was_polygon = bool(self._polygon and self._polygon.layer is not None)
        if self._polygon:
            self._polygon.reset()
        if was_active:
            widget = panel.vtk_widget
            widget.setMouseTracking(self._previous_tracking)
            widget.setCursor(self._previous_cursor)
            panel.update_status(
                "Surface polygon drawing finished."
                if was_polygon
                else "Polyline drawing finished.",
                render=False,
            )
        self.mw.mesh_polyline_mesh_id = None
        self._draft_redo.clear()
        self._right_press = None
        self.mw.mesh_polyline_points = []
        self._position = None
        if self._camera_drag:
            panel.interactor.GetInteractorStyle().OnLeftButtonUp()
            self._camera_drag = False
        self._picker = None
        self._pick_actor = None
        if self._preview is not None:
            panel.scene.RemoveActor(self._preview)
            panel._release_mesh_component(self._preview)
            self._preview = None
            if self._preview_markers is not None:
                panel.scene.RemoveActor(self._preview_markers)
                panel._release_mesh_component(self._preview_markers)
                self._preview_markers = None
            if self._floating is not None:
                panel.scene.RemoveActor(self._floating)
                panel._release_mesh_component(self._floating)
                self._floating = None
            self._floating_source = None
            if render:
                request_render(panel, panel.render_window)

        if was_active:
            self.mw.data_panel_manager.on_data_item_selected()
        if update_actions:
            self.mw._update_action_states()

    def sync_visibility(self, mesh_id: str | None, *, render: bool = True) -> None:
        parent_visible = mesh_id is None or self.mw.mesh_layers[mesh_id]["visible"]
        cancelled = (
            mesh_id is not None and self.mw.mesh_polyline_mesh_id == mesh_id
            and not self.available(mesh_id)
        )
        if cancelled:
            self.cancel(render=False)
        records = self.container(mesh_id)
        for record in records:
            actor = self.actors[record["id"]]
            visible = parent_visible and record["visible"]
            actor.SetVisibility(visible)
            if hasattr(actor, "patch"):
                actor.patch.SetVisibility(visible)
            actor.polyline_markers.SetVisibility(visible and record["show_points"])
        if cancelled or (render and records):
            request_render(self.mw.vtk_panel, self.mw.vtk_panel.render_window)

    def _release_actor(self, actor) -> None:
        parts = [actor, actor.polyline_markers]
        if hasattr(actor, 'patch'):
            parts.append(actor.patch)
        for part in parts:
            self.mw.vtk_panel.scene.RemoveActor(part)
            self.mw.vtk_panel._release_mesh_component(part)

    def remove_mesh(self, mesh_id: str, *, update_actions: bool = True) -> None:
        self._forget_history(mesh_id)
        if self._polygon:
            self._polygon.cache.pop(mesh_id, None)
        self.cancel_io(mesh_id)
        if self.mw.mesh_polyline_mesh_id == mesh_id:
            self.cancel(render=False, update_actions=update_actions)
        for record in self.mw.mesh_layers[mesh_id].get("polylines", []):
            actor = self.actors.pop(record["id"], None)
            if actor is not None:
                self._release_actor(actor)
        if not self.actors and not self.active:
            self._style_timer.stop()
            self._style_dirty = False

    def clear_standalone(self) -> None:
        self._forget_history(None)
        self.cancel_io()
        for record in self.mw.standalone_polylines:
            actor = self.actors.pop(record["id"], None)
            if actor is not None:
                self._release_actor(actor)
        self.mw.standalone_polylines.clear()
        if not self.actors:
            self._style_timer.stop()
            self._style_dirty = False
        request_render(self.mw.vtk_panel, self.mw.vtk_panel.render_window)

    def close(self) -> None:
        self.cancel_io(all_targets=True)
        self.cancel(render=False)
        self._style_timer.stop()
        self._style_dirty = False
        for actor in self.actors.values():
            self._release_actor(actor)
        self.actors.clear()
        if self._polygon is not None:
            self._polygon.cache.clear()
        self.mw.standalone_polylines.clear()

    def _record(self, mesh_id: str | None, identity: str) -> MeshPolyline | None:
        if mesh_id is not None and mesh_id not in self.mw.mesh_layers:
            return None
        return next(
            (
                record
                for record in self.container(mesh_id)
                if record["id"] == identity
            ),
            None,
        )

    def set_color(self, mesh_id: str | None, identity: str, color) -> None:
        record = self._record(mesh_id, identity)
        if record is not None and self.mw.mesh_manager._can_mutate():
            color = validated_color(color)
            self.actors[identity].GetProperty().SetColor(*color)
            self.actors[identity].polyline_markers.GetProperty().SetColor(*color)
            if hasattr(self.actors[identity], 'patch'):
                self.actors[identity].patch.GetProperty().SetColor(*color)
            record["color"] = color
            self.mw._update_data_panel_display()
            self.mw.vtk_panel.render_window.Render()

    def set_style(
        self, mesh_id: str | None, identity: str | None, *,
        line_width=None, point_size=None, show_points=None, final=False,
    ) -> None:
        draft = identity is None and self.active
        record = self._draft_style if draft else self._record(mesh_id, identity)
        if record is None or not self.mw.mesh_manager._can_mutate():
            return
        style = validated_style(
            record["line_width"] if line_width is None else line_width,
            record["point_size"] if point_size is None else point_size,
            record["show_points"] if show_points is None else show_points,
        )
        if any(record[key] != value for key, value in style.items()):
            record.update(style)
            actor = self._preview if draft else self.actors[identity]
            markers = self._preview_markers if draft else actor.polyline_markers
            actor.GetProperty().SetLineWidth(style["line_width"])
            actor.GetProperty().SetPointSize(style["point_size"])
            markers.GetProperty().SetPointSize(style["point_size"])
            markers.SetVisibility(actor.GetVisibility() and style["show_points"])
            if draft:
                self._floating.GetProperty().SetLineWidth(style["line_width"])
            self._style_dirty = True
            if not self._style_timer.isActive():
                self._style_timer.start()
        if final:
            self.flush_style_render()

    def flush_style_render(self) -> None:
        self._style_timer.stop()
        if self._style_dirty:
            self._style_dirty = False
            self.mw.vtk_panel.render_window.Render()

    def change_color(self, mesh_id: str | None, identity: str) -> None:
        record = self._record(mesh_id, identity)
        if record is None or not self.mw.mesh_manager._can_mutate():
            return
        title = "Surface Polygon Color" if "region" in record else "Polyline Color"
        color = QColorDialog.getColor(
            QColor.fromRgbF(*record["color"]),
            self.mw,
            title,
        )
        if color.isValid() and self._record(mesh_id, identity) is record:
            self.set_color(
                mesh_id, identity, (color.redF(), color.greenF(), color.blueF())
            )

    def set_visible(self, mesh_id: str | None, identity: str, visible: bool) -> None:
        record = self._record(mesh_id, identity)
        if record is not None and self.mw.mesh_manager._can_mutate():
            record["visible"] = bool(visible)
            self.sync_visibility(mesh_id)

    def remove(self, mesh_id: str | None, identity: str) -> None:
        record = self._record(mesh_id, identity)
        if record is not None and self.mw.mesh_manager._can_mutate():
            worker = self.mw._polyline_io_thread
            if worker is not None and getattr(worker, "record_id", None) == identity:
                self.cancel_io(mesh_id)
            records = self.container(mesh_id)
            index = next(i for i, item in enumerate(records) if item is record)
            records[:] = [item for item in records if item is not record]
            if mesh_id is not None and not self.container(mesh_id):
                del self.mw.mesh_layers[mesh_id]["polylines"]
            actor = self.actors.pop(identity)
            self._release_actor(actor)
            if not self.actors and not self.active:
                self._style_timer.stop()
                self._style_dirty = False
            self.mw.mesh_manager._refresh_tree()
            self.mw.vtk_panel.render_window.Render()

            self._remember(mesh_id, "remove", [record], index)

    def rename(self, mesh_id: str | None, identity: str) -> None:
        record = self._record(mesh_id, identity)
        if record is None or not self.mw.mesh_manager._can_mutate():
            return
        is_polygon = "region" in record
        title = "Rename Surface Polygon" if is_polygon else "Rename Polyline"
        name, ok = QInputDialog.getText(
            self.mw, title, "Name:", text=record["name"],
        )
        name = name.strip()
        if (not ok or self._record(mesh_id, identity) is not record
                or not self.mw.mesh_manager._can_mutate() or name == record["name"]):
            return
        try:
            validated_name(name)
            if any(
                item["name"] == name and ("region" in item) == is_polygon
                for item in self.container(mesh_id)
            ):
                noun = "surface polygon" if is_polygon else "polyline"
                raise ValueError(f"A {noun} with this name already exists.")
        except ValueError as error:
            mesh_polyline_manager.QMessageBox.warning(self.mw, title, str(error))
            return
        previous = record["name"]
        record["name"] = name
        self._remember(mesh_id, "rename", [record], names=(previous, name))
        self.mw.mesh_manager._refresh_tree()

    def build_tree(self, parent, mesh_id: str | None) -> None:
        records = self.container(mesh_id)
        if not records:
            return
        if mesh_id is None:
            _build_annotation_group(
                parent, None, records,
                label="Standalone polylines",
                group_type="standalone_polylines",
                item_type="standalone_polyline",
                component="polylines",
            )
            return
        polylines = [record for record in records if "region" not in record]
        polygons = [record for record in records if "region" in record]
        if polylines:
            _build_annotation_group(
                parent, mesh_id, polylines,
                label="Polylines",
                group_type="mesh_polylines",
                item_type="mesh_polyline",
                component="polylines",
                tooltip="Right-click to draw or import a polyline for this mesh.",
            )
        if polygons:
            _build_annotation_group(
                parent, mesh_id, polygons,
                label="Surface polygons",
                group_type="mesh_polygons",
                item_type="mesh_polygon",
                component="polygons",
                tooltip="Right-click to draw or import a surface polygon for this mesh.",
            )

    def add_menu_actions(self, menu, data) -> bool:
        standalone = data["type"] in ("standalone_polyline", "standalone_polylines")
        mesh_id = None if standalone else data["id"]
        if data["type"] in ("mesh_polyline", "mesh_polygon", "standalone_polyline"):
            identity = data["component"]
            record = self._record(mesh_id, identity)
            noun = "Surface Polygon" if record and "region" in record else "Polyline"
            for label, callback in (
                (f"Rename {noun}...", self.rename),
                ("Change Color...", self.change_color),
                (f"Export {noun}...", self.export),
                (f"Fit {noun.lower()} in view", self.fit),
                (f"Remove {noun}", self.remove),
            ):
                action = menu.addAction(label)
                if callback != self.fit:
                    action.setEnabled(not self.active and not self.io_busy)
                action.triggered.connect(
                    lambda checked=False, callback=callback: callback(mesh_id, identity)
                )
            return True
        if data["type"] == "mesh_polygons":
            can_draw = self.available(mesh_id) and not self.active and not self.io_busy
            for label, method, enabled in (
                ("Draw Surface Polygon...", "start", can_draw),
                ("Import Surface Polygon...", "import_pair", self.can_import(mesh_id)),
            ):
                option = menu.addAction(label)
                option.setEnabled(enabled)
                option.triggered.connect(
                    lambda checked=False, method=method: getattr(self.polygon_editor(), method)(mesh_id)
                )
            return True
        if data["type"] not in ("mesh", "mesh_polylines", "standalone_polylines"):
            return False
        if mesh_id is not None:
            action = menu.addAction("Draw Polyline...")
            action.setEnabled(
                self.available(mesh_id) and not self.active and not self.io_busy
            )
            action.triggered.connect(lambda: self.start(mesh_id))
            for label, method in (("Draw Surface Polygon...", "start"),
                                  ("Import Surface Polygon...", "import_pair")):
                option = menu.addAction(label)
                option.setEnabled(self.can_import(mesh_id) if method == "import_pair" else action.isEnabled())
                option.triggered.connect(
                    lambda checked=False, method=method: getattr(self.polygon_editor(), method)(mesh_id)
                )
        action = menu.addAction("Import Polyline...")
        action.setEnabled(self.can_import(mesh_id))
        action.triggered.connect(lambda: self.trigger_import(mesh_id))
        return data["type"] in ("mesh_polylines", "standalone_polylines")


def _build_annotation_group(
    parent,
    mesh_id: str | None,
    records,
    *,
    label: str,
    group_type: str,
    item_type: str,
    component: str,
    tooltip: str | None = None,
) -> None:
    group = QTreeWidgetItem(parent, [f"{label} ({len(records)})"])
    group.setData(
        0,
        Qt.ItemDataRole.UserRole,
        {"type": group_type, "id": mesh_id, "component": component},
    )
    if tooltip is not None:
        group.setToolTip(0, tooltip)
    is_polygon = group_type == "mesh_polygons"
    for record in records:
        item = QTreeWidgetItem(group, [record["name"]])
        swatch = QPixmap(10, 10) if is_polygon else QPixmap(12, 4)
        swatch.fill(QColor.fromRgbF(*record["color"]))
        item.setIcon(0, QIcon(swatch))
        item.setData(
            0,
            Qt.ItemDataRole.UserRole,
            {
                "type": item_type,
                "id": mesh_id if mesh_id is not None else record["id"],
                "component": record["id"],
            },
        )
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            0,
            Qt.CheckState.Checked if record["visible"] else Qt.CheckState.Unchecked,
        )
        item_tooltip = f"{len(record['points'])} vertices; RAS (mm)."
        if "region" in record:
            item_tooltip = f"Surface polygon: {len(record['region']['faces']):,} original triangles."
        if mesh_id is not None:
            item_tooltip += " Right-click to rename, change color or export."
        item.setToolTip(0, item_tooltip)
