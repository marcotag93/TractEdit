"""Mesh polygon drawing and paired annotation import/export."""

from pathlib import Path

import numpy as np
import vtk
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QProgressDialog

from ..workers import CancellableWorkerThread, create_progress_dialog, run_modal_worker
from ..file_dialogs import choose_input
from ..mesh.polylines import protected_export_path, validated_style
from ..render_batch import defer_render
from ..mesh.polygon import load_surface
from ..mesh.polygon_io import (
    patch_geometry,
    output_paths,
    ras_points,
    read_obj_pair,
    read_pair,
    resolve_import_pair,
    write_obj_pair,
    write_pair,
)

PROGRESS_DELAY_MS = 500


class _Job(CancellableWorkerThread):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.stop = self._cancelled
        self.error = None
        self.value = None

    def check(self):
        if self.stop.is_set():
            raise InterruptedError("Polygon operation cancelled")

    def run(self):
        try:
            self.value = self.operation(self.check)
        except Exception as error:
            self.error = error


class SurfacePolygonEditor:
    def __init__(self, manager):
        self.manager = manager
        self.mw = manager.mw
        self.cache = {}
        self.layer = None
        self.busy = False
        self._active_job = None
        self.generation = 0
        self.closed = False
        self.fill = None
        self.region = None
        self.paths = {}
        self.closure_redo = []

    def work(self, title, operation, *, commits_files=False, show_delay_ms=0):
        if self.busy:
            raise ValueError("Another polygon operation is active")
        generation = self.generation
        job = _Job(operation)
        dialog = None
        timer = None
        self._active_job = job
        self.busy = True
        try:
            if show_delay_ms > 0:
                loop = QEventLoop()
                job.finished.connect(loop.quit)

                def show_dialog():
                    nonlocal dialog
                    if job.isFinished() or dialog is not None:
                        return
                    dialog = create_progress_dialog(
                        self.mw, title, factory=QProgressDialog
                    )
                    dialog.canceled.connect(job.cancel)
                    dialog.show()

                timer = QTimer(self.mw)
                timer.setSingleShot(True)
                timer.timeout.connect(show_dialog)
                job.finished.connect(timer.stop)
                timer.start(show_delay_ms)
                job.start()
                if not job.isFinished():
                    loop.exec()
                timer.stop()
                job.wait()
            else:
                dialog = create_progress_dialog(self.mw, title, factory=QProgressDialog)
                dialog.canceled.connect(job.cancel)
                run_modal_worker(job, dialog)
            if not commits_files:
                job.check()
            if not commits_files and generation != self.generation:
                raise InterruptedError("Surface polygon operation superseded")
            if job.error:
                raise job.error
            return job.value
        finally:
            if timer is not None:
                timer.stop()
                timer.deleteLater()
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()
            self._active_job = None
            self.busy = False

    def surface(self, layer):
        key = layer["id"]
        if key not in self.cache:
            surface = self.work(
                "Preparing original mesh topology...",
                lambda check: load_surface(layer, check),
                show_delay_ms=PROGRESS_DELAY_MS,
            )
            if self.mw.mesh_layers.get(key) is not layer:
                raise ValueError("Mesh changed during preparation")
            self.cache[key] = surface
        return self.cache[key]

    def start(self, key):
        layer = self.mw.mesh_layers[key]
        try:
            self.surface(layer)
            with defer_render(self.mw.vtk_panel):
                if self.manager.start(key):
                    self.layer = layer
                    self.mw.data_panel_manager.update_polyline_properties()
                    self.mw.vtk_panel.update_status(
                        "Click boundary vertices. Alt+drag rotates. Enter closes the boundary."
                    )
        except InterruptedError:
            return
        except (ValueError, OSError) as error:
            self.mw.vtk_panel.update_status(str(error))

    def reset(self):
        self.generation += 1
        if self._active_job is not None:
            self._active_job.cancel()
        if self.fill is not None:
            self.mw.vtk_panel.scene.RemoveActor(self.fill)
            self.mw.vtk_panel._release_mesh_component(self.fill)
        self.fill = None
        self.closed = False
        self.region = None
        self.layer = None
        self.paths.clear()
        self.closure_redo.clear()

    def route(self, anchors, close=False):
        if not anchors:
            return []
        route = [anchors[0]]
        for end in [*anchors[1:], *([anchors[0]] if close else [])]:
            pair = route[-1], end
            if pair not in self.paths:
                surface = self.cache[self.layer["id"]]
                self.paths[pair] = self.work(
                    "Finding surface path...",
                    lambda check: surface.path(*pair, check=check),
                    show_delay_ms=PROGRESS_DELAY_MS,
                )
            route.extend(self.paths[pair][1:])
        return route

    def click(self, point, cell):
        try:
            surface = self.cache[self.layer["id"]]
            if self.busy:
                return
            if cell < 0 or cell >= len(surface.faces):
                raise ValueError("Pick an original triangle")
            ids = surface.faces[cell]
            coords = ras_points(surface, ids, self.layer)
            if not self.closed:
                anchor = int(ids[np.linalg.norm(coords - point, axis=1).argmin()])
                anchors = self.mw.mesh_polyline_points
                route = self.route([*anchors, anchor])
                if len(route) != len(set(route)):
                    raise ValueError(
                        "Boundary revisits a vertex. Choose another point."
                    )
                anchors.append(anchor)
                self.closure_redo.clear()
                self.manager._draft_redo.clear()
                self.manager._update_preview()
                return
            loop = self.route(self.mw.mesh_polyline_points, True)
            faces = self.work(
                "Selecting surface territory...",
                lambda check: surface.region(loop, cell, check=check),
                show_delay_ms=PROGRESS_DELAY_MS,
            )
            region = {
                "loop": np.asarray(loop),
                "faces": faces,
                "digest": surface.digest,
            }
            fill = self.actor(self.layer, region, self.manager._draft_color)
            if self.fill is not None:
                self.mw.vtk_panel.scene.RemoveActor(self.fill)
                self.mw.vtk_panel._release_mesh_component(self.fill)
            self.fill, self.region = fill, region
            self.mw.vtk_panel.scene.AddActor(fill)
            triangles = surface.vertices[surface.faces[faces]]
            area = (
                np.linalg.norm(
                    np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    ),
                    axis=1,
                ).sum()
                / 2
            )
            _, counts = np.unique(surface.face_edges[faces], return_counts=True)
            irregular = np.count_nonzero(counts > 2)
            detail = f" ({irregular} internal non-manifold edges)" if irregular else ""
            self.mw.vtk_panel.update_status(
                f"{len(faces):,} triangles{detail}, {area:.2f} mm². Enter saves; Esc cancels."
            )
        except InterruptedError:
            return
        except (ValueError, OSError) as error:
            self.mw.vtk_panel.update_status(str(error))

    def actor(self, layer, region, color):
        surface = self.surface(layer)
        if region["digest"] != surface.digest:
            raise ValueError("Polygon source fingerprint mismatch")
        patch, _ = patch_geometry(surface, region, layer)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(patch)
        mapper.ScalarVisibilityOff()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOff()
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.35)
        return actor

    def finish(self):
        anchors = self.mw.mesh_polyline_points
        try:
            if not self.closed:
                if len(anchors) < 3:
                    raise ValueError("Pick at least three distinct vertices")
                loop = self.route(anchors, True)
                if len(set(loop[:-1])) != len(loop) - 1:
                    raise ValueError("Closing path intersects the boundary")
                self.closure_redo.clear()
                self.manager._draft_redo.clear()
                self.closed = True
                self.mw.vtk_panel.update_status("Click a triangle on the side to keep.", render=False)
                self.manager._update_preview()
            elif self.region is not None:
                layer, region = self.layer, self.region
                color = self.manager._draft_color
                try:
                    published = self.publish(
                        layer, region, "Surface polygon", color, confirm=True
                    )
                except Exception as error:
                    self.mw.vtk_panel.update_status(
                        f"Could not save surface polygon: {error}"
                    )
                    return
                if published:
                    self.manager.cancel()
        except InterruptedError:
            return
        except (ValueError, OSError) as error:
            self.mw.vtk_panel.update_status(str(error))

    def restore_closure(self):
        region = self.closure_redo[-1]
        if region is not None:
            self.fill = self.actor(self.layer, region, self.manager._draft_color)
            self.mw.vtk_panel.scene.AddActor(self.fill)
        self.region = region
        self.closed = True
        self.closure_redo.clear()

    def publish(self, layer, region, name, color, *, confirm=False):
        points = ras_points(self.surface(layer), region["loop"], layer)
        return self.manager.publish_polylines(
            [
                dict(
                    points=points,
                    region=region,
                    name=name,
                    color=color,
                    visible=True,
                    **(self.manager._draft_style if confirm else validated_style()),
                )
            ],
            layer["id"],
            confirming_polygon=confirm,
        )

    def _resolve_import_mesh(self, key):
        if key is not None:
            return key if key in self.mw.mesh_layers else None
        if not self.mw.mesh_layers:
            return None
        items = self.mw.data_tree_widget.selectedItems()
        data = items[0].data(0, Qt.ItemDataRole.UserRole) if items else None
        selected = data.get("id") if isinstance(data, dict) else None
        if selected in self.mw.mesh_layers:
            return selected
        if len(self.mw.mesh_layers) == 1:
            return next(iter(self.mw.mesh_layers))
        choices = {
            f"{layer['display_name']} [{identity[:8]}]": identity
            for identity, layer in self.mw.mesh_layers.items()
        }
        choice, accepted = QInputDialog.getItem(
            self.mw, "Import Surface Polygon", "Mesh:", list(choices), 0, False,
        )
        return choices[choice] if accepted else None

    def import_pair(self, key=None):
        if not self.manager.can_import(key):
            return
        key = self._resolve_import_mesh(key)
        if key is None or not self.manager.can_import(key):
            return
        layer = self.mw.mesh_layers[key]
        paths, _ = choose_input(
            QFileDialog.getOpenFileNames,
            self.mw,
            "Import Surface Polygon (boundary and patch)",
            "",
            "Surface Polygons (*.vtp *.obj);;"
            "VTK XML PolyData (*.vtp);;Wavefront OBJ (*.obj)",
        )
        if not paths:
            return
        try:
            line, patch = resolve_import_pair(paths)
            surface = self.surface(layer)
            if line.suffix.lower() == ".obj":
                region, line, color = self.work(
                    "Checking surface territory...",
                    lambda check: read_obj_pair(
                        line, patch, surface, layer, check,
                        return_boundary=True, return_color=True,
                    ),
                    show_delay_ms=PROGRESS_DELAY_MS,
                )
            else:
                region, line = self.work(
                    "Checking surface territory...",
                    lambda check: read_pair(
                        line, patch, surface, layer, check, return_boundary=True,
                    ),
                    show_delay_ms=PROGRESS_DELAY_MS,
                )
                color = None
            if self.mw.mesh_layers.get(key) is not layer:
                raise ValueError("Mesh changed during import")
            self.publish(
                layer,
                region,
                Path(line).stem.removesuffix("_polygon"),
                color if color is not None else next(self.manager._available_colors()),
            )
        except InterruptedError:
            return
        except (ValueError, OSError) as error:
            QMessageBox.warning(self.mw, "Surface polygon", str(error))

    def export(self, layer, record):
        name, selected = QFileDialog.getSaveFileName(
            self.mw,
            "Export Surface Polygon",
            record["name"] + "_polygon.vtp",
            "VTK XML PolyData - boundary & patch (*.vtp);;"
            "Wavefront OBJ - boundary & patch (*.obj)",
        )
        if not name:
            return
        if not Path(name).suffix:
            name += ".obj" if "*.obj" in selected else ".vtp"
        try:
            paths = output_paths(name)
            sources = self.manager.export_sources()
            for path in paths:
                protected_export_path(path, sources)
            if (
                any(p.exists() or p.is_symlink() for p in paths)
                and QMessageBox.question(
                    self.mw, "Replace annotation files?", "\n".join(map(str, paths)),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
            if (not self.mw.mesh_manager._can_mutate()
                    or self.mw.mesh_layers.get(layer["id"]) is not layer
                    or self.manager._record(layer["id"], record["id"]) is not record):
                return
            surface = self.surface(layer)
            if paths[0].suffix.lower() == ".obj":
                writer = lambda check: write_obj_pair(
                    paths, surface, record["region"], layer, record["color"], check,
                    protected_paths=sources,
                )
            else:
                writer = lambda check: write_pair(
                    paths, surface, record["region"], layer, check,
                    protected_paths=sources,
                )
            self.work(
                "Exporting surface territory...",
                writer,
                commits_files=True,
                show_delay_ms=PROGRESS_DELAY_MS,
            )
            self.mw.vtk_panel.update_status(
                f"Surface polygon exported: {paths[0].name}, {paths[1].name}"
            )
        except InterruptedError:
            return
        except (OSError, ValueError) as error:
            QMessageBox.warning(self.mw, "Surface polygon export", str(error))
