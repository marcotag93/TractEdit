from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vtkmodules.vtkCommonExecutionModel import vtkPolyDataAlgorithm

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from ..data_contracts import MeshLoadResult
from ..mesh.model import (
    mesh_metadata_from_ply_comments,
    mesh_metadata_from_vtp,
    mesh_space_metadata,
)
from .validation import MeshLoadError, _check_cancel
from .ply import _DEFAULT_CHECK_PLY_FILE
from .obj import _DEFAULT_CHECK_OBJ_FILE
from .. import mesh_io


def _create_reader(extension: str) -> "vtkPolyDataAlgorithm":
    if extension == ".ply":
        from vtkmodules.vtkIOPLY import vtkPLYReader

        reader = vtkPLYReader()
        # Face textures must not duplicate source vertices or alter topology.
        reader.SetDuplicatePointsForFaceTexture(False)
        return reader
    if extension == ".obj":
        from vtkmodules.vtkIOGeometry import vtkOBJReader

        return vtkOBJReader()
    if extension == ".vtp":
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

        return vtkXMLPolyDataReader()
    raise MeshLoadError("Unsupported surface reader format")


def load_mesh(
    path: str | os.PathLike[str], *, cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> MeshLoadResult:
    """Read a surface or VTP lines, preserving coordinates and metadata.

    Cancellable OBJ reads isolate native Update() in a child process. Other
    reads check cancellation around Update() and between validation blocks.
    The detached polydata belongs to the receiver and must not be mutated after publication.
    SPACE resolution is a separate mesh_model operation, allowing explicit
    overrides of missing or conflicting metadata before scene publication.
    VTK 9.2.6 reads OBJ and PLY points as float32, including declared doubles;
    VTP retains its stored float32/float64 precision. No further narrowing or
    attempt to recover lost source precision is performed here.
    """
    _check_cancel(cancel_check)
    source = Path(path).absolute()
    if not source.is_file():
        raise MeshLoadError(f"Mesh path is not a regular file: {source}")
    extension = source.suffix.lower()
    if extension not in (".ply", ".vtp", ".obj"):
        raise MeshLoadError("Surface format must be .ply, .vtp or .obj")
    from ..session.io import source_stamp

    stamp = source_stamp(source)
    obj_face_count = None
    obj_texture_index = None
    source_digest = None
    if extension == ".ply":
        if mesh_io._check_ply_file is _DEFAULT_CHECK_PLY_FILE:
            candidate_digest = hashlib.sha256()
            if mesh_io._check_ply_file(source, cancel_check, candidate_digest):
                source_digest = candidate_digest
        elif mesh_io._check_ply_file(source, cancel_check):
            # The display reader can discard precision needed by territory export.
            source_digest = hashlib.sha256()
            with source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    _check_cancel(cancel_check)
                    source_digest.update(block)
    elif extension == ".obj":
        source_digest = hashlib.sha256()
        consumed = 0
        reported = -1

        def scanned(block):
            nonlocal consumed, reported
            source_digest.update(block)
            consumed += len(block)
            value = min(30, consumed * 30 // max(1, stamp[2]))
            if progress is not None and value != reported:
                progress(value, "Checking OBJ geometry...")
                reported = value

        def preflight(consume):
            if mesh_io._check_obj_file is _DEFAULT_CHECK_OBJ_FILE:
                return mesh_io._scan_obj_file(
                    source, cancel_check, consume, collect_texture_index=True
                )
            return mesh_io._check_obj_file(source, cancel_check, consume), None

        if cancel_check is None:
            obj_face_count, obj_texture_index = preflight(scanned)
        else:
            from ..mesh.process import load_obj_in_process

            with TemporaryDirectory(prefix="tractedit-source-") as directory:
                snapshot = Path(directory) / source.name
                with snapshot.open("wb") as output:
                    def capture(block):
                        output.write(block)
                        scanned(block)

                    obj_face_count, obj_texture_index = preflight(capture)
                _check_cancel(cancel_check)
                result = load_obj_in_process(
                    snapshot, obj_face_count, cancel_check,
                    **({"progress": progress} if progress is not None else {}),
                )
            result["path"] = str(source)
            result["verified_geometry_sha256"] = source_digest.hexdigest()
    _check_cancel(cancel_check)
    if extension != ".obj" or cancel_check is None:
        result = mesh_io._read_mesh(
            source, obj_face_count, cancel_check,
            **({"progress": progress} if progress is not None else {}),
        )
    if source_stamp(source) != stamp:
        raise MeshLoadError(f"{extension[1:].upper()} source changed while loading")
    result["source_stamp"] = stamp
    if source_digest is not None:
        result["source_sha256"] = source_digest.hexdigest()
    if obj_texture_index is not None:
        result["obj_texture_index"] = obj_texture_index
    return result


def read_annotation_vtp(path: str | os.PathLike[str], cancel_check=None):
    source = Path(path)
    if source.suffix.lower() != ".vtp" or not source.is_file():
        raise MeshLoadError("Expected a VTP annotation file")
    return mesh_io._read_mesh(source, None, cancel_check, annotation=True)["polydata"]


def _read_mesh(
    source: Path,
    obj_face_count: int | None,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[int, str], None] | None = None,
    *, annotation: bool = False,
) -> MeshLoadResult:
    from vtkmodules.util.misc import calldata_type
    from vtkmodules.vtkCommonCore import VTK_STRING
    from vtkmodules.vtkCommonDataModel import vtkPolyData

    extension = source.suffix.lower()
    reader = mesh_io._create_reader(extension)
    diagnostics = []

    @calldata_type(VTK_STRING)
    def on_error(caller, event: str, message: str | None) -> None:
        if len(diagnostics) < 8:
            diagnostics.append(str(message or event)[:2048])

    observers = []
    try:
        if progress is not None:
            reported = [-1]

            def reading(caller, event):
                value = 30 + int(40 * caller.GetProgress())
                if value != reported[0]:
                    progress(value, "Reading OBJ surface...")
                    reported[0] = value

            observers.append((reader, reader.AddObserver("ProgressEvent", reading)))
        for emitter in (reader, reader.GetExecutive()):
            for event in ("ErrorEvent", "WarningEvent"):
                observers.append((emitter, emitter.AddObserver(event, on_error)))
        reader.SetFileName(str(source))
        _check_cancel(cancel_check)
        try:
            # Algorithm.Update() discards this status; events may be disabled.
            pipeline_status = reader.GetExecutive().Update()
        except (RuntimeError, OSError, ValueError) as error:
            raise MeshLoadError(f"VTK could not read surface: {error}") from error
        _check_cancel(cancel_check)
        error_code = reader.GetErrorCode()
        if not pipeline_status or diagnostics or error_code:
            detail = (
                "; ".join(diagnostics)
                or f"pipeline status {pipeline_status}, error code {error_code}"
            )
            raise MeshLoadError(f"VTK surface read failed: {detail}")
        output = reader.GetOutput()
        if output is None:
            raise MeshLoadError("VTK returned no surface data")
        if obj_face_count is not None and output.GetNumberOfPolys() != obj_face_count:
            raise MeshLoadError("VTK OBJ reader did not preserve every polygon face")
        # Retain isolated zero-area OBJ triangles without cleaning or reindexing
        # the reader output; the surface must still contain nonzero-area faces.
        if progress is not None:
            progress(75, "Validating surface geometry...")
        bounds = mesh_io._validate_polydata(
            output, cancel_check, allow_zero_area=extension == ".obj" or annotation,
            annotation=annotation,
            allow_lines=extension == ".vtp" and bool(output.GetNumberOfLines()),
        )
        if extension == ".ply":
            comments = reader.GetComments()
            metadata = mesh_metadata_from_ply_comments(
                tuple(comments.GetValue(i) for i in range(comments.GetNumberOfValues()))
            )
        elif extension == ".vtp":
            metadata = mesh_metadata_from_vtp(output)
        else:
            metadata = mesh_space_metadata("obj_file", None)
        detached = vtkPolyData()
        detached.ShallowCopy(output)
        _check_cancel(cancel_check)
        return {
            "path": str(source),
            "polydata": detached,
            "space_metadata": metadata,
            "point_count": detached.GetNumberOfPoints(),
            "polygon_count": detached.GetNumberOfPolys(),
            "strip_count": detached.GetNumberOfStrips(),
            "source_bounds": bounds,
        }
    finally:
        for emitter, observer in observers:
            emitter.RemoveObserver(observer)
