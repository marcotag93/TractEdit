from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
from PyQt6.QtCore import QEvent

from tractedit_pkg.macos_documents import DocumentApplication


class FinderApp(SimpleNamespace):
    release_initial_window = DocumentApplication.release_initial_window
    _open_document = DocumentApplication._open_document

    def __init__(self, pending, *, empty=False):
        window = SimpleNamespace(
            isVisible=lambda: True, _background_workers=[], _session_busy=False, session_path=None,
            unified_undo_stack=[], roi_layers={}, tractogram_data=None,
            anatomical_image_data=None, parcellation_data=None, odf_data=None,
            mesh_layers={}, standalone_polylines=[], load_initial_files=Mock(),
        )
        super().__init__(
            _pending=pending, _initial_window=window if empty else None,
            _open_separately=Mock(), removeEventFilter=Mock(), activeModalWidget=lambda: None,
        )


@pytest.mark.parametrize("first, third", [
    ("first.TRK", "third.tck"),
    ("first.VTK", "third.vtp"),
    ("first.VTP", "third.vtk"),
    ("Superficie è 01.OBJ", "second mesh.obj"),
    ("first.NII.GZ", "third.nii"),
    ("first.nii", "third.nii.gz"),
    ("MRI è.TRACTEDIT-SESSION", "second session.tractedit-session"),
])
def test_finder_open_during_startup_and_with_an_active_document(first, third):
    app = FinderApp([], empty=True)
    initial = app._initial_window.load_initial_files
    for name in (first, "second file.trx", "./second file.trx"):
        event = Mock()
        event.type.return_value = QEvent.Type.FileOpen
        event.file.return_value = name
        assert DocumentApplication.event(app, event)
    DocumentApplication.finish_startup(app, None)
    if first.lower().endswith((".nii", ".nii.gz")):
        initial.assert_called_once_with(None, first)
    else:
        initial.assert_called_once_with(first, None)
    app._open_separately.assert_called_once_with("second file.trx")

    event.file.return_value = third
    assert DocumentApplication.event(app, event)
    app._open_separately.assert_called_with(third)


@pytest.mark.parametrize("document", ["cli.trk", "Superficie è 01.OBJ"])
def test_cli_document_is_preserved(document):
    app = FinderApp([document, "finder.tck", "finder.tck"])
    DocumentApplication.finish_startup(app, document)
    app._open_separately.assert_called_once_with("finder.tck")


def test_cli_and_finder_aliases_open_one_document(tmp_path):
    document = tmp_path / "Superficie è 01.OBJ"
    document.touch()
    alias = tmp_path / "alias.obj"
    if sys.platform == "win32":
        alias.hardlink_to(document)
    else:
        alias.symlink_to(document)
    other = tmp_path / "other.obj"
    other.touch()
    app = FinderApp([str(document), str(other)])
    DocumentApplication.finish_startup(app, str(alias))
    app._open_separately.assert_called_once_with(str(other))


def test_finder_preserves_initial_image_and_opens_next_separately():
    app = FinderApp(["T1.nii", "T2.nii.gz"])
    DocumentApplication.finish_startup(app, None, "T1.nii")
    app._open_separately.assert_called_once_with("T2.nii.gz")


@pytest.mark.parametrize("in_use", [None, "interaction", "loaded", "loading"])
def test_first_finder_event_after_startup(in_use):
    app = FinderApp([], empty=True)
    initial = app._initial_window.load_initial_files
    DocumentApplication.finish_startup(app, None)
    if in_use == "interaction":
        app.release_initial_window()
    elif in_use == "loaded":
        app._initial_window.tractogram_data = object()
    elif in_use == "loading":
        app._initial_window._background_workers.append(object())
    event = Mock()
    event.type.return_value = QEvent.Type.FileOpen
    event.file.return_value = "late.nii.gz"
    assert DocumentApplication.event(app, event)
    if in_use:
        initial.assert_not_called()
        app._open_separately.assert_called_once_with("late.nii.gz")
    else:
        initial.assert_called_once_with(None, "late.nii.gz")
        app._open_separately.assert_not_called()
    app._open_separately.reset_mock()
    assert DocumentApplication.event(app, event)
    app._open_separately.assert_called_once_with("late.nii.gz")


def test_running_bundle_starts_independent_packaged_process(monkeypatch):
    launch = Mock()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr("tractedit_pkg.macos_documents.subprocess.Popen", launch)
    path = "/Volumes/Diffusion MRI/fascicolo sinistro.trk"
    DocumentApplication._open_separately(None, path)
    assert launch.call_args.args == ([sys.executable, "--", path],)
    assert launch.call_args.kwargs["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
