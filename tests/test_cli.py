"""Regression tests for the TractEdit command-line interface."""

import sys
from unittest.mock import Mock

import pytest

import main as tractedit_main
from tractedit_pkg import __version__
from tractedit_pkg.cli import build_argument_parser, gui_document_paths


@pytest.mark.parametrize("document", ["T1.nii", "T1 scan.NII.GZ"])
def test_gui_image_document(document):
    assert gui_document_paths(document, None) == (None, document)
    with pytest.raises(ValueError):
        gui_document_paths(document, "other.nii")


@pytest.mark.parametrize("extension", ["trk", "trx", "tck", "vtk", "vtp", "OBJ", "tractedit-session", "nii", "NII.GZ"])
def test_open_with_document_argument(extension):
    document = rf"C:\Dati MRI\Fascicolo è 01.{extension}"
    args = build_argument_parser().parse_args([document])
    expected = (None, document) if extension in {"nii", "NII.GZ"} else (document, None)
    assert gui_document_paths(args.bundle, args.anat) == expected


def test_gui_bundle_with_explicit_reference_and_generic_gzip():
    assert gui_document_paths("AF.trk", "T1.nii.gz") == ("AF.trk", "T1.nii.gz")
    assert gui_document_paths("archive.gz", None) == ("archive.gz", None)


def test_obj_document_uses_interactive_mesh_loader(tmp_path, monkeypatch):
    from tractedit_pkg.main_window import MainWindow, file_io

    path = tmp_path / "Superficie è 01.OBJ"
    path.touch()
    window = Mock()
    load_streamlines = Mock()
    monkeypatch.setattr(file_io, "load_streamlines_file", load_streamlines)

    MainWindow.load_initial_files(window, *gui_document_paths(str(path), None))

    window.mesh_manager.load.assert_called_once_with(str(path))
    load_streamlines.assert_not_called()


@pytest.mark.parametrize("options", [
    ["--anat", "T1.nii"], ["--load-roi", "roi.nii"],
    ["--roi", "1", "2", "3"], ["--radius", "2"],
    ["--convert-to", "out.trk"], ["--density-map", "out.nii"],
])
def test_session_rejects_additional_inputs_and_exports(monkeypatch, capsys, options):
    monkeypatch.setattr(sys, "argv", ["tractedit", "MRI è.TRACTEDIT-SESSION", *options])
    with pytest.raises(SystemExit) as error:
        tractedit_main.main()
    assert error.value.code == 2
    assert "Open a saved session on its own" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["-v", "-V", "--version"])
def test_version_flags_exit_cleanly_and_print_current_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["tractedit", flag])

    with pytest.raises(SystemExit) as exc_info:
        tractedit_main.main()

    assert exc_info.value.code == 0
    assert f"Version: {__version__}" in capsys.readouterr().out
