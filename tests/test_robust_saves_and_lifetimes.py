# -*- coding: utf-8 -*-
"""Regression tests for transactional saves and background resource ownership."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from tractedit_pkg import file_io


class _FakeTrx:
    def __init__(self) -> None:
        self.closed = False
        self.header = {}
        self.streamlines = nib.streamlines.ArraySequence(
            [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
        )
        self.data_per_vertex = {}
        self.data_per_streamline = {}
        self.groups = {}
        self.data_per_group = {}

    def close(self) -> None:
        self.closed = True


class _Signal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback, **_kwargs) -> None:
        self.callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in self.callbacks:
            callback(*args)


class _ResultLoader:
    result = None

    def __init__(self, *_args) -> None:
        self.progress = _Signal()
        self.error = _Signal()
        self.finished = _Signal()
        self.is_cancelled = False
        self.has_pending_result = False
        self.is_consuming_result = False

    def start(self) -> None:
        self.finished.emit(self.result)

    def cancel(self) -> None:
        self.is_cancelled = True

    def complete_result(self) -> None:
        self.has_pending_result = False
        self.is_consuming_result = False

    def begin_result(self) -> None:
        self.is_consuming_result = True

    def isRunning(self) -> bool:
        return False

    def take_trx_owner(self, _result):
        return None

    def discard_result(self, _result) -> None:
        pass

    def take_result(self, result):
        return result


def test_staged_output_preserves_destination_and_removes_stage_on_failure(
    tmp_path,
):
    from tractedit_pkg.transactional_io import staged_output

    destination = tmp_path / "bundle.trk"
    destination.write_bytes(b"valid previous output")
    staged_path = None

    with pytest.raises(OSError, match="disk full"):
        with staged_output(destination) as pending:
            staged_path = pending
            assert pending.suffix == ".trk"
            pending.write_bytes(b"partial output")
            raise OSError("disk full")

    assert destination.read_bytes() == b"valid previous output"
    assert staged_path is not None and not staged_path.exists()


def test_staged_output_replaces_destination_only_after_success(tmp_path):
    from tractedit_pkg.transactional_io import staged_output

    destination = tmp_path / "roi.nii.gz"
    destination.write_bytes(b"old")

    with staged_output(destination) as pending:
        assert pending.name.endswith(".nii.gz")
        pending.write_bytes(b"new complete output")
        assert destination.read_bytes() == b"old"

    assert destination.read_bytes() == b"new complete output"
    assert not list(tmp_path.glob(".roi.*.nii.gz"))


def test_staged_output_preserves_locked_destination(tmp_path, monkeypatch):
    from tractedit_pkg import transactional_io

    destination = tmp_path / "bundle.trx"
    destination.write_bytes(b"old")

    def reject_replace(_source, _destination):
        raise PermissionError("destination is locked")

    monkeypatch.setattr(transactional_io.os, "replace", reject_replace)
    with pytest.raises(PermissionError, match="locked"):
        with transactional_io.staged_output(destination) as staged:
            staged.write_bytes(b"new")

    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tractedit-stage-*.trx"))


def test_staged_output_set_rolls_back_all_destinations_on_commit_failure(
    tmp_path, monkeypatch
):
    from tractedit_pkg import transactional_io

    matrix = tmp_path / "matrix.npy"
    labels = tmp_path / "matrix_labels.json"
    matrix.write_bytes(b"old matrix")
    labels.write_bytes(b"old labels")
    real_replace = transactional_io.os.replace
    commit_count = 0

    def fail_second_commit(source, destination):
        nonlocal commit_count
        source_path = Path(source)
        if ".tractedit-stage-" in source_path.name:
            commit_count += 1
            if commit_count == 2:
                raise PermissionError("locked destination")
        real_replace(source, destination)

    monkeypatch.setattr(transactional_io.os, "replace", fail_second_commit)

    with pytest.raises(PermissionError, match="locked destination"):
        with transactional_io.staged_output_set([matrix, labels]) as pending:
            pending[matrix].write_bytes(b"new matrix")
            pending[labels].write_bytes(b"new labels")

    assert matrix.read_bytes() == b"old matrix"
    assert labels.read_bytes() == b"old labels"
    assert not list(tmp_path.glob("*.tractedit-backup-*.npy"))
    assert not list(tmp_path.glob("*.tractedit-backup-*.json"))


def test_tractogram_writer_failure_preserves_existing_destination(
    tmp_path, monkeypatch
):
    destination = tmp_path / "bundle.trk"
    destination.write_bytes(b"previous tractogram")
    tractogram = nib.streamlines.Tractogram(
        [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)],
        affine_to_rasmm=np.eye(4),
    )

    def fail_after_partial_write(_file, path):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(file_io.nib.streamlines, "save", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated write failure"):
        file_io._save_tractogram_file(
            tractogram,
            {},
            str(destination),
            ".trk",
        )

    assert destination.read_bytes() == b"previous tractogram"


def test_cancelled_trx_loader_closes_unpublished_owner(monkeypatch):
    trx = _FakeTrx()
    loader = file_io.StreamlineLoaderThread("cancelled.trx")

    def load_and_cancel(_path):
        loader.cancel()
        return trx

    monkeypatch.setattr(file_io.tbx, "load", load_and_cancel)
    completed = []
    loader.finished.connect(completed.append)

    loader.run()

    assert not completed
    assert trx.closed


def test_successful_trx_loader_transfers_owner_explicitly(monkeypatch):
    trx = _FakeTrx()
    monkeypatch.setattr(file_io.tbx, "load", lambda _path: trx)
    monkeypatch.setattr(
        file_io,
        "_compute_bboxes_numba",
        lambda *_args: np.zeros((1, 2, 3), dtype=np.float32),
    )
    loader = file_io.StreamlineLoaderThread("loaded.trx")
    completed = []
    loader.finished.connect(completed.append)

    loader.run()

    assert len(completed) == 1
    assert not trx.closed
    assert loader.take_trx_owner(completed[0]) is trx
    assert loader.take_trx_owner(completed[0]) is None


def test_native_trx_save_closes_subset_on_success_and_failure(tmp_path, monkeypatch):
    subset = _FakeTrx()
    source = SimpleNamespace(
        streamlines=[np.zeros((2, 3), dtype=np.float32)],
        groups={},
        select=Mock(return_value=subset),
    )
    monkeypatch.setattr(
        file_io.tbx,
        "save",
        lambda _trx, path: Path(path).write_bytes(b"trx"),
    )

    file_io._save_trx_native(source, {0}, None, str(tmp_path / "ok.trx"))
    assert subset.closed

    failed_subset = _FakeTrx()
    source.select.return_value = failed_subset
    monkeypatch.setattr(
        file_io.tbx,
        "save",
        Mock(side_effect=OSError("write failed")),
    )

    with pytest.raises(OSError, match="write failed"):
        file_io._save_trx_native(source, {0}, None, str(tmp_path / "failed.trx"))
    assert failed_subset.closed


def test_generic_trx_save_closes_temporary_owner(tmp_path, monkeypatch):
    owner = _FakeTrx()
    tractogram = nib.streamlines.Tractogram(
        [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)],
        affine_to_rasmm=np.eye(4),
    )
    monkeypatch.setattr(
        file_io.tbx.TrxFile,
        "from_lazy_tractogram",
        Mock(return_value=owner),
    )
    monkeypatch.setattr(
        file_io.tbx,
        "save",
        lambda _trx, path: Path(path).write_bytes(b"trx"),
    )

    file_io._save_tractogram_file(
        tractogram,
        {"dimensions": (2, 2, 2), "voxel_to_rasmm": np.eye(4)},
        str(tmp_path / "output.trx"),
        ".trx",
    )

    assert owner.closed


def test_headless_trx_same_path_conversion_preserves_source(
    tmp_path,
):
    from main import _run_headless_conversion
    import trx.trx_file_memmap as tbx

    path = tmp_path / "same.trx"
    streamlines = [
        np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32),
        np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float32),
    ]
    tractogram = nib.streamlines.Tractogram(
        streamlines,
        affine_to_rasmm=np.eye(4),
    )
    reference = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), np.eye(4))
    source = tbx.TrxFile.from_tractogram(tractogram, reference)
    try:
        tbx.save(source, str(path))
    finally:
        source.close()

    _run_headless_conversion(str(path), str(path))
    assert not list(tmp_path.glob(".tractedit-source-*.trx"))

    loaded = tbx.load(str(path))
    try:
        actual = [np.asarray(streamline).copy() for streamline in loaded.streamlines]
    finally:
        loaded.close()
    for observed, expected in zip(actual, streamlines):
        np.testing.assert_array_equal(observed, expected)


def test_anatomical_loader_cancellation_is_cooperative(monkeypatch):
    data = np.zeros((2, 2, 2), dtype=np.float32)
    image = nib.Nifti1Image(data, np.eye(4))
    loader = file_io.AnatomicalImageLoaderThread("image.nii.gz")

    def load_and_cancel(_path):
        loader.cancel()
        return image

    monkeypatch.setattr(file_io.nib, "load", load_and_cancel)
    completed = []
    errors = []
    loader.finished.connect(completed.append)
    loader.error.connect(errors.append)

    loader.run()

    assert not completed
    assert not errors
    assert loader.is_cancelled


def test_bundle_finalization_failure_restores_previous_state(monkeypatch):
    old_streamlines = [np.zeros((2, 3), dtype=np.float32)]
    old_image = np.zeros((2, 2, 2), dtype=np.float32)
    old_mmap = Mock()
    new_streamlines = nib.streamlines.ArraySequence([np.ones((2, 3), dtype=np.float32)])
    _ResultLoader.result = {
        "streamlines": new_streamlines,
        "bboxes": np.ones((1, 2, 3), dtype=np.float32),
        "header": {},
        "affine": np.eye(4),
        "path": "new.trk",
        "ext": ".trk",
        "reference_grid": None,
        "scalars": {},
        "data_per_streamline": {},
        "active_scalar": None,
    }
    panel = Mock()
    panel.scene = object()
    panel.update_main_streamlines_actor.side_effect = RuntimeError("render failed")
    window = SimpleNamespace(
        vtk_panel=panel,
        theme_manager=Mock(),
        tractogram_data=old_streamlines,
        _tractogram_data_version=3,
        streamline_bboxes=np.zeros((1, 2, 3), dtype=np.float32),
        original_trk_header={"old": True},
        original_trk_affine=np.eye(4),
        original_trk_path="old.trk",
        original_file_extension=".trk",
        tractogram_reference_grid=None,
        trx_file_reference=None,
        scalar_data_per_point={"old": object()},
        data_per_streamline={"old": np.ones(1)},
        active_scalar_name="old",
        manual_visible_indices={0},
        visible_indices={0},
        _visibility_version=5,
        roi_states={"roi": {"include": True}},
        roi_intersection_cache={"roi": {0}},
        roi_highlight_indices={0},
        selected_streamline_indices={0},
        _inversion_active=True,
        _inversion_keeper_indices={0},
        unified_undo_stack=[{"old": "history"}],
        unified_redo_stack=[],
        current_color_mode=object(),
        scalar_min_val=0.25,
        scalar_max_val=0.75,
        scalar_data_min=0.0,
        scalar_data_max=1.0,
        anatomical_image_data=old_image,
        anatomical_image_affine=np.eye(4),
        anatomical_image_path="old.nii.gz",
        anatomical_mmap_image=old_mmap,
        anatomical_reference_grid=object(),
        image_is_visible=True,
        _update_action_states=Mock(),
        _update_bundle_info_display=Mock(),
        _update_data_panel_display=Mock(),
    )

    def mutate_scalar_state():
        window.scalar_min_val = -10.0
        window.scalar_max_val = 10.0
        window.scalar_data_min = -20.0
        window.scalar_data_max = 20.0

    window._auto_calculate_skip_level = mutate_scalar_state
    monkeypatch.setattr(file_io, "StreamlineLoaderThread", _ResultLoader)
    monkeypatch.setattr(file_io, "QProgressDialog", Mock())
    monkeypatch.setattr(file_io, "QMessageBox", Mock())

    file_io.load_streamlines_file(window, file_path="new.trk")

    assert window.tractogram_data is old_streamlines
    assert window.original_trk_path == "old.trk"
    assert window.anatomical_image_data is old_image
    assert window.anatomical_image_path == "old.nii.gz"
    old_mmap.clear_cache.assert_not_called()
    assert window.selected_streamline_indices == {0}
    assert window._inversion_active
    assert window._inversion_keeper_indices == {0}
    panel.update_invert_contour.assert_called_once_with()
    assert window.scalar_min_val == 0.25
    assert window.scalar_max_val == 0.75
    assert window.scalar_data_min == 0.0
    assert window.scalar_data_max == 1.0
    assert window.unified_undo_stack == [{"old": "history"}]


@pytest.mark.parametrize("failure", ["render", "roi_cleanup", "cancel"])
def test_image_finalization_failure_restores_previous_state(monkeypatch, failure):
    from tractedit_pkg import main_window

    old_data = np.zeros((2, 2, 2), dtype=np.float32)
    old_affine = np.eye(4)
    old_mmap = Mock()
    new_mmap = Mock()
    _ResultLoader.result = {
        "data": np.ones((2, 2, 2), dtype=np.float32),
        "affine": np.diag([2.0, 2.0, 2.0, 1.0]),
        "path": "new.nii.gz",
        "mmap_image": new_mmap,
        "reference_grid": object(),
    }
    panel = Mock()
    if failure == "render":
        panel.update_anatomical_slices.side_effect = RuntimeError("render failed")
    window = SimpleNamespace(
        anatomical_image_data=old_data,
        anatomical_image_affine=old_affine,
        anatomical_image_path="old.nii.gz",
        anatomical_mmap_image=old_mmap,
        anatomical_reference_grid=object(),
        image_is_visible=True,
        original_trk_path="old.trk",
        roi_layers={"roi": {"data": old_data, "affine": old_affine}},
        is_drawing_mode=True,
        current_drawing_roi="roi",
        theme_manager=Mock(),
        vtk_panel=panel,
        _update_bundle_info_display=Mock(),
        _update_action_states=Mock(),
        _update_data_panel_display=Mock(),
        _trigger_clear_all_rois=Mock(),
        _reset_all_drawing_modes=Mock(),
    )
    if failure == "roi_cleanup":

        def fail_roi_cleanup(notify=False):
            window.roi_layers.clear()
            window.is_drawing_mode = False
            window.current_drawing_roi = None
            raise RuntimeError("ROI cleanup failed")

        window._trigger_clear_all_rois.side_effect = fail_roi_cleanup
    if failure == "cancel":
        monkeypatch.setattr(
            main_window,
            "QApplication",
            SimpleNamespace(processEvents=lambda: window._image_loader_thread.cancel()),
        )
    confirmation = Mock()
    yes_button, no_button = object(), object()
    confirmation.addButton.side_effect = [yes_button, no_button]
    confirmation.clickedButton.return_value = yes_button
    message_box = Mock(return_value=confirmation)
    monkeypatch.setattr(main_window, "QMessageBox", message_box)
    monkeypatch.setattr(
        main_window,
        "QFileDialog",
        Mock(getOpenFileName=Mock(return_value=("new.nii.gz", ""))),
    )
    monkeypatch.setattr(main_window, "QProgressDialog", Mock())
    monkeypatch.setattr(file_io, "AnatomicalImageLoaderThread", _ResultLoader)

    main_window.MainWindow._trigger_load_anatomical_image(window)

    assert window.anatomical_image_data is old_data
    assert window.anatomical_image_affine is old_affine
    assert window.anatomical_image_path == "old.nii.gz"
    assert "roi" in window.roi_layers
    assert window.is_drawing_mode
    assert window.current_drawing_roi == "roi"
    if failure in {"render", "cancel"}:
        window._trigger_clear_all_rois.assert_not_called()
    else:
        window._trigger_clear_all_rois.assert_called_once_with(notify=False)
        panel.add_roi_layer.assert_called_once_with(
            "roi", old_data, old_affine, render=False
        )
    new_mmap.clear_cache.assert_called_once_with()


def test_image_commit_survives_old_cache_cleanup_failure(monkeypatch):
    from tractedit_pkg import main_window

    old_mmap = Mock()
    old_mmap.clear_cache.side_effect = OSError("cache is locked")
    new_data = np.ones((2, 2, 2), dtype=np.float32)
    new_mmap = Mock()
    _ResultLoader.result = {
        "data": new_data,
        "affine": np.diag([2.0, 2.0, 2.0, 1.0]),
        "path": "new.nii.gz",
        "mmap_image": new_mmap,
        "reference_grid": object(),
    }
    yes_button, no_button = object(), object()
    confirmation = Mock()
    confirmation.addButton.side_effect = [yes_button, no_button]
    confirmation.clickedButton.return_value = yes_button
    window = SimpleNamespace(
        anatomical_image_data=np.zeros((2, 2, 2), dtype=np.float32),
        anatomical_image_affine=np.eye(4),
        anatomical_image_path="old.nii.gz",
        anatomical_mmap_image=old_mmap,
        anatomical_reference_grid=object(),
        image_is_visible=True,
        original_trk_path="old.trk",
        roi_layers={},
        theme_manager=Mock(),
        vtk_panel=Mock(),
        _update_bundle_info_display=Mock(),
        _update_action_states=Mock(),
        _update_data_panel_display=Mock(),
        _trigger_clear_all_rois=Mock(),
        _reset_all_drawing_modes=Mock(),
    )
    monkeypatch.setattr(main_window, "QMessageBox", Mock(return_value=confirmation))
    monkeypatch.setattr(
        main_window,
        "QFileDialog",
        Mock(getOpenFileName=Mock(return_value=("new.nii.gz", ""))),
    )
    monkeypatch.setattr(main_window, "QProgressDialog", Mock())
    monkeypatch.setattr(file_io, "AnatomicalImageLoaderThread", _ResultLoader)

    main_window.MainWindow._trigger_load_anatomical_image(window)

    assert window.anatomical_image_data is new_data
    assert window.anatomical_image_path == "new.nii.gz"
    assert window.anatomical_mmap_image is new_mmap
    old_mmap.clear_cache.assert_called_once_with()
    new_mmap.clear_cache.assert_not_called()
    assert window._image_loader_thread is None
    assert not window._background_workers


def test_worker_is_retained_until_result_delivery_finishes():
    worker = Mock()
    worker.has_pending_result = True
    worker.isRunning.return_value = False
    window = SimpleNamespace(_background_workers=[worker], _loader_thread=worker)

    file_io._release_finished_worker(window, "_loader_thread", worker)
    assert window._loader_thread is worker

    worker.has_pending_result = False
    file_io._release_finished_worker(window, "_loader_thread", worker)
    assert window._loader_thread is None
    assert not window._background_workers


def test_trx_owner_is_closed_only_after_mapped_medoid_worker_stops():
    owner = Mock()
    worker = Mock(trx_owner=owner)
    worker.is_consuming_result = False
    worker.has_pending_result = False
    worker.isRunning.side_effect = [True, True, False]
    window = SimpleNamespace(_background_workers=[worker])

    file_io._retire_trx_owner(window, owner)
    worker.cancel.assert_called_once_with()
    owner.close.assert_not_called()

    file_io._release_deferred_trx_owners(window)
    owner.close.assert_not_called()

    file_io._release_deferred_trx_owners(window)
    owner.close.assert_called_once_with()
    assert not window._deferred_trx_owners
