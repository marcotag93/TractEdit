"""Qt worker cancellation, modal execution and deferred resource ownership."""

import logging
from threading import Event
from typing import Any

from PyQt6.QtCore import QEventLoop, QThread, QTimer, Qt
from PyQt6.QtWidgets import QProgressDialog

logger = logging.getLogger(__name__)


class CancellableWorkerThread(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = Event()
        self._result_pending = False
        self.is_consuming_result = False

    def cancel(self):
        self._cancelled.set()

    @property
    def is_cancelled(self):
        return self._cancelled.is_set()

    @property
    def has_pending_result(self):
        return self._result_pending

    def begin_result(self):
        self.is_consuming_result = True

    def complete_result(self):
        self._result_pending = False
        self.is_consuming_result = False


def create_progress_dialog(parent, title, *, factory=QProgressDialog):
    dialog = factory(title, "Cancel", 0, 0, parent)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(0)
    return dialog


def run_modal_worker(worker, dialog):
    """Keep GUI events flowing until the worker finishes, including fast jobs."""
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    dialog.show()
    worker.start()
    if not worker.isFinished():
        loop.exec()
    worker.wait()


def register_background_worker(
    main_window: Any,
    worker_attr: str,
    generation_attr: str,
    worker: Any,
) -> int:
    """Register one owned worker and invalidate any preceding operation."""
    previous = getattr(main_window, worker_attr, None)
    if previous is not None and previous is not worker:
        cancel = getattr(previous, "cancel", None)
        if callable(cancel):
            cancel()
        progress_dialog = getattr(previous, "progress_dialog", None)
        if progress_dialog is not None:
            try:
                progress_dialog.close()
            except RuntimeError:
                logger.debug("Previous worker progress dialog is already closed.")

    generation = getattr(main_window, generation_attr, 0) + 1
    setattr(main_window, generation_attr, generation)
    setattr(main_window, worker_attr, worker)

    workers = getattr(main_window, "_background_workers", None)
    if workers is None:
        workers = []
        setattr(main_window, "_background_workers", workers)
    workers.append(worker)
    return generation


def worker_is_current(
    main_window: Any,
    worker_attr: str,
    generation_attr: str,
    worker: Any,
    generation: int,
) -> bool:
    return (
        getattr(main_window, worker_attr, None) is worker
        and getattr(main_window, generation_attr, 0) == generation
        and not getattr(worker, "is_cancelled", False)
    )


def release_background_worker(
    main_window: Any,
    worker_attr: str,
    worker: Any,
) -> None:
    workers = getattr(main_window, "_background_workers", None)
    if workers is not None:
        try:
            workers.remove(worker)
        except ValueError:
            pass
    if getattr(main_window, worker_attr, None) is worker:
        setattr(main_window, worker_attr, None)
    if hasattr(worker, "progress_dialog"):
        progress_dialog = worker.progress_dialog
        if progress_dialog is not None:
            try:
                progress_dialog.deleteLater()
            except RuntimeError:
                logger.debug("Worker progress dialog was already deleted.")
        worker.progress_dialog = None

    update_actions = getattr(main_window, "_update_action_states", None)
    if callable(update_actions):
        try:
            update_actions()
        except Exception:
            logger.warning(
                "Failed to refresh actions after worker completion.", exc_info=True
            )


def release_finished_worker(
    main_window: Any,
    worker_attr: str,
    worker: Any,
) -> None:
    if worker.has_pending_result:
        return
    if worker.isRunning():
        QTimer.singleShot(
            10,
            lambda: release_finished_worker(main_window, worker_attr, worker),
        )
        return
    release_background_worker(main_window, worker_attr, worker)


def release_deferred_trx_owners(main_window: Any) -> None:
    deferred = getattr(main_window, "_deferred_trx_owners", [])
    workers = getattr(main_window, "_background_workers", [])
    remaining = []
    for owner in deferred:
        in_use = any(
            getattr(worker, "trx_owner", None) is owner
            and (
                worker.isRunning()
                or getattr(worker, "is_consuming_result", False)
                or getattr(worker, "has_pending_result", False)
            )
            for worker in workers
        )
        if in_use:
            remaining.append(owner)
            continue
        try:
            owner.close()
        except OSError:
            logger.warning("Failed to close retired TRX owner.", exc_info=True)
            remaining.append(owner)
    main_window._deferred_trx_owners = remaining


def retire_trx_owner(main_window: Any, owner: Any) -> None:
    if owner is None:
        return
    panel = getattr(
        getattr(main_window, "data_panel_manager", None), "tractogram_data", None
    )
    if panel is not None:
        panel.retire(owner)
    workers = getattr(main_window, "_background_workers", [])
    users = [
        worker for worker in workers if getattr(worker, "trx_owner", None) is owner
    ]
    for worker in users:
        worker.cancel()
        progress_dialog = getattr(worker, "progress_dialog", None)
        if progress_dialog is not None:
            try:
                progress_dialog.close()
            except RuntimeError:
                logger.debug("Previous worker progress dialog is already closed.")
    if any(
        worker.isRunning()
        or getattr(worker, "is_consuming_result", False)
        or getattr(worker, "has_pending_result", False)
        for worker in users
    ):
        deferred = getattr(main_window, "_deferred_trx_owners", None)
        if deferred is None:
            deferred = []
            main_window._deferred_trx_owners = deferred
        if not any(item is owner for item in deferred):
            deferred.append(owner)
        return
    try:
        owner.close()
    except OSError:
        logger.warning("Failed to close retired TRX owner.", exc_info=True)
        deferred = getattr(main_window, "_deferred_trx_owners", None)
        if deferred is None:
            deferred = []
            main_window._deferred_trx_owners = deferred
        if not any(item is owner for item in deferred):
            deferred.append(owner)
