# -*- coding: utf-8 -*-
"""Optional non-modal AppImage desktop integration UI for Linux."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)

from ..linux_appimage_integration import (
    AppImageContext,
    IntegrationState,
    IntegrationStatus,
    OperationResult,
    detect_appimage_context,
    inspect_integration_status,
    register_appimage,
    unregister_appimage,
)

if TYPE_CHECKING:
    from ..main_window import MainWindow


logger = logging.getLogger(__name__)

_PROMPT_DISABLED_KEY = "appimage_integration/prompt_disabled"
_DISMISSED_KEY = "appimage_integration/dismissed"
_OPTED_IN_KEY = "appimage_integration/user_opted_in"
_DISMISSED_UPDATE_KEY = "appimage_integration/dismissed_update_key"


class _IntegrationWorker(QThread):
    """Run AppImage registration or removal off the GUI thread."""

    def __init__(
        self,
        action: str,
        context: AppImageContext,
        data_home: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.action = action
        self._context = context
        self._data_home = data_home
        self.result: IntegrationStatus | OperationResult | None = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            if self.action == "inspect":
                self.result = inspect_integration_status(self._context, self._data_home)
            elif self.action == "register":
                self.result = register_appimage(self._context, self._data_home)
            else:
                self.result = unregister_appimage(
                    self._data_home, context=self._context
                )
        except (OSError, ValueError, RuntimeError) as exc:
            self.error = str(exc)


class AppImageIntegrationController(QObject):
    """Manage non-modal first-run prompt and repeatable Help menu actions."""

    idle = pyqtSignal()

    def __init__(
        self,
        main_window: "MainWindow",
        *,
        context: AppImageContext | None = None,
        data_home: Path | None = None,
    ) -> None:
        super().__init__(main_window)
        self.mw = main_window
        self._context = context
        self._started = False
        self._data_home = data_home
        self._status: IntegrationStatus | None = None
        self._closed = False
        self._worker: _IntegrationWorker | None = None
        self._tip_pending = False
        self._tip_mode = "register"
        self._tip: QFrame | None = None
        self._tip_label: QLabel | None = None
        self._tip_accept: QPushButton | None = None
        self._tip_skip: QPushButton | None = None
        self._tip_close: QToolButton | None = None
        self._message: QMessageBox | None = None

        self.menu_separator = QAction(self)
        self.menu_separator.setSeparator(True)
        self.integrate_action = QAction("Add TractEdit to 'Open With'...", self)
        self.integrate_action.triggered.connect(lambda: self._start_worker("register"))
        self.remove_action = QAction("Remove TractEdit from 'Open With'...", self)
        self.remove_action.triggered.connect(lambda: self._start_worker("unregister"))
        self.remove_action.setVisible(False)
        self.menu_separator.setVisible(False)
        self.integrate_action.setVisible(False)

    @property
    def active(self) -> bool:
        """Return True when running inside a valid Linux AppImage."""
        return self._context is not None

    @property
    def is_tip_visible(self) -> bool:
        """Return True while the non-modal integration banner is shown."""
        return bool(self._tip is not None and self._tip.isVisible())

    @property
    def working(self) -> bool:
        return self._worker is not None

    def start(self) -> None:
        """Inspect current integration state after the main window is shown."""
        if self._closed or self._started:
            return
        self._started = True
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            return
        if self._context is None:
            try:
                self._context = detect_appimage_context()
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("Could not identify AppImage: %s", exc)
            if self._context is None:
                return
        self.mw.installEventFilter(self)
        self._update_actions()
        if self.mw.settings.value(_PROMPT_DISABLED_KEY, False, type=bool):
            self.mw.help_menu.aboutToShow.connect(self.refresh_status)
            return
        self._tip_pending = True
        self._start_worker("inspect")

    def _prompt_for_status(self) -> None:
        self._tip_pending = False
        status = self._status
        if (
            status is None or status.system_deb_present
            or status.external_integrator_present
        ):
            return
        settings = self.mw.settings
        if status.state in (IntegrationState.UNREGISTERED, IntegrationState.PARTIAL):
            if settings.value(_DISMISSED_KEY, False, type=bool):
                return
            self._tip_mode = "register"
        elif status.state in (
            IntegrationState.REGISTERED_BROKEN, IntegrationState.REGISTERED_OUTDATED
        ):
            if not settings.value(_OPTED_IN_KEY, False, type=bool):
                return
            dismissed = settings.value(_DISMISSED_UPDATE_KEY, "", type=str)
            if dismissed == self._update_key(status):
                return
            self._tip_mode = "update"
        else:
            return
        self._tip_pending = True
        self._show_tip()

    def refresh_status(self) -> IntegrationStatus | None:
        """Read current user-scope integration state and update Help actions."""
        if self._context is None:
            return None
        try:
            self._status = inspect_integration_status(self._context, self._data_home)
        except (OSError, ValueError, RuntimeError) as exc:
            self._status = None
            logger.warning("Could not inspect AppImage integration: %s", exc)
        self._update_actions()
        return self._status

    def _update_key(self, status: IntegrationStatus) -> str:
        return f"{status.state.value}:{status.registered_exec}:{status.payload_sha256}"

    def _update_actions(self) -> None:
        if self._context is None:
            return
        status = self._status
        self.menu_separator.setVisible(True)
        self.integrate_action.setVisible(True)
        enabled = not self.working and not self._closed
        self.integrate_action.setEnabled(enabled)
        self.remove_action.setEnabled(enabled)
        action = self._worker.action if self._worker else None
        if action == "register":
            label = "Updating 'Open With'..."
        elif status is None or status.state == IntegrationState.UNREGISTERED:
            label = "Add TractEdit to 'Open With'..."
        elif status.state == IntegrationState.REGISTERED_CURRENT:
            label = "Refresh 'Open With' Integration..."
        else:
            label = "Update 'Open With' to This AppImage..."
        self.integrate_action.setText(label)
        self.remove_action.setText(
            "Removing 'Open With'..." if action == "unregister"
            else "Remove TractEdit from 'Open With'..."
        )
        self.remove_action.setVisible(status is not None and status.removable)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowActivate and self._tip_pending:
            QTimer.singleShot(0, self._show_tip)
        if event.type() == QEvent.Type.Resize and self._tip is not None:
            self._position_tip()
        return super().eventFilter(watched, event)

    def _tip_text(self) -> tuple[str, str]:
        status = self._status
        if self._tip_mode == "update":
            if (
                status is not None
                and status.state == IntegrationState.REGISTERED_BROKEN
            ):
                text = (
                    "Previous AppImage path is missing. "
                    "Update 'Open With' to this copy?"
                )
            else:
                text = (
                    "Updated file types are available. "
                    "Update 'Open With' for this AppImage?"
                )
            return text, "Update"

        text = "Open NIfTI, tracts and meshes with TractEdit?"
        if status is not None and status.is_ephemeral_location:
            text = "Add this temporary copy to 'Open With'?"
        return text, "Add to Open With"

    def _show_tip(self) -> None:
        if (
            self._closed
            or self.working
            or not self._tip_pending
            or not self.mw.isActiveWindow()
            or self.mw.isMinimized()
            or self.mw.update_notifier.is_tip_visible
        ):
            return
        if self._tip is None:
            self._tip = QFrame(self.mw)
            self._tip.setObjectName("appImageIntegrationTip")
            self._tip.setFrameShape(QFrame.Shape.StyledPanel)
            self._tip.setAutoFillBackground(True)
            self._tip.setFocusPolicy(Qt.FocusPolicy.NoFocus)

            outer = QVBoxLayout(self._tip)
            outer.setContentsMargins(10, 8, 8, 8)
            outer.setSpacing(6)

            top_row = QHBoxLayout()
            top_row.setSpacing(6)
            self._tip_label = QLabel(self._tip)
            self._tip_label.setWordWrap(True)
            top_row.addWidget(self._tip_label, 1)

            self._tip_close = QToolButton(self._tip)
            self._tip_close.setText("×")
            self._tip_close.setAutoRaise(True)
            self._tip_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._tip_close.setAccessibleName("Dismiss Open With prompt")
            self._tip_close.setToolTip("Dismiss")
            self._tip_close.clicked.connect(self._on_skip_clicked)
            top_row.addWidget(self._tip_close, 0, Qt.AlignmentFlag.AlignTop)
            outer.addLayout(top_row)

            buttons = QHBoxLayout()
            buttons.setSpacing(6)
            self._tip_accept = QPushButton(self._tip)
            self._tip_accept.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._tip_accept.clicked.connect(self.integrate_action.trigger)
            buttons.addWidget(self._tip_accept)

            self._tip_skip = QPushButton("Not now", self._tip)
            self._tip_skip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._tip_skip.clicked.connect(self._on_skip_clicked)
            buttons.addWidget(self._tip_skip)
            buttons.addStretch(1)
            outer.addLayout(buttons)

        text, accept_label = self._tip_text()
        self._tip_label.setText(text)
        self._tip_accept.setText(accept_label)
        self._position_tip()
        self._tip.show()
        self._tip.raise_()
        self._tip_pending = False

    def _position_tip(self) -> None:
        if self._tip is None:
            return
        # Keep width <= 290px anchored at left margin so it stays outside
        # the central VTK comparison region while remaining readable.
        self._tip.setFixedWidth(min(290, max(1, self.mw.width() - 16)))
        self._tip.adjustSize()
        menu_bar = self.mw.menuBar()
        anchor_y = menu_bar.mapTo(self.mw, QPoint(0, menu_bar.height())).y()
        x = max(8, min(8, self.mw.width() - self._tip.width() - 8))
        y = max(0, min(anchor_y + 6, self.mw.height() - self._tip.height()))
        self._tip.move(x, y)

    def _notify_update_tip_released(self) -> None:
        notifier = getattr(self.mw, "update_notifier", None)
        if notifier is not None and getattr(notifier, "_tip_pending", False):
            QTimer.singleShot(0, notifier._show_update_tip)

    def dismiss_tip(self) -> None:
        """Hide the banner without changing user preferences."""
        self._tip_pending = False
        if self._tip is not None and self._tip.isVisible():
            self._tip.hide()
            self._notify_update_tip_released()

    def _on_skip_clicked(self) -> None:
        if self._tip_mode == "update" and self._status is not None:
            self.mw.settings.setValue(
                _DISMISSED_UPDATE_KEY, self._update_key(self._status)
            )
        else:
            self.mw.settings.setValue(_DISMISSED_KEY, True)
        self.dismiss_tip()

    def _start_worker(self, action: str) -> None:
        if not self._started or self._closed or self.working or self._context is None:
            return
        if action != "inspect":
            self.dismiss_tip()
            self._set_status_message(
                "Updating 'Open With' integration..."
                if action == "register"
                else "Removing 'Open With' integration..."
            )
        worker = _IntegrationWorker(
            action, self._context, self._data_home, parent=self
        )
        self._worker = worker
        self._update_actions()
        worker.finished.connect(
            self._on_worker_finished, Qt.ConnectionType.QueuedConnection
        )
        worker.start()

    def _on_worker_completed(
        self, action: str, result: OperationResult | IntegrationStatus
    ) -> None:
        if isinstance(result, IntegrationStatus):
            if not self._closed:
                self._status = result
                self._update_actions()
                if self._tip_pending:
                    self._prompt_for_status()
            return
        self.mw.settings.setValue(_OPTED_IN_KEY, action == "register")
        self.mw.settings.setValue(_DISMISSED_KEY, True)
        if action == "register":
            self.mw.settings.setValue(_DISMISSED_UPDATE_KEY, "")
        self.mw.settings.sync()
        if self._closed:
            return
        if action == "register":
            self._status = result.status
            self._update_actions()
        else:
            self.refresh_status()
        self._set_status_message(result.message)
        for warning in result.warnings:
            logger.warning("AppImage integration warning: %s", warning)

    def _on_worker_failed(self, action: str, message: str) -> None:
        if self._closed:
            return
        if action == "inspect":
            self._tip_pending = False
            logger.warning("Could not inspect AppImage integration: %s", message)
            self._update_actions()
            return
        self.refresh_status()
        verb = "register" if action == "register" else "remove"
        self._set_status_message(f"Could not {verb} 'Open With' integration.")
        self._show_message(f"Could not {verb} TractEdit in 'Open With'.\n{message}")

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        try:
            if worker.result is not None:
                self._on_worker_completed(worker.action, worker.result)
            else:
                self._on_worker_failed(worker.action, worker.error or "Worker failed")
        finally:
            worker.deleteLater()
            if not self.working:
                self.idle.emit()

    def _set_status_message(self, text: str) -> None:
        vtk_panel = getattr(self.mw, "vtk_panel", None)
        if vtk_panel is not None and hasattr(vtk_panel, "update_status"):
            vtk_panel.update_status(text)
            return
        status_bar = self.mw.statusBar()
        if status_bar is not None:
            status_bar.showMessage(text, 5000)

    def _show_message(self, text: str) -> None:
        if self._message is None:
            self._message = QMessageBox(self.mw)
            self._message.setWindowTitle("Desktop Integration")
            self._message.setTextFormat(Qt.TextFormat.PlainText)
            self._message.setWindowModality(Qt.WindowModality.NonModal)
            self._message.addButton(QMessageBox.StandardButton.Close)
        self._message.setText(text)
        self._message.show()

    def close(self) -> None:
        """Retire the UI while queued worker results finish draining."""
        self._closed = True
        self._tip_pending = False
        if self._tip is not None:
            self._tip.hide()
        if self._message is not None:
            self._message.close()
