"""Optional, asynchronous notifications for published GitHub releases."""

import json
import re
import sys
import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import QDialogButtonBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QToolButton

from .. import __version__

if TYPE_CHECKING:
    from ..main_window import MainWindow


RELEASES_URL = "https://github.com/marcotag93/TractEdit/releases"
API_URL = "https://api.github.com/repos/marcotag93/TractEdit/releases/latest"
CHECK_INTERVAL = 24 * 60 * 60
MAX_RESPONSE_BYTES = 1024 * 1024


def _version_tuple(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(
        r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value
    ):
        raise ValueError("Unrecognized release version.")
    return tuple(int(part) for part in value.removeprefix("v").split("."))


def _release_tag(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Invalid release response.")
    if payload.get("draft") is not False or payload.get("prerelease") is not False:
        raise ValueError("No published stable release is available.")
    tag = payload.get("tag_name")
    _version_tuple(tag)
    suffixes = {
        "win32": (".zip",),
        "darwin": (".dmg",),
        "linux": (".deb", ".appimage"),
    }.get(sys.platform, ())
    assets = payload.get("assets")
    if not isinstance(assets, list) or not any(
        isinstance(asset, dict)
        and asset.get("state") == "uploaded"
        and isinstance(asset.get("size"), int)
        and asset["size"] > 0
        and isinstance(asset.get("name"), str)
        and asset["name"].lower().endswith(suffixes)
        for asset in assets
    ):
        raise ValueError("No release package is available for this platform yet.")
    return tag


class UpdateNotifier(QObject):
    """Own update requests and their UI independently of scientific operations."""

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self.mw = main_window
        self._network = None
        self._reply = None
        self._manual = False
        self._closed = False
        self._body = bytearray()
        self._failure = ""
        self._message = None
        self._release_button = None
        self._ignore_button = None
        self._dialog_dismissed = False
        self._notified_tag = ""
        self._tip_pending = False
        self._tip = None
        self._tip_label = None
        self._tip_close = None
        self._tag = ""
        self.mw.installEventFilter(self)

        self.check_action = QAction("Check for Updates...", self)
        self.check_action.triggered.connect(lambda: self.check(manual=True))
        self.release_action = QAction("Install TractEdit...", self)
        self.release_action.setVisible(False)
        self.release_action.triggered.connect(self._open_release)
        self.auto_action = QAction("Automatically Check for Updates", self)
        self.auto_action.setCheckable(True)
        self.auto_action.setChecked(
            self.mw.settings.value("updates/enabled", True, type=bool)
        )
        self.auto_action.setStatusTip(
            "Contact GitHub at most once a day; no research data is sent"
        )
        self.auto_action.toggled.connect(self._set_enabled)

        self._schedule = QTimer(self)
        self._schedule.setSingleShot(True)
        self._schedule.setTimerType(Qt.TimerType.PreciseTimer)
        self._schedule.timeout.connect(self._automatic_check)
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._timed_out)

    def start(self) -> None:
        if self.auto_action.isChecked() and not self._closed:
            self._schedule.start(5000)

    def _set_enabled(self, enabled: bool) -> None:
        self.mw.settings.setValue("updates/enabled", enabled)
        if enabled:
            self.start()
        else:
            self._schedule.stop()
            if self._reply is not None and not self._manual:
                self._reply.abort()

    def _automatic_check(self) -> None:
        if not self._closed and self.auto_action.isChecked():
            self.check()
            self._schedule.start(CHECK_INTERVAL * 1000)

    def check(self, manual: bool = False) -> None:
        if self._closed or self._reply is not None:
            return
        if not manual:
            if not self.auto_action.isChecked():
                return
            try:
                last_check = float(self.mw.settings.value("updates/last_check", 0))
            except (TypeError, ValueError):
                last_check = 0
            elapsed = time.time() - last_check
            if 0 <= elapsed < CHECK_INTERVAL:
                return

        self._manual = manual
        self._dialog_dismissed = False
        self._failure = ""
        self._body.clear()
        self.mw.settings.setValue("updates/last_check", time.time())
        self.check_action.setEnabled(False)
        self.check_action.setText("Checking for Updates...")
        if self._network is None:
            self._network = QNetworkAccessManager(self)
        request = QNetworkRequest(QUrl(API_URL))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"User-Agent", f"TractEdit/{__version__}".encode())
        request.setAttribute(
            QNetworkRequest.Attribute.RedirectPolicyAttribute,
            QNetworkRequest.RedirectPolicy.SameOriginRedirectPolicy,
        )
        self._reply = self._network.get(request)
        self._reply.setReadBufferSize(MAX_RESPONSE_BYTES + 1)
        self._reply.readyRead.connect(self._read_response)
        self._reply.finished.connect(self._finished)
        self._timeout.start(8000)

    def _read_response(self) -> None:
        if self._reply is None or self._failure:
            return
        remaining = MAX_RESPONSE_BYTES - len(self._body) + 1
        chunk = self._reply.read(remaining)
        if chunk:
            self._body.extend(bytes(chunk))
        if len(self._body) > MAX_RESPONSE_BYTES:
            self._failure = "The release response is too large."
            if not self._reply.isFinished():
                self._reply.abort()

    def _timed_out(self) -> None:
        if self._reply is not None:
            self._failure = "The connection timed out."
            self._reply.abort()

    def _finished(self) -> None:
        self._timeout.stop()
        reply = self._reply
        if reply is None:
            return
        if not self._failure and reply.error() == QNetworkReply.NetworkError.NoError:
            self._read_response()
        self._reply = None
        self.check_action.setEnabled(not self._closed)
        self.check_action.setText("Check for Updates...")
        try:
            if self._closed or (not self._manual and not self.auto_action.isChecked()):
                return
            status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if self._failure:
                raise ValueError(self._failure)
            if reply.error() != QNetworkReply.NetworkError.NoError or status != 200:
                raise ValueError("GitHub could not be reached or rejected the request.")
            tag = _release_tag(json.loads(self._body))
            if _version_tuple(tag) <= _version_tuple(__version__):
                if self._manual:
                    self._show_message(
                        "No newer stable release is available.\n"
                        f"Installed version: {__version__}"
                    )
                return
            self._show_update(tag)
        except (ValueError, UnicodeError, RecursionError) as exc:
            if self._manual:
                self._show_message(f"Unable to check for updates.\n{exc}")
        finally:
            self._body.clear()
            reply.deleteLater()

    def _show_update(self, tag: str) -> None:
        self._tag = tag
        description = f"TractEdit {tag.removeprefix('v')} available"
        self.release_action.setText(f"Install TractEdit v{tag.removeprefix('v')}...")
        self.release_action.setStatusTip(description)
        self.release_action.setVisible(True)
        if self._manual:
            self._show_message(description, show_release_actions=True)
        else:
            ignored = self.mw.settings.value("updates/ignored_version", "")
            if ignored != tag.removeprefix("v") and self._notified_tag != tag:
                self._tip_pending = True
                self._show_update_tip()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowActivate and self._tip_pending:
            QTimer.singleShot(0, self._show_update_tip)
        if event.type() == QEvent.Type.Resize and self._tip is not None:
            self._position_tip()
        return super().eventFilter(watched, event)

    def _show_update_tip(self) -> None:
        integration = getattr(self.mw, "appimage_integration", None)
        if (
            self._closed or not self._tip_pending
            or not self.mw.isActiveWindow() or self.mw.isMinimized()
            or (integration is not None and integration.is_tip_visible)
        ):
            return
        if self._tip is None:
            self._tip = QFrame(self.mw)
            self._tip.setObjectName("updateTip")
            self._tip.setFrameShape(QFrame.Shape.StyledPanel)
            self._tip.setAutoFillBackground(True)
            layout = QHBoxLayout(self._tip)
            layout.setContentsMargins(10, 8, 6, 8)
            layout.setSpacing(8)
            self._tip_label = QLabel(self._tip)
            self._tip_label.setWordWrap(True)
            layout.addWidget(self._tip_label, 1)
            self._tip_close = QToolButton(self._tip)
            self._tip_close.setText("×")
            self._tip_close.setAutoRaise(True)
            self._tip_close.setAccessibleName("Dismiss update notification")
            self._tip_close.setToolTip("Dismiss update notification")
            self._tip_close.clicked.connect(self.dismiss_tip)
            layout.addWidget(self._tip_close, 0, Qt.AlignmentFlag.AlignTop)
        self._tip_label.setText(
            f"TractEdit {self._tag.removeprefix('v')} available.\n"
            f"See Help → {self.release_action.text()}"
        )
        self._position_tip()
        self._tip.show()
        self._tip.raise_()
        self._notified_tag = self._tag
        self._tip_pending = False

    @property
    def is_tip_visible(self) -> bool:
        return self._tip is not None and self._tip.isVisible()

    def dismiss_tip(self) -> None:
        self._tip_pending = False
        if self._tag:
            self._notified_tag = self._tag
        if self.is_tip_visible:
            self._tip.hide()
            integration = getattr(self.mw, "appimage_integration", None)
            if integration is not None and integration._tip_pending:
                QTimer.singleShot(0, integration._show_tip)

    def _position_tip(self) -> None:
        self._tip.setFixedWidth(min(320, max(1, self.mw.width() - 16)))
        self._tip.adjustSize()
        menu_bar = self.mw.menuBar()
        help_menu = getattr(self.mw, "help_menu", None)
        position = (
            menu_bar.actionGeometry(help_menu.menuAction()).bottomLeft()
            if help_menu is not None else QPoint(0, menu_bar.height())
        )
        anchor = menu_bar.mapTo(self.mw, position)
        self._tip.move(
            max(8, min(anchor.x(), self.mw.width() - self._tip.width() - 8)),
            max(0, min(anchor.y() + 6, self.mw.height() - self._tip.height())),
        )

    def _open_release(self) -> None:
        if not QDesktopServices.openUrl(QUrl(f"{RELEASES_URL}/tag/{self._tag}")):
            self._dialog_dismissed = False
            self._show_message(f"Could not open the browser.\nVisit {RELEASES_URL}")

    def _ignore_version(self) -> None:
        self.mw.settings.setValue(
            "updates/ignored_version", self._tag.removeprefix("v")
        )
        self._tip_pending = False
        if self._message is not None:
            self._message.close()

    def _show_message(self, text: str, *, show_release_actions: bool = False) -> None:
        if self._message is None:
            self._message = QMessageBox(self.mw)
            self._message.setWindowTitle("Check for Updates")
            self._message.setTextFormat(Qt.TextFormat.PlainText)
            self._message.setWindowModality(Qt.WindowModality.NonModal)
            self._release_button = self._message.addButton(
                self.release_action.text(), QMessageBox.ButtonRole.ActionRole
            )
            self._release_button.clicked.connect(self._open_release)
            self._ignore_button = self._message.addButton(
                "Ignore this version", QMessageBox.ButtonRole.ActionRole
            )
            self._ignore_button.clicked.connect(self._ignore_version)
            self._message.addButton(QMessageBox.StandardButton.Close)
            self._message.findChild(QDialogButtonBox).setOrientation(Qt.Orientation.Vertical)
            self._message.finished.connect(self._dismiss_dialog)
        self._message.setText(text)
        self._release_button.setText(self.release_action.text())
        self._release_button.setVisible(show_release_actions)
        self._ignore_button.setVisible(show_release_actions)
        if not self._dialog_dismissed:
            self._message.show()

    def _dismiss_dialog(self) -> None:
        self._dialog_dismissed = True

    def close(self) -> None:
        self._closed = True
        self._schedule.stop()
        self._timeout.stop()
        if self._reply is not None:
            self._reply.abort()
        self._tip_pending = False
        if self._tip is not None:
            self._tip.hide()
        if self._message is not None:
            self._message.close()
