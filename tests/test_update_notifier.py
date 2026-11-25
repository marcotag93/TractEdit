"""Release notifications without network access or scientific GUI startup."""

import json
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QEventLoop, QObject, QSettings, QTimer, QUrl, pyqtSignal
from PyQt6.QtNetwork import (
    QHostAddress,
    QNetworkAccessManager,
    QNetworkReply,
    QNetworkRequest,
    QTcpServer,
)
from PyQt6.QtWidgets import QApplication, QMainWindow

from tractedit_pkg.ui import update_notifier as updates


class Reply(QObject):
    readyRead = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = b""
        self.status = 200
        self.aborted = False
        self.done = False

    def setReadBufferSize(self, size):
        self.buffer_size = size

    def read(self, size):
        if not self.data:
            return None
        result, self.data = self.data[:size], self.data[size:]
        return result

    def error(self):
        if self.aborted:
            return QNetworkReply.NetworkError.OperationCanceledError
        return QNetworkReply.NetworkError.NoError

    def attribute(self, key):
        return self.status

    def isFinished(self):
        return self.done

    def abort(self):
        self.aborted = True
        self.complete()

    def complete(self, payload=None):
        if payload is not None:
            self.data = json.dumps(payload).encode()
        self.done = True
        self.finished.emit()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def notifier(app, tmp_path, monkeypatch):
    monkeypatch.setattr(updates.sys, "platform", "win32")
    monkeypatch.setattr(updates, "__version__", "3.9.0")
    monkeypatch.setattr(updates.time, "time", lambda: 1_000_000)
    window = QMainWindow()
    window.settings = QSettings(
        str(tmp_path / "updates.ini"), QSettings.Format.IniFormat
    )
    reply = Reply()
    requests = []

    def get(request):
        requests.append(request)
        return reply

    monkeypatch.setattr(
        updates, "QNetworkAccessManager", lambda parent: SimpleNamespace(get=get)
    )
    controller = updates.UpdateNotifier(window)
    messages = []
    monkeypatch.setattr(controller, "_show_message", lambda text, **kwargs: messages.append(text))
    yield controller, reply, requests, messages
    controller.close()
    window.close()
    window.deleteLater()
    app.processEvents()


def release(tag="v3.10.0"):
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": False,
        "assets": [
            {"name": "TractEdit-windows-x64.zip", "state": "uploaded", "size": 10}
        ],
    }


def test_real_qt_reply_can_be_read_again_at_end_of_response(notifier):
    controller, fake_reply, requests, messages = notifier
    server = QTcpServer(controller)
    assert server.listen(QHostAddress.SpecialAddress.LocalHost, 0)

    def accept_connection():
        connection = server.nextPendingConnection()

        def respond():
            connection.readAll()
            connection.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                b"Connection: close\r\n\r\n{}"
            )
            connection.disconnectFromHost()

        connection.readyRead.connect(respond)

    server.newConnection.connect(accept_connection)
    network = QNetworkAccessManager(controller)
    url = QUrl(f"http://127.0.0.1:{server.serverPort()}/release")
    reply = network.get(QNetworkRequest(url))
    controller._reply = reply
    errors = []

    def read_available():
        try:
            controller._read_response()
        except Exception as exc:
            errors.append(exc)

    reply.readyRead.connect(read_available)
    loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    reply.finished.connect(loop.quit)
    timeout.start(2000)
    if not reply.isFinished():
        loop.exec()
    timeout.stop()
    try:
        assert reply.isFinished()
        assert reply.error() == QNetworkReply.NetworkError.NoError
        assert not errors
        assert controller._body == b"{}"
        controller._read_response()
        assert controller._body == b"{}"
    finally:
        controller._reply = None
        reply.abort()
        reply.deleteLater()
        server.close()


def test_finished_after_response_was_already_consumed(notifier):
    controller, reply, requests, messages = notifier
    controller.check(manual=True)
    reply.data = json.dumps(release("v3.9.0")).encode()
    reply.readyRead.emit()
    reply.done = True
    controller._finished()
    assert "No newer stable release" in messages[-1]
    assert controller._reply is None
    assert controller.check_action.isEnabled()


def test_checks_default_on_and_daily_but_manual_check_bypasses(notifier):
    controller, reply, requests, messages = notifier
    assert controller.auto_action.isChecked()
    controller.auto_action.setChecked(False)
    controller.start()
    controller.check()
    assert not controller._schedule.isActive()
    assert requests == []
    controller.auto_action.setChecked(True)
    controller._automatic_check()
    assert len(requests) == 1
    assert requests[0].url().toString() == updates.API_URL
    assert not controller.check_action.isEnabled()
    controller.check(manual=True)
    assert len(requests) == 1
    reply.complete(release("v3.8.0"))
    controller.check()
    assert len(requests) == 1
    assert messages == []
    controller.auto_action.setChecked(False)
    controller.check_action.trigger()
    assert len(requests) == 2
    reply.complete(release("v3.9.0"))
    assert "No newer stable release" in messages[-1]


@pytest.mark.parametrize("enabled", [False, True])
def test_automatic_check_preference_survives_reopening(notifier, enabled):
    controller, reply, requests, messages = notifier
    controller.auto_action.setChecked(False)
    controller.auto_action.setChecked(enabled)
    controller.close()
    controller.mw.settings.sync()
    window = QMainWindow()
    window.settings = QSettings(
        controller.mw.settings.fileName(), QSettings.Format.IniFormat
    )
    reopened = updates.UpdateNotifier(window)
    try:
        assert reopened.auto_action.isChecked() is enabled
        reopened.start()
        assert reopened._schedule.isActive() is enabled
        assert requests == []
    finally:
        reopened.close()
        window.close()
        window.deleteLater()


def test_new_release_ignore_and_manual_override(notifier, monkeypatch):
    controller, reply, requests, messages = notifier
    controller.check(manual=True)
    reply.complete(release())
    assert controller.release_action.isVisible()
    assert "3.10.0" in messages[-1]
    urls = []
    monkeypatch.setattr(
        updates.QDesktopServices,
        "openUrl",
        lambda url: urls.append(url.toString()) or True,
    )
    controller._open_release()
    assert urls == [f"{updates.RELEASES_URL}/tag/v3.10.0"]
    controller._ignore_version()
    assert controller.release_action.isVisible()
    assert controller.mw.settings.value("updates/ignored_version") == "3.10.0"
    controller.auto_action.setChecked(True)
    controller.mw.settings.setValue("updates/last_check", 0)
    controller.check()
    reply.complete(release())
    assert controller.release_action.isVisible()
    controller.check(manual=True)
    reply.complete(release())
    assert controller.release_action.isVisible()
    assert "3.10.0" in messages[-1]


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("failure", ["http", "timeout", "malformed", "oversized"])
def test_failures_are_silent_unless_requested(notifier, manual, failure):
    controller, reply, requests, messages = notifier
    controller.auto_action.setChecked(True)
    controller.check(manual=manual)
    if failure == "http":
        reply.status = 403
        reply.complete()
    elif failure == "timeout":
        controller._timeout.timeout.emit()
    elif failure == "malformed":
        reply.data = b"not JSON"
        reply.complete()
    else:
        reply.data = b"x" * (updates.MAX_RESPONSE_BYTES + 1)
        reply.readyRead.emit()
        assert reply.aborted
    assert controller._reply is None
    assert controller.check_action.isEnabled()
    assert not controller._timeout.isActive()
    assert bool(messages) == manual
    if manual:
        assert "Unable to check" in messages[-1]


@pytest.mark.parametrize("operation", ["disable", "close"])
def test_cancellation_has_no_late_notification(notifier, operation):
    controller, reply, requests, messages = notifier
    controller.auto_action.setChecked(True)
    controller.check()
    if operation == "disable":
        controller.auto_action.setChecked(False)
    else:
        controller.close()
    assert reply.aborted
    assert not controller._schedule.isActive()
    assert controller._reply is None
    reply.finished.emit()
    assert messages == []
    assert not controller.release_action.isVisible()


@pytest.mark.parametrize(
    "change",
    [
        {"tag_name": "v3.10.0rc1"},
        {"tag_name": "https://example.com"},
        {"tag_name": None},
        {"draft": True},
        {"prerelease": True},
        {"assets": []},
        {"assets": [{"name": "source.zip", "state": "new", "size": 10}]},
    ],
)
def test_invalid_or_unpublished_releases_are_rejected(change, monkeypatch):
    monkeypatch.setattr(updates.sys, "platform", "win32")
    payload = release()
    payload.update(change)
    with pytest.raises(ValueError):
        updates._release_tag(payload)
