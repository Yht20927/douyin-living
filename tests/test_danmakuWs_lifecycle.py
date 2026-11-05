# -*- coding: utf-8 -*-
"""DanmakuWs lifecycle tests — reconnect, heartbeat cleanup and stats."""

import threading
import time
from unittest.mock import MagicMock

import pytest

import src.danmakuWs as danmakuWs
from src.danmakuWs import DanmakuWs

HEARTBEAT_THREAD_NAME = "danmaku-ws-heartbeat"


def _makeWs(**kwargs) -> DanmakuWs:
    return DanmakuWs(
        params={"test": "1"},
        cookieStr="test=1",
        onDanmaku=None,
        **kwargs,
    )


def _aliveHeartbeats() -> list[threading.Thread]:
    return [
        t
        for t in threading.enumerate()
        if t.name == HEARTBEAT_THREAD_NAME and t.is_alive()
    ]


class _FakeWsApp:
    """Drop-in stand-in for ``websocket.WebSocketApp``.

    Each ``run_forever`` call opens, briefly lives and closes — enough to
    exercise the reconnect loop without touching the network.
    """

    instances: list["_FakeWsApp"] = []
    reached: threading.Event | None = None
    threshold = 0

    def __init__(
        self,
        url,
        header=None,
        cookie=None,
        on_open=None,
        on_message=None,
        on_error=None,
        on_close=None,
    ):
        self.url = url
        self.header = header
        self.cookie = cookie
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.closed = False
        self.sent: list[tuple[bytes, int | None]] = []

    def run_forever(self, origin=None):
        type(self).instances.append(self)
        if self.on_open:
            self.on_open(self)
        time.sleep(0.005)
        if self.on_close:
            self.on_close(self, 1000, "test-close")
        reached = type(self).reached
        if reached is not None and len(type(self).instances) >= type(self).threshold:
            reached.set()

    def close(self):
        self.closed = True

    def send(self, payload, opcode=None):
        self.sent.append((payload, opcode))


class _FakeSocket:
    def __init__(self, failSend: bool = False):
        self.failSend = failSend
        self.closed = False
        self.sent: list[tuple[bytes, int | None]] = []

    def send(self, payload, opcode=None):
        if self.failSend:
            raise RuntimeError("send failed")
        self.sent.append((payload, opcode))

    def close(self):
        self.closed = True


@pytest.fixture
def fakeWs(monkeypatch):
    _FakeWsApp.instances = []
    _FakeWsApp.reached = threading.Event()
    _FakeWsApp.threshold = 4
    monkeypatch.setattr(danmakuWs, "WebSocketApp", _FakeWsApp)
    return _FakeWsApp


class TestReconnectLifecycle:
    def test_heartbeat_threads_do_not_leak_after_reconnects(self, fakeWs):
        ws = _makeWs(pingInterval=0.001, reconnectDelay=0.001)
        ws.start()
        try:
            assert fakeWs.reached.wait(5.0), "reconnect loop did not run"
            stats = ws.stats
            assert stats["connectAttempts"] >= 4
            assert stats["reconnects"] >= 3
        finally:
            ws.stop()
            ws.join(timeout=5.0)

        assert ws._thread is not None and not ws._thread.is_alive()
        assert _aliveHeartbeats() == []
        assert ws.stats["connected"] is False
        assert ws.stats["running"] is False

    def test_stop_interrupts_long_backoff_wait(self, monkeypatch):
        class NoConnectWs:
            def __init__(self, *args, **kwargs):
                pass

            def run_forever(self, origin=None):
                return

            def close(self):
                pass

        monkeypatch.setattr(danmakuWs, "WebSocketApp", NoConnectWs)
        ws = _makeWs(reconnectDelay=30.0)
        ws.start()

        deadline = time.monotonic() + 2.0
        while ws.stats["connectAttempts"] < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert ws.stats["connectAttempts"] >= 1

        started = time.monotonic()
        ws.stop()
        ws.join(timeout=2.0)
        elapsed = time.monotonic() - started

        assert ws._thread is not None and not ws._thread.is_alive()
        assert elapsed < 1.0, f"stop took {elapsed:.2f}s, backoff was not interruptible"

    def test_backoff_delay_is_exponential_and_capped(self):
        ws = _makeWs(reconnectDelay=1.0)

        ws._consecutiveErrors = 0
        assert ws._backoffDelay() == 1.0
        ws._consecutiveErrors = 1
        assert ws._backoffDelay() == 1.0
        ws._consecutiveErrors = 3
        assert ws._backoffDelay() == 4.0
        ws._consecutiveErrors = 99
        assert ws._backoffDelay() == float(danmakuWs._cfg.backoff_max)


class TestConnectionCallbacks:
    def test_on_open_starts_and_on_close_stops_heartbeat(self):
        ws = _makeWs(pingInterval=0.001)
        ws._running = True
        sock = _FakeSocket()

        ws._onOpen(sock)
        assert ws.stats["connected"] is True
        assert ws.stats["connectedAt"] is not None
        assert len(_aliveHeartbeats()) == 1

        ws._onClose(sock, 1000, "bye")
        assert ws.stats["connected"] is False
        assert _aliveHeartbeats() == []

    def test_on_open_after_stop_closes_socket(self):
        ws = _makeWs()
        ws._running = False
        sock = _FakeSocket()

        ws._onOpen(sock)

        assert sock.closed is True
        assert ws.stats["connected"] is False
        assert _aliveHeartbeats() == []

    def test_ping_loop_exits_when_send_fails(self):
        ws = _makeWs(pingInterval=0.001)
        ws._running = True
        sock = _FakeSocket(failSend=True)

        ws._onOpen(sock)
        ws._hbThread.join(timeout=2.0)

        assert not ws._hbThread.is_alive()
        assert _aliveHeartbeats() == []

    def test_ping_loop_closes_stale_connection(self):
        ws = _makeWs(pingInterval=0.001)
        ws._running = True
        ws._pingTimeout = 0.02
        sock = _FakeSocket()

        ws._onOpen(sock)
        ws._hbThread.join(timeout=2.0)

        assert not ws._hbThread.is_alive()
        assert sock.closed is True
        assert "heartbeat timeout" in ws.stats["lastError"]
        assert ws.stats["consecutiveErrors"] == 1

    def test_on_error_records_metrics(self):
        ws = _makeWs()

        ws._onError(None, RuntimeError("boom"))

        stats = ws.stats
        assert stats["consecutiveErrors"] == 1
        assert "RuntimeError" in stats["lastError"]
        assert "boom" in stats["lastError"]
        assert stats["lastErrorAt"] is not None

    def test_on_error_accepts_plain_strings(self):
        ws = _makeWs()

        ws._onError(None, "socket reset")

        assert ws.stats["lastError"] == "socket reset"
        assert ws.stats["consecutiveErrors"] == 1

    def test_on_open_resets_error_counter(self):
        ws = _makeWs(pingInterval=0.001)
        ws._running = True
        ws._onError(None, "first failure")
        assert ws.stats["consecutiveErrors"] == 1

        ws._onOpen(_FakeSocket())
        try:
            assert ws.stats["consecutiveErrors"] == 0
        finally:
            ws._onClose(None, 1000, "done")

    def test_on_message_updates_last_activity(self):
        ws = _makeWs()
        assert ws.stats["lastMessageAt"] is None

        ws._onMessage(None, "text frame")

        assert ws.stats["lastMessageAt"] is not None


class TestUnknownMethodMetrics:
    def test_unknown_methods_are_counted(self):
        ws = _makeWs()
        msg = MagicMock()
        msg.method = "WebcastMysteryMessage"
        msg.payload = b""

        result = ws._parseMessage(msg)
        assert result is not None
        assert result["type"] == "unknown"
        assert ws.unknownMethods == {"WebcastMysteryMessage": 1}

        ws._parseMessage(msg)
        assert ws.stats["unknownMethods"] == {"WebcastMysteryMessage": 2}

    def test_stats_contains_lifecycle_fields(self):
        stats = _makeWs().stats
        for key in (
            "running",
            "connected",
            "messages",
            "connectAttempts",
            "reconnects",
            "consecutiveErrors",
            "lastError",
            "lastErrorAt",
            "connectedAt",
            "lastMessageAt",
            "unknownMethods",
        ):
            assert key in stats
