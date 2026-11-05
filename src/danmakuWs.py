# -*- coding: utf-8 -*-
"""Live danmaku WebSocket client.

Connects to ``wss://webcast100-ws-web-hl.douyin.com/webcast/im/push/v2/``,
receives Protobuf PushFrame messages, decompresses with gzip, and parses
individual message types (chat, gift, like, member, social, stats).
"""

import gzip
import threading
import time
from collections import Counter
from urllib.parse import urlencode
from typing import Callable, Any

from websocket import WebSocketApp

import src.protobuf.Live_pb2 as LivePb
from src.log.logger import getLogger
from src.config import load_settings

log = getLogger(__name__)
_cfg = load_settings().recording


class DanmakuWs:
    """Douyin live danmaku WebSocket client.

    Usage::

        ws = DanmakuWs(params, cookieStr, onDanmaku=myCallback)
        ws.start()        # blocks in background thread
        ws.stop()
    """

    def __init__(
        self,
        params: dict[str, str],
        cookieStr: str,
        onDanmaku: Callable[[dict[str, Any]], None] | None = None,
        pingInterval: float | None = None,
        reconnectDelay: float | None = None,
    ):
        self._params = params
        self._cookieStr = cookieStr
        self._onDanmaku = onDanmaku
        self._pingInterval = pingInterval if pingInterval is not None else _cfg.ws_ping_interval
        self._reconnectDelay = reconnectDelay if reconnectDelay is not None else _cfg.ws_reconnect_delay
        self._pingTimeout = _cfg.ws_ping_timeout

        self._ws: WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False
        self._msgCount = 0
        self._consecutiveErrors = 0
        self._stopEvent = threading.Event()
        self._hbStop = threading.Event()
        self._hbThread: threading.Thread | None = None
        self._reconnectCount = 0
        self._connectAttempts = 0
        self._lastError: str | None = None
        self._lastErrorAt: float | None = None
        self._connectedAt: float | None = None
        self._lastMessageAt: float | None = None
        self._unknownMethods: Counter[str] = Counter()

    # ── public API ────────────────────────────────────────────────

    @property
    def url(self) -> str:
        qs = urlencode(self._params)
        return f"wss://webcast100-ws-web-hl.douyin.com/webcast/im/push/v2/?{qs}"

    def start(self):
        """Start the WebSocket connection in a background thread."""
        if self._running:
            return
        self._running = True
        self._connected = False
        self._stopEvent.clear()
        self._hbStop.clear()
        self._thread = threading.Thread(
            target=self._runForever, name="danmaku-ws", daemon=True
        )
        self._thread.start()
        log.info("Danmaku WS starting: %s", self.url[:100])

    def stop(self):
        """Signal the WebSocket to close and wake any backoff wait."""
        self._running = False
        self._stopEvent.set()
        self._hbStop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def join(self, timeout: float | None = None):
        """Wait for the WS thread to finish."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def msgCount(self) -> int:
        return self._msgCount

    @property
    def unknownMethods(self) -> dict[str, int]:
        return dict(self._unknownMethods)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "connected": self._connected,
            "messages": self._msgCount,
            "connectAttempts": self._connectAttempts,
            "reconnects": self._reconnectCount,
            "consecutiveErrors": self._consecutiveErrors,
            "lastError": self._lastError,
            "lastErrorAt": self._lastErrorAt,
            "connectedAt": self._connectedAt,
            "lastMessageAt": self._lastMessageAt,
            "unknownMethods": dict(self._unknownMethods),
        }

    # ── internal ──────────────────────────────────────────────────

    def _recordError(self, error: BaseException | str, increment: bool = True) -> None:
        """Record an error for observability and backoff calculation."""
        if increment:
            self._consecutiveErrors += 1
        if isinstance(error, BaseException):
            self._lastError = f"{type(error).__name__}: {error}"
        else:
            self._lastError = str(error)
        self._lastErrorAt = time.time()

    def _backoffDelay(self) -> float:
        """Exponential backoff capped at ``recording.backoff_max``."""
        exponent = max(self._consecutiveErrors - 1, 0)
        return min(self._reconnectDelay * (2 ** exponent), float(_cfg.backoff_max))

    def _joinHeartbeat(self, timeout: float = 1.0) -> None:
        """Wait for the heartbeat thread to exit, never joining ourselves."""
        thread = self._hbThread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

    def _stopHeartbeat(self) -> None:
        self._hbStop.set()
        self._joinHeartbeat()

    def _pingLoop(self, ws, stopEvent: threading.Event) -> None:
        """Send heartbeat frames until the connection or client stops.

        If no frame has arrived for ``ws_ping_timeout`` seconds the socket is
        closed so ``_runForever`` can reconnect; this turns a silently dead
        connection into a measurable failure instead of an idle thread.
        """
        while not stopEvent.is_set() and not self._stopEvent.is_set():
            try:
                frame = LivePb.PushFrame()
                frame.payloadType = "hb"
                ws.send(frame.SerializeToString(), opcode=0x02)
            except Exception as exc:
                log.debug(f"Heartbeat send failed: {exc}")
                return
            if stopEvent.wait(self._pingInterval):
                return

            lastActivity = self._lastMessageAt or self._connectedAt or time.time()
            if time.time() - lastActivity > self._pingTimeout:
                self._recordError(f"heartbeat timeout after {self._pingTimeout:.1f}s")
                log.warning(
                    "Danmaku WS heartbeat timeout (%.1fs without data), closing",
                    self._pingTimeout,
                )
                try:
                    ws.close()
                except Exception:
                    pass
                return

    def _runForever(self):
        """Maintain one or more WS sessions until ``stop()`` is called.

        A fresh ``WebSocketApp`` is created per attempt.  Reconnects use
        exponential backoff that is interruptible by ``stop()`` instead of
        a blocking ``time.sleep``.
        """
        while self._running and not self._stopEvent.is_set():
            self._connectAttempts += 1
            try:
                self._ws = WebSocketApp(
                    self.url,
                    header={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/125.0.0.0 Safari/537.36"
                        ),
                        "Connection": "Upgrade",
                        "Upgrade": "websocket",
                    },
                    cookie=self._cookieStr,
                    on_open=self._onOpen,
                    on_message=self._onMessage,
                    on_error=self._onError,
                    on_close=self._onClose,
                )
                self._ws.run_forever(origin="https://live.douyin.com")
            except Exception as exc:
                self._recordError(exc)
                log.warning(f"WS loop error: {exc}")
            finally:
                self._stopHeartbeat()

            if not self._running or self._stopEvent.is_set():
                break

            self._reconnectCount += 1
            delay = self._backoffDelay()
            log.info(
                "WS reconnecting in %.1fs "
                "(attempts=%d, reconnects=%d, consecutiveErrors=%d)",
                delay,
                self._connectAttempts,
                self._reconnectCount,
                self._consecutiveErrors,
            )
            if self._stopEvent.wait(delay):
                break

        self._connected = False
        self._stopHeartbeat()
        self._stopEvent.set()
        log.info(
            "Danmaku WS stopped (messages=%d, attempts=%d, reconnects=%d, "
            "unknownMessages=%d)",
            self._msgCount,
            self._connectAttempts,
            self._reconnectCount,
            sum(self._unknownMethods.values()),
        )

    def _onOpen(self, ws):
        if not self._running or self._stopEvent.is_set():
            try:
                ws.close()
            except Exception:
                pass
            return

        self._connected = True
        self._connectedAt = time.time()
        self._consecutiveErrors = 0
        log.info("Danmaku WS connected ✓")

        # Each connection owns its heartbeat; replacing the event guarantees
        # a stop() racing with this callback cannot revive an old thread.
        self._stopHeartbeat()
        self._hbStop = threading.Event()
        self._hbThread = threading.Thread(
            target=self._pingLoop,
            args=(ws, self._hbStop),
            name="danmaku-ws-heartbeat",
            daemon=True,
        )
        self._hbThread.start()

    def _onMessage(self, ws, message):
        self._lastMessageAt = time.time()
        # Only process binary frames; text frames are ignored.
        if isinstance(message, str):
            return

        try:
            frame = LivePb.PushFrame()
            frame.ParseFromString(message)

            if frame.payloadType == "hb":
                return  # heartbeat response
            if frame.payloadType == "ack":
                return  # ack response

            if not frame.payload:
                return

            # Decompress gzip payload
            try:
                originBytes = gzip.decompress(frame.payload)
            except Exception:
                originBytes = frame.payload

            response = LivePb.LiveResponse()
            response.ParseFromString(originBytes)

            # Send ack if needed
            if response.needAck and response.internalExt:
                ack = LivePb.PushFrame()
                ack.payloadType = "ack"
                ack.payload = response.internalExt.encode("utf-8")
                ack.logId = frame.logId
                try:
                    ws.send(ack.SerializeToString(), opcode=0x02)
                except Exception:
                    pass

            # Process messages
            for msg in response.messagesList:
                self._msgCount += 1
                parsed = self._parseMessage(msg)
                if parsed and self._onDanmaku:
                    self._onDanmaku(parsed)

        except Exception:
            log.debug("Frame parse error", exc_info=True)

    def _onError(self, ws, error):
        self._recordError(error)
        log.warning(f"WS error: {error}")

    def _onClose(self, ws, code, msg):
        self._connected = False
        self._stopHeartbeat()
        log.info(f"WS closed: code={code} msg={msg}")

    # ── message parsing ───────────────────────────────────────────

    def _parseMessage(self, msg) -> dict[str, Any] | None:
        """Parse a single LiveResponse.Message into a structured dict."""
        method = msg.method
        payload = msg.payload
        result: dict[str, Any] = {"method": method, "raw": {}}

        try:
            if method == "WebcastChatMessage":
                chat = LivePb.ChatMessage()
                chat.ParseFromString(payload)
                result["type"] = "chat"
                result["content"] = chat.content
                # Guard against null user sub-message
                result["userName"] = chat.user.nickname if chat.HasField("user") else ""
                result["userId"] = chat.user.id if chat.HasField("user") else ""
                result["secUid"] = chat.user.sec_uid if chat.HasField("user") else ""
                return result

            elif method == "WebcastGiftMessage":
                gift = LivePb.GiftMessage()
                gift.ParseFromString(payload)
                result["type"] = "gift"
                result["giftName"] = gift.gift.name if gift.HasField("gift") and gift.gift else ""
                result["comboCount"] = gift.comboCount
                result["userName"] = gift.user.nickname if gift.HasField("user") else ""
                result["toUserName"] = gift.toUser.nickname if gift.HasField("toUser") else ""
                return result

            elif method == "WebcastLikeMessage":
                like = LivePb.LikeMessage()
                like.ParseFromString(payload)
                result["type"] = "like"
                result["count"] = like.count
                result["total"] = like.total
                result["userName"] = like.user.nickname if like.HasField("user") else ""
                return result

            elif method == "WebcastMemberMessage":
                member = LivePb.MemberMessage()
                member.ParseFromString(payload)
                result["type"] = "member"
                result["userName"] = member.user.nickname if member.HasField("user") else ""
                result["memberCount"] = member.memberCount
                return result

            elif method == "WebcastSocialMessage":
                social = LivePb.SocialMessage()
                social.ParseFromString(payload)
                result["type"] = "social"
                result["userName"] = social.user.nickname if social.HasField("user") else ""
                result["action"] = social.action
                return result

            elif method == "WebcastRoomStatsMessage":
                stats = LivePb.RoomStatsMessage()
                stats.ParseFromString(payload)
                result["type"] = "roomStats"
                result["displayLong"] = stats.displayLong
                result["displayShort"] = stats.displayShort
                return result

            else:
                self._unknownMethods[method] += 1
                result["type"] = "unknown"
                return result

        except Exception:
            log.debug(f"Failed to parse {method}", exc_info=True)
            return None
