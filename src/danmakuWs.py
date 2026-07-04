# -*- coding: utf-8 -*-
"""Live danmaku WebSocket client.

Connects to ``wss://webcast100-ws-web-hl.douyin.com/webcast/im/push/v2/``,
receives Protobuf PushFrame messages, decompresses with gzip, and parses
individual message types (chat, gift, like, member, social, stats).
"""

import gzip
import json
import threading
import time
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

        self._ws: WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._msgCount = 0
        self._consecutiveErrors = 0

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
        self._thread = threading.Thread(target=self._runForever, daemon=True)
        self._thread.start()
        log.info("Danmaku WS starting: %s", self.url[:100])

    def stop(self):
        """Signal the WebSocket to close."""
        self._running = False
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

    # ── internal ──────────────────────────────────────────────────

    def _runForever(self):
        while self._running:
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
                self._consecutiveErrors = 0
            except Exception as e:
                self._consecutiveErrors += 1
                # Exponential backoff on repeated failures
                delay = min(
                    self._reconnectDelay * (2 ** (self._consecutiveErrors - 1)),
                    _cfg.backoff_max,
                )
                log.warning(f"WS error, reconnecting in {delay:.1f}s (error #{self._consecutiveErrors}): {e}")
            if self._running:
                time.sleep(self._reconnectDelay)

    def _onOpen(self, ws):
        log.info("Danmaku WS connected ✓")
        # Start heartbeat thread
        def pingLoop():
            while self._running:
                try:
                    frame = LivePb.PushFrame()
                    frame.payloadType = "hb"
                    ws.send(frame.SerializeToString(), opcode=0x02)
                except Exception:
                    break
                time.sleep(self._pingInterval)

        t = threading.Thread(target=pingLoop, daemon=True)
        t.start()

    def _onMessage(self, ws, message):
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
        log.warning(f"WS error: {error}")

    def _onClose(self, ws, code, msg):
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
                result["type"] = "unknown"
                return result

        except Exception:
            log.debug(f"Failed to parse {method}", exc_info=True)
            return None
