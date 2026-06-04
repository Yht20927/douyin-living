# -*- coding: utf-8 -*-
"""Signature generation for douyin API requests.

msToken: random 107-char string (not cryptographic).
a_bogus: JS-generated signature (dyAb.js via PyExecJS).
X-Bogus (live danmaku): MD5 + JS signature (dyLiveSign.js via PyExecJS).
"""

import hashlib
import random
import string
import execjs
from pathlib import Path
from src.log.logger import getLogger

log = getLogger(__name__)

# JS signing scripts (relative to project root)
_SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
_AB_JS_PATH = _SCRIPT_DIR / "dyAb.js"
_LIVE_SIGN_JS_PATH = _SCRIPT_DIR / "dyLiveSign.js"

# Cached JS contexts
_abJs: "execjs._ExternalRuntime.Context | None" = None  # type: ignore
_liveSignJs: "execjs._ExternalRuntime.Context | None" = None  # type: ignore


class Signer:
    """Douyin signature generator.

    All methods are static/class-level — no instance needed.
    """

    _msTokenAlphabet = (
        "ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789="
    )

    # ── msToken ──────────────────────────────────────────────────

    @staticmethod
    def generateMsToken(length: int = 107) -> str:
        """Generate a random msToken (not verified by server)."""
        return "".join(random.choices(Signer._msTokenAlphabet, k=length))

    # ── a_bogus ─────────────────────────────────────────────────

    @staticmethod
    def generateABogus(query: str, data: str = "") -> str:
        """Generate a_bogus for HTTP API requests.

        Args:
            query: URL query string (e.g. 'aid=6383&web_rid=...')
            data: POST body string (empty for GET requests)
        """
        ctx = Signer._getAbJs()
        return ctx.call("get_ab", query, data)

    # ── Live danmaku signature (X-Bogus) ─────────────────────────

    @staticmethod
    def generateLiveSignature(roomId: str, userId: str) -> str:
        """Generate the X-Bogus signature for live danmaku WebSocket.

        Args:
            roomId: Internal room ID (from getLiveInfo).
            userId: User unique ID (from getLiveInfo).
        """
        raw = (
            f"live_id=1,aid=6383,version_code=180800,"
            f"webcast_sdk_version=1.0.15,room_id={roomId},"
            f"sub_room_id=,sub_channel_id=,did_rule=3,"
            f"user_unique_id={userId},device_platform=web,"
            f"device_type=,ac=,identity=audience"
        )
        xMsStub = hashlib.md5(raw.encode("utf-8")).hexdigest()
        ctx = Signer._getLiveSignJs()
        result = ctx.call("get_signature", xMsStub)
        return result.get("X-Bogus", "")

    # ── JS context management ────────────────────────────────────

    @staticmethod
    def _getAbJs():
        global _abJs
        if _abJs is None:
            if not _AB_JS_PATH.exists():
                raise FileNotFoundError(f"dyAb.js not found at {_AB_JS_PATH}")
            _abJs = execjs.compile(
                _AB_JS_PATH.read_text(encoding="utf-8"),
                cwd=str(_SCRIPT_DIR),
            )
            log.info("dyAb.js compiled")
        return _abJs

    @staticmethod
    def _getLiveSignJs():
        global _liveSignJs
        if _liveSignJs is None:
            if not _LIVE_SIGN_JS_PATH.exists():
                raise FileNotFoundError(f"dyLiveSign.js not found at {_LIVE_SIGN_JS_PATH}")
            _liveSignJs = execjs.compile(
                _LIVE_SIGN_JS_PATH.read_text(encoding="utf-8"),
                cwd=str(_SCRIPT_DIR),
            )
            log.info("dyLiveSign.js compiled")
        return _liveSignJs

    # ── Utility ──────────────────────────────────────────────────

    @staticmethod
    def generateWebId(length: int = 19) -> str:
        """Generate a fake webid (19-digit numeric string)."""
        return "".join(random.choices("0123456789", k=length))

    @staticmethod
    def generateMsTokenValue() -> str:
        """Alias for generateMsToken (compatibility)."""
        return Signer.generateMsToken()
