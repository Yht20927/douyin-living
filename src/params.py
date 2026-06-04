# -*- coding: utf-8 -*-
"""Query parameter builder for douyin API requests.

Mirrors Douyin_Spider's builder/params.py with camelCase naming.
"""

from urllib.parse import urlencode
from src.log.logger import getLogger

log = getLogger(__name__)


class Params:
    """Accumulates query parameters for douyin API and WebSocket requests.

    Usage::

        p = Params()
        p.withPlatform().withMsToken().withABogus()
        url = f"https://...?{p.build()}"
    """

    def __init__(self):
        self._params: dict[str, str] = {}

    # ── fluent builders ───────────────────────────────────────────

    def withPlatform(self) -> "Params":
        """Add common platform / browser fingerprint params."""
        self._params.update({
            "device_platform": "webapp",
            "aid": "6383",
            "channel": "channel_pc_web",
            "pc_client_type": "1",
            "update_version_code": "170400",
            "version_code": "170400",
            "version_name": "17.4.0",
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "browser_language": "zh-CN",
            "browser_platform": "Win32",
            "browser_name": "Chrome",
            "browser_version": "125.0.0.0",
            "browser_online": "true",
            "engine_name": "Blink",
            "engine_version": "125.0.0.0",
            "os_name": "Windows",
            "os_version": "10",
            "cpu_core_num": "12",
            "device_memory": "16",
            "platform": "PC",
            "downlink": "10",
            "effective_type": "4g",
            "round_trip_time": "100",
        })
        return self

    def withWebId(self, webId: str | None = None) -> "Params":
        """Add webid (generated if not provided)."""
        from src.signer import Signer
        self._params["webid"] = webId or Signer.generateWebId()
        return self

    def withMsToken(self, msToken: str | None = None) -> "Params":
        """Add msToken (generated if not provided)."""
        from src.signer import Signer
        self._params["msToken"] = msToken or Signer.generateMsToken()
        return self

    def withABogus(self, data: dict | None = None) -> "Params":
        """Add a_bogus, computed from current params + optional data."""
        from src.signer import Signer
        query = urlencode(self._params)
        dataStr = urlencode(data) if data else ""
        self._params["a_bogus"] = Signer.generateABogus(query, dataStr)
        return self

    def add(self, key: str, value: str) -> "Params":
        """Add a single parameter."""
        self._params[key] = value
        return self

    def addAll(self, mapping: dict[str, str]) -> "Params":
        """Add multiple parameters."""
        self._params.update(mapping)
        return self

    # ── output ────────────────────────────────────────────────────

    def build(self) -> str:
        """Return URL-encoded query string."""
        return urlencode(self._params)

    def toDict(self) -> dict[str, str]:
        """Return raw parameter dict."""
        return dict(self._params)
