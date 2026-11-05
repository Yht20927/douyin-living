# -*- coding: utf-8 -*-
"""Room API — douyin live room info and stream URLs."""

import asyncio
import random
import threading
from typing import Any
import httpx
from src.log.logger import getLogger
from src.util import transCookies
from src.douyin.room_parser import (
    extractRoomInfo,
    extractStreamUrls,
    extractStreamUrlsFromJson,
)

log = getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECS = 0.5
BACKOFF_MAX_SECS = 8.0
RETRY_STATUSES = {429, 500, 502, 503, 504}

HEADERS_PC = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://live.douyin.com",
}

# Module-level shared client — initialised once at import time so we
# never block the event loop inside an async function with a lock.
_client: httpx.AsyncClient | None = None
_clientLock = threading.Lock()


def _get_client() -> httpx.AsyncClient:
    """Return the module-level shared httpx client, creating it if needed."""
    global _client
    if _client is None:
        with _clientLock:
            if _client is None:
                _client = httpx.AsyncClient(
                    timeout=httpx.Timeout(15.0), follow_redirects=True
                )
    return _client


async def _sleepBackoff(seconds: float) -> None:
    """Sleep helper kept separate so tests can patch retry delays."""
    await asyncio.sleep(seconds)


def _retryDelay(attempt: int) -> float:
    """Exponential backoff with jitter for transient HTTP failures."""
    base = min(BACKOFF_BASE_SECS * (2 ** (attempt - 1)), BACKOFF_MAX_SECS)
    return base + random.uniform(0, base * 0.25)


async def _request(url: str, **kwargs) -> httpx.Response:
    """GET with retry/backoff for 429 and 5xx responses.

    Cookie failures (401/403) are not retried; they log a clear hint so the
    operator knows to refresh ``cookie.txt``.
    """
    client = _get_client()
    lastError: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = await client.get(url, **kwargs)
        except httpx.TransportError as exc:
            lastError = exc
            if attempt >= MAX_ATTEMPTS:
                break
            delay = _retryDelay(attempt)
            log.warning(
                f"HTTP transport error (attempt {attempt}/{MAX_ATTEMPTS}), "
                f"retrying in {delay:.2f}s: {exc}"
            )
            await _sleepBackoff(delay)
            continue

        status = getattr(resp, "status_code", 200)
        if status in (401, 403):
            log.warning(
                f"HTTP {status} from live.douyin.com — Cookie may be expired "
                "or invalid; refresh cookie.txt and retry"
            )
        if status in RETRY_STATUSES:
            lastError = httpx.HTTPStatusError(
                f"transient HTTP {status}", request=None, response=resp  # type: ignore[arg-type]
            )
            if attempt < MAX_ATTEMPTS:
                delay = _retryDelay(attempt)
                log.warning(
                    f"HTTP {status} (attempt {attempt}/{MAX_ATTEMPTS}), "
                    f"retrying in {delay:.2f}s"
                )
                await _sleepBackoff(delay)
                continue
            log.error(f"HTTP {status} after {MAX_ATTEMPTS} attempts")
        resp.raise_for_status()
        return resp

    if lastError is not None:
        raise lastError
    raise RuntimeError("request failed without an exception")


class RoomApi:
    """Douyin live room API client.

    Usage::

        info = await RoomApi.getLiveInfo(auth, "799525583657")
        detail = await RoomApi.getWebcastDetail(auth, info["userId"], info["roomId"])
    """

    @staticmethod
    async def close():
        """Explicitly close the shared HTTP client and release connections."""
        global _client
        if _client is not None:
            await _client.aclose()
            _client = None
            log.debug("HTTP client closed")

    @staticmethod
    async def getLiveInfo(auth, webRid: str) -> dict[str, str]:
        """Fetch room info from the live page HTML.

        Returns dict with keys: roomId, userId, userUniqueId, anchorId,
        secUid, ttwid, roomStatus, roomTitle.
        """
        url = f"https://live.douyin.com/{webRid}"
        headers = dict(HEADERS_PC)

        # Pass cookies as dict (httpx format), not as header string
        cookies = None
        if hasattr(auth, "cookie") and auth.cookie:
            cookies = auth.cookie
        elif hasattr(auth, "cookieStr") and auth.cookieStr:
            cookies = transCookies(auth.cookieStr)

        resp = await _request(
            url, headers=headers, cookies=cookies, follow_redirects=True
        )
        html = resp.text

        # Extract ttwid from response cookies
        ttwid = ""
        for cookie in resp.cookies.jar:
            if cookie.name == "ttwid":
                ttwid = cookie.value
                break

        # Extract room info from SSR data in the page
        result = extractRoomInfo(html, webRid)
        if ttwid:
            result["ttwid"] = ttwid

        log.info(f"Room {webRid}: {result.get('roomTitle','?')} status={result.get('roomStatus','?')}")
        return result

    @staticmethod
    async def getWebcastDetail(auth, userId: str, roomId: str, referer: str = "") -> bytes:
        """Fetch webcast detail (Protobuf binary) for danmaku WS initialization.

        Returns raw protobuf bytes (LiveResponse).
        """
        from src.params import Params
        from src.signer import Signer

        p = Params()
        p.addAll({
            "resp_content_type": "protobuf",  # ← KEY: tells server to return protobuf
            "did_rule": "3",
            "device_id": "",
            "app_name": "douyin_web",
            "endpoint": "live_pc",
            "support_wrds": "1",
            "user_unique_id": str(userId),
            "identity": "audience",
            "need_persist_msg_count": "15",
            "insert_task_id": "",
            "live_reason": "",
            "room_id": roomId,
            "version_code": "180800",
            "last_rtt": "0",
            "live_id": "1",
            "aid": "6383",
            "fetch_rule": "1",
            "cursor": "",
            "internal_ext": "",
            "device_platform": "web",
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "browser_language": "zh-CN",
            "browser_platform": "Win32",
            "browser_name": "Mozilla",
            "browser_version": "5.0",
            "browser_online": "true",
            "tz_name": "Asia/Shanghai",
            "msToken": getattr(auth, "msToken", "") or Signer.generateMsToken(),
        })
        p.withABogus()

        url = "https://live.douyin.com/webcast/im/fetch/"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept": "application/x-protobuf",
            "Origin": "https://live.douyin.com",
            "Referer": referer or "https://live.douyin.com/",
        }

        # Add CSRF token if available
        csrfToken = auth.cookie.get("passport_csrf_token", "") if hasattr(auth, "cookie") else ""
        if csrfToken:
            headers["x-secsdk-csrf-token"] = csrfToken

        cookies = None
        if hasattr(auth, "cookie") and auth.cookie:
            cookies = auth.cookie
        elif hasattr(auth, "cookieStr") and auth.cookieStr:
            cookies = transCookies(auth.cookieStr)

        resp = await _request(
            url,
            params=p.toDict(),
            headers=headers,
            cookies=cookies,
            follow_redirects=True,
        )
        log.debug(f"Webcast detail: {len(resp.content)} bytes")
        return resp.content

    @staticmethod
    async def getStreamUrls(auth, webRid: str) -> dict[str, str]:
        """Extract signed FLV stream URLs from the live page HTML.

        The SSR data contains signed FLV URLs with quality levels.
        Returns dict like::

            {"or4": "http://...or4.flv?expire=...&sign=...",
             "hd": "http://...hd.flv?expire=...&sign=...",
             "sd": "http://...sd.flv?expire=...&sign=...",
             "ld": "http://...ld.flv?expire=...&sign=..."}
        """
        url = f"https://live.douyin.com/{webRid}"
        headers = dict(HEADERS_PC)

        cookies = None
        if hasattr(auth, "cookie") and auth.cookie:
            cookies = auth.cookie
        elif hasattr(auth, "cookieStr") and auth.cookieStr:
            cookies = transCookies(auth.cookieStr)

        result: dict[str, str] = {}
        html = ""
        try:
            resp = await _request(
                url, headers=headers, cookies=cookies, follow_redirects=True
            )
            html = resp.text
            result = extractStreamUrls(html)
        except httpx.HTTPError as exc:
            log.warning(
                f"Live page fetch failed ({type(exc).__name__}: {exc}); "
                "trying enterRoom API"
            )

        if result:
            log.info(f"Extracted {len(result)} FLV URLs: {list(result.keys())}")
            return result

        roomId = ""
        if html:
            roomInfo = extractRoomInfo(html, webRid)
            roomId = roomInfo.get("roomId", "")

        try:
            payload = await RoomApi.enterRoom(auth, webRid, roomId)
            result = extractStreamUrlsFromJson(payload)
        except httpx.HTTPError as exc:
            log.warning(
                f"enterRoom fallback failed ({type(exc).__name__}: {exc})"
            )
        except Exception:
            log.exception("enterRoom fallback failed unexpectedly")

        if result:
            log.info(
                f"enterRoom fallback extracted {len(result)} FLV URLs: "
                f"{list(result.keys())}"
            )
        else:
            log.warning("No FLV URLs found in page SSR or enterRoom API")
        return result

    @staticmethod
    async def enterRoom(auth, webRid: str, roomId: str = "") -> dict[str, Any]:
        """Call web/enter API for full room info (including FLV stream URLs).

        Args:
            auth: Auth instance with cookies.
            webRid: Short room ID (e.g. '300294032039').
            roomId: Internal room ID (from getLiveInfo). Required for stream URLs.

        Returns the parsed JSON response data.
        """
        from src.params import Params
        p = Params()
        p.withPlatform()
        p.add("web_rid", webRid)
        if roomId:
            p.add("room_id_str", roomId)
        p.add("live_id", "1")
        p.add("enter_from", "link_share")
        p.add("is_need_double_stream", "false")
        p.withMsToken()
        p.withABogus()  # ← must be LAST, includes all params

        url = f"https://live.douyin.com/webcast/room/web/enter/?{p.build()}"
        headers = dict(HEADERS_PC)
        headers["Accept"] = "application/json, text/plain, */*"
        headers["Referer"] = f"https://live.douyin.com/{webRid}"

        cookies = None
        if hasattr(auth, "cookie") and auth.cookie:
            cookies = auth.cookie

        resp = await _request(
            url, headers=headers, cookies=cookies, follow_redirects=True
        )
        return resp.json()


# ── internal helpers ──────────────────────────────────────────────

def _extractRoomInfo(html: str, webRid: str) -> dict[str, str]:
    """Backward-compatible wrapper around :func:`extractRoomInfo`."""
    return extractRoomInfo(html, webRid)

