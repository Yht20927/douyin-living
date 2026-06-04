# -*- coding: utf-8 -*-
"""Room API — douyin live room info and stream URLs."""

import re
import json
from typing import Any
import httpx
from src.log.logger import getLogger
from src.util import transCookies

log = getLogger(__name__)

HEADERS_PC = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://live.douyin.com",
}


class RoomApi:
    """Douyin live room API client.

    Usage::

        info = await RoomApi.getLiveInfo(auth, "799525583657")
        detail = await RoomApi.getWebcastDetail(auth, info["userId"], info["roomId"])
    """

    _client: httpx.AsyncClient | None = None

    @classmethod
    def _get_client(cls) -> httpx.AsyncClient:
        """Lazy-init and return the shared httpx client."""
        if cls._client is None:
            cls._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0), follow_redirects=True)
        return cls._client

    @classmethod
    async def close(cls):
        """Explicitly close the shared HTTP client and release connections."""
        if cls._client is not None:
            await cls._client.aclose()
            cls._client = None
            log.debug("HTTP client closed")

    @staticmethod
    async def getLiveInfo(auth, webRid: str) -> dict[str, str]:
        """Fetch room info from the live page HTML.

        Returns dict with keys: roomId, userId, userUniqueId, anchorId,
        secUid, ttwid, roomStatus, roomTitle.
        """
        url = f"https://live.douyin.com/{webRid}"
        headers = dict(HEADERS_PC)

        client = RoomApi._get_client()

        # Pass cookies as dict (httpx format), not as header string
        cookies = None
        if hasattr(auth, "cookie") and auth.cookie:
            cookies = auth.cookie
        elif hasattr(auth, "cookieStr") and auth.cookieStr:
            cookies = transCookies(auth.cookieStr)

        resp = await client.get(url, headers=headers, cookies=cookies, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        # Extract ttwid from response cookies
        ttwid = ""
        for cookie in resp.cookies.jar:
            if cookie.name == "ttwid":
                ttwid = cookie.value
                break

        # Extract room info from SSR data in the page
        result = _extractRoomInfo(html, webRid)
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
            "Referer": referer or f"https://live.douyin.com/",
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

        client = RoomApi._get_client()
        resp = await client.get(url, params=p.toDict(), headers=headers, cookies=cookies, follow_redirects=True)
        resp.raise_for_status()
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

        client = RoomApi._get_client()
        cookies = None
        if hasattr(auth, "cookie") and auth.cookie:
            cookies = auth.cookie

        resp = await client.get(url, headers=headers, cookies=cookies, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        # Find script tags with FLV URLs
        scripts = re.findall(r"<script[^>]*nonce[^>]*>(.*?)</script>", html, re.DOTALL)
        result: dict[str, str] = {}

        for scriptContent in scripts:
            if "pull-flv" not in scriptContent:
                continue

            # First, decode all unicode escapes in the script content
            decoded = scriptContent
            # Replace \u0026 → &, \u002F → /, \u003D → =, etc.
            decoded = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), decoded)
            decoded = decoded.replace("\\/", "/")

            # Match signed FLV URLs — capture full URL including all query params
            # Pattern: http(s)://pull-flv-{cdn}/third/stream-{id}_{quality}.flv?params...
            # The URL ends at the next unescaped double-quote
            matches = re.findall(
                r'(https?://pull-flv[^"]+?'
                r'(?:stream-[\d]+)_(or4|hd|sd|ld|md)\.flv'
                r'\?[^"]+)',
                decoded
            )
            for fullUrl, quality in matches:
                result[quality] = fullUrl

            if result:
                break

        log.info(f"Extracted {len(result)} FLV URLs: {list(result.keys())}")
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

        client = RoomApi._get_client()
        resp = await client.get(url, headers=headers, cookies=cookies, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()


# ── internal helpers ──────────────────────────────────────────────

def _extractRoomInfo(html: str, webRid: str) -> dict[str, str]:
    """Parse room info from the live page HTML.

    Uses the same regex patterns as Douyin_Spider's get_live_info.
    The key data is embedded in script tags with escaped JSON.
    """
    result: dict[str, str] = {
        "roomId": "", "userId": "", "userUniqueId": "",
        "anchorId": "", "secUid": "", "ttwid": "",
        "roomStatus": "", "roomTitle": "",
    }

    # Find script tags with nonce attribute (contains SSR data)
    scripts = re.findall(r'<script[^>]*nonce[^>]*>(.*?)</script>', html, re.DOTALL)
    if not scripts:
        # Fallback: find all script tags
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)

    for scriptContent in scripts:
        if not scriptContent or 'roomId' not in scriptContent:
            continue

        try:
            # Regex patterns from Douyin_Spider (note: double-escaped quotes)
            userIdMatches = re.findall(r'\\"user_unique_id\\":\\"(\d+)\\"', scriptContent)
            roomIdMatches = re.findall(r'\\"roomId\\":\\"(\d+)\\"', scriptContent)
            userUniqueIdMatches = re.findall(r'\\"user_unique_id\\":\\"(\d+)\\"', scriptContent)

            # Room info: status + title
            roomInfoMatches = re.findall(
                r'\\"roomInfo\\":\{\\"room\\":\{\\"id_str\\":\\".*?\\",\\"status\\":(.*?),\\"status_str\\":\\".*?\\",\\"title\\":\\"(.*?)\\"',
                scriptContent
            )

            anchorMatches = re.findall(r'\\"anchor\\":\{\\"id_str\\":\\"(\d+)\\"', scriptContent)
            secUidMatches = re.findall(r'\\"sec_uid\\":\\"(.*?)\\"', scriptContent)

            if roomIdMatches:
                result["roomId"] = roomIdMatches[0]
            if userIdMatches:
                result["userId"] = userIdMatches[0]
            if userUniqueIdMatches:
                result["userUniqueId"] = userUniqueIdMatches[0]
            if roomInfoMatches:
                result["roomStatus"] = roomInfoMatches[0][0]
                result["roomTitle"] = roomInfoMatches[0][1]
            if anchorMatches:
                result["anchorId"] = anchorMatches[0]
            if secUidMatches:
                result["secUid"] = secUidMatches[0]

            # If we got roomId, we're done
            if result["roomId"]:
                break
        except Exception:
            continue

    return result
