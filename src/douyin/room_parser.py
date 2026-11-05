# -*- coding: utf-8 -*-
"""Douyin live page parser — structured JSON first, regex only as fallback.

The live page has embedded room metadata in several historical shapes:

* ``<script id="RENDER_DATA">`` with URL-encoded JSON;
* ``window.__INITIAL_STATE__ = {...}``;
* nonce ``<script>`` payloads whose JSON is double-escaped.

Keeping all of that knowledge here means a protocol change only touches one
module instead of being spread across ``roomApi`` call sites.
"""

import html as htmlModule
import json
import re
from typing import Any, Iterator
from urllib.parse import unquote

DEFAULT_ROOM_INFO: dict[str, str] = {
    "roomId": "",
    "userId": "",
    "userUniqueId": "",
    "anchorId": "",
    "secUid": "",
    "ttwid": "",
    "roomStatus": "",
    "roomTitle": "",
}

_SCRIPT_RE = re.compile(
    r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>",
    re.DOTALL | re.IGNORECASE,
)
_RENDER_DATA_RE = re.compile(
    r'<script[^>]*id=["\']RENDER_DATA["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
_FLV_URL_RE = re.compile(
    r'(https?://pull-flv[^"\s\\]+?_(or4|hd|sd|ld|md)\.flv\?[^"\s\\]+)',
    re.IGNORECASE,
)
_INITIAL_STATE_MARKERS = (
    "__INITIAL_STATE__",
    "__NUXT__",
    "__SSR_DATA__",
    "__NEXT_DATA__",
)

_QUALITY_ALIASES = {
    "FULL_HD1": "or4",
    "ORIGIN": "or4",
    "OR4": "or4",
    "HD1": "hd",
    "HD": "hd",
    "SD1": "sd",
    "SD": "sd",
    "SD2": "ld",
    "LD1": "ld",
    "LD": "ld",
    "MD": "md",
}

_JSON_DECODER = json.JSONDecoder()


# ── public API ───────────────────────────────────────────────────


def extractRoomInfo(html: str, webRid: str = "") -> dict[str, str]:
    """Return room metadata from a live page.

    Structured JSON is tried first; regex fills any fields still missing so
    older/partial SSR payloads keep working.
    """
    result = dict(DEFAULT_ROOM_INFO)

    for data in _iterJsonCandidates(html):
        _mergeRoomInfo(result, data)
        if result["roomId"]:
            break

    regexResult = _extractRoomInfoRegex(html)
    for key, value in regexResult.items():
        if not result.get(key) and value:
            result[key] = value

    return result


def extractStreamUrls(html: str) -> dict[str, str]:
    """Extract signed FLV URLs (quality → URL) from a live page."""
    for data in _iterJsonCandidates(html):
        found = _collectUrlsFromJson(data)
        if found:
            return found

    # Regex fallback: decode escapes first, then scan every script body.
    for match in _SCRIPT_RE.finditer(html):
        decoded = _decodeEscapes(match.group("body"))
        if "pull-flv" not in decoded:
            continue
        found: dict[str, str] = {}
        for url, quality in _FLV_URL_RE.findall(htmlModule.unescape(decoded)):
            found[quality.lower()] = url
        if found:
            return found
    return {}


def extractStreamUrlsFromJson(payload: Any) -> dict[str, str]:
    """Extract FLV URLs from an ``enterRoom`` JSON payload."""
    result: dict[str, str] = {}
    _collectEnterUrls(payload, result)
    return result


def extractStreamUrlsFromPayload(payload: Any) -> dict[str, str]:
    """Compatibility alias used by callers that only have decoded JSON."""
    return extractStreamUrlsFromJson(payload)


# ── JSON discovery ───────────────────────────────────────────────


def _iterJsonCandidates(html: str) -> Iterator[Any]:
    """Yield decoded JSON documents embedded in the page, in priority order."""
    for match in _RENDER_DATA_RE.finditer(html):
        body = htmlModule.unescape(match.group(1).strip())
        for candidate in (unquote(body), body):
            parsed = _loadJson(candidate)
            if parsed is not None:
                yield parsed
                break

    for marker in _INITIAL_STATE_MARKERS:
        start = 0
        while True:
            index = html.find(marker, start)
            if index < 0:
                break
            brace = html.find("{", index)
            if brace >= 0:
                parsed = _rawDecode(html, brace)
                if parsed is not None:
                    yield parsed
            start = index + len(marker)

    for match in _SCRIPT_RE.finditer(html):
        body = match.group("body")
        if not any(token in body for token in ("roomId", "pull-flv", "flv_pull_url")):
            continue
        for candidate in _candidateTexts(body):
            parsed = _loadJson(candidate)
            if parsed is not None:
                yield parsed
                break


def _candidateTexts(text: str) -> Iterator[str]:
    """Yield progressively unescaped variants of an SSR script body."""
    stripped = text.strip()
    if not stripped:
        return
    yield stripped
    decoded = _decodeEscapes(stripped)
    if decoded != stripped:
        yield decoded


def _decodeEscapes(text: str) -> str:
    text = text.replace("\\/", "/")
    text = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )
    return text.replace('\\"', '"')


def _loadJson(text: str) -> Any:
    """Parse a JSON document, tolerating wrappers around the object."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    for opener in ("{", "["):
        index = stripped.find(opener)
        if index < 0:
            continue
        parsed = _rawDecode(stripped, index)
        if parsed is not None:
            return parsed
    return None


def _rawDecode(text: str, start: int) -> Any:
    try:
        parsed, _ = _JSON_DECODER.raw_decode(text, start)
        return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ── room info extraction ─────────────────────────────────────────


def _mergeRoomInfo(result: dict[str, str], data: Any) -> None:
    roomId = _firstValue(
        data,
        [
            ("roomInfo", "room", "id_str"),
            ("roomInfo", "room", "id"),
            ("room", "id_str"),
            ("room", "id"),
            ("roomId",),
            ("room_id",),
            ("data", "roomInfo", "room", "id_str"),
        ],
    )
    if roomId:
        result["roomId"] = roomId

    userId = _firstValue(
        data,
        [
            ("user_unique_id",),
            ("userUniqueId",),
            ("user", "unique_id"),
            ("user", "id_str"),
        ],
    )
    if userId:
        result["userId"] = userId

    userUniqueId = _firstValue(
        data,
        [("user_unique_id",), ("userUniqueId",), ("user", "unique_id")],
    )
    if userUniqueId:
        result["userUniqueId"] = userUniqueId

    anchorId = _firstValue(
        data,
        [
            ("anchor", "id_str"),
            ("anchor", "id"),
            ("roomInfo", "room", "owner", "id_str"),
            ("room", "owner", "id_str"),
        ],
    )
    if anchorId:
        result["anchorId"] = anchorId

    secUid = _firstValue(
        data,
        [
            ("sec_uid",),
            ("anchor", "sec_uid"),
            ("roomInfo", "room", "owner", "sec_uid"),
            ("room", "owner", "sec_uid"),
            ("user", "sec_uid"),
        ],
    )
    if secUid:
        result["secUid"] = secUid

    roomStatus = _firstValue(
        data,
        [
            ("roomInfo", "room", "status"),
            ("room", "status"),
            ("room_status",),
            ("status",),
        ],
    )
    if roomStatus:
        result["roomStatus"] = roomStatus

    roomTitle = _firstValue(
        data,
        [
            ("roomInfo", "room", "title"),
            ("room", "title"),
            ("roomInfo", "room", "title_str"),
            ("title",),
        ],
    )
    if roomTitle:
        result["roomTitle"] = roomTitle


def _firstValue(data: Any, paths: list[tuple[str, ...]]) -> str:
    for path in paths:
        value = data
        for key in path:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float)):
            return str(value)
    return ""


def _extractRoomInfoRegex(html: str) -> dict[str, str]:
    """Historical regex fallback for partial or malformed SSR payloads."""
    result = dict(DEFAULT_ROOM_INFO)

    scripts = re.findall(r"<script[^>]*nonce[^>]*>(.*?)</script>", html, re.DOTALL)
    if not scripts:
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)

    for scriptContent in scripts:
        if not scriptContent or "roomId" not in scriptContent:
            continue
        try:
            escaped = scriptContent
            plain = _decodeEscapes(scriptContent)
            for text, escapedQuotes in ((escaped, True), (plain, False)):
                if not escapedQuotes:
                    _applyPlainRegex(result, text)
                _applyEscapedRegex(result, text)
            if result["roomId"]:
                break
        except Exception:  # pragma: no cover - regex never raises today
            continue
    return result


def _applyEscapedRegex(result: dict[str, str], text: str) -> None:
    userIdMatches = re.findall(r'\\"user_unique_id\\":\\"(\d+)\\"', text)
    roomIdMatches = re.findall(r'\\"roomId\\":\\"(\d+)\\"', text)
    roomInfoMatches = re.findall(
        r'\\"roomInfo\\":\{\\"room\\":\{\\"id_str\\":\\".*?\\",'
        r'\\"status\\":(.*?),\\"status_str\\":\\".*?\\",\\"title\\":\\"(.*?)\\"',
        text,
    )
    anchorMatches = re.findall(r'\\"anchor\\":\{\\"id_str\\":\\"(\d+)\\"', text)
    secUidMatches = re.findall(r'\\"sec_uid\\":\\"(.*?)\\"', text)

    if not result["userId"] and userIdMatches:
        result["userId"] = userIdMatches[0]
    if not result["roomId"] and roomIdMatches:
        result["roomId"] = roomIdMatches[0]
    if not result["roomStatus"] and roomInfoMatches:
        result["roomStatus"] = roomInfoMatches[0][0]
    if not result["roomTitle"] and roomInfoMatches:
        result["roomTitle"] = roomInfoMatches[0][1]
    if not result["anchorId"] and anchorMatches:
        result["anchorId"] = anchorMatches[0]
    if not result["secUid"] and secUidMatches:
        result["secUid"] = secUidMatches[0]


def _applyPlainRegex(result: dict[str, str], text: str) -> None:
    roomIdMatches = re.findall(r'"roomId"\s*:\s*"?(\d+)"?', text)
    userIdMatches = re.findall(r'"user_unique_id"\s*:\s*"(\d+)"', text)
    anchorMatches = re.findall(r'"anchor"\s*:\s*\{\s*"id_str"\s*:\s*"(\d+)"', text)
    secUidMatches = re.findall(r'"sec_uid"\s*:\s*"([^"]+)"', text)
    roomInfoMatches = re.findall(
        r'"roomInfo"\s*:\s*\{\s*"room"\s*:\s*\{.*?'
        r'"status"\s*:\s*"?(\d+)"?.*?'
        r'"title"\s*:\s*"([^"]*)"',
        text,
        re.DOTALL,
    )

    if not result["roomId"] and roomIdMatches:
        result["roomId"] = roomIdMatches[0]
    if not result["userId"] and userIdMatches:
        result["userId"] = userIdMatches[0]
    if not result["anchorId"] and anchorMatches:
        result["anchorId"] = anchorMatches[0]
    if not result["secUid"] and secUidMatches:
        result["secUid"] = secUidMatches[0]
    if not result["roomStatus"] and roomInfoMatches:
        result["roomStatus"] = roomInfoMatches[0][0]
    if not result["roomTitle"] and roomInfoMatches:
        result["roomTitle"] = roomInfoMatches[0][1]


# ── stream URL extraction ────────────────────────────────────────


def _collectUrlsFromJson(data: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    _walkJson(data, result)
    return result


def _walkJson(value: Any, result: dict[str, str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _walkJson(item, result)
    elif isinstance(value, list):
        for item in value:
            _walkJson(item, result)
    elif isinstance(value, str) and "pull-flv" in value and ".flv" in value:
        decoded = htmlModule.unescape(_decodeEscapes(value))
        for url, quality in _FLV_URL_RE.findall(decoded):
            result.setdefault(quality.lower(), url)


def _collectEnterUrls(value: Any, result: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and ".flv" in item:
                quality = _normalizeQuality(str(key), item)
                if quality:
                    result.setdefault(quality, htmlModule.unescape(item))
                continue
            _collectEnterUrls(item, result)
    elif isinstance(value, list):
        for item in value:
            _collectEnterUrls(item, result)


def _normalizeQuality(key: str, url: str) -> str | None:
    upper = key.upper()
    if upper in _QUALITY_ALIASES:
        return _QUALITY_ALIASES[upper]

    match = re.search(r"_(or4|hd|sd|ld|md)\.flv", url, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    lowered = key.lower()
    if lowered in {"or4", "hd", "sd", "ld", "md"}:
        return lowered
    return None
