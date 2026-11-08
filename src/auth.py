# -*- coding: utf-8 -*-
"""Authentication management — loads cookies from env, cookie.txt, or a string.

Cookie values are credentials: never log their contents.  Browser copy/paste
sometimes truncates a value with ``…`` or other non-ASCII text; HTTP cookies
must be ASCII, so such entries are skipped instead of crashing every request.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.log.logger import getLogger

log = getLogger(__name__)

COOKIE_FILE_ENV = "DY_COOKIE_FILE"
COOKIE_FILE_NAME = "cookie.txt"

_COOKIE_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_NETSCAPE_FIELDS = 7


def _isPrintableAscii(value: str) -> bool:
    """Return True when every character can be sent in an HTTP Cookie header."""
    return all(32 <= ord(ch) <= 126 for ch in value)


def _parseCookiePairs(cookieStr: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Split a raw cookie string into valid pairs and skipped entry names."""
    pairs: list[tuple[str, str]] = []
    skipped: list[str] = []
    for item in (cookieStr or "").split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not _COOKIE_NAME_RE.match(name) or not _isPrintableAscii(value):
            if name:
                skipped.append(name)
            continue
        pairs.append((name, value))
    return pairs, skipped


def _parseNetscapeCookies(lines: list[str]) -> list[str]:
    """Parse a Netscape ``cookies.txt`` export into ``name=value`` pairs."""
    pairs: list[str] = []
    for line in lines:
        line = line.removeprefix("#HttpOnly_")
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < _NETSCAPE_FIELDS:
            continue
        name, value = fields[5].strip(), fields[6].strip()
        if name and value:
            pairs.append(f"{name}={value}")
    return pairs


def _readCookieFile(path: Path) -> str:
    """Read a raw or Netscape cookie file, returning a single cookie string."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any("\t" in line for line in lines):
        pairs = _parseNetscapeCookies(lines)
        if pairs:
            return "; ".join(pairs)
    rawLines = [line.rstrip(";") for line in lines if not line.startswith("#")]
    return "; ".join(rawLines)


def _findCookieFile(explicit: str | os.PathLike[str] | None) -> Path | None:
    """Resolve an explicit, env-configured, or cwd ``cookie.txt`` file."""
    candidate = explicit or os.getenv(COOKIE_FILE_ENV)
    path = Path(candidate).expanduser() if candidate else Path.cwd() / COOKIE_FILE_NAME
    return path if path.is_file() else None


class Auth:
    """Douyin authentication container.

    Usage::

        auth = Auth.fromEnv()
        print(auth.cookieStr)   # "; ".join formatted
        print(auth.cookie)      # dict
    """

    def __init__(self):
        self.cookie: dict[str, str] = {}
        self.cookieStr: str = ""
        self.msToken: str = ""
        self.privateKey: str = ""
        self.ticket: str = ""
        self.tsSign: str = ""
        self.uid: str = ""

    @classmethod
    def fromEnv(
        cls,
        envPath: str | None = None,
        cookiePath: str | os.PathLike[str] | None = None,
    ) -> "Auth":
        """Load cookies from env vars, then fall back to ``cookie.txt``.

        Precedence:

        1. ``DY_LIVE_COOKIES`` / ``DY_COOKIES`` (loaded from ``.env`` too);
        2. explicit ``cookiePath`` argument;
        3. ``DY_COOKIE_FILE`` environment variable;
        4. ``./cookie.txt``.

        Args:
            envPath: Path to .env file. If None, searches current dir upward.
            cookiePath: Optional explicit cookie file (raw or Netscape format).
        """
        if envPath and os.path.exists(envPath):
            load_dotenv(envPath)
        else:
            load_dotenv()

        cookieStr = os.getenv("DY_LIVE_COOKIES") or os.getenv("DY_COOKIES") or ""
        source = "environment"
        if not cookieStr:
            candidate = _findCookieFile(cookiePath)
            if candidate is not None:
                cookieStr = _readCookieFile(candidate)
                source = str(candidate)
            elif cookiePath:
                log.warning(f"Cookie file not found: {cookiePath}")

        auth = cls.fromString(cookieStr)
        if cookieStr:
            log.info(f"Auth cookie source: {source}")
        return auth

    @classmethod
    def fromString(cls, cookieStr: str) -> "Auth":
        """Parse and sanitize a cookie string into an Auth instance.

        Invalid entries (non-ASCII values, malformed names, truncated paste
        artefacts) are skipped.  Only entry names are logged, never values.
        """
        auth = cls()
        pairs, skipped = _parseCookiePairs(cookieStr)
        auth.cookie = dict(pairs)
        auth.cookieStr = "; ".join(f"{name}={value}" for name, value in pairs)
        auth.msToken = auth.cookie.get("msToken", "")

        if skipped:
            names = ", ".join(sorted(set(skipped)))
            log.warning(
                f"Skipped {len(skipped)} invalid cookie entries "
                f"(non-ASCII or malformed): {names}"
            )
        if not pairs:
            log.warning("No usable cookies — auth may fail")
        log.info(f"Auth loaded: {len(auth.cookie)} cookies")
        return auth
