# -*- coding: utf-8 -*-
"""Authentication management — loads cookies from .env or direct string."""

import os
from dotenv import load_dotenv
from src.log.logger import getLogger
from src.util import transCookies

log = getLogger(__name__)


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
    def fromEnv(cls, envPath: str | None = None) -> "Auth":
        """Load cookies from .env file (DY_LIVE_COOKIES or DY_COOKIES).

        Args:
            envPath: Path to .env file. If None, searches current dir upward.
        """
        if envPath and os.path.exists(envPath):
            load_dotenv(envPath)
        else:
            load_dotenv()

        cookieStr = os.getenv("DY_LIVE_COOKIES") or os.getenv("DY_COOKIES") or ""
        return cls.fromString(cookieStr)

    @classmethod
    def fromString(cls, cookieStr: str) -> "Auth":
        """Parse a cookie string into an Auth instance."""
        auth = cls()
        auth.cookieStr = cookieStr
        auth.cookie = transCookies(cookieStr)
        auth.msToken = auth.cookie.get("msToken", "")
        log.info(f"Auth loaded: {len(auth.cookie)} cookies")
        return auth



