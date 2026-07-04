# -*- coding: utf-8 -*-
"""Unit tests for auth module — fromString, fromEnv edge cases."""

import os
import pytest
from src.auth import Auth


class TestAuthFromString:
    def test_empty_string(self):
        auth = Auth.fromString("")
        assert auth.cookieStr == ""
        assert auth.cookie == {}
        assert auth.msToken == ""

    def test_single_cookie(self):
        auth = Auth.fromString("foo=bar")
        assert auth.cookieStr == "foo=bar"
        assert auth.cookie.get("foo") == "bar"

    def test_multiple_cookies(self):
        auth = Auth.fromString("a=1; b=2; c=3")
        assert auth.cookie.get("a") == "1"
        assert auth.cookie.get("b") == "2"
        assert auth.cookie.get("c") == "3"

    def test_msToken_extracted_from_cookie(self):
        auth = Auth.fromString("msToken=abc123; other=val")
        assert auth.msToken == "abc123"

    def test_malformed_cookie_string(self):
        """Malformed cookie strings should not crash."""
        auth = Auth.fromString(";;;==;;")
        assert isinstance(auth.cookie, dict)

    def test_cookie_with_spaces(self):
        auth = Auth.fromString(" key = value ")
        # transCookies trims whitespace
        assert isinstance(auth.cookie, dict)


class TestAuthFromEnv:
    def test_missing_env_returns_empty(self, monkeypatch):
        """When neither env var is set, returns empty Auth."""
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        # Also patch load_dotenv to prevent it from loading the real .env
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        auth = Auth.fromEnv()
        assert auth.cookieStr == ""

    def test_dy_live_cookies_takes_precedence(self, monkeypatch):
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.setenv("DY_LIVE_COOKIES", "live=1")
        monkeypatch.setenv("DY_COOKIES", "main=2")
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        auth = Auth.fromEnv()
        assert auth.cookie.get("live") == "1"

    def test_fallback_to_dy_cookies(self, monkeypatch):
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.setenv("DY_COOKIES", "main=2")
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        auth = Auth.fromEnv()
        assert auth.cookie.get("main") == "2"
