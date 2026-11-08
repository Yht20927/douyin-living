# -*- coding: utf-8 -*-
"""Unit tests for auth module — fromString, fromEnv, cookie.txt fallback."""

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

    def test_non_ascii_value_is_skipped(self):
        auth = Auth.fromString("good=ok; broken=1…2; also=ok")

        assert auth.cookie == {"good": "ok", "also": "ok"}
        assert auth.cookieStr == "good=ok; also=ok"
        assert all(ord(ch) < 128 for ch in auth.cookieStr)

    def test_invalid_name_is_skipped(self):
        auth = Auth.fromString("bad name=value; good=1")

        assert auth.cookie == {"good": "1"}

    def test_value_with_equals_is_preserved(self):
        auth = Auth.fromString("token=abc==; other=1")

        assert auth.cookie["token"] == "abc=="


class TestAuthFromEnv:
    def test_missing_env_returns_empty(self, monkeypatch, tmp_path):
        """When neither env var nor cookie.txt exists, returns empty Auth."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIE_FILE", raising=False)
        # Also patch load_dotenv to prevent it from loading the real .env
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        auth = Auth.fromEnv()
        assert auth.cookieStr == ""

    def test_dy_live_cookies_takes_precedence(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.setenv("DY_LIVE_COOKIES", "live=1")
        monkeypatch.setenv("DY_COOKIES", "main=2")
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        auth = Auth.fromEnv()
        assert auth.cookie.get("live") == "1"

    def test_fallback_to_dy_cookies(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.setenv("DY_COOKIES", "main=2")
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        auth = Auth.fromEnv()
        assert auth.cookie.get("main") == "2"

    def test_cookie_txt_fallback(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIE_FILE", raising=False)
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        (tmp_path / "cookie.txt").write_text(
            "ttwid=abc; broken=1…2; sessionid=xyz\n", encoding="utf-8"
        )

        auth = Auth.fromEnv()

        assert auth.cookie == {"ttwid": "abc", "sessionid": "xyz"}

    def test_env_cookie_takes_precedence_over_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DY_LIVE_COOKIES", "env=1")
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        (tmp_path / "cookie.txt").write_text("file=2", encoding="utf-8")

        auth = Auth.fromEnv()

        assert auth.cookie == {"env": "1"}

    def test_dy_cookie_file_override(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        custom = tmp_path / "custom-cookies.txt"
        custom.write_text("custom=1", encoding="utf-8")
        monkeypatch.setenv("DY_COOKIE_FILE", str(custom))

        auth = Auth.fromEnv()

        assert auth.cookie == {"custom": "1"}

    def test_netscape_cookie_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DY_LIVE_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIES", raising=False)
        monkeypatch.delenv("DY_COOKIE_FILE", raising=False)
        monkeypatch.setattr("src.auth.load_dotenv", lambda *a, **kw: None)
        (tmp_path / "cookie.txt").write_text(
            "# Netscape HTTP Cookie File\n"
            ".douyin.com\tTRUE\t/\tTRUE\t0\tttwid\tabc\n"
            "#HttpOnly_.douyin.com\tTRUE\t/\tTRUE\t0\tsessionid\txyz\n",
            encoding="utf-8",
        )

        auth = Auth.fromEnv()

        assert auth.cookie == {"ttwid": "abc", "sessionid": "xyz"}
