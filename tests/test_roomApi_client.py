# -*- coding: utf-8 -*-
"""RoomApi client tests — httpx calls are fully mocked (no network)."""

import asyncio
import types

import httpx
import pytest

import src.roomApi as roomApi


class FakeResponse:
    def __init__(self, text="", content=b"", jsonData=None, cookies=None, status_code=200):
        self.text = text
        self.content = content
        self._json = jsonData
        self.cookies = types.SimpleNamespace(jar=cookies or [])
        self.status_code = status_code
        self.raised = False

    def raise_for_status(self):
        self.raised = True
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self  # type: ignore[arg-type]
            )

    def json(self):
        return self._json


class FakeClient:
    def __init__(self, response):
        self.responses = response if isinstance(response, list) else [response]
        self.calls = []
        self.closed = False

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    async def aclose(self):
        self.closed = True


class FakeAuth:
    def __init__(self, cookie=None, cookieStr=""):
        self.cookie = cookie or {}
        self.cookieStr = cookieStr


class FakeParams:
    def __init__(self):
        self.values = {}
        self.abogus = False

    def addAll(self, values):
        self.values.update(values)

    def add(self, key, value):
        self.values[key] = value

    def withPlatform(self):
        self.values["platform"] = "web"

    def withMsToken(self):
        self.values["msToken"] = "token"

    def withABogus(self):
        self.abogus = True
        self.values["a_bogus"] = "sig"

    def build(self):
        return "&".join(f"{k}={v}" for k, v in self.values.items())

    def toDict(self):
        return dict(self.values)


class TestSharedClient:
    def test_lazy_init_and_close(self, monkeypatch):
        created = FakeClient(FakeResponse())

        def factory(**kwargs):
            return created

        monkeypatch.setattr(roomApi, "_client", None)
        monkeypatch.setattr(roomApi.httpx, "AsyncClient", factory)
        assert roomApi._get_client() is created
        asyncio.run(roomApi.RoomApi.close())
        assert created.closed is True
        assert roomApi._client is None


class TestGetLiveInfo:
    def test_parses_html_and_ttwid(self, monkeypatch):
        html = (
            '<script nonce="abc">'
            r'{\"roomId\":\"987\",\"user_unique_id\":\"111\",'
            r'\"roomInfo\":{\"room\":{\"id_str\":\"987\",\"status\":2,'
            r'\"status_str\":\"LIVE\",\"title\":\"测试\"}},'
            r'\"anchor\":{\"id_str\":\"444\"},\"sec_uid\":\"sec-1\"}'
            '</script>'
        )
        response = FakeResponse(
            text=html,
            cookies=[types.SimpleNamespace(name="ttwid", value="tt-1")],
        )
        client = FakeClient(response)
        monkeypatch.setattr(roomApi, "_client", client)

        result = asyncio.run(
            roomApi.RoomApi.getLiveInfo(FakeAuth(cookie={"a": "1"}), "12345")
        )
        assert result["roomId"] == "987"
        assert result["userId"] == "111"
        assert result["anchorId"] == "444"
        assert result["secUid"] == "sec-1"
        assert result["ttwid"] == "tt-1"
        assert response.raised is True
        _, kwargs = client.calls[0]
        assert kwargs["cookies"] == {"a": "1"}


class TestGetWebcastDetail:
    def test_builds_signed_request(self, monkeypatch):
        response = FakeResponse(content=b"protobuf")
        client = FakeClient(response)
        monkeypatch.setattr(roomApi, "_client", client)
        monkeypatch.setattr("src.params.Params", FakeParams)

        class FakeSigner:
            @staticmethod
            def generateMsToken():
                return "generated"

        monkeypatch.setattr("src.signer.Signer", FakeSigner)

        auth = FakeAuth(cookie={"passport_csrf_token": "csrf"})
        result = asyncio.run(
            roomApi.RoomApi.getWebcastDetail(auth, "user-1", "room-1", referer="https://ref")
        )
        assert result == b"protobuf"
        url, kwargs = client.calls[0]
        assert url.endswith("/webcast/im/fetch/")
        assert kwargs["headers"]["x-secsdk-csrf-token"] == "csrf"
        assert kwargs["headers"]["Referer"] == "https://ref"
        assert kwargs["params"]["room_id"] == "room-1"
        assert kwargs["params"]["a_bogus"] == "sig"


class TestGetStreamUrls:
    def test_decodes_escaped_urls(self, monkeypatch):
        html = (
            '<script nonce="abc">'
            r'var x = "https:\/\/pull-flv-cdn.example.com\/third\/stream-123_hd.flv?expire=1\u0026sign=abc";'
            r'var y = "https:\/\/pull-flv-cdn.example.com\/third\/stream-123_sd.flv?expire=2\u0026sign=def";'
            '</script>'
        )
        client = FakeClient(FakeResponse(text=html))
        monkeypatch.setattr(roomApi, "_client", client)

        result = asyncio.run(roomApi.RoomApi.getStreamUrls(FakeAuth(), "12345"))
        assert set(result) == {"hd", "sd"}
        assert result["hd"].startswith("https://pull-flv")
        assert "sign=abc" in result["hd"]

    def test_no_scripts_returns_empty(self, monkeypatch):
        client = FakeClient(FakeResponse(text="<html></html>"))
        monkeypatch.setattr(roomApi, "_client", client)

        async def emptyEnter(*args, **kwargs):
            return {}

        monkeypatch.setattr(
            roomApi.RoomApi, "enterRoom", staticmethod(emptyEnter)
        )
        assert asyncio.run(roomApi.RoomApi.getStreamUrls(FakeAuth(), "1")) == {}


class TestRequestRetries:
    def test_retries_429_then_succeeds(self, monkeypatch):
        responses = [
            FakeResponse(status_code=429),
            FakeResponse(text="ok", status_code=200),
        ]
        client = FakeClient(responses)
        monkeypatch.setattr(roomApi, "_client", client)
        delays = []

        async def fakeSleep(seconds):
            delays.append(seconds)

        monkeypatch.setattr(roomApi, "_sleepBackoff", fakeSleep)

        resp = asyncio.run(roomApi._request("https://example.com"))

        assert resp.text == "ok"
        assert len(client.calls) == 2
        assert len(delays) == 1

    def test_retries_transport_error_then_succeeds(self, monkeypatch):
        class FlakyClient(FakeClient):
            async def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                if len(self.calls) == 1:
                    raise httpx.ConnectError("boom")
                return FakeResponse(text="ok")

        client = FlakyClient(FakeResponse())
        monkeypatch.setattr(roomApi, "_client", client)

        async def fakeSleep(seconds):
            return None

        monkeypatch.setattr(roomApi, "_sleepBackoff", fakeSleep)

        resp = asyncio.run(roomApi._request("https://example.com"))

        assert resp.text == "ok"
        assert len(client.calls) == 2

    def test_exhausted_5xx_raises(self, monkeypatch):
        client = FakeClient(
            [
                FakeResponse(status_code=503),
                FakeResponse(status_code=503),
                FakeResponse(status_code=503),
            ]
        )
        monkeypatch.setattr(roomApi, "_client", client)

        async def fakeSleep(seconds):
            return None

        monkeypatch.setattr(roomApi, "_sleepBackoff", fakeSleep)

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(roomApi._request("https://example.com"))

        assert len(client.calls) == roomApi.MAX_ATTEMPTS

    def test_retry_delay_is_capped(self):
        delays = [roomApi._retryDelay(attempt) for attempt in range(1, 10)]
        assert all(delay <= roomApi.BACKOFF_MAX_SECS * 1.25 for delay in delays)
        assert delays[-1] > delays[0]


class TestStreamUrlFallback:
    def test_falls_back_to_enter_room(self, monkeypatch):
        client = FakeClient(FakeResponse(text="<html></html>"))
        monkeypatch.setattr(roomApi, "_client", client)

        async def fakeEnter(auth, webRid, roomId=""):
            return {
                "data": {
                    "data": [
                        {
                            "stream_url": {
                                "flv_pull_url": {
                                    "FULL_HD1": "https://pull-flv/x_or4.flv?sign=1"
                                }
                            }
                        }
                    ]
                }
            }

        monkeypatch.setattr(
            roomApi.RoomApi, "enterRoom", staticmethod(fakeEnter)
        )

        result = asyncio.run(roomApi.RoomApi.getStreamUrls(FakeAuth(), "12345"))

        assert set(result) == {"or4"}
        assert "sign=1" in result["or4"]

    def test_page_failure_still_tries_enter_room(self, monkeypatch):
        class DownClient(FakeClient):
            async def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                raise httpx.ConnectError("network down")

        client = DownClient(FakeResponse())
        monkeypatch.setattr(roomApi, "_client", client)

        async def fakeSleep(seconds):
            return None

        async def fakeEnter(auth, webRid, roomId=""):
            return {
                "flv_pull_url": {"HD1": "https://pull-flv/x_hd.flv?sign=2"}
            }

        monkeypatch.setattr(roomApi, "_sleepBackoff", fakeSleep)
        monkeypatch.setattr(
            roomApi.RoomApi, "enterRoom", staticmethod(fakeEnter)
        )

        result = asyncio.run(roomApi.RoomApi.getStreamUrls(FakeAuth(), "12345"))

        assert set(result) == {"hd"}
        assert len(client.calls) == roomApi.MAX_ATTEMPTS

    def test_enter_room_failure_returns_empty(self, monkeypatch):
        client = FakeClient(FakeResponse(text="<html></html>"))
        monkeypatch.setattr(roomApi, "_client", client)

        async def brokenEnter(*args, **kwargs):
            raise RuntimeError("enter failed")

        monkeypatch.setattr(
            roomApi.RoomApi, "enterRoom", staticmethod(brokenEnter)
        )

        assert asyncio.run(roomApi.RoomApi.getStreamUrls(FakeAuth(), "1")) == {}


class TestEnterRoom:
    def test_returns_json(self, monkeypatch):
        payload = {"data": {"room": {"id_str": "1"}}}
        client = FakeClient(FakeResponse(jsonData=payload))
        monkeypatch.setattr(roomApi, "_client", client)
        monkeypatch.setattr("src.params.Params", FakeParams)

        result = asyncio.run(
            roomApi.RoomApi.enterRoom(FakeAuth(cookie={"a": "1"}), "12345", "room-1")
        )
        assert result == payload
        url, kwargs = client.calls[0]
        assert "web_rid=12345" in url
        assert "room_id_str=room-1" in url
        assert kwargs["cookies"] == {"a": "1"}
