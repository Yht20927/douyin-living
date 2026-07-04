# -*- coding: utf-8 -*-
"""Unit tests for data models — StreamInfo, Quality, RoomInfo, DanmakuMessage."""

import pytest
from src.models import StreamInfo, Quality, RoomInfo, RoomStatus, DanmakuMessage


class TestQuality:
    def test_default_is_hd1(self):
        assert Quality.default() == Quality.HD1

    def test_all_qualities_are_strings(self):
        for attr in ("SD2", "SD1", "HD1", "FULL_HD1"):
            assert isinstance(getattr(Quality, attr), str)


class TestStreamInfo:
    def test_flvUrl_returns_correct_quality(self):
        s = StreamInfo(flvUrls={"hd": "http://example.com/hd.flv"})
        assert s.flvUrl("hd") == "http://example.com/hd.flv"

    def test_flvUrl_raises_keyerror_for_missing_quality(self):
        s = StreamInfo(flvUrls={"sd": "http://example.com/sd.flv"})
        with pytest.raises(KeyError):
            s.flvUrl("hd")

    def test_hlsUrl_returns_correct_quality(self):
        s = StreamInfo(hlsUrls={"hd": "http://example.com/hd.m3u8"})
        assert s.hlsUrl("hd") == "http://example.com/hd.m3u8"

    def test_hlsUrl_raises_keyerror_for_missing_quality(self):
        s = StreamInfo(hlsUrls={})
        with pytest.raises(KeyError):
            s.hlsUrl("hd")

    def test_default_quality_is_preserved(self):
        s = StreamInfo(defaultQuality="or4")
        assert s.defaultQuality == "or4"

    def test_empty_streaminfo_has_empty_dicts(self):
        s = StreamInfo()
        assert s.flvUrls == {}
        assert s.hlsUrls == {}


class TestRoomInfo:
    def test_defaults(self):
        r = RoomInfo()
        assert r.status == RoomStatus.UNKNOWN
        assert r.userCount == 0
        assert r.stream is None

    def test_room_status_values(self):
        assert RoomStatus.UNKNOWN == 0
        assert RoomStatus.LIVING == 2
        assert RoomStatus.ENDED == 4


class TestDanmakuMessage:
    def test_defaults(self):
        d = DanmakuMessage()
        assert d.roomId == ""
        assert d.msgType == ""
        assert d.rawPayload is None
        assert d.extra == {}
