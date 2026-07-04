# -*- coding: utf-8 -*-
"""Unit tests for FlvRecorder — URL fallback, stats, properties."""

import os
import pytest
import asyncio
from src.flvRecorder import FlvRecorder
from src.models import StreamInfo


class TestFlvRecorderInit:
    def test_requires_stream_or_url(self):
        with pytest.raises(ValueError, match="Either"):
            FlvRecorder(roomId="123")

    def test_with_url(self):
        r = FlvRecorder(roomId="123", url="http://example.com/stream.flv")
        assert r.flvUrl == "http://example.com/stream.flv"

    def test_with_stream(self):
        s = StreamInfo(flvUrls={"hd": "http://example.com/hd.flv"})
        r = FlvRecorder(roomId="123", stream=s)
        assert r.flvUrl == "http://example.com/hd.flv"

    def test_direct_url_takes_precedence(self):
        s = StreamInfo(flvUrls={"hd": "http://example.com/hd.flv"})
        r = FlvRecorder(roomId="123", stream=s, url="http://direct/flv")
        assert r.flvUrl == "http://direct/flv"

    def test_updateUrl_replaces_url(self):
        r = FlvRecorder(roomId="123", url="http://old/url")
        r.updateUrl("http://new/url")
        assert r.flvUrl == "http://new/url"

    def test_default_quality(self):
        s = StreamInfo(defaultQuality="or4", flvUrls={"or4": "http://example.com/or4.flv"})
        r = FlvRecorder(roomId="123", stream=s)
        assert r.quality == "or4"

    def test_quality_default_when_stream_none(self):
        r = FlvRecorder(roomId="123", url="http://x.flv")
        assert r.quality == "hd"


class TestFlvUrlFallback:
    def test_missing_quality_falls_back_to_first(self):
        s = StreamInfo(flvUrls={"sd": "http://example.com/sd.flv"})
        r = FlvRecorder(roomId="123", stream=s, quality="hd")
        # hd not in flvUrls → should fall back to sd
        assert r.flvUrl == "http://example.com/sd.flv"

    def test_empty_flvUrls_raises(self):
        s = StreamInfo(flvUrls={})
        r = FlvRecorder(roomId="123", stream=s)
        with pytest.raises(ValueError, match="No FLV URLs"):
            _ = r.flvUrl


class TestStats:
    def test_initial_stats(self):
        r = FlvRecorder(roomId="abc", url="http://x.flv")
        s = r.stats
        assert s["roomId"] == "abc"
        assert s["quality"] == "hd"
        assert s["bytesWritten"] == 0
        assert s["currentFile"] is None
        assert s["downloadedFiles"] == []
        assert s["startedAt"] is None
