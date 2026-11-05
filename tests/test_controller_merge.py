# -*- coding: utf-8 -*-
"""Controller FLV merge tests — validation, quarantine and re-encode fallback."""

import asyncio
import os

import pytest

from src.controller import Controller


@pytest.fixture
def ctrl(tmp_path):
    return Controller("room", outputDir=str(tmp_path))


def _writeSegments(tmp_path, names=("a.flv", "b.flv")):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"flv-data")
        paths.append(str(path))
    return paths


def _acceptAll(path):
    return {"durationSec": 10.0, "hasVideo": True, "hasAudio": True}


class TestSegmentValidation:
    def test_corrupt_segment_is_quarantined_and_excluded(
        self, tmp_path, monkeypatch, ctrl
    ):
        good, bad = _writeSegments(tmp_path)
        monkeypatch.setattr("src.controller.shutil.which", lambda name: "/usr/bin/ffprobe")
        monkeypatch.setattr(
            "src.controller.probeMedia",
            lambda path: _acceptAll(path) if str(path) == good else {},
        )

        result = asyncio.run(ctrl._mergeFlvSegments([good, bad]))

        assert result == good
        assert not os.path.exists(bad)
        assert (tmp_path / "corrupt" / "b.flv").exists()

    def test_all_corrupt_returns_none(self, tmp_path, monkeypatch, ctrl):
        segments = _writeSegments(tmp_path)
        monkeypatch.setattr("src.controller.shutil.which", lambda name: "/usr/bin/ffprobe")
        monkeypatch.setattr("src.controller.probeMedia", lambda path: {})

        result = asyncio.run(ctrl._mergeFlvSegments(segments))

        assert result is None
        assert len(list((tmp_path / "corrupt").glob("*.flv"))) == 2

    def test_missing_files_ignored(self, ctrl, tmp_path):
        assert asyncio.run(ctrl._mergeFlvSegments([str(tmp_path / "nope.flv")])) is None


class TestMergeFallback:
    def test_stream_copy_success_skips_reencode(self, tmp_path, monkeypatch, ctrl):
        segments = _writeSegments(tmp_path)
        monkeypatch.setattr("src.controller.shutil.which", lambda name: "/usr/bin/ffprobe")
        monkeypatch.setattr("src.controller.probeMedia", _acceptAll)
        calls = []

        async def fakeConcat(listPath, outPath, reencode):
            calls.append(reencode)
            with open(outPath, "wb") as fh:
                fh.write(b"merged")
            return True

        monkeypatch.setattr(ctrl, "_concatFfmpeg", fakeConcat)

        result = asyncio.run(ctrl._mergeFlvSegments(segments))

        assert calls == [False]
        assert result is not None and result.endswith("_merged.mp4")
        assert os.path.getsize(result) > 0
        assert not os.path.exists(os.path.join(ctrl.outputDir, ".concat_list.txt"))

    def test_failed_copy_retries_with_reencode(self, tmp_path, monkeypatch, ctrl):
        segments = _writeSegments(tmp_path)
        monkeypatch.setattr("src.controller.shutil.which", lambda name: "/usr/bin/ffprobe")
        monkeypatch.setattr("src.controller.probeMedia", _acceptAll)
        calls = []

        async def fakeConcat(listPath, outPath, reencode):
            calls.append(reencode)
            if not reencode:
                return False
            with open(outPath, "wb") as fh:
                fh.write(b"re-encoded")
            return True

        monkeypatch.setattr(ctrl, "_concatFfmpeg", fakeConcat)

        result = asyncio.run(ctrl._mergeFlvSegments(segments))

        assert calls == [False, True]
        assert result is not None and result.endswith("_merged.mp4")

    def test_both_attempts_failing_returns_none(self, tmp_path, monkeypatch, ctrl):
        segments = _writeSegments(tmp_path)
        monkeypatch.setattr("src.controller.shutil.which", lambda name: "/usr/bin/ffprobe")
        monkeypatch.setattr("src.controller.probeMedia", _acceptAll)

        async def fakeConcat(listPath, outPath, reencode):
            return False

        monkeypatch.setattr(ctrl, "_concatFfmpeg", fakeConcat)

        assert asyncio.run(ctrl._mergeFlvSegments(segments)) is None
