# -*- coding: utf-8 -*-
"""Unit tests for clipper.generateClipSrt — pure-text SRT slicing.

Other clipper functions (clipVideo, extractThumbnail, burnSubtitles) shell
out to ffmpeg and are integration-level; we cover them only by syntactic
checks elsewhere.
"""

import os
import tempfile
import pytest

from src.clipper import generateClipSrt


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,000
hello world

2
00:00:05,500 --> 00:00:08,000
mid clip

3
00:00:10,000 --> 00:00:13,000
late clip
"""


def _writeSrt(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".srt")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


class TestGenerateClipSrt:
    def test_extracts_within_range(self):
        srt = _writeSrt(SAMPLE_SRT)
        out = srt + ".out"
        try:
            generateClipSrt(srt, clipStart=5.0, clipEnd=9.0, outputPath=out)
            assert os.path.exists(out)
            with open(out, encoding="utf-8") as fh:
                body = fh.read()
            # Only the middle entry should survive
            assert "mid clip" in body
            assert "hello world" not in body
            assert "late clip" not in body
        finally:
            os.remove(srt)
            if os.path.exists(out):
                os.remove(out)

    def test_offsets_to_zero(self):
        srt = _writeSrt(SAMPLE_SRT)
        out = srt + ".out"
        try:
            generateClipSrt(srt, clipStart=5.0, clipEnd=9.0, outputPath=out)
            with open(out, encoding="utf-8") as fh:
                body = fh.read()
            # mid clip was 5.5–8.0; with clipStart=5.0 → 0.5–3.0
            assert "00:00:00,500 --> 00:00:03,000" in body
        finally:
            os.remove(srt)
            if os.path.exists(out):
                os.remove(out)

    def test_no_overlap_writes_nothing(self):
        srt = _writeSrt(SAMPLE_SRT)
        out = srt + ".out"
        try:
            # Range that contains no entries
            generateClipSrt(srt, clipStart=100.0, clipEnd=200.0, outputPath=out)
            # No file should have been created
            assert not os.path.exists(out)
        finally:
            os.remove(srt)
            if os.path.exists(out):
                os.remove(out)

    def test_missing_input_is_noop(self):
        # Non-existent SRT → silent no-op (function returns None,
        # never raises).  This protects callers that always invoke it.
        out = tempfile.mkstemp(suffix=".srt")[1]
        os.remove(out)  # delete the placeholder so we can detect re-creation
        generateClipSrt("/nonexistent/path.srt", 0.0, 10.0, out)
        assert not os.path.exists(out)

    def test_partial_overlap_truncated(self):
        # Partial overlap is now kept and truncated to clip boundaries.
        srt = _writeSrt(SAMPLE_SRT)
        out = srt + ".out"
        try:
            # 6.0–7.5 partially overlaps mid clip (5.5–8.0)
            generateClipSrt(srt, clipStart=6.0, clipEnd=7.5, outputPath=out)
            assert os.path.exists(out)
            with open(out, encoding="utf-8") as fh:
                body = fh.read()
            # Truncated to clip boundaries: 6.0→0.0, 7.5→1.5
            assert "00:00:00,000 --> 00:00:01,500" in body
            assert "mid clip" in body
        finally:
            os.remove(srt)
            if os.path.exists(out):
                os.remove(out)
