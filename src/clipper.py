# -*- coding: utf-8 -*-
"""ffmpeg batch video clipper — reads time_table.json, clips video segments."""

import json
import subprocess
import os
from pathlib import Path
from src.log.logger import getLogger
from src.util import fmtSize, fmtSrtTime

log = getLogger(__name__)


def clipVideo(
    inputPath: str,
    timeTablePath: str,
    outputDir: str,
    dryRun: bool = False,
    resolution: str = "720p",
) -> list[tuple[str, dict]]:
    """Clip video segments per time_table.json.

    Args:
        inputPath: Path to source video (FLV/MP4).
        timeTablePath: Path to time_table.json from scorer.
        outputDir: Output directory for highlight clips.
        dryRun: If True, only print commands, don't execute.
        resolution: Output resolution — '720p', '1080p', or '4k'.
            '720p' (default) uses stream copy; others re-encode with scale.

    Returns:
        List of (outputPath, originalClipDict) tuples.  The original clip
        dict is preserved so callers can correlate files with metadata
        (start, end, trigger, etc.) without relying on fragile index math.
    """
    # Resolution → width:height
    RES_MAP = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
    scaleW, scaleH = RES_MAP.get(resolution, (1280, 720))
    needsScale = resolution != "720p"
    log.info(f"Clipping video: {inputPath}")
    os.makedirs(outputDir, exist_ok=True)

    with open(timeTablePath, encoding="utf-8") as f:
        data = json.load(f)

    outputs: list[tuple[str, dict]] = []
    for clip in data.get("clips", []):
        start = clip["start"]
        end = clip["end"]
        duration = end - start
        if duration < 1.0:
            log.warning(f"Skipping clip < 1s: {start}-{end}")
            continue

        trigger = clip.get("trigger", str(int(start)))
        safeName = "".join(c for c in trigger if c.isalnum() or c in " _-+")[:30]
        name = f"clip_{int(start)}s_{safeName}"
        outPath = os.path.join(outputDir, f"{name}.mp4")

        if needsScale:
            cmd = [
                "ffmpeg", "-ss", str(start), "-i", inputPath,
                "-t", str(duration),
                "-vf", f"scale={scaleW}:{scaleH}:force_original_aspect_ratio=decrease,pad={scaleW}:{scaleH}:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-y", outPath,
            ]
        else:
            cmd = [
                "ffmpeg", "-ss", str(start), "-i", inputPath,
                "-t", str(duration),
                "-c", "copy", "-y", outPath,
            ]

        if dryRun:
            log.info(f"[DRY RUN] {' '.join(cmd)}")
        else:
            try:
                result = subprocess.run(
                    cmd, check=True,
                    capture_output=True, text=True,
                    timeout=300,
                )
            except subprocess.TimeoutExpired:
                log.error(f"Clip timeout {name}: ffmpeg hung for >300s")
                continue
            except subprocess.CalledProcessError as e:
                log.error(f"Clip failed {name}: {e.stderr.strip()[:200] if e.stderr else e}")
                continue

        size = os.path.getsize(outPath) if os.path.exists(outPath) else 0
        log.info(f"✓ {name}.mp4  ({duration:.0f}s, {fmtSize(size)})")
        outputs.append((outPath, clip))

    log.info(f"Clipping done: {len(outputs)} clips → {outputDir}")
    return outputs


def extractThumbnail(
    inputPath: str, timeSec: float, outputPath: str, dryRun: bool = False
):
    """Extract a frame as thumbnail at given time."""
    cmd = [
        "ffmpeg", "-ss", str(timeSec), "-i", inputPath,
        "-vframes", "1", "-q:v", "2", "-y", outputPath,
    ]
    if dryRun:
        log.info(f"[DRY RUN] {' '.join(cmd)}")
    else:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            log.info(f"Thumbnail: {outputPath}")
        except subprocess.TimeoutExpired:
            log.warning(f"Thumbnail timeout: ffmpeg hung for >60s at {timeSec}s")


def burnSubtitles(
    videoPath: str, srtPath: str, outputPath: str, dryRun: bool = False
):
    """Burn SRT subtitles into video file."""
    cmd = [
        "ffmpeg", "-i", videoPath,
        "-vf", f"subtitles={srtPath}",
        "-c:a", "copy", "-y", outputPath,
    ]
    if dryRun:
        log.info(f"[DRY RUN] {' '.join(cmd)}")
    else:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            log.info(f"Subtitles burned: {outputPath}")
        except subprocess.CalledProcessError as e:
            log.warning(f"Subtitle burn failed: {e.stderr.strip()[:200] if e.stderr else 'unknown'}")
        except subprocess.TimeoutExpired:
            log.warning(f"Subtitle burn timeout: ffmpeg hung for >300s")


def generateClipSrt(
    srtPath: str, clipStart: float, clipEnd: float, outputPath: str
):
    """Extract SRT lines within a time range and offset to start at 0."""
    if not os.path.exists(srtPath):
        return
    import re
    with open(srtPath, encoding="utf-8") as f:
        content = f.read()

    # Normalize Windows line endings to Unix
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Parse SRT blocks
    blocks = re.split(r"\n\n+", content.strip())
    outBlocks = []
    counter = 0

    for block in blocks:
        m = re.match(
            r"(\d+)\n(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> "
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\n(.*)",
            block, re.DOTALL,
        )
        if not m:
            continue
        sH, sM, sS, sMs = map(int, m.group(2, 3, 4, 5))
        eH, eM, eS, eMs = map(int, m.group(6, 7, 8, 9))
        text = m.group(10).strip()
        startSec = sH * 3600 + sM * 60 + sS + sMs / 1000.0
        endSec = eH * 3600 + eM * 60 + eS + eMs / 1000.0

        # Keep subtitles that overlap with the clip, not just fully-contained.
        # Truncate to clip boundaries so partial entries are still visible.
        if endSec < clipStart or startSec > clipEnd:
            continue
        counter += 1
        newStart = max(0.0, startSec - clipStart)
        newEnd = min(clipEnd - clipStart, endSec - clipStart)

        outBlocks.append(f"{counter}\n{fmtSrtTime(newStart)} --> {fmtSrtTime(newEnd)}\n{text}")

    if outBlocks:
        with open(outputPath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(outBlocks))
        log.debug(f"Clip SRT: {outputPath} ({counter} entries)")



