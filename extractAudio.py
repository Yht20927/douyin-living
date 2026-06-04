#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract audio track from FLV video file without re-encoding.

Usage:
    python extractAudio.py <input.flv> [output.aac]
    python extractAudio.py data/300294032039/video.flv
    python extractAudio.py video.flv audio.mp3    # re-encode to MP3
"""

import subprocess
import sys
import os

from src.util import fmtSize


def extractAudio(
    inputPath: str,
    outputPath: str | None = None,
    codec: str = "copy",
) -> bool:
    """Extract audio from video file.

    Args:
        inputPath: Path to input video file (FLV, MP4, etc.)
        outputPath: Path to output audio file. If None, auto-generates.
        codec: Audio codec — 'copy' (no re-encode) or 'libmp3lame' (MP3) etc.

    Returns:
        True if extraction succeeded.
    """
    if not os.path.exists(inputPath):
        print(f"✗ Input file not found: {inputPath}")
        return False

    if outputPath is None:
        base = inputPath.rsplit(".", 1)[0]
        outputPath = base + ".aac"

    # Auto-select codec based on output extension
    if outputPath.endswith(".mp3"):
        codec = "libmp3lame"
    elif codec == "copy" and outputPath.endswith(".aac"):
        codec = "copy"  # AAC copy is fine
    elif codec == "copy":
        codec = "copy"

    inputSize = os.path.getsize(inputPath)
    print(f"Input : {inputPath} ({fmtSize(inputSize)})")

    args = ["ffmpeg", "-i", inputPath, "-vn", "-y", "-acodec", codec]
    if codec != "copy":
        args += ["-b:a", "128k"]  # MP3 bitrate
    args.append(outputPath)

    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if proc.returncode == 0 and os.path.exists(outputPath):
            outputSize = os.path.getsize(outputPath)
            print(f"Output: {outputPath} ({fmtSize(outputSize)})")
            print("✓ Done")
            return True
        else:
            err = proc.stderr.decode("utf-8", errors="replace")[-300:]
            print(f"✗ Failed (code={proc.returncode}): {err}")
            return False
    except FileNotFoundError:
        print("✗ ffmpeg not found. Install: https://ffmpeg.org/download.html")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inputFile = sys.argv[1]
    outputFile = sys.argv[2] if len(sys.argv) > 2 else None

    ok = extractAudio(inputFile, outputFile)
    sys.exit(0 if ok else 1)
