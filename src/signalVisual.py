# -*- coding: utf-8 -*-
"""Visual signal extraction — proxy-first scene change + motion + face slot.

Sprint 5 / S5-4 reduces source decoding from two passes to one:

1. ffmpeg transcodes the source into a low-fps, low-height proxy;
2. a single OpenCV pass over that proxy computes both scene-cut scores
   (HSV frame difference) and optical-flow motion.

When ffmpeg is unavailable or fails, the legacy PySceneDetect + OpenCV
double-pass path is used so extraction still succeeds.
"""

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.log.logger import getLogger
from src.config import load_settings

log = getLogger(__name__)
_cfg = load_settings().visual

VISUAL_GENERATOR_VERSION = "signalVisual/3"


def _fit1Hz(values: list, target: int, padValue: float = 0.0) -> list:
    """Fit a 1 Hz feature array to an expected timeline length."""
    if target <= 0 or len(values) == target:
        return values
    if len(values) > target:
        return values[:target]
    return values + [padValue] * (target - len(values))


def extractFeatures(
    videoPath: str,
    outputPath: str | None = None,
    faceDetection: bool = False,
    durationSec: float | None = None,
) -> dict:
    """Extract visual features from a video file.

    Args:
        videoPath: Path to FLV/MP4 video.
        outputPath: If set, save JSON.
        faceDetection: If True, request face detection on candidate frames.
        durationSec: Authoritative timeline length from the controller.  When
            provided, arrays are fitted to this length so all signals share a
            single timeline.

    Returns:
        dict with 'sampleRate', 'duration', 'features', 'status' keys.
    """
    log.info(f"Extracting visual features: {videoPath}")
    try:
        return _extractVisualFeaturesImpl(
            videoPath, outputPath, faceDetection, durationSec
        )
    except Exception as exc:
        log.exception(f"Visual feature extraction failed for {videoPath}")
        empty = {
            "sampleRate": 1,
            "duration": 0,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "generatorVersion": VISUAL_GENERATOR_VERSION,
            "featureCount": 0,
            "features": {"sceneChange": [], "motion": [], "faceCount": []},
        }
        if outputPath:
            with open(outputPath, "w", encoding="utf-8") as f:
                json.dump(empty, f, ensure_ascii=False)
        return empty


def _probeVideo(videoPath: str) -> dict:
    """Read duration/geometry without decoding frames (best effort)."""
    import cv2

    cap = cv2.VideoCapture(videoPath)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frameCount = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if fps <= 0:
        fps = float(_cfg.fps_fallback)
    duration = frameCount / fps if frameCount > 0 and fps > 0 else 0.0
    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "frameCount": frameCount,
    }


def _buildProxy(
    videoPath: str,
    sourceHeight: int = 0,
    sourceFps: float = 0.0,
    proxyHeight: int | None = None,
    sampleFps: float | None = None,
    timeoutSec: int | None = None,
) -> tuple[str | None, dict]:
    """Transcode a low-fps, low-height proxy used for visual analysis.

    Returns ``(proxyPath, info)``.  ``proxyPath`` is None when the source is
    missing or ffmpeg fails; callers then fall back to the legacy path.
    """
    if not Path(videoPath).exists():
        return None, {"error": "input file not found"}

    proxyHeight = int(_cfg.proxy_height if proxyHeight is None else proxyHeight)
    sampleFps = float(_cfg.sample_fps if sampleFps is None else sampleFps)
    timeoutSec = int(_cfg.proxy_timeout_sec if timeoutSec is None else timeoutSec)

    targetHeight = proxyHeight
    if sourceHeight > 0:
        targetHeight = min(proxyHeight, sourceHeight)
    targetHeight = max(2, targetHeight - (targetHeight % 2))

    targetFps = sampleFps
    if sourceFps > 0:
        targetFps = min(sampleFps, sourceFps)
    targetFps = max(0.1, targetFps)

    try:
        fd, proxyPath = tempfile.mkstemp(prefix="dyci-visual-", suffix=".mp4")
        os.close(fd)
    except OSError as exc:
        return None, {"error": f"proxy temp file failed: {exc}"}

    videoFilter = f"fps={targetFps:g},scale=-2:{targetHeight}"
    baseCmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", videoPath,
        "-an", "-sn",
        "-map", "0:v:0",
        "-vf", videoFilter,
    ]
    attempts = [
        ("libx264", baseCmd + [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p", proxyPath,
        ]),
        ("mpeg4", baseCmd + [
            "-c:v", "mpeg4", "-q:v", "5", proxyPath,
        ]),
    ]

    errors: list[str] = []
    try:
        for encoder, command in attempts:
            try:
                if os.path.exists(proxyPath):
                    os.remove(proxyPath)
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeoutSec,
                    check=False,
                )
            except FileNotFoundError:
                if os.path.exists(proxyPath):
                    os.remove(proxyPath)
                return None, {"error": "ffmpeg not installed"}
            except subprocess.TimeoutExpired:
                errors.append(f"{encoder}: timeout after {timeoutSec}s")
                break
            if proc.returncode == 0 and os.path.exists(proxyPath):
                if os.path.getsize(proxyPath) > 0:
                    return proxyPath, {
                        "encoder": encoder,
                        "height": targetHeight,
                        "sampleFps": targetFps,
                        "videoFilter": videoFilter,
                    }
            stderr = (proc.stderr or "").strip()[-300:]
            errors.append(f"{encoder}: {stderr or 'unknown ffmpeg error'}")
            # A bad input path will fail for every encoder — don't retry.
            if "No such file" in stderr or "Invalid data" in stderr:
                break
    finally:
        if errors and os.path.exists(proxyPath):
            os.remove(proxyPath)

    return None, {"error": "; ".join(errors) or "proxy generation failed"}


def _sceneChangeScore(previousHsv: np.ndarray, currentHsv: np.ndarray) -> float:
    """ContentDetector-style HSV difference, in the 0..765 range.

    PySceneDetect's default threshold is 27; keeping the same score scale lets
    the existing ``visual.scene_threshold`` config remain meaningful.
    """
    previous = previousHsv.astype(np.int16)
    current = currentHsv.astype(np.int16)
    rawHue = np.abs(current[..., 0] - previous[..., 0])
    # OpenCV hue is 0..179: red sits at both ends, so measure the shorter arc.
    deltaHue = float(np.mean(np.minimum(rawHue, 180 - rawHue)))
    deltaSat = float(np.mean(np.abs(current[..., 1] - previous[..., 1])))
    deltaVal = float(np.mean(np.abs(current[..., 2] - previous[..., 2])))
    return deltaHue + deltaSat + deltaVal


def _extractProxySignals(proxyPath: str, expectedSecs: int = 0) -> dict:
    """Single-pass scene + motion extraction over the low-fps proxy."""
    import cv2

    cap = cv2.VideoCapture(proxyPath)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = float(_cfg.sample_fps)
    fps = max(fps, 0.1)

    resW, resH = _cfg.opticalflow_resolution
    motionSamples: dict[int, list[float]] = {}
    sceneCuts: set[int] = set()
    previousGray = None
    previousHsv = None
    lastCutTime = -1e9
    frameIndex = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (resW, resH))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            frameTime = frameIndex / fps
            sec = int(frameTime)

            if previousGray is not None and previousHsv is not None:
                score = _sceneChangeScore(previousHsv, hsv)
                if (
                    score >= _cfg.scene_threshold
                    and frameTime - lastCutTime >= _cfg.scene_min_gap_sec
                ):
                    sceneCuts.add(sec)
                    lastCutTime = frameTime

                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        previousGray, gray, None,
                        _cfg.farneback_pyr_scale,
                        _cfg.farneback_levels,
                        _cfg.farneback_winsize,
                        _cfg.farneback_iterations,
                        _cfg.farneback_poly_n,
                        _cfg.farneback_poly_sigma,
                        _cfg.farneback_flags,
                    )
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motionSamples.setdefault(sec, []).append(float(np.mean(magnitude)))
                except Exception:
                    log.debug(
                        f"Optical flow failed at proxy frame {frameIndex}",
                        exc_info=True,
                    )

            previousGray = gray
            previousHsv = hsv
            frameIndex += 1
    finally:
        cap.release()

    proxyDuration = frameIndex / fps
    nSecs = expectedSecs if expectedSecs > 0 else int(math.ceil(proxyDuration))
    sceneChange = [0] * nSecs
    for sec in sceneCuts:
        if 0 <= sec < nSecs:
            sceneChange[sec] = 1

    motion = [0.0] * nSecs
    for sec, samples in motionSamples.items():
        if 0 <= sec < nSecs and samples:
            motion[sec] = sum(samples) / len(samples)

    return {
        "sceneChange": sceneChange,
        "motion": motion,
        "faceCount": [0] * nSecs,
        "duration": nSecs,
        "proxyDuration": proxyDuration,
        "sceneCount": len(sceneCuts),
        "frames": frameIndex,
        "fps": fps,
    }


def _extractLegacySignals(videoPath: str, expectedSecs: int = 0) -> dict:
    """Legacy double-decode path: PySceneDetect + per-frame OpenCV flow."""
    from scenedetect import open_video, SceneManager, ContentDetector
    import cv2

    video = open_video(videoPath)

    # ── Scene detection ────────────────────────────────────────
    log.debug("Running scene detection (PySceneDetect, legacy path)...")
    sceneManager = SceneManager()
    sceneManager.add_detector(ContentDetector(threshold=_cfg.scene_threshold))
    sceneManager.detect_scenes(video)
    sceneList = sceneManager.get_scene_list()

    # PySceneDetect's video.duration may return None/0 for some FLV files.
    # Fallback to OpenCV frame count / fps when that happens.
    duration = float(video.duration) if video.duration else 0
    if duration <= 0:
        capProbe = cv2.VideoCapture(videoPath)
        frameCount = capProbe.get(cv2.CAP_PROP_FRAME_COUNT)
        fpsProbe = capProbe.get(cv2.CAP_PROP_FPS) or _cfg.fps_fallback
        capProbe.release()
        if frameCount > 0 and fpsProbe > 0:
            duration = frameCount / fpsProbe
            log.debug(f"Duration from OpenCV: {duration:.0f}s")
    nSecs = int(math.ceil(duration)) if duration > 0 else 0
    if expectedSecs > 0:
        nSecs = expectedSecs
    log.debug(f"Duration: {duration:.0f}s, scenes: {len(sceneList)}")

    sceneChange = [0] * nSecs
    for _, frameTimecode in sceneList:
        sec = int(frameTimecode.get_seconds())
        if 0 <= sec < nSecs:
            sceneChange[sec] = 1  # mark only the cut point, not the entire scene

    # ── Motion detection (optical flow, sampled) ───────────────
    log.debug("Computing optical flow motion (legacy path)...")
    cap = cv2.VideoCapture(videoPath)
    motion = [0.0] * nSecs
    fps = cap.get(cv2.CAP_PROP_FPS) or _cfg.fps_fallback
    if fps == 0:
        fps = _cfg.fps_fallback

    resW, resH = _cfg.opticalflow_resolution
    sampleInterval = _cfg.opticalflow_sample_interval

    ret, prevFrame = cap.read()
    prevGray = None
    if ret:
        prevGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
        prevGray = cv2.resize(prevGray, (resW, resH))  # reduce for speed

    # Accumulate motion magnitude per second (multiple samples per second
    # are averaged instead of overwriting each other).
    motionSamples: dict[int, list[float]] = {}

    frameCount = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frameCount += 1

        # Sample at configured interval
        if frameCount % sampleInterval != 0:
            continue

        currGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        currGray = cv2.resize(currGray, (resW, resH))

        try:
            flow = cv2.calcOpticalFlowFarneback(
                prevGray, currGray, None,
                _cfg.farneback_pyr_scale,
                _cfg.farneback_levels,
                _cfg.farneback_winsize,
                _cfg.farneback_iterations,
                _cfg.farneback_poly_n,
                _cfg.farneback_poly_sigma,
                _cfg.farneback_flags,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motionSec = int(frameCount / fps)
            if 0 <= motionSec < nSecs:
                motionSamples.setdefault(motionSec, []).append(float(np.mean(mag)))
        except Exception:
            # Optical flow occasionally fails on bad frames; one bad frame
            # zeroes only that second instead of crashing the whole sweep.
            log.debug(f"Optical flow failed at frame {frameCount}", exc_info=True)
        prevGray = currGray

        if frameCount % 500 == 0 and frameCount > 0:
            log.debug(f"  Processed {frameCount} frames...")

    cap.release()

    # Average accumulated samples per second
    for sec, samples in motionSamples.items():
        motion[sec] = sum(samples) / len(samples)

    return {
        "sceneChange": sceneChange,
        "motion": motion,
        "faceCount": [0] * nSecs,
        "duration": nSecs,
        "videoDuration": duration,
        "sceneCount": len(sceneList),
        "frames": frameCount,
        "fps": fps,
    }


def _extractVisualFeaturesImpl(
    videoPath: str,
    outputPath: str | None = None,
    faceDetection: bool = False,
    durationSec: float | None = None,
) -> dict:
    """Inner implementation — wrapped by extractFeatures with try/except."""
    probe = _probeVideo(videoPath)
    duration = float(probe["duration"])
    if durationSec and durationSec > 0:
        duration = float(durationSec)
    expectedSecs = int(math.ceil(duration)) if duration > 0 else 0

    requestedFace = bool(faceDetection or _cfg.enable_face)
    mode = "legacy"
    fallbackReason: str | None = None
    proxyInfo: dict = {}
    proxyPath: str | None = None
    result: dict | None = None

    if _cfg.proxy_enabled:
        proxyPath, proxyInfo = _buildProxy(
            videoPath,
            sourceHeight=probe["height"],
            sourceFps=probe["fps"],
        )
        if proxyPath:
            try:
                candidate = _extractProxySignals(proxyPath, expectedSecs)
                if candidate["frames"] <= 0:
                    raise RuntimeError("proxy produced no decodable frames")
                result = candidate
                mode = "proxy"
            except Exception as exc:
                fallbackReason = (
                    f"proxy decode failed: {type(exc).__name__}: {exc}"
                )
                log.warning(
                    f"{fallbackReason} — falling back to double-decode path"
                )
            finally:
                try:
                    os.remove(proxyPath)
                except OSError:
                    pass
        else:
            fallbackReason = str(proxyInfo.get("error") or "proxy unavailable")
            log.warning(
                f"Visual proxy unavailable ({fallbackReason}) — "
                "falling back to double-decode path"
            )

    if result is None:
        result = _extractLegacySignals(videoPath, expectedSecs)

    expectedSecs = result["duration"]
    featureDict = {
        "sceneChange": _fit1Hz(result["sceneChange"], expectedSecs, 0),
        "motion": _fit1Hz(
            result["motion"],
            expectedSecs,
            result["motion"][-1] if result["motion"] else 0.0,
        ),
        "faceCount": _fit1Hz(result["faceCount"], expectedSecs, 0),
    }

    if requestedFace:
        log.debug("Face detection is deferred to candidate-window keyframes")

    # ── Build output ───────────────────────────────────────────
    features = {
        "sampleRate": 1,
        "duration": expectedSecs,
        "status": "ok" if any(featureDict.values()) else "empty",
        "generatorVersion": VISUAL_GENERATOR_VERSION,
        "featureCount": sum(1 for values in featureDict.values() if values),
        "stats": {
            "videoDurationSec": round(float(duration), 3),
            "durationSource": "explicit" if durationSec else "video",
            "scenes": result["sceneCount"],
            "mode": mode,
            "sceneDetector": "hsv-diff" if mode == "proxy" else "pyscenedetect",
            "sourceDecodePasses": 1 if mode == "proxy" else 2,
            "proxy": {
                "requested": bool(_cfg.proxy_enabled),
                "enabled": mode == "proxy",
                "height": (
                    int(proxyInfo["height"]) if proxyInfo.get("height") else None
                ),
                "sampleFps": (
                    round(float(proxyInfo["sampleFps"]), 3)
                    if proxyInfo.get("sampleFps")
                    else None
                ),
                "encoder": proxyInfo.get("encoder"),
                "fallbackReason": fallbackReason,
            },
            "faceDetectionRequested": requestedFace,
            "faces": 0,
        },
        "features": featureDict,
    }

    if outputPath:
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False)
        log.info(f"Visual features saved: {outputPath}")

    log.info(
        f"Visual features ({mode}): {expectedSecs}s, "
        f"{features['featureCount']} non-empty features"
    )
    return features
