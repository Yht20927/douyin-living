# -*- coding: utf-8 -*-
"""Visual signal extraction — scene change + motion + face detection."""

import json
import numpy as np
from src.log.logger import getLogger
from src.config import load_settings

log = getLogger(__name__)
_cfg = load_settings().visual


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
        faceDetection: If True, run insightface on candidate frames (Phase 2).
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
            "generatorVersion": "signalVisual/2",
            "featureCount": 0,
            "features": {"sceneChange": [], "motion": [], "faceCount": []},
        }
        if outputPath:
            with open(outputPath, "w", encoding="utf-8") as f:
                json.dump(empty, f, ensure_ascii=False)
        return empty


def _extractVisualFeaturesImpl(
    videoPath: str,
    outputPath: str | None = None,
    faceDetection: bool = False,
    durationSec: float | None = None,
) -> dict:
    """Inner implementation — wrapped by extractFeatures with try/except."""

    from scenedetect import open_video, SceneManager, ContentDetector
    import cv2

    video = open_video(videoPath)

    # ── Scene detection ────────────────────────────────────────
    log.debug("Running scene detection (PySceneDetect)...")
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
    nSecs = int(np.ceil(duration)) if duration > 0 else 0
    expectedSecs = (
        int(np.ceil(durationSec)) if durationSec and durationSec > 0 else nSecs
    )
    log.debug(f"Duration: {duration:.0f}s, scenes: {len(sceneList)}")

    sceneChange = [0] * nSecs
    for _, frameTimecode in sceneList:
        sec = int(frameTimecode.get_seconds())
        if 0 <= sec < nSecs:
            sceneChange[sec] = 1  # mark only the cut point, not the entire scene

    # ── Motion detection (optical flow, sampled) ───────────────
    log.debug("Computing optical flow motion...")
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

    # ── Face detection (Phase 2: only on candidate frames) ─────
    faceCount = [0] * nSecs
    if faceDetection:
        log.debug("Skipping face detection in Phase 1 (use --face to enable)")
        # Phase 2: insightface on frames where score > threshold

    # ── Align to expected timeline ─────────────────────────────
    featureDict = {
        "sceneChange": _fit1Hz(sceneChange, expectedSecs, 0),
        "motion": _fit1Hz(motion, expectedSecs, motion[-1] if motion else 0.0),
        "faceCount": _fit1Hz(faceCount, expectedSecs, 0),
    }

    # ── Build output ───────────────────────────────────────────
    features = {
        "sampleRate": 1,
        "duration": expectedSecs,
        "status": "ok" if any(featureDict.values()) else "empty",
        "generatorVersion": "signalVisual/2",
        "featureCount": sum(1 for values in featureDict.values() if values),
        "stats": {
            "videoDurationSec": round(float(duration), 3),
            "durationSource": "explicit" if durationSec else "video",
            "scenes": len(sceneList),
        },
        "features": featureDict,
    }

    if outputPath:
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False)
        log.info(f"Visual features saved: {outputPath}")

    log.info(f"Visual features: {expectedSecs}s, {features['featureCount']} non-empty features")
    return features
