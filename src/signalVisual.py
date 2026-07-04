# -*- coding: utf-8 -*-
"""Visual signal extraction — scene change + motion + face detection."""

import json
import numpy as np
from src.log.logger import getLogger
from src.config import load_settings

log = getLogger(__name__)
_cfg = load_settings().visual


def extractFeatures(
    videoPath: str,
    outputPath: str | None = None,
    faceDetection: bool = False,
) -> dict:
    """Extract visual features from a video file.

    Args:
        videoPath: Path to FLV/MP4 video.
        outputPath: If set, save JSON.
        faceDetection: If True, run insightface on candidate frames (Phase 2).

    Returns:
        dict with 'sampleRate', 'duration', 'features' keys.
    """
    log.info(f"Extracting visual features: {videoPath}")
    try:
        return _extractVisualFeaturesImpl(videoPath, outputPath, faceDetection)
    except Exception:
        log.exception(f"Visual feature extraction failed for {videoPath}")
        empty = {
            "sampleRate": 1,
            "duration": 0,
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

    # ── Build output ───────────────────────────────────────────
    features = {
        "sampleRate": 1,
        "duration": nSecs,
        "features": {
            "sceneChange": sceneChange,
            "motion": motion,
            "faceCount": faceCount,
        },
    }

    if outputPath:
        with open(outputPath, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False)
        log.info(f"Visual features saved: {outputPath}")

    log.info(f"Visual features: {nSecs}s, {len(list(features['features'].keys()))} features")
    return features
