# -*- coding: utf-8 -*-
"""Visual signal extraction — scene change + motion + face detection."""

import json
import numpy as np
from src.log.logger import getLogger

log = getLogger(__name__)


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

    from scenedetect import open_video, SceneManager, ContentDetector
    import cv2

    video = open_video(videoPath)

    # ── Scene detection ────────────────────────────────────────
    log.debug("Running scene detection (PySceneDetect)...")
    sceneManager = SceneManager()
    sceneManager.add_detector(ContentDetector(threshold=27.0))
    sceneManager.detect_scenes(video)
    sceneList = sceneManager.get_scene_list()

    # PySceneDetect's video.duration may return None/0 for some FLV files.
    # Fallback to OpenCV frame count / fps when that happens.
    duration = float(video.duration) if video.duration else 0
    if duration <= 0:
        capProbe = cv2.VideoCapture(videoPath)
        frameCount = capProbe.get(cv2.CAP_PROP_FRAME_COUNT)
        fpsProbe = capProbe.get(cv2.CAP_PROP_FPS) or 30
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps == 0:
        fps = 30

    ret, prevFrame = cap.read()
    prevGray = None
    if ret:
        prevGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
        prevGray = cv2.resize(prevGray, (320, 240))  # reduce for speed

    frameCount = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frameCount += 1

        # Sample every 5 frames (1/5 of fps)
        if frameCount % 5 != 0:
            continue

        currGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        currGray = cv2.resize(currGray, (320, 240))

        try:
            flow = cv2.calcOpticalFlowFarneback(
                prevGray, currGray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motionSec = int(frameCount / fps)
            if 0 <= motionSec < nSecs:
                motion[motionSec] = float(np.mean(mag))
        except Exception:
            # Optical flow occasionally fails on bad frames; one bad frame
            # zeroes only that second instead of crashing the whole sweep.
            log.debug(f"Optical flow failed at frame {frameCount}", exc_info=True)
        prevGray = currGray

        if frameCount % 500 == 0 and frameCount > 0:
            log.debug(f"  Processed {frameCount} frames...")

    cap.release()

    # ── Face detection (Phase 2: only on candidate frames) ─────
    faceCount = [0] * nSecs
    if faceDetection:
        log.debug("Skipping face detection in Phase 1 (use --face to enable)")
        # Phase 2: insightface on frames where score > 0.6

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
