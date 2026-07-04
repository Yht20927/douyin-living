# -*- coding: utf-8 -*-
"""Speech-to-text with timestamps and speaker diarization.

Uses faster-whisper for transcription + WhisperX for speaker identification.

Usage:
    python asr.py audio.aac
    python asr.py audio.aac --diarize
    python asr.py audio.aac --model tiny --device cpu
"""

import argparse
import json
import sys
import time

from src.log.logger import getLogger
from src.util import fmtSrtTime

log = getLogger(__name__)


def transcribe(
    audioPath: str,
    modelSize: str = "base",
    language: str | None = "zh",
    device: str = "cuda",
    computeType: str = "int8",
    diarize: bool = True,
    outputPath: str | None = None,
) -> list[dict]:
    """Transcribe audio with timestamps and optional speaker labels.

    Args:
        audioPath: Path to audio file (AAC/MP3/WAV).
        modelSize: 'tiny'|'base'|'small'|'medium'|'large-v3'.
        language: Language code or None for auto.
        device: 'cuda' or 'cpu'.
        computeType: 'int8'|'float16'|'auto'.
        diarize: If True, run WhisperX speaker diarization.
        outputPath: Save JSON + SRT.

    Returns:
        List of segment dicts with start, end, text, speaker (if diarized).
    """
    from faster_whisper import WhisperModel

    log.info(f"Loading ASR model: {modelSize} ({device}/{computeType})")
    t0 = time.time()

    model = WhisperModel(modelSize, device=device, compute_type=computeType)
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    log.info(f"Transcribing: {audioPath}")
    t1 = time.time()

    segments, info = model.transcribe(
        audioPath,
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    segmentsList = list(segments)  # materialize
    elapsed = time.time() - t1
    log.info(f"Transcribed: {len(segmentsList)} segments in {elapsed:.1f}s")
    log.info(f"Detected language: {info.language} (p={info.language_probability:.2f})")

    # ── WhisperX diarization ────────────────────────────────────
    if diarize:
        log.info("Running speaker diarization...")
        try:
            import whisperx
            # Re-use the transcribed data
            asrResult = {
                "segments": [],
                "language": info.language or language or "zh",
            }
            for seg in segmentsList:
                asrResult["segments"].append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                })

            # Align model for word-level timestamps (required by WhisperX)
            alignModel, alignMeta = whisperx.load_align_model(
                language_code=info.language or "zh", device=device
            )
            asrResult = whisperx.align(
                asrResult["segments"], alignModel, alignMeta, audioPath, device=device
            )

            # Diarize
            diarizeModel = whisperx.DiarizationPipeline(
                use_auth_token=None, device=device
            )
            diarizeSegments = diarizeModel(audioPath, batch_size=32)

            # Assign speakers
            asrResult = whisperx.assign_word_speakers(diarizeSegments, asrResult)
            segmentsList = asrResult["segments"]

            del alignModel, diarizeModel
            import torch
            torch.cuda.empty_cache()

            log.info("Diarization done")
        except ImportError:
            log.warning("whisperx not installed — falling back to mono transcription")
        except Exception:
            log.warning("Diarization failed — proceeding without speaker labels", exc_info=True)

    # ── Normalize segmentsList to dicts (faster-whisper → objects, WhisperX → dicts)
    normalized: list[dict] = []
    for seg in segmentsList:
        if isinstance(seg, dict):
            normalized.append(seg)
        else:
            # faster-whisper Segment object
            entry: dict = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            # Extract speaker from word-level data if available
            if hasattr(seg, "words") and seg.words:
                speaker = seg.words[0].get("speaker") if isinstance(seg.words[0], dict) else None
                if speaker:
                    entry["speaker"] = speaker
            normalized.append(entry)

    # ── Build result ──────────────────────────────────────────
    results = []
    for seg in normalized:
        entry = {
            "start": round(seg.get("start", 0), 1),
            "end": round(seg.get("end", 0), 1),
            "text": seg.get("text", ""),
        }
        speaker = seg.get("speaker")
        if speaker:
            entry["speaker"] = speaker
        results.append(entry)

    if outputPath:
        # Save JSON
        jsonPath = outputPath if outputPath.endswith(".json") else outputPath + ".json"
        with open(jsonPath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info(f"ASR JSON saved: {jsonPath}")

        # Save SRT
        srtPath = jsonPath.replace(".json", ".srt")
        with open(srtPath, "w", encoding="utf-8") as f:
            for i, seg in enumerate(results, 1):
                speaker = seg.get("speaker", "")
                text = f"[{speaker}] {seg['text']}" if speaker else seg["text"]
                f.write(f"{i}\n")
                f.write(f"{fmtSrtTime(seg['start'])} --> {fmtSrtTime(seg['end'])}\n")
                f.write(f"{text}\n\n")
        log.info(f"SRT saved: {srtPath}")

    return results





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech-to-text with diarization")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--output", help="Output path (without extension)")
    args = parser.parse_args()

    results = transcribe(
        audioPath=args.audio,
        modelSize=args.model,
        language=args.language,
        device=args.device,
        computeType=args.compute_type,
        diarize=not args.no_diarize,
        outputPath=args.output,
    )

    for seg in results[:10]:
        speaker = seg.get("speaker", "")
        prefix = f"[{speaker}] " if speaker else ""
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {prefix}{seg['text']}")

    if len(results) > 10:
        print(f"... ({len(results) - 10} more)")

    print(f"\nTotal: {len(results)} segments")
