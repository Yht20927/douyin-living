# -*- coding: utf-8 -*-
"""Speech-to-text with timestamps and speaker diarization.

Uses faster-whisper for transcription + WhisperX for speaker identification.

S5-5 adds:
- ``config/asr.yaml`` runtime settings (auto model selection, VAD, batch size);
- 30-minute chunking with a resumable checkpoint for long recordings;
- graceful degradation: a failed run leaves ``status=failed`` for the text
  feature extractor, which then falls back to danmaku-only signals.

Usage:
    python asr.py audio.aac
    python asr.py audio.aac --no-diarize
    python asr.py audio.aac --model tiny --device cpu
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.asrRuntime import resolveAsrRuntime
from src.contracts import atomicWriteJson, probeMedia
from src.log.logger import getLogger
from src.util import fmtSrtTime

log = getLogger(__name__)

_CHECKPOINT_VERSION = 1

__all__ = ["resolveAsrRuntime", "transcribe"]


def _loadModel(modelSize: str, device: str, computeType: str):
    from faster_whisper import WhisperModel

    log.info(f"Loading ASR model: {modelSize} ({device}/{computeType})")
    startedAt = time.time()
    model = WhisperModel(modelSize, device=device, compute_type=computeType)
    log.info(f"Model loaded in {time.time() - startedAt:.1f}s")
    return model


def _diarize(
    audioPath: str,
    segmentsList: list,
    language: str | None,
    runtime: dict,
) -> list:
    """Run WhisperX alignment + diarization, returning the original list on failure."""
    log.info("Running speaker diarization...")
    try:
        import whisperx

        asrResult = {
            "segments": [],
            "language": language or "zh",
        }
        for seg in segmentsList:
            if isinstance(seg, dict):
                asrResult["segments"].append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", ""),
                })
            else:
                asrResult["segments"].append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                })

        alignModel, alignMeta = whisperx.load_align_model(
            language_code=language or "zh", device=runtime["device"]
        )
        asrResult = whisperx.align(
            asrResult["segments"], alignModel, alignMeta, audioPath,
            device=runtime["device"],
        )

        diarizeModel = whisperx.DiarizationPipeline(
            use_auth_token=None, device=runtime["device"]
        )
        diarizeSegments = diarizeModel(
            audioPath, batch_size=runtime["batchSize"]
        )

        asrResult = whisperx.assign_word_speakers(diarizeSegments, asrResult)
        segmentsList = asrResult["segments"]

        del alignModel, diarizeModel
        if runtime["device"] == "cuda":
            import torch
            torch.cuda.empty_cache()

        log.info("Diarization done")
    except ImportError:
        log.warning("whisperx not installed — falling back to mono transcription")
    except Exception:
        log.warning(
            "Diarization failed — proceeding without speaker labels",
            exc_info=True,
        )
    return segmentsList


def _normalizeSegments(segmentsList: list) -> list[dict]:
    """Normalize faster-whisper objects / WhisperX dicts into plain dicts."""
    normalized: list[dict] = []
    for seg in segmentsList:
        if isinstance(seg, dict):
            normalized.append(seg)
        else:
            entry: dict = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            if hasattr(seg, "words") and seg.words:
                word = seg.words[0]
                speaker = word.get("speaker") if isinstance(word, dict) else None
                if speaker:
                    entry["speaker"] = speaker
            normalized.append(entry)

    results = []
    for seg in normalized:
        entry = {
            "start": round(float(seg.get("start", 0)), 1),
            "end": round(float(seg.get("end", 0)), 1),
            "text": seg.get("text", ""),
        }
        speaker = seg.get("speaker")
        if speaker:
            entry["speaker"] = speaker
        results.append(entry)
    return results


def _transcribeOnce(audioPath: str, runtime: dict, model=None):
    """Transcribe one file/chunk; returns (segments, info, loadedModel)."""
    if model is None:
        model = _loadModel(
            runtime["model"], runtime["device"], runtime["computeType"]
        )

    log.info(f"Transcribing: {audioPath}")
    startedAt = time.time()
    vadParameters = None
    if runtime["vadFilter"]:
        vadParameters = {
            "min_silence_duration_ms": runtime["vadMinSilenceMs"],
            "speech_pad_ms": runtime["vadSpeechPadMs"],
        }
    segments, info = model.transcribe(
        audioPath,
        language=runtime["language"],
        vad_filter=runtime["vadFilter"],
        vad_parameters=vadParameters,
        beam_size=runtime["beamSize"],
    )

    segmentsList = list(segments)
    log.info(
        f"Transcribed: {len(segmentsList)} segments in "
        f"{time.time() - startedAt:.1f}s"
    )
    if getattr(info, "language", None):
        probability = getattr(info, "language_probability", 0.0) or 0.0
        log.info(
            f"Detected language: {info.language} (p={probability:.2f})"
        )

    if runtime["diarize"]:
        language = getattr(info, "language", None) or runtime["language"]
        segmentsList = _diarize(audioPath, segmentsList, language, runtime)

    return _normalizeSegments(segmentsList), info, model


def _writeOutput(results: list[dict], outputPath: str | None) -> None:
    if not outputPath:
        return
    jsonPath = outputPath if outputPath.endswith(".json") else outputPath + ".json"
    with open(jsonPath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"ASR JSON saved: {jsonPath}")

    srtPath = jsonPath.replace(".json", ".srt")
    with open(srtPath, "w", encoding="utf-8") as f:
        for index, seg in enumerate(results, 1):
            speaker = seg.get("speaker", "")
            text = f"[{speaker}] {seg['text']}" if speaker else seg["text"]
            f.write(f"{index}\n")
            f.write(f"{fmtSrtTime(seg['start'])} --> {fmtSrtTime(seg['end'])}\n")
            f.write(f"{text}\n\n")
    log.info(f"SRT saved: {srtPath}")


def _checkpointPath(outputPath: str) -> str:
    jsonPath = outputPath if outputPath.endswith(".json") else outputPath + ".json"
    return jsonPath[:-5] + ".chunks.json"


def _audioSignature(audioPath: str, runtime: dict) -> dict:
    stat = os.stat(audioPath)
    return {
        "version": _CHECKPOINT_VERSION,
        "path": str(Path(audioPath).resolve()),
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "chunkSec": runtime["chunkSec"],
        "model": runtime["model"],
        "device": runtime["device"],
        "computeType": runtime["computeType"],
        "language": runtime["language"],
        "diarize": runtime["diarize"],
        "batchSize": runtime["batchSize"],
        "vadFilter": runtime["vadFilter"],
        "vadMinSilenceMs": runtime["vadMinSilenceMs"],
        "vadSpeechPadMs": runtime["vadSpeechPadMs"],
        "beamSize": runtime["beamSize"],
    }


def _loadCheckpoint(outputPath: str | None, signature: dict) -> dict:
    if not outputPath:
        return {}
    path = _checkpointPath(outputPath)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("signature") != signature:
        log.info(f"ASR checkpoint ignored (settings/audio changed): {path}")
        return {}
    completed = data.get("completed")
    if not isinstance(completed, dict):
        return {}
    log.info(f"ASR checkpoint loaded: {len(completed)} completed chunk(s)")
    return {str(key): value for key, value in completed.items()}


def _saveCheckpoint(outputPath: str | None, signature: dict, completed: dict) -> None:
    if not outputPath:
        return
    atomicWriteJson(
        _checkpointPath(outputPath),
        {"signature": signature, "completed": completed},
    )


def _removeCheckpoint(outputPath: str | None) -> None:
    if not outputPath:
        return
    try:
        os.remove(_checkpointPath(outputPath))
    except OSError:
        pass


def _splitAudio(audioPath: str, chunkSec: float, workDir: str) -> list[str]:
    """Split audio into mono 16 kHz AAC chunks; returns paths in order."""
    pattern = str(Path(workDir) / "chunk_%04d.m4a")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", audioPath,
        "-vn", "-sn", "-ac", "1", "-ar", "16000",
        "-c:a", "aac", "-b:a", "64k",
        "-f", "segment",
        "-segment_time", f"{chunkSec:g}",
        "-reset_timestamps", "1",
        pattern,
    ]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=900, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not installed — cannot split ASR audio") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while splitting ASR audio") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()[-300:]
        raise RuntimeError(f"ffmpeg chunking failed: {detail or 'unknown error'}")

    chunks = sorted(Path(workDir).glob("chunk_*.m4a"))
    if not chunks:
        raise RuntimeError("ffmpeg produced no ASR chunks")
    return [str(path) for path in chunks]


def _chunkOffsets(chunks: list[str], fallbackSec: float) -> list[float]:
    """Cumulative start time for each chunk, using real durations when possible."""
    offsets: list[float] = []
    cursor = 0.0
    for chunk in chunks:
        offsets.append(cursor)
        duration = float(probeMedia(chunk).get("durationSec") or 0.0)
        cursor += duration if duration > 0 else fallbackSec
    return offsets


def _offsetSegments(segments: list[dict], offsetSec: float) -> list[dict]:
    if offsetSec <= 0:
        return segments
    shifted = []
    for seg in segments:
        entry = dict(seg)
        entry["start"] = round(float(entry.get("start", 0.0)) + offsetSec, 1)
        entry["end"] = round(float(entry.get("end", 0.0)) + offsetSec, 1)
        shifted.append(entry)
    return shifted


def _transcribeChunked(
    audioPath: str,
    runtime: dict,
    outputPath: str | None,
) -> list[dict]:
    signature = _audioSignature(audioPath, runtime)
    completed = _loadCheckpoint(outputPath, signature) if runtime["resume"] else {}

    with tempfile.TemporaryDirectory(prefix="dyci-asr-") as workDir:
        chunks = _splitAudio(audioPath, runtime["chunkSec"], workDir)
        offsets = _chunkOffsets(chunks, runtime["chunkSec"])
        log.info(
            f"ASR chunking: {len(chunks)} chunk(s), "
            f"{runtime['chunkSec']:.0f}s each"
        )

        model = None
        merged: list[dict] = []
        for index, (chunkPath, offset) in enumerate(zip(chunks, offsets)):
            key = str(index)
            segments = completed.get(key)
            if not isinstance(segments, list):
                segments, _, model = _transcribeOnce(chunkPath, runtime, model)
                segments = _offsetSegments(segments, offset)
                completed[key] = segments
                _saveCheckpoint(outputPath, signature, completed)
                log.info(
                    f"ASR chunk {index + 1}/{len(chunks)} done "
                    f"({len(segments)} segments)"
                )
            else:
                log.info(
                    f"ASR chunk {index + 1}/{len(chunks)} reused from checkpoint"
                )
            merged.extend(segments)

    merged.sort(key=lambda seg: float(seg.get("start", 0.0)))
    _writeOutput(merged, outputPath)
    _removeCheckpoint(outputPath)
    return merged


_CUDA_RUNTIME_TOKENS = ("cublas", "cudnn", "cuda", "libcuda")


def _isCudaRuntimeError(exc: BaseException) -> bool:
    """Return True for CUDA driver/library failures that CPU can handle."""
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(token in message for token in _CUDA_RUNTIME_TOKENS)


def _transcribeWithRuntime(
    audioPath: str,
    runtime: dict,
    outputPath: str | None,
) -> list[dict]:
    """Run one full transcription pass with a resolved runtime."""
    duration = 0.0
    if runtime["chunkSec"] > 0:
        duration = float(probeMedia(audioPath).get("durationSec") or 0.0)
    if duration > runtime["chunkSec"]:
        return _transcribeChunked(audioPath, runtime, outputPath)

    results, _, _ = _transcribeOnce(audioPath, runtime)
    _writeOutput(results, outputPath)
    _removeCheckpoint(outputPath)
    return results


def transcribe(
    audioPath: str,
    modelSize: str = "base",
    language: str | None = "zh",
    device: str = "cuda",
    computeType: str = "int8",
    diarize: bool = True,
    outputPath: str | None = None,
    batchSize: int = 32,
    vadFilter: bool = True,
    vadMinSilenceMs: int = 500,
    vadSpeechPadMs: int = 200,
    beamSize: int = 5,
    chunkSec: float = 0.0,
    resume: bool = True,
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
        batchSize: WhisperX diarization batch size.
        vadFilter: Enable faster-whisper VAD.
        vadMinSilenceMs: VAD minimum silence duration.
        vadSpeechPadMs: VAD speech padding.
        beamSize: faster-whisper beam size.
        chunkSec: Split recordings longer than this; 0 disables chunking.
        resume: Reuse completed chunks from the checkpoint.

    Returns:
        List of segment dicts with start, end, text, speaker (if diarized).
    """
    runtime = {
        "model": modelSize,
        "device": device,
        "computeType": computeType,
        "language": language,
        "diarize": bool(diarize),
        "batchSize": int(batchSize),
        "vadFilter": bool(vadFilter),
        "vadMinSilenceMs": int(vadMinSilenceMs),
        "vadSpeechPadMs": int(vadSpeechPadMs),
        "beamSize": int(beamSize),
        "chunkSec": float(chunkSec),
        "resume": bool(resume),
    }

    try:
        return _transcribeWithRuntime(audioPath, runtime, outputPath)
    except Exception as exc:
        if runtime["device"] != "cuda" or not _isCudaRuntimeError(exc):
            raise
        log.warning(
            f"CUDA ASR failed ({type(exc).__name__}: {exc}) — "
            "retrying on CPU with int8"
        )
        runtime["device"] = "cpu"
        runtime["computeType"] = "int8"
        return _transcribeWithRuntime(audioPath, runtime, outputPath)


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech-to-text with diarization")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", help="tiny/base/small/medium/large-v3")
    parser.add_argument("--language", help="Language code or 'auto'")
    parser.add_argument("--device", help="auto/cuda/cpu")
    parser.add_argument("--compute-type", help="int8/float16/...")
    parser.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--chunk-sec", type=float, help="Chunk length in seconds (0 disables)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore completed chunks")
    parser.add_argument("--output", help="Output path (without extension)")
    return parser.parse_args()


def main() -> int:
    args = _parseArgs()
    runtime = resolveAsrRuntime()
    results = transcribe(
        audioPath=args.audio,
        modelSize=args.model or runtime["model"],
        language=runtime["language"] if args.language is None else args.language,
        device=args.device or runtime["device"],
        computeType=args.compute_type or runtime["computeType"],
        diarize=runtime["diarize"] and not args.no_diarize,
        outputPath=args.output,
        batchSize=runtime["batchSize"],
        vadFilter=runtime["vadFilter"],
        vadMinSilenceMs=runtime["vadMinSilenceMs"],
        vadSpeechPadMs=runtime["vadSpeechPadMs"],
        beamSize=runtime["beamSize"],
        chunkSec=runtime["chunkSec"] if args.chunk_sec is None else args.chunk_sec,
        resume=not args.no_resume,
    )

    for seg in results[:10]:
        speaker = seg.get("speaker", "")
        prefix = f"[{speaker}] " if speaker else ""
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {prefix}{seg['text']}")

    if len(results) > 10:
        print(f"... ({len(results) - 10} more)")

    print(f"\nTotal: {len(results)} segments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
