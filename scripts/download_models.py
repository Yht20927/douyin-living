#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and verify the optional model assets used by the clipper.

Heavy dependencies are deliberately not imported here: the script only needs
PyYAML and (for Hugging Face downloads) ``huggingface_hub``.  The FastText
model is a 4+ GB download, so it is never fetched implicitly.

Usage::

    python scripts/download_models.py --check
    python scripts/download_models.py --check --strict
    python scripts/download_models.py --fasttext
    python scripts/download_models.py --hf
    python scripts/download_models.py --asr base
    python scripts/download_models.py --all          # HF text models + ASR base

Set ``HF_ENDPOINT=https://hf-mirror.com`` to use a Hugging Face mirror.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_CONFIG = PROJECT_ROOT / "config" / "text.yaml"

DEFAULT_FASTTEXT_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz"
)
DEFAULT_FASTTEXT_PATH = "models/cc.zh.300.bin"

ASR_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


# ── helpers ──────────────────────────────────────────────────────────


def _humanSize(numBytes: int) -> str:
    value = float(numBytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _loadTextConfig() -> dict:
    if not TEXT_CONFIG.exists():
        return {}
    with open(TEXT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _hfCacheHit(repoId: str) -> bool | None:
    """Return True/False when the cache can be inspected, None if unknown."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    result = try_to_load_from_cache(repoId, "config.json")
    return isinstance(result, str)


# ── checks ───────────────────────────────────────────────────────────


def checkAssets(asrSize: str) -> int:
    """Print model asset status and return the number of missing assets."""
    cfg = _loadTextConfig()
    fasttextPath = _resolve(cfg.get("fasttext_model_path", DEFAULT_FASTTEXT_PATH))
    textModels = [
        cfg.get("text2vec_model", "shibing624/text2vec-base-chinese"),
        cfg.get("sentiment_model", "uer/roberta-base-finetuned-jd-binary-chinese"),
    ]
    asrRepo = ASR_REPOS.get(asrSize)

    missing = 0
    print("\n📦 FastText")
    if fasttextPath.exists():
        size = fasttextPath.stat().st_size
        if size < 1024 * 1024:
            print(f"  ⚠️  {fasttextPath} looks too small ({_humanSize(size)})")
        else:
            print(f"  ✅ {fasttextPath} ({_humanSize(size)})")
    else:
        print(f"  ❌ {fasttextPath} — run: python scripts/download_models.py --fasttext")
        missing += 1

    print("\n🤗 Hugging Face models")
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        print(f"  ℹ️  HF_ENDPOINT={endpoint}")
    for repoId in textModels:
        hit = _hfCacheHit(repoId)
        if hit is True:
            print(f"  ✅ {repoId}")
        elif hit is False:
            print(f"  ❌ {repoId} (not cached) — run: python scripts/download_models.py --hf")
            missing += 1
        else:
            print(f"  ⚠️  {repoId} — huggingface_hub not installed, cannot verify")

    if asrRepo:
        hit = _hfCacheHit(asrRepo)
        if hit is True:
            print(f"  ✅ {asrRepo}")
        elif hit is False:
            print(f"  ❌ {asrRepo} (not cached) — run: python scripts/download_models.py --asr {asrSize}")
            missing += 1
        else:
            print(f"  ⚠️  {asrRepo} — huggingface_hub not installed, cannot verify")

    print("\nℹ️  PANNs checkpoint is downloaded by panns-inference on first run.")
    return missing


# ── downloads ────────────────────────────────────────────────────────


def downloadFile(url: str, dest: Path, force: bool = False) -> None:
    """Stream a URL to *dest*, resuming a partial ``.part`` file when possible."""
    if dest.exists() and not force:
        print(f"  ✅ already present: {dest} ({_humanSize(dest.stat().st_size)})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    partPath = dest.with_name(dest.name + ".part")
    if force:
        partPath.unlink(missing_ok=True)

    offset = partPath.stat().st_size if partPath.exists() else 0
    headers = {"User-Agent": "douyin-live-model-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    print(f"  ⬇️  {url}")
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and response.status != 206:
            offset = 0  # server ignored the Range header — restart
        total = int(response.headers.get("Content-Length") or 0) + offset
        mode = "ab" if offset else "wb"
        written = offset
        lastReport = 0.0
        with open(partPath, mode) as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - lastReport >= 1.0:
                    lastReport = now
                    pct = f" ({written * 100 // total:.0f}%)" if total else ""
                    print(
                        f"     {_humanSize(written)} / {_humanSize(total)}{pct}",
                        end="\r",
                        flush=True,
                    )
    print()
    partPath.replace(dest)
    print(f"  ✅ saved {dest} ({_humanSize(dest.stat().st_size)})")


def downloadFastText(url: str, target: Path, force: bool = False) -> None:
    print("\n📦 FastText")
    if target.exists() and not force:
        print(f"  ✅ already present: {target} ({_humanSize(target.stat().st_size)})")
        return

    if url.endswith(".gz"):
        archive = target.with_name(target.name + ".gz")
        downloadFile(url, archive, force=force)
        tmpPath = target.with_name(target.name + ".tmp")
        print(f"  📦 decompressing {archive.name} …")
        with gzip.open(archive, "rb") as src, open(tmpPath, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        tmpPath.replace(target)
        archive.unlink(missing_ok=True)
        print(f"  ✅ saved {target} ({_humanSize(target.stat().st_size)})")
    else:
        downloadFile(url, target, force=force)


def downloadHfModel(repoId: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ❌ huggingface_hub is not installed. Run: pip install huggingface_hub")
        return
    endpoint = os.environ.get("HF_ENDPOINT")
    suffix = f" via {endpoint}" if endpoint else ""
    print(f"  ⬇️  {repoId}{suffix}")
    snapshot_download(repo_id=repoId)
    print(f"  ✅ {repoId}")


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download/verify optional model assets (nothing is fetched by default)."
    )
    parser.add_argument("--check", action="store_true", help="verify local assets (default)")
    parser.add_argument("--strict", action="store_true", help="exit 1 when --check finds missing assets")
    parser.add_argument("--fasttext", action="store_true", help="download the FastText model")
    parser.add_argument("--hf", action="store_true", help="download text2vec + sentiment models")
    parser.add_argument("--asr", nargs="?", const="base", choices=ASR_REPOS, help="download a faster-whisper model (default: base)")
    parser.add_argument("--all", action="store_true", help="--hf + --asr base (FastText still requires --fasttext)")
    parser.add_argument("--asr-size", default="base", choices=ASR_REPOS, help="ASR model used by --check (default: base)")
    parser.add_argument("--fasttext-url", default=DEFAULT_FASTTEXT_URL, help="override the FastText archive URL")
    parser.add_argument("--fasttext-path", default=None, help="override the output path")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    cfg = _loadTextConfig()
    downloadRequested = bool(args.fasttext or args.hf or args.asr or args.all)
    if not downloadRequested:
        missing = checkAssets(args.asr_size)
        print()
        if missing and args.strict:
            print(f"🚨 {missing} asset(s) missing.")
            return 1
        print("✅ Model check complete." if not missing else "⚠️  Some model assets are missing (optional).")
        return 0

    if args.fasttext:
        target = _resolve(
            args.fasttext_path
            or cfg.get("fasttext_model_path", DEFAULT_FASTTEXT_PATH)
        )
        downloadFastText(args.fasttext_url, target, force=args.force)

    if args.hf or args.all:
        print("\n🤗 Hugging Face models")
        repos = [
            cfg.get("text2vec_model", "shibing624/text2vec-base-chinese"),
            cfg.get("sentiment_model", "uer/roberta-base-finetuned-jd-binary-chinese"),
        ]
        for repoId in repos:
            downloadHfModel(repoId)

    asrSize = args.asr or ("base" if args.all else None)
    if asrSize:
        print("\n🎙️ ASR model")
        downloadHfModel(ASR_REPOS[asrSize])

    print("\n✅ Done. Run `python scripts/download_models.py --check` to verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
