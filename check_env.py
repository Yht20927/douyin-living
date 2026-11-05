# -*- coding: utf-8 -*-
"""Environment pre-flight check — run before recording to catch missing deps early.

Usage::

    python check_env.py              # recording baseline (AI deps are warnings)
    python check_env.py --clip       # AI deps and model files are required
    python check_env.py --skip-auth  # containers / CI without cookies
"""

import argparse
import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # reported by the core dependency check below
    yaml = None

_PROJECT_ROOT = Path(__file__).resolve().parent


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str, fix: str = ""):
    print(f"  ❌ {msg}")
    if fix:
        print(f"     💡 {fix}")


def _warn(msg: str, fix: str = ""):
    print(f"  ⚠️  {msg}")
    if fix:
        print(f"     💡 {fix}")


def _hfCacheHit(repoId: str) -> bool | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    return isinstance(try_to_load_from_cache(repoId, "config.json"), str)


def check(clip: bool = False, skipAuth: bool = False):
    print("=" * 50)
    print("Douyin Live AI Clipper — Environment Check")
    print("=" * 50)
    errors = 0

    # ── Python version ─────────────────────────────────────────
    print("\n📦 Python")
    py = sys.version_info
    if py >= (3, 11):
        _ok(f"Python {py.major}.{py.minor}.{py.micro}")
    else:
        _fail(f"Python {py.major}.{py.minor}.{py.micro}", "Requires Python >= 3.11")
        errors += 1

    # ── Core deps ──────────────────────────────────────────────
    print("\n📦 Core dependencies")
    core = [
        ("aiohttp", "aiohttp"),
        ("aiofiles", "aiofiles"),
        ("httpx", "httpx"),
        ("websocket", "websocket-client"),
        ("google.protobuf", "protobuf"),
        ("execjs", "PyExecJS"),
        ("dotenv", "python-dotenv"),
        ("loguru", "loguru"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("yaml", "PyYAML"),
    ]
    for mod, pkg in core:
        try:
            importlib.import_module(mod)
            _ok(mod)
        except ImportError:
            _fail(mod, f"pip install {pkg}")
            errors += 1

    # ── AI deps (optional for recording, required for clipping) ─
    level = "required for --clip" if clip else "optional until --clip"
    print(f"\n📦 AI dependencies ({level})")
    ai = [
        ("librosa", "librosa"),
        ("jieba", "jieba"),
        ("fasttext", "fasttext"),
        ("text2vec", "text2vec"),
        ("scenedetect", "scenedetect[opencv]"),
        ("transformers", "transformers"),
        ("panns_inference", "panns-inference"),
        ("faster_whisper", "faster-whisper"),
    ]
    for mod, pkg in ai:
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if found:
            _ok(mod)
        elif clip:
            _fail(mod, f"pip install -r requirements-ai.txt  # includes {pkg}")
            errors += 1
        else:
            _warn(mod, f"pip install {pkg}")

    # ── Model assets ───────────────────────────────────────────
    print(f"\n📦 Model assets ({level})")
    textConfig = {}
    configPath = _PROJECT_ROOT / "config" / "text.yaml"
    if yaml is not None and configPath.exists():
        with open(configPath, encoding="utf-8") as f:
            textConfig = yaml.safe_load(f) or {}

    fasttextPath = Path(
        textConfig.get("fasttext_model_path", "models/cc.zh.300.bin")
    )
    if not fasttextPath.is_absolute():
        fasttextPath = _PROJECT_ROOT / fasttextPath
    if fasttextPath.exists() and fasttextPath.stat().st_size > 1024 * 1024:
        _ok(f"FastText: {fasttextPath}")
    else:
        fix = "python scripts/download_models.py --fasttext  # ~4 GB"
        if clip:
            _fail(f"FastText model missing: {fasttextPath}", fix)
            errors += 1
        else:
            _warn(f"FastText model missing: {fasttextPath}", fix)

    hfModels = [
        textConfig.get("text2vec_model", "shibing624/text2vec-base-chinese"),
        textConfig.get("sentiment_model", "uer/roberta-base-finetuned-jd-binary-chinese"),
        "Systran/faster-whisper-base",
    ]
    for repoId in hfModels:
        hit = _hfCacheHit(repoId)
        if hit is True:
            _ok(f"HF cache: {repoId}")
        elif hit is False:
            if repoId.startswith("Systran/faster-whisper"):
                fix = "python scripts/download_models.py --asr base"
            else:
                fix = "python scripts/download_models.py --hf"
            if clip:
                _fail(f"HF model not cached: {repoId}", fix)
                errors += 1
            else:
                _warn(f"HF model not cached: {repoId}", fix)
        else:
            _warn(f"HF cache unknown: {repoId}", "pip install huggingface_hub")

    # ── System binaries ────────────────────────────────────────
    print("\n🔧 System binaries")
    for bin_name in ["ffmpeg", "node"]:
        path = shutil.which(bin_name)
        if path:
            ver = subprocess.run([bin_name, "-version"], capture_output=True, text=True, timeout=5)
            first = ver.stdout.splitlines()[0] if ver.stdout else ""
            _ok(f"{bin_name}: {first[:60]}")
        else:
            if bin_name == "ffmpeg":
                _fail(bin_name, "sudo apt install ffmpeg")
                errors += 1
            else:
                _warn(bin_name, "sudo apt install nodejs")

    # ── GPU ────────────────────────────────────────────────────
    print("\n🔧 GPU / CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            _ok(f"CUDA available: {dev.name} ({dev.total_memory // (1024**2)} MiB)")
        else:
            _warn("CUDA not available", "ASR will run on CPU (slower)")
    except ImportError:
        _warn("torch not installed", "pip install torch")

    # ── Cookies ────────────────────────────────────────────────
    print("\n🔐 Authentication")
    if skipAuth:
        _warn("Auth check skipped (--skip-auth)")
    else:
        try:
            from src.auth import Auth
            auth = Auth.fromEnv()
            if auth.cookieStr:
                _ok(f"Cookies loaded: {len(auth.cookie)} entries")
            else:
                _fail("No cookies found", "Create .env file with DY_LIVE_COOKIES=...")
                errors += 1
        except Exception as e:
            _fail(f"Failed to load auth: {e}", "Create .env file with DY_LIVE_COOKIES=...")
            errors += 1

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if errors == 0:
        if clip:
            print("🚀 Environment OK — ready to record and run AI clipping")
        else:
            print("🚀 Environment OK — ready to record")
    else:
        print(f"🚨 Found {errors} critical issue(s). Fix them before recording.")
    print("=" * 50)
    return errors == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-flight environment check")
    parser.add_argument("--clip", action="store_true", help="require AI dependencies and model files")
    parser.add_argument("--skip-auth", action="store_true", help="skip cookie loading")
    args = parser.parse_args()

    ok = check(clip=args.clip, skipAuth=args.skip_auth)
    sys.exit(0 if ok else 1)
