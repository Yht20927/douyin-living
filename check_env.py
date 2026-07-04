# -*- coding: utf-8 -*-
"""Environment pre-flight check — run before recording to catch missing deps early."""

import importlib
import shutil
import subprocess
import sys


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


def check():
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
        ("httpx", "httpx"),
        ("websocket", "websocket-client"),
        ("google.protobuf", "protobuf"),
        ("execjs", "PyExecJS"),
        ("dotenv", "python-dotenv"),
        ("loguru", "loguru"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ]
    for mod, pkg in core:
        try:
            importlib.import_module(mod)
            _ok(mod)
        except ImportError:
            _fail(mod, f"pip install {pkg}")
            errors += 1

    # ── AI deps (optional for recording, required for clipping) ─
    print("\n📦 AI dependencies (required for --clip)")
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
            importlib.import_module(mod)
            _ok(mod)
        except ImportError:
            _warn(mod, f"pip install {pkg}")

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
        print("🚀 Environment OK — ready to record")
    else:
        print(f"🚨 Found {errors} critical issue(s). Fix them before recording.")
    print("=" * 50)
    return errors == 0


if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
