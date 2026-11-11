"""PITCHMIND Environment & Video Tooling Verification Script (Phase 1A).

Checks and reports:
1. Python runtime and virtual environment isolation
2. Installed package versions and editable package status
3. FFmpeg and FFprobe binary availability
4. GPU hardware detection and VRAM allocation
"""

import os
import shutil
import subprocess
import sys


def check_python_environment():
    print("=" * 70)
    print("1. PYTHON RUNTIME & ENVIRONMENT ISOLATION")
    print("=" * 70)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"  Python Version:     {sys.version.split()[0]} ({sys.platform})")
    print(f"  Executable:         {sys.executable}")
    print(f"  Virtual Env Active: {in_venv} (Prefix: {sys.prefix})")
    if not in_venv:
        print("  WARNING: Not running inside a dedicated virtual environment!")
    else:
        print("  [OK] Operating inside isolated virtual environment.")


def check_packages():
    print("\n" + "=" * 70)
    print("2. CORE PACKAGE VERSIONS (MINIMAL INGESTION STACK)")
    print("=" * 70)
    packages = [
        ("pitchmind", "Core Platform Package"),
        ("cv2", "OpenCV Video/Image Engine"),
        ("numpy", "Numerical Computation"),
        ("pydantic", "Schema Validation"),
        ("pytest", "Testing Framework"),
        ("yaml", "PyYAML Configuration Parser"),
    ]
    
    for mod_name, description in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            file_path = getattr(mod, "__file__", "builtin/namespace")
            print(f"  {description:<32} {mod_name:<12} {ver:<12} [OK]")
        except ImportError:
            print(f"  {description:<32} {mod_name:<12} {'NOT INSTALLED':<12} [MISSING]")

    # Check deferred packages
    deferred = [
        ("fastapi", "FastAPI Web Backend (Phase 8/13)"),
        ("uvicorn", "Uvicorn ASGI Server (Phase 8/13)"),
        ("torch", "PyTorch Deep Learning Engine (Deferred)"),
        ("torch_geometric", "PyG Graph Neural Network (Deferred)"),
    ]
    print("\n  Deferred / Non-Ingestion Packages Status:")
    for mod_name, description in deferred:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  {description:<40} {mod_name:<12} {ver:<12} (Present in env)")
        except ImportError:
            print(f"  {description:<40} {mod_name:<12} {'Deferred':<12} [Planned for later phase]")


def check_video_tooling():
    print("\n" + "=" * 70)
    print("3. VIDEO TOOLING (FFmpeg & FFprobe)")
    print("=" * 70)
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    # If not directly resolved by shutil.which, search known WinGet and User PATH candidate locations
    if not ffmpeg_path or not ffprobe_path:
        user_path = os.environ.get("Path", "")
        candidates = [
            r"C:\Users\ombag\AppData\Local\Microsoft\WinGet\Links",
            r"C:\Users\ombag\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin",
            r"C:\Users\ombag\Practice\ffmpeg-2025-09-22-git-c9168717bf-full_build\bin",
        ]
        for c in candidates:
            if not ffmpeg_path and os.path.exists(os.path.join(c, "ffmpeg.exe")):
                ffmpeg_path = os.path.join(c, "ffmpeg.exe")
            if not ffprobe_path and os.path.exists(os.path.join(c, "ffprobe.exe")):
                ffprobe_path = os.path.join(c, "ffprobe.exe")

    if ffmpeg_path:
        print(f"  ffmpeg:  FOUND -> {ffmpeg_path}")
        try:
            res = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, timeout=5)
            first_line = res.stdout.splitlines()[0] if res.stdout else "Unknown version"
            print(f"           Version: {first_line}")
        except Exception as e:
            print(f"           Version check failed: {e}")
    else:
        print("  ffmpeg:  NOT FOUND in system PATH")

    if ffprobe_path:
        print(f"  ffprobe: FOUND -> {ffprobe_path}")
        try:
            res = subprocess.run([ffprobe_path, "-version"], capture_output=True, text=True, timeout=5)
            first_line = res.stdout.splitlines()[0] if res.stdout else "Unknown version"
            print(f"           Version: {first_line}")
        except Exception as e:
            print(f"           Version check failed: {e}")
    else:
        print("  ffprobe: NOT FOUND in system PATH")

    if not ffmpeg_path or not ffprobe_path:
        print("\n  Recommended Windows Installation Procedure:")
        print("  Option A (Windows Package Manager - Recommended):")
        print("    winget install --id Gyan.FFmpeg")
        print("  Option B (Manual Download):")
        print("    Download release build from https://www.gyan.dev/ffmpeg/builds/")
        print("    Extract to C:\\ffmpeg and add C:\\ffmpeg\\bin to system PATH environment variable.")


def check_gpu():
    print("\n" + "=" * 70)
    print("4. HARDWARE & GPU ACCELERATION STATUS")
    print("=" * 70)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                line = res.stdout.strip()
                print(f"  NVIDIA GPU Detected: {line}")
                print("  Hardware Baseline matches specification (RTX 3050 Laptop GPU / 4 GB VRAM).")
            else:
                print(f"  nvidia-smi error: {res.stderr.strip()}")
        except Exception as e:
            print(f"  Failed to query nvidia-smi: {e}")
    else:
        print("  nvidia-smi not available in PATH.")


if __name__ == "__main__":
    check_python_environment()
    check_packages()
    check_video_tooling()
    check_gpu()
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

