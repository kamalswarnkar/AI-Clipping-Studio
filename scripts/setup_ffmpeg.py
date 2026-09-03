"""Download a project-local FFmpeg build.

Used when FFmpeg is not already on the PATH. Everything lands in tools/ffmpeg,
so nothing outside the project is touched and no admin rights are needed.
"""

from __future__ import annotations

import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "tools" / "ffmpeg"

WINDOWS_BUILD = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def already_available() -> bool:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("FFmpeg and ffprobe are already on your PATH.")
        return True
    suffix = ".exe" if platform.system() == "Windows" else ""
    if (TARGET / "bin" / f"ffmpeg{suffix}").exists():
        print(f"FFmpeg is already installed at {TARGET}")
        return True
    return False


def install_windows() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / "ffmpeg.zip"
        print(f"Downloading {WINDOWS_BUILD} ...")

        def report(block: int, size: int, total: int) -> None:
            if total > 0:
                pct = min(100, block * size * 100 // total)
                print(f"\r  {pct}%", end="", flush=True)

        urllib.request.urlretrieve(WINDOWS_BUILD, archive, reporthook=report)
        print("\nExtracting ...")

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(td)

        extracted = next(
            (p for p in Path(td).iterdir() if p.is_dir() and p.name.startswith("ffmpeg")),
            None,
        )
        if extracted is None:
            print("Could not find the extracted FFmpeg directory.", file=sys.stderr)
            return 1

        if TARGET.exists():
            shutil.rmtree(TARGET)
        shutil.move(str(extracted), str(TARGET))

    ffmpeg = TARGET / "bin" / "ffmpeg.exe"
    if not ffmpeg.exists():
        print("Install finished but ffmpeg.exe is missing.", file=sys.stderr)
        return 1

    print(f"\nInstalled to {TARGET / 'bin'}")
    print("The app finds this automatically; no PATH changes are needed.")
    return 0


def main() -> int:
    if already_available():
        return 0

    system = platform.system()
    if system == "Windows":
        return install_windows()

    print(
        f"Automatic install is only implemented for Windows (detected {system}).\n"
        "Install FFmpeg with your package manager:\n"
        "  macOS:         brew install ffmpeg\n"
        "  Debian/Ubuntu: sudo apt install ffmpeg\n"
        "  Fedora:        sudo dnf install ffmpeg\n"
        "Or set FFMPEG_BIN and FFPROBE_BIN in .env.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
