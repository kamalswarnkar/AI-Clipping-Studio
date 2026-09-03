"""Thin, safe wrapper around the ffmpeg/ffprobe binaries.

Everything that touches video goes through here. Commands are always built as
argument *lists* (never shell strings), so filenames containing spaces, quotes
or shell metacharacters cannot be misinterpreted or injected.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from ..config import resolve_binary
from ..jobs.states import PipelineError
from ..models.domain import MediaInfo

log = logging.getLogger(__name__)

# Extensions we accept on upload.
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

# Suppress ffmpeg banner noise; keep real errors.
_BASE_FLAGS = ["-hide_banner", "-loglevel", "error", "-nostdin"]


class FFmpegError(PipelineError):
    def __init__(self, message: str, *, cmd: Sequence[str] = (), stderr: str = "") -> None:
        super().__init__(message, stage="render")
        self.cmd = list(cmd)
        self.stderr = stderr


def _no_window_kwargs() -> dict:
    """Stop Windows from flashing a console window for every ffmpeg call."""
    import os

    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def run_ffmpeg(
    args: Sequence[str],
    *,
    timeout: Optional[float] = None,
    cwd: Optional[Path] = None,
) -> str:
    """Run ffmpeg with the given arguments. Returns stderr on success."""
    cmd = [resolve_binary("ffmpeg"), *_BASE_FLAGS, *args]
    log.debug("ffmpeg: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            check=False,
            **_no_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"FFmpeg timed out after {timeout:.0f}s.", cmd=cmd
        ) from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        detail = tail[-1] if tail else "no error output"
        raise FFmpegError(
            f"FFmpeg failed: {detail}", cmd=cmd, stderr=proc.stderr or ""
        )
    return proc.stderr or ""


def run_ffprobe(args: Sequence[str], *, timeout: float = 120.0) -> str:
    cmd = [resolve_binary("ffprobe"), "-hide_banner", "-loglevel", "error", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            **_no_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError("FFprobe timed out.", cmd=cmd) from exc

    if proc.returncode != 0:
        raise FFmpegError(
            f"FFprobe failed: {(proc.stderr or '').strip()[:400]}",
            cmd=cmd,
            stderr=proc.stderr or "",
        )
    return proc.stdout


def _parse_fraction(value: str) -> float:
    """Parse ffprobe rationals such as '30000/1001'."""
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            d = float(den)
            return float(num) / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: Path) -> MediaInfo:
    """Inspect a media file, raising user-facing errors for unusable input."""
    if not path.exists():
        raise PipelineError(
            f"Video file not found: {path.name}",
            stage="ingest",
            remedy="Re-upload the file.",
        )

    raw = run_ffprobe(
        [
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise PipelineError(
            "Could not read this file. It may be corrupt or not a video.",
            stage="ingest",
            remedy="Try re-exporting the video as MP4 (H.264 + AAC).",
        ) from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video is None:
        raise PipelineError(
            "This file has no video stream.",
            stage="ingest",
            remedy="Upload a video file rather than an audio-only file.",
        )

    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise PipelineError(
            "Could not determine the video duration; the file looks corrupt.",
            stage="ingest",
            remedy="Try re-exporting or remuxing the video.",
        )

    fps = _parse_fraction(str(video.get("avg_frame_rate") or "")) or _parse_fraction(
        str(video.get("r_frame_rate") or "")
    )

    return MediaInfo(
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=round(fps, 3) if fps else 0.0,
        video_codec=str(video.get("codec_name") or ""),
        audio_codec=str(audio.get("codec_name") or "") if audio else "",
        has_audio=audio is not None,
        audio_sample_rate=int(audio.get("sample_rate") or 0) if audio else 0,
        audio_channels=int(audio.get("channels") or 0) if audio else 0,
        bitrate=int(float(fmt.get("bit_rate") or 0)),
        size_bytes=int(float(fmt.get("size") or 0)),
        container=str(fmt.get("format_name") or ""),
    )


def extract_audio(
    source: Path,
    dest: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Extract mono PCM WAV for ASR and audio analysis.

    16 kHz mono is what Whisper expects and is plenty for the loudness/event
    features we compute.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        timeout=None,
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise PipelineError(
            "Audio extraction produced an empty file.",
            stage="extract_audio",
            remedy="The source may have a broken or silent audio track.",
        )
    return dest


def make_proxy(source: Path, dest: Path, *, height: int = 480, fps: int = 8) -> Path:
    """Build a small proxy used for cheap frame-by-frame visual analysis.

    Decoding a downscaled, low-fps copy is dramatically faster than seeking the
    full-resolution source thousands of times.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            f"scale=-2:{height}:flags=fast_bilinear,fps={fps}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        timeout=None,
    )
    return dest


def extract_frame(source: Path, timestamp: float, dest: Path, *, width: int = 640) -> Path:
    """Grab a single frame as JPEG (used for thumbnails and vision montages)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2:flags=bicubic",
            "-q:v",
            "4",
            str(dest),
        ],
        timeout=180,
    )
    return dest


def has_encoder(name: str) -> bool:
    """Check whether this ffmpeg build exposes a given encoder."""
    try:
        out = subprocess.run(
            [resolve_binary("ffmpeg"), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            **_no_window_kwargs(),
        ).stdout
    except Exception:  # noqa: BLE001
        return False
    return name in out


def escape_filter_path(path: Path) -> str:
    """Escape a path for use *inside* an ffmpeg filter argument.

    Filter graphs parse ':' and '\\' themselves, so a Windows path like
    C:\\a\\b.ass must become C\\:/a/b.ass or the filter fails to load.
    """
    text = str(path).replace("\\", "/")
    return text.replace(":", "\\:")
