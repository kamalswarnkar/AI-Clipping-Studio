"""Clip rendering: trim, reframe, normalise audio, burn subtitles, encode.

This is the deterministic half of the system. It receives a validated ClipPlan
and produces a real MP4. It never asks a model anything.

The filter chain is built as a single graph so ffmpeg decodes the source once:

    trim -> crop (smart 9:16) -> scale -> pad -> subtitles -> encode
    trim -> loudnorm -> aac
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import get_settings
from ..models.domain import (
    ClipPlan,
    CropKeyframe,
    Diarization,
    MediaInfo,
    Transcript,
    VisualAnalysis,
)
from . import ffmpeg, reframe, subtitles

log = logging.getLogger(__name__)


@dataclass
class RenderOptions:
    vertical: bool = True
    captions: bool = True
    smart_reframe: bool = True
    width: int = 1080
    height: int = 1920
    fps: int = 30
    crf: int = 20
    preset: str = "veryfast"
    target_lufs: float = -14.0
    font: str = "Arial"
    font_size: int = 68

    @classmethod
    def from_settings(cls, overrides: Optional[dict] = None) -> "RenderOptions":
        s = get_settings()
        options = cls(
            width=s.render_width,
            height=s.render_height,
            fps=s.render_fps,
            crf=s.render_crf,
            preset=s.render_preset,
            target_lufs=s.render_audio_lufs,
            font=s.subtitle_font,
            font_size=s.subtitle_font_size,
        )
        for key, value in (overrides or {}).items():
            if hasattr(options, key) and value is not None:
                setattr(options, key, value)
        return options


@dataclass
class RenderResult:
    video_path: Path
    thumbnail_path: Optional[Path]
    subtitle_path: Optional[Path]
    crop_strategy: str
    keyframes: int
    duration: float


def _build_video_filter(
    *,
    plan: ClipPlan,
    media: MediaInfo,
    visual: Optional[VisualAnalysis],
    options: RenderOptions,
    subtitle_file: Optional[Path],
) -> tuple[str, str, int]:
    """Assemble the video filter chain. Returns (filter, strategy, keyframes)."""
    # Rebase timestamps to zero FIRST.
    #
    # We fast-seek with -ss before -i and trim with -ss after it. That trim
    # drops frames but does NOT rezero the PTS the filter graph sees, so frames
    # arrive with PTS ~= the pre-roll length. The subtitles filter renders by
    # PTS, so without this every caption appears the pre-roll duration early.
    parts: list[str] = ["setpts=PTS-STARTPTS"]
    strategy = "no reframing"
    keyframe_count = 0

    # Exclude any burned-in caption band at the bottom of the source, so the
    # vertical crop is computed against picture we actually want to keep.
    usable_h = media.height
    band = getattr(visual, "subtitle_band_top", None) if visual else None
    if band and 0.5 < band < 1.0:
        usable_h = max(int(media.height * 0.5), int(media.height * band))
        usable_h -= usable_h % 2

    if options.vertical:
        if options.smart_reframe:
            keyframes, strategy = reframe.compute_crop_path(
                visual=visual,
                start=plan.start,
                end=plan.end,
                source_w=media.width,
                source_h=usable_h,
                target_w=options.width,
                target_h=options.height,
            )
            if usable_h != media.height:
                strategy += "; source captions cropped out"
        else:
            crop_w, crop_h = reframe._crop_size(
                media.width, usable_h, options.width, options.height
            )
            keyframes = [
                CropKeyframe(
                    t=0.0,
                    x=max(0, (media.width - crop_w) // 2),
                    y=max(0, (usable_h - crop_h) // 2),
                    w=crop_w,
                    h=crop_h,
                )
            ]
            strategy = "centre crop (smart reframing disabled)"

        keyframe_count = len(keyframes)
        crop_w, crop_h = keyframes[0].w, keyframes[0].h

        if crop_w < media.width or crop_h < usable_h or usable_h < media.height:
            x_expr = reframe.build_crop_expression(keyframes, "x")
            y_expr = reframe.build_crop_expression(keyframes, "y")
            parts.append(f"crop={crop_w}:{crop_h}:x='{x_expr}':y='{y_expr}'")

        # Fit inside the target, then pad: never distort the image.
        parts.append(
            f"scale={options.width}:{options.height}:"
            f"force_original_aspect_ratio=decrease:flags=bicubic"
        )
        parts.append(
            f"pad={options.width}:{options.height}:"
            f"(ow-iw)/2:(oh-ih)/2:color=black"
        )
    else:
        parts.append(f"scale=-2:{options.height}:flags=bicubic")

    parts.append(f"fps={options.fps}")

    if subtitle_file is not None:
        # The path is escaped for filter-graph parsing; see ffmpeg.escape_filter_path.
        parts.append(f"subtitles='{ffmpeg.escape_filter_path(subtitle_file)}'")

    parts.append("format=yuv420p")
    return ",".join(parts), strategy, keyframe_count


def render_clip(
    *,
    plan: ClipPlan,
    source: Path,
    media: MediaInfo,
    transcript: Optional[Transcript],
    visual: Optional[VisualAnalysis],
    diarization: Optional[Diarization],
    output_path: Path,
    subtitle_path: Optional[Path] = None,
    thumbnail_path: Optional[Path] = None,
    options: Optional[RenderOptions] = None,
) -> RenderResult:
    """Cut, reframe and encode one clip. Returns paths to the produced files."""
    options = options or RenderOptions.from_settings()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(0.1, plan.end - plan.start)

    # --- subtitles ----------------------------------------------------------
    written_subtitles: Optional[Path] = None
    if options.captions and transcript is not None and subtitle_path is not None:
        ass = subtitles.build_ass(
            transcript=transcript,
            clip_start=plan.start,
            clip_end=plan.end,
            width=options.width if options.vertical else media.width,
            height=options.height,
            font=options.font,
            font_size=options.font_size,
            diarization=diarization,
        )
        if "Dialogue:" in ass:
            written_subtitles = subtitles.write_ass(subtitle_path, ass)
        else:
            log.info("Clip %s has no words to caption; skipping subtitles", plan.name)

    video_filter, strategy, keyframes = _build_video_filter(
        plan=plan,
        media=media,
        visual=visual,
        options=options,
        subtitle_file=written_subtitles,
    )

    # --- audio --------------------------------------------------------------
    # loudnorm brings every clip to a consistent level; without it, clips cut
    # from different parts of a recording vary noticeably in volume.
    # asetpts mirrors the video rebase so the two streams stay aligned.
    audio_filter = (
        "asetpts=PTS-STARTPTS,"
        f"loudnorm=I={options.target_lufs}:TP=-1.5:LRA=11,"
        "aresample=async=1:first_pts=0"
    )

    args = [
        "-y",
        # A single accurate seek before -i.
        #
        # Do NOT reintroduce the "fast pre-roll seek + output -ss trim" pattern
        # here. Input -ss already rebases timestamps to zero, so the filter graph
        # burns subtitles from that zero point and the output-side -ss then
        # discards the first seconds of ALREADY-SUBTITLED video, shifting every
        # caption earlier by the pre-roll length. Modern ffmpeg decodes to the
        # exact frame on input seek, so the pre-roll bought nothing anyway.
        "-ss",
        f"{plan.start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-vf",
        video_filter,
    ]

    if media.has_audio:
        args += ["-af", audio_filter, "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    else:
        args += ["-an"]

    args += [
        "-c:v",
        "libx264",
        "-preset",
        options.preset,
        "-crf",
        str(options.crf),
        "-profile:v",
        "high",
        "-level",
        "4.1",
        # Streaming-friendly: moov atom up front so the browser can play
        # immediately instead of downloading the whole file first.
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    ffmpeg.run_ffmpeg(args, timeout=None)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ffmpeg.FFmpegError(f"Render produced no output for {plan.name}.")

    # --- thumbnail ----------------------------------------------------------
    produced_thumbnail: Optional[Path] = None
    if thumbnail_path is not None:
        try:
            ffmpeg.extract_frame(
                output_path, min(1.0, duration / 3.0), thumbnail_path, width=480
            )
            produced_thumbnail = thumbnail_path
        except Exception as exc:  # noqa: BLE001 - a missing thumbnail is cosmetic
            log.warning("Thumbnail failed for %s: %s", plan.name, exc)

    return RenderResult(
        video_path=output_path,
        thumbnail_path=produced_thumbnail,
        subtitle_path=written_subtitles,
        crop_strategy=strategy,
        keyframes=keyframes,
        duration=duration,
    )

