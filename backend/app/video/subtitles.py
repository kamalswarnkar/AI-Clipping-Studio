"""Subtitle generation in ASS format.

Built from word-level timings so cues appear with the speech rather than drifting
against it. Styling stays deliberately minimal: white text, heavy outline, safe
margins, no animation (Architecture.md section 17).

ASS is used rather than SRT because it carries positioning and styling, which
matters when burning into a 1080x1920 frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..models.domain import Diarization, Transcript, Word

log = logging.getLogger(__name__)

# Line length for a 9:16 frame. A narrow frame wraps badly past this.
MAX_CHARS_PER_LINE = 22
MAX_LINES = 2
# The frame the styling above was tuned against; other sizes scale from it.
DESIGN_HEIGHT = 1920
# Side margin, as a fraction of frame width, on each side.
MARGIN_H_FRACTION = 0.07
MAX_CUE_SECONDS = 3.2
# A pause longer than this ends the cue, so captions breathe with the speech.
CUE_BREAK_GAP = 0.55


def _timestamp(seconds: float) -> str:
    """ASS uses H:MM:SS.cc (centiseconds)."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _escape(text: str) -> str:
    """Escape ASS markup characters in dialogue text."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .strip()
    )


def chars_per_line(width: int, font_size: int) -> int:
    """How many characters fit on one line, at this width and font size.

    This is load-bearing, not cosmetic. WrapStyle 2 means libass breaks lines
    only where we put a break, so a line we allow to be too long does not wrap
    -- it runs off the edge of the frame. Deriving the limit from the actual
    font size keeps text inside the margins at any size or frame shape.

    The 0.62 is the average glyph advance as a fraction of font size, measured
    against the 22-character line the vertical layout was tuned with. Capped at
    42 because a longer line is hard to read in one glance however much room
    there is.
    """
    if width <= 0 or font_size <= 0:
        return MAX_CHARS_PER_LINE
    usable = width * (1 - 2 * MARGIN_H_FRACTION)
    return max(12, min(42, int(usable / (0.62 * font_size))))


def _inside(t: float, spans: list[tuple[float, float]]) -> bool:
    return any(start <= t <= end for start, end in spans)


def _wrap(words: list[str], max_chars: int = MAX_CHARS_PER_LINE) -> str:
    """Wrap into at most MAX_LINES balanced lines using the ASS break code."""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if len(lines) > MAX_LINES:
        # Too long to show at once; merge the overflow into the last line and let
        # libass shrink-wrap it rather than dropping words.
        head = lines[: MAX_LINES - 1]
        head.append(" ".join(lines[MAX_LINES - 1 :]))
        lines = head

    return "\\N".join(lines)


def group_words_into_cues(
    words: list[Word],
    *,
    clip_start: float,
    clip_end: float,
    diarization: Optional[Diarization] = None,
    max_chars: int = MAX_CHARS_PER_LINE,
) -> list[tuple[float, float, str]]:
    """Group words into readable cues, timed relative to the clip start.

    When speaker turns are known, a cue never spans two speakers -- a caption
    that runs one person's answer straight into another's question is actively
    misleading about who said what.
    """
    cues: list[tuple[float, float, str]] = []
    buffer: list[Word] = []
    current_speaker: Optional[str] = None

    def flush() -> None:
        if not buffer:
            return
        start = max(0.0, buffer[0].start - clip_start)
        end = min(clip_end - clip_start, buffer[-1].end - clip_start)
        if end <= start:
            end = start + 0.4
        text = _wrap([_escape(w.word) for w in buffer if w.word.strip()], max_chars)
        if text:
            cues.append((start, end, text))
        buffer.clear()

    for word in words:
        speaker = (
            diarization.speaker_at((word.start + word.end) / 2.0)
            if diarization
            else None
        )

        if buffer:
            gap = word.start - buffer[-1].end
            span = word.end - buffer[0].start
            projected = sum(len(w.word.strip()) + 1 for w in buffer) + len(word.word)
            if (
                gap > CUE_BREAK_GAP
                or span > MAX_CUE_SECONDS
                or projected > max_chars * MAX_LINES
                or (speaker is not None and current_speaker is not None and speaker != current_speaker)
            ):
                flush()

        if not buffer:
            current_speaker = speaker
        buffer.append(word)

        # End the cue on sentence-final punctuation so cues match thoughts.
        if word.word.strip().endswith((".", "!", "?")):
            flush()

    flush()

    # Remove overlaps that would make two cues render at once.
    for i in range(len(cues) - 1):
        start, end, text = cues[i]
        next_start = cues[i + 1][0]
        if end > next_start:
            cues[i] = (start, max(start + 0.2, next_start - 0.02), text)

    return cues


def build_ass(
    *,
    transcript: Transcript,
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    font: str = "Arial",
    font_size: int = 120,
    diarization: Optional[Diarization] = None,
    reserved_bottom: float = 0.0,
    suppress_spans: Optional[list[tuple[float, float]]] = None,
) -> str:
    """Render an ASS subtitle file for one clip.

    `font_size` is specified for a 1920-tall frame and scaled to whatever frame
    is actually being rendered, so captions occupy the same share of the
    picture at any output size.

    `reserved_bottom` is the fraction of the frame already occupied by captions
    burned into the source. When the source band is not being cropped away,
    these captions are lifted above it rather than printed on top of it.

    `suppress_spans` are windows in source time where the source is already
    showing its own captions. Cues landing inside one are dropped: a second set
    of captions saying the same words is worse than none.
    """
    words = [
        w
        for w in transcript.words_between(clip_start, clip_end)
        if w.word.strip()
    ]

    font_size = max(12, round(font_size * height / DESIGN_HEIGHT))
    cues = group_words_into_cues(
        words,
        clip_start=clip_start,
        clip_end=clip_end,
        diarization=diarization,
        max_chars=chars_per_line(width, font_size),
    )

    if suppress_spans:
        kept = [
            cue
            for cue in cues
            if not _inside(clip_start + (cue[0] + cue[1]) / 2.0, suppress_spans)
        ]
        if len(kept) != len(cues):
            log.info(
                "Dropped %d cue(s) that fell where the source shows its own captions",
                len(cues) - len(kept),
            )
        cues = kept

    # Margins keep text clear of platform UI overlays at the bottom of the frame.
    margin_v = int(height * 0.13)
    if reserved_bottom > 0:
        margin_v = max(margin_v, int(height * (reserved_bottom + 0.02)))
    margin_h = int(width * MARGIN_H_FRACTION)
    outline = max(2, round(font_size * 0.06))
    shadow = 0

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [
        f"Dialogue: 0,{_timestamp(start)},{_timestamp(end)},Caption,,0,0,0,,{text}"
        for start, end, text in cues
    ]

    return header + "\n".join(lines) + "\n"


def write_ass(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # libass reads UTF-8; BOM confuses some builds, so write without one.
    path.write_text(content, encoding="utf-8")
    return path
