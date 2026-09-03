"""Intelligent start/end boundary refinement.

Architecture.md section 10 calls this one of the most important parts of the
system. The LLM proposes semantic boundaries; this module snaps them to real
acoustic and linguistic events using word-level timestamps, so a clip never
opens mid-word or ends before the payoff lands.

Everything here is deterministic. The model says *where the thought is*; this
code decides the exact frame-accurate cut.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models.domain import AudioAnalysis, Scene, Transcript, Word

# A sentence-final punctuation mark, allowing for trailing quotes/brackets.
_SENTENCE_END = re.compile(r'[.!?]["\')\]]*\s*$')

# Openers that signal the sentence depends on something already said. Starting a
# clip here usually produces a confusing first line.
_DEPENDENT_OPENERS = (
    "but ", "and ", "so ", "because ", "which ", "that is why ", "then ",
    "also ", "however ", "therefore ", "anyway ", "plus ", "or ",
    "it ", "he ", "she ", "they ", "this ", "that ", "those ", "these ",
)

# Padding so speech is never clipped by frame rounding.
LEAD_IN = 0.18
LEAD_OUT = 0.35


@dataclass(frozen=True)
class BoundaryResult:
    start: float
    end: float
    start_snapped_to: str
    end_snapped_to: str
    opens_mid_sentence: bool
    ends_mid_sentence: bool


def _words_sorted(transcript: Transcript) -> list[Word]:
    return sorted(transcript.words(), key=lambda w: w.start)


def sentence_starts(transcript: Transcript) -> list[float]:
    """Timestamps where a new sentence begins."""
    starts: list[float] = []
    for seg in transcript.segments:
        if seg.words:
            starts.append(seg.words[0].start)
        else:
            starts.append(seg.start)

        # Segments can contain several sentences; split on terminal punctuation.
        for i, word in enumerate(seg.words[:-1]):
            if _SENTENCE_END.search(word.word.strip()):
                starts.append(seg.words[i + 1].start)
    return sorted(set(starts))


def sentence_ends(transcript: Transcript) -> list[float]:
    """Timestamps where a sentence completes."""
    ends: list[float] = []
    for seg in transcript.segments:
        for word in seg.words:
            if _SENTENCE_END.search(word.word.strip()):
                ends.append(word.end)
        if seg.words:
            ends.append(seg.words[-1].end)
        else:
            ends.append(seg.end)
    return sorted(set(ends))


def pause_points(transcript: Transcript, *, min_gap: float = 0.35) -> list[float]:
    """Midpoints of silences between words -- natural places to cut."""
    words = _words_sorted(transcript)
    points: list[float] = []
    for prev, nxt in zip(words, words[1:]):
        gap = nxt.start - prev.end
        if gap >= min_gap:
            points.append(prev.end + min(gap / 2.0, 0.4))
    return points


def _nearest(points: list[float], target: float, window: float) -> Optional[float]:
    """Closest point within `window` seconds of target, or None."""
    best, best_delta = None, window
    for p in points:
        delta = abs(p - target)
        if delta <= best_delta:
            best, best_delta = p, delta
    return best


def _word_at(words: list[Word], t: float) -> Optional[Word]:
    for w in words:
        if w.start <= t <= w.end:
            return w
    return None


def is_mid_word(transcript: Transcript, t: float) -> bool:
    """True if t falls strictly inside a spoken word."""
    word = _word_at(_words_sorted(transcript), t)
    return word is not None and (word.start + 0.02) < t < (word.end - 0.02)


def text_starts_dependently(text: str) -> bool:
    """Heuristic: does this opening line lean on missing context?"""
    lowered = text.strip().lower()
    return any(lowered.startswith(opener) for opener in _DEPENDENT_OPENERS)


def refine_boundaries(
    start: float,
    end: float,
    *,
    transcript: Transcript,
    audio: Optional[AudioAnalysis] = None,
    scenes: Optional[list[Scene]] = None,
    min_duration: float = 10.0,
    max_duration: float = 60.0,
    total_duration: float = 0.0,
    search_window: float = 2.5,
) -> BoundaryResult:
    """Snap a proposed window to clean linguistic boundaries.

    Start prefers, in order: a sentence start, a scene cut, a pause.
    End prefers: a sentence end, a pause, a scene cut.
    """
    starts = sentence_starts(transcript)
    ends = sentence_ends(transcript)
    pauses = pause_points(transcript)
    cuts = [s.start for s in (scenes or [])]

    # --- start --------------------------------------------------------------
    new_start, start_kind = start, "requested"
    candidate = _nearest(starts, start, search_window)
    if candidate is not None:
        new_start, start_kind = candidate, "sentence"
    else:
        cut = _nearest(cuts, start, 1.2)
        if cut is not None:
            new_start, start_kind = cut, "scene"
        else:
            pause = _nearest(pauses, start, search_window)
            if pause is not None:
                new_start, start_kind = pause, "pause"

    # --- end ----------------------------------------------------------------
    new_end, end_kind = end, "requested"
    candidate = _nearest(ends, end, search_window)
    if candidate is not None:
        new_end, end_kind = candidate, "sentence"
    else:
        pause = _nearest(pauses, end, search_window)
        if pause is not None:
            new_end, end_kind = pause, "pause"
        else:
            cut = _nearest(cuts, end, 1.2)
            if cut is not None:
                new_end, end_kind = cut, "scene"

    # Never cut through a word, whatever the snapping produced.
    words = _words_sorted(transcript)
    word = _word_at(words, new_start)
    if word and new_start > word.start + 0.02:
        new_start = word.start
        start_kind += "+word"
    word = _word_at(words, new_end)
    if word and new_end < word.end - 0.02:
        new_end = word.end
        end_kind += "+word"

    # --- padding ------------------------------------------------------------
    new_start = max(0.0, new_start - LEAD_IN)
    new_end = new_end + LEAD_OUT
    if total_duration > 0:
        new_end = min(new_end, total_duration)

    # --- duration constraints ----------------------------------------------
    duration = new_end - new_start
    if duration < min_duration:
        # Grow toward whichever side has room, preferring to add lead-in context.
        deficit = min_duration - duration
        grow_start = min(deficit * 0.6, new_start)
        new_start -= grow_start
        new_end += deficit - grow_start
        if total_duration > 0 and new_end > total_duration:
            new_start = max(0.0, new_start - (new_end - total_duration))
            new_end = total_duration

    if new_end - new_start > max_duration:
        # Trim from the front: the payoff is usually at the end.
        trimmed = new_end - max_duration
        snapped = _nearest(starts, trimmed, 2.0)
        new_start = snapped if snapped is not None and snapped < new_end else trimmed

    opening = transcript.text_in_window(new_start, new_start + 6.0)
    ending_text = transcript.text_in_window(max(0.0, new_end - 6.0), new_end)

    return BoundaryResult(
        start=round(max(0.0, new_start), 3),
        end=round(new_end, 3),
        start_snapped_to=start_kind,
        end_snapped_to=end_kind,
        opens_mid_sentence=text_starts_dependently(opening),
        ends_mid_sentence=bool(ending_text) and not _SENTENCE_END.search(ending_text),
    )


def expand_for_context(
    start: float,
    end: float,
    *,
    transcript: Transcript,
    max_duration: float,
    total_duration: float,
    lookback: float = 12.0,
) -> tuple[float, float]:
    """Pull the start earlier to a sentence that does not depend on prior context.

    Used when validation finds the opening incomprehensible on its own. Accuracy
    beats brevity: a slightly longer clip that makes sense is better than a tight
    one that misleads (Architecture.md section 11).
    """
    starts = [s for s in sentence_starts(transcript) if start - lookback <= s < start]
    for candidate in sorted(starts, reverse=True):
        if end - candidate > max_duration:
            continue
        opening = transcript.text_in_window(candidate, candidate + 6.0)
        if opening and not text_starts_dependently(opening):
            return max(0.0, candidate - LEAD_IN), end
    return start, end
