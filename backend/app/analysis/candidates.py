"""Candidate moment generation and cheap multi-factor scoring.

This is the funnel that keeps the pipeline affordable: a long video produces
30-100 candidate windows here using only transcript/audio/scene signals, and
only the strongest survive to the expensive LLM and vision stages
(Architecture.md section 9).

Nothing in this module calls a model.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from ..models.domain import (
    AudioAnalysis,
    Candidate,
    Diarization,
    Scene,
    ScoreBreakdown,
    Transcript,
    VisualAnalysis,
)
from . import audio_analysis as aa
from . import conflict
from . import scenes as scene_utils
from . import visual as visual_utils
from .boundaries import refine_boundaries, text_starts_dependently

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r'[.!?]["\')\]]*\s*$')

# Structural cues that a moment is worth examining. These only *nominate*
# windows -- they never select a final clip, so an emotive word alone cannot
# promote weak content (Architecture.md section 9).
_TRIGGER_PATTERNS: dict[str, re.Pattern[str]] = {
    "story": re.compile(
        r"\b(so here is what happened|here is what happened|let me tell you|"
        r"the story|what happened was|i remember when|years ago|back when|"
        r"the first time)\b",
        re.I,
    ),
    "lesson": re.compile(
        r"\b(the mistake|what i learned|the lesson|the biggest thing|"
        r"what people get wrong|people keep missing|the truth is|"
        r"here is the thing|the reality is)\b",
        re.I,
    ),
    "claim": re.compile(
        r"\b(i think|i believe|in my opinion|the fact is|i would argue|"
        r"i will defend|the point is)\b",
        re.I,
    ),
    "disagreement": re.compile(
        r"\b(but surely|i disagree|that is not true|you are wrong|"
        r"that is fair, but|i push back|actually, no|i would challenge)\b",
        re.I,
    ),
    "explanation": re.compile(
        r"\b(because|the reason|which means|that is why|the way it works|"
        r"in other words|for example)\b",
        re.I,
    ),
    "conclusion": re.compile(
        r"\b(so the takeaway|bottom line|in the end|what that means|"
        r"that is the whole|the answer is)\b",
        re.I,
    ),
    "confrontation": re.compile(
        r"(don't touch|do not touch|get out|back off|shut up|let go|"
        r"give it back|you're lying|youre lying|that's a lie|thats a lie|"
        r"excuse me|hey hey|whoa|stop it|get off|get away from|"
        r"who do you think|how dare you|are you serious|"
        r"call the police|i'm calling|im calling)",
        re.I,
    ),
    "challenge": re.compile(
        r"(answer the question|why won't you|why wont you|prove it|"
        r"do you support|do you think|what about|explain to me|"
        r"can you name|justify)",
        re.I,
    ),
    "number": re.compile(
        r"\b\d{1,3}(?:[.,]\d+)?\s?(?:percent|%|x|times|million|billion|"
        r"thousand|dollars|years|months|weeks|days)\b",
        re.I,
    ),
}

_QUESTION = re.compile(r"\?\s*$")

# How many boundary variants to keep per anchor sentence.
VARIANTS_PER_ANCHOR = 3

# Filler that signals logistics/small talk rather than content worth clipping.
_LOW_VALUE = re.compile(
    r"\b(welcome back|thanks for having me|happy to be here|before we start|"
    r"quick reminder|let us take a short break|see you next week|"
    r"swap the microphone|sorry to interrupt|sponsored by|subscribe|"
    r"episodes recorded|go out in january|we record every|"
    r"back after the break|let us pause|microphone battery|"
    r"the boring logistics|flew in this morning|the flight was delayed|"
    # Greetings and channel intros: a reel that opens on one is dead.
    r"welcome to the|welcome everybody|thanks for watching|"
    r"like and subscribe|hit the bell|in this video|today we are here|"
    r"my name is [a-z]+ and)\b",
    re.I,
)


def trim_filler_opening(
    start: float,
    end: float,
    *,
    transcript: Transcript,
    min_duration: float,
    max_duration: float = 60.0,
    total_duration: float = 0.0,
) -> tuple[float, float, bool]:
    """Drop an opening sentence that is pure logistics or small talk.

    A deterministic guard: a clip that opens on "we have three more episodes
    recorded" reads as a mistake regardless of how the rest scores, and the
    model cannot be relied on to notice.

    If removing the filler would leave the clip under the minimum duration, the
    end is extended to the next sentence boundary instead of keeping the filler.
    Returns (start, end, trimmed).
    """
    sentences = build_sentences(transcript)
    # Overlap-based, not start-based: the sentence a clip opens on frequently
    # begins a fraction of a second before the cut, and a start-based filter
    # would skip exactly the sentence we need to inspect.
    overlapping = [s for s in sentences if s.end > start + 0.2 and s.start < end]

    trimmed = False
    for position, sentence in enumerate(overlapping):
        if not _LOW_VALUE.search(sentence.text):
            # A tiny pleasantry ("Thank you.") often sits in front of the real
            # filler; look one sentence past it rather than giving up.
            following = overlapping[position + 1] if position + 1 < len(overlapping) else None
            if (
                len(sentence.text.split()) <= 3
                and following is not None
                and _LOW_VALUE.search(following.text)
            ):
                continue
            break

        new_start = sentence.end
        new_end = end

        # Recover the minimum duration by extending forward rather than by
        # keeping filler at the front.
        if new_end - new_start < min_duration:
            needed = new_start + min_duration
            following = [s.end for s in sentences if s.end >= needed]
            candidate_end = following[0] if following else needed
            if total_duration:
                candidate_end = min(candidate_end, total_duration)
            if (
                candidate_end - new_start > max_duration
                or candidate_end - new_start < min_duration
            ):
                break  # cannot fix this one; leave it for the LLM to judge
            new_end = candidate_end

        start, end = new_start, new_end
        trimmed = True

    return start, end, trimmed


class _Sentence:
    """A sentence with its timing, speaker and text."""

    __slots__ = ("start", "end", "text", "speaker")

    def __init__(self, start: float, end: float, text: str, speaker: Optional[str]) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.speaker = speaker

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def build_sentences(
    transcript: Transcript, diarization: Optional[Diarization] = None
) -> list[_Sentence]:
    """Split the transcript into sentence units with speaker attribution."""
    sentences: list[_Sentence] = []

    for seg in transcript.segments:
        if not seg.words:
            sentences.append(
                _Sentence(
                    seg.start,
                    seg.end,
                    seg.text.strip(),
                    diarization.speaker_at(seg.start) if diarization else None,
                )
            )
            continue

        buf: list[str] = []
        buf_start = seg.words[0].start
        for word in seg.words:
            buf.append(word.word.strip())
            if _SENTENCE_END.search(word.word.strip()):
                text = " ".join(buf).strip()
                if text:
                    sentences.append(
                        _Sentence(
                            buf_start,
                            word.end,
                            text,
                            diarization.speaker_at(buf_start) if diarization else None,
                        )
                    )
                buf = []
                buf_start = word.end
        if buf:
            text = " ".join(buf).strip()
            if text:
                sentences.append(
                    _Sentence(
                        buf_start,
                        seg.words[-1].end,
                        text,
                        diarization.speaker_at(buf_start) if diarization else None,
                    )
                )

    return [s for s in sentences if s.text]


def _anchor_score(sentence: _Sentence, following: str) -> tuple[float, str]:
    """How promising is this sentence as the *start* of a clip?"""
    text = sentence.text
    score, trigger = 0.0, "generic"

    for name, pattern in _TRIGGER_PATTERNS.items():
        if pattern.search(text):
            weight = {
                "confrontation": 0.95,
                "challenge": 0.8,
                "story": 0.9,
                "lesson": 0.85,
                "disagreement": 0.8,
                "conclusion": 0.7,
                "number": 0.65,
                "claim": 0.5,
                "explanation": 0.35,
            }[name]
            if weight > score:
                score, trigger = weight, name

    # A question is a strong opener when an answer follows.
    if _QUESTION.search(text) and len(following.split()) > 12:
        if score < 0.75:
            score, trigger = 0.75, "question"

    if _LOW_VALUE.search(text):
        score -= 0.6

    # Dependent openers ("But it...") make poor first lines.
    if text_starts_dependently(text):
        score -= 0.25

    return score, trigger


def _lexical_density(text: str) -> float:
    """Fraction of content-bearing words, as a proxy for information density."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    stop = {
        "the", "a", "an", "and", "or", "but", "so", "if", "of", "to", "in", "on",
        "for", "with", "is", "are", "was", "were", "be", "been", "it", "that",
        "this", "i", "you", "we", "they", "he", "she", "at", "as", "by", "from",
        "not", "do", "did", "does", "have", "has", "had", "will", "would", "can",
        "could", "just", "like", "know", "think", "really", "very", "there",
        "what", "when", "how", "then", "than", "about", "my", "me", "our",
    }
    content = [w for w in words if w not in stop and len(w) > 2]
    return float(np.clip(len(content) / len(words) / 0.55, 0.0, 1.0))


def _score_window(
    *,
    text: str,
    opening: str,
    ending: str,
    start: float,
    end: float,
    trigger: str,
    audio: Optional[AudioAnalysis],
    visual: Optional[VisualAnalysis],
    scene_list: list[Scene],
    speakers: list[str],
) -> ScoreBreakdown:
    """Cheap approximation of the section 12 factors.

    These are heuristics, not judgements. The LLM stage revisits clarity and
    completeness with actual comprehension; this exists to rank the pool.
    """
    words = text.split()
    word_count = len(words)
    duration = max(1e-6, end - start)
    wps = word_count / duration

    # --- content clarity: enough speech, not rushed, not sparse -------------
    clarity = float(np.clip(wps / 2.6, 0.0, 1.0))
    if wps < 0.9:  # long silences or very sparse speech
        clarity *= 0.6
    if _LOW_VALUE.search(text):
        clarity *= 0.35

    # --- standalone completeness -------------------------------------------
    completeness = 0.5
    if not text_starts_dependently(opening):
        completeness += 0.25
    if _SENTENCE_END.search(ending.strip()):
        completeness += 0.25
    completeness = float(np.clip(completeness, 0.0, 1.0))

    # --- narrative structure -----------------------------------------------
    structure = {
        "confrontation": 0.85,
        "challenge": 0.8,
        "story": 0.9,
        "lesson": 0.85,
        "question": 0.8,
        "disagreement": 0.8,
        "conclusion": 0.75,
        "number": 0.6,
        "claim": 0.55,
        "explanation": 0.45,
        "generic": 0.3,
    }.get(trigger, 0.3)
    # A back-and-forth exchange usually reads as more structured than a monologue.
    if len(speakers) > 1:
        structure = min(1.0, structure + 0.1)

    # --- supporting signals -------------------------------------------------
    visual_score = (
        visual_utils.visual_interest(visual, start, end) if visual else 0.45
    )
    dynamics = aa.dynamics_score(audio, start, end) if audio else 0.5
    reaction = aa.reaction_score(audio, start, end) if audio else 0.0

    density = _lexical_density(text)

    # --- opening / ending strength -----------------------------------------
    opening_strength = 0.35
    if _TRIGGER_PATTERNS["story"].search(opening) or _QUESTION.search(opening.strip()):
        opening_strength = 0.9
    elif any(p.search(opening) for p in _TRIGGER_PATTERNS.values()):
        opening_strength = 0.7
    if text_starts_dependently(opening):
        opening_strength *= 0.5

    ending_payoff = 0.35
    if _SENTENCE_END.search(ending.strip()):
        ending_payoff = 0.7
    if any(
        _TRIGGER_PATTERNS[k].search(ending) for k in ("conclusion", "lesson", "number")
    ):
        ending_payoff = 0.95
    if reaction > 0.25:  # audience responded right at the end
        ending_payoff = min(1.0, ending_payoff + 0.2)

    # Excessive cutting inside a short clip is visually noisy.
    cuts = scene_utils.scene_count_between(scene_list, start, end)
    if cuts > 6:
        visual_score *= 0.8

    return ScoreBreakdown(
        content_clarity=round(clarity, 4),
        standalone_completeness=round(completeness, 4),
        narrative_structure=round(structure, 4),
        visual_interest=round(visual_score, 4),
        audio_dynamics=round(dynamics, 4),
        emotional_reaction=round(reaction, 4),
        information_density=round(density, 4),
        opening_strength=round(opening_strength, 4),
        ending_payoff=round(ending_payoff, 4),
    )


def generate_candidates(
    *,
    transcript: Transcript,
    diarization: Optional[Diarization],
    audio: Optional[AudioAnalysis],
    visual: Optional[VisualAnalysis],
    scene_list: list[Scene],
    weights: dict[str, float],
    min_duration: float,
    opening_window: float = 3.0,
    opening_weight: float = 0.22,
    conflict_weight: float = 0.18,
    max_duration: float,
    total_duration: float,
    pool_max: int = 80,
) -> list[Candidate]:
    """Produce a ranked candidate pool.

    Windows are anchored on promising sentence starts and grown to end on a
    sentence boundary, so every candidate is already roughly well-formed before
    any model sees it.
    """
    sentences = build_sentences(transcript, diarization)
    if not sentences:
        return []

    seen: set[tuple[int, int]] = set()
    candidates: list[Candidate] = []
    # Anchor index per candidate, so thinning can distinguish "another view of
    # the same moment" from "a different moment".
    anchors: list[int] = []

    for i, sentence in enumerate(sentences):
        following = " ".join(s.text for s in sentences[i + 1 : i + 4])
        anchor, trigger = _anchor_score(sentence, following)

        # Anchors below the bar are still sampled sparsely so a video with no
        # obvious cue words still yields a pool.
        if anchor < 0.3 and i % 4 != 0:
            continue

        # Grow the window sentence by sentence, emitting each valid length.
        for j in range(i, min(len(sentences), i + 40)):
            end_sentence = sentences[j]
            raw_start = sentence.start
            raw_end = end_sentence.end
            duration = raw_end - raw_start

            if duration < min_duration:
                continue
            if duration > max_duration:
                break

            key = (int(raw_start * 4), int(raw_end * 4))
            if key in seen:
                continue
            seen.add(key)

            refined = refine_boundaries(
                raw_start,
                raw_end,
                transcript=transcript,
                audio=audio,
                scenes=scene_list,
                min_duration=min_duration,
                max_duration=max_duration,
                total_duration=total_duration,
            )
            if refined.end - refined.start < min_duration:
                continue

            text = transcript.text_in_window(refined.start, refined.end)
            if len(text.split()) < 18:  # too little actually said
                continue

            opening = transcript.text_in_window(refined.start, refined.start + 6.0)
            ending = transcript.text_in_window(max(0.0, refined.end - 7.0), refined.end)
            window_speakers = (
                diarization.speakers_between(refined.start, refined.end)
                if diarization
                else []
            )

            breakdown = _score_window(
                text=text,
                opening=opening,
                ending=ending,
                start=refined.start,
                end=refined.end,
                trigger=trigger,
                audio=audio,
                visual=visual,
                scene_list=scene_list,
                speakers=window_speakers,
            )

            # Conflict intensity and opening punch, from cheap signals.
            profile = conflict.profile_window(
                transcript=transcript,
                audio=audio,
                diarization=diarization,
                start=refined.start,
                end=refined.end,
                opening_window=opening_window,
            )

            # A clip nobody watches past three seconds has no other quality, so
            # the opening is weighted directly rather than averaged away.
            breakdown.opening_strength = round(
                min(1.0, 0.45 * breakdown.opening_strength + 0.55 * profile.opening_punch),
                4,
            )
            breakdown.emotional_reaction = round(
                min(1.0, max(breakdown.emotional_reaction, profile.intensity)), 4
            )

            base = breakdown.weighted_total(weights)
            # Blend in the anchor: a strong structural cue is real evidence.
            base = 0.82 * base + 0.18 * float(np.clip(anchor, 0.0, 1.0))

            residual = max(0.0, 1.0 - opening_weight - conflict_weight)
            score = (
                residual * base
                + opening_weight * profile.opening_punch
                + conflict_weight * profile.intensity
            )

            anchors.append(i)
            candidates.append(
                Candidate(
                    id=f"candidate_{len(candidates) + 1:03d}",
                    start=refined.start,
                    end=refined.end,
                    text=text,
                    context_before=transcript.text_between(
                        max(0.0, refined.start - 25.0), refined.start
                    )[-600:],
                    context_after=transcript.text_between(
                        refined.end, refined.end + 25.0
                    )[:600],
                    speakers=window_speakers,
                    scene_count=scene_utils.scene_count_between(
                        scene_list, refined.start, refined.end
                    ),
                    audio_events=sorted(
                        {
                            e.kind
                            for e in (audio.events_between(refined.start, refined.end) if audio else [])
                            if e.kind != "silence"
                        }
                    ),
                    breakdown=breakdown,
                    heuristic_score=round(float(score), 4),
                    trigger=trigger,
                    opening_punch=profile.opening_punch,
                    conflict_intensity=profile.intensity,
                    conflict_note=profile.summary,
                )
            )

    # Keep a few boundary variants per anchor, then rank.
    #
    # Thinning across *different* moments here would be a mistake: it would
    # decide which window survives using only the heuristic score, before the
    # LLM has judged meaning. That is how a well-bounded story loses to a
    # badly-bounded sibling that merely scores higher on cheap signals. Real
    # cross-moment deduplication happens in selection.py, after quality is known.
    by_anchor: dict[int, list[Candidate]] = {}
    for anchor, candidate in zip(anchors, candidates):
        by_anchor.setdefault(anchor, []).append(candidate)

    kept: list[Candidate] = []
    for group in by_anchor.values():
        group.sort(key=lambda c: c.heuristic_score, reverse=True)
        # Distinct lengths from the same start are genuinely different edits;
        # keep the strongest few so the LLM can pick the right boundaries.
        kept.extend(_thin_overlapping(group, max_overlap=0.92)[:VARIANTS_PER_ANCHOR])

    kept.sort(key=lambda c: c.heuristic_score, reverse=True)
    capped = kept[:pool_max]

    # Renumber for stable, readable ids in prompts and logs.
    for idx, candidate in enumerate(capped, start=1):
        candidate.id = f"candidate_{idx:03d}"

    log.info(
        "Generated %d candidates (from %d windows) over %d sentences",
        len(capped),
        len(candidates),
        len(sentences),
    )
    return capped


def _thin_overlapping(
    candidates: list[Candidate], *, max_overlap: float
) -> list[Candidate]:
    """Keep the best candidate among heavily overlapping windows."""
    kept: list[Candidate] = []
    for candidate in candidates:  # already sorted best-first
        redundant = False
        for existing in kept:
            overlap = min(candidate.end, existing.end) - max(candidate.start, existing.start)
            if overlap <= 0:
                continue
            shorter = min(candidate.duration, existing.duration)
            if shorter > 0 and overlap / shorter > max_overlap:
                redundant = True
                break
        if not redundant:
            kept.append(candidate)
    return kept
