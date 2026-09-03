"""Conflict intensity and opening-hook detection.

Short-form retention is decided in the first seconds: a viewer who is not caught
by the opening never sees the payoff. This module scores two things a clip is
selected on:

* **conflict intensity** -- shouting, interruption, rapid back-and-forth,
  confrontational language across the whole clip
* **opening punch** -- how much of that lands inside the first few seconds

Both are computed from signals we already have (loudness envelope, speaker turns,
transcript). Nothing here calls a model, so it costs effectively nothing compared
with the multimodal vision pass it replaces for this purpose.

Real-world confrontation footage is loud and noisy throughout, so every measure
here is *relative to the recording's own baseline* rather than an absolute
threshold -- an outdoor rally and a quiet studio interview must both work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models.domain import AudioAnalysis, Diarization, Transcript

# Words that mark confrontation rather than mere emphasis. Deliberately about
# *interaction* (accusation, refusal, challenge) rather than topic.
_CONFRONTATION = re.compile(
    r"\b(you|your|you're|youre)\b.{0,28}\b(lying|lie|wrong|racist|stupid|idiot|"
    r"crazy|insane|disgusting|pathetic|coward|fake|fraud|criminal|liar)\b"
    r"|\b(shut up|get out|back off|leave|go away|don't touch|do not touch|"
    r"stop it|stop touching|let go|give it back|get off|get away|move|"
    r"excuse me|hey hey|whoa whoa)\b"
    r"|\b(that's a lie|thats a lie|you're lying|youre lying|prove it|"
    r"answer the question|why won't you|why wont you|admit it|"
    r"you refused|you ignored|you blocked)\b"
    r"|\b(arrest|assault|attacked|grabbed|shoved|pushed|threatened|"
    r"stole|steal|theft|trespassing|illegal)\b",
    re.I,
)

_CHALLENGE = re.compile(
    r"\b(do you|did you|why do you|why did you|how can you|what about|"
    r"are you saying|so you think|explain|justify|defend)\b",
    re.I,
)

_DISAGREE = re.compile(
    r"\b(no|nope|absolutely not|that's not|thats not|i disagree|wrong|"
    r"actually|but|however|nonsense|ridiculous)\b",
    re.I,
)


@dataclass(frozen=True)
class ConflictProfile:
    """Conflict measures for one window, each in 0..1."""

    intensity: float
    shouting: float
    exchange_rate: float
    confrontation: float
    opening_punch: float

    @property
    def summary(self) -> str:
        """Short human-readable note for Info.txt and prompts."""
        bits: list[str] = []
        if self.shouting > 0.5:
            bits.append("raised voices")
        if self.exchange_rate > 0.5:
            bits.append("rapid back-and-forth")
        if self.confrontation > 0.4:
            bits.append("direct confrontation")
        if self.opening_punch > 0.6:
            bits.append("strong opening")
        return ", ".join(bits)


def _energy_percentiles(audio: AudioAnalysis) -> tuple[float, float]:
    """Baseline and loud reference levels for this recording."""
    if not audio.rms_db:
        return -30.0, -12.0
    arr = np.asarray(audio.rms_db, dtype=np.float32)
    voiced = arr[arr > (arr.max() - 35.0)]
    if voiced.size == 0:
        voiced = arr
    return float(np.percentile(voiced, 50)), float(np.percentile(voiced, 95))


def shouting_score(
    audio: Optional[AudioAnalysis], start: float, end: float
) -> float:
    """Fraction of the window spent well above the speaker's normal level.

    Measured against the recording's own loud reference, so it works on a noisy
    street recording as well as a studio one.
    """
    if audio is None or not audio.rms_db:
        return 0.0
    window = audio.energy_between(start, end)
    if not window:
        return 0.0

    median, loud = _energy_percentiles(audio)
    span = max(3.0, loud - median)
    arr = np.asarray(window, dtype=np.float32)

    # How far into the upper range this window sits, on average and at peak.
    above = np.clip((arr - median) / span, 0.0, 1.5)
    sustained = float(np.mean(above > 0.75))
    peak = float(np.clip(np.percentile(above, 95), 0.0, 1.0))

    return float(np.clip(0.6 * sustained + 0.4 * peak, 0.0, 1.0))


def exchange_rate_score(
    diarization: Optional[Diarization], start: float, end: float
) -> float:
    """Speaker alternations per minute, normalised.

    A heated argument is structurally short alternating turns; a monologue is
    not. This is the cheapest reliable indicator of a live dispute.
    """
    if diarization is None or not diarization.turns:
        return 0.0
    duration = max(1e-6, end - start)

    turns = [t for t in diarization.turns if t.end > start and t.start < end]
    if len(turns) < 2:
        return 0.0

    switches = sum(
        1 for a, b in zip(turns, turns[1:]) if a.speaker != b.speaker
    )
    per_minute = switches / (duration / 60.0)
    # ~12 switches/min is a brisk argument; treat that as the top of the scale.
    return float(np.clip(per_minute / 12.0, 0.0, 1.0))


def confrontation_score(text: str) -> float:
    """Confrontational language density in the window's speech."""
    if not text.strip():
        return 0.0
    words = max(8, len(text.split()))

    hits = len(_CONFRONTATION.findall(text)) * 1.0
    hits += len(_CHALLENGE.findall(text)) * 0.45
    hits += len(_DISAGREE.findall(text)) * 0.2

    # Normalise per 40 words so long clips are not automatically favoured.
    return float(np.clip(hits / (words / 40.0) / 3.0, 0.0, 1.0))


def opening_punch_score(
    *,
    transcript: Transcript,
    audio: Optional[AudioAnalysis],
    diarization: Optional[Diarization],
    start: float,
    window: float = 3.0,
) -> float:
    """How strongly the FIRST seconds grab attention.

    This is the metric that decides whether a reel is watched at all, so it is
    scored on its own rather than being diluted into a whole-clip average.
    """
    end = start + window
    opening_text = transcript.text_in_window(start, end)

    # Dead air at the start is fatal for a short-form clip.
    if not opening_text.strip():
        return 0.05

    score = 0.0

    # Someone is already shouting / at full intensity as the clip opens.
    score += 0.40 * shouting_score(audio, start, end)

    # Confrontational or challenging words in the very first line.
    score += 0.30 * confrontation_score(opening_text)

    # An exchange already in progress reads as "walking into" a fight.
    score += 0.15 * exchange_rate_score(diarization, start, end + 2.0)

    # Speech starts immediately rather than after a pause.
    words = transcript.words_between(start, end)
    if words:
        lead_in = words[0].start - start
        score += 0.15 * float(np.clip(1.0 - lead_in / 1.2, 0.0, 1.0))

    return float(np.clip(score, 0.0, 1.0))


def profile_window(
    *,
    transcript: Transcript,
    audio: Optional[AudioAnalysis],
    diarization: Optional[Diarization],
    start: float,
    end: float,
    opening_window: float = 3.0,
) -> ConflictProfile:
    """Full conflict profile for one candidate window."""
    text = transcript.text_in_window(start, end)

    shouting = shouting_score(audio, start, end)
    exchange = exchange_rate_score(diarization, start, end)
    confrontation = confrontation_score(text)
    opening = opening_punch_score(
        transcript=transcript,
        audio=audio,
        diarization=diarization,
        start=start,
        window=opening_window,
    )

    intensity = float(
        np.clip(
            0.34 * shouting + 0.28 * exchange + 0.38 * confrontation,
            0.0,
            1.0,
        )
    )

    return ConflictProfile(
        intensity=round(intensity, 4),
        shouting=round(shouting, 4),
        exchange_rate=round(exchange, 4),
        confrontation=round(confrontation, 4),
        opening_punch=round(opening, 4),
    )
