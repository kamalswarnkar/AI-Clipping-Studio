"""Bounded multimodal visual reasoning.

OpenCV already gives us faces, motion and scene cuts across the whole video for
free. This stage adds *semantic* visual understanding -- reactions, gestures,
on-screen text -- and is deliberately expensive, so it runs only on the strongest
candidates and only on a few sampled frames each (Architecture.md section 7).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from ...models.domain import Candidate, VisionObservation
from ...video import ffmpeg
from ..base import ProviderError, VisionProvider
from ..prompts import clip_prompts as P

log = logging.getLogger(__name__)


def _sample_times(candidate: Candidate, count: int) -> list[float]:
    """Evenly spaced timestamps inside the candidate, avoiding the very edges."""
    duration = candidate.duration
    if duration <= 0 or count <= 0:
        return [candidate.start]
    if count == 1:
        return [candidate.start + duration / 2.0]
    inset = min(0.8, duration * 0.1)
    span = duration - 2 * inset
    return [
        candidate.start + inset + span * i / (count - 1) for i in range(count)
    ]


def analyze_candidates(
    candidates: list[Candidate],
    *,
    source: Path,
    frames_dir: Path,
    provider: VisionProvider,
    max_candidates: int = 20,
    frames_per_candidate: int = 4,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, VisionObservation]:
    """Attach visual observations to the top candidates.

    Failures are non-fatal: a candidate simply goes to the LLM without a visual
    note rather than blocking the run.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Candidates arrive ranked; only the strongest justify the cost.
    targets = candidates[:max_candidates]
    observations: dict[str, VisionObservation] = {}

    for i, candidate in enumerate(targets):
        image_paths: list[Path] = []
        try:
            for j, t in enumerate(_sample_times(candidate, frames_per_candidate)):
                frame_path = frames_dir / f"{candidate.id}_{j}.jpg"
                if not frame_path.exists():
                    ffmpeg.extract_frame(source, t, frame_path, width=512)
                image_paths.append(frame_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Frame extraction failed for %s: %s", candidate.id, exc)
            if progress:
                progress((i + 1) / len(targets), "frame extraction failed")
            continue

        try:
            raw = provider.describe_frames(
                image_paths,
                system=P.VISION_SYSTEM,
                prompt=P.VISION_PROMPT,
                schema=P.VISION_SCHEMA,
            )
        except ProviderError as exc:
            log.warning("Vision analysis failed for %s: %s", candidate.id, exc)
            if progress:
                progress((i + 1) / len(targets), "vision call failed")
            continue

        try:
            interest = float(raw.get("visual_interest", 0.5))
        except (TypeError, ValueError):
            interest = 0.5

        events = raw.get("notable_events") or []
        if not isinstance(events, list):
            events = []

        try:
            people = int(raw.get("people_visible", 0))
        except (TypeError, ValueError):
            people = 0

        observation = VisionObservation(
            candidate_id=candidate.id,
            description=str(raw.get("description", ""))[:600],
            people_visible=max(0, people),
            on_screen_text=str(raw.get("on_screen_text", ""))[:300],
            notable_events=[str(e)[:120] for e in events][:8],
            visual_interest=max(0.0, min(1.0, interest)),
            is_talking_head=bool(raw.get("is_talking_head", True)),
        )
        observations[candidate.id] = observation
        candidate.vision = observation

        # Let a strong or weak visual read nudge the cheap score.
        candidate.breakdown.visual_interest = round(
            0.5 * candidate.breakdown.visual_interest + 0.5 * observation.visual_interest,
            4,
        )

        if progress:
            progress((i + 1) / len(targets), f"{len(observations)} analysed")

    return observations
