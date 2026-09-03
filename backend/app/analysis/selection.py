"""Final clip selection: deduplication, ranking and honest counting.

Two product rules drive this module:

* Never ship two clips of the same moment (Architecture.md section 13).
* Never pad the list with weak clips to hit a requested number. If only eight
  moments are strong, return eight and say so (section 14).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..models.domain import (
    Candidate,
    ClipPlan,
    ContextDependency,
    LLMEvaluation,
)

log = logging.getLogger(__name__)

# A clip must clear this blended score to be shippable at all.
QUALITY_FLOOR = 0.42

_WORD = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his i if in
    is it its me my not of on or our she so that the their them then there these they
    this to too us was we were what when which who will with would you your""".split()
)


@dataclass
class SelectionResult:
    clips: list[ClipPlan]
    requested: int
    strong_available: int
    duplicates_removed: int = 0
    below_floor: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def fell_short(self) -> bool:
        return len(self.clips) < self.requested


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def text_similarity(a: str, b: str) -> float:
    """Jaccard overlap of content words.

    Cheap, deterministic and language-agnostic. It catches the common case --
    two windows quoting the same sentences -- without needing an embedding model.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def time_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Overlap as a fraction of the shorter clip."""
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return 0.0
    shorter = min(a_end - a_start, b_end - b_start)
    return overlap / shorter if shorter > 0 else 0.0


def is_duplicate(
    a: ClipPlan,
    b: ClipPlan,
    *,
    iou_threshold: float,
    text_threshold: float,
) -> tuple[bool, str]:
    """Decide whether two clips represent the same moment."""
    overlap = time_overlap_ratio(a.start, a.end, b.start, b.end)
    if overlap >= iou_threshold:
        return True, f"overlaps {overlap * 100:.0f}% in time"

    similarity = text_similarity(a.transcript, b.transcript)
    if similarity >= text_threshold:
        return True, f"transcript {similarity * 100:.0f}% similar"

    # Adjacent windows that quote much of the same speech are still the same beat.
    if overlap > 0.15 and similarity > text_threshold * 0.75:
        return True, "adjacent window covering the same moment"

    return False, ""


def select_clips(
    *,
    candidates: list[Candidate],
    evaluations: list[LLMEvaluation],
    requested: int,
    iou_threshold: float = 0.35,
    text_threshold: float = 0.72,
    llm_available: bool = True,
) -> SelectionResult:
    """Rank, deduplicate and cut off at the honest number of strong clips."""
    eval_by_id = {e.candidate_id: e for e in evaluations}

    scored: list[tuple[float, ClipPlan]] = []

    for candidate in candidates:
        evaluation: Optional[LLMEvaluation] = eval_by_id.get(candidate.id)

        if evaluation is None:
            if llm_available:
                # The model saw it and returned nothing usable: do not promote it.
                continue
            blended = candidate.heuristic_score
            start, end = candidate.start, candidate.end
            topic, reason = "", "Selected without LLM evaluation."
            dependency = ContextDependency.MEDIUM
        else:
            # The LLM judges meaning; heuristics judge deliverability. Both count.
            blended = 0.62 * evaluation.quality_score + 0.38 * candidate.heuristic_score
            if evaluation.recommended:
                blended += 0.10
            if evaluation.context_dependency == ContextDependency.HIGH:
                blended -= 0.12
            if not evaluation.complete_thought:
                blended -= 0.10
            start, end = evaluation.start, evaluation.end
            topic, reason = evaluation.topic, evaluation.reason
            dependency = evaluation.context_dependency

        blended = max(0.0, min(1.0, blended))

        plan = ClipPlan(
            id=candidate.id,
            index=0,  # assigned after ranking
            start=start,
            end=end,
            topic=topic,
            transcript=candidate.text,
            speakers=candidate.speakers,
            score=round(blended, 4),
            reason=reason,
            context_dependency=dependency,
            breakdown=candidate.breakdown,
            source_candidate_id=candidate.id,
        )
        scored.append((blended, plan))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    # --- quality floor ------------------------------------------------------
    above_floor = [plan for score, plan in scored if score >= QUALITY_FLOOR]
    below_floor = len(scored) - len(above_floor)

    # --- deduplication ------------------------------------------------------
    kept: list[ClipPlan] = []
    duplicates = 0
    for plan in above_floor:  # best-first, so the survivor is the stronger one
        duplicate_of = None
        for existing in kept:
            same, why = is_duplicate(
                plan, existing, iou_threshold=iou_threshold, text_threshold=text_threshold
            )
            if same:
                duplicate_of = (existing, why)
                break
        if duplicate_of:
            duplicates += 1
            log.debug(
                "Dropping %s as duplicate of %s (%s)",
                plan.id,
                duplicate_of[0].id,
                duplicate_of[1],
            )
            continue
        kept.append(plan)

    strong_available = len(kept)
    final = kept[:requested]

    # Present clips in source order: a viewer scanning results expects the
    # timeline, not a ranking they cannot see.
    final.sort(key=lambda p: p.start)
    for i, plan in enumerate(final, start=1):
        plan.index = i

    notes: list[str] = []
    if strong_available < requested:
        notes.append(
            f"Only {strong_available} genuinely strong moment"
            f"{'s' if strong_available != 1 else ''} were found in this video, "
            f"fewer than the {requested} requested. Weak clips were not added to "
            f"make up the number."
        )
    if duplicates:
        notes.append(f"{duplicates} near-duplicate moment(s) were merged.")

    log.info(
        "Selected %d clips (requested %d, %d above floor, %d duplicates dropped)",
        len(final),
        requested,
        len(above_floor),
        duplicates,
    )

    return SelectionResult(
        clips=final,
        requested=requested,
        strong_available=strong_available,
        duplicates_removed=duplicates,
        below_floor=below_floor,
        notes=notes,
    )
