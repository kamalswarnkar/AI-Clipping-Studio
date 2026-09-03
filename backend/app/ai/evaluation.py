"""LLM candidate evaluation and context validation.

This module is the boundary between "AI decides WHAT" and "code decides HOW".
Model output is treated as a *proposal*: every field is re-validated, every
timestamp is clamped into a legal range, and anything unusable is dropped rather
than passed downstream.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from ..config import get_settings
from ..models.domain import (
    Candidate,
    ContextDependency,
    LLMEvaluation,
    Transcript,
)
from .base import LLMProvider, ProviderError
from .prompts import clip_prompts as P

log = logging.getLogger(__name__)

# How far the model may move a boundary from the candidate window. Anything
# beyond this is a hallucinated timestamp, not an editorial judgement.
MAX_BOUNDARY_SHIFT = 15.0


def _clamp_evaluation(
    raw: dict,
    candidate: Candidate,
    *,
    min_duration: float,
    max_duration: float,
    total_duration: float,
) -> Optional[LLMEvaluation]:
    """Validate one raw evaluation object. Returns None if unusable.

    Schema-constrained decoding guarantees the *shape*, not the *values*: models
    routinely return timestamps from a different candidate, or a 400-second
    "clip". Those are caught here.
    """
    try:
        start = float(raw.get("start", candidate.start))
        end = float(raw.get("end", candidate.end))
        score = float(raw.get("quality_score", 0.0))
    except (TypeError, ValueError):
        log.warning("Evaluation for %s had non-numeric fields", candidate.id)
        return None

    # Reject nonsense outright rather than trying to repair it.
    if not (0.0 <= start < end):
        log.warning(
            "Evaluation for %s had invalid range %.1f-%.1f; using candidate window",
            candidate.id,
            start,
            end,
        )
        start, end = candidate.start, candidate.end

    # Boundaries may only move within a sane neighbourhood of the proposal.
    start = max(candidate.start - MAX_BOUNDARY_SHIFT, min(start, candidate.start + MAX_BOUNDARY_SHIFT))
    end = max(candidate.end - MAX_BOUNDARY_SHIFT, min(end, candidate.end + MAX_BOUNDARY_SHIFT))

    start = max(0.0, start)
    if total_duration > 0:
        end = min(end, total_duration)

    if end - start < min_duration:
        end = min(start + min_duration, total_duration or (start + min_duration))
    if end - start > max_duration:
        end = start + max_duration
    if end <= start:
        return None

    dependency = str(raw.get("context_dependency", "medium")).lower()
    if dependency not in ("low", "medium", "high"):
        dependency = "medium"

    # Models sometimes return an identifier-style topic ("retention_over_growth").
    # Normalise it here so both the UI and the exported Info.txt read as prose.
    topic = re.sub(r"[_\-]+", " ", str(raw.get("topic", ""))).strip()[:200]

    return LLMEvaluation(
        candidate_id=candidate.id,
        recommended=bool(raw.get("recommended", False)),
        start=round(start, 3),
        end=round(end, 3),
        reason=str(raw.get("reason", ""))[:500],
        context_dependency=ContextDependency(dependency),
        quality_score=max(0.0, min(1.0, score)),
        topic=topic,
        speakers=candidate.speakers,
        complete_thought=bool(raw.get("complete_thought", True)),
        needs_expansion=bool(raw.get("needs_expansion", False)),
    )


def evaluate_candidates(
    candidates: list[Candidate],
    *,
    llm: LLMProvider,
    min_duration: float,
    max_duration: float,
    total_duration: float,
    batch_size: int = 4,
    progress: Optional[Callable[[float, str], None]] = None,
) -> list[LLMEvaluation]:
    """Score candidates in batches, tolerating per-batch failures.

    A failed batch degrades the result (those candidates fall back to their
    heuristic score) but never aborts the run.
    """
    by_id = {c.id: c for c in candidates}
    batches = [
        candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)
    ]
    if not batches:
        return []

    settings = get_settings()
    tokens_each = max(40, settings.llm_eval_tokens_per_candidate)

    def run_batch(batch: list[Candidate]) -> list[LLMEvaluation]:
        payload = [
            {
                "id": c.id,
                "start": c.start,
                "end": c.end,
                "duration": c.duration,
                "text": c.text,
                # Context is trimmed hard: prompt prefill is cheap per token but
                # not free, and this stage dominates total runtime.
                "context_before": c.context_before[-200:],
                "context_after": c.context_after[:200],
                "speakers": c.speakers,
                "audio_events": c.audio_events,
                "opening_text": c.text[:110],
                "conflict_note": c.conflict_note,
                "vision": c.vision.description if c.vision else "",
            }
            for c in batch
        ]

        prompt = P.build_evaluation_prompt(
            payload, min_duration=min_duration, max_duration=max_duration
        )

        try:
            result = llm.complete_json(
                system=P.EVALUATION_SYSTEM,
                prompt=prompt,
                schema=P.EVALUATION_SCHEMA,
                max_tokens=tokens_each * len(batch),
            )
        except ProviderError as exc:
            log.warning("LLM evaluation failed for a batch: %s", exc)
            return []

        items = result.get("evaluations")
        if not isinstance(items, list):
            return []

        out: list[LLMEvaluation] = []
        seen: set[str] = set()
        for raw in items:
            if not isinstance(raw, dict):
                continue
            cid = str(raw.get("candidate_id", "")).strip()
            candidate = by_id.get(cid)
            if candidate is None or cid in seen:
                # Models occasionally invent ids or repeat one; ignore both.
                continue
            seen.add(cid)

            evaluation = _clamp_evaluation(
                raw,
                candidate,
                min_duration=min_duration,
                max_duration=max_duration,
                total_duration=total_duration,
            )
            if evaluation is not None:
                out.append(evaluation)
        return out

    # Batches are independent, so run several concurrently. Ollama serves
    # overlapping requests and the wall-clock win is substantial even though
    # aggregate token throughput is roughly flat.
    evaluations: list[LLMEvaluation] = []
    workers = max(1, min(settings.llm_parallel, len(batches)))
    done = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-eval") as pool:
        futures = [pool.submit(run_batch, batch) for batch in batches]
        for future in as_completed(futures):
            try:
                evaluations.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - one batch must not kill the stage
                log.warning("Evaluation batch raised: %s", exc)
            done += 1
            if progress:
                progress(done / len(batches), f"{len(evaluations)} evaluated")

    return evaluations


def validate_context(
    *,
    start: float,
    end: float,
    transcript: Transcript,
    llm: LLMProvider,
    context_window: float = 30.0,
) -> dict:
    """Ask whether a clip stands alone and whether trimming changed its meaning.

    Returns a dict with keys: verdict, understandable, meaning_preserved,
    missing_context, start_shift, note. On provider failure it returns an
    'accept' verdict so a flaky model cannot silently discard good clips.
    """
    before = transcript.text_between(max(0.0, start - context_window), start)
    after = transcript.text_between(end, end + context_window)
    text = transcript.text_in_window(start, end)

    try:
        raw = llm.complete_json(
            system=P.VALIDATION_SYSTEM,
            prompt=P.build_validation_prompt(
                before=before[-900:],
                text=text,
                after=after[:900],
                start=start,
                end=end,
            ),
            schema=P.VALIDATION_SCHEMA,
            max_tokens=320,
        )
    except ProviderError as exc:
        log.warning("Context validation unavailable (%s); accepting clip", exc)
        return {
            "verdict": "accept",
            "understandable": True,
            "meaning_preserved": True,
            "missing_context": "",
            "start_shift": 0.0,
            "note": "Context validation was skipped.",
            "degraded": True,
        }

    verdict = str(raw.get("verdict", "accept")).lower()
    if verdict not in ("accept", "expand", "reject"):
        verdict = "accept"

    try:
        shift = float(raw.get("suggested_start_shift", 0.0))
    except (TypeError, ValueError):
        shift = 0.0
    # Only ever earlier, and never absurdly so.
    shift = max(-20.0, min(0.0, shift))

    return {
        "verdict": verdict,
        "understandable": bool(raw.get("understandable", True)),
        "meaning_preserved": bool(raw.get("meaning_preserved", True)),
        "missing_context": str(raw.get("missing_context", ""))[:400],
        "start_shift": shift,
        "note": str(raw.get("note", ""))[:300],
        "degraded": False,
    }
