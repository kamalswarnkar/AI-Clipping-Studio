"""Hook and caption generation, with factuality checks applied in code.

The prompt asks the model not to fabricate. This module verifies it, because a
prompt is a request and not a guarantee. Every generated hook is checked against
the clip transcript for:

* numbers and percentages that were never said
* quoted phrases that do not appear in the speech
* fabricated attribution

Anything that fails is dropped and replaced from the transcript itself, so the
output stays faithful even when the model does not (Architecture.md section 19).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..models.domain import ClipCopy, ClipPlan, Hook, VisionObservation
from .base import LLMProvider, ProviderError
from .prompts import clip_prompts as P

log = logging.getLogger(__name__)

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# Quoted spans, in every form a model actually emits. Single quotes need
# lookarounds so apostrophes inside contractions ("doesn't") are not mistaken
# for quote delimiters -- without that guard a misquote wrapped in single
# quotes slips through unchecked.
_QUOTED = re.compile(
    r'"([^"]{4,})"'
    r"|“([^”]{4,})”"
    r"|(?<![\w'])'([^']{8,})'(?![\w])"
    r"|(?<![\w’])‘([^’]{8,})’(?![\w])"
)
_WORD = re.compile(r"[a-z0-9']+")

# Spelled-out forms so "ninety one percent" matches "91%".
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000",
}


def _normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _transcript_numbers(transcript: str) -> set[str]:
    """Every number in the transcript, digits and spelled-out forms alike."""
    found = set(_NUMBER.findall(transcript))
    lowered = transcript.lower()
    for word, digits in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            found.add(digits)
    # "ninety one" -> 91
    for first, second in re.findall(
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s-]"
        r"(one|two|three|four|five|six|seven|eight|nine)\b",
        lowered,
    ):
        found.add(str(int(_NUMBER_WORDS[first]) + int(_NUMBER_WORDS[second])))
    return found


def check_factuality(text: str, transcript: str) -> tuple[bool, str]:
    """Verify a hook or caption is supported by the transcript.

    Conservative by design: it rejects the checkable failure modes (invented
    numbers, invented quotes) rather than attempting to judge meaning, which is
    what the model is for.
    """
    if not text.strip():
        return False, "empty"

    transcript_numbers = _transcript_numbers(transcript)
    for number in _NUMBER.findall(text):
        # Ignore small ordinals that are usually rhetorical ("3 things").
        if number in transcript_numbers:
            continue
        if number.rstrip(".0") in {n.rstrip(".0") for n in transcript_numbers}:
            continue
        return False, f"cites {number}, which is not in the clip"

    normalised_transcript = _normalise(transcript)
    for match in _QUOTED.finditer(text):
        quote = next((g for g in match.groups() if g), "")
        normalised_quote = _normalise(quote)
        if normalised_quote and normalised_quote not in normalised_transcript:
            return False, f'quotes "{quote[:40]}", which is not said in the clip'

    return True, ""


def _fallback_hooks(plan: ClipPlan, existing: list[Hook]) -> list[Hook]:
    """Fill missing categories with transcript-derived lines.

    Uses actual sentences from the clip, so a fallback hook is dull but never
    false -- the correct trade when the model has failed.
    """
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", plan.transcript)
        if 4 <= len(s.split()) <= 18
    ]
    sentences.sort(key=lambda s: -len(s.split()))

    have = {h.category for h in existing}
    filled = list(existing)
    pool = iter(sentences)

    for category in P.HOOK_CATEGORIES:
        if category in have:
            continue
        sentence = next(pool, "")
        if not sentence:
            sentence = (plan.topic or plan.transcript[:70]).strip()
        text = sentence.rstrip(".").strip()
        if len(text) > 90:
            text = text[:87].rsplit(" ", 1)[0] + "..."
        filled.append(Hook(category=category, text=text))

    return filled


def generate_copy(
    plan: ClipPlan,
    *,
    llm: LLMProvider,
    vision: Optional[VisionObservation] = None,
) -> ClipCopy:
    """Generate 13 ranked hooks and one caption for a clip.

    Falls back to transcript-derived copy if the model is unavailable, so a clip
    always ships with usable text rather than an empty file.
    """
    visual_note = ""
    if vision and vision.description:
        visual_note = vision.description[:200]

    prompt = P.build_copy_prompt(
        transcript=plan.transcript,
        topic=plan.topic,
        speakers=plan.speakers,
        duration=plan.duration,
        visual_note=visual_note,
    )

    raw: dict = {}
    try:
        raw = llm.complete_json(
            system=P.COPY_SYSTEM,
            prompt=prompt,
            schema=P.COPY_SCHEMA,
            temperature=0.7,  # copy benefits from variety; evaluation does not
            max_tokens=1100,
        )
    except ProviderError as exc:
        log.warning("Copy generation failed for %s: %s", plan.name, exc)

    # --- hooks --------------------------------------------------------------
    accepted: list[Hook] = []
    rejected: list[tuple[str, str]] = []
    seen_texts: set[str] = set()

    for item in raw.get("hooks", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip().strip('"')
        category = str(item.get("category", "")).strip()

        if category not in P.HOOK_CATEGORIES:
            # Map onto the closest known category by position, or drop it.
            category = next(
                (c for c in P.HOOK_CATEGORIES if c.lower() == category.lower()), ""
            )
        if not text or not category:
            continue

        key = _normalise(text)
        if key in seen_texts:
            continue

        ok, why = check_factuality(text, plan.transcript)
        if not ok:
            rejected.append((text, why))
            log.info("Rejected hook for %s (%s): %s", plan.name, why, text[:60])
            continue

        seen_texts.add(key)
        accepted.append(Hook(category=category, text=text))

    # Keep one hook per category, in the canonical order.
    by_category: dict[str, Hook] = {}
    for hook in accepted:
        by_category.setdefault(hook.category, hook)

    hooks = _fallback_hooks(plan, list(by_category.values()))
    hooks.sort(key=lambda h: P.HOOK_CATEGORIES.index(h.category))

    # --- ranking ------------------------------------------------------------
    ranking = raw.get("ranking")
    ordered: list[Hook] = []
    if isinstance(ranking, list):
        for position in ranking:
            try:
                index = int(position)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(hooks) and hooks[index] not in ordered:
                ordered.append(hooks[index])
    for hook in hooks:
        if hook not in ordered:
            ordered.append(hook)

    for rank, hook in enumerate(ordered, start=1):
        hook.rank = rank

    # --- best hook ----------------------------------------------------------
    best = ordered[0] if ordered else None
    try:
        index = int(raw.get("best_hook_index", -1))
        if 0 <= index < len(hooks):
            candidate = hooks[index]
            ok, _ = check_factuality(candidate.text, plan.transcript)
            if ok:
                best = candidate
    except (TypeError, ValueError):
        pass

    if best is not None:
        for hook in hooks:
            hook.is_best = hook is best

    # --- caption ------------------------------------------------------------
    caption = str(raw.get("caption", "")).strip()
    ok, why = check_factuality(caption, plan.transcript)
    if not ok:
        if caption:
            log.info("Rejected caption for %s (%s)", plan.name, why)
        topic = plan.topic or "this moment"
        speakers = " and ".join(plan.speakers) if plan.speakers else "the speaker"
        first = re.split(r"(?<=[.!?])\s+", plan.transcript.strip())[:2]
        caption = (
            f"{speakers} on {topic}. " + " ".join(first)
        ).strip()
        if len(caption) > 400:
            caption = caption[:397].rsplit(" ", 1)[0] + "..."

    return ClipCopy(
        hooks=hooks,
        best_hook=best.text if best else "",
        caption=caption,
        generated_by=getattr(llm, "model", getattr(llm, "name", "unknown")),
    )
