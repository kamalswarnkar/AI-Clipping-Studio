"""Hook and caption generation, with factuality checks applied in code.

The prompt asks the model not to fabricate. This module verifies it, because a
prompt is a request and not a guarantee. Every generated hook is checked for:

* numbers and percentages that were never said
* quoted phrases that do not appear in the speech
* banned generic phrases that would fit any video

Hooks and the caption are produced in a SINGLE call: they share all their
context, and two calls per clip doubled prompt prefill and request overhead
for no quality benefit. Generation tokens dominate this stage's runtime.
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

# "Speaker A" and friends are internal diarization tags, not people. They must
# never surface in copy, so they are stripped from output as well as withheld
# from the prompt.
# The trailing boundary is essential: without it "Speaker Demands" matches
# "Speaker D" and the substitution swallows the D, yielding "speakeremands".
_SPEAKER_LABEL = re.compile(r"\bspeakers?\s+[A-Z]\b", re.I)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)

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

DEFAULT_EMOJIS = ("🚨", "👀", "😳", "😶")


def _normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _transcript_numbers(transcript: str) -> set[str]:
    """Every number in the transcript, digits and spelled-out forms alike."""
    found = set(_NUMBER.findall(transcript))
    lowered = transcript.lower()
    for word, digits in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            found.add(digits)
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


def check_hook_quality(text: str) -> tuple[bool, str]:
    """Reject hooks that are generic, banned, or the wrong shape.

    The brief is explicit that a hook which could fit hundreds of videos is a
    failure, so that is enforced here rather than hoped for.
    """
    stripped = _EMOJI.sub("", text).strip()
    if not stripped:
        return False, "empty"

    lowered = stripped.lower()
    for phrase in P.BANNED_HOOK_PHRASES:
        if phrase in lowered:
            return False, f"uses banned phrase '{phrase}'"

    words = len(stripped.split())
    if words > 14:
        return False, f"{words} words, too long to read at a glance"

    return True, ""


def strip_speaker_labels(text: str) -> str:
    """Remove internal diarization tags from generated copy.

    Diarization knows there were three distinct voices, not who they were.
    "Speaker C says..." is meaningless to a viewer, so any label that survives
    the prompt is rewritten into role-neutral phrasing here.
    """
    if not text or not _SPEAKER_LABEL.search(text):
        return text

    text = _SPEAKER_LABEL.sub("one speaker", text)
    # "one speaker and one speaker" reads worse than the thing it replaced.
    text = re.sub(
        r"one speaker\s+and\s+one speaker", "two people", text, flags=re.I
    )
    # Tidy the articles the substitution leaves behind.
    text = re.sub(r"(?:a|an|the)\s+one speaker", "one speaker", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:1].upper() + text[1:] if text else text


def _clean_paragraph(text: str) -> str:
    """Strip padding the model adds to satisfy a minimum length.

    A length floor makes a small model repeat hashtags and emoji inside body
    paragraphs to reach the count. Hashtags belong only in the final line, and
    the reference captions use emoji solely in the headline and the question.
    """
    text = strip_speaker_labels(text.strip())
    text = re.sub(r"\s*#\w+", "", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _location_is_supported(location: str, transcript: str) -> bool:
    """Keep a geotag only when the place is actually named in the clip.

    The model will happily supply a plausible city. Treating that as verified
    would put an invented location on a published post.
    """
    if not location.strip():
        return False
    lowered = transcript.lower()
    parts = [p.strip() for p in re.split(r"[,/]", location) if p.strip()]
    if not parts:
        return False
    # Require the most specific component -- the city. Accepting any part lets
    # "Sacramento, California" through on the strength of "California" alone,
    # publishing a city the clip never mentions.
    city = parts[0]
    return len(city) > 2 and city.lower() in lowered


def _clean_hook(text: str) -> str:
    """Tidy the artifacts a small model leaves on hook text.

    Typical output includes stray leading punctuation ("/Area Tensions Erupt"),
    trailing hashtags that belong in the caption, and wrapping quotes.
    """
    text = text.strip().strip('"').strip()
    # Hashtags belong in the caption, not in a hook.
    text = re.sub(r"\s*#\w+", "", text)
    # Leading punctuation left over from a truncated or malformed generation.
    text = re.sub(r"^[\s\-–—:;,./\\|*>]+", "", text)
    # Collapse whitespace introduced by the removals.
    return re.sub(r"\s{2,}", " ", text).strip()


def _ensure_emoji(text: str) -> str:
    """Guarantee the hook carries an emoji, as the brief requires."""
    if _EMOJI.search(text):
        return text
    return f"🚨 {text.strip()}"


def _fallback_hooks(plan: ClipPlan, existing: list[Hook]) -> list[Hook]:
    """Fill missing categories with transcript-derived lines.

    Uses actual sentences from the clip, so a fallback hook is dull but never
    false -- the correct trade when the model has failed.
    """
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", plan.transcript)
        if 4 <= len(s.split()) <= 14
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
            sentence = (plan.topic or plan.transcript[:60]).strip()
        text = sentence.rstrip(".").strip()
        if len(text) > 90:
            text = text[:87].rsplit(" ", 1)[0] + "..."
        filled.append(Hook(category=category, text=_ensure_emoji(text)))

    return filled


def _parse_hooks(raw: dict, plan: ClipPlan) -> tuple[list[Hook], str]:
    """Turn a raw response into ranked hooks. Returns (hooks, best hook text)."""
    accepted: list[Hook] = []
    seen: set[str] = set()

    for item in raw.get("hooks", []) or []:
        if not isinstance(item, dict):
            continue
        text = _clean_hook(strip_speaker_labels(str(item.get("text", ""))))
        category = str(item.get("category", "")).strip()
        category = next(
            (c for c in P.HOOK_CATEGORIES if c.lower() == category.lower()), ""
        )
        if not text or not category:
            continue

        key = _normalise(text)
        if key in seen:
            continue

        ok, why = check_factuality(text, plan.transcript)
        if not ok:
            log.info("Rejected hook for %s (%s): %s", plan.name, why, text[:60])
            continue

        ok, why = check_hook_quality(text)
        if not ok:
            log.info("Rejected hook for %s (%s): %s", plan.name, why, text[:60])
            continue

        seen.add(key)
        accepted.append(Hook(category=category, text=_ensure_emoji(text)))

    by_category: dict[str, Hook] = {}
    for hook in accepted:
        by_category.setdefault(hook.category, hook)

    hooks = _fallback_hooks(plan, list(by_category.values()))
    hooks.sort(key=lambda h: P.HOOK_CATEGORIES.index(h.category))

    # --- ranking ------------------------------------------------------------
    ordered: list[Hook] = []
    ranking = raw.get("ranking")
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
            if check_factuality(candidate.text, plan.transcript)[0]:
                best = candidate
    except (TypeError, ValueError):
        pass

    for hook in hooks:
        hook.is_best = hook is best

    return hooks, (best.text if best else "")


def _format_caption(raw: dict, plan: ClipPlan) -> str:
    """Assemble the caption in the required layout.

    The structure is built in code rather than asked for as free text, so the
    headline banner, the side-taking question, the location rule and the exact
    hashtag count are guaranteed instead of hoped for.
    """
    headline = strip_speaker_labels(
        str(raw.get("headline", "")).strip().strip('"').rstrip("!.")
    )
    # The model sometimes answers with a bare hashtag ("#FRAUDCONTROVERSY").
    # Hashtags belong in the final line only, so unpack it into words instead.
    headline = re.sub(
        r"#(\w+)", lambda m: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m.group(1)), headline
    ).strip()
    if not headline:
        headline = (plan.topic or "CONFRONTATION CAUGHT ON CAMERA").upper()
    headline = _EMOJI.sub("", headline).strip().upper()

    parts: list[str] = [f"🚨 {headline} 🚨", ""]

    for key in ("trigger", "escalation", "reaction", "debate"):
        paragraph = _clean_paragraph(str(raw.get(key, "")))
        if paragraph:
            parts.extend([paragraph, ""])

    question = str(raw.get("question", "")).strip()
    if question:
        question = question.lstrip("👇").strip()
        question = re.sub(r"\s*#\w+", "", question).strip()
        # The model often ends on an emoji. Check for the question mark against
        # the text with any trailing emoji removed, or we produce "...🤔?".
        without_emoji = _EMOJI.sub("", question).strip()
        if without_emoji and not without_emoji.endswith("?"):
            question = f"{without_emoji}?"
        else:
            question = without_emoji or question
        parts.extend([f"👇 {question}", ""])

    # Only include a location the model was given evidence for.
    location = str(raw.get("location", "")).strip()
    if (
        location
        and location.lower() not in ("unknown", "n/a", "none", "null")
        and _location_is_supported(location, plan.transcript)
    ):
        parts.extend([f"📍 Geotag / Location: {location}", ""])

    tags = raw.get("hashtags") or []
    cleaned: list[str] = []
    for tag in tags:
        tag = str(tag).strip().replace(" ", "")
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        if tag.lower() not in {t.lower() for t in cleaned}:
            cleaned.append(tag)
    while len(cleaned) < 5:
        for filler in ("#StreetInterview", "#PublicConfrontation", "#ViralVideo",
                       "#Debate", "#CaughtOnCamera"):
            if filler.lower() not in {t.lower() for t in cleaned}:
                cleaned.append(filler)
                break
    parts.append(" ".join(cleaned[:5]))

    return "\n".join(parts).strip()


def _build_caption(raw: dict, plan: ClipPlan) -> str:
    caption = _format_caption(raw, plan)

    ok, why = check_factuality(caption, plan.transcript)
    if not ok:
        log.info("Caption for %s failed the factuality check (%s)", plan.name, why)
        # Rebuild without the offending free text, keeping the structure.
        safe = {
            "headline": (plan.topic or "confrontation caught on camera").upper(),
            "trigger": " ".join(
                re.split(r"(?<=[.!?])\s+", plan.transcript.strip())[:2]
            )[:400],
            "escalation": "",
            "reaction": "",
            "debate": "",
            "question": "Who crossed the line",
            "location": "",
            "hashtags": raw.get("hashtags") or [],
        }
        caption = _format_caption(safe, plan)

    return caption


def generate_copy(
    plan: ClipPlan,
    *,
    llm: LLMProvider,
    vision: Optional[VisionObservation] = None,
    conflict_note: str = "",
    opening_line: str = "",
) -> ClipCopy:
    """Generate 13 ranked hooks and one formatted caption for a clip."""
    visual_note = vision.description[:200] if vision and vision.description else ""
    opening = opening_line or " ".join(plan.transcript.split()[:14])

    # Withhold the internal speaker tags: given "Speaker C" the model will
    # faithfully write "Speaker C says...". It only needs to know how many
    # distinct voices there are.
    voices = len([s for s in plan.speakers if s])
    speaker_hint = (
        [f"{voices} different voices"] if voices > 1 else ["one speaker"]
    )

    # One call for both. Hooks and the caption share every piece of context, so
    # splitting them doubled prompt prefill and request overhead for no benefit.
    raw: dict = {}
    try:
        raw = llm.complete_json(
            system=P.COPY_SYSTEM,
            prompt=P.build_copy_prompt(
                transcript=plan.transcript,
                opening_line=opening,
                topic=plan.topic,
                speakers=speaker_hint,
                duration=plan.duration,
                conflict_note=conflict_note,
                visual_note=visual_note,
            ),
            schema=P.COPY_SCHEMA,
            temperature=0.75,
            max_tokens=1700,
        )
    except ProviderError as exc:
        log.warning("Copy generation failed for %s: %s", plan.name, exc)

    hooks, best = _parse_hooks(raw, plan)
    caption = _build_caption(raw, plan)

    return ClipCopy(
        hooks=hooks,
        best_hook=best,
        caption=caption,
        generated_by=getattr(llm, "model", getattr(llm, "name", "unknown")),
    )
