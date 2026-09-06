"""Hook and caption writing, in the formats `hooks.txt` and `caption.txt` specify.

The model supplies the writing; this module supplies the format and the rules
that can be checked. Model output is a proposal: hooks that are the wrong
length, generic, first-person or duplicated are rejected, the ranking is
repaired if it is not a permutation, the hashtag count is forced to five, and
the caption is trimmed if it runs past the word limit.

Rendering the fixed template here rather than asking for it is deliberate. A 7B
model reproducing emoji headers, separators, thirteen numbered categories and a
ranking block gets it wrong often enough to be unusable, and there is nothing to
gain from letting it try.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ..models.domain import ClipCopy, Hook, VideoContext
from .base import LLMProvider, ProviderError, VisionProvider
from .prompts import copy_prompts as P

log = logging.getLogger(__name__)

# caption.txt asks for 80-150 words; the cap here is the product decision to
# allow a longer caption when the clip earns it.
CAPTION_MAX_WORDS = 190
# The floor asked of the model. caption.txt's own guidance is 80-150; the
# product decision is a fuller caption, so the model is aimed at the top of
# the range and trimmed back if it overshoots.
CAPTION_TARGET_WORDS = 140
CAPTION_MIN_WORDS = 80

HOOK_MIN_WORDS = 4
HOOK_MAX_WORDS = 12  # the spec prefers 4-9; 12 is the hard reject line

# First person outside quotation marks. A Quote-Inspired hook may legitimately
# contain "I" inside quotes, so only unquoted narration is checked.
_FIRST_PERSON = re.compile(
    r"\b(i|i'm|im|me|my|mine|we|we're|our|ours|us)\b", re.IGNORECASE
)
_QUOTED = re.compile(r"[\"“”'‘’][^\"“”]*[\"“”]")
_SPEAKER_LABEL = re.compile(r"\bspeakers?\s+[A-Z]\b", re.IGNORECASE)

# Small models drift out of English mid-sentence, especially near the token
# limit -- a caption question ended "...or was he just trying to draw\u6ce8\u610f\u529b".
# CJK, Hangul, Cyrillic, Arabic, Hebrew, Devanagari.
_NON_LATIN = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
    "\u0400-\u04ff\u0600-\u06ff\u0590-\u05ff\u0900-\u097f]"
)


def has_non_latin(text: str) -> bool:
    """True when the model has drifted out of the Latin alphabet."""
    return bool(_NON_LATIN.search(text or ''))


# ---------------------------------------------------------------------------
# Cleaning and checks
# ---------------------------------------------------------------------------


def clean_hook(text: str) -> str:
    """Strip the decorations models add around a hook."""
    text = _SPEAKER_LABEL.sub("someone", str(text or "")).strip()
    text = re.sub(r"\s+", " ", text)
    # Leading list markers, category labels and wrapping quotes.
    text = re.sub(r"^\s*\d+[.)]\s*", "", text)
    text = re.sub(r"^(hook|best hook)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    for category in P.HOOK_CATEGORIES:
        text = re.sub(rf"^{re.escape(category)}\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    if len(text) > 1 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()
    return text.rstrip(" .")


def is_first_person(text: str) -> bool:
    """True when the narration speaks as a participant rather than about them."""
    return bool(_FIRST_PERSON.search(_QUOTED.sub(" ", text)))


def hook_is_usable(text: str) -> tuple[bool, str]:
    """The checks hooks.txt states as rules, applied in code."""
    if not text:
        return False, "empty"
    words = text.split()
    if len(words) < HOOK_MIN_WORDS:
        return False, f"too short ({len(words)} words)"
    if len(words) > HOOK_MAX_WORDS:
        return False, f"too long ({len(words)} words)"
    lowered = text.lower()
    for phrase in P.BANNED_HOOK_PHRASES:
        if phrase in lowered:
            return False, f"generic phrase {phrase!r}"
    if is_first_person(text):
        return False, "first person"
    if has_non_latin(text):
        return False, "not English"
    return True, ""


def _hashtag(tag: str) -> str:
    tag = re.sub(r"[^0-9A-Za-z]", "", str(tag or ""))
    return f"#{tag}" if tag else ""


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _trim_to_words(paragraphs: list[str], question: str, limit: int) -> list[str]:
    """Drop whole sentences from the end until the caption fits.

    Sentences, not words: cutting a caption mid-sentence to hit a count is
    worse than a slightly shorter caption.
    """
    budget = limit - _word_count(question)
    while _word_count(" ".join(paragraphs)) > budget:
        longest = max(range(len(paragraphs)), key=lambda i: _word_count(paragraphs[i]))
        sentences = re.split(r"(?<=[.!?])\s+", paragraphs[longest].strip())
        if len(sentences) <= 1:
            break
        paragraphs[longest] = " ".join(sentences[:-1]).strip()
    return paragraphs


# ---------------------------------------------------------------------------
# Rendering the fixed layouts
# ---------------------------------------------------------------------------


def render_hooks_txt(copy: ClipCopy) -> str:
    """The exact layout hooks.txt specifies."""
    lines = ["🏆 BEST HOOK", "", copy.best_hook, "", "---", ""]
    for index, hook in enumerate(copy.hooks, start=1):
        lines.append(f"{index}. {hook.category}: {hook.text}")
    lines += ["", "---", "", "🏆 FINAL RANKING", ""]
    for position, hook in enumerate(copy.ranked_hooks, start=1):
        lines.append(f"{position}. {hook.text}")
    return "\n".join(lines).rstrip() + "\n"


def render_caption_txt(copy: ClipCopy) -> str:
    """The exact layout caption.txt specifies."""
    return copy.caption.rstrip() + "\n"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def watch_clip(
    *,
    video_path: "Path",
    duration: float,
    frames: int,
    vision: "VisionProvider",
    workdir: "Path",
) -> str:
    """Describe what is visible in the clip, from the rendered file itself.

    hooks.txt and caption.txt both name the video as the primary source and the
    transcript as secondary. A text model cannot watch anything, so the clip is
    watched here and the description is what the writing prompts receive.

    Returns "" on any failure: copy written from the transcript and context
    alone is worse, but it is not broken.
    """
    from ..video import ffmpeg

    if not video_path.exists() or duration <= 0:
        return ""

    # Spread the samples across the clip rather than bunching at the start:
    # what makes these clips work usually happens in the middle.
    fractions = [(i + 0.5) / frames for i in range(max(1, frames))]
    images: list[Path] = []
    workdir.mkdir(parents=True, exist_ok=True)
    for index, fraction in enumerate(fractions):
        target = workdir / f"{video_path.stem}_f{index}.jpg"
        try:
            ffmpeg.extract_frame(video_path, duration * fraction, target, width=640)
            images.append(target)
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not sample %s at %.1fs: %s", video_path.name, fraction, exc)

    if not images:
        return ""

    try:
        raw = vision.describe_frames(
            images,
            system=P.CLIP_VISION_SYSTEM,
            prompt=P.CLIP_VISION_PROMPT,
            schema=P.CLIP_VISION_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001 - vision is an enhancement
        log.warning("Could not watch %s: %s", video_path.name, exc)
        return ""
    finally:
        for image in images:
            image.unlink(missing_ok=True)

    description = re.sub(r"\s+", " ", str(raw.get("description", "") or "")).strip()
    if has_non_latin(description):
        log.warning("Vision description drifted out of English; ignoring")
        return ""

    extras = []
    people = raw.get("people_visible")
    if isinstance(people, int) and people > 0:
        extras.append(f"People visible: {people}.")
    on_screen = str(raw.get("on_screen_text", "") or "").strip()
    if on_screen:
        extras.append(f"On-screen text: {on_screen}")
    return " ".join([description, *extras]).strip()


def generate_hooks(
    *,
    context: str,
    transcript: str,
    video_context: VideoContext,
    llm: LLMProvider,
    seen: str = "",
) -> tuple[str, list[Hook]]:
    """Return (best_hook, hooks). Empty on failure -- never raises."""
    if not transcript.strip():
        return "", []

    try:
        raw = llm.complete_json(
            system=P.HOOKS_SYSTEM,
            prompt=P.build_hooks_prompt(
                context=context,
                transcript=transcript,
                video_context=video_context.as_prompt_block(),
                seen=seen,
            ),
            schema=P.HOOKS_SCHEMA,
            max_tokens=1500,
        )
    except ProviderError as exc:
        log.warning("Hook generation unavailable: %s", exc)
        return "", []

    proposed = raw.get("hooks") or []
    by_category: dict[str, str] = {}
    # Duplicates are rejected here rather than when the list is assembled. Drop
    # one later and its category is already marked as filled, so the follow-up
    # call never asks for it and the set comes out short.
    taken: set[str] = set()
    for item in proposed:
        if not isinstance(item, dict):
            continue
        text = clean_hook(item.get("text", ""))
        usable, reason = hook_is_usable(text)
        if not usable:
            log.debug("Rejected hook %r: %s", text, reason)
            continue
        if text.lower() in taken:
            log.debug("Rejected hook %r: duplicate", text)
            continue
        category = str(item.get("category", "")).strip()
        # Match the category back to the canonical list; models paraphrase.
        match = next(
            (c for c in P.HOOK_CATEGORIES if c.lower() == category.lower()), None
        )
        if match is None:
            match = next(
                (c for c in P.HOOK_CATEGORIES if c not in by_category), None
            )
        if match and match not in by_category:
            by_category[match] = text
            taken.add(text.lower())

    # A 7B model rarely delivers all thirteen in one answer. Three follow-up
    # rounds gets there most of the time; past that the returns stop justifying
    # the calls, and a set of twelve is still usable.
    for _ in range(3):
        missing = [c for c in P.HOOK_CATEGORIES if c not in by_category]
        if not missing:
            break
        filled = _fill_missing_hooks(
            missing=missing,
            existing=list(by_category.values()),
            context=context,
            transcript=transcript,
            video_context=video_context,
            llm=llm,
        )
        if not filled:
            break
        by_category.update(filled)

    seen: set[str] = set()
    hooks: list[Hook] = []
    for category in P.HOOK_CATEGORIES:
        text = by_category.get(category, "")
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        hooks.append(Hook(category=category, text=text))

    best = clean_hook(raw.get("best_hook", ""))
    usable, reason = hook_is_usable(best)
    if not usable:
        log.debug("Rejected best hook %r: %s", best, reason)
        best = hooks[0].text if hooks else ""

    # The ranking must be a permutation of the hooks that survived.
    ranking = [
        int(n) - 1
        for n in (raw.get("ranking") or [])
        if isinstance(n, int) and 1 <= n <= len(hooks)
    ]
    order: list[int] = []
    for index in ranking:
        if index not in order:
            order.append(index)
    order += [i for i in range(len(hooks)) if i not in order]
    for position, index in enumerate(order):
        hooks[index].rank = position + 1

    return best, hooks


def _fill_missing_hooks(
    *,
    missing: list[str],
    existing: list[str],
    context: str,
    transcript: str,
    video_context: VideoContext,
    llm: LLMProvider,
) -> dict[str, str]:
    """Second, smaller call for the categories the first answer skipped."""
    log.info("Asking again for %d hook categor(ies): %s", len(missing), ", ".join(missing))
    try:
        raw = llm.complete_json(
            system=P.HOOKS_SYSTEM,
            prompt=P.build_missing_hooks_prompt(
                context=context,
                transcript=transcript,
                video_context=video_context.as_prompt_block(),
                categories=missing,
                existing=existing,
            ),
            schema=P.MISSING_HOOKS_SCHEMA,
            max_tokens=700,
        )
    except ProviderError as exc:
        log.warning("Could not fill missing hooks: %s", exc)
        return {}

    filled: dict[str, str] = {}
    taken = {e.lower() for e in existing}
    for item in raw.get("hooks") or []:
        if not isinstance(item, dict):
            continue
        text = clean_hook(item.get("text", ""))
        usable, reason = hook_is_usable(text)
        if not usable or text.lower() in taken:
            log.debug("Rejected fill-in hook %r: %s", text, reason or "duplicate")
            continue
        category = str(item.get("category", "")).strip()
        match = next((c for c in missing if c.lower() == category.lower()), None)
        if match is None:
            match = next((c for c in missing if c not in filled), None)
        if match and match not in filled:
            filled[match] = text
            taken.add(text.lower())
    return filled


def generate_caption(
    *,
    context: str,
    transcript: str,
    video_context: VideoContext,
    llm: LLMProvider,
    seen: str = "",
) -> str:
    """Return the finished caption text, or "" on failure.

    Tried twice: the failure this guards against is the model drifting out of
    English part-way through, which produces a caption that is unusable but
    otherwise well-formed, so nothing else would catch it.
    """
    for attempt in range(2):
        caption = _caption_once(
            context=context,
            transcript=transcript,
            video_context=video_context,
            llm=llm,
            seen=seen,
        )
        if caption:
            return caption
        log.info("Caption attempt %d produced nothing usable", attempt + 1)
    return ""


def _caption_once(
    *,
    context: str,
    transcript: str,
    video_context: VideoContext,
    llm: LLMProvider,
    seen: str = "",
) -> str:
    if not transcript.strip():
        return ""

    try:
        raw = llm.complete_json(
            system=P.CAPTION_SYSTEM,
            prompt=P.build_caption_prompt(
                context=context,
                transcript=transcript,
                video_context=video_context.as_prompt_block(),
                min_words=CAPTION_TARGET_WORDS,
                max_words=CAPTION_MAX_WORDS,
                seen=seen,
            ),
            schema=P.CAPTION_SCHEMA,
            max_tokens=1600,
        )
    except ProviderError as exc:
        log.warning("Caption generation unavailable: %s", exc)
        return ""

    def field(name: str) -> str:
        value = _SPEAKER_LABEL.sub("someone", str(raw.get(name, "") or "")).strip()
        return re.sub(r"\s+", " ", value)

    headline = field("headline").strip("🚨 ").strip()
    paragraphs = [field("trigger"), field("escalation"), field("division")]
    paragraphs = [p for p in paragraphs if p]
    question = field("question").lstrip("👇 ").strip()

    if not headline or not paragraphs or not question:
        log.warning("Caption missing required parts; discarding")
        return ""

    drifted = [
        part
        for part in [headline, question, *paragraphs]
        if has_non_latin(part)
    ]
    if drifted:
        log.warning("Caption drifted out of English; discarding")
        return ""

    # A question with no question mark means the model ran out of room
    # mid-sentence -- the last word is usually cut in half ("drug traffc").
    # Appending a "?" hides that, so treat it as the failed attempt it is and
    # let the caller try again.
    if not question.rstrip().endswith("?"):
        log.info("Caption question was cut off; discarding this attempt")
        return ""

    paragraphs = _trim_to_words(paragraphs, question, CAPTION_MAX_WORDS)

    tags = [_hashtag(t) for t in (raw.get("hashtags") or [])]
    tags = [t for t in dict.fromkeys(tags) if t][:5]

    lines = [f"🚨 {headline} 🚨", ""]
    for paragraph in paragraphs:
        lines += [paragraph, ""]
    lines.append(f"👇 {question}")

    location = field("location")
    # Only a "City, State" shape counts; anything else is a guess.
    if location and re.fullmatch(r"[A-Za-z .'-]+,\s*[A-Za-z .]+", location):
        lines += ["", f"📍 {location}"]

    if tags:
        lines += ["", " ".join(tags)]

    caption = "\n".join(lines).strip()
    words = _word_count(" ".join(paragraphs)) + _word_count(question)
    if words < CAPTION_MIN_WORDS:
        log.info("Caption is %d words, below the %d target", words, CAPTION_MIN_WORDS)
    return caption


def write_copy(
    *,
    context: str,
    transcript: str,
    video_context: VideoContext,
    llm: LLMProvider,
    seen: str = "",
) -> Optional[ClipCopy]:
    """Hooks and caption for one clip. None when nothing usable was produced."""
    best, hooks = generate_hooks(
        context=context,
        transcript=transcript,
        video_context=video_context,
        llm=llm,
        seen=seen,
    )
    caption = generate_caption(
        context=context,
        transcript=transcript,
        video_context=video_context,
        llm=llm,
        seen=seen,
    )
    if not hooks and not caption:
        return None
    return ClipCopy(best_hook=best, hooks=hooks, caption=caption)
