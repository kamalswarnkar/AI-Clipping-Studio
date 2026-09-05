"""Video-level context, and per-clip context drawn from it.

A clip taken out of an hour of footage loses everything that made it legible:
who is speaking, where, and what they are arguing about. This module rebuilds
that in two steps.

First the whole transcript is summarised into a description of the video --
setting, participants, subject, what happens. Long transcripts do not fit in a
local model's context window, so this is map-reduce: each section is summarised
on its own, then the section summaries are synthesised into one description.

Then each clip is described *against* that global description, which is what
lets a clip's context name the bill, the city or the person that the clip
itself only refers to as "it" or "he".

Everything is third person and grounded in the transcript. As everywhere else,
model output is a proposal: it is length-checked and cleaned before use.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from ..models.domain import Diarization, Transcript, VideoContext
from .base import LLMProvider, ProviderError
from .prompts import clip_prompts as P

log = logging.getLogger(__name__)

# Roughly the transcript one summarisation call can take without crowding an
# 8k-token context window. Characters, not tokens, because the transcript is
# plain prose and the ratio is stable enough for a budget.
CHUNK_CHARS = 5000
# Beyond this many sections the reduce step gets its own context problem, so
# the middle is sampled rather than included whole.
MAX_CHUNKS = 14

# Internal diarization labels must never reach a description: the system knows
# there were four voices, not who they were.
_SPEAKER_LABEL = re.compile(r"\bspeakers?\s+[A-Z]\b", re.IGNORECASE)


def _clean(text: str) -> str:
    """Trim model padding and strip labels the viewer should never see."""
    text = _SPEAKER_LABEL.sub("someone", str(text or "")).strip()
    text = re.sub(r"\s+", " ", text)
    # Models like to open with a restatement of the task.
    for opener in (
        "in this clip,", "in this video,", "this clip shows",
        "here is the context:", "context:", "summary:",
    ):
        if text.lower().startswith(opener):
            text = text[len(opener):].strip()
            text = text[:1].upper() + text[1:] if text else text
    return text


def _chunks(transcript: Transcript) -> list[tuple[float, float, str]]:
    """Split the transcript into summarisable sections on segment boundaries."""
    sections: list[tuple[float, float, str]] = []
    buffer: list[str] = []
    start: Optional[float] = None
    end = 0.0
    size = 0

    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue
        if start is None:
            start = segment.start
        buffer.append(text)
        end = segment.end
        size += len(text) + 1
        if size >= CHUNK_CHARS:
            sections.append((start, end, " ".join(buffer)))
            buffer, start, size = [], None, 0

    if buffer and start is not None:
        sections.append((start, end, " ".join(buffer)))

    if len(sections) > MAX_CHUNKS:
        # Keep the opening and the ending -- they carry the framing and the
        # conclusion -- and sample evenly through the middle.
        head, tail = sections[0], sections[-1]
        middle = sections[1:-1]
        step = max(1, len(middle) // (MAX_CHUNKS - 2))
        sections = [head] + middle[::step][: MAX_CHUNKS - 2] + [tail]

    return sections


def summarize_video(
    *,
    transcript: Transcript,
    llm: LLMProvider,
    diarization: Optional[Diarization] = None,
    duration: float = 0.0,
    parallel: int = 3,
    progress: Optional[Callable[[float, str], None]] = None,
) -> VideoContext:
    """Describe the whole video, third person, from its transcript."""
    sections = _chunks(transcript)
    if not sections:
        return VideoContext()

    summaries: list[tuple[float, str]] = []
    workers = max(1, min(parallel, len(sections)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="summarize") as pool:
        futures = {
            pool.submit(
                llm.complete_json,
                system=P.SUMMARY_SYSTEM,
                prompt=P.build_chunk_summary_prompt(text=text, start=start, end=end),
                schema=P.CHUNK_SUMMARY_SCHEMA,
                max_tokens=200,
            ): start
            for start, end, text in sections
        }
        done = 0
        for future in as_completed(futures):
            start = futures[future]
            try:
                raw = future.result()
                summary = _clean(raw.get("summary", ""))
                if summary:
                    summaries.append((start, summary))
            except (ProviderError, Exception) as exc:  # noqa: BLE001
                log.warning("Section summary failed at %.0fs: %s", start, exc)
            done += 1
            if progress:
                progress(0.7 * done / len(sections), f"{done}/{len(sections)} sections")

    if not summaries:
        log.warning("No section summaries produced; video context unavailable")
        return VideoContext()

    summaries.sort(key=lambda item: item[0])
    joined = "\n".join(f"[{t:.0f}s] {text}" for t, text in summaries)
    # The count, not the labels: "Speaker C" is an internal id, not a person.
    voices = len(diarization.speakers) if diarization else 0

    try:
        raw = llm.complete_json(
            system=P.SUMMARY_SYSTEM,
            prompt=P.build_global_context_prompt(
                summaries=joined,
                speakers=f"{voices} distinct voices" if voices else "",
                duration=duration,
            ),
            schema=P.GLOBAL_CONTEXT_SCHEMA,
            max_tokens=600,
        )
    except ProviderError as exc:
        log.warning("Global context unavailable (%s); using section summaries", exc)
        return VideoContext(summary=" ".join(text for _, text in summaries)[:1500])

    if progress:
        progress(1.0, "video context ready")

    terms = [
        _clean(term)
        for term in (raw.get("key_terms") or [])
        if isinstance(term, str) and term.strip()
    ]
    return VideoContext(
        setting=_clean(raw.get("setting", "")),
        participants=_clean(raw.get("participants", "")),
        subject=_clean(raw.get("subject", "")),
        summary=_clean(raw.get("summary", "")),
        key_terms=terms[:12],
    )


def describe_clip(
    *,
    global_context: VideoContext,
    text: str,
    start: float,
    end: float,
    llm: LLMProvider,
) -> tuple[str, bool]:
    """Describe one clip in terms of the video it came from.

    Returns (context, standalone). An empty context means the model could not
    be reached -- callers treat that as "no description", never as a failure.
    """
    if not text.strip():
        return "", False

    try:
        raw = llm.complete_json(
            system=P.SUMMARY_SYSTEM,
            prompt=P.build_clip_context_prompt(
                global_context=global_context.as_prompt_block(),
                text=text,
                start=start,
                end=end,
            ),
            schema=P.CLIP_CONTEXT_SCHEMA,
            max_tokens=400,
        )
    except ProviderError as exc:
        log.warning("Clip context unavailable at %.0fs: %s", start, exc)
        return "", False

    return _clean(raw.get("context", "")), bool(raw.get("standalone", False))
