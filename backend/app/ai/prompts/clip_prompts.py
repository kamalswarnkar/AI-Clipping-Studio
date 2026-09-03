"""Prompts and JSON schemas for the reasoning stages.

Two rules shape everything here:

* The model reasons over *evidence we extracted*; it never sees or touches the
  video, and it never emits a command. It returns judgements as JSON.
* Factuality constraints are stated as hard rules, and are re-checked in code
  afterwards. A prompt is a request, not a guarantee.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Shared safety language (Architecture.md section 19)
# ---------------------------------------------------------------------------

FACTUALITY_RULES = """
ACCURACY RULES (these override every other instruction):
- Use only what is present in the supplied transcript and observations.
- Never invent quotes, statistics, names, places, events, motives or outcomes.
- Never state or imply something happens in the clip that the transcript does
  not support.
- Do not drop a qualifier if removing it changes the meaning of a claim.
- If something is uncertain, describe it cautiously rather than asserting it.
- For political or contested content: describe faithfully, never distort,
  never fabricate accusations, and never write persuasion aimed at a
  demographic or voter group.
""".strip()


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

EVALUATION_SYSTEM = f"""
You are a senior short-form video editor. You review candidate moments cut from a
long interview or podcast and decide which would work as standalone short videos.

You judge ONLY from the supplied transcript and observations. You never see the
video itself. You return JSON and nothing else.

A strong candidate is:
- self-contained: understandable without having watched the rest of the source
- a complete thought: it does not start or stop mid-idea
- structured: setup then payoff, question then answer, or claim then reasoning
- specific: it says something concrete rather than gesturing at a topic

Reject candidates that are:
- logistics, scheduling, small talk, introductions, sponsor reads or sign-offs
- dependent on context that is not inside the clip
- a fragment of a larger point, with the actual payoff outside the window
- interesting-sounding but substantively empty

Do not recommend a candidate merely because it contains emotive or dramatic
words. Substance decides.

{FACTUALITY_RULES}
""".strip()


EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "recommended": {"type": "boolean"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "reason": {"type": "string"},
                    "context_dependency": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "quality_score": {"type": "number"},
                    "topic": {"type": "string"},
                    "complete_thought": {"type": "boolean"},
                    "needs_expansion": {"type": "boolean"},
                },
                "required": [
                    "candidate_id",
                    "recommended",
                    "start",
                    "end",
                    "reason",
                    "context_dependency",
                    "quality_score",
                    "topic",
                    "complete_thought",
                    "needs_expansion",
                ],
            },
        }
    },
    "required": ["evaluations"],
}


def build_evaluation_prompt(
    candidates: list[dict[str, Any]],
    *,
    min_duration: float,
    max_duration: float,
) -> str:
    """Render the evidence bundle for a batch of candidates."""
    blocks: list[str] = []
    for c in candidates:
        lines = [
            f"### {c['id']}",
            f"Window: {c['start']:.1f}s to {c['end']:.1f}s ({c['duration']:.1f}s)",
            f"Speakers in window: {', '.join(c['speakers']) or 'unknown'}",
        ]
        if c.get("audio_events"):
            lines.append(f"Audio events: {', '.join(c['audio_events'])}")
        if c.get("scene_count"):
            lines.append(f"Camera/scene changes: {c['scene_count']}")
        if c.get("vision"):
            lines.append(f"Visual observation: {c['vision']}")
        if c.get("context_before"):
            lines.append(f"\n[Context immediately BEFORE the window]\n{c['context_before']}")
        lines.append(f"\n[TRANSCRIPT OF THE CANDIDATE]\n{c['text']}")
        if c.get("context_after"):
            lines.append(f"\n[Context immediately AFTER the window]\n{c['context_after']}")
        blocks.append("\n".join(lines))

    ids = ", ".join(c["id"] for c in candidates)

    return f"""
Evaluate each candidate below as a potential standalone short-form clip.

{chr(10).join(blocks)}

---
For EVERY candidate ({ids}) return one object with:

- candidate_id: the exact id above
- recommended: true only if this would genuinely work as a standalone short
- start / end: absolute seconds in the source video. Keep the supplied window
  unless it is wrong. You may adjust each edge by up to 15 seconds to capture a
  complete thought. Duration must stay between {min_duration:.0f} and
  {max_duration:.0f} seconds. Use the context sections to justify any change.
- reason: one sentence on why it works or fails
- context_dependency: low if it stands alone, high if it needs the wider video
- quality_score: 0.0 to 1.0
- topic: a short factual noun phrase describing what is discussed
- complete_thought: does the window contain a whole idea
- needs_expansion: true if it needs more surrounding context to make sense

Return JSON only.
""".strip()


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

VISION_SYSTEM = """
You describe frames sampled from one moment of a video, for an editor deciding
whether to cut it as a short-form clip.

Report only what is actually visible. Do not guess at identities, locations,
affiliations or intentions. Do not infer what is being said. If something is
unclear, say so rather than inventing a detail.

Return JSON only.
""".strip()


VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "people_visible": {"type": "integer"},
        "on_screen_text": {"type": "string"},
        "notable_events": {"type": "array", "items": {"type": "string"}},
        "visual_interest": {"type": "number"},
        "is_talking_head": {"type": "boolean"},
    },
    "required": [
        "description",
        "people_visible",
        "on_screen_text",
        "notable_events",
        "visual_interest",
        "is_talking_head",
    ],
}

VISION_PROMPT = """
These frames are sampled in order from one continuous moment of a video.

Describe:
- description: one or two sentences on what is visibly happening
- people_visible: how many people are clearly visible (0 if none)
- on_screen_text: any readable text, captions or graphics; empty string if none
- notable_events: visible actions such as gestures, pointing, reactions,
  laughter, someone standing up, a graphic appearing. Empty list if nothing
  notable.
- visual_interest: 0.0 to 1.0, how visually engaging this is for a short video
- is_talking_head: true if it is mostly people talking to camera or each other

Return JSON only.
""".strip()


# ---------------------------------------------------------------------------
# Copy generation (hooks + caption)
# ---------------------------------------------------------------------------

HOOK_CATEGORIES: tuple[str, ...] = (
    "Shock",
    "Curiosity",
    "Conflict",
    "Emotional",
    "Debate",
    "Irony",
    "Suspense",
    "One-Line Story",
    "Quote-Inspired",
    "Action-Based",
    "Cinematic",
    "High-Share Potential",
    "Controversial",
)


COPY_SYSTEM = f"""
You write hooks and captions for short-form video clips.

You are given the exact transcript of one clip. Everything you write must be
supported by that transcript.

Hooks must:
- be specific to THIS clip, never generic filler that would fit any video
- be short enough to read at a glance (under about 12 words)
- be grammatically correct with correct spelling
- create interest without deceiving
- avoid revealing the entire payoff, while never implying something that does
  not happen

Never write a hook that promises an event, revelation, confrontation or
statistic that the transcript does not contain. Curiosity is fine; false
promises are not.

{FACTUALITY_RULES}
""".strip()


COPY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hooks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["category", "text"],
            },
        },
        "best_hook_index": {"type": "integer"},
        "ranking": {"type": "array", "items": {"type": "integer"}},
        "caption": {"type": "string"},
    },
    "required": ["hooks", "best_hook_index", "ranking", "caption"],
}


def build_copy_prompt(
    *,
    transcript: str,
    topic: str,
    speakers: list[str],
    duration: float,
    visual_note: str = "",
) -> str:
    categories = "\n".join(
        f"{i + 1}. {name}" for i, name in enumerate(HOOK_CATEGORIES)
    )
    visual_line = f"\nVisible in the clip: {visual_note}" if visual_note else ""

    return f"""
CLIP TRANSCRIPT (this is everything the viewer will hear):
\"\"\"
{transcript}
\"\"\"

Topic: {topic or "not specified"}
Speakers: {", ".join(speakers) or "unknown"}
Duration: {duration:.0f} seconds{visual_line}

Write exactly {len(HOOK_CATEGORIES)} hooks, one for each category below, in this
order:

{categories}

Then:
- best_hook_index: the 0-based index of the single strongest hook
- ranking: all {len(HOOK_CATEGORIES)} indices (0-based), strongest first
- caption: one finished, ready-to-post caption describing this clip accurately.
  Two or three sentences. No hashtags unless they name something explicitly
  discussed. Do not invent context.

Every hook and the caption must be supported by the transcript above.

Return JSON only.
""".strip()


# ---------------------------------------------------------------------------
# Context validation
# ---------------------------------------------------------------------------

VALIDATION_SYSTEM = f"""
You are a fact-checking editor. You decide whether a proposed video clip can be
understood on its own and whether cutting it changes its meaning.

You are shown the clip transcript plus what was said immediately before and
after it in the source.

Be strict. A clip that is technically coherent but silently drops a qualifier,
a condition, or the fact that the speaker is describing someone else's view is
misleading, and must be flagged.

{FACTUALITY_RULES}
""".strip()


VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "understandable": {"type": "boolean"},
        "meaning_preserved": {"type": "boolean"},
        "missing_context": {"type": "string"},
        "suggested_start_shift": {"type": "number"},
        "verdict": {"type": "string", "enum": ["accept", "expand", "reject"]},
        "note": {"type": "string"},
    },
    "required": [
        "understandable",
        "meaning_preserved",
        "missing_context",
        "suggested_start_shift",
        "verdict",
        "note",
    ],
}


def build_validation_prompt(
    *, before: str, text: str, after: str, start: float, end: float
) -> str:
    return f"""
[SAID IMMEDIATELY BEFORE THE CLIP]
{before or "(nothing -- clip starts near the beginning of the video)"}

[THE CLIP ITSELF: {start:.1f}s to {end:.1f}s]
{text}

[SAID IMMEDIATELY AFTER THE CLIP]
{after or "(nothing -- clip runs to the end of the video)"}

Decide:
- understandable: can a viewer who has not seen the source follow this clip
- meaning_preserved: does the clip represent what the speaker actually meant,
  including any qualifier or condition
- missing_context: what a viewer would be missing, or empty string if nothing
- suggested_start_shift: negative seconds to move the start EARLIER to include
  needed setup (0 if none needed, never positive, never below -20)
- verdict: accept, expand (needs more context but is salvageable), or reject
  (misleading or incomprehensible even with more context)
- note: one short sentence explaining the verdict

Return JSON only.
""".strip()
