"""Prompts and JSON schemas for the reasoning stages.

Two rules shape everything here:

* The model reasons over *evidence we extracted*; it never sees or touches the
  video, and it never emits a command. It returns judgements as JSON.
* Factuality constraints are stated as hard rules, and are re-checked in code
  afterwards. A prompt is a request, not a guarantee.

The copy prompts target short-form confrontation content (street interviews,
protests, public disputes, political debates), where retention is decided in the
first seconds. Engagement and accuracy are not in tension here: a hook that
promises something the clip does not deliver loses the viewer anyway.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Shared safety language
# ---------------------------------------------------------------------------

FACTUALITY_RULES = """
ACCURACY RULES (these override every other instruction):
- Use only what is present in the supplied transcript and observations.
- Never invent quotes, statistics, names, places, events, motives or outcomes.
- Never promise a moment the clip does not contain.
- Do not drop a qualifier if removing it changes the meaning of a claim.
- Do not assert who was right, who broke the law, or what someone intended.
- Describe what happens; let viewers judge it.
- Never write persuasion aimed at a demographic, ethnic or voter group.
""".strip()


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

EVALUATION_SYSTEM = f"""
You select moments to cut from long videos into vertical short-form clips
(Reels / Shorts / TikTok).

You judge ONLY from the supplied transcript and observations. You never see the
video. You return JSON and nothing else. Be terse.

WHAT MAKES A STRONG CLIP, in priority order:

1. THE FIRST 3 SECONDS. A viewer who is not gripped immediately never sees the
   rest. The opening line must land: a confrontation, an accusation, a refusal,
   a shocking claim, a direct challenge, or raised voices already in progress.
   A clip that opens with throat-clearing, background, or a calm setup is weak
   no matter how good the middle is.
2. CONFLICT AND STAKES. Disagreement, confrontation, someone being challenged
   and having to answer, a reversal, a moment people will take sides over.
3. SELF-CONTAINED. Understandable without the rest of the source.
4. A COMPLETE BEAT. It resolves or lands on something, rather than stopping
   mid-idea.

REJECT: logistics, introductions, sponsor reads, sign-offs, background
explanation with no friction, and anything whose payoff sits outside the window.

Do not reward a clip merely for containing dramatic words. The moment itself
must be strong.

{FACTUALITY_RULES}
""".strip()


EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string", "maxLength": 24},
                    "recommended": {"type": "boolean"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "opening_strength": {"type": "number"},
                    "quality_score": {"type": "number"},
                    "topic": {"type": "string", "maxLength": 60},
                    "reason": {"type": "string", "maxLength": 90},
                    "context_dependency": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "complete_thought": {"type": "boolean"},
                },
                "required": [
                    "candidate_id",
                    "recommended",
                    "start",
                    "end",
                    "opening_strength",
                    "quality_score",
                    "topic",
                    "reason",
                    "context_dependency",
                    "complete_thought",
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
    """Render a compact evidence bundle for a batch of candidates."""
    blocks: list[str] = []
    for c in candidates:
        lines = [f"### {c['id']}  ({c['start']:.0f}s-{c['end']:.0f}s, {c['duration']:.0f}s)"]
        if c.get("speakers"):
            lines.append(f"Speakers: {', '.join(c['speakers'])}")
        signals = []
        if c.get("conflict_note"):
            signals.append(c["conflict_note"])
        if c.get("audio_events"):
            signals.append(", ".join(c["audio_events"]))
        if c.get("vision"):
            signals.append(c["vision"])
        if signals:
            lines.append(f"Signals: {' | '.join(signals)}")
        if c.get("context_before"):
            lines.append(f"[before] ...{c['context_before']}")
        lines.append(f"OPENS WITH: \"{c.get('opening_text', '')}\"")
        lines.append(f"[clip] {c['text']}")
        if c.get("context_after"):
            lines.append(f"[after] {c['context_after']}...")
        blocks.append("\n".join(lines))

    ids = ", ".join(c["id"] for c in candidates)

    return f"""
Evaluate each candidate as a vertical short-form clip.

{chr(10).join(blocks)}

---
Return one object per candidate ({ids}) with:

- candidate_id: exact id above
- recommended: true only if this would genuinely hold a scrolling viewer
- start / end: absolute seconds. Keep the given window unless it is wrong; you
  may move either edge by up to 15s to start on a stronger line or to finish a
  beat. Duration must stay between {min_duration:.0f}s and {max_duration:.0f}s.
  If a stronger opening line exists a few seconds later, move the start there.
- opening_strength: 0-1, how hard the FIRST 3 SECONDS hit
- quality_score: 0-1 overall
- topic: 3-6 plain words, no underscores
- reason: MAXIMUM 12 WORDS
- context_dependency: low / medium / high
- complete_thought: true or false

Return COMPACT JSON on a single line. Do not pretty-print or indent:
every wasted character costs generation time.
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

Report:
- description: one or two sentences on what is visibly happening
- people_visible: how many people are clearly visible (0 if none)
- on_screen_text: any readable text or graphics; empty string if none
- notable_events: visible actions -- pointing, shoving, grabbing, walking away,
  a crowd closing in, someone turning their back, a raised object. Empty list if
  nothing notable.
- visual_interest: 0.0 to 1.0
- is_talking_head: true if it is mostly people talking

JSON only.
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

# Generic openers that would fit any video. Rejected in code as well as here.
BANNED_HOOK_PHRASES: tuple[str, ...] = (
    "watch this",
    "you won't believe",
    "you wont believe",
    "then everything changed",
    "reporter covers",
    "political debate erupts",
    "things got heated",
    "nobody expected this",
    "this happened next",
    "what happens next",
    "wait for it",
    "must watch",
)

# Concrete action verbs the hooks should reach for.
PREFERRED_VERBS: tuple[str, ...] = (
    "blocked", "grabbed", "refused", "ignored", "interrupted", "walked away",
    "challenged", "stopped", "confronted", "laughed", "snapped", "turned around",
    "called out", "surrounded", "demanded",
)


_CAPTION_RULES = """
CAPTION -- you are also an expert viral Instagram Reels copywriter for US street
interviews, protests and public confrontations.

Write it as NARRATION, in the third person. Describe what happens. NEVER paste
the transcript back as a paragraph, and never open with the raw quote.

NAMING PEOPLE
- Use a real name ONLY if that exact name is spoken in the transcript.
- Otherwise describe people by role: "the reporter", "a demonstrator",
  "the official", "a woman in the crowd".
- NEVER write "Speaker A", "Speaker B" or any similar label. Those are internal
  audio-processing tags, not people.

LENGTH: aim for 170-220 words across the four body paragraphs. Short paragraphs,
but substantial ones -- one thin sentence per paragraph is far too little.

The four body paragraphs, in order:
1. trigger     - the exact moment that set the conflict off
2. escalation  - how it escalated, who did what next
3. reaction    - how the other side and any bystanders responded. END this
                 paragraph with a line pushing the viewer to watch, such as
                 "You have to watch the clip to see how fast this spiralled."
4. debate      - why viewers are split. Use the shape:
                 "Critics argue ... while others insist ..."

HASHTAGS: exactly 5, PascalCase, no spaces. Mix the topic, the event type and
any place or person actually named -- e.g. #StreetInterview
#PublicConfrontation #Sacramento #ViralVideo.

Here is the standard to match (from a reference reel):

    HEADLINE: SIDEWALK STANDOFF ESCALATES AS STATE OFFICIAL CLAIMS THEFT ON CAMERA
    TRIGGER: The confrontation turns hostile on the sidewalk when a state
    official aggressively snatches papers, igniting a heated dispute over
    personal property.
    ESCALATION: Tension spikes instantly outside the Capitol as he loudly
    accuses the reporter of committing a crime, repeatedly demanding his
    property back.
    REACTION: Instead of handing the papers back, the reporter holds them just
    out of reach, taunting the official while displaying the stack of printed
    insults. You have to watch the clip to see how fast this altercation
    spiralled on camera.
    DEBATE: This clash has divided social media. Critics argue a government
    official had no business parading personal insults in public, while others
    insist taking and holding someone else's property crossed a major line.
    QUESTION: Who crossed the line?
""".strip()


COPY_SYSTEM = f"""
You are one of the world's best short-form content strategists and Instagram
Reels hook writers.

Your specialty is ultra-high-retention hooks for US street interviews, public
confrontations, political debates, protests, breaking-news moments and social
conflicts.

You optimise for: scroll stopping, 3-second retention, watch time, shares,
comments, saves, and audience debate.

METHOD
First identify, from the transcript:
- the exact trigger that started the conflict
- the highest emotional moment
- the biggest unexpected action
- the moment viewers would most want to see

Then write hooks that make someone think: "What happened?", "Why did they react
like that?", "Whose side am I on?", "I need to see this."

RULES
- Under 9 words whenever possible.
- Create a strong information gap, but NEVER spoil the ending.
- Use concrete actions, not vague statements. Prefer verbs like:
  {", ".join(PREFERRED_VERBS)}.
- Build curiosity through specificity. Every hook must be unique to THIS clip.
- If a hook could fit hundreds of videos, rewrite it.
- Prioritise emotional tension over generic suspense.
- Each hook must use a DIFFERENT psychological trigger.
- Include 1-2 emojis from 🚨👀😶😳😳 at the start, the end, or both.

NEVER USE these phrases: {", ".join(BANNED_HOOK_PHRASES)}.

{FACTUALITY_RULES}

Never promise a confrontation, revelation or statistic the transcript does not
contain. Curiosity is the goal; false promises lose the viewer and the account.

{_CAPTION_RULES}

{FACTUALITY_RULES}
""".strip()




# Hooks and the caption are produced in ONE call. Two calls per clip doubled the
# request overhead and prompt prefill for output that shares all its context, and
# copy generation is the slowest stage once selection is fast.
COPY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # minItems/maxItems are load-bearing, not decoration: without them the
        # grammar-constrained decoder is free to emit "hooks": [] and satisfy
        # the schema, which silently drops every clip onto the fallback path.
        "hooks": {
            "type": "array",
            "minItems": 13,
            "maxItems": 13,
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "maxLength": 24},
                    "text": {"type": "string", "minLength": 12, "maxLength": 90},
                },
                "required": ["category", "text"],
            },
        },
        "best_hook_index": {"type": "integer"},
        "ranking": {
            "type": "array",
            "minItems": 13,
            "maxItems": 13,
            "items": {"type": "integer"},
        },
        "headline": {"type": "string", "minLength": 25, "maxLength": 120},
        "trigger": {"type": "string", "minLength": 150, "maxLength": 420},
        "escalation": {"type": "string", "minLength": 150, "maxLength": 420},
        "reaction": {"type": "string", "minLength": 150, "maxLength": 420},
        "debate": {"type": "string", "minLength": 170, "maxLength": 460},
        "question": {"type": "string", "minLength": 12, "maxLength": 90},
        "location": {"type": "string", "maxLength": 60},
        "hashtags": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {"type": "string", "maxLength": 30},
        },
    },
    "required": [
        "hooks", "best_hook_index", "ranking",
        "headline", "trigger", "escalation", "reaction", "debate", "question",
        "location", "hashtags",
    ],
}


def build_copy_prompt(
    *,
    transcript: str,
    opening_line: str,
    topic: str,
    speakers: list[str],
    duration: float,
    conflict_note: str = "",
    visual_note: str = "",
) -> str:
    """One prompt for hooks and caption -- they share all their context."""
    categories = "\n".join(
        f"{i + 1}. {name}" for i, name in enumerate(HOOK_CATEGORIES)
    )
    extras = []
    if conflict_note:
        extras.append(f"Detected in the clip: {conflict_note}")
    if visual_note:
        extras.append(f"Visible: {visual_note}")
    extra_block = ("\n" + "\n".join(extras)) if extras else ""

    return f"""
CLIP TRANSCRIPT (everything the viewer hears):
\"\"\"
{transcript}
\"\"\"

The clip OPENS on: "{opening_line}"
Topic: {topic or "unspecified"}
Speakers: {", ".join(speakers) or "unknown"}
Length: {duration:.0f}s{extra_block}

PART 1 - HOOKS
Write exactly {len(HOOK_CATEGORIES)} hooks, one per category, in this order:

{categories}

- best_hook_index: 0-based index of the strongest hook
- ranking: all {len(HOOK_CATEGORIES)} indices (0-based), strongest first

PART 2 - CAPTION (narrate in the third person; do NOT quote the transcript back)
- headline: breaking-news conflict headline IN CAPS, 8-14 words, no emojis
- trigger: paragraph on the exact moment that set the conflict off
- escalation: paragraph on how it escalated and who did what next
- reaction: paragraph on how others responded, ending with a line telling the
  viewer they have to watch the clip
- debate: paragraph on why viewers are split -- "Critics argue ... while others
  insist ..."
- question: a closed, side-taking question
- location: "City, State" ONLY if stated in the transcript, else ""
- hashtags: exactly 5, PascalCase, each starting with #

The four caption paragraphs together should total 170-220 words.
Put hashtags ONLY in the hashtags field, never inside a paragraph.
Use emoji ONLY in the headline and the question. Body paragraphs are plain
prose -- do not pad them with emoji or repeated tags to reach a length.
Refer to people by name only if the name is spoken in the transcript, otherwise
by role. Never write "Speaker A"/"Speaker B".

Everything must be supported by the transcript.
Return COMPACT single-line JSON. Do not indent or pretty-print.
""".strip()




# ---------------------------------------------------------------------------
# Context validation
# ---------------------------------------------------------------------------

VALIDATION_SYSTEM = f"""
You are a fact-checking editor. You decide whether a proposed video clip can be
understood on its own and whether cutting it changes its meaning.

You are shown the clip transcript plus what was said immediately before and
after it in the source.

Be strict about misrepresentation: a clip that silently drops a qualifier, a
condition, or the fact that a speaker is describing someone else's view is
misleading and must be flagged.

Be permissive about intensity: an argument that starts abruptly is normal for
short-form and is NOT a defect.

{FACTUALITY_RULES}
""".strip()


VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "understandable": {"type": "boolean"},
        "meaning_preserved": {"type": "boolean"},
        "missing_context": {"type": "string", "maxLength": 160},
        "suggested_start_shift": {"type": "number"},
        "verdict": {"type": "string", "enum": ["accept", "expand", "reject"]},
        "note": {"type": "string", "maxLength": 120},
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
[SAID IMMEDIATELY BEFORE]
{before or "(nothing -- clip starts near the beginning)"}

[THE CLIP: {start:.0f}s to {end:.0f}s]
{text}

[SAID IMMEDIATELY AFTER]
{after or "(nothing -- clip runs to the end)"}

Decide:
- understandable: can a viewer who has not seen the source follow this
- meaning_preserved: does it represent what the speaker actually meant,
  including any qualifier or condition
- missing_context: what a viewer would be missing, or ""
- suggested_start_shift: negative seconds to move the start EARLIER for needed
  setup (0 if none, never positive, never below -20)
- verdict: accept / expand / reject. Reject ONLY if misleading or
  incomprehensible -- not merely because it starts abruptly.
- note: one short sentence

JSON only.
""".strip()
