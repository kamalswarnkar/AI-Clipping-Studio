"""Prompts for hook and caption writing.

These follow `hooks.txt` and `caption.txt` in the project root, which are the
product spec for this feature. The rules there are reproduced closely; what is
deliberately *not* reproduced is the output layout.

The model is asked for the pieces as JSON and the exact layout is assembled in
`ai/copy.py`. Asking a 7B model to reproduce a template with emoji headers,
separators, numbered categories and a ranking block gets the format wrong often
enough to be useless, and there is nothing to gain from letting it try: the
template is fixed, so it belongs in code.

Everything is third person. These are notes for whoever publishes the clip, and
a hook written as "I confronted him" is wrong about who is speaking.
"""

from __future__ import annotations

# The 13 categories from hooks.txt, in order. The order is load-bearing: it is
# the order they are printed in.
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

# Phrases hooks.txt names as generic. Checked in code, not left to the model.
BANNED_HOOK_PHRASES: tuple[str, ...] = (
    "watch this",
    "you won't believe",
    "you wont believe",
    "then everything changed",
    "nobody expected this",
    "this happened next",
    "things got heated",
    "political debate erupts",
    "you need to see this",
    "wait for it",
    "this is why",
)

# Verbs hooks.txt lists as useful action language.
PREFERRED_VERBS: tuple[str, ...] = (
    "blocked", "grabbed", "refused", "ignored", "interrupted", "walked away",
    "challenged", "stopped", "confronted", "laughed", "snapped", "turned around",
    "called out", "surrounded", "demanded", "pointed", "shouted", "walked toward",
    "backed away", "cut off",
)

_SHARED_GROUNDING = """
The context and transcript are the only sources of truth. Never invent motives,
facts, locations, identities, events, quotes or statements. If something cannot
be supported from what you were given, leave it out.

Write in the third person throughout. Describe the people in the clip as "he",
"she", "they", "the interviewer", "a protester" -- never "I", never "we". The
one exception is text inside quotation marks, which reproduces what someone
actually said.
""".strip()

HOOKS_SYSTEM = f"""
You are an elite short-form content strategist and Instagram Reels hook writer,
specialising in US street interviews, public confrontations, political debates,
protests and heated arguments.

Your only objective is scroll-stopping power, 1-3 second retention, watch time,
rewatches, shares, comments and audience debate.

{_SHARED_GROUNDING}

Find the single strongest viral angle. Do not summarise the clip. Look for the
strongest combination of CURIOSITY + TENSION + SPECIFICITY + EMOTIONAL STAKES:
what starts the conflict, who confronts whom, which reaction escalates it, and
what a viewer would need to keep watching to find out.

Every hook must be specific to THIS clip. If a hook could describe hundreds of
unrelated videos, it is wrong -- write a more specific one.

Rules for every hook:
- 4 to 9 words. Never sacrifice clarity to be shorter.
- Concrete actions and strong verbs: {", ".join(PREFERRED_VERBS[:12])}.
- Create an information gap. Do not reveal the outcome.
- Do not exaggerate, and do not call something shocking unless the material
  supports it.
- No generic clickbait: {", ".join(repr(p) for p in BANNED_HOOK_PHRASES[:6])}.

BAD: "Political argument gets heated"
BETTER: "He refused to answer, then walked away"
BAD: "People argue at a protest"
BETTER: "She confronted him in front of the crowd"
""".strip()

HOOKS_SCHEMA = {
    "type": "object",
    "properties": {
        "best_hook": {"type": "string", "minLength": 12, "maxLength": 90},
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
        "ranking": {
            "type": "array",
            "minItems": 13,
            "maxItems": 13,
            "items": {"type": "integer"},
        },
    },
    "required": ["best_hook", "hooks", "ranking"],
}

CAPTION_SYSTEM = f"""
You are an elite viral Instagram Reels copywriter, specialising in US street
interviews, public confrontations, political debates, protests and social
conflicts.

You write for watch-through, shares, comments, audience debate and rewatching.

{_SHARED_GROUNDING}

Find the single strongest conflict narrative and build the caption around
TRIGGER -> ESCALATION -> DIVISION -> DEBATE:

- trigger: the exact moment that started the confrontation or disagreement.
- escalation: how it intensified, and the strongest reactions.
- division: why viewers may read the situation differently.
- question: a forced choice that makes viewers take a side.

Write like social media copy, not a news report: short paragraphs, strong verbs,
concrete details, simple language. Avoid policy analysis, academic language,
excessive background and filler. Prioritise information density over length.

The question must force a choice between at least two positions -- "Was he right
or wrong?", "Who crossed the line?", "Was this justified or out of line?". Never
"What do you think?", "Thoughts?", "Do you agree?".

Give a `location` as "City, State" ONLY if the material clearly establishes a US
location. Otherwise return "". Never guess.

Give EXACTLY 5 hashtags, each directly relevant to the subject, the kind of
event, the audience or the discussion. No generic hashtag spam.
""".strip()

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "minLength": 20, "maxLength": 120},
        "trigger": {"type": "string", "minLength": 160, "maxLength": 480},
        "escalation": {"type": "string", "minLength": 160, "maxLength": 480},
        "division": {"type": "string", "minLength": 160, "maxLength": 480},
        "question": {"type": "string", "minLength": 15, "maxLength": 140},
        "location": {"type": "string", "maxLength": 60},
        "hashtags": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 3, "maxLength": 40},
        },
    },
    "required": [
        "headline",
        "trigger",
        "escalation",
        "division",
        "question",
        "hashtags",
    ],
}


def build_hooks_prompt(
    *, context: str, transcript: str, video_context: str, seen: str = ""
) -> str:
    categories = "\n".join(
        f"{i}. {name}" for i, name in enumerate(HOOK_CATEGORIES, start=1)
    )
    return f"""
[THE VIDEO THIS CLIP COMES FROM]
{video_context or "(not available)"}

[WHAT IS VISIBLE IN THE CLIP -- THE PRIMARY SOURCE]
{seen or "(the clip was not watched; use the transcript and context)"}

[CONTEXT OF THE CLIP]
{context or "(not available)"}

[TRANSCRIPT OF THE CLIP -- SECONDARY, FOR DIALOGUE AND NAMES]
{transcript}

Write `best_hook`: the single hook most likely to make a stranger stop scrolling
and watch this clip.

Then write `hooks`: exactly 13 hooks, one per category, in this order, with
`category` set to the category name exactly as written here:

{categories}

Then write `ranking`: the numbers 1 to 13, each exactly once, ordering those 13
hooks strongest to weakest by how likely they are to produce scroll stops,
watch-through, rewatches, comments, shares and debate -- not by writing quality.

JSON only.
""".strip()


def build_caption_prompt(
    *,
    context: str,
    transcript: str,
    video_context: str,
    min_words: int,
    max_words: int,
    seen: str = "",
) -> str:
    return f"""
[THE VIDEO THIS CLIP COMES FROM]
{video_context or "(not available)"}

[WHAT IS VISIBLE IN THE CLIP -- THE PRIMARY SOURCE]
{seen or "(the clip was not watched; use the transcript and context)"}

[CONTEXT OF THE CLIP]
{context or "(not available)"}

[TRANSCRIPT OF THE CLIP -- SECONDARY, FOR DIALOGUE AND NAMES]
{transcript}

Write the caption pieces. The three paragraphs plus the question must come to
between {min_words} and {max_words} words in total. Use the room: name the
people, the bill, the place and the specific claims rather than gesturing at
them. Do not pad with filler, and do not exceed the limit.

- headline: the most compelling conflict or action in this clip. Specific, not
  generic. No emoji -- they are added afterwards.
- trigger, escalation, division: one short paragraph each.
- question: the forced-choice debate question.
- location: "City, State" only if clearly established, otherwise "".
- hashtags: exactly 5, without the leading '#'.

JSON only.
""".strip()


MISSING_HOOKS_SCHEMA = {
    "type": "object",
    "properties": {
        "hooks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 13,
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "maxLength": 24},
                    "text": {"type": "string", "minLength": 12, "maxLength": 90},
                },
                "required": ["category", "text"],
            },
        }
    },
    "required": ["hooks"],
}


def build_missing_hooks_prompt(
    *, context: str, transcript: str, video_context: str, categories: list[str],
    existing: list[str],
) -> str:
    """Ask for only the categories the first response did not deliver.

    Thirteen hooks plus a ranking is a long answer for a small model, and what
    comes back is often short. Asking again for the gap is cheaper and more
    reliable than retrying the whole thing.
    """
    wanted = "\n".join(f"- {c}" for c in categories)
    already = "\n".join(f"- {h}" for h in existing) or "(none yet)"
    return f"""
[THE VIDEO THIS CLIP COMES FROM]
{video_context or "(not available)"}

[CONTEXT OF THE CLIP]
{context or "(not available)"}

[TRANSCRIPT OF THE CLIP]
{transcript}

Hooks already written for this clip -- do not repeat these, and do not rephrase
them:
{already}

Write one hook for each of these categories, and no others:
{wanted}

Set `category` to the category name exactly as written above. 4 to 9 words each,
third person, specific to this clip.

JSON only.
""".strip()


# ---------------------------------------------------------------------------
# Looking at the clip
# ---------------------------------------------------------------------------
# hooks.txt and caption.txt both name the video as the primary source of truth
# and the transcript as secondary. A text model cannot see the video, so the
# clip is watched first by a vision model and its description is what the
# writing prompts receive.

CLIP_VISION_SYSTEM = """
You describe video frames for an editor, factually and in the third person.

Describe only what is visible. Do not infer motives, do not guess at identities,
do not describe anything you cannot see. If people are arguing, say what makes
that visible -- posture, gestures, distance, where they are looking.
""".strip()

CLIP_VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 80, "maxLength": 700},
        "people_visible": {"type": "integer"},
        "on_screen_text": {"type": "string", "maxLength": 200},
    },
    "required": ["description"],
}

CLIP_VISION_PROMPT = """
These frames are taken in order from one short video clip.

Describe what visibly happens across them: who is present, what they are doing,
how they are standing and moving, any confrontation or physical action, the
crowd, and any signs or on-screen text you can actually read.

Set `people_visible` to the number of people clearly visible, and
`on_screen_text` to text you can read in the frames, or "".

JSON only.
""".strip()
