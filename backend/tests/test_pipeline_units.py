"""Unit tests for the deterministic pieces of the pipeline.

These cover the logic that must be correct regardless of what any model says:
boundary snapping, duplicate detection, factuality checking, subtitle timing,
crop expressions and path safety.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.copy import check_factuality  # noqa: E402
from app.analysis.boundaries import (  # noqa: E402
    is_mid_word,
    refine_boundaries,
    sentence_ends,
    sentence_starts,
    text_starts_dependently,
)
from app.analysis.candidates import trim_filler_opening  # noqa: E402
from app.analysis.selection import (  # noqa: E402
    is_duplicate,
    text_similarity,
    time_overlap_ratio,
)
from app.models.domain import (  # noqa: E402
    ClipPlan,
    CropKeyframe,
    FaceBox,
    FrameAnalysis,
    Transcript,
    TranscriptSegment,
    VisualAnalysis,
    Word,
)
from app.services.storage import safe_stem, sanitize_filename  # noqa: E402
from app.video.reframe import build_crop_expression, compute_crop_path  # noqa: E402
from app.video.subtitles import build_ass, group_words_into_cues  # noqa: E402


def _words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(word=w, start=s, end=e) for w, s, e in spec]


@pytest.fixture
def transcript() -> Transcript:
    """Two sentences, then a pause, then two more."""
    seg_a = TranscriptSegment(
        id=0,
        start=0.0,
        end=4.0,
        text="This is the first sentence. Here is a second one.",
        words=_words(
            [
                ("This", 0.0, 0.3), ("is", 0.3, 0.5), ("the", 0.5, 0.7),
                ("first", 0.7, 1.1), ("sentence.", 1.1, 1.7),
                ("Here", 2.0, 2.3), ("is", 2.3, 2.5), ("a", 2.5, 2.6),
                ("second", 2.6, 3.2), ("one.", 3.2, 4.0),
            ]
        ),
    )
    seg_b = TranscriptSegment(
        id=1,
        start=6.0,
        end=10.0,
        text="But it depends on context. That is the whole point.",
        words=_words(
            [
                ("But", 6.0, 6.2), ("it", 6.2, 6.4), ("depends", 6.4, 6.9),
                ("on", 6.9, 7.1), ("context.", 7.1, 7.8),
                ("That", 8.2, 8.5), ("is", 8.5, 8.7), ("the", 8.7, 8.9),
                ("whole", 8.9, 9.3), ("point.", 9.3, 10.0),
            ]
        ),
    )
    return Transcript(language="en", duration=10.0, segments=[seg_a, seg_b])


# ---------------------------------------------------------------------------
# Transcript windowing
# ---------------------------------------------------------------------------

def test_text_in_window_excludes_words_outside(transcript: Transcript) -> None:
    """Exact-window text must not leak in words from an overlapping sentence."""
    # text_between is segment-granular and pulls the whole first segment in.
    assert "first sentence" in transcript.text_between(2.0, 4.0)
    # text_in_window is word-accurate.
    exact = transcript.text_in_window(2.0, 4.0)
    assert "first" not in exact
    assert exact.startswith("Here is a second")


def test_words_between_uses_midpoint(transcript: Transcript) -> None:
    # "sentence." spans 1.1-1.7, so its midpoint (1.4) falls inside a 0-1.5
    # window and the word is included; "Here" (2.0-2.3) is not.
    words = transcript.words_between(0.0, 1.5)
    assert [w.word for w in words] == ["This", "is", "the", "first", "sentence."]

    # A window ending before that midpoint excludes it.
    assert [w.word for w in transcript.words_between(0.0, 1.3)] == [
        "This", "is", "the", "first",
    ]


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def test_sentence_starts_and_ends(transcript: Transcript) -> None:
    assert 0.0 in sentence_starts(transcript)
    assert 2.0 in sentence_starts(transcript)  # after "sentence."
    assert 1.7 in sentence_ends(transcript)


def test_is_mid_word(transcript: Transcript) -> None:
    assert is_mid_word(transcript, 0.9)      # inside "first"
    assert not is_mid_word(transcript, 1.85)  # in the gap


def test_dependent_openers() -> None:
    assert text_starts_dependently("But it depends on context.")
    assert text_starts_dependently("So that is why we did it.")
    assert not text_starts_dependently("Retention is the growth engine.")


def test_refine_snaps_to_sentence_and_never_mid_word(transcript: Transcript) -> None:
    result = refine_boundaries(
        2.1, 3.9, transcript=transcript, min_duration=1.0, max_duration=10.0,
        total_duration=10.0,
    )
    assert result.start <= 2.0            # snapped back to the sentence start
    assert not is_mid_word(transcript, result.start)
    assert not is_mid_word(transcript, result.end)


def test_refine_enforces_minimum_duration(transcript: Transcript) -> None:
    result = refine_boundaries(
        2.0, 3.0, transcript=transcript, min_duration=6.0, max_duration=10.0,
        total_duration=10.0,
    )
    assert result.end - result.start >= 6.0 - 1e-6


def test_refine_respects_maximum_duration(transcript: Transcript) -> None:
    result = refine_boundaries(
        0.0, 10.0, transcript=transcript, min_duration=1.0, max_duration=5.0,
        total_duration=10.0,
    )
    assert result.end - result.start <= 5.0 + 1e-6


def test_refine_never_exceeds_video_length(transcript: Transcript) -> None:
    result = refine_boundaries(
        8.0, 12.0, transcript=transcript, min_duration=1.0, max_duration=30.0,
        total_duration=10.0,
    )
    assert result.end <= 10.0


# ---------------------------------------------------------------------------
# Filler trimming
# ---------------------------------------------------------------------------

def test_trim_filler_opening_extends_end_to_keep_minimum() -> None:
    """Trimming filler must not produce an under-length clip."""
    seg = TranscriptSegment(
        id=0, start=0.0, end=14.0,
        text="Quick reminder that we record every Tuesday. The real point is retention.",
        words=_words(
            [("Quick", 0.0, 0.4), ("reminder", 0.4, 1.0), ("that", 1.0, 1.2),
             ("we", 1.2, 1.4), ("record", 1.4, 1.9), ("every", 1.9, 2.3),
             ("Tuesday.", 2.3, 3.0),
             ("The", 3.4, 3.6), ("real", 3.6, 4.0), ("point", 4.0, 4.4),
             ("is", 4.4, 4.6), ("retention.", 4.6, 5.4)]
        ),
    )
    tr = Transcript(duration=30.0, segments=[seg])
    start, end, trimmed = trim_filler_opening(
        0.0, 8.0, transcript=tr, min_duration=10.0, max_duration=60.0,
        total_duration=30.0,
    )
    assert trimmed
    assert start >= 3.0                 # filler dropped
    assert end - start >= 10.0          # duration recovered by extending the end


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def test_time_overlap_ratio_uses_shorter_clip() -> None:
    assert time_overlap_ratio(0, 100, 0, 10) == pytest.approx(1.0)
    assert time_overlap_ratio(0, 10, 20, 30) == 0.0


def test_text_similarity() -> None:
    a = "retention is the growth engine for every company"
    assert text_similarity(a, a) == pytest.approx(1.0)
    assert text_similarity(a, "completely unrelated topic about weather") < 0.2


def _plan(index: int, start: float, end: float, text: str) -> ClipPlan:
    return ClipPlan(id=f"c{index}", index=index, start=start, end=end, transcript=text)


def test_is_duplicate_detects_time_overlap() -> None:
    a = _plan(1, 10, 40, "one text")
    b = _plan(2, 15, 45, "totally different words here entirely")
    same, why = is_duplicate(a, b, iou_threshold=0.35, text_threshold=0.72)
    assert same and "time" in why


def test_is_duplicate_detects_similar_text_without_overlap() -> None:
    text = "retention is the growth engine and acquisition follows from it"
    a = _plan(1, 10, 40, text)
    b = _plan(2, 200, 230, text)
    same, why = is_duplicate(a, b, iou_threshold=0.35, text_threshold=0.72)
    assert same and "similar" in why


def test_distinct_clips_are_not_duplicates() -> None:
    a = _plan(1, 10, 40, "retention is the growth engine for subscription products")
    b = _plan(2, 200, 230, "hiring senior engineers requires a different interview loop")
    same, _ = is_duplicate(a, b, iou_threshold=0.35, text_threshold=0.72)
    assert not same


# ---------------------------------------------------------------------------
# Factuality
# ---------------------------------------------------------------------------

TRANSCRIPT = (
    "We cut our support tickets by 60 percent without hiring a single person. "
    "Ninety one percent of those people never came back."
)


def test_factuality_accepts_supported_numbers() -> None:
    ok, _ = check_factuality("They cut support tickets by 60%", TRANSCRIPT)
    assert ok


def test_factuality_accepts_spelled_out_numbers() -> None:
    ok, _ = check_factuality("91% never returned", TRANSCRIPT)
    assert ok


def test_factuality_rejects_invented_number() -> None:
    ok, why = check_factuality("They cut support tickets by 85%", TRANSCRIPT)
    assert not ok and "85" in why


def test_factuality_rejects_invented_quote() -> None:
    ok, why = check_factuality('He said "we fired the whole team"', TRANSCRIPT)
    assert not ok and "quote" in why.lower()


def test_factuality_accepts_real_quote() -> None:
    ok, _ = check_factuality('He said "without hiring a single person"', TRANSCRIPT)
    assert ok


def test_factuality_rejects_single_quoted_misquote() -> None:
    """A paraphrase presented as a quote is still a misquote."""
    real = (
        "Most companies stay in acquisition mode long after they should have switched."
    )
    ok, why = check_factuality(
        "'Most companies stay in acquisition mode too long.'", real
    )
    assert not ok and "quote" in why.lower()


def test_factuality_accepts_single_quoted_real_quote() -> None:
    real = "Most companies stay in acquisition mode long after they should have switched."
    ok, _ = check_factuality("'stay in acquisition mode long after'", real)
    assert ok


def test_factuality_ignores_apostrophes_in_contractions() -> None:
    """Contractions must not be parsed as quote delimiters."""
    real = "It doesn't matter and we won't pretend it does for the audience."
    ok, why = check_factuality("It doesn't matter and we won't pretend", real)
    assert ok, why


def test_factuality_rejects_empty() -> None:
    ok, _ = check_factuality("   ", TRANSCRIPT)
    assert not ok


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------

def test_cues_are_relative_to_clip_start(transcript: Transcript) -> None:
    words = transcript.words_between(6.0, 10.0)
    cues = group_words_into_cues(words, clip_start=6.0, clip_end=10.0)
    assert cues
    assert cues[0][0] == pytest.approx(0.0, abs=0.05)
    assert all(end <= 4.05 for _, end, _ in cues)


def test_cues_do_not_overlap(transcript: Transcript) -> None:
    words = transcript.words_between(0.0, 10.0)
    cues = group_words_into_cues(words, clip_start=0.0, clip_end=10.0)
    for (_, end, _), (next_start, _, _) in zip(cues, cues[1:]):
        assert end <= next_start + 1e-6


def test_build_ass_has_style_and_dialogue(transcript: Transcript) -> None:
    ass = build_ass(
        transcript=transcript, clip_start=0.0, clip_end=10.0,
        width=1080, height=1920,
    )
    assert "[V4+ Styles]" in ass
    assert "Dialogue:" in ass
    assert "PlayResX: 1080" in ass


# ---------------------------------------------------------------------------
# Reframing
# ---------------------------------------------------------------------------

def test_crop_path_centres_when_no_faces() -> None:
    keyframes, strategy = compute_crop_path(
        visual=None, start=0.0, end=10.0, source_w=1920, source_h=1080,
    )
    assert len(keyframes) == 1
    assert "no faces" in strategy
    # 9:16 out of a 1920x1080 source is 607x1080, centred at x=656.
    assert keyframes[0].w == 606 or keyframes[0].w == 608
    assert keyframes[0].h == 1080


def test_crop_path_keeps_both_speakers_when_they_fit() -> None:
    frames = [
        FrameAnalysis(
            t=float(t),
            faces=[
                FaceBox(x=800, y=300, w=120, h=120),
                FaceBox(x=1000, y=300, w=120, h=120),
            ],
        )
        for t in range(0, 10)
    ]
    visual = VisualAnalysis(frames=frames, frame_width=1920, frame_height=1080)
    keyframes, strategy = compute_crop_path(
        visual=visual, start=0.0, end=10.0, source_w=1920, source_h=1080,
    )
    assert len(keyframes) == 1
    assert "static crop" in strategy
    # Both faces (800..1120) must sit inside the crop window.
    k = keyframes[0]
    assert k.x <= 800 and k.x + k.w >= 1120


def test_crop_expression_constant_and_interpolated() -> None:
    single = [CropKeyframe(t=0.0, x=100, y=0, w=600, h=1080)]
    assert build_crop_expression(single, "x") == "100"

    moving = [
        CropKeyframe(t=0.0, x=100, y=0, w=600, h=1080),
        CropKeyframe(t=2.0, x=500, y=0, w=600, h=1080),
    ]
    expr = build_crop_expression(moving, "x")
    assert "if(lt(t," in expr and "100" in expr and "500" in expr


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    ["../../etc/passwd.mp4", "..\\..\\windows\\system32\\a.mp4", "a/b/c.mp4"],
)
def test_sanitize_strips_directories(raw: str) -> None:
    cleaned = sanitize_filename(raw)
    assert "/" not in cleaned and "\\" not in cleaned and ".." not in cleaned


def test_sanitize_handles_reserved_windows_names() -> None:
    assert sanitize_filename("CON.mp4").startswith("_")
    assert sanitize_filename("NUL").startswith("_")


def test_sanitize_removes_illegal_characters() -> None:
    cleaned = sanitize_filename('my<vid>eo:"?*.mp4')
    assert not any(ch in cleaned for ch in '<>:"?*|')


def test_safe_stem_drops_extension() -> None:
    assert safe_stem("My Podcast Ep 12.mp4") == "My Podcast Ep 12"


def test_cues_break_on_speaker_change(transcript: Transcript) -> None:
    """A caption must never run two speakers together."""
    from app.models.domain import Diarization, SpeakerTurn

    diarization = Diarization(
        speakers=["Speaker A", "Speaker B"],
        turns=[
            SpeakerTurn(speaker="Speaker A", start=0.0, end=2.0),
            SpeakerTurn(speaker="Speaker B", start=2.0, end=10.0),
        ],
    )
    words = transcript.words_between(0.0, 4.0)
    cues = group_words_into_cues(
        words, clip_start=0.0, clip_end=4.0, diarization=diarization
    )
    # The turn changes at 2.0s, so no cue may straddle it.
    for start, end, _ in cues:
        assert not (start < 2.0 - 1e-6 and end > 2.0 + 1e-6)


# ---------------------------------------------------------------------------
# Truncated model output
# ---------------------------------------------------------------------------

def test_salvages_truncated_json_response() -> None:
    """A response cut off mid-object must not lose the items that completed."""
    from app.ai.llm.ollama_provider import _extract_json

    truncated = (
        '{"evaluations": ['
        '{"candidate_id": "c1", "quality_score": 0.9},'
        '{"candidate_id": "c2", "quality_score": 0.4},'
        '{"candidate_id": "c3", "quality_sc'
    )
    result = _extract_json(truncated)
    assert [e["candidate_id"] for e in result["evaluations"]] == ["c1", "c2"]


def test_extract_json_ignores_braces_inside_strings() -> None:
    from app.ai.llm.ollama_provider import _extract_json

    result = _extract_json('{"a": "a } brace", "b": [{"c": 1}, {"d": 2')
    assert result["a"] == "a } brace"
    assert result["b"] == [{"c": 1}]


def test_extract_json_still_parses_clean_output() -> None:
    from app.ai.llm.ollama_provider import _extract_json

    assert _extract_json('{"x": [1, 2, 3]}') == {"x": [1, 2, 3]}
    fenced = "```json\n" + '{"y": 5}' + "\n```"
    assert _extract_json(fenced) == {"y": 5}


# ---------------------------------------------------------------------------
# Opening quality (short-form retention)
# ---------------------------------------------------------------------------

def _continuation_transcript() -> Transcript:
    """Two ASR segments that split a single sentence in half.

    Whisper does this constantly on noisy audio, and it used to make the second
    segment look like a sentence start.
    """
    seg_a = TranscriptSegment(
        id=0, start=0.0, end=2.0, text="They are not going to be",
        words=_words([("They", 0.0, 0.3), ("are", 0.3, 0.6), ("not", 0.6, 0.9),
                      ("going", 0.9, 1.3), ("to", 1.3, 1.5), ("be", 1.5, 2.0)]),
    )
    # Starts 0.1s later -- a continuation, not a new sentence.
    seg_b = TranscriptSegment(
        id=1, start=2.1, end=4.0, text="paying taxes at all?",
        words=_words([("paying", 2.1, 2.5), ("taxes", 2.5, 2.9),
                      ("at", 2.9, 3.1), ("all?", 3.1, 4.0)]),
    )
    seg_c = TranscriptSegment(
        id=2, start=5.0, end=8.0, text="We should fix the fraud first.",
        words=_words([("We", 5.0, 5.2), ("should", 5.2, 5.5), ("fix", 5.5, 5.8),
                      ("the", 5.8, 6.0), ("fraud", 6.0, 6.5), ("first.", 6.5, 8.0)]),
    )
    return Transcript(duration=20.0, segments=[seg_a, seg_b, seg_c])


def test_segment_break_is_not_a_sentence_start() -> None:
    """A mid-sentence ASR split must not be offered as a clip opening."""
    starts = sentence_starts(_continuation_transcript())
    assert 0.0 in starts          # genuine start
    assert 2.1 not in starts      # mid-sentence continuation
    assert 5.0 in starts          # follows a completed sentence


def test_floor_start_prevents_resnapping_onto_trimmed_filler() -> None:
    """Snapping must not restore an opening the caller deliberately removed."""
    tr = _continuation_transcript()
    refined = refine_boundaries(
        5.0, 18.0, transcript=tr, min_duration=5.0, max_duration=60.0,
        total_duration=20.0, floor_start=5.0,
    )
    assert refined.start >= 5.0 - 0.2


def test_opening_punch_prefers_confrontation() -> None:
    """A confrontational opening must outscore a calm one."""
    from app.analysis.conflict import opening_punch_score

    calm = TranscriptSegment(
        id=0, start=0.0, end=3.0, text="So the weather was quite nice that day.",
        words=_words([("So", 0.0, 0.3), ("the", 0.3, 0.5), ("weather", 0.5, 1.0),
                      ("was", 1.0, 1.3), ("quite", 1.3, 1.7), ("nice", 1.7, 2.2),
                      ("that", 2.2, 2.6), ("day.", 2.6, 3.0)]),
    )
    hot = TranscriptSegment(
        id=0, start=0.0, end=3.0, text="Don't touch my things! You're lying.",
        words=_words([("Don't", 0.0, 0.3), ("touch", 0.3, 0.6), ("my", 0.6, 0.8),
                      ("things!", 0.8, 1.4), ("You're", 1.4, 1.8),
                      ("lying.", 1.8, 3.0)]),
    )
    calm_score = opening_punch_score(
        transcript=Transcript(duration=10.0, segments=[calm]),
        audio=None, diarization=None, start=0.0,
    )
    hot_score = opening_punch_score(
        transcript=Transcript(duration=10.0, segments=[hot]),
        audio=None, diarization=None, start=0.0,
    )
    assert hot_score > calm_score


def test_hook_quality_rejects_banned_phrases() -> None:
    from app.ai.copy import check_hook_quality

    assert not check_hook_quality("You won't believe what happened")[0]
    assert not check_hook_quality("Things got heated at the rally")[0]
    assert check_hook_quality("🚨 He grabbed the papers and walked away")[0]


def test_clean_hook_strips_model_artifacts() -> None:
    from app.ai.copy import _clean_hook

    assert _clean_hook("/Area Tensions Erupt") == "Area Tensions Erupt"
    assert _clean_hook('Why defend it? #FreeSpeech #Debate') == "Why defend it?"
    assert _clean_hook('  "He walked away"  ') == "He walked away"


# ---------------------------------------------------------------------------
# Copy hygiene
# ---------------------------------------------------------------------------

def test_strip_speaker_labels_rewrites_diarization_tags() -> None:
    """Internal Speaker A/B/C tags must never reach a viewer."""
    from app.ai.copy import strip_speaker_labels

    assert "Speaker C" not in strip_speaker_labels("Speaker C says the bill is bad.")
    assert strip_speaker_labels(
        "The clash between Speaker A and Speaker B escalated."
    ) == "The clash between two people escalated."


def test_strip_speaker_labels_leaves_real_words_alone() -> None:
    """The trailing word boundary stops 'Speaker Demands' becoming 'speakeremands'."""
    from app.ai.copy import strip_speaker_labels

    assert strip_speaker_labels("Speaker Demands attention now.") == (
        "Speaker Demands attention now."
    )
    assert strip_speaker_labels("A speaker demanded answers.") == (
        "A speaker demanded answers."
    )


def test_caption_schema_forbids_empty_paragraphs() -> None:
    """Empty strings satisfied the old schema, collapsing captions to one line."""
    from app.ai.prompts.clip_prompts import COPY_SCHEMA

    props = COPY_SCHEMA["properties"]
    for field in ("trigger", "escalation", "reaction", "debate"):
        assert props[field]["minLength"] >= 150, field
    assert COPY_SCHEMA["properties"]["hooks"]["minItems"] == 13


def test_confrontation_triggers_match() -> None:
    """These patterns were silently dead after heredoc damage ate the \b escapes."""
    from app.analysis.candidates import _TRIGGER_PATTERNS

    assert _TRIGGER_PATTERNS["confrontation"].search("Don't touch my things!")
    assert _TRIGGER_PATTERNS["challenge"].search("Answer the question, please.")
    assert not _TRIGGER_PATTERNS["confrontation"].search("We reviewed the budget.")


def test_caption_headline_never_a_hashtag() -> None:
    """A bare hashtag headline is unpacked into words, not printed raw."""
    from app.ai.copy import _format_caption

    plan = ClipPlan(id="x", index=1, start=0, end=20, transcript="some words here")
    out = _format_caption(
        {
            "headline": "#IllegalImmigrationControversy",
            "trigger": "t" * 40, "escalation": "e" * 40,
            "reaction": "r" * 40, "debate": "d" * 40,
            "question": "Who is right? #Debate",
            "location": "", "hashtags": ["#a", "#b", "#c", "#d", "#e"],
        },
        plan,
    )
    headline = out.splitlines()[0]
    assert "#" not in headline
    assert "ILLEGAL IMMIGRATION CONTROVERSY" in headline

    question = next(l for l in out.splitlines() if l.startswith("👇"))
    assert "#" not in question


def test_geotag_requires_the_city_to_be_spoken() -> None:
    """A state match must not smuggle in a city the clip never named."""
    from app.ai.copy import _location_is_supported

    assert _location_is_supported("Sacramento, CA", "we are here in Sacramento today")
    assert not _location_is_supported("Sacramento, CA", "here in California somewhere")
    assert not _location_is_supported("", "anything")


def test_clean_paragraph_removes_padding() -> None:
    """Length floors make small models pad with tags and emoji."""
    from app.ai.copy import _clean_paragraph

    assert _clean_paragraph("The clash escalated. 🤬 #Fraud #Viral 🤙") == (
        "The clash escalated."
    )
