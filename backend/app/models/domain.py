"""Domain objects shared across the pipeline.

These are plain pydantic models -- the vocabulary that analysis, AI and
rendering stages exchange. They are serialised to JSON artifacts on disk; the
database stores state and paths, not bulk analysis data.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

class MediaInfo(BaseModel):
    """Result of ffprobe on the source video."""

    duration: float
    width: int
    height: int
    fps: float
    video_codec: str = ""
    audio_codec: str = ""
    has_audio: bool = True
    audio_sample_rate: int = 0
    audio_channels: int = 0
    bitrate: int = 0
    size_bytes: int = 0
    container: str = ""

    @property
    def is_landscape(self) -> bool:
        return self.width >= self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

class Word(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0


class TranscriptSegment(BaseModel):
    """A sentence-ish unit as emitted by ASR, optionally speaker-attributed."""

    id: int
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    language: str = "unknown"
    language_probability: float = 0.0
    duration: float = 0.0
    segments: list[TranscriptSegment] = Field(default_factory=list)
    model: str = ""

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def words(self) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            out.extend(seg.words)
        return out

    def text_between(self, start: float, end: float) -> str:
        """Transcript text overlapping the interval.

        Segment-granular, so the result may include words spoken just outside
        the interval. Use `text_in_window` when the text must match exactly what
        a viewer of that window would hear.
        """
        parts = [s.text.strip() for s in self.segments if s.end > start and s.start < end]
        return " ".join(p for p in parts if p).strip()

    def segments_between(self, start: float, end: float) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.end > start and s.start < end]

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint falls inside the interval."""
        out: list[Word] = []
        for seg in self.segments:
            if seg.end <= start or seg.start >= end:
                continue
            for w in seg.words:
                if start <= (w.start + w.end) / 2.0 <= end:
                    out.append(w)
        return out

    def text_in_window(self, start: float, end: float) -> str:
        """Exactly the words audible within the window.

        This is what candidate evaluation and copy generation must see: judging a
        clip against text that includes a sentence starting two seconds before
        the cut makes the model reject good clips as 'starting mid-sentence',
        and lets copy cite words the viewer never hears.
        """
        words = self.words_between(start, end)
        if words:
            return " ".join(w.word.strip() for w in words if w.word.strip()).strip()
        # Fall back to segment text when word timings are unavailable.
        return self.text_between(start, end)


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------

class SpeakerTurn(BaseModel):
    speaker: str
    start: float
    end: float


class Diarization(BaseModel):
    speakers: list[str] = Field(default_factory=list)
    turns: list[SpeakerTurn] = Field(default_factory=list)
    method: str = "none"

    def speaker_at(self, t: float) -> Optional[str]:
        for turn in self.turns:
            if turn.start <= t < turn.end:
                return turn.speaker
        return None

    def speakers_between(self, start: float, end: float) -> list[str]:
        seen: list[str] = []
        for turn in self.turns:
            if turn.end > start and turn.start < end and turn.speaker not in seen:
                seen.append(turn.speaker)
        return seen


# ---------------------------------------------------------------------------
# Scenes / visual
# ---------------------------------------------------------------------------

class Scene(BaseModel):
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class FaceBox(BaseModel):
    """Face rectangle in absolute source-pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return self.w * self.h


class FrameAnalysis(BaseModel):
    """Cheap per-frame observations from OpenCV. No LLM involved."""

    t: float
    faces: list[FaceBox] = Field(default_factory=list)
    motion: float = 0.0
    brightness: float = 0.0
    sharpness: float = 0.0
    # Density of outlined-glyph pixels low in the frame: high while the source
    # is showing captions of its own.
    caption_score: float = 0.0


class VisualAnalysis(BaseModel):
    frames: list[FrameAnalysis] = Field(default_factory=list)
    sample_interval: float = 1.0
    frame_width: int = 0
    frame_height: int = 0
    # Fraction of frame height at which burned-in source subtitles begin, or
    # None when the source has none. The renderer crops this band away so the
    # 9:16 conversion does not slice someone else's captions in half.
    subtitle_band_top: Optional[float] = None
    # (start, end) windows in source time where the source is showing its own
    # captions. Burned-in captions usually come and go rather than running the
    # whole video, so this is a schedule, not a flag.
    subtitle_spans: list[tuple[float, float]] = Field(default_factory=list)

    def source_captions_at(self, t: float) -> bool:
        return any(start <= t <= end for start, end in self.subtitle_spans)

    def frames_between(self, start: float, end: float) -> list[FrameAnalysis]:
        return [f for f in self.frames if start <= f.t <= end]


class VisionObservation(BaseModel):
    """Higher-level visual reasoning from a multimodal model, per candidate."""

    candidate_id: str
    description: str = ""
    people_visible: int = 0
    on_screen_text: str = ""
    notable_events: list[str] = Field(default_factory=list)
    visual_interest: float = 0.5
    is_talking_head: bool = True


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class AudioEvent(BaseModel):
    kind: Literal["laughter", "applause", "cheer", "silence", "emphasis", "overlap"]
    start: float
    end: float
    strength: float = 0.5


class AudioAnalysis(BaseModel):
    sample_rate: int = 16000
    duration: float = 0.0
    hop: float = 0.05
    rms_db: list[float] = Field(default_factory=list)
    events: list[AudioEvent] = Field(default_factory=list)
    global_median_db: float = -30.0

    def events_between(self, start: float, end: float) -> list[AudioEvent]:
        return [e for e in self.events if e.end > start and e.start < end]

    def energy_between(self, start: float, end: float) -> list[float]:
        if not self.rms_db:
            return []
        i0 = max(0, int(start / self.hop))
        i1 = min(len(self.rms_db), int(end / self.hop))
        return self.rms_db[i0:i1]


# ---------------------------------------------------------------------------
# Candidates and clips
# ---------------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    content_clarity: float = 0.0
    standalone_completeness: float = 0.0
    narrative_structure: float = 0.0
    visual_interest: float = 0.0
    audio_dynamics: float = 0.0
    emotional_reaction: float = 0.0
    information_density: float = 0.0
    opening_strength: float = 0.0
    ending_payoff: float = 0.0

    def weighted_total(self, weights: dict[str, float]) -> float:
        mine = self.model_dump()
        return sum(mine.get(k, 0.0) * w for k, w in weights.items())


class Candidate(BaseModel):
    """A proposed moment, before LLM evaluation."""

    id: str
    start: float
    end: float
    text: str = ""
    context_before: str = ""
    context_after: str = ""
    speakers: list[str] = Field(default_factory=list)
    scene_count: int = 1
    audio_events: list[str] = Field(default_factory=list)
    breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    heuristic_score: float = 0.0
    trigger: str = "generic"
    # Attention signals: how hard the first seconds hit, and how much conflict
    # the window carries. Both drive selection for short-form.
    opening_punch: float = 0.0
    conflict_intensity: float = 0.0
    conflict_note: str = ""
    vision: Optional[VisionObservation] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ContextDependency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LLMEvaluation(BaseModel):
    """Strict schema the LLM must satisfy. Validated before anything is cut."""

    candidate_id: str
    recommended: bool
    start: float
    end: float
    reason: str = ""
    context_dependency: ContextDependency = ContextDependency.MEDIUM
    quality_score: float = 0.0
    topic: str = ""
    speakers: list[str] = Field(default_factory=list)
    complete_thought: bool = True
    needs_expansion: bool = False


class RenderStatus(str, Enum):
    PENDING = "pending"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"



class VideoContext(BaseModel):
    """What the source video is, described in the third person.

    Derived once from the whole transcript and then used to describe every
    clip, which is what lets a clip's description name the bill or the person
    the clip itself only calls "it" or "he".
    """

    setting: str = ""
    participants: str = ""
    subject: str = ""
    summary: str = ""
    key_terms: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.summary or self.subject)

    def as_prompt_block(self) -> str:
        """The form handed to the model when describing a clip."""
        parts = [
            f"Setting: {self.setting}" if self.setting else "",
            f"Participants: {self.participants}" if self.participants else "",
            f"Subject: {self.subject}" if self.subject else "",
            f"What happens: {self.summary}" if self.summary else "",
            f"Recurring terms: {', '.join(self.key_terms)}" if self.key_terms else "",
        ]
        return "\n".join(p for p in parts if p)

    def as_text(self) -> str:
        """The form written to disk and shown in the interface."""
        lines = []
        if self.setting:
            lines.append(self.setting)
        if self.participants:
            lines.append(f"Participants: {self.participants}")
        if self.subject:
            lines.append(self.subject)
        if self.summary:
            lines.append(self.summary)
        if self.key_terms:
            lines.append(f"Recurring terms: {', '.join(self.key_terms)}")
        return "\n\n".join(lines)


class ClipPlan(BaseModel):
    """A validated, deduplicated clip ready for rendering."""

    id: str
    index: int
    start: float
    end: float
    topic: str = ""
    transcript: str = ""
    speakers: list[str] = Field(default_factory=list)
    score: float = 0.0
    reason: str = ""
    context_dependency: ContextDependency = ContextDependency.LOW
    breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    source_candidate_id: str = ""
    analysis_notes: str = ""
    # Third-person description of this clip, written against the video's own
    # context so it can name what the clip only alludes to.
    context: str = ""
    standalone: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def name(self) -> str:
        return f"Clip_{self.index:02d}"


class CropKeyframe(BaseModel):
    """One point on the smart-reframe crop path, in source pixels."""

    t: float
    x: int
    y: int
    w: int
    h: int


class PipelineWarning(BaseModel):
    """Surfaced to the UI so degraded stages are visible rather than silent."""

    stage: str
    message: str
    severity: Literal["info", "warning"] = "warning"
    details: dict[str, Any] = Field(default_factory=dict)
