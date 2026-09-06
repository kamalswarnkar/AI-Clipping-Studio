"""Job and project state machine.

Every long-running stage is a Job row so the UI can report *real* progress and so
a failure can be retried in isolation -- a failed render of Clip 09 must not
force the whole project to restart.
"""

from __future__ import annotations

from enum import Enum


class JobType(str, Enum):
    """Ordered pipeline stages (Architecture.md section 27)."""

    INGEST = "INGEST"
    EXTRACT_AUDIO = "EXTRACT_AUDIO"
    TRANSCRIBE = "TRANSCRIBE"
    DIARIZE = "DIARIZE"
    SUMMARIZE = "SUMMARIZE"
    REFINE_TRANSCRIPT = "REFINE_TRANSCRIPT"
    ANALYZE_SCENES = "ANALYZE_SCENES"
    ANALYZE_AUDIO = "ANALYZE_AUDIO"
    ANALYZE_VISUALS = "ANALYZE_VISUALS"
    GENERATE_CANDIDATES = "GENERATE_CANDIDATES"
    LLM_EVALUATE = "LLM_EVALUATE"
    VALIDATE = "VALIDATE"
    DESCRIBE_CLIPS = "DESCRIBE_CLIPS"
    RENDER = "RENDER"
    WRITE_COPY = "WRITE_COPY"
    PACKAGE_EXPORT = "PACKAGE_EXPORT"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.SKIPPED,
            JobStatus.CANCELLED,
        )


class ProjectStatus(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# The canonical order the pipeline runs in. PACKAGE_EXPORT is on-demand, so it
# is not part of the automatic run.
PIPELINE_ORDER: tuple[JobType, ...] = (
    JobType.INGEST,
    JobType.EXTRACT_AUDIO,
    JobType.TRANSCRIBE,
    JobType.DIARIZE,
    JobType.SUMMARIZE,
    JobType.REFINE_TRANSCRIPT,
    JobType.ANALYZE_SCENES,
    JobType.ANALYZE_AUDIO,
    JobType.ANALYZE_VISUALS,
    JobType.GENERATE_CANDIDATES,
    JobType.LLM_EVALUATE,
    JobType.VALIDATE,
    JobType.DESCRIBE_CLIPS,
    # After rendering: the clip file is what the copy is written from.
    JobType.RENDER,
    JobType.WRITE_COPY,
)

# Human-facing labels for the processing screen.
JOB_LABELS: dict[JobType, str] = {
    JobType.INGEST: "Inspecting video",
    JobType.EXTRACT_AUDIO: "Extracting audio",
    JobType.TRANSCRIBE: "Transcribing",
    JobType.DIARIZE: "Identifying speakers",
    JobType.SUMMARIZE: "Understanding the video",
    JobType.REFINE_TRANSCRIPT: "Correcting names",
    JobType.ANALYZE_SCENES: "Detecting scenes",
    JobType.ANALYZE_AUDIO: "Analyzing audio",
    JobType.ANALYZE_VISUALS: "Analyzing video",
    JobType.GENERATE_CANDIDATES: "Finding moments",
    JobType.LLM_EVALUATE: "Evaluating moments",
    JobType.VALIDATE: "Validating context",
    JobType.DESCRIBE_CLIPS: "Describing clips",
    JobType.RENDER: "Rendering clips",
    JobType.WRITE_COPY: "Writing hooks and captions",
    JobType.PACKAGE_EXPORT: "Packaging export",
}

# Stages the pipeline can continue without. A failure here degrades quality but
# does not kill the run; the reason is surfaced as a warning in the UI.
OPTIONAL_JOBS: frozenset[JobType] = frozenset(
    {
        JobType.DIARIZE,
        JobType.ANALYZE_SCENES,
        JobType.ANALYZE_AUDIO,
        JobType.ANALYZE_VISUALS,
        # Descriptions are reference notes, not the product. If the model is
        # unavailable the clips are still correct, just undescribed.
        JobType.SUMMARIZE,
        JobType.REFINE_TRANSCRIPT,
        JobType.DESCRIBE_CLIPS,
        JobType.WRITE_COPY,
    }
)


class PipelineError(Exception):
    """An error with a message intended for the end user.

    `stage` and `remedy` let the UI show something actionable instead of a bare
    'Something went wrong'.
    """

    def __init__(self, message: str, *, stage: str = "", remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.remedy = remedy

    def to_dict(self) -> dict[str, str]:
        return {"message": self.message, "stage": self.stage, "remedy": self.remedy}
