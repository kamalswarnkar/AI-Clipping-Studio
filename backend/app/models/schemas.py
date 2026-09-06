"""API request/response schemas.

Kept separate from the ORM and from the internal domain models so the wire format
can stay stable while either side evolves.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from ..jobs.states import JOB_LABELS, JobType


class ProjectSettings(BaseModel):
    """User-facing options from the upload screen."""

    clip_count: int = Field(default=15, ge=1, le=50)
    min_duration: float = Field(default=20.0, ge=3.0, le=180.0)
    max_duration: float = Field(default=60.0, ge=5.0, le=300.0)
    vertical: bool = False
    captions: bool = True
    smart_reframe: bool = True
    # Names and terms Whisper would not otherwise get right.
    vocabulary: str = Field(default="", max_length=1000)

    @field_validator("max_duration")
    @classmethod
    def _max_above_min(cls, v: float, info) -> float:
        minimum = info.data.get("min_duration", 20.0)
        if v <= minimum:
            raise ValueError("Maximum duration must be greater than minimum duration.")
        return v


class CreateProjectRequest(BaseModel):
    name: str = Field(default="", max_length=200)


class JobResponse(BaseModel):
    id: str
    type: str
    label: str
    status: str
    progress: float
    message: str
    error: Optional[str] = None
    attempts: int = 0
    duration_seconds: Optional[float] = None

    @classmethod
    def from_row(cls, row) -> "JobResponse":  # noqa: ANN001
        try:
            label = JOB_LABELS[JobType(row.type)]
        except (KeyError, ValueError):
            label = row.type
        return cls(
            id=row.id,
            type=row.type,
            label=label,
            status=row.status,
            progress=round(row.progress, 4),
            message=row.message or "",
            error=row.error,
            attempts=row.attempts,
            duration_seconds=row.duration_seconds,
        )


class ClipResponse(BaseModel):
    id: str
    index: int
    name: str
    start: float
    end: float
    duration: float
    topic: str = ""
    transcript: str = ""
    reason: str = ""
    analysis_notes: str = ""
    context_dependency: str = "low"
    context: str = ""
    standalone: bool = False
    best_hook: str = ""
    hooks: list[dict[str, Any]] = Field(default_factory=list)
    caption: str = ""
    score: float = 0.0
    speakers: list[str] = Field(default_factory=list)
    render_status: str = "pending"
    render_error: Optional[str] = None
    has_video: bool = False
    has_thumbnail: bool = False
    breakdown: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_row(cls, row) -> "ClipResponse":  # noqa: ANN001
        from pathlib import Path

        return cls(
            id=row.id,
            index=row.index,
            name=row.name,
            start=round(row.start, 3),
            end=round(row.end, 3),
            duration=round(row.duration, 3),
            topic=row.topic or "",
            transcript=row.transcript or "",
            reason=row.reason or "",
            analysis_notes=row.analysis_notes or "",
            context_dependency=row.context_dependency or "low",
            context=row.context or "",
            standalone=bool(row.standalone),
            best_hook=row.best_hook or "",
            hooks=row.hooks,
            caption=row.caption or "",
            score=round(row.score, 4),
            speakers=row.speakers,
            render_status=row.render_status,
            render_error=row.render_error,
            has_video=bool(row.video_path and Path(row.video_path).exists()),
            has_thumbnail=bool(row.thumbnail_path and Path(row.thumbnail_path).exists()),
            breakdown=row.breakdown,
        )


class WarningResponse(BaseModel):
    stage: str
    message: str
    severity: str = "warning"


class ProjectResponse(BaseModel):
    id: str
    name: str
    source_filename: str
    status: str
    created_at: dt.datetime
    updated_at: dt.datetime
    settings: ProjectSettings
    media_info: dict[str, Any] = Field(default_factory=dict)
    # Third-person description of the source video, derived from its transcript.
    global_context: str = ""
    warnings: list[WarningResponse] = Field(default_factory=list)
    error: Optional[dict[str, str]] = None
    clip_count: int = 0

    @classmethod
    def from_row(cls, row) -> "ProjectResponse":  # noqa: ANN001
        error = None
        if row.error_message:
            error = {
                "message": row.error_message,
                "stage": row.error_stage or "",
                "remedy": row.error_remedy or "",
            }
        return cls(
            id=row.id,
            name=row.name,
            source_filename=row.source_filename,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
            settings=ProjectSettings(**(row.settings or {})),
            media_info=row.media_info,
            global_context=row.global_context or "",
            warnings=[WarningResponse(**w) for w in row.warnings],
            error=error,
            clip_count=len(row.clips),
        )


class ProjectStatusResponse(BaseModel):
    """Everything the processing screen needs in one poll."""

    id: str
    status: str
    overall_progress: float
    current_stage: str = ""
    jobs: list[JobResponse] = Field(default_factory=list)
    warnings: list[WarningResponse] = Field(default_factory=list)
    error: Optional[dict[str, str]] = None
    clips_ready: int = 0
    clips_total: int = 0


class ProcessResponse(BaseModel):
    started: bool
    message: str = ""


class AdjustClipRequest(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start", 0.0)
        if v <= start:
            raise ValueError("End must be after start.")
        return v


class ExportRequest(BaseModel):
    clip_ids: Optional[list[str]] = None


class ExportResponse(BaseModel):
    ready: bool
    filename: str = ""
    clip_count: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    ffmpeg: dict[str, Any]
    providers: dict[str, dict[str, Any]]
    settings: dict[str, Any]
