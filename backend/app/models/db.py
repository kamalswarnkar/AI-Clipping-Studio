"""SQLAlchemy models and session management.

The database holds *state and pointers*, not bulk data: transcripts, scene lists
and frame analyses are written to JSON artifacts on disk and referenced by path.
That keeps rows small and makes artifacts inspectable without a DB client.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from ..config import get_settings
from ..jobs.states import JobStatus, JobType, ProjectStatus
from .domain import RenderStatus


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class JSONMixin:
    """Helpers for the JSON-encoded TEXT columns used throughout."""

    @staticmethod
    def _load(raw: Optional[str], default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)


class Project(Base, JSONMixin):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), default="")
    source_filename: Mapped[str] = mapped_column(String(512), default="")
    source_path: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default=ProjectStatus.CREATED.value)

    media_info_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    settings_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    warnings_json: Mapped[Optional[str]] = mapped_column(Text, default=None)

    # Third-person description of the whole video, derived from the transcript
    # and used to describe every clip cut from it.
    global_context: Mapped[str] = mapped_column(Text, default="")

    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_stage: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    error_remedy: Mapped[Optional[str]] = mapped_column(Text, default=None)

    created_at: Mapped[dt.datetime] = mapped_column(default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_now, onupdate=_now)

    jobs: Mapped[list["Job"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Job.position"
    )
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Clip.index"
    )

    # --- JSON accessors -----------------------------------------------------
    @property
    def media_info(self) -> dict[str, Any]:
        return self._load(self.media_info_json, {})

    @media_info.setter
    def media_info(self, value: dict[str, Any]) -> None:
        self.media_info_json = self._dump(value)

    @property
    def settings(self) -> dict[str, Any]:
        return self._load(self.settings_json, {})

    @settings.setter
    def settings(self, value: dict[str, Any]) -> None:
        self.settings_json = self._dump(value)

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return self._load(self.warnings_json, [])

    @warnings.setter
    def warnings(self, value: list[dict[str, Any]]) -> None:
        self.warnings_json = self._dump(value)

    def add_warning(self, warning: dict[str, Any]) -> None:
        current = self.warnings
        current.append(warning)
        self.warnings = current


class Job(Base, JSONMixin):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(24), default=JobStatus.PENDING.value)
    position: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[Optional[str]] = mapped_column(Text, default=None)
    attempts: Mapped[int] = mapped_column(Integer, default=0)

    started_at: Mapped[Optional[dt.datetime]] = mapped_column(default=None)
    finished_at: Mapped[Optional[dt.datetime]] = mapped_column(default=None)

    project: Mapped[Project] = relationship(back_populates="jobs")

    @property
    def job_type(self) -> JobType:
        return JobType(self.type)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class Candidate(Base, JSONMixin):
    """Persisted candidate pool -- kept so the LLM stage is independently retryable."""

    __tablename__ = "candidates"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    candidate_key: Mapped[str] = mapped_column(String(48), default="")
    start: Mapped[float] = mapped_column(Float, default=0.0)
    end: Mapped[float] = mapped_column(Float, default=0.0)
    heuristic_score: Mapped[float] = mapped_column(Float, default=0.0)
    llm_score: Mapped[Optional[float]] = mapped_column(Float, default=None)
    recommended: Mapped[Optional[bool]] = mapped_column(default=None)
    rejected_reason: Mapped[Optional[str]] = mapped_column(Text, default=None)
    payload_json: Mapped[Optional[str]] = mapped_column(Text, default=None)

    @property
    def payload(self) -> dict[str, Any]:
        return self._load(self.payload_json, {})

    @payload.setter
    def payload(self, value: dict[str, Any]) -> None:
        self.payload_json = self._dump(value)


class Clip(Base, JSONMixin):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column(Integer, default=1)
    start: Mapped[float] = mapped_column(Float, default=0.0)
    end: Mapped[float] = mapped_column(Float, default=0.0)
    duration: Mapped[float] = mapped_column(Float, default=0.0)

    topic: Mapped[str] = mapped_column(Text, default="")
    transcript: Mapped[str] = mapped_column(Text, default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    analysis_notes: Mapped[str] = mapped_column(Text, default="")
    context_dependency: Mapped[str] = mapped_column(String(16), default="low")
    # What this clip is, written against the video's own context.
    context: Mapped[str] = mapped_column(Text, default="")
    standalone: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)

    speakers_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    breakdown_json: Mapped[Optional[str]] = mapped_column(Text, default=None)

    render_status: Mapped[str] = mapped_column(
        String(24), default=RenderStatus.PENDING.value
    )
    render_error: Mapped[Optional[str]] = mapped_column(Text, default=None)
    video_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    subtitle_path: Mapped[Optional[str]] = mapped_column(Text, default=None)

    created_at: Mapped[dt.datetime] = mapped_column(default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(default=_now, onupdate=_now)

    project: Mapped[Project] = relationship(back_populates="clips")

    @property
    def name(self) -> str:
        return f"Clip_{self.index:02d}"

    @property
    def speakers(self) -> list[str]:
        return self._load(self.speakers_json, [])

    @speakers.setter
    def speakers(self, value: list[str]) -> None:
        self.speakers_json = self._dump(value)

    @property
    def breakdown(self) -> dict[str, float]:
        return self._load(self.breakdown_json, {})

    @breakdown.setter
    def breakdown(self, value: dict[str, float]) -> None:
        self.breakdown_json = self._dump(value)


class Feedback(Base, JSONMixin):
    """Signals for a future ranking model (Architecture.md section 30).

    Recording these now costs almost nothing and means a later ranking model has
    training data available without re-instrumenting the app.
    """

    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    clip_id: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    event: Mapped[str] = mapped_column(String(48))
    payload_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(default=_now)

    @property
    def payload(self) -> dict[str, Any]:
        return self._load(self.payload_json, {})

    @payload.setter
    def payload(self, value: dict[str, Any]) -> None:
        self.payload_json = self._dump(value)


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

_settings = get_settings()

engine: Engine = create_engine(
    _settings.database_url,
    echo=False,
    future=True,
    # The job pool touches the DB from worker threads.
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
    """WAL keeps readers (the polling UI) from blocking on writers (workers)."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _add_missing_columns() -> None:
    """Bring an existing database up to the current model.

    `create_all` creates missing tables but never alters existing ones, so a
    column added to a model is invisible to a database created before it -- and
    every query then fails with "no such column". This is a local app the user
    starts by double-clicking; there is no migration step for them to run, so
    the schema catches itself up on startup.

    Only additive, and only for columns with a default. Anything more than that
    belongs in a real migration.
    """
    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            existing = {
                row[1]
                for row in connection.exec_driver_sql(
                    f"PRAGMA table_info({table.name})"
                ).fetchall()
            }
            if not existing:  # table did not exist; create_all just made it
                continue
            for column in table.columns:
                if column.name in existing:
                    continue
                default = column.default.arg if column.default is not None else None
                if isinstance(default, bool):
                    literal = "1" if default else "0"
                elif isinstance(default, (int, float)):
                    literal = str(default)
                elif isinstance(default, str):
                    escaped = default.replace("'", "''")
                    literal = f"'{escaped}'"
                else:
                    literal = "NULL"
                sql = (
                    f"ALTER TABLE {table.name} "
                    f"ADD COLUMN {column.name} {column.type.compile(engine.dialect)} "
                    f"DEFAULT {literal}"
                )
                connection.exec_driver_sql(sql)
                log.info("Added column %s.%s", table.name, column.name)


def init_db() -> None:
    _settings.ensure_dirs()
    Base.metadata.create_all(engine)
    _add_missing_columns()


def get_session() -> Session:
    return SessionLocal()
