"""Project lifecycle: create, upload, process, status."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from ...jobs.pipeline import create_jobs, run_pipeline
from ...jobs.queue import queue
from ...jobs.states import JOB_LABELS, JobStatus, JobType, ProjectStatus
from ...models.db import Clip, Project, get_session
from ...models.domain import RenderStatus
from ...models.schemas import (
    ClipResponse,
    JobResponse,
    ProcessResponse,
    ProjectResponse,
    ProjectSettings,
    ProjectStatusResponse,
    WarningResponse,
)
from ...services.storage import ProjectStorage, sanitize_filename
from ...video.ffmpeg import SUPPORTED_EXTENSIONS

log = logging.getLogger(__name__)
router = APIRouter(prefix="/projects", tags=["projects"])

# Read uploads in chunks so a multi-GB file never lands in memory.
CHUNK_SIZE = 4 * 1024 * 1024


def _get_project_or_404(session, project_id: str) -> Project:  # noqa: ANN001
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(name: str = Form(default="")) -> ProjectResponse:
    with get_session() as session:
        project = Project(name=name.strip()[:200], status=ProjectStatus.CREATED.value)
        project.settings = ProjectSettings().model_dump()
        session.add(project)
        session.commit()
        ProjectStorage(project.id).ensure()
        return ProjectResponse.from_row(project)


@router.post("/{project_id}/upload", response_model=ProjectResponse)
async def upload_video(
    project_id: str,
    file: UploadFile = File(...),
    clip_count: int = Form(default=15),
    min_duration: float = Form(default=20.0),
    max_duration: float = Form(default=60.0),
    vertical: bool = Form(default=False),
    captions: bool = Form(default=True),
    smart_reframe: bool = Form(default=True),
    vocabulary: str = Form(default=""),
) -> ProjectResponse:
    """Store the source video and the processing options."""
    filename = sanitize_filename(file.filename or "video.mp4")
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"'{extension or 'unknown'}' is not a supported format. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    try:
        settings = ProjectSettings(
            clip_count=clip_count,
            min_duration=min_duration,
            max_duration=max_duration,
            vertical=vertical,
            captions=captions,
            smart_reframe=smart_reframe,
            vocabulary=vocabulary,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with get_session() as session:
        project = _get_project_or_404(session, project_id)
        storage = ProjectStorage(project_id).ensure()
        destination = storage.source_dir / filename

        try:
            with destination.open("wb") as out:
                while chunk := await file.read(CHUNK_SIZE):
                    out.write(chunk)
        except OSError as exc:
            raise HTTPException(
                status_code=507,
                detail=f"Could not save the upload: {exc}. Check available disk space.",
            ) from exc
        finally:
            await file.close()

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        project.source_filename = filename
        project.source_path = str(destination)
        project.status = ProjectStatus.UPLOADED.value
        project.settings = settings.model_dump()
        if not project.name:
            project.name = Path(filename).stem
        session.commit()
        return ProjectResponse.from_row(project)


@router.post("/{project_id}/process", response_model=ProcessResponse)
def process_project(project_id: str) -> ProcessResponse:
    """Start the pipeline. Returns immediately; poll /status for progress."""
    with get_session() as session:
        project = _get_project_or_404(session, project_id)
        if not project.source_path or not Path(project.source_path).exists():
            raise HTTPException(
                status_code=400, detail="Upload a video before starting processing."
            )

    if queue.is_running(project_id):
        return ProcessResponse(started=False, message="This project is already processing.")

    create_jobs(project_id)
    started = queue.submit(project_id, run_pipeline)
    return ProcessResponse(
        started=started,
        message="Processing started." if started else "Already queued.",
    )


@router.post("/{project_id}/cancel", response_model=ProcessResponse)
def cancel_project(project_id: str) -> ProcessResponse:
    cancelled = queue.cancel(project_id)
    return ProcessResponse(
        started=False,
        message="Cancelling after the current step." if cancelled else "Nothing running.",
    )


@router.get("/{project_id}/status", response_model=ProjectStatusResponse)
def project_status(project_id: str) -> ProjectStatusResponse:
    """Real pipeline state -- never a simulated progress bar."""
    with get_session() as session:
        project = _get_project_or_404(session, project_id)
        jobs = sorted(project.jobs, key=lambda j: j.position)

        # Overall progress is the mean of per-job progress, so it advances
        # smoothly within a long stage rather than jumping between stages.
        if jobs:
            total = sum(
                1.0
                if j.status in (JobStatus.COMPLETED.value, JobStatus.SKIPPED.value)
                else (j.progress if j.status == JobStatus.RUNNING.value else 0.0)
                for j in jobs
            )
            overall = total / len(jobs)
        else:
            overall = 0.0

        running = next((j for j in jobs if j.status == JobStatus.RUNNING.value), None)
        current_stage = ""
        if running is not None:
            try:
                current_stage = JOB_LABELS[JobType(running.type)]
            except (KeyError, ValueError):
                current_stage = running.type

        clips = project.clips
        ready = sum(
            1 for c in clips if c.render_status == RenderStatus.COMPLETED.value
        )

        error = None
        if project.error_message:
            error = {
                "message": project.error_message,
                "stage": project.error_stage or "",
                "remedy": project.error_remedy or "",
            }

        return ProjectStatusResponse(
            id=project.id,
            status=project.status,
            overall_progress=round(min(1.0, overall), 4),
            current_stage=current_stage,
            jobs=[JobResponse.from_row(j) for j in jobs],
            warnings=[WarningResponse(**w) for w in project.warnings],
            error=error,
            clips_ready=ready,
            clips_total=len(clips),
        )


@router.get("", response_model=list[ProjectResponse])
def list_projects(limit: int = 50) -> list[ProjectResponse]:
    with get_session() as session:
        rows = (
            session.query(Project)
            .order_by(Project.created_at.desc())
            .limit(max(1, min(limit, 200)))
            .all()
        )
        return [ProjectResponse.from_row(r) for r in rows]


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str) -> ProjectResponse:
    with get_session() as session:
        return ProjectResponse.from_row(_get_project_or_404(session, project_id))


@router.get("/{project_id}/clips", response_model=list[ClipResponse])
def list_clips(project_id: str) -> list[ClipResponse]:
    with get_session() as session:
        _get_project_or_404(session, project_id)
        rows = (
            session.query(Clip)
            .filter(Clip.project_id == project_id)
            .order_by(Clip.index)
            .all()
        )
        return [ClipResponse.from_row(r) for r in rows]


@router.delete("/{project_id}", status_code=204, response_class=Response)
def delete_project(project_id: str) -> Response:
    if queue.is_running(project_id):
        raise HTTPException(
            status_code=409, detail="Cancel processing before deleting this project."
        )
    with get_session() as session:
        project = _get_project_or_404(session, project_id)
        session.delete(project)
        session.commit()

    try:
        ProjectStorage(project_id).delete()
    except Exception as exc:  # noqa: BLE001 - DB row is already gone
        log.warning("Could not remove files for %s: %s", project_id, exc)

    return Response(status_code=204)
