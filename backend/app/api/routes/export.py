"""Export: build the Clip_NN folder tree and package it as a ZIP."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...exporters.files import build_export_tree, zip_directory
from ...models.db import Clip, Project, get_session
from ...models.domain import (
    ClipPlan,
    ContextDependency,
    RenderStatus,
    ScoreBreakdown,
)
from ...models.schemas import ExportRequest, ExportResponse
from ...services.storage import ProjectStorage, safe_stem

log = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}/export", tags=["export"])


def _build(project_id: str, clip_ids: list[str] | None) -> tuple[Path, int, str]:
    """Assemble the export tree. Returns (zip_path, clip_count, source_filename)."""
    with get_session() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")

        query = session.query(Clip).filter(Clip.project_id == project_id)
        if clip_ids:
            query = query.filter(Clip.id.in_(clip_ids))
        clips = query.order_by(Clip.index).all()

        if not clips:
            raise HTTPException(status_code=404, detail="No clips to export.")

        source_filename = project.source_filename or "video.mp4"
        video_context = project.global_context or ""

        items = []
        for clip in clips:
            plan = ClipPlan(
                id=clip.id,
                index=clip.index,
                start=clip.start,
                end=clip.end,
                topic=clip.topic or "",
                transcript=clip.transcript or "",
                speakers=clip.speakers,
                score=clip.score,
                reason=clip.reason or "",
                analysis_notes=clip.analysis_notes or "",
                context_dependency=ContextDependency(clip.context_dependency or "low"),
                breakdown=ScoreBreakdown(**(clip.breakdown or {})),
                context=clip.context or "",
                standalone=bool(clip.standalone),
            )
            video = Path(clip.video_path) if clip.video_path else None
            if video is not None and not video.exists():
                video = None
            items.append((plan, video))

    storage = ProjectStorage(project_id)
    root = build_export_tree(
        export_root=storage.export_dir,
        source_filename=source_filename,
        items=items,
        video_context=video_context,
    )
    zip_path = zip_directory(root, storage.zip_path(source_filename))
    return zip_path, len(items), source_filename


@router.post("", response_model=ExportResponse)
def prepare_export(project_id: str, request: ExportRequest) -> ExportResponse:
    """Build the export package and report what it contains."""
    with get_session() as session:
        unrendered = (
            session.query(Clip)
            .filter(
                Clip.project_id == project_id,
                Clip.render_status != RenderStatus.COMPLETED.value,
            )
            .count()
        )

    zip_path, count, source = _build(project_id, request.clip_ids)
    message = f"Packaged {count} clip(s)."
    if unrendered and not request.clip_ids:
        message += (
            f" {unrendered} clip(s) have no video and were exported with text files only."
        )

    return ExportResponse(
        ready=zip_path.exists(),
        filename=f"{safe_stem(source)}.zip",
        clip_count=count,
        message=message,
    )


@router.get("")
def download_export(project_id: str):  # noqa: ANN201
    """Download the ZIP, building it first if it does not exist yet."""
    zip_path, _, source = _build(project_id, None)
    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="Export package could not be built.")

    filename = f"{safe_stem(source)}.zip"
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
