"""Clip-level operations: media streaming, copy regeneration, re-render, adjust."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from ...ai import copy as copy_gen
from ...ai.registry import get_providers
from ...analysis.boundaries import refine_boundaries
from ...models.db import Clip, Feedback, Project, get_session
from ...models.domain import (
    ClipPlan,
    ContextDependency,
    MediaInfo,
    RenderStatus,
    Transcript,
    VisualAnalysis,
)
from ...models.schemas import AdjustClipRequest, ClipResponse
from ...services.storage import ProjectStorage
from ...video import ffmpeg
from ...video.renderer import RenderOptions, render_clip

log = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}/clips", tags=["clips"])


def _get_clip(session, project_id: str, clip_id: str) -> Clip:  # noqa: ANN001
    clip = (
        session.query(Clip)
        .filter(Clip.project_id == project_id, Clip.id == clip_id)
        .one_or_none()
    )
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return clip


def _record_feedback(project_id: str, clip_id: str, event: str, payload: dict) -> None:
    """Store an interaction signal for future ranking work (section 30)."""
    try:
        with get_session() as session:
            row = Feedback(project_id=project_id, clip_id=clip_id, event=event)
            row.payload = payload
            session.add(row)
            session.commit()
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a request
        log.debug("Could not record feedback: %s", exc)


def _plan_from_clip(clip: Clip) -> ClipPlan:
    return ClipPlan(
        id=clip.id,
        index=clip.index,
        start=clip.start,
        end=clip.end,
        topic=clip.topic or "",
        transcript=clip.transcript or "",
        speakers=clip.speakers,
        score=clip.score,
        reason=clip.reason or "",
        context_dependency=ContextDependency(clip.context_dependency or "low"),
    )


@router.get("/{clip_id}", response_model=ClipResponse)
def get_clip(project_id: str, clip_id: str) -> ClipResponse:
    with get_session() as session:
        return ClipResponse.from_row(_get_clip(session, project_id, clip_id))


@router.get("/{clip_id}/video")
def stream_video(project_id: str, clip_id: str):  # noqa: ANN201
    """Serve the rendered MP4. FileResponse handles range requests, so the
    player can seek without downloading the whole clip."""
    with get_session() as session:
        clip = _get_clip(session, project_id, clip_id)
        path = Path(clip.video_path) if clip.video_path else None
        name = clip.name

    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="This clip has not been rendered yet.")

    # Confine the served path to the project directory.
    try:
        ProjectStorage(project_id).resolve_within(path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid clip path.") from None

    return FileResponse(path, media_type="video/mp4", filename=f"{name}.mp4")


@router.get("/{clip_id}/thumbnail")
def get_thumbnail(project_id: str, clip_id: str):  # noqa: ANN201
    with get_session() as session:
        clip = _get_clip(session, project_id, clip_id)
        path = Path(clip.thumbnail_path) if clip.thumbnail_path else None

    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="No thumbnail available.")

    try:
        ProjectStorage(project_id).resolve_within(path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid thumbnail path.") from None

    return FileResponse(path, media_type="image/jpeg")


@router.get("/{clip_id}/download")
def download_clip(project_id: str, clip_id: str):  # noqa: ANN201
    with get_session() as session:
        clip = _get_clip(session, project_id, clip_id)
        path = Path(clip.video_path) if clip.video_path else None
        name = clip.name

    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="This clip has not been rendered yet.")

    _record_feedback(project_id, clip_id, "clip_downloaded", {})
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{name}.mp4",
        headers={"Content-Disposition": f'attachment; filename="{name}.mp4"'},
    )


@router.post("/{clip_id}/regenerate-copy", response_model=ClipResponse)
def regenerate_copy(project_id: str, clip_id: str) -> ClipResponse:
    """Re-run hook and caption generation for one clip."""
    providers = get_providers()
    available, reason = providers.llm.is_available()
    if not available:
        raise HTTPException(status_code=503, detail=reason)

    with get_session() as session:
        clip = _get_clip(session, project_id, clip_id)
        plan = _plan_from_clip(clip)
        previous = clip.best_hook

    copy = copy_gen.generate_copy(plan, llm=providers.llm)

    with get_session() as session:
        clip = _get_clip(session, project_id, clip_id)
        clip.hooks = [h.model_dump() for h in copy.hooks]
        clip.best_hook = copy.best_hook
        clip.caption = copy.caption
        clip.copy_generated_by = copy.generated_by
        session.commit()
        result = ClipResponse.from_row(clip)

    _record_feedback(
        project_id, clip_id, "copy_regenerated", {"previous_best_hook": previous}
    )
    return result


@router.post("/{clip_id}/adjust", response_model=ClipResponse)
def adjust_clip(
    project_id: str, clip_id: str, request: AdjustClipRequest
) -> ClipResponse:
    """Move a clip's boundaries and re-render it.

    The requested times are snapped to word boundaries, so a manual adjustment
    still cannot land mid-word.
    """
    storage = ProjectStorage(project_id)

    with get_session() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        clip = _get_clip(session, project_id, clip_id)
        previous = (clip.start, clip.end)
        settings = project.settings or {}
        source = Path(project.source_path)
        media_info = project.media_info

    if not source.exists():
        raise HTTPException(status_code=410, detail="The source video is no longer available.")

    transcript_raw = storage.read_json("transcript")
    transcript = Transcript.model_validate(transcript_raw) if transcript_raw else None

    start, end = request.start, request.end
    duration = float(media_info.get("duration", 0.0))
    if duration and end > duration:
        raise HTTPException(
            status_code=422,
            detail=f"End time {end:.1f}s is beyond the video length ({duration:.1f}s).",
        )

    if transcript is not None:
        refined = refine_boundaries(
            start,
            end,
            transcript=transcript,
            min_duration=float(settings.get("min_duration", 10.0)),
            max_duration=float(settings.get("max_duration", 60.0)),
            total_duration=duration,
            search_window=1.2,  # respect the user's intent; only tidy the edges
        )
        start, end = refined.start, refined.end

    with get_session() as session:
        clip = _get_clip(session, project_id, clip_id)
        clip.start, clip.end, clip.duration = start, end, end - start
        if transcript is not None:
            clip.transcript = transcript.text_in_window(start, end)
        clip.render_status = RenderStatus.RENDERING.value
        clip.render_error = None
        session.commit()
        plan = _plan_from_clip(clip)
        index = clip.index

    visual_raw = storage.read_json("visual")
    visual = VisualAnalysis.model_validate(visual_raw) if visual_raw else None

    # Reuse the stored probe result; fall back to probing if it is missing.
    media = MediaInfo.model_validate(media_info) if media_info else ffmpeg.probe(source)

    options = RenderOptions.from_settings(
        {
            "vertical": settings.get("vertical", True),
            "captions": settings.get("captions", True),
            "smart_reframe": settings.get("smart_reframe", True),
        }
    )

    try:
        render_clip(
            plan=plan,
            source=source,
            media=media,
            transcript=transcript,
            visual=visual,
            diarization=None,
            output_path=storage.clip_video(index),
            subtitle_path=storage.clip_subtitles(index),
            thumbnail_path=storage.clip_thumbnail(index),
            options=options,
        )
        status, error = RenderStatus.COMPLETED, None
    except Exception as exc:  # noqa: BLE001
        log.warning("Re-render failed for clip %s: %s", clip_id, exc)
        status, error = RenderStatus.FAILED, str(exc)

    with get_session() as session:
        clip = _get_clip(session, project_id, clip_id)
        clip.render_status = status.value
        clip.render_error = error[:1000] if error else None
        if status is RenderStatus.COMPLETED:
            clip.video_path = str(storage.clip_video(index))
            clip.thumbnail_path = str(storage.clip_thumbnail(index))
        session.commit()
        result = ClipResponse.from_row(clip)

    _record_feedback(
        project_id,
        clip_id,
        "boundary_adjusted",
        {"from": previous, "to": (start, end)},
    )

    if status is RenderStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Re-render failed: {error}")
    return result


@router.post("/{clip_id}/rerender", response_model=ClipResponse)
def rerender_clip(project_id: str, clip_id: str) -> ClipResponse:
    """Retry a failed render without restarting the project."""
    with get_session() as session:
        clip = _get_clip(session, project_id, clip_id)
        return adjust_clip(
            project_id, clip_id, AdjustClipRequest(start=clip.start, end=clip.end)
        )


@router.post("/{clip_id}/feedback", status_code=204, response_class=Response)
def submit_feedback(
    project_id: str, clip_id: str, event: str, value: str = ""
) -> Response:
    """Record hook selection / rejection signals from the UI."""
    allowed = {
        "hook_selected",
        "hook_rejected",
        "caption_edited",
        "clip_selected",
        "clip_rejected",
    }
    if event not in allowed:
        raise HTTPException(status_code=422, detail=f"Unknown event '{event}'.")
    _record_feedback(project_id, clip_id, event, {"value": value[:500]})
    return Response(status_code=204)
