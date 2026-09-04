"""Pipeline orchestration.

Runs the stages in order, records real progress per job, and keeps optional
stages non-fatal: losing diarization or vision degrades quality and raises a
visible warning, but never kills a run that can still produce clips.

Artifacts are written to disk as each stage completes, so a retry resumes from
the last good state instead of redoing transcription.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from ..ai import copy as copy_gen
from ..ai import evaluation
from ..ai.registry import get_providers
from ..ai.vision import analyzer as vision_analyzer
from ..analysis import candidates as candidate_gen
from ..analysis import scenes as scene_analysis
from ..analysis import selection
from ..analysis.audio_analysis import analyze_audio
from ..analysis.boundaries import expand_for_context, refine_boundaries
from ..analysis.visual import analyze_visuals, detect_subtitle_band
from ..config import get_settings
from ..models.db import Candidate as CandidateRow
from ..models.db import Clip, Job, Project, get_session
from ..models.domain import (
    AudioAnalysis,
    Candidate,
    ClipCopy,
    ClipPlan,
    Diarization,
    MediaInfo,
    PipelineWarning,
    RenderStatus,
    Scene,
    Transcript,
    VisualAnalysis,
)
from ..services.storage import ProjectStorage
from ..video import ffmpeg
from ..video.renderer import RenderOptions, render_clip
from .queue import JobReporter, queue
from .states import (
    JOB_LABELS,
    OPTIONAL_JOBS,
    PIPELINE_ORDER,
    JobStatus,
    JobType,
    PipelineError,
    ProjectStatus,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_jobs(project_id: str) -> None:
    """Create (or reset) the job rows for a full run."""
    with get_session() as session:
        project = session.get(Project, project_id)
        if project is None:
            return
        for job in list(project.jobs):
            session.delete(job)
        session.flush()
        for position, job_type in enumerate(PIPELINE_ORDER):
            session.add(
                Job(
                    project_id=project_id,
                    type=job_type.value,
                    status=JobStatus.PENDING.value,
                    position=position,
                    message=JOB_LABELS[job_type],
                )
            )
        session.commit()


def _job_id(project_id: str, job_type: JobType) -> Optional[str]:
    with get_session() as session:
        job = (
            session.query(Job)
            .filter(Job.project_id == project_id, Job.type == job_type.value)
            .one_or_none()
        )
        return job.id if job else None


def _add_warning(project_id: str, warning: PipelineWarning) -> None:
    with get_session() as session:
        project = session.get(Project, project_id)
        if project is not None:
            project.add_warning(warning.model_dump())
            session.commit()


class _Context:
    """Everything the stages share for one run."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.settings = get_settings()
        self.storage = ProjectStorage(project_id).ensure()

        with get_session() as session:
            project = session.get(Project, project_id)
            if project is None:
                raise PipelineError("Project not found.", stage="ingest")
            self.source = Path(project.source_path)
            self.source_filename = project.source_filename
            self.options = project.settings or {}

        self.media: Optional[MediaInfo] = None
        self.transcript: Optional[Transcript] = None
        self.diarization: Optional[Diarization] = None
        self.scenes: list[Scene] = []
        self.audio: Optional[AudioAnalysis] = None
        self.visual: Optional[VisualAnalysis] = None
        self.candidates: list[Candidate] = []
        self.plans: list[ClipPlan] = []
        self.copies: dict[str, ClipCopy] = {}
        self.crop_strategies: dict[str, str] = {}
        self.selection_notes: list[str] = []

    # --- user-facing options ------------------------------------------------
    @property
    def clip_count(self) -> int:
        return int(self.options.get("clip_count", self.settings.clip_count_default))

    @property
    def min_duration(self) -> float:
        return float(self.options.get("min_duration", self.settings.clip_min_duration))

    @property
    def max_duration(self) -> float:
        return float(self.options.get("max_duration", self.settings.clip_max_duration))

    @property
    def render_options(self) -> RenderOptions:
        return RenderOptions.from_settings(
            {
                "vertical": self.options.get("vertical", True),
                "captions": self.options.get("captions", True),
                "smart_reframe": self.options.get("smart_reframe", True),
            }
        )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _stage_ingest(ctx: _Context, report: JobReporter) -> None:
    report.start("Reading video metadata")
    ctx.media = ffmpeg.probe(ctx.source)

    if not ctx.media.has_audio:
        raise PipelineError(
            "This video has no audio track, so there is no speech to analyse.",
            stage="ingest",
            remedy="Upload a video that contains spoken audio.",
        )
    if ctx.media.duration < ctx.min_duration * 2:
        raise PipelineError(
            f"This video is only {ctx.media.duration:.0f}s long, too short to cut "
            f"{ctx.min_duration:.0f}s clips from.",
            stage="ingest",
            remedy="Upload a longer video, or lower the minimum clip duration.",
        )

    with get_session() as session:
        project = session.get(Project, ctx.project_id)
        if project is not None:
            project.media_info = ctx.media.model_dump()
            session.commit()

    ctx.storage.write_json("media", ctx.media)
    report.progress(0.5, "Building analysis proxy")

    # A low-res proxy makes every later frame operation dramatically cheaper.
    try:
        ffmpeg.make_proxy(ctx.source, ctx.storage.proxy_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Proxy generation failed: %s", exc)
        _add_warning(
            ctx.project_id,
            PipelineWarning(
                stage="ingest",
                message="Could not build an analysis proxy; visual analysis will be slower.",
            ),
        )

    report.complete(
        f"{ctx.media.width}x{ctx.media.height}, {ctx.media.duration / 60:.1f} min"
    )


def _stage_extract_audio(ctx: _Context, report: JobReporter) -> None:
    report.start("Extracting audio")
    ffmpeg.extract_audio(ctx.source, ctx.storage.audio_path)
    report.complete("Audio ready")


def _stage_transcribe(ctx: _Context, report: JobReporter) -> None:
    report.start("Loading speech model")
    providers = get_providers()

    available, reason = providers.transcription.is_available()
    if not available:
        raise PipelineError(
            f"Speech recognition is unavailable: {reason}",
            stage="transcribe",
            remedy="Check TRANSCRIPTION_PROVIDER in .env, or install faster-whisper.",
        )

    cached = ctx.storage.read_json("transcript")
    if cached:
        ctx.transcript = Transcript.model_validate(cached)
        report.complete(f"{len(ctx.transcript.segments)} segments (cached)")
        return

    ctx.transcript = providers.transcription.transcribe(
        ctx.storage.audio_path,
        progress=lambda f, m: report.progress(f, m),
    )
    ctx.storage.write_json("transcript", ctx.transcript)
    report.complete(
        f"{len(ctx.transcript.segments)} segments, language {ctx.transcript.language}"
    )


def _stage_diarize(ctx: _Context, report: JobReporter) -> None:
    report.start("Separating speakers")
    providers = get_providers()

    available, reason = providers.diarization.is_available()
    if not available:
        report.skip(reason)
        return

    ctx.diarization = providers.diarization.diarize(
        ctx.storage.audio_path, transcript=ctx.transcript
    )
    ctx.storage.write_json("diarization", ctx.diarization)
    report.complete(
        f"{len(ctx.diarization.speakers)} speaker(s), {len(ctx.diarization.turns)} turns"
    )


def _stage_scenes(ctx: _Context, report: JobReporter) -> None:
    report.start("Detecting scenes")
    source = (
        ctx.storage.proxy_path if ctx.storage.proxy_path.exists() else ctx.source
    )
    ctx.scenes = scene_analysis.detect_scenes(
        source,
        duration=ctx.media.duration if ctx.media else 0.0,
        progress=lambda f, m: report.progress(f, m),
    )
    ctx.storage.write_json("scenes", [s.model_dump() for s in ctx.scenes])
    report.complete(f"{len(ctx.scenes)} scenes")


def _stage_audio_analysis(ctx: _Context, report: JobReporter) -> None:
    report.start("Analyzing audio")
    ctx.audio = analyze_audio(
        ctx.storage.audio_path, progress=lambda f, m: report.progress(f, m)
    )
    ctx.storage.write_json("audio_analysis", ctx.audio)
    reactions = sum(
        1 for e in ctx.audio.events if e.kind in ("laughter", "applause", "cheer")
    )
    report.complete(f"{len(ctx.audio.events)} events, {reactions} reactions")


def _stage_visual_analysis(ctx: _Context, report: JobReporter) -> None:
    report.start("Analyzing video")
    if not ctx.storage.proxy_path.exists():
        report.skip("No proxy available")
        return

    ctx.visual = analyze_visuals(
        ctx.storage.proxy_path,
        source_width=ctx.media.width if ctx.media else 0,
        source_height=ctx.media.height if ctx.media else 0,
        progress=lambda f, m: report.progress(f, m),
    )
    # Burned-in captions in the source must be found before rendering: the
    # vertical crop would otherwise cut them in half and the app would burn its
    # own captions on top of the remains.
    if ctx.settings.remove_source_subtitles:
        try:
            ctx.visual.subtitle_band_top = detect_subtitle_band(ctx.source)
        except Exception as exc:  # noqa: BLE001 - purely an enhancement
            log.warning("Subtitle band detection failed: %s", exc)

    ctx.storage.write_json("visual", ctx.visual)
    with_faces = sum(1 for f in ctx.visual.frames if f.faces)
    band = ctx.visual.subtitle_band_top
    extra = f", source captions from {band * 100:.0f}%" if band else ""
    report.complete(f"{len(ctx.visual.frames)} frames, faces in {with_faces}{extra}")


def _stage_candidates(ctx: _Context, report: JobReporter) -> None:
    report.start("Finding moments")
    if ctx.transcript is None:
        raise PipelineError("No transcript available.", stage="candidates")

    ctx.candidates = candidate_gen.generate_candidates(
        transcript=ctx.transcript,
        diarization=ctx.diarization,
        audio=ctx.audio,
        visual=ctx.visual,
        scene_list=ctx.scenes,
        weights=ctx.settings.scoring_weights().normalised(),
        min_duration=ctx.min_duration,
        opening_window=ctx.settings.opening_window_seconds,
        opening_weight=ctx.settings.opening_weight,
        conflict_weight=ctx.settings.conflict_weight,
        max_duration=ctx.max_duration,
        total_duration=ctx.media.duration if ctx.media else 0.0,
        pool_max=ctx.settings.candidate_pool_max,
    )

    if not ctx.candidates:
        raise PipelineError(
            "No usable moments were found. The video may contain very little "
            "continuous speech.",
            stage="candidates",
            remedy="Try a video with more spoken content, or lower the minimum duration.",
        )

    with get_session() as session:
        session.query(CandidateRow).filter(
            CandidateRow.project_id == ctx.project_id
        ).delete()
        for candidate in ctx.candidates:
            row = CandidateRow(
                project_id=ctx.project_id,
                candidate_key=candidate.id,
                start=candidate.start,
                end=candidate.end,
                heuristic_score=candidate.heuristic_score,
            )
            row.payload = candidate.model_dump()
            session.add(row)
        session.commit()

    report.complete(f"{len(ctx.candidates)} candidates")


def _stage_llm_evaluate(ctx: _Context, report: JobReporter) -> None:
    report.start("Evaluating moments")
    providers = get_providers()

    # --- bounded vision pass first -----------------------------------------
    # Runs before the text LLM so the two models are not swapped in and out of
    # VRAM repeatedly on machines where both cannot co-reside.
    if ctx.settings.vision_enabled:
        vision_ok, vision_reason = providers.vision.is_available()
        if vision_ok:
            try:
                observations = vision_analyzer.analyze_candidates(
                    ctx.candidates,
                    source=ctx.source,
                    frames_dir=ctx.storage.frames_dir,
                    provider=providers.vision,
                    max_candidates=ctx.settings.vision_max_candidates,
                    frames_per_candidate=ctx.settings.vision_frames_per_candidate,
                    progress=lambda f, m: report.progress(f * 0.35, f"visual: {m}"),
                )
                log.info("Vision analysed %d candidates", len(observations))
            except Exception as exc:  # noqa: BLE001 - optional enrichment
                log.warning("Vision stage failed: %s", exc)
                _add_warning(
                    ctx.project_id,
                    PipelineWarning(
                        stage="vision",
                        message=f"Visual analysis was skipped: {exc}",
                    ),
                )
        else:
            _add_warning(
                ctx.project_id,
                PipelineWarning(stage="vision", message=vision_reason, severity="info"),
            )

    # --- text evaluation ----------------------------------------------------
    llm_ok, llm_reason = providers.llm.is_available()
    if not llm_ok:
        _add_warning(
            ctx.project_id,
            PipelineWarning(
                stage="llm",
                message=(
                    f"{llm_reason} Clips were selected using signal analysis only, "
                    "which is less accurate at judging meaning."
                ),
            ),
        )
        report.skip(llm_reason)
        return

    shortlist = ctx.candidates[: ctx.settings.candidate_llm_max]
    evaluations = evaluation.evaluate_candidates(
        shortlist,
        llm=providers.llm,
        min_duration=ctx.min_duration,
        max_duration=ctx.max_duration,
        total_duration=ctx.media.duration if ctx.media else 0.0,
        batch_size=ctx.settings.llm_eval_batch,
        progress=lambda f, m: report.progress(0.35 + f * 0.65, m),
    )

    ctx.storage.write_json(
        "evaluations", [e.model_dump(mode="json") for e in evaluations]
    )

    with get_session() as session:
        rows = {
            r.candidate_key: r
            for r in session.query(CandidateRow)
            .filter(CandidateRow.project_id == ctx.project_id)
            .all()
        }
        for item in evaluations:
            row = rows.get(item.candidate_id)
            if row is not None:
                row.llm_score = item.quality_score
                row.recommended = item.recommended
        session.commit()

    ctx._evaluations = evaluations  # type: ignore[attr-defined]
    recommended = sum(1 for e in evaluations if e.recommended)
    report.complete(f"{len(evaluations)} evaluated, {recommended} recommended")


def _stage_validate(ctx: _Context, report: JobReporter) -> None:
    report.start("Validating context")
    providers = get_providers()
    evaluations = getattr(ctx, "_evaluations", [])

    result = selection.select_clips(
        candidates=ctx.candidates,
        evaluations=evaluations,
        requested=ctx.clip_count,
        iou_threshold=ctx.settings.dedupe_iou_threshold,
        text_threshold=ctx.settings.dedupe_text_similarity,
        llm_available=bool(evaluations),
    )
    ctx.selection_notes = result.notes

    llm_ok, _ = providers.llm.is_available()
    plans: list[ClipPlan] = []

    # Validation is one independent LLM call per clip, so run them concurrently
    # and apply the deterministic edits afterwards in order.
    verdicts: dict[str, dict] = {}
    if llm_ok and ctx.transcript is not None and result.clips:
        workers = max(1, min(ctx.settings.llm_parallel, len(result.clips)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="validate") as pool:
            futures = {
                pool.submit(
                    evaluation.validate_context,
                    start=p.start,
                    end=p.end,
                    transcript=ctx.transcript,
                    llm=providers.llm,
                ): p.id
                for p in result.clips
            }
            for future in as_completed(futures):
                clip_id = futures[future]
                try:
                    verdicts[clip_id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Validation failed for %s: %s", clip_id, exc)

    for i, plan in enumerate(result.clips):
        queue.raise_if_cancelled(ctx.project_id)

        # Context validation: does this stand alone, and did trimming change
        # what the speaker meant?
        verdict = verdicts.get(plan.id)
        if verdict is not None:

            if verdict["verdict"] == "reject":
                log.info("Dropping %s after validation: %s", plan.id, verdict["note"])
                continue

            if verdict["verdict"] == "expand" or verdict["start_shift"] < 0:
                new_start, new_end = expand_for_context(
                    plan.start,
                    plan.end,
                    transcript=ctx.transcript,
                    max_duration=ctx.max_duration,
                    total_duration=ctx.media.duration if ctx.media else 0.0,
                    lookback=abs(verdict["start_shift"]) or 12.0,
                )
                plan.start, plan.end = new_start, new_end

            plan.analysis_notes = verdict["note"]
            if verdict["missing_context"]:
                plan.analysis_notes += f" Missing context: {verdict['missing_context']}"

        # Deterministic guard: never open a clip on logistics or small talk,
        # whatever the model scored it.
        floor: Optional[float] = None
        if ctx.transcript is not None:
            new_start, new_end, was_trimmed = candidate_gen.trim_filler_opening(
                plan.start,
                plan.end,
                transcript=ctx.transcript,
                min_duration=ctx.min_duration,
                max_duration=ctx.max_duration,
                total_duration=ctx.media.duration if ctx.media else 0.0,
            )
            if was_trimmed:
                log.info(
                    "Trimmed filler opening from %s (%.1f-%.1f -> %.1f-%.1f)",
                    plan.id,
                    plan.start,
                    plan.end,
                    new_start,
                    new_end,
                )
                plan.start, plan.end = new_start, new_end
                floor = new_start

        # Final deterministic snap to word/sentence boundaries.
        if ctx.transcript is not None:
            refined = refine_boundaries(
                plan.start,
                plan.end,
                floor_start=floor,
                transcript=ctx.transcript,
                audio=ctx.audio,
                scenes=ctx.scenes,
                min_duration=ctx.min_duration,
                max_duration=ctx.max_duration,
                total_duration=ctx.media.duration if ctx.media else 0.0,
            )
            plan.start, plan.end = refined.start, refined.end
            plan.transcript = ctx.transcript.text_in_window(plan.start, plan.end)
            if ctx.diarization:
                plan.speakers = ctx.diarization.speakers_between(plan.start, plan.end)

        plans.append(plan)
        report.progress((i + 1) / max(1, len(result.clips)), f"{len(plans)} validated")

    # Renumber after any drops so clips read 01..N with no gaps.
    plans.sort(key=lambda p: p.start)
    for index, plan in enumerate(plans, start=1):
        plan.index = index

    ctx.plans = plans

    if not plans:
        raise PipelineError(
            "No moments passed context validation. Every candidate needed "
            "surrounding context to make sense on its own.",
            stage="validate",
            remedy="Try increasing the maximum clip duration so clips can include more setup.",
        )

    # Persist clip rows now so the UI can show them while rendering runs.
    with get_session() as session:
        session.query(Clip).filter(Clip.project_id == ctx.project_id).delete()
        for plan in plans:
            clip = Clip(
                project_id=ctx.project_id,
                index=plan.index,
                start=plan.start,
                end=plan.end,
                duration=plan.duration,
                topic=plan.topic,
                transcript=plan.transcript,
                reason=plan.reason,
                analysis_notes=plan.analysis_notes,
                context_dependency=plan.context_dependency.value,
                score=plan.score,
                render_status=RenderStatus.PENDING.value,
            )
            clip.speakers = plan.speakers
            clip.breakdown = plan.breakdown.model_dump()
            session.add(clip)
        session.commit()

    for note in result.notes:
        _add_warning(
            ctx.project_id,
            PipelineWarning(stage="selection", message=note, severity="info"),
        )

    report.complete(f"{len(plans)} clips selected")


def _stage_generate_copy(ctx: _Context, report: JobReporter) -> None:
    report.start("Generating hooks and captions")
    providers = get_providers()

    llm_ok, reason = providers.llm.is_available()
    if not llm_ok:
        # Still produce transcript-derived copy so no clip ships empty.
        for plan in ctx.plans:
            ctx.copies[plan.id] = copy_gen.generate_copy(
                plan, llm=providers.llm, vision=None
            )
        report.skip(reason)
        return

    vision_by_candidate = {
        c.id: c.vision for c in ctx.candidates if c.vision is not None
    }
    notes_by_candidate = {c.id: c.conflict_note for c in ctx.candidates}

    def one(plan: ClipPlan) -> tuple[str, ClipCopy]:
        opening = (
            ctx.transcript.text_in_window(plan.start, plan.start + 3.0)
            if ctx.transcript
            else ""
        )
        return plan.id, copy_gen.generate_copy(
            plan,
            llm=providers.llm,
            vision=vision_by_candidate.get(plan.source_candidate_id),
            conflict_note=notes_by_candidate.get(plan.source_candidate_id, ""),
            opening_line=opening,
        )

    # Clips are independent, so generate copy concurrently.
    done = 0
    workers = max(1, min(ctx.settings.llm_parallel, len(ctx.plans)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="copy") as pool:
        futures = [pool.submit(one, plan) for plan in ctx.plans]
        for future in as_completed(futures):
            queue.raise_if_cancelled(ctx.project_id)
            try:
                plan_id, copy = future.result()
                ctx.copies[plan_id] = copy
            except Exception as exc:  # noqa: BLE001 - one clip must not kill the stage
                log.warning("Copy generation failed for a clip: %s", exc)
            done += 1
            report.progress(done / len(ctx.plans), f"{done}/{len(ctx.plans)} clips")

    with get_session() as session:
        clips = {
            c.index: c
            for c in session.query(Clip).filter(Clip.project_id == ctx.project_id).all()
        }
        for plan in ctx.plans:
            clip = clips.get(plan.index)
            copy = ctx.copies.get(plan.id)
            if clip is None or copy is None:
                continue
            clip.hooks = [h.model_dump() for h in copy.hooks]
            clip.best_hook = copy.best_hook
            clip.caption = copy.caption
            clip.copy_generated_by = copy.generated_by
        session.commit()

    report.complete(f"Copy for {len(ctx.copies)} clips")


def _render_one(
    ctx: _Context, plan: ClipPlan
) -> tuple[ClipPlan, Optional[Exception], Optional[str]]:
    """Render a single clip. Returns (plan, error, crop_strategy)."""
    try:
        result = render_clip(
            plan=plan,
            source=ctx.source,
            media=ctx.media,  # type: ignore[arg-type]
            transcript=ctx.transcript,
            visual=ctx.visual,
            diarization=ctx.diarization,
            output_path=ctx.storage.clip_video(plan.index),
            subtitle_path=ctx.storage.clip_subtitles(plan.index),
            thumbnail_path=ctx.storage.clip_thumbnail(plan.index),
            options=ctx.render_options,
        )
        return plan, None, result.crop_strategy
    except Exception as exc:  # noqa: BLE001 - reported per clip, not fatal
        return plan, exc, None


def _stage_render(ctx: _Context, report: JobReporter) -> None:
    report.start("Rendering clips")
    if not ctx.plans:
        report.skip("No clips to render")
        return

    with get_session() as session:
        for clip in session.query(Clip).filter(Clip.project_id == ctx.project_id).all():
            clip.render_status = RenderStatus.PENDING.value
        session.commit()

    completed = 0
    failures: list[str] = []
    workers = max(1, ctx.settings.render_workers)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render") as pool:
        futures = {pool.submit(_render_one, ctx, plan): plan for plan in ctx.plans}

        for future in as_completed(futures):
            plan, error, strategy = future.result()
            completed += 1

            with get_session() as session:
                clip = (
                    session.query(Clip)
                    .filter(Clip.project_id == ctx.project_id, Clip.index == plan.index)
                    .one_or_none()
                )
                if clip is not None:
                    if error is None:
                        clip.render_status = RenderStatus.COMPLETED.value
                        clip.video_path = str(ctx.storage.clip_video(plan.index))
                        clip.thumbnail_path = str(ctx.storage.clip_thumbnail(plan.index))
                        subs = ctx.storage.clip_subtitles(plan.index)
                        clip.subtitle_path = str(subs) if subs.exists() else None
                    else:
                        clip.render_status = RenderStatus.FAILED.value
                        clip.render_error = str(error)[:1000]
                    session.commit()

            if error is None:
                ctx.crop_strategies[plan.id] = strategy or ""
            else:
                log.warning("Render failed for %s: %s", plan.name, error)
                failures.append(plan.name)

            report.progress(completed / len(ctx.plans), f"{completed}/{len(ctx.plans)}")

    if failures and len(failures) == len(ctx.plans):
        raise PipelineError(
            "Every clip failed to render.",
            stage="render",
            remedy="Check that the source video is readable and that disk space is available.",
        )

    if failures:
        _add_warning(
            ctx.project_id,
            PipelineWarning(
                stage="render",
                message=(
                    f"{len(failures)} clip(s) failed to render: {', '.join(failures)}. "
                    "They can be retried individually."
                ),
            ),
        )

    report.complete(f"{completed - len(failures)}/{len(ctx.plans)} rendered")


STAGE_FUNCTIONS: dict[JobType, Callable[[_Context, JobReporter], None]] = {
    JobType.INGEST: _stage_ingest,
    JobType.EXTRACT_AUDIO: _stage_extract_audio,
    JobType.TRANSCRIBE: _stage_transcribe,
    JobType.DIARIZE: _stage_diarize,
    JobType.ANALYZE_SCENES: _stage_scenes,
    JobType.ANALYZE_AUDIO: _stage_audio_analysis,
    JobType.ANALYZE_VISUALS: _stage_visual_analysis,
    JobType.GENERATE_CANDIDATES: _stage_candidates,
    JobType.LLM_EVALUATE: _stage_llm_evaluate,
    JobType.VALIDATE: _stage_validate,
    JobType.GENERATE_COPY: _stage_generate_copy,
    JobType.RENDER: _stage_render,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(project_id: str) -> None:
    """Execute the full pipeline for one project."""
    log.info("Starting pipeline for project %s", project_id)

    with get_session() as session:
        project = session.get(Project, project_id)
        if project is None:
            return
        project.status = ProjectStatus.PROCESSING.value
        project.error_message = None
        project.error_stage = None
        project.error_remedy = None
        project.warnings = []
        session.commit()

    ctx = _Context(project_id)

    for job_type in PIPELINE_ORDER:
        queue.raise_if_cancelled(project_id)

        job_id = _job_id(project_id, job_type)
        if job_id is None:
            continue
        report = JobReporter(project_id, job_id)
        stage = STAGE_FUNCTIONS[job_type]

        # Transient failures (a model still loading, a busy GPU, a locked file)
        # are common enough that one retry saves a whole re-run. A PipelineError
        # is a considered verdict about the input, so it is never retried.
        attempts_allowed = 1 + max(0, ctx.settings.job_max_retries)
        last_error: Exception | None = None

        for attempt in range(attempts_allowed):
            try:
                stage(ctx, report)
                last_error = None
                break
            except PipelineError:
                report.fail("")
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                queue.raise_if_cancelled(project_id)
                if attempt + 1 < attempts_allowed:
                    log.warning(
                        "Stage %s failed (attempt %d/%d): %s -- retrying",
                        job_type.value,
                        attempt + 1,
                        attempts_allowed,
                        exc,
                    )
                    report.progress(0.0, f"Retrying after error: {exc}"[:120])

        if last_error is not None:
            if job_type in OPTIONAL_JOBS:
                # Degrade rather than abort: the run can still produce clips.
                log.warning("Optional stage %s failed: %s", job_type.value, last_error)
                report.skip(f"Skipped after error: {last_error}")
                _add_warning(
                    project_id,
                    PipelineWarning(
                        stage=job_type.value.lower(),
                        message=(
                            f"{JOB_LABELS[job_type]} failed and was skipped: {last_error}"
                        ),
                    ),
                )
                continue
            report.fail(str(last_error))
            raise PipelineError(
                f"{JOB_LABELS[job_type]} failed: {last_error}",
                stage=job_type.value.lower(),
            ) from last_error

    with get_session() as session:
        project = session.get(Project, project_id)
        if project is not None:
            project.status = ProjectStatus.COMPLETED.value
            session.commit()

    log.info("Pipeline complete for project %s", project_id)
