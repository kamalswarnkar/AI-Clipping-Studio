"""End-to-end pipeline run against the real backend, without the HTTP layer.

Creates a project, points it at a video, runs every stage, then reports what was
produced and builds the export tree.

Usage:  python scripts/run_e2e.py [path/to/video.mp4] [--clips N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

# Hooks and captions contain emoji. The Windows console defaults to cp1252 and
# raises UnicodeEncodeError on them, which would kill the run at the report step.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from app.exporters.files import build_export_tree, zip_directory  # noqa: E402
from app.jobs.pipeline import create_jobs, run_pipeline  # noqa: E402
from app.jobs.states import ProjectStatus  # noqa: E402
from app.main import configure_logging  # noqa: E402
from app.models.db import Project, get_session, init_db  # noqa: E402
from app.models.domain import (  # noqa: E402
    ClipCopy,
    ClipPlan,
    ContextDependency,
    Hook,
    ScoreBreakdown,
)
from app.models.schemas import ProjectSettings  # noqa: E402
from app.services.storage import ProjectStorage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video", nargs="?", default=str(REPO / "data" / "samples" / "interview.mp4")
    )
    parser.add_argument("--clips", type=int, default=15)
    parser.add_argument("--min", type=float, default=10.0)
    parser.add_argument("--max", type=float, default=60.0)
    args = parser.parse_args()

    configure_logging("INFO")
    init_db()

    source = Path(args.video).resolve()
    if not source.exists():
        print(f"No such video: {source}")
        return 1

    # --- create project and stage the source --------------------------------
    with get_session() as session:
        project = Project(
            name=source.stem,
            source_filename=source.name,
            status=ProjectStatus.UPLOADED.value,
        )
        project.settings = ProjectSettings(
            clip_count=args.clips, min_duration=args.min, max_duration=args.max
        ).model_dump()
        session.add(project)
        session.commit()
        project_id = project.id

    storage = ProjectStorage(project_id).ensure()
    staged = storage.source_dir / source.name
    if not staged.exists():
        import shutil

        shutil.copy2(source, staged)

    with get_session() as session:
        project = session.get(Project, project_id)
        project.source_path = str(staged)
        session.commit()

    print(f"\nProject {project_id}  source={source.name}\n" + "=" * 70)

    create_jobs(project_id)
    started = time.time()
    try:
        run_pipeline(project_id)
    except Exception as exc:  # noqa: BLE001
        print(f"\nPIPELINE FAILED: {exc}")

    elapsed = time.time() - started

    # --- report --------------------------------------------------------------
    with get_session() as session:
        project = session.get(Project, project_id)
        print("\n" + "=" * 70)
        print(f"STATUS: {project.status}   elapsed {elapsed / 60:.1f} min")
        if project.error_message:
            print(f"ERROR : {project.error_message}")
            print(f"REMEDY: {project.error_remedy}")

        print("\nJOBS")
        for job in sorted(project.jobs, key=lambda j: j.position):
            secs = job.duration_seconds
            timing = f"{secs:6.1f}s" if secs else "     -"
            print(f"  {job.status:10s} {timing}  {job.type:20s} {job.message[:44]}")

        if project.warnings:
            print("\nWARNINGS")
            for w in project.warnings:
                print(f"  [{w['severity']}] {w['stage']}: {w['message'][:100]}")

        clips = sorted(project.clips, key=lambda c: c.index)
        print(f"\nCLIPS: {len(clips)}")
        for clip in clips:
            size = ""
            if clip.video_path and Path(clip.video_path).exists():
                size = f"{Path(clip.video_path).stat().st_size / 1e6:.1f}MB"
            print(
                f"  {clip.name}  {clip.start:7.1f}-{clip.end:7.1f} "
                f"({clip.duration:5.1f}s) {clip.render_status:9s} {size:8s} "
                f"score={clip.score:.2f} | {clip.topic[:38]}"
            )
            if clip.best_hook:
                print(f"      BEST HOOK: {clip.best_hook}")
            if clip.render_error:
                print(f"      RENDER ERROR: {clip.render_error[:90]}")

        # --- export ----------------------------------------------------------
        if clips:
            items = []
            for clip in clips:
                plan = ClipPlan(
                    id=clip.id, index=clip.index, start=clip.start, end=clip.end,
                    topic=clip.topic or "", transcript=clip.transcript or "",
                    speakers=clip.speakers, score=clip.score, reason=clip.reason or "",
                    analysis_notes=clip.analysis_notes or "",
                    context_dependency=ContextDependency(clip.context_dependency or "low"),
                    breakdown=ScoreBreakdown(**(clip.breakdown or {})),
                )
                copy = ClipCopy(
                    hooks=[Hook(**h) for h in clip.hooks],
                    best_hook=clip.best_hook or "", caption=clip.caption or "",
                    generated_by=clip.copy_generated_by or "",
                )
                video = Path(clip.video_path) if clip.video_path else None
                items.append((plan, copy, video if video and video.exists() else None, ""))

            root = build_export_tree(
                export_root=storage.export_dir,
                source_filename=project.source_filename,
                items=items,
            )
            zip_path = zip_directory(root, storage.zip_path(project.source_filename))
            print(f"\nEXPORT: {root}")
            print(f"ZIP   : {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\nProject id: {project_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
