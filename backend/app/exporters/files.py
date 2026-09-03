"""Export writers: the Clip_NN folder structure, Hooks.txt, Caption.txt, Info.txt.

The exact layout is a product requirement (Architecture.md section 20), so the
formatting here is intentionally literal rather than clever. Every clip folder is
self-contained: video, hooks, caption and metadata together.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from ..models.domain import ClipCopy, ClipPlan
from ..services.storage import safe_stem

log = logging.getLogger(__name__)


def format_timecode(seconds: float) -> str:
    """HH:MM:SS for human-readable metadata."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_hooks_txt(copy: ClipCopy) -> str:
    """Render Hooks.txt in the required layout."""
    lines: list[str] = ["🏆 BEST HOOK", ""]
    lines.append(copy.best_hook or "(none generated)")
    lines.extend(["", ""])

    for i, hook in enumerate(copy.hooks, start=1):
        lines.append(f"{i}. {hook.category}")
        lines.append(hook.text)
        lines.extend(["", ""])

    lines.append("RANKING")
    lines.append("")
    for hook in copy.ranked_hooks:
        lines.append(f"{hook.rank}. {hook.text}")

    return "\n".join(lines).rstrip() + "\n"


def render_caption_txt(copy: ClipCopy) -> str:
    """Caption.txt holds only the caption, ready to copy and paste."""
    return (copy.caption or "").strip() + "\n"


def render_info_txt(
    plan: ClipPlan,
    *,
    source_filename: str,
    crop_strategy: str = "",
    generated_by: str = "",
    warnings: Optional[list[str]] = None,
) -> str:
    """Render Info.txt with the source metadata and analysis trail."""
    speakers = ", ".join(plan.speakers) if plan.speakers else "Not identified"

    sections = [
        ("Source", source_filename),
        ("Clip", plan.name),
        ("Start", format_timecode(plan.start)),
        ("End", format_timecode(plan.end)),
        ("Duration", f"{plan.duration:.0f} seconds"),
        ("Topic", plan.topic or "Not specified"),
        ("Speakers", speakers),
        ("Quality score", f"{plan.score:.2f}"),
        ("Context dependency", plan.context_dependency.value),
        ("Transcript", plan.transcript or "(no speech detected)"),
        ("Analysis", plan.reason or "Not recorded"),
    ]

    if plan.analysis_notes:
        sections.append(("Validation", plan.analysis_notes))
    if crop_strategy:
        sections.append(("Framing", crop_strategy))

    breakdown = plan.breakdown.model_dump()
    if any(breakdown.values()):
        detail = "\n".join(
            f"  {key.replace('_', ' ')}: {value:.2f}"
            for key, value in breakdown.items()
        )
        sections.append(("Score breakdown", "\n" + detail))

    if warnings:
        sections.append(("Notes", "\n" + "\n".join(f"  - {w}" for w in warnings)))

    sections.append(
        ("Generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M") + (f" using {generated_by}" if generated_by else ""))
    )

    out: list[str] = []
    for label, value in sections:
        out.append(f"{label}:")
        out.append(str(value))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def write_clip_folder(
    *,
    root: Path,
    plan: ClipPlan,
    copy: ClipCopy,
    video_path: Optional[Path],
    source_filename: str,
    crop_strategy: str = "",
    warnings: Optional[list[str]] = None,
) -> Path:
    """Create one self-contained Clip_NN folder."""
    folder = root / plan.name
    folder.mkdir(parents=True, exist_ok=True)

    if video_path and video_path.exists():
        shutil.copy2(video_path, folder / f"{plan.name}.mp4")
    else:
        log.warning("No rendered video for %s; folder will lack the MP4", plan.name)

    (folder / "Hooks.txt").write_text(render_hooks_txt(copy), encoding="utf-8")
    (folder / "Caption.txt").write_text(render_caption_txt(copy), encoding="utf-8")
    (folder / "Info.txt").write_text(
        render_info_txt(
            plan,
            source_filename=source_filename,
            crop_strategy=crop_strategy,
            generated_by=copy.generated_by,
            warnings=warnings,
        ),
        encoding="utf-8",
    )
    return folder


def build_export_tree(
    *,
    export_root: Path,
    source_filename: str,
    items: Iterable[tuple[ClipPlan, ClipCopy, Optional[Path], str]],
    project_warnings: Optional[list[str]] = None,
) -> Path:
    """Build <SourceVideoName>/Clip_NN/... under the export root."""
    root = export_root / safe_stem(source_filename)
    root.mkdir(parents=True, exist_ok=True)

    count = 0
    for plan, copy, video_path, crop_strategy in items:
        write_clip_folder(
            root=root,
            plan=plan,
            copy=copy,
            video_path=video_path,
            source_filename=source_filename,
            crop_strategy=crop_strategy,
            warnings=project_warnings,
        )
        count += 1

    log.info("Export tree built at %s with %d clips", root, count)
    return root


def zip_directory(directory: Path, zip_path: Path) -> Path:
    """Zip a directory, preserving the folder structure inside the archive."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                # arcname keeps <SourceVideoName>/Clip_01/... inside the zip.
                archive.write(path, path.relative_to(directory.parent))

    return zip_path
