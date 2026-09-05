"""Export writers: the Clip_NN folder structure and Info.txt.

The exact layout is a product requirement (Architecture.md section 20), so the
formatting here is intentionally literal rather than clever. Every clip folder is
self-contained: the rendered video plus its transcript.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from ..models.domain import ClipPlan
from ..services.storage import safe_stem

log = logging.getLogger(__name__)


def format_timecode(seconds: float) -> str:
    """HH:MM:SS for human-readable metadata."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_info_txt(plan: ClipPlan) -> str:
    """Info.txt holds the clip transcript and nothing else."""
    return (plan.transcript or "(no speech detected)").strip() + "\n"


def write_clip_folder(
    *,
    root: Path,
    plan: ClipPlan,
    video_path: Optional[Path],
) -> Path:
    """Create one self-contained Clip_NN folder."""
    folder = root / plan.name
    folder.mkdir(parents=True, exist_ok=True)

    if video_path and video_path.exists():
        shutil.copy2(video_path, folder / f"{plan.name}.mp4")
    else:
        log.warning("No rendered video for %s; folder will lack the MP4", plan.name)

    (folder / "Info.txt").write_text(render_info_txt(plan), encoding="utf-8")
    return folder


def build_export_tree(
    *,
    export_root: Path,
    source_filename: str,
    items: Iterable[tuple[ClipPlan, Optional[Path]]],
) -> Path:
    """Build <SourceVideoName>/Clip_NN/... under the export root."""
    root = export_root / safe_stem(source_filename)
    root.mkdir(parents=True, exist_ok=True)

    count = 0
    for plan, video_path in items:
        write_clip_folder(root=root, plan=plan, video_path=video_path)
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
