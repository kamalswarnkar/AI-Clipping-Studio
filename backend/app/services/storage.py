"""Project-scoped filesystem layout and safe path handling.

Every project owns one directory. Nothing outside it is ever written, and all
user-supplied names are sanitised before touching the filesystem
(Architecture.md section 31).
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from ..config import get_settings

# Characters Windows forbids in filenames, plus separators and control chars.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WINDOWS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename(name: str, *, fallback: str = "video", max_length: int = 120) -> str:
    """Reduce an arbitrary upload name to something safe to write to disk.

    Strips directory components, control characters, reserved device names and
    trailing dots/spaces (which Windows silently drops).
    """
    name = unicodedata.normalize("NFKD", name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = _UNSAFE.sub("_", name).strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"_{3,}", "__", name)

    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    if stem.upper() in _RESERVED_WINDOWS:
        stem = f"_{stem}"
    stem = stem[:max_length].strip() or fallback

    return f"{stem}.{ext}" if ext else stem


def safe_stem(name: str, *, fallback: str = "video") -> str:
    """Sanitised name without extension -- used as the export root folder name."""
    cleaned = sanitize_filename(name, fallback=fallback)
    return Path(cleaned).stem or fallback


class ProjectStorage:
    """Resolves every path belonging to a single project.

    Layout::

        data/projects/<project_id>/
            source/<original>.mp4      untouched source
            work/                      audio.wav, proxy.mp4, frames/
            analysis/                  transcript.json, scenes.json, ...
            clips/                     Clip_01.mp4, Clip_01.ass, Clip_01.jpg
            export/                    <SourceName>/Clip_01/... and the ZIP
    """

    def __init__(self, project_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", project_id):
            raise ValueError(f"Invalid project id: {project_id!r}")
        self.project_id = project_id
        self.root = get_settings().projects_dir / project_id

    # --- directories --------------------------------------------------------
    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def frames_dir(self) -> Path:
        return self.work_dir / "frames"

    def ensure(self) -> "ProjectStorage":
        for d in (
            self.root,
            self.source_dir,
            self.work_dir,
            self.analysis_dir,
            self.clips_dir,
            self.export_dir,
            self.frames_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # --- well-known files ---------------------------------------------------
    @property
    def audio_path(self) -> Path:
        return self.work_dir / "audio.wav"

    @property
    def proxy_path(self) -> Path:
        return self.work_dir / "proxy.mp4"

    def analysis_file(self, name: str) -> Path:
        return self.analysis_dir / f"{sanitize_filename(name)}.json"

    def clip_video(self, index: int) -> Path:
        return self.clips_dir / f"Clip_{index:02d}.mp4"

    def clip_subtitles(self, index: int) -> Path:
        return self.clips_dir / f"Clip_{index:02d}.ass"

    def clip_thumbnail(self, index: int) -> Path:
        return self.clips_dir / f"Clip_{index:02d}.jpg"

    def zip_path(self, source_name: str) -> Path:
        return self.export_dir / f"{safe_stem(source_name)}.zip"

    # --- JSON artifacts -----------------------------------------------------
    def write_json(self, name: str, payload: Any) -> Path:
        """Write an analysis artifact atomically, so a crash cannot leave a
        half-written file that later parses as valid-but-truncated JSON."""
        path = self.analysis_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        return path

    def read_json(self, name: str) -> Any | None:
        path = self.analysis_file(name)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def has_artifact(self, name: str) -> bool:
        return self.analysis_file(name).exists()

    # --- lifecycle ----------------------------------------------------------
    def delete(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def clear_work(self) -> None:
        """Drop intermediates (audio, proxy, frames) but keep clips and analysis."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def resolve_within(self, candidate: Path) -> Path:
        """Guard against path traversal: the result must stay inside the project."""
        resolved = candidate.resolve()
        root = self.root.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Path escapes the project directory.")
        return resolved
