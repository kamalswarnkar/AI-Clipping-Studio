"""Application configuration.

All settings come from environment variables (optionally via a .env file at the
repository root). Nothing here is secret in the default local-only setup, but the
loader is the single place credentials would live, and it is server-side only --
the frontend never receives this object.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class ScoringWeights(BaseModel):
    """Multi-factor candidate scoring weights (Architecture.md section 12).

    This is a content-quality / short-form-suitability score. It is deliberately
    *not* framed as a virality prediction.
    """

    content_clarity: float = 0.20
    standalone_completeness: float = 0.20
    narrative_structure: float = 0.15
    visual_interest: float = 0.10
    audio_dynamics: float = 0.10
    emotional_reaction: float = 0.10
    information_density: float = 0.05
    opening_strength: float = 0.05
    ending_payoff: float = 0.05

    def normalised(self) -> dict[str, float]:
        raw = self.model_dump()
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: Path = REPO_ROOT / "data"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- FFmpeg -------------------------------------------------------------
    ffmpeg_bin: str = ""
    ffprobe_bin: str = ""

    # --- Provider selection -------------------------------------------------
    transcription_provider: Literal["faster_whisper", "null"] = "faster_whisper"
    llm_provider: Literal["ollama", "null"] = "ollama"
    vision_provider: Literal["ollama", "null"] = "ollama"
    diarization_provider: Literal["clustering", "pyannote", "null"] = "clustering"

    # --- Ollama -------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen2.5:7b-instruct"
    ollama_vision_model: str = "qwen2.5vl:7b"
    ollama_timeout: float = 600.0
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.2
    ollama_keep_alive: str = "10m"
    ollama_json_retries: int = 3

    # --- Whisper ------------------------------------------------------------
    whisper_model: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    whisper_cpu_threads: int = 0
    whisper_language: str = ""
    whisper_vad_filter: bool = True
    # Names and terms that recur across your videos. Whisper is biased toward
    # these, which is the only thing that reliably fixes unusual proper nouns.
    whisper_vocabulary: str = ""

    # --- Pipeline -----------------------------------------------------------
    clip_count_default: int = 15
    clip_min_duration: float = 20.0
    clip_max_duration: float = 60.0
    candidate_pool_max: int = 80
    candidate_llm_max: int = 32
    llm_eval_batch: int = 6
    llm_parallel: int = 3
    llm_eval_tokens_per_candidate: int = 150
    vision_enabled: bool = False
    vision_max_candidates: int = 10
    vision_frames_per_candidate: int = 2
    dedupe_iou_threshold: float = 0.35
    dedupe_text_similarity: float = 0.72

    # --- Rendering ----------------------------------------------------------
    render_width: int = 1080
    render_height: int = 1920
    render_fps: int = 30
    render_crf: int = 20
    render_preset: str = "veryfast"
    render_audio_lufs: float = -14.0
    render_workers: int = 3
    subtitle_font: str = "Arial"
    subtitle_font_size: int = 120
    # Detect and crop away burned-in captions already present in the source.
    remove_source_subtitles: bool = True

    # --- Attention / conflict scoring ---------------------------------------
    opening_window_seconds: float = 3.0
    opening_weight: float = 0.22
    conflict_weight: float = 0.18

    # --- Jobs ---------------------------------------------------------------
    job_workers: int = 2
    job_max_retries: int = 2

    # --- Scoring ------------------------------------------------------------
    scoring_weights_file: Path = REPO_ROOT / "backend" / "config" / "scoring.json"

    @field_validator("data_dir", mode="before")
    @classmethod
    def _resolve_data_dir(cls, v: object) -> Path:
        p = Path(str(v)).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "studio.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.projects_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def scoring_weights(self) -> ScoringWeights:
        """Load weights from the JSON config file, falling back to defaults."""
        try:
            if self.scoring_weights_file.exists():
                data = json.loads(self.scoring_weights_file.read_text("utf-8"))
                return ScoringWeights(**data)
        except Exception:  # noqa: BLE001 - config must never block startup
            pass
        return ScoringWeights()


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

def _bundled(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return REPO_ROOT / "tools" / "ffmpeg" / "bin" / f"{name}{suffix}"


@functools.lru_cache(maxsize=8)
def resolve_binary(name: Literal["ffmpeg", "ffprobe"]) -> str:
    """Locate ffmpeg/ffprobe.

    Order: explicit env setting -> system PATH -> repo-local tools/ffmpeg ->
    the binary bundled with imageio-ffmpeg (ffmpeg only).

    Raising here (rather than at import) keeps the API able to boot and report a
    precise, actionable error instead of crashing on startup.
    """
    settings = get_settings()
    explicit = settings.ffmpeg_bin if name == "ffmpeg" else settings.ffprobe_bin
    if explicit and Path(explicit).exists():
        return str(Path(explicit))

    on_path = shutil.which(name)
    if on_path:
        return on_path

    local = _bundled(name)
    if local.exists():
        return str(local)

    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            pass

    raise FileNotFoundError(
        f"{name} could not be found. Install it, put it on PATH, set "
        f"{name.upper()}_BIN in .env, or run: python scripts/setup_ffmpeg.py"
    )
