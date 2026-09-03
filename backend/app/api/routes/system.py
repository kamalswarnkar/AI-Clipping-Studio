"""System health.

The upload screen calls this so problems (Ollama not running, model not pulled,
ffmpeg missing) are visible *before* a user waits through a long upload.
"""

from __future__ import annotations

from fastapi import APIRouter

from ...ai.registry import get_providers
from ...config import get_settings, resolve_binary
from ...models.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()

    ffmpeg_info: dict[str, object] = {}
    try:
        ffmpeg_info = {
            "available": True,
            "ffmpeg": resolve_binary("ffmpeg"),
            "ffprobe": resolve_binary("ffprobe"),
        }
    except FileNotFoundError as exc:
        ffmpeg_info = {"available": False, "error": str(exc)}

    providers = get_providers().health()

    # Everything essential must be present for the pipeline to run at all.
    essential_ok = bool(ffmpeg_info.get("available")) and providers["transcription"]["available"]
    degraded = not providers["llm"]["available"]

    return HealthResponse(
        status="ok" if essential_ok and not degraded else ("degraded" if essential_ok else "error"),
        ffmpeg=ffmpeg_info,
        providers=providers,
        settings={
            "clip_count_default": settings.clip_count_default,
            "clip_min_duration": settings.clip_min_duration,
            "clip_max_duration": settings.clip_max_duration,
            "vision_enabled": settings.vision_enabled,
            "whisper_model": settings.whisper_model,
            "render_resolution": f"{settings.render_width}x{settings.render_height}",
        },
    )
