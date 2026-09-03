"""Provider registry.

One place that maps configuration to concrete provider instances. The pipeline
only ever asks for `get_providers()`, so switching backends is a config change.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

from ..config import get_settings
from .base import (
    DiarizationProvider,
    LLMProvider,
    TranscriptionProvider,
    VisionProvider,
)
from .diarization.clustering_provider import (
    ClusteringDiarizationProvider,
    NullDiarizationProvider,
)
from .llm.ollama_provider import (
    NullLLMProvider,
    NullVisionProvider,
    OllamaLLMProvider,
    OllamaVisionProvider,
)
from .transcription.faster_whisper_provider import (
    FasterWhisperProvider,
    NullTranscriptionProvider,
)

log = logging.getLogger(__name__)


@dataclass
class Providers:
    transcription: TranscriptionProvider
    diarization: DiarizationProvider
    llm: LLMProvider
    vision: VisionProvider

    def health(self) -> dict[str, dict[str, object]]:
        """Availability of each provider, for the API health endpoint."""
        out: dict[str, dict[str, object]] = {}
        for key in ("transcription", "diarization", "llm", "vision"):
            provider = getattr(self, key)
            try:
                available, reason = provider.is_available()
            except Exception as exc:  # noqa: BLE001
                available, reason = False, str(exc)
            out[key] = {
                "provider": getattr(provider, "name", "unknown"),
                "model": getattr(provider, "model", ""),
                "available": available,
                "reason": reason,
            }
        return out


def _build_diarization(name: str) -> DiarizationProvider:
    if name == "clustering":
        return ClusteringDiarizationProvider()
    if name == "pyannote":
        try:
            from .diarization.pyannote_provider import PyannoteDiarizationProvider

            return PyannoteDiarizationProvider()
        except ImportError as exc:
            log.warning(
                "pyannote requested but unavailable (%s); using clustering instead.",
                exc,
            )
            return ClusteringDiarizationProvider()
    return NullDiarizationProvider()


@functools.lru_cache(maxsize=1)
def get_providers() -> Providers:
    s = get_settings()
    return Providers(
        transcription=(
            FasterWhisperProvider()
            if s.transcription_provider == "faster_whisper"
            else NullTranscriptionProvider()
        ),
        diarization=_build_diarization(s.diarization_provider),
        llm=OllamaLLMProvider() if s.llm_provider == "ollama" else NullLLMProvider(),
        vision=(
            OllamaVisionProvider() if s.vision_provider == "ollama" else NullVisionProvider()
        ),
    )


def reset_providers() -> None:
    """Drop cached providers (used after a settings change or in tests)."""
    get_providers.cache_clear()
