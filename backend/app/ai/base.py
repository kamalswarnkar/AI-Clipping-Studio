"""Provider protocols.

The pipeline depends on these interfaces, never on a concrete vendor. Swapping
Ollama for another backend, or faster-whisper for a hosted ASR, means adding one
class and changing one environment variable -- no pipeline changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from ..models.domain import Diarization, Transcript


class ProviderError(Exception):
    """A provider failed in a way the pipeline may be able to work around."""

    def __init__(self, message: str, *, provider: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retryable = retryable


class ProviderUnavailable(ProviderError):
    """The provider is not reachable or not configured (e.g. Ollama not running)."""


@runtime_checkable
class TranscriptionProvider(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str]:
        """Return (available, human-readable reason if not)."""
        ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        progress: Optional[Any] = None,
    ) -> Transcript:
        ...


@runtime_checkable
class DiarizationProvider(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str]:
        ...

    def diarize(
        self,
        audio_path: Path,
        *,
        transcript: Optional[Transcript] = None,
        max_speakers: int = 6,
    ) -> Diarization:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def is_available(self) -> tuple[bool, str]:
        ...

    def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Return parsed JSON conforming to `schema`.

        Implementations must validate before returning -- callers are entitled to
        assume the shape is right, because nothing downstream should ever act on
        unvalidated model output.
        """
        ...

    def complete_text(
        self,
        *,
        system: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        ...


@runtime_checkable
class VisionProvider(Protocol):
    name: str
    model: str

    def is_available(self) -> tuple[bool, str]:
        ...

    def describe_frames(
        self,
        image_paths: list[Path],
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        ...
