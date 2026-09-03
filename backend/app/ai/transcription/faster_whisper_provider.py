"""Local speech recognition via faster-whisper (CTranslate2, no torch required).

Produces sentence segments with word-level timestamps. Word timings are what
make intelligent clip boundaries possible, so they are requested unconditionally.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from ...config import get_settings
from ...models.domain import Transcript, TranscriptSegment, Word
from ..base import ProviderError, ProviderUnavailable

log = logging.getLogger(__name__)

# Loading a Whisper model costs seconds and hundreds of MB; cache per config.
_model_cache: dict[tuple[str, str, str], Any] = {}
_model_lock = threading.Lock()

# Fillers Whisper emits for non-speech audio; dropping them keeps candidate text
# clean and prevents applause tags from being treated as spoken content.
_NON_SPEECH = re.compile(
    r"^\s*[\[\(\*](applause|laughter|music|silence|inaudible|noise|clapping)[^\])\*]*[\]\)\*]\s*$",
    re.IGNORECASE,
)


class FasterWhisperProvider:
    name = "faster_whisper"

    def __init__(self) -> None:
        s = get_settings()
        self.model_size = s.whisper_model
        self.device = s.whisper_device
        self.compute_type = s.whisper_compute_type
        self.beam_size = s.whisper_beam_size
        self.cpu_threads = s.whisper_cpu_threads
        self.vad_filter = s.whisper_vad_filter
        self.configured_language = s.whisper_language or None

    # --- availability -------------------------------------------------------
    def is_available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            return False, f"faster-whisper is not installed ({exc})."
        return True, ""

    def _load(self):
        from faster_whisper import WhisperModel

        key = (self.model_size, self.device, self.compute_type)
        with _model_lock:
            if key not in _model_cache:
                log.info(
                    "Loading Whisper model %s (device=%s, compute=%s)",
                    self.model_size,
                    self.device,
                    self.compute_type,
                )
                try:
                    kwargs = {}
                    if self.cpu_threads > 0:
                        kwargs["cpu_threads"] = self.cpu_threads
                    _model_cache[key] = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        **kwargs,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise ProviderUnavailable(
                        f"Could not load the Whisper model '{self.model_size}': {exc}",
                        provider=self.name,
                    ) from exc
            return _model_cache[key]

    # --- transcription ------------------------------------------------------
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> Transcript:
        if not audio_path.exists():
            raise ProviderError(
                f"Audio file missing: {audio_path.name}", provider=self.name
            )

        model = self._load()
        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language or self.configured_language,
                beam_size=self.beam_size,
                word_timestamps=True,
                vad_filter=self.vad_filter,
                vad_parameters={"min_silence_duration_ms": 400},
                condition_on_previous_text=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"Transcription failed: {exc}", provider=self.name, retryable=True
            ) from exc

        total = float(getattr(info, "duration", 0.0) or 0.0)
        segments: list[TranscriptSegment] = []

        # faster-whisper streams lazily; consuming the generator is the work.
        for raw in segments_iter:
            text = (raw.text or "").strip()
            if not text or _NON_SPEECH.match(text):
                continue

            words = [
                Word(
                    word=(w.word or "").strip(),
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(getattr(w, "probability", 1.0) or 0.0),
                )
                for w in (raw.words or [])
                if w.word and w.start is not None and w.end is not None
            ]

            segments.append(
                TranscriptSegment(
                    id=len(segments),
                    start=float(raw.start),
                    end=float(raw.end),
                    text=text,
                    words=words,
                    avg_logprob=float(getattr(raw, "avg_logprob", 0.0) or 0.0),
                    no_speech_prob=float(getattr(raw, "no_speech_prob", 0.0) or 0.0),
                )
            )

            if progress and total > 0:
                progress(min(0.99, float(raw.end) / total), f"{len(segments)} segments")

        if not segments:
            raise ProviderError(
                "No speech was detected in this video.",
                provider=self.name,
            )

        return Transcript(
            language=str(getattr(info, "language", "") or "unknown"),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=total,
            segments=segments,
            model=f"faster-whisper/{self.model_size}",
        )


class NullTranscriptionProvider:
    """Explicit no-op provider so the failure mode is a clear message, not a crash."""

    name = "null"

    def is_available(self) -> tuple[bool, str]:
        return False, "Transcription is disabled (TRANSCRIPTION_PROVIDER=null)."

    def transcribe(self, audio_path: Path, **_: Any) -> Transcript:
        raise ProviderUnavailable(
            "Transcription is disabled, so clips cannot be selected.",
            provider=self.name,
        )
