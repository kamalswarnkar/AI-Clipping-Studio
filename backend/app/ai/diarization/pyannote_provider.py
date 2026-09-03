"""Optional pyannote.audio diarization.

Stronger than the built-in clustering provider, but it needs torch, a model
download and a HuggingFace token with the gated pyannote terms accepted. It is
therefore opt-in via DIARIZATION_PROVIDER=pyannote rather than a default.

Install with:
    pip install pyannote.audio
    set HUGGINGFACE_TOKEN=hf_...
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ...models.domain import Diarization, SpeakerTurn, Transcript
from ..base import ProviderError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


class PyannoteDiarizationProvider:
    name = "pyannote"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        self._pipeline = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            return False, (
                "pyannote.audio is not installed. Install it with "
                "'pip install pyannote.audio', or use DIARIZATION_PROVIDER=clustering."
            )
        if not self.token:
            return False, (
                "pyannote needs a HuggingFace token. Set HUGGINGFACE_TOKEN and accept "
                f"the model terms for {self.model}."
            )
        return True, ""

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline

        from pyannote.audio import Pipeline

        try:
            self._pipeline = Pipeline.from_pretrained(
                self.model, use_auth_token=self.token
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"Could not load {self.model}: {exc}", provider=self.name
            ) from exc
        return self._pipeline

    def diarize(
        self,
        audio_path: Path,
        *,
        transcript: Optional[Transcript] = None,
        max_speakers: int = 6,
    ) -> Diarization:
        available, reason = self.is_available()
        if not available:
            raise ProviderError(reason, provider=self.name)

        pipeline = self._load()
        try:
            annotation = pipeline(str(audio_path), max_speakers=max_speakers)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"Diarization failed: {exc}", provider=self.name, retryable=True
            ) from exc

        # Map pyannote's SPEAKER_00 labels onto our A/B/C naming, ordered by
        # first appearance so labels stay stable and readable.
        order: dict[str, str] = {}
        turns: list[SpeakerTurn] = []

        for segment, _, label in annotation.itertracks(yield_label=True):
            if label not in order:
                order[label] = f"Speaker {chr(ord('A') + len(order))}"
            speaker = order[label]

            if turns and turns[-1].speaker == speaker and segment.start - turns[-1].end < 1.0:
                turns[-1] = SpeakerTurn(
                    speaker=speaker, start=turns[-1].start, end=float(segment.end)
                )
            else:
                turns.append(
                    SpeakerTurn(
                        speaker=speaker,
                        start=float(segment.start),
                        end=float(segment.end),
                    )
                )

        return Diarization(
            speakers=sorted(order.values()),
            turns=turns,
            method=f"pyannote/{self.model}",
        )
