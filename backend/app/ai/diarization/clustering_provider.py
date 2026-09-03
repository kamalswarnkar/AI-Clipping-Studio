"""Lightweight speaker diarization, fully local and dependency-light.

Approach: cut the audio at transcript segment boundaries, compute an MFCC-based
embedding per segment, and cluster them. This is meaningfully weaker than
pyannote.audio, but it needs no torch, no model download and no HuggingFace
token -- and it is enough to label turns 'Speaker A/B/C' and detect exchanges,
which is what clip selection actually consumes.

pyannote remains selectable via DIARIZATION_PROVIDER=pyannote for users who want
the stronger model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.fftpack import dct

from ...models.domain import Diarization, SpeakerTurn, Transcript

log = logging.getLogger(__name__)

N_MELS = 26
N_MFCC = 13
FRAME_MS = 25
HOP_MS = 10


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    low, high = 80.0, min(7600.0, sr / 2.0 - 1.0)
    points = _mel_to_hz(np.linspace(_hz_to_mel(low), _hz_to_mel(high), n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        if centre == left:
            centre = left + 1
        if right <= centre:
            right = centre + 1
        if right > n_fft // 2:
            break
        fb[m - 1, left:centre] = np.linspace(0, 1, centre - left, endpoint=False)
        fb[m - 1, centre:right] = np.linspace(1, 0, right - centre, endpoint=False)
    return fb


def _mfcc(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute MFCCs for a mono signal. Returns (frames, N_MFCC)."""
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    if len(signal) < frame_len:
        return np.empty((0, N_MFCC), dtype=np.float32)

    # Pre-emphasis lifts the higher formants that distinguish voices.
    emphasised = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    n_frames = 1 + (len(emphasised) - frame_len) // hop_len
    strides = (emphasised.strides[0] * hop_len, emphasised.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        emphasised, shape=(n_frames, frame_len), strides=strides
    )

    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2

    windowed = frames * np.hamming(frame_len)
    power = (np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)) ** 2) / n_fft

    fb = _mel_filterbank(sr, n_fft)
    mel = np.maximum(power @ fb.T, 1e-10)
    return dct(np.log(mel), type=2, axis=1, norm="ortho")[:, :N_MFCC].astype(np.float32)


def _embed(signal: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """One fixed-length vector per speech segment (mean + std of MFCCs)."""
    coeffs = _mfcc(signal, sr)
    if coeffs.shape[0] < 5:
        return None
    # Drop C0 (overall energy) so loudness does not dominate voice identity.
    coeffs = coeffs[:, 1:]
    return np.concatenate([coeffs.mean(axis=0), coeffs.std(axis=0)])


def _estimate_speaker_count(features: np.ndarray, max_speakers: int) -> int:
    """Pick k by silhouette score, defaulting to 1 when clusters are not separable."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(features)
    upper = min(max_speakers, n - 1)
    if upper < 2:
        return 1

    best_k, best_score = 1, -1.0
    for k in range(2, upper + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
                features
            )
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(features, labels))
        except Exception:  # noqa: BLE001
            continue
        if score > best_score:
            best_k, best_score = k, score

    # Below this, the "clusters" are noise rather than distinct voices.
    return best_k if best_score >= 0.12 else 1


class ClusteringDiarizationProvider:
    name = "clustering"

    def is_available(self) -> tuple[bool, str]:
        try:
            import sklearn  # noqa: F401
        except ImportError as exc:
            return False, f"scikit-learn is not installed ({exc})."
        return True, ""

    def diarize(
        self,
        audio_path: Path,
        *,
        transcript: Optional[Transcript] = None,
        max_speakers: int = 6,
    ) -> Diarization:
        if transcript is None or not transcript.segments:
            return Diarization(method="none")

        try:
            audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("Diarization could not read audio: %s", exc)
            return Diarization(method="failed")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Embed each transcript segment that is long enough to characterise.
        embeddings: list[np.ndarray] = []
        indices: list[int] = []
        for i, seg in enumerate(transcript.segments):
            if seg.duration < 0.6:
                continue
            i0, i1 = int(seg.start * sr), int(seg.end * sr)
            chunk = audio[max(0, i0) : min(len(audio), i1)]
            vec = _embed(chunk, sr)
            if vec is not None and np.isfinite(vec).all():
                embeddings.append(vec)
                indices.append(i)

        if len(embeddings) < 2:
            return Diarization(
                speakers=["Speaker A"],
                turns=[
                    SpeakerTurn(speaker="Speaker A", start=s.start, end=s.end)
                    for s in transcript.segments
                ],
                method="single-speaker",
            )

        features = np.vstack(embeddings)
        # Standardise so no coefficient dominates the distance metric.
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

        k = _estimate_speaker_count(features, max_speakers)
        if k <= 1:
            labels = np.zeros(len(features), dtype=int)
        else:
            from sklearn.cluster import AgglomerativeClustering

            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
                features
            )

        # Map cluster ids to stable A/B/C labels ordered by first appearance.
        order: dict[int, str] = {}
        for label in labels:
            if label not in order:
                order[int(label)] = f"Speaker {chr(ord('A') + len(order))}"

        assigned: dict[int, str] = {
            seg_idx: order[int(label)] for seg_idx, label in zip(indices, labels)
        }

        # Segments too short to embed inherit the previous speaker.
        turns: list[SpeakerTurn] = []
        last = order[int(labels[0])]
        for i, seg in enumerate(transcript.segments):
            speaker = assigned.get(i, last)
            last = speaker
            if turns and turns[-1].speaker == speaker and seg.start - turns[-1].end < 1.2:
                turns[-1] = SpeakerTurn(
                    speaker=speaker, start=turns[-1].start, end=seg.end
                )
            else:
                turns.append(
                    SpeakerTurn(speaker=speaker, start=seg.start, end=seg.end)
                )

        return Diarization(
            speakers=sorted(order.values()),
            turns=turns,
            method=f"mfcc-agglomerative(k={len(order)})",
        )


class NullDiarizationProvider:
    name = "null"

    def is_available(self) -> tuple[bool, str]:
        return False, "Diarization is disabled (DIARIZATION_PROVIDER=null)."

    def diarize(self, audio_path: Path, **_: object) -> Diarization:
        return Diarization(method="disabled")
