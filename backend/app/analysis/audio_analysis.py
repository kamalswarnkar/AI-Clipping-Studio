"""Audio signal analysis: loudness, silence, emphasis and crowd reactions.

Implemented directly on numpy/scipy rather than via librosa, which would pull in
numba and its Python-version lag for features we can compute in a few lines.

Audio is a *supporting* signal. Loud does not mean good, so these features feed
the score with modest weight and never select a clip on their own
(Architecture.md section 8).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from ..models.domain import AudioAnalysis, AudioEvent

log = logging.getLogger(__name__)

HOP_SECONDS = 0.05
WIN_SECONDS = 0.10


def _frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Split into overlapping frames without copying (stride tricks)."""
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop_len
    if n_frames <= 0:
        return np.empty((0, frame_len), dtype=x.dtype)
    strides = (x.strides[0] * hop_len, x.strides[0])
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame_len), strides=strides, writeable=False
    )


def _spectral_flatness(frames: np.ndarray) -> np.ndarray:
    """Geometric / arithmetic mean of the power spectrum.

    Near 1.0 for noise-like signals (applause, crowd), low for voiced speech,
    which is what separates a reaction from someone simply talking loudly.
    """
    window = np.hanning(frames.shape[1]).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    spec = np.maximum(spec, 1e-12)
    geo = np.exp(np.mean(np.log(spec), axis=1))
    arith = np.mean(spec, axis=1)
    return (geo / np.maximum(arith, 1e-12)).astype(np.float32)


def _envelope_modulation(rms: np.ndarray, hop: float) -> np.ndarray:
    """Strength of 3-8 Hz amplitude modulation, per frame.

    Laughter has a characteristic syllabic pulsing in that band; sustained
    applause is comparatively flat.
    """
    win = max(4, int(0.6 / hop))
    out = np.zeros(len(rms), dtype=np.float32)
    if len(rms) < win * 2:
        return out

    for i in range(len(rms)):
        lo = max(0, i - win)
        hi = min(len(rms), i + win)
        seg = rms[lo:hi]
        if len(seg) < 4:
            continue
        seg = seg - seg.mean()
        if not np.any(seg):
            continue
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), d=hop)
        band = (freqs >= 3.0) & (freqs <= 8.0)
        total = spec.sum()
        if total > 0 and band.any():
            out[i] = float(spec[band].sum() / total)
    return out


def _merge_runs(
    mask: np.ndarray, hop: float, *, min_duration: float
) -> list[tuple[float, float]]:
    """Convert a boolean per-frame mask into (start, end) spans."""
    spans: list[tuple[float, float]] = []
    if not mask.any():
        return spans

    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for i in range(0, len(edges) - 1, 2):
        start, end = edges[i] * hop, edges[i + 1] * hop
        if end - start >= min_duration:
            spans.append((float(start), float(end)))
    return spans


def analyze_audio(
    audio_path: Path,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> AudioAnalysis:
    """Compute the loudness envelope and detect notable audio events."""
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read audio for analysis: %s", exc)
        return AudioAnalysis()

    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size == 0:
        return AudioAnalysis(sample_rate=sr)

    duration = len(data) / sr
    hop_len = max(1, int(HOP_SECONDS * sr))
    frame_len = max(hop_len, int(WIN_SECONDS * sr))

    frames = _frame_signal(data, frame_len, hop_len)
    if frames.shape[0] == 0:
        return AudioAnalysis(sample_rate=sr, duration=duration)

    if progress:
        progress(0.3, "loudness")

    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-7))

    if progress:
        progress(0.6, "spectral features")

    flatness = _spectral_flatness(frames)
    modulation = _envelope_modulation(rms, HOP_SECONDS)

    # Reference levels from speech-active frames, so long silences do not drag
    # the baseline down and make everything look like "emphasis".
    voiced = rms_db > (rms_db.max() - 35.0)
    median_db = float(np.median(rms_db[voiced])) if voiced.any() else float(np.median(rms_db))

    events: list[AudioEvent] = []

    # --- silence ------------------------------------------------------------
    for start, end in _merge_runs(
        rms_db < (median_db - 22.0), HOP_SECONDS, min_duration=0.45
    ):
        events.append(AudioEvent(kind="silence", start=start, end=end, strength=1.0))

    # --- emphasis (raised voice / stressed delivery) ------------------------
    for start, end in _merge_runs(
        rms_db > (median_db + 5.0), HOP_SECONDS, min_duration=0.35
    ):
        i0, i1 = int(start / HOP_SECONDS), int(end / HOP_SECONDS)
        peak = float(rms_db[i0:i1].max() - median_db) if i1 > i0 else 0.0
        events.append(
            AudioEvent(
                kind="emphasis",
                start=start,
                end=end,
                strength=float(np.clip(peak / 15.0, 0.0, 1.0)),
            )
        )

    # --- crowd reactions ----------------------------------------------------
    # Reactions are noise-like, but two separate thresholds are needed.
    #
    # Flatness: derived from clearly-voiced frames only. Silence and room tone
    # are *also* noise-like, so a global percentile lands on the reaction level
    # itself and detects nothing.
    speech_frames = rms_db > (median_db - 12.0)
    speech_flatness = (
        float(np.percentile(flatness[speech_frames], 90)) if speech_frames.any() else 0.05
    )
    flat_thresh = float(np.clip(speech_flatness * 3.0, 0.12, 0.45))

    # Level: gate above the noise floor rather than near speech level. Applause
    # is spiky, so its RMS sits well below speech even when clearly audible;
    # requiring near-speech loudness would only ever catch sustained laughter.
    noise_floor = float(np.percentile(rms_db, 5))
    level_gate = max(noise_floor + 12.0, median_db - 30.0)

    reaction_mask = (rms_db > level_gate) & (flatness > flat_thresh)

    for start, end in _merge_runs(reaction_mask, HOP_SECONDS, min_duration=0.8):
        i0, i1 = int(start / HOP_SECONDS), int(end / HOP_SECONDS)
        mod = float(modulation[i0:i1].mean()) if i1 > i0 else 0.0
        flat = float(flatness[i0:i1].mean()) if i1 > i0 else 0.0
        # Pulsing envelope reads as laughter; steady broadband reads as applause.
        kind = "laughter" if mod > 0.32 else "applause"
        events.append(
            AudioEvent(
                kind=kind,
                start=start,
                end=end,
                strength=float(np.clip(flat / max(flat_thresh, 1e-6) / 2.0, 0.2, 1.0)),
            )
        )

    if progress:
        progress(1.0, f"{len(events)} events")

    events.sort(key=lambda e: e.start)
    return AudioAnalysis(
        sample_rate=int(sr),
        duration=duration,
        hop=HOP_SECONDS,
        rms_db=[round(float(v), 2) for v in rms_db],
        events=events,
        global_median_db=median_db,
    )


def dynamics_score(analysis: AudioAnalysis, start: float, end: float) -> float:
    """How dynamic the audio is across a window, in 0..1.

    Rewards variation in level (engaging delivery) and penalises long dead air.
    """
    energy = analysis.energy_between(start, end)
    if not energy:
        return 0.3

    arr = np.asarray(energy, dtype=np.float32)
    spread = float(np.percentile(arr, 90) - np.percentile(arr, 10))
    variation = float(np.clip(spread / 25.0, 0.0, 1.0))

    silent_frames = float(np.mean(arr < (analysis.global_median_db - 22.0)))
    dead_air_penalty = float(np.clip((silent_frames - 0.25) * 1.6, 0.0, 0.6))

    return float(np.clip(0.35 + 0.65 * variation - dead_air_penalty, 0.0, 1.0))


def reaction_score(analysis: AudioAnalysis, start: float, end: float) -> float:
    """Presence of audience reaction inside a window, in 0..1."""
    score = 0.0
    for event in analysis.events_between(start, end):
        if event.kind in ("laughter", "cheer"):
            score += 0.45 * event.strength
        elif event.kind == "applause":
            score += 0.35 * event.strength
        elif event.kind == "emphasis":
            score += 0.12 * event.strength
    return float(np.clip(score, 0.0, 1.0))


def trailing_silence(analysis: AudioAnalysis, t: float, lookahead: float = 2.0) -> float:
    """Seconds of silence starting at t -- used to find natural end points."""
    for event in analysis.events:
        if event.kind == "silence" and event.start <= t + 0.25 <= event.end + 0.25:
            return min(event.end, t + lookahead) - t
    return 0.0
