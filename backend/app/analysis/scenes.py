"""Scene / shot boundary detection.

Runs on the low-resolution proxy rather than the source: scene detection only
needs coarse colour statistics, and decoding a 480p/8fps proxy is roughly an
order of magnitude cheaper than the original stream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from ..models.domain import Scene

log = logging.getLogger(__name__)


def detect_scenes(
    video_path: Path,
    *,
    duration: float,
    threshold: float = 27.0,
    min_scene_seconds: float = 1.5,
    progress: Optional[Callable[[float, str], None]] = None,
) -> list[Scene]:
    """Detect shot boundaries, falling back to a single scene on failure.

    A failure here degrades clip *scoring* but must not stop the pipeline, so the
    caller receives a usable single-scene result rather than an exception.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError:
        log.warning("PySceneDetect unavailable; treating video as one scene.")
        return [Scene(index=0, start=0.0, end=duration)]

    try:
        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=max(1, int(min_scene_seconds * 8)),  # proxy is 8fps
            )
        )

        if progress:
            def _cb(_frame, num: int) -> None:
                if duration > 0:
                    progress(min(0.99, (num / 8.0) / duration), "")

            manager.detect_scenes(video, callback=_cb, show_progress=False)
        else:
            manager.detect_scenes(video, show_progress=False)

        raw = manager.get_scene_list()
    except Exception as exc:  # noqa: BLE001 - optional stage, never fatal
        log.warning("Scene detection failed (%s); treating video as one scene.", exc)
        return [Scene(index=0, start=0.0, end=duration)]

    if not raw:
        return [Scene(index=0, start=0.0, end=duration)]

    scenes = [
        Scene(
            index=i,
            start=float(start.get_seconds()),
            end=min(float(end.get_seconds()), duration),
        )
        for i, (start, end) in enumerate(raw)
    ]

    # Guarantee full coverage even if the detector trimmed the tail.
    if scenes and scenes[-1].end < duration - 0.5:
        scenes[-1] = Scene(index=scenes[-1].index, start=scenes[-1].start, end=duration)

    return scenes


def scene_boundaries_near(
    scenes: list[Scene], t: float, window: float = 1.5
) -> list[float]:
    """Cut points within `window` seconds of t.

    Used by boundary refinement: starting a clip just *after* a cut, or ending it
    just *before* one, looks intentional rather than accidental.
    """
    points: list[float] = []
    for s in scenes:
        for edge in (s.start, s.end):
            if abs(edge - t) <= window:
                points.append(edge)
    return sorted(set(points))


def scene_count_between(scenes: list[Scene], start: float, end: float) -> int:
    return sum(1 for s in scenes if s.end > start and s.start < end)
