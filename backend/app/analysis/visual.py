"""Cheap visual analysis: face presence, motion, exposure and sharpness.

Runs over the whole proxy with OpenCV -- no model API calls. These signals feed
candidate scoring and, critically, drive the 9:16 reframing crop path.

Expensive multimodal reasoning is a separate, bounded stage (see ai/vision.py);
this module is the "sample cheaply, everywhere" half of the strategy in
Architecture.md section 7.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..models.domain import FaceBox, FrameAnalysis, VisualAnalysis

log = logging.getLogger(__name__)


def _load_cascades() -> tuple[Optional[cv2.CascadeClassifier], Optional[cv2.CascadeClassifier]]:
    """Haar cascades ship inside the opencv wheel, so nothing extra downloads."""
    try:
        base = Path(cv2.data.haarcascades)
        frontal = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        profile = cv2.CascadeClassifier(str(base / "haarcascade_profileface.xml"))
        return (
            frontal if not frontal.empty() else None,
            profile if not profile.empty() else None,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load face cascades: %s", exc)
        return None, None


def _dedupe_faces(boxes: list[FaceBox], iou_threshold: float = 0.35) -> list[FaceBox]:
    """Frontal and profile cascades often fire on the same face; keep one."""
    kept: list[FaceBox] = []
    for box in sorted(boxes, key=lambda b: b.area, reverse=True):
        overlaps = False
        for existing in kept:
            x0 = max(box.x, existing.x)
            y0 = max(box.y, existing.y)
            x1 = min(box.x + box.w, existing.x + existing.w)
            y1 = min(box.y + box.h, existing.y + existing.h)
            inter = max(0, x1 - x0) * max(0, y1 - y0)
            union = box.area + existing.area - inter
            if union > 0 and inter / union > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(box)
    return kept


def analyze_visuals(
    proxy_path: Path,
    *,
    source_width: int,
    source_height: int,
    sample_interval: float = 0.5,
    progress: Optional[Callable[[float, str], None]] = None,
) -> VisualAnalysis:
    """Sample the proxy and record per-frame observations.

    Face coordinates are scaled back to *source* pixels so the renderer can crop
    the original video directly.
    """
    cap = cv2.VideoCapture(str(proxy_path))
    if not cap.isOpened():
        log.warning("Could not open proxy for visual analysis: %s", proxy_path)
        return VisualAnalysis(sample_interval=sample_interval)

    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    proxy_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    proxy_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    scale_x = (source_width / proxy_w) if proxy_w else 1.0
    scale_y = (source_height / proxy_h) if proxy_h else 1.0

    frontal, profile = _load_cascades()
    step = max(1, int(round(sample_interval * fps)))

    frames: list[FrameAnalysis] = []
    prev_gray: Optional[np.ndarray] = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            t = idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- motion: mean absolute difference against the previous sample
            motion = 0.0
            if prev_gray is not None and prev_gray.shape == gray.shape:
                motion = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            prev_gray = gray

            # --- faces
            boxes: list[FaceBox] = []
            if frontal is not None:
                equalised = cv2.equalizeHist(gray)
                detections = frontal.detectMultiScale(
                    equalised, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24)
                )
                for (x, y, w, h) in detections:
                    boxes.append(
                        FaceBox(
                            x=int(x * scale_x),
                            y=int(y * scale_y),
                            w=int(w * scale_x),
                            h=int(h * scale_y),
                            confidence=0.8,
                        )
                    )
                if profile is not None and len(boxes) < 2:
                    for (x, y, w, h) in profile.detectMultiScale(
                        equalised, scaleFactor=1.2, minNeighbors=5, minSize=(24, 24)
                    ):
                        boxes.append(
                            FaceBox(
                                x=int(x * scale_x),
                                y=int(y * scale_y),
                                w=int(w * scale_x),
                                h=int(h * scale_y),
                                confidence=0.55,
                            )
                        )

            frames.append(
                FrameAnalysis(
                    t=round(t, 3),
                    faces=_dedupe_faces(boxes),
                    motion=round(motion, 4),
                    brightness=round(float(gray.mean()) / 255.0, 4),
                    sharpness=round(
                        float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0, 4
                    ),
                )
            )

            if progress and total_frames:
                progress(min(0.99, idx / total_frames), f"{len(frames)} frames")

        idx += 1

    cap.release()

    return VisualAnalysis(
        frames=frames,
        sample_interval=sample_interval,
        frame_width=source_width,
        frame_height=source_height,
    )


def visual_interest(analysis: VisualAnalysis, start: float, end: float) -> float:
    """How visually usable a window is, in 0..1.

    Rewards a visible subject and some movement; penalises very dark, very flat
    or badly out-of-focus stretches.
    """
    frames = analysis.frames_between(start, end)
    if not frames:
        return 0.4  # unknown, not bad

    with_faces = sum(1 for f in frames if f.faces) / len(frames)
    motion = float(np.mean([f.motion for f in frames]))
    brightness = float(np.mean([f.brightness for f in frames]))
    sharpness = float(np.mean([f.sharpness for f in frames]))

    score = 0.30
    score += 0.30 * with_faces
    # Some motion is good; a static frame or a chaotic one both read worse.
    score += 0.20 * float(np.clip(motion / 0.08, 0.0, 1.0))
    score += 0.10 * float(np.clip(sharpness / 0.5, 0.0, 1.0))

    if brightness < 0.12 or brightness > 0.94:
        score -= 0.20

    return float(np.clip(score, 0.0, 1.0))


def face_track(
    analysis: VisualAnalysis, start: float, end: float
) -> list[tuple[float, list[FaceBox]]]:
    """Face observations inside a window, as (timestamp, boxes)."""
    return [(f.t, f.faces) for f in analysis.frames_between(start, end)]


def dominant_face_count(analysis: VisualAnalysis, start: float, end: float) -> int:
    """Typical number of faces on screen -- median avoids single-frame flicker."""
    frames = analysis.frames_between(start, end)
    if not frames:
        return 0
    return int(np.median([len(f.faces) for f in frames]))
