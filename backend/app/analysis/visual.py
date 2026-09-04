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
    sample_interval: float = 1.0,
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
            # Detect on a downscaled copy: Haar cost is quadratic in pixels and
            # face positions only need to be accurate to a few source pixels for
            # cropping. This is the difference between minutes and seconds on a
            # long video.
            boxes: list[FaceBox] = []
            if frontal is not None:
                det_w = 320
                det_scale = det_w / gray.shape[1] if gray.shape[1] > det_w else 1.0
                small = (
                    cv2.resize(gray, (det_w, int(gray.shape[0] * det_scale)))
                    if det_scale < 1.0
                    else gray
                )
                back = (1.0 / det_scale) if det_scale else 1.0
                equalised = cv2.equalizeHist(small)
                detections = frontal.detectMultiScale(
                    equalised, scaleFactor=1.2, minNeighbors=5, minSize=(18, 18)
                )
                for (x, y, w, h) in detections:
                    boxes.append(
                        FaceBox(
                            x=int(x * back * scale_x),
                            y=int(y * back * scale_y),
                            w=int(w * back * scale_x),
                            h=int(h * back * scale_y),
                            confidence=0.8,
                        )
                    )
                # The profile cascade roughly doubles cost, so only fall back to
                # it when the frontal pass found nothing at all.
                if profile is not None and not boxes:
                    for (x, y, w, h) in profile.detectMultiScale(
                        equalised, scaleFactor=1.25, minNeighbors=5, minSize=(18, 18)
                    ):
                        boxes.append(
                            FaceBox(
                                x=int(x * back * scale_x),
                                y=int(y * back * scale_y),
                                w=int(w * back * scale_x),
                                h=int(h * back * scale_y),
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


def detect_subtitle_band(
    video_path: Path,
    *,
    samples: int = 48,
) -> Optional[float]:
    """Find burned-in subtitles along the bottom of the source.

    Returns the fraction of frame height at which the band starts (so the caller
    can crop it away), or None when no band is found.

    Why this matters: converting 16:9 to 9:16 keeps full height but throws away
    most of the width, which slices burned-in captions down the middle. The
    result looks broken, and the app then burns its own captions on top of the
    wreckage. Removing the band first leaves one clean set of captions.

    Detection keys on the signature of outlined caption text -- a very bright
    pixel with a very dark pixel within a few pixels. Ordinary bright scenery
    (sky, signage, clothing) lacks that dark companion, which is what makes this
    separable where plain edge density is not.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    if total <= 0 or height <= 0 or width <= 0:
        cap.release()
        return None

    # Captions are centred; the outer fifths hold logos and lower-third banners.
    x0, x1 = int(width * 0.20), int(width * 0.80)
    kernel = np.ones((5, 5), np.uint8)

    profiles: list[np.ndarray] = []
    for index in np.linspace(total * 0.05, total * 0.95, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, x0:x1]
        glyph = (gray > 200) & (cv2.erode(gray, kernel) < 80)
        profiles.append(glyph.astype(np.float32).mean(axis=1))

    cap.release()
    if len(profiles) < 8:
        return None

    profile = np.vstack(profiles).mean(axis=0)

    lower_start = int(height * 0.60)
    baseline = float(np.percentile(profile[:lower_start], 90))
    region = profile[lower_start:]
    peak = float(region.max())

    # Require both an absolute presence and a clear margin over the rest of the
    # frame, so videos without burned-in captions are left alone.
    if peak < 0.018 or peak < baseline * 1.6:
        log.info(
            "No burned-in subtitle band (peak=%.4f, baseline=%.4f)", peak, baseline
        )
        return None

    # Grow outward from the PEAK, not from the topmost hit anywhere below.
    # Lower-third banners and station logos also clear the threshold, and
    # anchoring on them would crop away a quarter of the frame.
    threshold = max(baseline * 1.2, peak * 0.25)
    above = profile >= threshold
    peak_row = lower_start + int(region.argmax())

    band_top = peak_row
    while band_top - 1 >= lower_start and above[band_top - 1]:
        band_top -= 1

    # Captions wrap to two lines with a gap between them; look a little further
    # up for a second line rather than slicing its top off.
    lookup = int(height * 0.08)
    probe = band_top - 1
    while probe >= max(lower_start, band_top - lookup):
        if above[probe]:
            band_top = probe
            while band_top - 1 >= lower_start and above[band_top - 1]:
                band_top -= 1
            probe = band_top - 1
            continue
        probe -= 1

    # A small margin above the glyphs catches descenders and outline bleed.
    band_top = max(lower_start, band_top - int(height * 0.015))

    # Never sacrifice more than a fifth of the frame on a heuristic.
    fraction = max(0.80, min(0.97, band_top / height))

    log.info(
        "Burned-in subtitle band detected from %.1f%% of height "
        "(peak=%.4f vs baseline=%.4f)",
        fraction * 100,
        peak,
        baseline,
    )
    return fraction
