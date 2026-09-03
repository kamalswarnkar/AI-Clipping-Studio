"""Smart 9:16 reframing.

Computes a crop path over the source frame so the people talking stay in shot
when a landscape video becomes vertical (Architecture.md section 16).

Strategy, in order of preference:

1. If a crop can hold *every* face on screen, use it -- a two-person exchange
   should keep both people visible.
2. Otherwise follow the dominant/active face, moving smoothly.
3. If no faces were detected at all, fall back to a centre crop biased slightly
   above middle, where heads usually sit.

The output is a list of keyframes, converted by the renderer into an ffmpeg crop
expression. The source file is never modified.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..models.domain import CropKeyframe, FaceBox, VisualAnalysis

log = logging.getLogger(__name__)

# Faces sit higher than centre; place the crop so heads are in the upper third.
VERTICAL_BIAS = 0.42

# Movement smaller than this is imperceptible and only adds jitter.
MIN_SHIFT_PX = 24

# Seconds between keyframes on the crop path.
KEYFRAME_INTERVAL = 0.5


def _crop_size(
    source_w: int, source_h: int, target_w: int, target_h: int
) -> tuple[int, int]:
    """Largest region of the source matching the target aspect ratio."""
    target_ratio = target_w / target_h

    crop_h = source_h
    crop_w = int(round(crop_h * target_ratio))
    if crop_w > source_w:
        crop_w = source_w
        crop_h = int(round(crop_w / target_ratio))

    # H.264 needs even dimensions.
    return (crop_w - crop_w % 2, crop_h - crop_h % 2)


def _weighted_centre(faces: list[FaceBox]) -> tuple[float, float]:
    """Face centroid weighted by area -- the closest face dominates."""
    total = sum(f.area for f in faces) or 1
    cx = sum(f.cx * f.area for f in faces) / total
    cy = sum(f.cy * f.area for f in faces) / total
    return cx, cy


def _faces_span(faces: list[FaceBox]) -> tuple[float, float]:
    left = min(f.x for f in faces)
    right = max(f.x + f.w for f in faces)
    return float(left), float(right)


def _smooth(values: list[float], window: int = 5) -> list[float]:
    """Median filter then moving average: kills detector flicker, keeps motion."""
    if len(values) < 3:
        return values

    arr = np.asarray(values, dtype=np.float64)
    window = min(window, len(arr) if len(arr) % 2 else len(arr) - 1)
    if window >= 3:
        pad = window // 2
        padded = np.pad(arr, pad, mode="edge")
        arr = np.array(
            [np.median(padded[i : i + window]) for i in range(len(arr))]
        )

    kernel = min(5, len(arr))
    if kernel >= 2:
        weights = np.ones(kernel) / kernel
        arr = np.convolve(np.pad(arr, (kernel // 2, kernel // 2), mode="edge"), weights, mode="same")
        arr = arr[kernel // 2 : kernel // 2 + len(values)]

    return [float(v) for v in arr]


def compute_crop_path(
    *,
    visual: Optional[VisualAnalysis],
    start: float,
    end: float,
    source_w: int,
    source_h: int,
    target_w: int = 1080,
    target_h: int = 1920,
) -> tuple[list[CropKeyframe], str]:
    """Build the crop path for one clip.

    Returns (keyframes, strategy) where strategy explains the choice for the
    clip's Info.txt and for debugging.
    """
    crop_w, crop_h = _crop_size(source_w, source_h, target_w, target_h)
    max_x = max(0, source_w - crop_w)
    max_y = max(0, source_h - crop_h)

    # Default: centred horizontally, biased upward vertically.
    default_x = max_x // 2
    default_y = int(min(max_y, max(0, (source_h * VERTICAL_BIAS) - crop_h / 2)))

    if crop_w >= source_w and crop_h >= source_h:
        return (
            [CropKeyframe(t=0.0, x=0, y=0, w=crop_w, h=crop_h)],
            "no crop needed (source already matches target aspect)",
        )

    frames = visual.frames_between(start, end) if visual else []
    frames_with_faces = [f for f in frames if f.faces]

    if not frames_with_faces:
        return (
            [CropKeyframe(t=0.0, x=default_x, y=default_y, w=crop_w, h=crop_h)],
            "centre crop (no faces detected)",
        )

    # --- can one static crop hold everyone? ---------------------------------
    all_faces = [face for f in frames_with_faces for face in f.faces]
    span_left, span_right = _faces_span(all_faces)
    span = span_right - span_left

    if span <= crop_w * 0.92:
        centre = (span_left + span_right) / 2.0
        x = int(round(np.clip(centre - crop_w / 2.0, 0, max_x)))
        cy = float(np.mean([f.cy for f in all_faces]))
        y = int(round(np.clip(cy - crop_h * VERTICAL_BIAS, 0, max_y)))
        strategy = (
            "static crop holding all speakers"
            if len({round(f.cx / 100) for f in all_faces}) > 1
            else "static crop on speaker"
        )
        return ([CropKeyframe(t=0.0, x=x, y=y, w=crop_w, h=crop_h)], strategy)

    # --- otherwise track the dominant face ----------------------------------
    times: list[float] = []
    centres_x: list[float] = []
    centres_y: list[float] = []

    last_cx = float(np.mean([f.cx for f in all_faces]))
    last_cy = float(np.mean([f.cy for f in all_faces]))

    for frame in frames:
        if frame.faces:
            # The largest face is the closest to camera, which in an interview
            # is nearly always the person being featured at that moment.
            cx, cy = _weighted_centre(frame.faces)
            last_cx, last_cy = cx, cy
        else:
            cx, cy = last_cx, last_cy  # hold position through detection gaps
        times.append(frame.t - start)
        centres_x.append(cx)
        centres_y.append(cy)

    centres_x = _smooth(centres_x, window=7)
    centres_y = _smooth(centres_y, window=7)

    keyframes: list[CropKeyframe] = []
    last_x: Optional[int] = None
    last_y: Optional[int] = None

    for t, cx, cy in zip(times, centres_x, centres_y):
        x = int(round(np.clip(cx - crop_w / 2.0, 0, max_x)))
        y = int(round(np.clip(cy - crop_h * VERTICAL_BIAS, 0, max_y)))

        # Only emit a keyframe when the crop actually needs to move.
        if (
            last_x is not None
            and abs(x - last_x) < MIN_SHIFT_PX
            and abs(y - last_y) < MIN_SHIFT_PX
        ):
            continue

        keyframes.append(
            CropKeyframe(t=round(max(0.0, t), 3), x=x, y=y, w=crop_w, h=crop_h)
        )
        last_x, last_y = x, y

    if not keyframes:
        keyframes = [CropKeyframe(t=0.0, x=default_x, y=default_y, w=crop_w, h=crop_h)]
    elif keyframes[0].t > 0.0:
        first = keyframes[0]
        keyframes.insert(
            0, CropKeyframe(t=0.0, x=first.x, y=first.y, w=crop_w, h=crop_h)
        )

    return keyframes, f"active-speaker tracking ({len(keyframes)} keyframes)"


def build_crop_expression(keyframes: list[CropKeyframe], axis: str) -> str:
    """Render a crop coordinate as an ffmpeg expression over clip time `t`.

    A single keyframe becomes a constant. Multiple keyframes become nested
    linear interpolations, so the crop glides between positions instead of
    snapping, which would read as a jump cut.
    """
    if not keyframes:
        return "0"

    values = [getattr(k, axis) for k in keyframes]
    if len(keyframes) == 1 or len(set(values)) == 1:
        return str(values[0])

    # Build from the last span backwards so each `if` falls through correctly.
    expression = str(values[-1])
    for i in range(len(keyframes) - 2, -1, -1):
        t0, t1 = keyframes[i].t, keyframes[i + 1].t
        v0, v1 = values[i], values[i + 1]
        span = max(1e-6, t1 - t0)
        # Linear ramp from v0 at t0 to v1 at t1.
        ramp = f"({v0}+({v1}-{v0})*(t-{t0:.3f})/{span:.3f})"
        expression = f"if(lt(t,{t1:.3f}),{ramp},{expression})"

    return expression
