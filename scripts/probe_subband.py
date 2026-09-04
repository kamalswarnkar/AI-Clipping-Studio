"""Prototype: detect a burned-in subtitle band using the outline signature."""

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "sample.mp4")

cap = cv2.VideoCapture(str(src))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
print(f"{w}x{h}, {total} frames")

N = 60
idxs = np.linspace(total * 0.05, total * 0.95, N).astype(int)

# Subtitles are centred; ignore the outer thirds where logos/banners live.
x0, x1 = int(w * 0.20), int(w * 0.80)

profiles = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, frame = cap.read()
    if not ok:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, x0:x1]

    # Signature of outlined caption text: a very bright pixel with a very dark
    # pixel within a few px. Ordinary bright scenery lacks the dark companion.
    bright = gray > 200
    dark_near = cv2.erode(gray, np.ones((5, 5), np.uint8)) < 80
    glyph = (bright & dark_near).astype(np.float32)
    profiles.append(glyph.mean(axis=1))

cap.release()
P = np.vstack(profiles)
mean_profile = P.mean(axis=0)
present = (P > 0.02).mean(axis=0)

print("\nrow    glyph_density  present")
for y in range(0, h, max(1, h // 36)):
    bar = "#" * int(mean_profile[y] * 600)
    print(f"{y:4d}  {mean_profile[y]:.4f}  {present[y]*100:5.1f}%  {bar}")

# Look only in the lower half, where burned-in captions live.
lower = int(h * 0.55)
region = mean_profile[lower:]
peak = float(region.max())
peak_row = lower + int(region.argmax())
print(f"\nlower-half peak={peak:.4f} at row {peak_row} ({peak_row/h:.3f})")
print(f"upper-half peak={float(mean_profile[:lower].max()):.4f}")

if peak > 0.02:
    rows = np.flatnonzero((mean_profile >= peak * 0.30) & (np.arange(h) >= lower))
    # Contiguous run containing the peak.
    lo = hi = peak_row
    while lo - 1 in rows:
        lo -= 1
    while hi + 1 in rows:
        hi += 1
    print(f"BAND rows {lo}..{hi}  ({lo/h:.3f}..{hi/h:.3f})  "
          f"height={hi-lo+1}px  present_in={present[lo:hi+1].mean()*100:.0f}% of frames")
else:
    print("no subtitle band detected")
