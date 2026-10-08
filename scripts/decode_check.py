"""Is sampled frame *i* the same picture no matter how FrameSource reaches it?

Diagnostic for seeking in variable-frame-rate videos.
Ground truth is a sequential decode of the whole clip (one md5 per real frame); then
raw ``CAP_PROP_POS_FRAMES`` seeks and the app's own ``FrameSource`` in several scrub
orders are mapped back to the real frame they returned. A correct FrameSource prints
``off=+0`` on every line.

    uv run python scripts/decode_check.py <clip.mp4> [sampled_idx=136] [stride=5]

Decodes the full clip once (~70 s for a 10-min 4K video).
"""

import hashlib
import sys

import cv2
import numpy as np

from kumo_track.annotate.frames import FrameSource

path = sys.argv[1]
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 136
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 5
WANT = TARGET * STRIDE


def sig(bgr):
    return hashlib.md5(bgr.tobytes()).hexdigest()[:10]


def thumb(bgr):
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (48, 27)).astype(np.int16)


# --- 1. sequential ground truth ---------------------------------------------
cap = cv2.VideoCapture(path)
declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
sigs, thumbs, pts = [], [], []
while True:
    ok, f = cap.read()
    if not ok:
        break
    sigs.append(sig(f))
    thumbs.append(thumb(f))
    pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
cap.release()
real = len(sigs)
print(f"declared={declared} fps={fps:.3f} really_decodable={real}")
dts = np.diff(pts)
med = np.median(dts)
print(f"frame interval ms: median={med:.2f} min={dts.min():.2f} max={dts.max():.2f} "
      f"(#gaps>1.5x median: {(dts > 1.5 * med).sum()})  <- gaps = variable frame rate")
sig_to_idx = {}
for i, s in enumerate(sigs):
    sig_to_idx.setdefault(s, i)
print(f"byte-identical duplicate frames in stream: {real - len(sig_to_idx)}")


def which(bgr):
    """Real frame index for a decoded picture (exact md5, else nearest thumbnail)."""
    s = sig(bgr)
    if s in sig_to_idx:
        return sig_to_idx[s], "exact"
    t = thumb(bgr)
    d = [np.abs(x - t).mean() for x in thumbs]
    j = int(np.argmin(d))
    return j, f"nearest(diff={d[j]:.1f})"


# --- 2. raw seek accuracy -----------------------------------------------------
print("\n-- cap.set(POS_FRAMES, n); read(): requested -> real --")
cap = cv2.VideoCapture(path)
grid = sorted({0, real // 4, real // 2, WANT - 10, WANT, WANT + 10, 3 * real // 4, real - 6})
for n in grid:
    if not 0 <= n < real:
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, f = cap.read()
    if not ok:
        print(f"  {n:5d} -> unreadable")
        continue
    j, how = which(f)
    print(f"  {n:5d} -> {j:5d}  off={j - n:+d}  {how}")
cap.release()

# --- 3. the app's FrameSource in different scrub orders -----------------------
print(f"\n-- FrameSource.get_frame({TARGET}) (source {WANT}) by access order --")
orders = {
    "jump straight to target":           [TARGET],
    "jump to target-36, scrub fwd":      list(range(max(0, TARGET - 36), TARGET + 1)),
    "jump to target-6, scrub fwd":       list(range(max(0, TARGET - 6), TARGET + 1)),
    "jump to target+4, scrub back":      list(range(TARGET + 4, TARGET - 1, -1)),
    "play from 0 to target":             list(range(0, TARGET + 1)),
    "target, far away, back to target":  [TARGET, TARGET // 2, TARGET],
}
for name, order in orders.items():
    fs = FrameSource(path, STRIDE, cache_mb=64)
    got = None
    for i in order:
        if i >= fs.n_frames:
            continue
        fs._cache.clear(); fs._cache_bytes = 0  # defeat the RAM LRU: force a decode per step
        rgb = fs.get_frame(i)
        if i == TARGET:
            got = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    j, how = which(got)
    print(f"  {name:34s} -> real frame {j:5d} (wanted {WANT}, off={j - WANT:+d}) {how}")
    fs.close()
