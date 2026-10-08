"""On-demand video frame access for the annotation app.

Replaces the old eager "decode the whole clip into RAM" model so 20-minute clips
open instantly and stay memory-bounded. A :class:`FrameSource` exposes the clip as
a list of *sampled* frames (every ``stride``-th source frame) and decodes each one
only when asked. Two caches keep it fast:

* a byte-budgeted **RAM LRU** of decoded RGB frames (``FRAME_CACHE_MB``), used by
  tracking-window construction and export;
* a **JPEG disk cache** (``outputs/cache/<stem>-s<stride>-pts/``) for the frame-serving
  HTTP endpoint — source files are immutable, so cached JPEGs never invalidate.

Index conventions used throughout the app:

* *sampled idx* ``i`` — index into this source's frame list (``0 .. n_frames-1``);
* *source idx* — the original frame number in the file, ``i * stride``.

A single :class:`cv2.VideoCapture` is shared and guarded by a lock; near-forward
reads ``grab()`` ahead (cheap), far jumps seek and then *verify* where they landed.

The recordings are variable frame rate, and OpenCV's ``CAP_PROP_POS_FRAMES`` seeks
by time (``n / fps``), so on them it lands tens of frames off — differently per
scrub path. We therefore index the pts of every frame once per clip (one grab()
pass) and make every seek land by pts, so sampled ``i`` is always source frame
``i * stride`` counted from 0 (the contract the DB, exports and SAM3 rely on).
"""

import bisect
import os
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

CACHE_ROOT = Path(os.environ.get("FRAME_CACHE_DIR", "outputs/cache"))
_DEFAULT_CACHE_MB = float(os.environ.get("FRAME_CACHE_MB", "2048"))
# Sampled frames the container may over-declare before a clip counts as truncated.
_TAIL_SLACK = 5
# Source-frame gap below which we grab() forward instead of seeking. A forward
# scrub of one sampled frame is `stride` source frames, so this covers normal
# scrubbing without ever paying a keyframe seek.
_SEEK_GAP = 240


class FrameSource:
    """Lazily-decoded, cached view of one clip sampled every ``stride`` frames.

    Thread-safe: all VideoCapture access and cache mutation happen under a lock,
    so the HTTP frame endpoint, tracking, and export can share one instance.
    """

    def __init__(self, path: str, stride: int, name: str | None = None, cache_mb: float | None = None):
        self.path = str(path)
        self.stride = max(1, int(stride))
        self.name = name or Path(self.path).name
        self._lock = threading.RLock()
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"could not open video: {self.path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._declared = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._cursor: int | None = None  # next source idx the cap will read

        # RAM LRU of decoded RGB frames, byte-budgeted.
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_bytes = 0
        budget_mb = cache_mb if cache_mb is not None else _DEFAULT_CACHE_MB
        self._budget = int(budget_mb * 1024 * 1024)

        stem = Path(self.name).stem
        # `-pts`: cache dirs written by the pre-pts-seek code hold wrong pictures.
        self._disk = CACHE_ROOT / f"{stem}-s{self.stride}-pts"

        pts = self._index()
        self._n_real = len(pts)
        # pts-guided seeking needs monotonic timestamps; otherwise _land() decodes from 0.
        self._pts = pts if all(a < b for a, b in zip(pts, pts[1:])) else None
        self.n_frames, self.truncated_at = self._probe_length()
        if self.width <= 0 or self.height <= 0:
            first = self.get_frame(0)
            self.height, self.width = first.shape[:2]

    # --- index mapping ---------------------------------------------------------

    def source_index(self, sampled_idx: int) -> int:
        """Original file frame number for sampled index ``sampled_idx``."""
        return sampled_idx * self.stride

    @property
    def source_indices(self) -> list[int]:
        """The original frame number of every sampled frame (for the DB/export)."""
        return [i * self.stride for i in range(self.n_frames)]

    # --- length / truncation ---------------------------------------------------

    def _index(self) -> list[float]:
        """pts (ms) of every real frame in decode order — one grab() pass (~3 s per
        10-min clip). Its length is the true frame count; containers over-declare."""
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            pts = []
            while self._cap.grab():
                pts.append(self._cap.get(cv2.CAP_PROP_POS_MSEC))
            self._cursor = None
        if not pts:
            raise ValueError(f"no frames decoded from {self.path}")
        return pts

    def _probe_length(self) -> tuple[int, int | None]:
        """``(n_frames, truncated_at)``: real sampled count vs. the container's claim."""
        n_sampled = (self._n_real + self.stride - 1) // self.stride
        declared = (self._declared + self.stride - 1) // self.stride
        # ponytail: some recorders declare exactly one frame more than it holds.
        # That is metadata slack, not corruption; only flag (→ UI "looks corrupt" toast)
        # when a real chunk of the clip is gone.
        if declared - n_sampled <= _TAIL_SLACK:
            return n_sampled, None
        print(
            f"[annotate] WARNING: {self.path} is corrupt/truncated past source frame "
            f"~{self._n_real}; recovered {n_sampled} of {declared} sampled frame(s)."
        )
        return n_sampled, self._n_real

    def _land(self, target: int) -> None:
        """Seek so that ``_cursor`` is truthful and ``<= target`` (assumes the lock is held).

        OpenCV lands near *time* ``n / fps``, so ask for the frame number whose time is
        the target's pts, aim a little early, and check by pts which real frame came out;
        landing late (or past EOF) → aim earlier. Grab-forward from there is exact.
        """
        cap = self._cap
        pts = self._pts
        if target == 0 or pts is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # seeking to the start is always exact
            self._cursor = 0
            return
        guess = int(pts[target] / 1000.0 * self.fps)
        back = 2
        while True:
            at = max(0, guess - back)
            cap.set(cv2.CAP_PROP_POS_FRAMES, at)
            landed = target  # a failed grab means we ran past EOF: too late
            if cap.grab():
                t = cap.get(cv2.CAP_PROP_POS_MSEC)
                k = bisect.bisect_left(pts, t)  # nearest pts (frames 0/1 can be 11 µs apart)
                landed = k - 1 if k and (k == len(pts) or pts[k] - t > t - pts[k - 1]) else k
            if landed < target:
                self._cursor = landed + 1
                return
            if at == 0:  # even from the start we "landed late": pts are untrustworthy
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._cursor = 0
                return
            back = max(back * 2, landed - target + 2)  # aim past the observed lateness

    # --- decode ----------------------------------------------------------------

    def _decode(self, sampled_idx: int) -> np.ndarray:
        """Decode one sampled frame to RGB (assumes the lock is held)."""
        target = sampled_idx * self.stride
        cap = self._cap
        if not (self._cursor is not None and self._cursor <= target < self._cursor + _SEEK_GAP):
            self._land(target)
        while self._cursor < target:
            if not cap.grab():
                self._cursor = None
                raise RuntimeError(f"frame {sampled_idx} (source {target}) is unreadable")
            self._cursor += 1
        ok, bgr = cap.read()
        self._cursor = target + 1
        if not ok or bgr is None:
            self._cursor = None
            raise RuntimeError(f"frame {sampled_idx} (source {target}) is unreadable")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _put(self, sampled_idx: int, frame: np.ndarray) -> None:
        """Insert into the RAM LRU and evict oldest frames over the byte budget."""
        if sampled_idx in self._cache:
            self._cache_bytes -= self._cache[sampled_idx].nbytes
        self._cache[sampled_idx] = frame
        self._cache.move_to_end(sampled_idx)
        self._cache_bytes += frame.nbytes
        while self._cache_bytes > self._budget and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes

    def get_frame(self, sampled_idx: int) -> np.ndarray:
        """Return the sampled frame as a full-resolution RGB uint8 array (cached)."""
        if sampled_idx < 0 or sampled_idx >= self.n_frames:
            raise IndexError(f"frame {sampled_idx} out of range (0..{self.n_frames - 1})")
        with self._lock:
            hit = self._cache.get(sampled_idx)
            if hit is not None:
                self._cache.move_to_end(sampled_idx)
                return hit
            frame = self._decode(sampled_idx)
            self._put(sampled_idx, frame)
            return frame

    def get_frames(self, start: int, end: int) -> list[np.ndarray]:
        """RGB frames for the inclusive sampled range ``[start, end]`` (for windows)."""
        return [self.get_frame(i) for i in range(start, end + 1)]

    def frame_bgr(self, sampled_idx: int) -> np.ndarray:
        """Full-resolution BGR frame for export via ``cv2.imwrite``."""
        return cv2.cvtColor(self.get_frame(sampled_idx), cv2.COLOR_RGB2BGR)

    # --- JPEG (disk-cached) for the HTTP frame endpoint ------------------------

    def jpeg(self, sampled_idx: int, quality: int = 80) -> bytes:
        """JPEG-encode a sampled frame, caching the bytes on disk (immutable)."""
        if sampled_idx < 0 or sampled_idx >= self.n_frames:
            raise IndexError(f"frame {sampled_idx} out of range (0..{self.n_frames - 1})")
        path = self._disk / f"{sampled_idx}-q{quality}.jpg"
        if path.exists():
            return path.read_bytes()
        bgr = self.frame_bgr(sampled_idx)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        data = buf.tobytes()
        self._disk.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic — concurrent endpoint workers never see a partial file
        return data

    # --- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._cache.clear()
            self._cache_bytes = 0
