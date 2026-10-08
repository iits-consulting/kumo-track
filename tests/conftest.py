"""Shared fixtures: a synthetic clip + a GPU-free fake tracker.

Tests must run without SAM3 weights or a GPU, so we generate a tiny clip with
``cv2.VideoWriter`` (MJPG/AVI — every frame is a keyframe, so frame-index mapping
is exact) and inject :class:`FakeTracker` via ``create_app(tracker_factory=...)``.
Each sampled frame is filled with a constant equal to its *source* frame index, so
tests can assert that ``get_frame(i)`` decoded the right frame.
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kumo_track.annotate import rle
from kumo_track.annotate.app import create_app
from kumo_track.masks import mask_target_size

CLIP_W, CLIP_H = 64, 48
CLIP_N = 40  # source frames; keep < 256 so frame index fits in a pixel value


def make_clip(path: Path, n_frames: int = CLIP_N, w: int = CLIP_W, h: int = CLIP_H, fps: int = 10):
    """Write an AVI where source frame i is a solid image of intensity i."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open (no MJPG support?)")
    for i in range(n_frames):
        vw.write(np.full((h, w, 3), i, dtype=np.uint8))
    vw.release()


class FakeTracker:
    """Deterministic, GPU-free stand-in for TrackerManager.

    ``segment`` returns a box from the prompt; ``propagate`` walks frames in the
    requested direction copying each seed's box. Same interface the app uses.
    """

    def __init__(self, frame_source):
        self.fs = frame_source
        self.removed: set[int] = set()

    @staticmethod
    def _corners(box):
        x1, y1, x2, y2 = box
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def _mask_rle(self, corners):
        """Rasterise the box rectangle into a ds-res RLE, mirroring the real _fit."""
        ds_h, ds_w = mask_target_size(self.fs.height, self.fs.width)
        binary = rle.rasterize_polygon(corners, ds_h, ds_w, self.fs.height, self.fs.width)
        return rle.encode(binary)

    def segment(self, frame_idx, obj_id, points=None, labels=None, box=None):
        if box:
            c = self._corners(box)
        elif points:
            x, y = points[0]
            c = self._corners([x - 5, y - 5, x + 5, y + 5])
        else:
            return None
        return {"corners": c, "polygon": c, "mask_rle": self._mask_rle(c), "score": None}

    def propagate(self, start, reverse, max_steps, seeds):
        seeds = {o: a for o, a in seeds.items() if o not in self.removed and a}
        if not seeds:
            return
        step = -1 if reverse else 1
        i = start + step
        emitted = 0
        while 0 <= i < self.fs.n_frames:
            if max_steps is not None and emitted >= max_steps:
                break
            boxes = {
                oid: {"corners": a["corners"], "polygon": a.get("polygon"),
                      "mask_rle": a.get("mask_rle"), "score": None}
                for oid, a in seeds.items()
            }
            yield i, boxes
            emitted += 1
            i += step

    def forget_object(self, obj_id):
        self.removed.add(obj_id)

    def unforget_object(self, obj_id):
        self.removed.discard(obj_id)

    def close(self):
        pass


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.avi"
    make_clip(p)
    return p


@pytest.fixture
def vfr_clip(tmp_path):
    """A *variable-frame-rate* h264 clip like many camera recordings: 200 source frames at
    2 fps with frames 30..70 dropped (clustered — a uniform drop pattern does not
    reproduce the seek bug). Real frame p has intensity p if p < 30 else p + 41."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed (cv2.VideoWriter cannot write VFR)")
    cfr = tmp_path / "cfr.avi"
    make_clip(cfr, n_frames=200, fps=2)
    vfr = tmp_path / "vfr.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(cfr), "-vf", "select='not(between(n\\,30\\,70))'",
         "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", "-crf", "18", str(vfr)],
        check=True,
    )
    return vfr


@pytest.fixture
def client(tmp_path, clip):
    # Pin polygon mode so the suite is independent of the repo's config.toml; brush
    # mode is exercised through the dedicated `brush_opened` fixture.
    app = create_app(
        tracker_factory=FakeTracker,
        load_on_start=False,
        db_path=str(tmp_path / "test.db"),
        video_dir=clip.parent,
        config={"edit_tool": "polygon"},
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture
def opened(client):
    """A client with the clip opened; returns (client, open_response_json)."""
    r = client.post("/api/open", json={"name": "clip.avi", "stride": 4})
    assert r.status_code == 200, r.text
    return client, r.json()


@pytest.fixture
def brush_opened(tmp_path, clip):
    """Like `opened`, but the app runs in brush mode (edit_tool='brush')."""
    app = create_app(
        tracker_factory=FakeTracker,
        load_on_start=False,
        db_path=str(tmp_path / "brush.db"),
        video_dir=clip.parent,
        config={"edit_tool": "brush"},
    )
    with TestClient(app) as c:
        r = c.post("/api/open", json={"name": "clip.avi", "stride": 4})
        assert r.status_code == 200, r.text
        yield c, r.json()
