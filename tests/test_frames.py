"""FrameSource: index mapping, decode correctness (grab + seek paths), LRU."""

import cv2
import numpy as np

from kumo_track.annotate.frames import FrameSource
from tests.conftest import CLIP_N


def _value(frame: np.ndarray) -> int:
    return int(round(float(frame.mean())))


def test_index_mapping_and_length(clip):
    fs = FrameSource(str(clip), stride=4)
    assert fs.n_frames == (CLIP_N + 3) // 4  # ceil(40/4) = 10
    assert fs.source_index(0) == 0
    assert fs.source_index(3) == 12
    assert fs.source_indices[:3] == [0, 4, 8]
    assert fs.truncated_at is None
    assert (fs.width, fs.height) == (64, 48)
    fs.close()


def test_probe_tolerates_overdeclared_tail(clip):
    """Recorders over-declare the frame count by a few frames: that is slack, not
    corruption. Only a real missing chunk sets ``truncated_at``."""
    fs = FrameSource(str(clip), stride=4)
    n = fs.n_frames  # 10 sampled frames really decode
    fs._declared = CLIP_N + 3 * 4  # 3 sampled frames short → within slack
    assert fs._probe_length() == (n, None)
    fs._declared = CLIP_N + 8 * 4  # 8 sampled frames short → flagged
    assert fs._probe_length() == (n, CLIP_N)
    fs.close()


def test_decode_content_forward_and_seek(clip):
    fs = FrameSource(str(clip), stride=4)
    # Forward scrub exercises the grab() path; each sampled frame's value == source idx.
    for i in range(fs.n_frames):
        assert abs(_value(fs.get_frame(i)) - i * 4) <= 2, f"frame {i}"
    # Jump backward exercises the seek path.
    assert abs(_value(fs.get_frame(0)) - 0) <= 2
    assert abs(_value(fs.get_frame(7)) - 28) <= 2
    fs.close()


def test_cache_hit_returns_same_object(clip):
    fs = FrameSource(str(clip), stride=4)
    a = fs.get_frame(2)
    b = fs.get_frame(2)
    assert a is b  # served from RAM cache, not re-decoded
    fs.close()


def test_lru_eviction_respects_byte_budget(clip):
    # ~9 KB per frame; a 0.01 MB budget holds exactly one frame.
    fs = FrameSource(str(clip), stride=4, cache_mb=0.01)
    for i in range(5):
        fs.get_frame(i)
    assert len(fs._cache) == 1
    assert fs._cache_bytes <= fs._budget
    fs.close()


def test_jpeg_disk_cache(clip):
    fs = FrameSource(str(clip), stride=4)
    data1 = fs.jpeg(3, quality=80)
    assert data1[:2] == b"\xff\xd8"  # JPEG SOI marker
    cached = fs._disk / "3-q80.jpg"
    assert cached.exists()
    assert fs.jpeg(3, quality=80) == data1  # second call hits disk
    fs.close()


def test_vfr_frame_identity_independent_of_scrub_order(vfr_clip):
    """On VFR clips a raw POS_FRAMES seek lands up to dozens of frames off, early or
    late. get_frame(i) must be source frame i*stride whichever way it is reached."""
    cap = cv2.VideoCapture(str(vfr_clip))
    seq = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        seq.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(seq) == 159
    stride = 5
    n = (len(seq) + stride - 1) // stride
    # Fresh source + one-frame cache per order: every step is a real decode.
    orders = [list(range(n)), list(range(n - 1, -1, -1))]          # play fwd; scrub back
    orders += [[i] for i in range(n)]                               # jump straight to i
    orders += [list(range(max(0, i - 8), i + 1)) for i in range(n)]  # jump to i-8, scrub to i
    for order in orders:
        fs = FrameSource(str(vfr_clip), stride=stride, cache_mb=0.01)
        assert fs.n_frames == n and fs.truncated_at is None
        for i in order:
            assert np.array_equal(fs.get_frame(i), seq[i * stride]), f"frame {i} via {order[:3]}..."
        fs.close()
