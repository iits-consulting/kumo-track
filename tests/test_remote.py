"""Remote SAM3 drop-ins: contract mapping against a stubbed client (no network).

These lock the transport-swap glue: NDJSON propagate → ``(frame_idx, {int: fit})``,
start-frame exclusion, ``removed``-set filtering, 404→reopen recovery, and the
detector's ``detect_image`` shape. No torch, no GPU, no HTTP.
"""

import httpx
import pytest
from PIL import Image

from kumo_track.annotate.remote import (
    RemoteSAM3Detector,
    RemoteTrackerManager,
    SessionNotFound,
    _raise_for_status,
)

FIT = {"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "polygon": None,
       "mask_rle": {"size": [2, 2], "counts": [1, 3]}, "score": None}


class StubClient:
    def __init__(self):
        self.opened = 0
        self.deleted = []
        self.segment_calls = []
        self.propagate_calls = []
        self.detect_calls = []
        self.segment_404_times = 0
        self.propagate_records = []
        self.detections = []

    def open_session(self, data, name, stride):
        self.opened += 1
        return {"session_id": f"sid-{self.opened}", "n_frames": 10}

    def segment(self, sid, frame_idx, obj_id, box=None, points=None, labels=None):
        self.segment_calls.append((sid, frame_idx, obj_id, box, points, labels))
        if self.segment_404_times > 0:
            self.segment_404_times -= 1
            raise SessionNotFound("gone")
        return FIT

    def propagate(self, sid, start, reverse, max_steps, seeds):
        self.propagate_calls.append((sid, start, reverse, max_steps, seeds))
        yield from self.propagate_records

    def detect(self, image_bytes, prompts, threshold, box_mode="obb", filename="frame.jpg"):
        self.detect_calls.append((prompts, threshold, box_mode))
        return self.detections

    def delete_session(self, sid):
        self.deleted.append(sid)


class FS:
    def __init__(self, path):
        self.path = str(path)
        self.name = "clip.mp4"
        self.stride = 5


@pytest.fixture
def fs(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake-video-bytes")
    return FS(p)


def test_segment_opens_session_once_and_returns_fit(fs):
    client = StubClient()
    mgr = RemoteTrackerManager(fs, client=client)
    assert mgr.segment(0, 1, box=[1, 2, 3, 4]) == FIT
    assert mgr.segment(0, 2, box=[5, 6, 7, 8]) == FIT
    assert client.opened == 1  # uploaded once, reused
    assert client.segment_calls[0][:3] == ("sid-1", 0, 1)


def test_propagate_maps_keys_and_skips_start(fs):
    client = StubClient()
    client.propagate_records = [
        {"frame_idx": 0, "objects": {"1": FIT}},   # start frame — must be skipped
        {"frame_idx": 1, "objects": {"1": FIT, "2": FIT}},
        {"frame_idx": 2, "objects": {"1": FIT}},
    ]
    mgr = RemoteTrackerManager(fs, client=client)
    out = list(mgr.propagate(0, reverse=False, max_steps=None, seeds={1: FIT, 2: FIT}))
    assert [fi for fi, _ in out] == [1, 2]                 # start frame excluded
    assert all(isinstance(k, int) for _, objs in out for k in objs)  # str keys → int
    assert out[0][1] == {1: FIT, 2: FIT}


def test_propagate_filters_removed_objects(fs):
    client = StubClient()
    client.propagate_records = [{"frame_idx": 1, "objects": {"1": FIT}}]
    mgr = RemoteTrackerManager(fs, client=client)
    mgr.forget_object(2)
    list(mgr.propagate(0, False, None, seeds={1: FIT, 2: FIT}))
    sent_seeds = client.propagate_calls[0][4]
    assert set(sent_seeds) == {1}            # removed obj 2 not sent as a seed
    mgr.unforget_object(2)
    list(mgr.propagate(0, False, None, seeds={1: FIT, 2: FIT}))
    assert set(client.propagate_calls[1][4]) == {1, 2}


def test_propagate_empty_seeds_is_noop(fs):
    client = StubClient()
    mgr = RemoteTrackerManager(fs, client=client)
    assert list(mgr.propagate(0, False, None, seeds={})) == []
    assert client.opened == 0  # never even uploads


def test_segment_reopens_on_session_lost(fs):
    client = StubClient()
    client.segment_404_times = 1
    mgr = RemoteTrackerManager(fs, client=client)
    assert mgr.segment(0, 1, box=[1, 2, 3, 4]) == FIT
    assert client.opened == 2                      # reopened after 404
    assert client.segment_calls[1][0] == "sid-2"   # retried on the new session


def test_close_deletes_session(fs):
    client = StubClient()
    mgr = RemoteTrackerManager(fs, client=client)
    mgr.segment(0, 1, box=[1, 2, 3, 4])
    mgr.close()
    assert client.deleted == ["sid-1"]
    mgr.close()  # idempotent
    assert client.deleted == ["sid-1"]


def test_detector_detect_image(fs):
    client = StubClient()
    client.detections = [
        {"corners": [[0, 0], [4, 0], [4, 4], [0, 4]], "score": 0.8, "label": "truck"},
        {"corners": None},  # dropped
    ]
    det = RemoteSAM3Detector(client=client, box_mode="aabb")
    det.text_prompts = ["truck"]
    det.threshold = 0.42
    results = det.detect_image(Image.new("RGB", (8, 8)))
    assert len(results) == 1
    assert results[0].label == "truck" and results[0].score == 0.8
    prompts, threshold, box_mode = client.detect_calls[0]
    assert prompts == "truck" and threshold == 0.42 and box_mode == "aabb"


def test_raise_for_status_maps_codes():
    _raise_for_status(httpx.Response(200))  # no raise
    with pytest.raises(SessionNotFound):
        _raise_for_status(httpx.Response(404, text="session not found"))
    with pytest.raises(RuntimeError, match="507"):
        _raise_for_status(httpx.Response(507, text="CUDA OOM"))
    with pytest.raises(RuntimeError, match="401"):
        _raise_for_status(httpx.Response(401, text="bad key"))
