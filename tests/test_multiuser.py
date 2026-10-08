"""Shared-workspace behaviour (§5): per-video session cache, concurrent writes,
last-writer-wins, header attribution, the auth gate, and a storage round-trip.

All against the fake tracker and the local/SQLite backend (no GPU, no Postgres).
"""

import json
import threading

from fastapi.testclient import TestClient

import kumo_track.annotate.app as app_module
from kumo_track.annotate.app import create_app

from .conftest import FakeTracker, make_clip


# --- §5.1 no cross-eviction ----------------------------------------------------


def test_two_clips_coexist_no_cross_eviction(client, tmp_path):
    make_clip(tmp_path / "clip2.avi")  # a second clip in the same video dir
    a = client.post("/api/open", json={"name": "clip.avi", "stride": 4}).json()
    b = client.post("/api/open", json={"name": "clip2.avi", "stride": 4}).json()
    assert a["video_id"] != b["video_id"]  # distinct sessions

    # Opening B must not have evicted A: operating on A still works.
    oid = client.post("/api/objects", json={"video_id": a["video_id"], "label": "bag"}).json()["obj_id"]
    seg = client.post("/api/segment", json={
        "video_id": a["video_id"], "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    assert seg.status_code == 200, seg.text
    assert client.get(f"/api/frame/{a['video_id']}/2.jpg").status_code == 200


def test_inflight_propagate_survives_over_capacity_open(tmp_path, clip, monkeypatch):
    # The eviction regression: LRU victims are picked by recency, but an in-flight
    # /propagate stream doesn't re-touch its session — with the cache full, another
    # user opening one more clip used to close the streaming session's VideoCapture
    # mid-flight. The stream pins its session; the open must overflow the cap
    # instead of killing the stream.
    monkeypatch.setattr(app_module, "MAX_ACTIVE_SESSIONS", 1)
    started, release = threading.Event(), threading.Event()

    class GatedTracker(FakeTracker):
        def propagate(self, start, reverse, max_steps, seeds):
            started.set()
            release.wait(timeout=10)  # hold the stream mid-flight
            yield from super().propagate(start, reverse, max_steps, seeds)

    app = create_app(
        tracker_factory=GatedTracker, load_on_start=False,
        db_path=str(tmp_path / "pin.db"), video_dir=clip.parent,
        config={"edit_tool": "polygon"},
    )
    make_clip(clip.parent / "clip2.avi")
    with TestClient(app) as client:
        a = client.post("/api/open", json={"name": "clip.avi", "stride": 4}).json()
        oid = client.post("/api/objects",
                          json={"video_id": a["video_id"], "label": "bag"}).json()["obj_id"]
        client.post("/api/segment", json={
            "video_id": a["video_id"], "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30]})

        # TestClient buffers a streaming body inside the request call, so the
        # concurrent open has to come from another thread while the stream is
        # gated open mid-flight.
        results: dict = {}

        def concurrent_open():
            if not started.wait(5):
                results["started"] = False
                return
            results["started"] = True
            # Cache cap is 1 and A is the only (LRU) entry — this open would
            # evict + close A without the pin.
            results["open"] = client.post(
                "/api/open", json={"name": "clip2.avi", "stride": 4}).status_code
            release.set()

        t = threading.Thread(target=concurrent_open)
        t.start()
        try:
            with client.stream("POST", "/api/propagate", json={
                    "video_id": a["video_id"], "start_frame_idx": 2, "n_frames": 3}) as r:
                assert r.status_code == 200
                frames = [json.loads(line) for line in r.iter_lines() if line]
        finally:
            release.set()  # never leave the gated tracker blocking the threadpool
            t.join(15)

        assert results.get("started") is True, "stream never reached the tracker"
        assert results.get("open") == 200
        assert frames, "stream produced no output"
        assert not [f for f in frames if "error" in f], frames
        # A is still open and usable after the stream (it was never evicted).
        assert client.get(f"/api/frame/{a['video_id']}/2.jpg").status_code == 200


# --- §5.2 shared video, concurrent writes --------------------------------------


def test_shared_video_two_users_both_persist(opened):
    client, info = opened
    vid = info["video_id"]
    # Two users annotate different objects/frames of the SAME video.
    a = client.post("/api/objects", json={"video_id": vid, "label": "a"},
                    headers={"X-Kumo-User": "alice"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": a, "box": [10, 10, 30, 30]})
    b = client.post("/api/objects", json={"video_id": vid, "label": "b"},
                    headers={"X-Kumo-User": "bob"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 5, "obj_id": b, "box": [40, 10, 60, 30]})

    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert str(a) in ann["frames"]["2"] and str(b) in ann["frames"]["5"]
    by_id = {o["id"]: o["created_by"] for o in ann["objects"]}
    assert by_id[a] == "alice" and by_id[b] == "bob"


# --- §5.3 same object+frame: last-writer-wins ----------------------------------


def test_same_object_and_frame_is_last_writer_wins(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    first = [[1, 1], [9, 1], [9, 9], [1, 9]]
    second = [[2, 2], [8, 2], [8, 8], [2, 8]]
    r1 = client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 3, "corners": first})
    r2 = client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 3, "corners": second})
    assert r1.status_code == 200 and r2.status_code == 200  # no error, no corruption

    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert ann["frames"]["3"][str(oid)]["corners"] == second  # the last write persisted


# --- §5.4 attribution ----------------------------------------------------------


def test_created_by_from_header_body_value_ignored(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post(
        "/api/objects",
        json={"video_id": vid, "label": "bag", "created_by": "attacker"},  # body is untrusted
        headers={"X-Kumo-User": "real-user"},
    ).json()["obj_id"]
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    obj = next(o for o in ann["objects"] if o["id"] == oid)
    assert obj["created_by"] == "real-user"


def test_created_by_defaults_to_dev_user_without_header(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    obj = next(o for o in ann["objects"] if o["id"] == oid)
    assert obj["created_by"] == "dev"  # auth off → default dev user


# --- §5.5 auth gate ------------------------------------------------------------


def test_auth_gate_rejects_anonymous_when_enabled(tmp_path, clip):
    app = create_app(
        tracker_factory=FakeTracker, load_on_start=False,
        db_path=str(tmp_path / "auth.db"), video_dir=clip.parent,
        config={"edit_tool": "polygon"}, require_auth=True,
    )
    with TestClient(app) as c:
        assert c.get("/api/videos").status_code == 401  # missing X-Kumo-User
        assert c.get("/api/videos", headers={"X-Kumo-User": "alice"}).status_code == 200


def test_health_is_gate_exempt_even_with_auth_on(tmp_path, clip):
    # Orchestrator probes send no X-Kumo-User; health must still be 200 or the
    # container never becomes healthy with REQUIRE_AUTH=on.
    app = create_app(
        tracker_factory=FakeTracker, load_on_start=False,
        db_path=str(tmp_path / "auth.db"), video_dir=clip.parent,
        config={"edit_tool": "polygon"}, require_auth=True,
    )
    with TestClient(app) as c:
        for path in ("/api/health", "/healthz"):
            r = c.get(path)  # no identity header
            assert r.status_code == 200, (path, r.text)
            assert r.json()["status"] == "ok"


def test_proxy_secret_gates_every_request(tmp_path, clip):
    # With a proxy secret configured, X-Kumo-User alone is NOT enough — it is a
    # client-writable header; only requests proving they came through the gateway
    # (X-Kumo-Proxy-Auth) are accepted. Health probes stay exempt.
    app = create_app(
        tracker_factory=FakeTracker, load_on_start=False,
        db_path=str(tmp_path / "auth.db"), video_dir=clip.parent,
        config={"edit_tool": "polygon"}, require_auth=True,
        proxy_secret="gateway-only-secret",
    )
    with TestClient(app) as c:
        r = c.get("/api/videos", headers={"X-Kumo-User": "attacker"})
        assert r.status_code == 401  # forged identity, no proxy proof
        r = c.get("/api/videos",
                  headers={"X-Kumo-User": "attacker", "X-Kumo-Proxy-Auth": "guess"})
        assert r.status_code == 401  # wrong secret
        r = c.get("/api/videos",
                  headers={"X-Kumo-User": "alice",
                           "X-Kumo-Proxy-Auth": "gateway-only-secret"})
        assert r.status_code == 200  # the gateway's own requests pass
        for path in ("/api/health", "/healthz"):
            assert c.get(path).status_code == 200  # probes carry no secret


# --- §5.7 upload of an existing name is rejected -------------------------------


def test_upload_rejects_duplicate_name(opened):
    # clip.avi is already present (the `clip` fixture). Re-uploading under the same
    # name would rebind its annotations to new footage — reject with 409 instead.
    client, _ = opened
    up = client.post(
        "/api/upload",
        files={"file": ("clip.avi", b"not-a-real-clip", "video/x-msvideo")},
    )
    assert up.status_code == 409, up.text


# --- §5.6 storage round-trip ---------------------------------------------------


def test_storage_roundtrip_upload_list_open_export(client, tmp_path, monkeypatch):
    src = tmp_path / "_src.avi"
    make_clip(src)
    with src.open("rb") as fh:
        up = client.post("/api/upload", files={"file": ("fresh.avi", fh, "video/x-msvideo")})
    assert up.status_code == 200 and up.json()["name"] == "fresh.avi"

    assert "fresh.avi" in client.get("/api/videos").json()["videos"]

    info = client.post("/api/open", json={"name": "fresh.avi", "stride": 4}).json()
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30]})

    monkeypatch.chdir(tmp_path)
    res = client.post("/api/export", json={
        "video_id": vid, "out_name": "ds", "include_images": True}).json()
    assert res["n_images"] == 1 and res["n_boxes"] == 1
    assert (tmp_path / "data" / "ds" / "labels.json").exists()
    assert (tmp_path / "data" / "ds" / "images").exists()
