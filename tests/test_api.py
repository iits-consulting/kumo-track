"""End-to-end API flows against a fake tracker: open, segment, propagate,
cancellation-shape, manual precedence, static export."""

import json

import pytest

import kumo_track.annotate.app as app_module

from .conftest import make_clip


def _ndjson(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_open_returns_metadata(opened):
    _, info = opened
    assert info["n_frames"] == 10
    assert (info["width"], info["height"]) == (64, 48)
    assert info["truncated_at"] is None
    assert info["annotations"]["objects"] == []
    assert info["stride"] == 4


def test_frame_endpoint_is_cacheable(opened):
    client, info = opened
    r = client.get(f"/api/frame/{info['video_id']}/3.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert "immutable" in r.headers["cache-control"]
    assert r.content[:2] == b"\xff\xd8"


def test_segment_then_propagate_persists(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    seg = client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30],
    }).json()
    assert seg["corners"]

    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 2, "direction": "fwd", "n_frames": 3,
    })
    assert r.status_code == 200
    lines = _ndjson(r)
    assert [ln["frame_idx"] for ln in lines] == [3, 4, 5]  # start frame not re-emitted

    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert ann["frames"]["2"][str(oid)]["origin"] == "seed"
    assert ann["frames"]["5"][str(oid)]["origin"] == "propagated"


def test_propagate_without_seed_is_409(opened):
    client, info = opened
    r = client.post("/api/propagate", json={
        "video_id": info["video_id"], "start_frame_idx": 1, "direction": "fwd",
    })
    assert r.status_code == 409


def test_session_rebuilds_after_eviction_instead_of_409(opened, tmp_path, monkeypatch):
    # The evicted-session bug: the DB row for `vid` survives, so a cache miss
    # (LRU eviction, or a restart) must rebuild the session instead of 409ing on
    # every subsequent frame/annotation/segment call.
    client, info = opened
    vid = info["video_id"]

    monkeypatch.setattr(app_module, "MAX_ACTIVE_SESSIONS", 1)
    make_clip(tmp_path / "clip2.avi")
    r2 = client.post("/api/open", json={"name": "clip2.avi", "stride": 4})
    assert r2.status_code == 200, r2.text  # evicts vid's session (cap=1, vid is LRU)

    frame = client.get(f"/api/frame/{vid}/2.jpg")
    assert frame.status_code == 200
    assert frame.content[:2] == b"\xff\xd8"

    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    put = client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 2,
        "corners": [[1, 1], [9, 1], [9, 9], [1, 9]],
    })
    assert put.status_code == 200, put.text

    # The async path (_pinned_session_async) rebuilds too, not just the sync one.
    seg = client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    assert seg.status_code == 200, seg.text

    # A video_id that never existed is a real 404, not a rebuild target.
    assert client.get("/api/frame/999999/0.jpg").status_code == 404


def test_manual_precedence_survives_propagation(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    manual_corners = [[1, 1], [9, 1], [9, 9], [1, 9]]
    client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 4, "corners": manual_corners,
    })

    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 2, "direction": "fwd", "n_frames": 5,
    })
    lines = {ln["frame_idx"]: ln for ln in _ndjson(r)}
    assert lines[4]["boxes"][str(oid)]["corners"] == manual_corners  # stream keeps manual

    ann = client.get(f"/api/annotations?video_id={vid}").json()
    f4 = ann["frames"]["4"][str(oid)]
    assert f4["origin"] == "manual"
    assert f4["corners"] == manual_corners


def test_manual_polygon_recomputes_obb(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    # Vertex edit: send polygon only → server derives the OBB corners.
    poly = [[0, 0], [10, 0], [10, 6], [0, 6]]
    r = client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 3, "polygon": poly,
    }).json()
    assert r["polygon"] == poly
    assert len(r["corners"]) == 4  # a real rotated box, not the polygon
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert ann["frames"]["3"][str(oid)]["origin"] == "manual"


def test_manual_box_transform_trusts_both(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    corners = [[2, 2], [12, 2], [12, 9], [2, 9]]
    poly = [[3, 3], [11, 3], [11, 8], [3, 8]]
    r = client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 3, "corners": corners, "polygon": poly,
    }).json()
    assert r["corners"] == corners and r["polygon"] == poly  # both kept as-is


def test_legacy_propagate_fields(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 5, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    # Old client shape: reverse + max_frames instead of direction + n_frames.
    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 5, "reverse": True, "max_frames": 2,
    })
    assert [ln["frame_idx"] for ln in _ndjson(r)] == [4, 3]


def test_track_seeds_selected_object_from_nearest_frame(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    # Track from frame 6 where the object has no box — selected, so it seeds itself.
    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 6, "direction": "fwd", "n_frames": 2,
        "active_obj": oid,
    })
    assert r.status_code == 200
    lines = {ln["frame_idx"]: ln for ln in _ndjson(r)}
    box = [[10, 10], [30, 10], [30, 30], [10, 30]]
    assert lines[6]["boxes"][str(oid)]["corners"] == box  # seed echoed on start frame
    assert lines[7]["boxes"][str(oid)]["corners"] == box  # then tracked onward

    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert ann["frames"]["6"][str(oid)]["origin"] == "seed"  # copy persisted as seed


def test_track_skips_unselected_boxless_objects(opened):
    client, info = opened
    vid = info["video_id"]
    a = client.post("/api/objects", json={"video_id": vid, "label": "a"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": a, "box": [10, 10, 30, 30],
    })
    b = client.post("/api/objects", json={"video_id": vid, "label": "b"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 6, "obj_id": b, "box": [40, 10, 60, 30],
    })
    # b is selected; a has no box on frame 6 and must NOT be dragged in.
    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 6, "direction": "fwd", "n_frames": 2,
        "active_obj": b,
    })
    lines = {ln["frame_idx"]: ln for ln in _ndjson(r)}
    assert str(b) in lines[7]["boxes"]
    assert str(a) not in lines[7]["boxes"]


def test_propagate_copies_static_boxes_alongside_tracking(opened):
    client, info = opened
    vid = info["video_id"]
    s = client.post("/api/objects", json={"video_id": vid, "label": "zone"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 0, "obj_id": s, "box": [0, 0, 20, 20],
    })
    client.patch(f"/api/objects/{s}", json={"static": True})
    n = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 1, "obj_id": n, "box": [10, 10, 30, 30],
    })

    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 1, "direction": "fwd", "n_frames": 2,
    })
    lines = {ln["frame_idx"]: ln for ln in _ndjson(r)}
    # The pinned box rides along on every tracked frame, same corners everywhere.
    zone = [[0, 0], [20, 0], [20, 20], [0, 20]]
    assert lines[2]["boxes"][str(s)]["corners"] == zone
    assert lines[3]["boxes"][str(s)]["corners"] == zone

    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert ann["frames"]["2"][str(s)]["corners"] == zone  # persisted, not just streamed


def test_propagate_with_only_static_objects_copies_without_tracker(opened):
    client, info = opened
    vid = info["video_id"]
    s = client.post("/api/objects", json={"video_id": vid, "label": "zone"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 0, "obj_id": s, "box": [0, 0, 20, 20],
    })
    client.patch(f"/api/objects/{s}", json={"static": True})

    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 0, "direction": "fwd", "n_frames": 5,
        "active_obj": s,  # selecting a static object must NOT seed it into SAM3
    })
    assert r.status_code == 200  # no non-static seed needed
    lines = {ln["frame_idx"]: ln for ln in _ndjson(r)}
    zone = [[0, 0], [20, 0], [20, 20], [0, 20]]
    assert sorted(lines) == [1, 2, 3, 4, 5]
    assert all(ln["boxes"][str(s)]["corners"] == zone for ln in lines.values())


def test_patch_hidden_and_color_round_trip(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    obj = client.get(f"/api/annotations?video_id={vid}").json()["objects"][0]
    assert obj["hidden"] is False and obj["color"] is None  # defaults exposed by the API

    client.patch(f"/api/objects/{oid}", json={"hidden": True, "color": "#ff8800"})
    obj = client.get(f"/api/annotations?video_id={vid}").json()["objects"][0]
    assert obj["hidden"] is True and obj["color"] == "#ff8800"


def test_static_toggle_preserves_annotations(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 1, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 1, "direction": "fwd", "n_frames": 3,
    })
    for static in (True, False):  # a pure mode toggle — annotations untouched
        client.patch(f"/api/objects/{oid}", json={"static": static})
        ann = client.get(f"/api/annotations?video_id={vid}").json()
        frames_with_box = sorted(k for k, v in ann["frames"].items() if str(oid) in v)
        assert frames_with_box == ["1", "2", "3", "4"]


def test_static_toggle_flow_matches_schematic(opened):
    """The intended workflow: track → make static (copies) → unmake (SAM3 again).

    |o---------|  seed frame 0
    |oooo------|  track 3 frames with SAM3
    |++++------|  make static: nothing deleted, just a mode flip
    |+++++++---|  track 3 more: box copied, no SAM3
    |ooooooo---|  user sees movement on frame 6 → make trackable again
    |oooooooooo|  track on: frame 6's box seeds SAM3
    """
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    box = [[10, 10], [30, 10], [30, 30], [10, 30]]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 0, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 0, "direction": "fwd", "n_frames": 3,
    })

    client.patch(f"/api/objects/{oid}", json={"static": True})
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert sorted(ann["frames"]) == ["0", "1", "2", "3"]  # nothing lost

    # Static tracking: the tracker is bypassed; frame 3's box is copied onward.
    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 3, "direction": "fwd", "n_frames": 3,
    })
    lines = {ln["frame_idx"]: ln for ln in _ndjson(r)}
    assert all(lines[fi]["boxes"][str(oid)]["corners"] == box for fi in (4, 5, 6))

    # Back to trackable: SAM3 takes over from the current frame's box.
    client.patch(f"/api/objects/{oid}", json={"static": False})
    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 6, "direction": "fwd", "n_frames": 3,
    })
    lines = {ln["frame_idx"]: ln for ln in _ndjson(r)}
    assert all(str(oid) in lines[fi]["boxes"] for fi in (7, 8, 9))
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert sorted(int(k) for k in ann["frames"]) == list(range(10))  # whole clip annotated


def test_recreated_object_id_tracks_again(opened):
    """SQLite reuses rowids: a new object landing on a deleted object's id must
    not inherit its tracker tombstone (forget_object) — it would silently never
    track until the clip is reopened."""
    client, info = opened
    vid = info["video_id"]
    old = client.post("/api/objects", json={"video_id": vid, "label": "junk"}).json()["obj_id"]
    client.delete(f"/api/objects/{old}")
    new = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    assert new == old  # rowid recycled

    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": new, "box": [10, 10, 30, 30],
    })
    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 2, "direction": "fwd", "n_frames": 2,
    })
    lines = _ndjson(r)
    assert lines and all(str(new) in ln["boxes"] for ln in lines)


class FakeSAM3Detector:
    """Stands in for models.sam3.SAM3Detector — one fixed detection per prompt."""

    def __init__(self, threshold=0.05, box_mode="aabb"):
        self.threshold = threshold
        self.text_prompts = ["object"]

    def detect_image(self, image):
        from kumo_track.base import DetectionResult
        return [DetectionResult(
            corners=[[10, 10], [30, 10], [30, 30], [10, 30]], score=0.9,
            label=self.text_prompts[0],
        )]


def test_find_all_objects_track(opened, monkeypatch):
    """Find-all created objects must track — including on a recycled rowid."""
    # Monkeypatching the detector still imports models.sam3, which pulls in torch
    # (the `local` extra). Skip cleanly on a CPU-only/base install, like test_tracker.
    pytest.importorskip("torch")
    import kumo_track.models.sam3 as sam3mod
    monkeypatch.setattr(sam3mod, "SAM3Detector", FakeSAM3Detector)
    client, info = opened
    vid = info["video_id"]
    old = client.post("/api/objects", json={"video_id": vid, "label": "junk"}).json()["obj_id"]
    client.delete(f"/api/objects/{old}")

    res = client.post("/api/find_all", json={
        "video_id": vid, "frame_idx": 2, "label": "kangaroo", "query": "kangaroo",
    }).json()
    assert len(res["created"]) == 1
    obj = res["created"][0]
    assert obj["id"] == old  # rowid recycled
    assert obj["label"] == "kangaroo"

    r = client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 2, "direction": "fwd", "n_frames": 2,
    })
    lines = _ndjson(r)
    assert lines and all(str(obj["id"]) in ln["boxes"] for ln in lines)


def test_export_labels_only(opened, tmp_path, monkeypatch):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    monkeypatch.chdir(tmp_path)
    res = client.post("/api/export", json={
        "video_id": vid, "out_name": "ds", "include_images": False,
    }).json()
    assert res["n_images"] == 1 and res["n_boxes"] == 1
    assert (tmp_path / "data" / "ds" / "labels.json").exists()
    assert not (tmp_path / "data" / "ds" / "images").exists()


def test_static_object_exports_from_its_real_rows(opened, tmp_path, monkeypatch):
    client, info = opened
    vid = info["video_id"]
    # Static object: seeded on frame 0, copied onto frames 2-3 by the propagation.
    s = client.post("/api/objects", json={"video_id": vid, "label": "zone"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 0, "obj_id": s, "box": [0, 0, 20, 20],
    })
    client.patch(f"/api/objects/{s}", json={"static": True})
    # Normal tracked object on frames 1..3.
    n = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 1, "obj_id": n, "box": [10, 10, 30, 30],
    })
    client.post("/api/propagate", json={
        "video_id": vid, "start_frame_idx": 1, "direction": "fwd", "n_frames": 2,
    })

    monkeypatch.chdir(tmp_path)
    res = client.post("/api/export", json={"video_id": vid, "out_name": "ds"}).json()
    labels = json.loads((tmp_path / "data" / "ds" / "labels.json").read_text())
    # Annotations export exactly where they exist — no expansion onto other frames:
    # frame 0: zone seed; frame 1: bag seed; frames 2-3: bag tracked + zone copied.
    assert res["n_images"] == 4
    per_frame = {f: sorted(row["label"] for row in rows) for f, rows in labels.items()}
    assert sorted(per_frame.values()) == [["bag"], ["bag", "zone"], ["bag", "zone"], ["zone"]]


# --- config + brush mask -------------------------------------------------------


def _mask_png(w, h, rect):
    """base64 RGBA PNG with `rect`=(x1,y1,x2,y2) painted opaque, rest transparent."""
    import base64

    import cv2
    import numpy as np

    img = np.zeros((h, w, 4), np.uint8)
    x1, y1, x2, y2 = rect
    img[y1:y2, x1:x2] = (255, 255, 255, 255)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def test_config_endpoint_returns_valid_tool(client):
    # snake_case key for the frontend; value constrained to the known tools.
    body = client.get("/api/config").json()
    assert body["edit_tool"] in ("polygon", "brush")


def test_load_config_reads_choice(tmp_path, monkeypatch):
    from kumo_track.annotate import app as app_module

    cfg = tmp_path / "config.toml"
    cfg.write_text('[annotation]\nedit_tool = "brush"\n')
    monkeypatch.setattr(app_module, "CONFIG_FILE", cfg)
    assert app_module.load_config() == {"edit_tool": "brush"}


def test_load_config_bad_or_missing_falls_back(tmp_path, monkeypatch):
    from kumo_track.annotate import app as app_module

    bad = tmp_path / "config.toml"
    bad.write_text('[annotation]\nedit_tool = "lasso"\n')
    monkeypatch.setattr(app_module, "CONFIG_FILE", bad)
    assert app_module.load_config() == {"edit_tool": "polygon"}
    monkeypatch.setattr(app_module, "CONFIG_FILE", tmp_path / "nope.toml")
    assert app_module.load_config() == {"edit_tool": "polygon"}


def test_brush_mask_creates_manual_annotation(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    # Paint a box at the clip's full resolution (64x48) → scale factor 1.
    png = _mask_png(64, 48, (10, 8, 40, 36))
    r = client.put("/api/mask", json={
        "video_id": vid, "object_id": oid, "frame_idx": 3, "mask_png": png,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["corners"]) == 4 and body["polygon"]
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    f3 = ann["frames"]["3"][str(oid)]
    assert f3["origin"] == "manual"
    xs = [p[0] for p in f3["corners"]]
    ys = [p[1] for p in f3["corners"]]
    assert 8 <= min(xs) <= 12 and 38 <= max(xs) <= 42  # ~bounds the painted region
    assert 6 <= min(ys) <= 10 and 34 <= max(ys) <= 38


def test_brush_empty_mask_deletes_annotation(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 3,
        "corners": [[1, 1], [9, 1], [9, 9], [1, 9]],
    })
    r = client.put("/api/mask", json={
        "video_id": vid, "object_id": oid, "frame_idx": 3,
        "mask_png": _mask_png(64, 48, (0, 0, 0, 0)),  # nothing painted
    })
    assert r.status_code == 200, r.text
    assert r.json()["corners"] is None
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert str(oid) not in ann["frames"].get("3", {})


# --- mask as source of truth ---------------------------------------------------


def test_get_mask_returns_png_for_segmented_object(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    r = client.get(f"/api/mask?video_id={vid}&object_id={oid}&frame_idx=2")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get(f"/api/mask?video_id={vid}&object_id={oid}&frame_idx=9").status_code == 404


def test_export_includes_full_res_coco_rle_mask(opened, tmp_path, monkeypatch):
    from kumo_track.annotate import rle

    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    client.post("/api/segment", json={
        "video_id": vid, "frame_idx": 2, "obj_id": oid, "box": [10, 10, 30, 30],
    })
    monkeypatch.chdir(tmp_path)
    client.post("/api/export", json={"video_id": vid, "out_name": "ds", "include_images": False})
    labels = json.loads((tmp_path / "data" / "ds" / "labels.json").read_text())
    rows = next(iter(labels.values()))
    mask = rows[0]["mask"]
    assert mask["size"] == [48, 64]  # full image resolution, not the ds res
    dec = rle.decode(mask)
    assert dec.shape == (48, 64) and dec.sum() > 0


def test_brush_mode_stores_mask_without_polygon(brush_opened):
    import cv2
    import numpy as np

    client, info = brush_opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    r = client.put("/api/mask", json={
        "video_id": vid, "object_id": oid, "frame_idx": 3,
        "mask_png": _mask_png(64, 48, (10, 8, 40, 36)),
    }).json()
    assert len(r["corners"]) == 4
    assert r["polygon"] is None  # brush mode keeps no polygon
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert ann["frames"]["3"][str(oid)]["polygon"] is None
    # the stored mask is fetchable and decodes to a non-empty RGBA PNG
    m = client.get(f"/api/mask?video_id={vid}&object_id={oid}&frame_idx=3")
    assert m.status_code == 200 and m.headers["content-type"] == "image/png"
    arr = cv2.imdecode(np.frombuffer(m.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert arr.shape[2] == 4 and arr[:, :, 3].any()


def test_sam3_health_local_is_ready(client):
    r = client.get("/api/sam3/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_sam3_health_remote_unreachable_is_loading(tmp_path, clip, monkeypatch):
    # Remote mode with a dead endpoint: the probe fails → "loading" (never 5xx).
    from fastapi.testclient import TestClient

    from kumo_track.annotate.app import create_app
    from tests.conftest import FakeTracker

    monkeypatch.setenv("SAM3_URL", "http://127.0.0.1:9")  # nothing listens here
    app = create_app(
        tracker_factory=FakeTracker,
        load_on_start=False,
        db_path=str(tmp_path / "remote.db"),
        video_dir=clip.parent,
        config={"edit_tool": "polygon"},
    )
    with TestClient(app) as c:
        r = c.get("/api/sam3/health")
    assert r.status_code == 200
    assert r.json() == {"status": "loading"}


def test_manual_box_stores_no_mask(opened, tmp_path, monkeypatch):
    # A pure bounding box is coordinates only — no polygon, no rasterised mask.
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "crate"}).json()["obj_id"]
    corners = [[4, 4], [30, 4], [30, 20], [4, 20]]
    r = client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 2, "corners": corners,
    }).json()
    assert r["corners"] == corners
    assert r["polygon"] is None
    assert r["has_mask"] is False
    fit = client.get(f"/api/annotations?video_id={vid}").json()["frames"]["2"][str(oid)]
    assert fit["polygon"] is None and fit["has_mask"] is False
    m = client.get(f"/api/mask?video_id={vid}&object_id={oid}&frame_idx=2")
    assert m.status_code == 404
    # exported row: box + label only, no "mask" key
    monkeypatch.chdir(tmp_path)
    client.post("/api/export", json={"video_id": vid, "out_name": "boxonly",
                                     "include_images": False})
    labels = json.loads((tmp_path / "data" / "boxonly" / "labels.json").read_text())
    rows = next(iter(labels.values()))
    assert rows[0]["corners"] == [[4, 4], [30, 4], [30, 20], [4, 20]]
    assert "mask" not in rows[0]


def test_manual_polygon_stores_mask(opened):
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "bag"}).json()["obj_id"]
    r = client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 1,
        "polygon": [[0, 0], [20, 0], [20, 12], [0, 12]],
    }).json()
    assert r["has_mask"] is True
    assert client.get(f"/api/mask?video_id={vid}&object_id={oid}&frame_idx=1").status_code == 200


def test_delete_object_removes_annotations_for_good(opened):
    # Deleting an object must take its annotations with it: SQLite reuses row
    # ids, so a survivor would re-attach to the next created object.
    client, info = opened
    vid = info["video_id"]
    oid = client.post("/api/objects", json={"video_id": vid, "label": "object"}).json()["obj_id"]
    client.put("/api/annotations", json={
        "video_id": vid, "object_id": oid, "frame_idx": 0,
        "corners": [[1, 1], [9, 1], [9, 9], [1, 9]],
    })
    client.delete(f"/api/objects/{oid}")
    oid2 = client.post("/api/objects", json={"video_id": vid, "label": "object"}).json()["obj_id"]
    assert oid2 == oid  # rowid reuse — exactly the hazard this guards against
    ann = client.get(f"/api/annotations?video_id={vid}").json()
    assert ann["frames"] == {}
