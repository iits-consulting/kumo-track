"""DB: migration idempotency, static toggle, manual/seed/static query helpers."""

import sqlite3

from kumo_track.annotate import db


def _fresh(tmp_path):
    return db.connect(tmp_path / "a.db")


def test_migration_updates_old_schemas(tmp_path):
    # Pre-static DB: gains the flag. Static-range-era DB: loses the range columns.
    p = tmp_path / "old.db"
    raw = sqlite3.connect(p)
    raw.executescript(
        "CREATE TABLE object (id INTEGER PRIMARY KEY, video_id INTEGER, label TEXT,"
        " created_by TEXT, created_at TEXT, static_start INTEGER, static_end INTEGER);"
    )
    raw.commit()
    raw.close()
    # connect() must migrate it without error, twice (idempotent).
    for _ in range(2):
        c = db.connect(p)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(object)")}
        assert {"static", "hidden", "color"} <= cols
        assert not {"static_start", "static_end"} & cols
        c.close()


def test_static_toggle_preserves_annotations(tmp_path):
    c = _fresh(tmp_path)
    vid = db.upsert_video(c, "v", "/v", 100, 100, 10.0, 4, None, 10, list(range(0, 40, 4)))
    oid = db.create_object(c, vid, "bag", None)
    db.upsert_annotation(c, vid, oid, 0, 0, [[0, 0], [1, 0], [1, 1], [0, 1]], None, None, "propagated")
    db.upsert_annotation(c, vid, oid, 1, 4, [[0, 0], [2, 0], [2, 2], [0, 2]], None, None, "seed")
    db.upsert_annotation(c, vid, oid, 2, 8, [[0, 0], [3, 0], [3, 3], [0, 3]], None, None, "propagated")
    for static in (True, False):  # a pure mode toggle — rows untouched both ways
        db.set_object_static(c, oid, static)
        rows = c.execute("SELECT frame_idx FROM annotation WHERE object_id=?", (oid,)).fetchall()
        assert [r["frame_idx"] for r in rows] == [0, 1, 2]
        assert db.get_annotations(c, vid)["objects"][0]["static"] is static
    c.close()


def test_hidden_and_color_are_display_only(tmp_path):
    # Both are pure display metadata: setters round-trip through get_annotations
    # and never touch the object's annotations.
    c = _fresh(tmp_path)
    vid = db.upsert_video(c, "v", "/v", 100, 100, 10.0, 4, None, 10, list(range(0, 40, 4)))
    oid = db.create_object(c, vid, "bag", None)
    db.upsert_annotation(c, vid, oid, 0, 0, [[0, 0], [1, 0], [1, 1], [0, 1]], None, None, "seed")
    obj = db.get_annotations(c, vid)["objects"][0]
    assert obj["hidden"] is False and obj["color"] is None  # defaults

    db.set_object_hidden(c, oid, True)
    db.set_object_color(c, oid, "#ff8800")
    obj = db.get_annotations(c, vid)["objects"][0]
    assert obj["hidden"] is True and obj["color"] == "#ff8800"
    # annotation untouched
    assert c.execute("SELECT COUNT(*) FROM annotation WHERE object_id=?", (oid,)).fetchone()[0] == 1

    db.set_object_hidden(c, oid, False)
    db.set_object_color(c, oid, None)  # clears the override
    obj = db.get_annotations(c, vid)["objects"][0]
    assert obj["hidden"] is False and obj["color"] is None
    c.close()


def test_static_annotations_picks_box_nearest_start_frame(tmp_path):
    c = _fresh(tmp_path)
    vid = db.upsert_video(c, "v", "/v", 100, 100, 10.0, 4, None, 10, list(range(0, 40, 4)))
    oid = db.create_object(c, vid, "zone", None)
    near = [[0, 0], [1, 0], [1, 1], [0, 1]]
    far = [[5, 5], [6, 5], [6, 6], [5, 6]]
    db.upsert_annotation(c, vid, oid, 1, 4, far, None, None, "seed")
    db.upsert_annotation(c, vid, oid, 5, 20, near, None, None, "propagated")
    db.set_object_static(c, oid, True)
    assert db.static_annotations(c, vid, 4)[oid]["corners"] == near  # frame 5 beats frame 1
    assert db.static_annotations(c, vid, 1)[oid]["corners"] == far
    c.close()


def test_seed_annotations_excludes_static(tmp_path):
    c = _fresh(tmp_path)
    vid = db.upsert_video(c, "v", "/v", 100, 100, 10.0, 4, None, 10, list(range(0, 40, 4)))
    a = db.create_object(c, vid, "a", None)
    b = db.create_object(c, vid, "b", None)
    box = [[0, 0], [1, 0], [1, 1], [0, 1]]
    db.upsert_annotation(c, vid, a, 5, 20, box, box, None, "seed")
    db.upsert_annotation(c, vid, b, 5, 20, box, box, None, "seed")
    db.set_object_static(c, b, True)
    seeds = db.seed_annotations(c, vid, 5)
    assert set(seeds) == {a}  # static object b is not a tracking seed
    c.close()


def test_manual_annotations_lookup(tmp_path):
    c = _fresh(tmp_path)
    vid = db.upsert_video(c, "v", "/v", 100, 100, 10.0, 4, None, 10, list(range(0, 40, 4)))
    oid = db.create_object(c, vid, "a", None)
    box = [[0, 0], [1, 0], [1, 1], [0, 1]]
    db.upsert_annotation(c, vid, oid, 3, 12, box, box, None, "manual")
    db.upsert_annotation(c, vid, oid, 4, 16, box, box, None, "propagated")
    manual = db.manual_annotations(c, vid)
    assert set(manual) == {(oid, 3)}
    c.close()
