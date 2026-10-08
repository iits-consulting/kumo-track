"""GET /api/annotation-status: the aggregate the project hub joins on clip name.

Must sum objects across strides (multiple video rows, same name) and collect the
distinct X-Kumo-User annotators. Clips with no objects stay absent (= unlabeled)."""


def _open(client, stride):
    r = client.post("/api/open", json={"name": "clip.avi", "stride": stride})
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


def _add_object(client, vid, label, user):
    r = client.post("/api/objects", json={"video_id": vid, "label": label},
                    headers={"X-Kumo-User": user})
    assert r.status_code == 200, r.text


def test_status_aggregates_across_strides_and_annotators(client):
    # Same clip, two decoding configs → two video rows sharing the name.
    v4 = _open(client, 4)
    v5 = _open(client, 5)
    assert v4 != v5
    _add_object(client, v4, "bag", "a@x.de")
    _add_object(client, v4, "bag", "a@x.de")
    _add_object(client, v5, "box", "b@x.de")

    status = client.get("/api/annotation-status").json()["videos"]
    assert set(status) == {"clip.avi"}
    entry = status["clip.avi"]
    assert entry["object_count"] == 3
    assert entry["annotators"] == ["a@x.de", "b@x.de"]   # distinct, sorted
    assert entry["last_activity"] is not None


def test_status_omits_clips_without_objects(client):
    _open(client, 4)   # opened, but no objects created
    assert client.get("/api/annotation-status").json()["videos"] == {}
