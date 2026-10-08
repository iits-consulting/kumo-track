"""Storage backend units that don't need a real Azure account.

BlobStorage is exercised with a faked container client (no `azure` import needed):
we construct the instance directly and set the two attributes materialize_video
touches. The focus is the concurrent cold-cache download race (FIX-4).
"""

import threading
import time

from kumo_track.annotate.storage import BlobStorage


class _FakeDownload:
    """Mimics azure's StorageStreamDownloader: readinto(out) writes the blob.

    Writes in small chunks with a tiny sleep so two concurrent downloads of the
    same name reliably interleave — the condition that corrupts a shared .part.
    """

    def __init__(self, data: bytes):
        self._data = data

    def readinto(self, out) -> int:
        for i in range(0, len(self._data), 4096):
            out.write(self._data[i : i + 4096])
            out.flush()
            time.sleep(0.001)
        return len(self._data)


class _FakeVideos:
    container_name = "videos"

    def __init__(self, data: bytes):
        self._data = data

    def download_blob(self, name: str) -> _FakeDownload:
        return _FakeDownload(self._data)


def _blob_store(tmp_path, data: bytes) -> BlobStorage:
    # Bypass __init__ (which needs the azure SDK + a connection string): wire up
    # only what materialize_video uses.
    store = object.__new__(BlobStorage)
    store._videos = _FakeVideos(data)
    store._scratch = tmp_path / "scratch"
    store._scratch.mkdir()
    return store


def test_concurrent_materialize_same_clip_no_partial(tmp_path):
    # "Two teammates open the same new clip": both reach materialize concurrently.
    # Each must get a complete, correct local file — never a truncated/interleaved
    # one, and no spurious error from a second rename of a shared temp.
    data = bytes(range(256)) * 2000  # 512 KB, > one chunk
    store = _blob_store(tmp_path, data)

    results: list[bytes] = []
    errors: list[Exception] = []

    def worker():
        try:
            path = store.materialize_video("clip.mp4")
            results.append(path.read_bytes())
        except Exception as exc:  # noqa: BLE001 — surface it as a failure
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert results and all(r == data for r in results)
    assert (store._scratch / "clip.mp4").read_bytes() == data
    # No leftover temp files after either download.
    assert list(store._scratch.glob("*.part")) == []


class _Blob:
    def __init__(self, name):
        self.name = name


def test_list_videos_skips_nested_and_nonvideo():
    # Video container: root-level clips are real samples; derived/*
    # previews and stray non-video keys must not show up as samples.
    store = object.__new__(BlobStorage)
    store._videos = type("V", (), {
        "list_blobs": lambda self: [
            _Blob("clip_a.mp4"),
            _Blob("clip_b.mp4"),
            _Blob("derived/clip_a.preview.mp4"),  # nested → skip
            _Blob("derived/clip_a.jpg"),          # nested + non-video → skip
            _Blob("notes.txt"),                     # non-video → skip
        ],
    })()
    assert store.list_videos() == ["clip_a.mp4", "clip_b.mp4"]


def test_materialize_caches_by_name(tmp_path):
    # A present local copy is reused (source blobs are immutable per name).
    store = _blob_store(tmp_path, b"hello")
    first = store.materialize_video("clip.mp4")
    assert first.read_bytes() == b"hello"
    # Swap the fake blob contents; a cached file must NOT be re-downloaded.
    store._videos = _FakeVideos(b"different")
    second = store.materialize_video("clip.mp4")
    assert second == first and second.read_bytes() == b"hello"
