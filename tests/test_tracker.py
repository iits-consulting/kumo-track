"""TrackerManager.propagate window-chaining + global/local index mapping.

The #1 correctness risk is the global↔window-local frame mapping and the
multi-window chaining. We exercise it with a fake window that mirrors the real
``propagate_in_video_iterator`` index contract, so a tiny window_len forces many
chained windows over the clip. Each fit encodes its global frame, so a mis-mapping
would surface as a wrong corner value, not just a wrong count.
"""

import pytest

# The local windowed tracker needs torch (the `local` extra). Skip on a CPU-only
# install that runs purely against the hosted SAM3 service.
pytest.importorskip("torch")

from kumo_track.annotate.tracker import TrackerManager  # noqa: E402


class FakeFS:
    n_frames = 10
    height = 100
    width = 100


class FakeOut:
    def __init__(self, frame_idx):
        self.frame_idx = frame_idx
        self.pred_masks = None


class FakeWindow:
    def __init__(self, w0, w1):
        self.w0, self.w1 = w0, w1
        self.seeded: set[int] = set()

    @property
    def length(self):
        return self.w1 - self.w0 + 1

    def contains(self, g):
        return self.w0 <= g <= self.w1

    def local(self, g):
        return g - self.w0

    def seed_from_ann(self, obj_id, local_idx, ann):
        self.seeded.add(obj_id)
        return True

    def mark_seeded(self, obj_ids):
        pass

    def iterate(self, local_start, max_steps, reverse):
        # Mirror propagate_in_video_iterator's processing order exactly.
        if reverse:
            end = max(local_start - max_steps, 0)
            order = range(local_start, end - 1, -1)
        else:
            end = min(local_start + max_steps, self.length - 1)
            order = range(local_start, end + 1)
        for local in order:
            yield FakeOut(local)

    def fits(self, out, exclude):
        g = self.w0 + out.frame_idx
        # Encode the global frame index into the corners so the test can verify mapping.
        corners = [[g, g], [g + 1, g], [g + 1, g + 1], [g, g + 1]]
        return {oid: {"corners": corners, "polygon": corners, "score": None}
                for oid in self.seeded if oid not in exclude}

    def close(self):
        pass


class ManagerUnderTest(TrackerManager):
    def __init__(self, window_len):
        self.fs = FakeFS()
        self.window_len = window_len
        self.n_frames = FakeFS.n_frames
        self.cur = None
        self.removed = set()

    def _new_window(self, w0, w1):
        return FakeWindow(w0, w1)


def _run(mgr, start, reverse, max_steps):
    seed = {7: {"corners": [[0, 0]], "polygon": [[0, 0]]}}
    out = list(mgr.propagate(start, reverse, max_steps, seed))
    frames = [g for g, _ in out]
    # Every fit must encode its own global frame (mapping correctness).
    for g, boxes in out:
        assert boxes[7]["corners"][0] == [g, g], f"frame {g} mis-mapped: {boxes[7]}"
    return frames


def test_forward_to_clip_end_chains_windows():
    # window_len=3 → forces ~5 chained windows across the 10-frame clip.
    assert _run(ManagerUnderTest(3), start=0, reverse=False, max_steps=None) == list(range(1, 10))


def test_forward_bounded_steps():
    assert _run(ManagerUnderTest(3), start=0, reverse=False, max_steps=4) == [1, 2, 3, 4]


def test_reverse_to_clip_start_chains_windows():
    assert _run(ManagerUnderTest(3), start=9, reverse=True, max_steps=None) == [8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_reverse_bounded_steps():
    assert _run(ManagerUnderTest(4), start=9, reverse=True, max_steps=3) == [8, 7, 6]


def test_midclip_start_forward():
    assert _run(ManagerUnderTest(3), start=4, reverse=False, max_steps=None) == [5, 6, 7, 8, 9]


def test_two_objects_chained():
    mgr = ManagerUnderTest(3)
    seed = {7: {"corners": [[0, 0]], "polygon": [[0, 0]]}, 9: {"corners": [[0, 0]], "polygon": [[0, 0]]}}
    out = list(mgr.propagate(0, False, None, seed))
    assert [g for g, _ in out] == list(range(1, 10))
    for g, boxes in out:
        assert set(boxes) == {7, 9}
        assert boxes[7]["corners"][0] == [g, g]


def test_removed_object_not_emitted():
    mgr = ManagerUnderTest(3)
    mgr.forget_object(7)
    seed = {7: {"corners": [[0, 0]], "polygon": [[0, 0]]}}
    assert list(mgr.propagate(0, False, None, seed)) == []
