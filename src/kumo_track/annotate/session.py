"""One open clip: a :class:`FrameSource` + a windowed SAM3 tracker.

This object is the per-request handle the app holds for the single active video.
It owns lazy frame access (decode/JPEG/export) and delegates all SAM3 work to a
tracker (the local :class:`~kumo_track.annotate.tracker.TrackerManager` or the
:class:`~kumo_track.annotate.remote.RemoteTrackerManager`). The tracker is injected
via a factory so tests can swap in a fake (no GPU / no weights) and so the heavy
local tracker — which imports torch — is only loaded when actually used. The DB
remains the source of truth; this session is disposable and reconstructible from it.
"""

from collections.abc import Callable

from kumo_track.annotate.frames import FrameSource

__all__ = ["VideoSession"]


class VideoSession:
    """An open clip: frame access + a windowed tracker. Guard with the GPU lock."""

    def __init__(
        self,
        video_id: int,
        name: str,
        path: str,
        stride: int,
        tracker_factory: Callable[[FrameSource], object] | None = None,
    ):
        self.video_id = video_id
        self.name = name
        self.stride = stride
        self.fs = FrameSource(path, stride, name=name)
        if tracker_factory is None:  # default: the local torch tracker (lazy import)
            from kumo_track.annotate.tracker import TrackerManager

            tracker_factory = TrackerManager
        self.tracker = tracker_factory(self.fs)

    # --- clip info -------------------------------------------------------------

    @property
    def width(self) -> int:
        return self.fs.width

    @property
    def height(self) -> int:
        return self.fs.height

    @property
    def fps(self) -> float:
        return self.fs.fps

    @property
    def n_frames(self) -> int:
        return self.fs.n_frames

    @property
    def source_indices(self) -> list[int]:
        return self.fs.source_indices

    @property
    def truncated_at(self) -> int | None:
        return self.fs.truncated_at

    def source_frame(self, frame_idx: int) -> int:
        return self.fs.source_index(frame_idx)

    # --- frames ----------------------------------------------------------------

    def jpeg(self, frame_idx: int, quality: int = 80) -> bytes:
        return self.fs.jpeg(frame_idx, quality)

    def frame_bgr(self, frame_idx: int):
        return self.fs.frame_bgr(frame_idx)

    # --- tracking (delegated) --------------------------------------------------

    def segment(self, frame_idx, obj_id, points=None, labels=None, box=None):
        return self.tracker.segment(frame_idx, obj_id, points, labels, box)

    def propagate(self, start_frame_idx, reverse, max_steps, seeds):
        return self.tracker.propagate(start_frame_idx, reverse, max_steps, seeds)

    def forget_object(self, obj_id: int) -> None:
        self.tracker.forget_object(obj_id)

    def unforget_object(self, obj_id: int) -> None:
        self.tracker.unforget_object(obj_id)

    # --- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        self.tracker.close()
        self.fs.close()
