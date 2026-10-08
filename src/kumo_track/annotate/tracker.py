"""Windowed SAM3 video tracking for the annotation app.

The transformers SAM3 video session ingests its whole frame list at init, so a
20-minute clip can't live in one session. :class:`TrackerManager` runs SAM3 over
**sliding windows** of ``TRACK_WINDOW`` sampled frames pulled on demand from a
:class:`~kumo_track.annotate.frames.FrameSource`. Long propagations *chain*
windows: when tracking reaches a window edge, the next window is built and every
tracked object is re-seeded on the boundary frame from its last mask (rasterised
from the stored outline polygon, falling back to the box).

Index conventions:

* *global* sampled idx — the app/DB frame index (``0 .. n_frames-1``);
* *window-local* idx — index inside one window's frame slice; ``local = global - w0``.
  SAM3 only ever sees window-local indices.

The manager is a disposable cache: it never touches the DB. Seeds for propagation
are passed in by the caller (read from the DB), so any window is reconstructible.
The heavy SAM3 model is a module-level singleton shared across sessions; only the
per-window inference state is transient. Not thread-safe — guard with the GPU lock.
"""

import os

import cv2
import numpy as np
import torch

from kumo_track.annotate import rle
from kumo_track.masks import _mask_polygon, _mask_to_corners, mask_target_size

_MODEL_NAME = "facebook/sam3"
_TRACK_WINDOW = int(os.environ.get("TRACK_WINDOW", "200"))  # sampled frames per window (upper bound)
# A propagation window decodes every one of its frames into RAM at once. A 4K frame
# is ~25 MB, so 200 frames would be ~5 GB in a single list and OOM-kill the process.
# Cap the *decoded* window to this many bytes; the effective window length is derived
# from the clip's frame size (override the budget with TRACK_WINDOW_MB).
_WINDOW_BUDGET_BYTES = int(float(os.environ.get("TRACK_WINDOW_MB", "1024")) * 1024 * 1024)

_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load (once) and return the SAM3 video tracker model + processor."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        _PROCESSOR = Sam3TrackerVideoProcessor.from_pretrained(_MODEL_NAME)
        _MODEL = Sam3TrackerVideoModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _PROCESSOR


def _ann_aabb(ann: dict) -> list[float] | None:
    """Axis-aligned [x1,y1,x2,y2] from an annotation's OBB corners."""
    corners = ann.get("corners")
    if not corners:
        return None
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


class TrackerWindow:
    """One SAM3 inference session over the sampled-frame range ``[w0, w1]``."""

    def __init__(self, model, proc, frames_rgb, w0, w1, device, full_h, full_w):
        self.model = model
        self.proc = proc
        self.w0 = w0
        self.w1 = w1
        self.device = device
        self.full_h, self.full_w = full_h, full_w
        self.ds_h, self.ds_w = mask_target_size(full_h, full_w)
        self.session = proc.init_video_session(
            video=frames_rgb,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.float32,
        )

    @property
    def length(self) -> int:
        return self.w1 - self.w0 + 1

    def contains(self, global_idx: int) -> bool:
        return self.w0 <= global_idx <= self.w1

    def local(self, global_idx: int) -> int:
        return global_idx - self.w0

    # --- seeding ---------------------------------------------------------------

    def add_prompt(self, obj_id, local_idx, points=None, labels=None, box=None) -> None:
        """Add point/box prompts for one object on a window-local frame."""
        kw: dict = {}
        if points:
            kw["input_points"] = [[[[float(x), float(y)] for x, y in points]]]
            pt_labels = labels if labels is not None else [1] * len(points)
            kw["input_labels"] = [[[int(v) for v in pt_labels]]]
        if box:
            kw["input_boxes"] = [[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]]
        self.proc.add_inputs_to_inference_session(
            inference_session=self.session,
            frame_idx=local_idx,
            obj_ids=obj_id,
            clear_old_inputs=True,
            **kw,
        )

    def add_mask(self, obj_id, local_idx, mask: np.ndarray) -> None:
        """Add a binary mask prompt for one object on a window-local frame."""
        self.proc.add_inputs_to_inference_session(
            inference_session=self.session,
            frame_idx=local_idx,
            obj_ids=obj_id,
            input_masks=mask,
            clear_old_inputs=True,
        )

    def seed_from_ann(self, obj_id, local_idx, ann: dict) -> bool:
        """Re-seed an object from a stored annotation (mask preferred, box fallback)."""
        mask_rle = ann.get("mask_rle")
        if mask_rle:
            mask = rle.decode(mask_rle)
            if mask.shape != (self.ds_h, self.ds_w):  # defensive: stored at a different ds
                mask = cv2.resize(mask, (self.ds_w, self.ds_h), interpolation=cv2.INTER_NEAREST)
            self.add_mask(obj_id, local_idx, mask)
            return True
        polygon = ann.get("polygon")  # old rows without a stored mask
        if polygon:
            self.add_mask(obj_id, local_idx, self._rasterize(polygon))
            return True
        box = _ann_aabb(ann)
        if box:
            self.add_prompt(obj_id, local_idx, box=box)
            return True
        return False

    def mark_seeded(self, obj_ids) -> None:
        """Flag every object as having new inputs for the next forward.

        ``add_inputs_to_inference_session`` *assigns* ``obj_with_new_inputs`` per
        call, so seeding several objects before one propagate would leave only the
        last flagged — the others' conditioning would be silently ignored. Set the
        full set after seeding so the boundary forward conditions on all of them."""
        self.session.obj_with_new_inputs = list(obj_ids)

    def _rasterize(self, polygon: list[list[float]]) -> np.ndarray:
        """Full-res outline polygon → binary mask at the downscaled mask size."""
        return rle.rasterize_polygon(polygon, self.ds_h, self.ds_w, self.full_h, self.full_w)

    # --- inference -------------------------------------------------------------

    def run_frame(self, local_idx):
        with torch.inference_mode():
            return self.model(inference_session=self.session, frame_idx=local_idx)

    def iterate(self, local_start, max_steps, reverse):
        with torch.inference_mode():
            yield from self.model.propagate_in_video_iterator(
                inference_session=self.session,
                start_frame_idx=local_start,
                max_frame_num_to_track=max_steps,
                reverse=reverse,
            )

    def _masks(self, out):
        return self.proc.post_process_masks(
            [out.pred_masks], original_sizes=[[self.ds_h, self.ds_w]], binarize=False
        )[0]

    def fit_obj(self, out, obj_id) -> dict | None:
        obj_ids = list(self.session.obj_ids)
        if obj_id not in obj_ids:
            return None
        return self._fit(self._masks(out)[obj_ids.index(obj_id)])

    def fits(self, out, exclude: set[int]) -> dict[int, dict | None]:
        masks = self._masks(out)
        obj_ids = list(self.session.obj_ids)
        return {oid: self._fit(masks[j]) for j, oid in enumerate(obj_ids) if oid not in exclude}

    def _fit(self, mask) -> dict | None:
        if mask.ndim == 3:
            mask = mask[0]
        binary = (mask > 0).to(torch.uint8).cpu().numpy()
        if binary.sum() == 0:
            return None
        scale_x = self.full_w / binary.shape[1]
        scale_y = self.full_h / binary.shape[0]
        corners = _mask_to_corners(binary, scale_x, scale_y)
        if corners is None:
            return None
        return {
            "corners": corners,
            "polygon": _mask_polygon(binary, scale_x, scale_y),
            "mask_rle": rle.encode(binary),
            "score": None,
        }

    def close(self) -> None:
        self.session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TrackerManager:
    """Sliding-window SAM3 tracker over a :class:`FrameSource`. GPU-lock guarded."""

    def __init__(self, frame_source, window_len: int | None = None, device: str | None = None):
        self.fs = frame_source
        self.model, self.proc = load_model()
        self.device = device or _DEVICE
        # Cap the window so a decoded window fits the RAM budget (see _WINDOW_BUDGET_BYTES).
        # On a 4K clip this yields ~40 frames/window; on 1080p ~165; small clips keep the
        # full TRACK_WINDOW. Long clips just chain more windows (re-seeded at each boundary).
        frame_bytes = max(1, int(frame_source.height) * int(frame_source.width) * 3)
        self.window_len = max(8, min(window_len or _TRACK_WINDOW, _WINDOW_BUDGET_BYTES // frame_bytes))
        self.n_frames = frame_source.n_frames
        self.cur: TrackerWindow | None = None
        self.removed: set[int] = set()

    # --- window lifecycle ------------------------------------------------------

    def _new_window(self, w0: int, w1: int) -> TrackerWindow:
        frames = self.fs.get_frames(w0, w1)
        return TrackerWindow(
            self.model, self.proc, frames, w0, w1, self.device, self.fs.height, self.fs.width
        )

    def ensure_window(self, frame_idx: int) -> TrackerWindow:
        """Single-frame window for interactive segment/refine on ``frame_idx``.

        Interactive segmentation only ever runs the model on the seed frame — it
        never propagates — so the window needs just that one frame. Building a full
        ``window_len`` window here (the old behaviour) decoded hundreds of frames
        into a single RAM list (~5 GB on a 4K clip), OOM-killing the process. The
        reused-if-contained check keeps refine clicks on the same frame cheap;
        propagation builds its own (chained, memory-budgeted) windows separately.
        """
        if self.cur is not None and self.cur.contains(frame_idx):
            return self.cur
        if self.cur is not None:
            self.cur.close()
            self.cur = None
        self.cur = self._new_window(frame_idx, frame_idx)
        return self.cur

    def forget_object(self, obj_id: int) -> None:
        """Stop emitting an object after it's deleted or pinned (safe if never seeded here)."""
        self.removed.add(obj_id)

    def unforget_object(self, obj_id: int) -> None:
        """Let an object track again after it's unpinned."""
        self.removed.discard(obj_id)

    # --- interactive segment ---------------------------------------------------

    def segment(self, frame_idx, obj_id, points=None, labels=None, box=None) -> dict | None:
        """Seed/refine ``obj_id`` on ``frame_idx`` with points/box; return its fit."""
        win = self.ensure_window(frame_idx)
        win.add_prompt(obj_id, win.local(frame_idx), points, labels, box)
        out = win.run_frame(win.local(frame_idx))
        return win.fit_obj(out, obj_id)

    # --- propagation -----------------------------------------------------------

    def propagate(self, start_frame_idx: int, reverse: bool, max_steps: int | None, seeds: dict[int, dict]):
        """Track ``seeds`` from ``start_frame_idx``; yield ``(global_idx, {obj_id: fit|None})``.

        ``seeds`` maps object id → its annotation on the start frame (corners/polygon).
        ``max_steps`` is how many frames *beyond* the start frame to annotate
        (``None`` = to clip start/end). Windows are chained transparently; the start
        frame itself is never re-emitted (the caller already has it).
        """
        # Free any interactive window so propagation windows own the GPU budget.
        if self.cur is not None:
            self.cur.close()
            self.cur = None

        seeds = {oid: ann for oid, ann in seeds.items() if oid not in self.removed and ann}
        if not seeds:
            return

        remaining = max_steps  # None = unbounded
        boundary = start_frame_idx
        cur_seeds = seeds

        while True:
            if reverse:
                w1 = boundary
                w0 = max(0, boundary - self.window_len + 1)
            else:
                w0 = boundary
                w1 = min(self.n_frames - 1, boundary + self.window_len - 1)

            win = self._new_window(w0, w1)
            local_boundary = win.local(boundary)
            seeded = []
            for oid, ann in cur_seeds.items():
                if win.seed_from_ann(oid, local_boundary, ann):
                    seeded.append(oid)
            if not seeded:
                win.close()
                return
            win.mark_seeded(seeded)  # condition the boundary forward on ALL seeds

            win_span = local_boundary if reverse else (win.length - 1 - local_boundary)
            max_track = win_span if remaining is None else min(win_span, remaining)

            edge_global = w0 if reverse else w1
            edge_fits: dict[int, dict] = {}
            try:
                for out in win.iterate(local_boundary, max_track, reverse):
                    g = w0 + out.frame_idx
                    fits = win.fits(out, self.removed)
                    if g == edge_global:
                        edge_fits = {oid: f for oid, f in fits.items() if f}
                    if g == boundary:
                        continue  # seed/boundary frame — already annotated, don't re-emit
                    yield g, fits
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            return
            finally:
                win.close()

            reached_clip_end = (w0 <= 0) if reverse else (w1 >= self.n_frames - 1)
            if reached_clip_end or not edge_fits:
                return
            boundary = edge_global
            cur_seeds = edge_fits

    def close(self) -> None:
        if self.cur is not None:
            self.cur.close()
            self.cur = None
