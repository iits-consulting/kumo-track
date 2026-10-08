"""Tiled (sliding-window) inference for a detector that exposes `detect_image`.

SAM3 resizes every input to 1008x1008 internally, so on a 3-4K frame a thin
object is downsampled ~3x and its mask comes out coarse, with boxes too loose
to clear IoU 0.5. Running the detector on overlapping crops raises the
effective resolution on each object, then the
per-tile detections are mapped back to full-image coordinates and merged with
rotated-box NMS (polygon IoU, so it handles the rotated quads SAM3 now emits).

Public API:
    tile_boxes(w, h, n_cols, n_rows, overlap) -> list[(x0, y0, x1, y1)]
    nms(dets, iou_thr) -> list[DetectionResult]
    tiled_detect(detector, image, n_cols, n_rows, overlap, nms_iou, full_frame)
        -> list[DetectionResult]
"""

from pathlib import Path

from PIL import Image

from kumo_track.base import DetectionResult
from kumo_track.geometry import polygon_iou


def tile_boxes(
    w: int, h: int, n_cols: int, n_rows: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Crop windows for an n_cols x n_rows grid with fractional `overlap`.

    Tiles are sized so neighbours share `overlap` of their extent (an object that
    straddles a seam is then likely fully inside at least one tile). Returns
    integer (x0, y0, x1, y1) pixel boxes covering the whole image.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    tw = w / (n_cols - (n_cols - 1) * overlap) if n_cols > 1 else w
    th = h / (n_rows - (n_rows - 1) * overlap) if n_rows > 1 else h
    step_x = tw * (1.0 - overlap) if n_cols > 1 else 0.0
    step_y = th * (1.0 - overlap) if n_rows > 1 else 0.0

    boxes = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = round(c * step_x)
            y0 = round(r * step_y)
            x1 = w if c == n_cols - 1 else round(c * step_x + tw)
            y1 = h if r == n_rows - 1 else round(r * step_y + th)
            boxes.append((x0, y0, min(x1, w), min(y1, h)))
    return boxes


def nms(dets: list[DetectionResult], iou_thr: float) -> list[DetectionResult]:
    """Greedy per-label rotated-box NMS, highest score first (polygon IoU)."""
    order = sorted(range(len(dets)), key=lambda i: dets[i].score, reverse=True)
    suppressed = [False] * len(dets)
    keep: list[DetectionResult] = []
    for rank, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(dets[i])
        for j in order[rank + 1:]:
            if suppressed[j] or dets[j].label != dets[i].label:
                continue
            if polygon_iou(dets[i].corners, dets[j].corners) > iou_thr:
                suppressed[j] = True
    return keep


def tiled_detect(
    detector,
    image: str | Path | Image.Image,
    n_cols: int,
    n_rows: int,
    overlap: float = 0.2,
    nms_iou: float = 0.3,
    full_frame: bool = False,
) -> list[DetectionResult]:
    """Run `detector.detect_image` on overlapping tiles; merge with rotated NMS.

    `image` may be a path or a PIL image. Per-tile corners are offset by the tile
    origin back to full-image coordinates. With `full_frame=True` the whole-frame
    detections are added too (catches objects longer than one tile), then NMS
    dedupes overlaps across tiles and scales.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    w, h = image.width, image.height

    dets: list[DetectionResult] = []
    for x0, y0, x1, y1 in tile_boxes(w, h, n_cols, n_rows, overlap):
        crop = image.crop((x0, y0, x1, y1))
        for d in detector.detect_image(crop):
            shifted = [[x + x0, y + y0] for x, y in d.corners]
            dets.append(DetectionResult(corners=shifted, score=d.score, label=d.label))

    if full_frame:
        dets.extend(detector.detect_image(image))

    return nms(dets, nms_iou)
