"""Uncompressed COCO run-length encoding for binary instance masks.

The annotation store keeps the *real* mask as its source of truth, serialised as
COCO RLE ``{"size": [h, w], "counts": [...]}``. ``counts`` is a list of run lengths
in **column-major** (Fortran) order, starting with the run of background (0) pixels —
the same layout pycocotools reads via ``frPyObjects``. We control both ends here, so
this small pure-numpy codec avoids a pycocotools build dependency. Swap in
``pycocotools.mask`` if the compressed ``counts`` string form is ever needed.
"""

import cv2
import numpy as np


def encode(binary: np.ndarray) -> dict:
    """HxW array (any non-zero is foreground) → uncompressed COCO RLE dict."""
    h, w = binary.shape
    flat = (np.asarray(binary) > 0).flatten(order="F").astype(np.uint8)
    bounds = np.concatenate(([0], np.where(np.diff(flat) != 0)[0] + 1, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:  # first run is foreground → lead with an empty bg run
        counts = [0] + counts
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def decode(rle: dict) -> np.ndarray:
    """COCO RLE dict → HxW uint8 mask of {0, 1}."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((h, w), order="F")


def rasterize_polygon(
    polygon: list[list[float]], ds_h: int, ds_w: int, full_h: int, full_w: int
) -> np.ndarray:
    """Full-res outline polygon → filled binary mask at the downscaled (ds_h, ds_w) size."""
    mask = np.zeros((ds_h, ds_w), dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.float32)
    if pts.shape[0] < 3:
        return mask
    pts[:, 0] *= ds_w / full_w
    pts[:, 1] *= ds_h / full_h
    cv2.fillPoly(mask, [pts.round().astype(np.int32)], 1)
    return mask
