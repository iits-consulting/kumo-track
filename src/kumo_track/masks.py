"""Mask geometry helpers shared by the local model code and the annotation app.

These are pure numpy/cv2 utilities (no torch/transformers), kept in one torch-free
module so the app can import them on a CPU-only box that talks to the remote SAM3
service instead of loading SAM 3 in-process. The local detector/tracker and the app
all import from here.
"""

import cv2
import numpy as np

# Cap the post-processed mask resolution, then scale fitted box/polygon back up.
# Full-res masks per object per frame blow up memory on 4K clips; 1024px long-side
# is ample for a tight minAreaRect / contour.
_MASK_MAX_SIDE = 1024
_POLY_MAX_POINTS = 48  # cap mask-outline vertices stored per annotation


def mask_target_size(height: int, width: int) -> tuple[int, int]:
    """Downscaled (h, w) for mask post-processing, capping the long side."""
    long_side = max(height, width)
    if long_side <= _MASK_MAX_SIDE:
        return height, width
    s = _MASK_MAX_SIDE / long_side
    return max(1, round(height * s)), max(1, round(width * s))


def _mask_polygon(binary: np.ndarray, scale_x: float, scale_y: float) -> list[list[float]] | None:
    """Largest external contour, simplified and capped, rescaled to full-res."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(c, True)
    pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
    if len(pts) > _POLY_MAX_POINTS:
        keep = np.linspace(0, len(pts) - 1, _POLY_MAX_POINTS).round().astype(int)
        pts = pts[keep]
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    return pts.tolist()


def _mask_to_corners(mask: np.ndarray, scale_x: float, scale_y: float) -> list[list[float]] | None:
    """Fit a rotated (minimum-area) box to a binary mask and return its 4 corners.

    `mask` is an HxW array of {0,1}; corners are rescaled by (scale_x, scale_y)
    to map a downscaled mask back to full image coordinates. Returns None for an
    empty or degenerate (≈zero-area) mask.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
    (_, _), (rw, rh), _ = rect
    if rw * rh < 1.0:
        return None
    corners = cv2.boxPoints(rect)  # 4 [x,y] points in consistent cyclic order
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y
    return corners.tolist()
