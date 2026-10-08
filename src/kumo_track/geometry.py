"""Polygon IoU for rotated/axis-aligned 4-corner boxes.

Both ground truth and detector predictions may be rotated boxes (4-corner
polygons), so overlap is computed as the IoU of two convex quadrilaterals.
Used by :mod:`kumo_track.tiled` to merge per-tile detections with rotated NMS.
"""

import cv2
import numpy as np


def _hull(corners: list[list[float]]) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.convexHull(pts)


def polygon_iou(corners_a: list[list[float]], corners_b: list[list[float]]) -> float:
    """IoU of two convex quadrilaterals (rotated or axis-aligned)."""
    ha, hb = _hull(corners_a), _hull(corners_b)
    area_a = abs(float(cv2.contourArea(ha)))
    area_b = abs(float(cv2.contourArea(hb)))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter = float(cv2.intersectConvexConvex(ha, hb)[0])
    if inter <= 0.0:
        return 0.0
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0
