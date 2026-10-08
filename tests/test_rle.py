"""Unit tests for the uncompressed COCO RLE codec."""

import numpy as np

from kumo_track.annotate import rle


def test_round_trip_preserves_mask():
    masks = [
        np.zeros((5, 7), np.uint8),
        np.ones((5, 7), np.uint8),
        (np.arange(35).reshape(5, 7) % 3 == 0).astype(np.uint8),
    ]
    for m in masks:
        d = rle.decode(rle.encode(m))
        assert d.shape == m.shape
        assert (d == (m > 0)).all()


def test_counts_are_column_major_with_leading_background():
    # 2x2 with one foreground pixel at row=1,col=0. Column-major (Fortran) flatten
    # is [m00, m10, m01, m11] = [0, 1, 0, 0] → runs 1 bg, 1 fg, 2 bg.
    m = np.zeros((2, 2), np.uint8)
    m[1, 0] = 1
    assert rle.encode(m) == {"size": [2, 2], "counts": [1, 1, 2]}


def test_encode_accepts_0_255_masks():
    m = np.zeros((3, 3), np.uint8)
    m[0, 0] = 255
    assert rle.decode(rle.encode(m)).sum() == 1


def test_rasterize_polygon_fills_rectangle():
    poly = [[1, 1], [4, 1], [4, 3], [1, 3]]
    m = rle.rasterize_polygon(poly, 5, 6, 5, 6)
    assert m.shape == (5, 6) and m.sum() > 0
    # degenerate (<3 points) → empty mask, no crash
    assert rle.rasterize_polygon([[1, 1], [2, 2]], 5, 6, 5, 6).sum() == 0
