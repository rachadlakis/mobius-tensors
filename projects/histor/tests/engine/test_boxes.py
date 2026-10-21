"""Box geometry (engine/boxes.py), shared by the tracker, motion, abandoned and the Pipeline."""

import numpy as np
import pytest

from histor.engine.boxes import (area, centre, clip_box, greedy_match, inside, iou_matrix,
                                 loose_overlap, scale_box)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], float)
    iou = iou_matrix(a, b)
    assert iou.shape == (2, 3)
    assert iou[0, 0] == pytest.approx(1.0) and iou[0, 1] == pytest.approx(50 / 150)
    assert iou[1].tolist() == [0, 0, 0]
    assert iou_matrix(a, np.zeros((0, 4))).shape == (2, 0)
    empty = np.array([[5, 5, 5, 5], [10, 10, 0, 0]], float)  # no area, and inside out
    assert iou_matrix(empty, a).tolist() == [[0, 0], [0, 0]]


def test_greedy_match_best_first_each_once():
    iou = np.array([[0.9, 0.8], [0.85, 0.2], [0.1, 0.05]])
    assert greedy_match(iou, 0.3) == [(0, 0)]  # row 1 lost col 0, col 1 is below for it
    assert sorted(greedy_match(iou, 0.15)) == [(0, 0), (1, 1)]
    assert greedy_match(np.zeros((0, 3)), 0.3) == []
    assert greedy_match(np.array([[0.5, 0.5]]), 0.3) == [(0, 0)]  # a tie: the first column


def test_area_centre_inside():
    assert area([0, 0, 4, 5]) == 20 and area([4, 4, 0, 0]) == 0
    assert centre([0, 0, 4, 6]).tolist() == [2, 3]
    assert inside([0, 0, 10, 10], [5, 0, 20, 10]) == pytest.approx(0.5)
    assert inside([0, 0, 10, 10], [-5, -5, 50, 50]) == pytest.approx(1.0)
    assert inside([3, 3, 3, 3], [0, 0, 10, 10]) == 0.0  # an empty box is inside nothing


def test_loose_overlap():
    a = [0, 0, 10, 10]
    assert loose_overlap(a, [1, 1, 11, 11])             # IoU 0.68
    assert loose_overlap(a, [10, 0, 14, 10])            # touching: centre 12 within 10 + 5
    assert not loose_overlap(a, [30, 0, 40, 10])
    assert not loose_overlap(a, [10, 0, 14, 10], grow=0.1)
    assert not loose_overlap([5, 5, 5, 5], [30, 30, 30, 30])  # empty boxes


def test_scale_and_clip():
    assert scale_box([10, 20, 30, 40], 0.5).tolist() == [5, 10, 15, 20]
    assert scale_box([10, 20, 30, 40], 2, 0.5).tolist() == [20, 10, 60, 20]
    assert clip_box([-5, 10, 700, 500], 640, 360).tolist() == [0, 10, 640, 360]
