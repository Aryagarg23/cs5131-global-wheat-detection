import numpy as np
import pytest

from wheat.metric import calculate_final_score


def score(gt, pred, scores):
    return calculate_final_score([{'image_id': 'x', 'gt_boxes': np.array(gt, float),
                                   'pred_boxes': np.array(pred, float).reshape(-1, 4),
                                   'scores': np.array(scores, float)}], score_threshold=0.0)


A = [0, 0, 99, 99]
B = [20, 0, 119, 99]


def test_perfect_and_empty():
    assert score([A, B], [B, A], [0.5, 0.9]) == pytest.approx(1.0)
    assert score([A, B], [], []) == 0.0


def test_partial_match():
    # IoU 0.62 matches at thresholds 0.5, 0.55, 0.6 out of six
    assert score([A], [[0, 0, 99, 61]], [1]) == pytest.approx(0.5)


def test_sorted_by_score():
    # The weak box overlaps B more than A. Without sorting it would take B first and
    # the confident box (exactly B) would be left with A, which it only matches at IoU 0.67.
    assert score([A, B], [[12, 0, 111, 99], B], [0.1, 0.9]) == pytest.approx(1.0)


def test_no_ground_truth_rejected():
    with pytest.raises(ValueError):
        score(np.zeros((0, 4)), [], [])
