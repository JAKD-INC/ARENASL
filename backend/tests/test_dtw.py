import numpy as np
import pytest
from asl.dtw import dtw_distance


def test_identical_sequences_zero_distance():
    seq = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert dtw_distance(seq, seq) == 0.0


def test_time_warped_copy_is_zero():
    seq = np.array([[0.0], [1.0], [2.0]])
    warped = np.array([[0.0], [1.0], [1.0], [2.0]])
    assert dtw_distance(seq, warped) == 0.0


def test_offset_sequence_has_positive_distance():
    seq = np.array([[0.0], [1.0], [2.0]])
    shifted = np.array([[5.0], [6.0], [7.0]])
    assert dtw_distance(seq, shifted) > 0.0


def test_closer_sequence_scores_lower():
    seq = np.array([[0.0], [1.0], [2.0]])
    near = np.array([[0.1], [1.1], [2.1]])
    far = np.array([[3.0], [4.0], [5.0]])
    assert dtw_distance(seq, near) < dtw_distance(seq, far)


def test_length_normalization_removes_length_bias():
    a = np.array([[0.0], [1.0], [2.0]])
    b = np.array([[0.5], [1.5], [2.5]])
    base = dtw_distance(a, b)

    a2 = np.repeat(a, 2, axis=0)
    b2 = np.repeat(b, 2, axis=0)
    doubled = dtw_distance(a2, b2)

    assert doubled == pytest.approx(base)
