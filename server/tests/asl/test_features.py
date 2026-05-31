import numpy as np
import pytest
from asl.features import normalize_frame, LEFT_SHOULDER, RIGHT_SHOULDER


def _frame(shoulder_l, shoulder_r, extra):
    pts = np.zeros((3, 3), dtype=float)
    pts[LEFT_SHOULDER] = shoulder_l
    pts[RIGHT_SHOULDER] = shoulder_r
    pts[2] = extra
    return pts


def test_centers_on_shoulder_midpoint():
    f = _frame([0, 0, 0], [2, 0, 0], [1, 1, 0])
    out = normalize_frame(f)
    mid = (out[LEFT_SHOULDER] + out[RIGHT_SHOULDER]) / 2
    assert np.allclose(mid, [0, 0, 0])


def test_translation_invariance():
    base = _frame([0, 0, 0], [2, 0, 0], [1, 1, 0])
    shifted = _frame([5, 7, 0], [7, 7, 0], [6, 8, 0])
    assert np.allclose(normalize_frame(base), normalize_frame(shifted))


def test_scale_invariance():
    base = _frame([0, 0, 0], [2, 0, 0], [1, 1, 0])
    scaled = _frame([0, 0, 0], [4, 0, 0], [2, 2, 0])  # 2x shoulder width
    assert np.allclose(normalize_frame(base), normalize_frame(scaled))


def test_zero_shoulder_width_raises():
    degenerate = _frame([1, 1, 0], [1, 1, 0], [2, 2, 0])
    with pytest.raises(ValueError):
        normalize_frame(degenerate)


def test_near_zero_shoulder_width_raises():
    degenerate = _frame([1, 1, 0], [1 + 1e-12, 1, 0], [2, 2, 0])
    with pytest.raises(ValueError):
        normalize_frame(degenerate)
