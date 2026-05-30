import numpy as np
import pytest
from asl.matcher import Matcher


def _seq(values):
    return np.array([[v] for v in values], dtype=float)


def test_exact_match_strength_is_one():
    m = Matcher({"HELLO": [_seq([0, 1, 2])]}, scale=1.0)
    assert m.strength(_seq([0, 1, 2]), "HELLO") == pytest.approx(1.0)


def test_distant_window_has_low_strength():
    m = Matcher({"HELLO": [_seq([0, 1, 2])]}, scale=1.0)
    assert m.strength(_seq([10, 11, 12]), "HELLO") < 0.01


def test_strength_is_monotonic_in_distance():
    m = Matcher({"HELLO": [_seq([0, 1, 2])]}, scale=1.0)
    near = m.strength(_seq([0.1, 1.1, 2.1]), "HELLO")
    far = m.strength(_seq([2, 3, 4]), "HELLO")
    assert 1.0 > near > far > 0.0


def test_best_exemplar_wins():
    m = Matcher({"HELLO": [_seq([5, 6, 7]), _seq([0, 1, 2])]}, scale=1.0)
    assert m.strength(_seq([0, 1, 2]), "HELLO") == pytest.approx(1.0)


def test_unknown_target_raises():
    m = Matcher({"HELLO": [_seq([0, 1, 2])]}, scale=1.0)
    with pytest.raises(KeyError):
        m.strength(_seq([0, 1, 2]), "MISSING")
