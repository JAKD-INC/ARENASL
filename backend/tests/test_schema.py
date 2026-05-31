import numpy as np
import pytest
from asl.schema import (
    assemble_frame,
    match_features,
    HAND_XY_COLS,
    N_KEYPOINTS,
    POSE_INDICES,
)


def _pose():  # 33 distinct points so we can verify index mapping
    return [[float(i), float(i) + 0.5, 0.0] for i in range(33)]


def _hand(base):  # 21 points
    return [[base + j * 0.01, base, 0.0] for j in range(21)]


def test_shape_and_shoulders_first():
    frame = assemble_frame(_pose(), _hand(1.0), _hand(2.0))
    assert frame.shape == (N_KEYPOINTS, 3)
    # idx 0/1 must be pose 11/12 (shoulders)
    assert frame[0, 0] == 11.0
    assert frame[1, 0] == 12.0


def test_pose_subset_mapped_in_order():
    frame = assemble_frame(_pose(), _hand(1.0), _hand(2.0))
    for slot, pose_idx in enumerate(POSE_INDICES):
        assert frame[slot, 0] == float(pose_idx)


def test_hands_placed_in_slots():
    frame = assemble_frame(_pose(), _hand(1.0), _hand(2.0))
    assert frame[7, 0] == pytest.approx(1.0)    # first left-hand point
    assert frame[28, 0] == pytest.approx(2.0)   # first right-hand point


def test_missing_hands_zero_filled():
    frame = assemble_frame(_pose(), None, None)
    assert np.all(frame[7:28] == 0.0)   # left hand slots
    assert np.all(frame[28:49] == 0.0)  # right hand slots
    assert frame[0, 0] == 11.0          # pose still present


def test_missing_pose_raises():
    with pytest.raises(ValueError):
        assemble_frame(None, _hand(1.0), _hand(2.0))


def test_match_features_selects_from_full_frame():
    # Full (T, 147) sequence -> hand-xy (T, 84) by HAND_XY_COLS selection.
    seq = np.arange(3 * N_KEYPOINTS * 3, dtype=float).reshape(3, N_KEYPOINTS * 3)
    out = match_features(seq)
    assert out.shape == (3, len(HAND_XY_COLS))
    np.testing.assert_array_equal(out, seq[..., HAND_XY_COLS])


def test_match_features_idempotent_on_84():
    # Already-reduced (T, 84) input is returned unchanged (as float).
    seq = np.arange(5 * len(HAND_XY_COLS), dtype=np.float32).reshape(5, len(HAND_XY_COLS))
    out = match_features(seq)
    assert out.shape == seq.shape
    np.testing.assert_array_equal(out, seq.astype(float))
    # Double application is a no-op too.
    np.testing.assert_array_equal(match_features(out), out)


def test_match_features_84_float64_returned_unchanged_no_copy():
    # A float64 (T, 84) input is already hand-xy: returned as the SAME object
    # (no copy), proving the idempotency guard short-circuits before selection.
    seq = np.arange(4 * len(HAND_XY_COLS), dtype=np.float64).reshape(4, len(HAND_XY_COLS))
    out = match_features(seq)
    assert out is seq
    np.testing.assert_array_equal(match_features(out), out)


def test_match_features_idempotent_across_boundary():
    # Starting from a full (T, 147) frame, applying match_features twice is a
    # no-op past the first reduction: 147 -> 84 (select) -> 84 (unchanged).
    seq = np.arange(3 * N_KEYPOINTS * 3, dtype=float).reshape(3, N_KEYPOINTS * 3)
    once = match_features(seq)
    twice = match_features(once)
    assert once.shape == (3, len(HAND_XY_COLS))
    np.testing.assert_array_equal(twice, once)


def test_match_features_bad_dim_raises():
    with pytest.raises(ValueError):
        match_features(np.zeros((2, 50)))
