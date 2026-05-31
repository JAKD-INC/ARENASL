import numpy as np
import pytest
from asl.schema import assemble_frame, N_KEYPOINTS, POSE_INDICES


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
