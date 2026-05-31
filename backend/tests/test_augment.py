import numpy as np
import pytest

from train.augment import (
    add_noise,
    augment,
    frame_dropout,
    spatial_jitter,
    temporal_crop,
    time_warp,
)


def _seq(T=20):
    rng = np.random.default_rng(0)
    return rng.standard_normal((T, 84)).astype(np.float32)


def test_time_warp_resamples_length():
    seq = _seq(20)
    out = time_warp(seq, 2.0)
    assert out.shape == (40, 84)
    out = time_warp(seq, 0.5)
    assert out.shape == (10, 84)


def test_time_warp_identity_factor_preserves_values():
    seq = _seq(15)
    out = time_warp(seq, 1.0)
    assert out.shape == seq.shape
    assert np.allclose(out, seq, atol=1e-4)


def test_time_warp_single_frame_safe():
    seq = _seq(1)
    out = time_warp(seq, 2.0)
    assert out.shape[0] >= 1
    assert out.shape[1] == 84


def test_frame_dropout_keeps_at_least_one():
    rng = np.random.default_rng(1)
    out = frame_dropout(_seq(10), 1.0, rng)  # p=1 would drop all
    assert out.shape[0] >= 1
    assert out.shape[1] == 84


def test_frame_dropout_reduces_or_keeps_length():
    rng = np.random.default_rng(2)
    seq = _seq(50)
    out = frame_dropout(seq, 0.3, rng)
    assert 1 <= out.shape[0] <= 50


def test_temporal_crop_length():
    rng = np.random.default_rng(3)
    out = temporal_crop(_seq(30), 10, rng)
    assert out.shape == (10, 84)


def test_temporal_crop_longer_than_seq_returns_all():
    rng = np.random.default_rng(4)
    seq = _seq(8)
    out = temporal_crop(seq, 50, rng)
    assert out.shape == (8, 84)


def test_spatial_jitter_shape_and_finite():
    rng = np.random.default_rng(5)
    out = spatial_jitter(_seq(12), rng)
    assert out.shape == (12, 84)
    assert np.isfinite(out).all()


def test_spatial_jitter_zero_magnitude_is_identity():
    rng = np.random.default_rng(6)
    seq = _seq(12)
    out = spatial_jitter(seq, rng, max_shift=0.0, max_scale=0.0, max_rot_deg=0.0)
    assert out.shape == seq.shape
    assert np.allclose(out, seq, atol=1e-5)


def test_spatial_jitter_deterministic_given_rng():
    a = spatial_jitter(_seq(12), np.random.default_rng(7))
    b = spatial_jitter(_seq(12), np.random.default_rng(7))
    assert np.allclose(a, b)


def test_spatial_jitter_preserves_absent_hand_blocks():
    # Absence is encoded by all-zero 21-point hand blocks; the affine shift must
    # not turn those into fake non-zero hand positions.
    rng = np.random.default_rng(12)
    seq = _seq(12)
    seq[:, :42] = 0.0          # left hand absent on every frame
    seq[3, 42:] = 0.0          # right hand absent on one frame only
    out = spatial_jitter(seq, rng)
    assert np.all(out[:, :42] == 0.0)
    assert np.all(out[3, 42:] == 0.0)
    # Present hand frames must still have moved (non-identity affine).
    assert not np.allclose(out[0, 42:], seq[0, 42:])


def test_add_noise_changes_values_but_close():
    rng = np.random.default_rng(8)
    seq = _seq(10)
    out = add_noise(seq, 0.01, rng)
    assert out.shape == seq.shape
    assert not np.allclose(out, seq)
    assert np.abs(out - seq).max() < 0.2


def test_augment_never_empty_or_nan():
    rng = np.random.default_rng(9)
    for _ in range(50):
        out = augment(_seq(np.random.default_rng(_).integers(2, 40)), rng)
        assert out.shape[0] >= 1
        assert out.shape[1] == 84
        assert np.isfinite(out).all()


def test_augment_deterministic_given_rng():
    a = augment(_seq(25), np.random.default_rng(11))
    b = augment(_seq(25), np.random.default_rng(11))
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_rejects_wrong_width():
    with pytest.raises(ValueError):
        time_warp(np.zeros((5, 10), dtype=np.float32), 1.5)
