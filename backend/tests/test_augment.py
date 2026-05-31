import numpy as np
import pytest

from train.augment import (
    add_noise,
    augment,
    frame_dropout,
    horizontal_flip,
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


def test_horizontal_flip_twice_is_identity():
    seq = _seq(12)
    out = horizontal_flip(horizontal_flip(seq))
    assert out.shape == seq.shape
    assert np.allclose(out, seq, atol=1e-6)


def test_horizontal_flip_swaps_hand_blocks_and_mirrors_x():
    # Left-hand block = dims 0:42, right-hand block = dims 42:84; x are even dims,
    # y are odd. A mirror swaps the blocks together AND negates x.
    seq = _seq(5)
    out = horizontal_flip(seq)
    # The mirrored old left hand (x negated) now occupies the right-hand slot.
    expected_right = seq[:, :42].copy()
    expected_right[:, 0::2] = -expected_right[:, 0::2]
    assert np.allclose(out[:, 42:], expected_right, atol=1e-6)
    # The mirrored old right hand (x negated) now occupies the left-hand slot.
    expected_left = seq[:, 42:].copy()
    expected_left[:, 0::2] = -expected_left[:, 0::2]
    assert np.allclose(out[:, :42], expected_left, atol=1e-6)


def test_horizontal_flip_preserves_absent_hand_blocks():
    # Absence is an all-zero 21-point block; the flip must not invent a hand.
    seq = _seq(6)
    seq[:, :42] = 0.0  # left hand absent on every frame
    out = horizontal_flip(seq)
    # Absent left block lands in the right slot, still all-zero.
    assert np.all(out[:, 42:] == 0.0)


def test_augment_flip_prob_zero_never_flips():
    # flip_prob=0 must be reproducible run-to-run.
    a = augment(_seq(20), np.random.default_rng(3), flip_prob=0.0)
    b = augment(_seq(20), np.random.default_rng(3), flip_prob=0.0)
    assert np.allclose(a, b)


def test_augment_flip_prob_gates_the_flip():
    # augment() always consumes one rng.random() for the flip decision regardless
    # of flip_prob, so with the SAME seed the ONLY difference between flip_prob=0
    # and flip_prob=1 is whether horizontal_flip runs before the rest of the
    # pipeline (downstream rng draws are identical). The two results must differ,
    # proving the flip is genuinely conditional (not always-on, not always-off).
    seq = _seq(20)
    never = augment(seq, np.random.default_rng(42), flip_prob=0.0)
    always = augment(seq, np.random.default_rng(42), flip_prob=1.0)
    assert never.shape == always.shape
    assert not np.allclose(never, always)
    # flip_prob=1 with the same seed equals flip_prob=0 fed the pre-flipped input,
    # confirming flip_prob=0 leaves the hand blocks / x-coords unmirrored.
    pre = augment(horizontal_flip(seq), np.random.default_rng(42), flip_prob=0.0)
    assert np.allclose(always, pre)


def test_augment_deterministic_with_flip_default_on():
    a = augment(_seq(20), np.random.default_rng(99))
    b = augment(_seq(20), np.random.default_rng(99))
    assert a.shape == b.shape
    assert np.allclose(a, b)


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
        # Draw a genuinely random length from the shared rng each iteration
        # (not a deterministic length derived from the loop counter).
        out = augment(_seq(int(rng.integers(2, 40))), rng)
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
