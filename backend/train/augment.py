"""On-the-fly data augmentation for (T, 84) hands-xy training sequences.

Each function is pure numpy on a float32 (T, 84) array and deterministic given a
numpy Generator. These bake in the invariances the old DTW matcher had for free:
tempo (time_warp), partial windows / detector dropouts (frame_dropout,
temporal_crop), and camera framing / hand size / shake (spatial_jitter,
add_noise). No function ever returns an empty or NaN sequence.
"""
import numpy as np

N_HAND_POINTS = 42  # 2 hands * 21 keypoints; (T, 84) reshapes to (T, 42, 2)


def _as_seq(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 84:
        raise ValueError(f"expected (T, 84) sequence; got shape {seq.shape}")
    return seq


def time_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Resample to round(T*factor) frames via per-column linear interpolation.

    factor ~0.5..2.0 ("slow down / speed up the video") so signing tempo doesn't
    matter. Always returns at least one frame.
    """
    seq = _as_seq(seq)
    T = seq.shape[0]
    new_T = max(1, int(round(T * float(factor))))
    if T == 1 or new_T == 1:
        # Nothing to interpolate along; repeat the single frame.
        return np.repeat(seq[:1], new_T, axis=0).astype(np.float32)
    src = np.arange(T, dtype=np.float64)
    dst = np.linspace(0.0, T - 1, new_T)
    out = np.empty((new_T, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(dst, src, seq[:, c])
    return out


def frame_dropout(seq: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Drop each frame independently with probability p (detector dropouts).

    Always keeps at least one frame (if all would drop, keep one at random).
    """
    seq = _as_seq(seq)
    T = seq.shape[0]
    if T <= 1 or p <= 0.0:
        return seq.copy()
    keep = rng.random(T) >= p
    if not keep.any():
        keep[rng.integers(T)] = True
    return seq[keep].astype(np.float32)


def temporal_crop(seq: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Random contiguous subwindow of `length` frames (the sliding window often
    catches only part of a sign). If length >= T, return the whole sequence."""
    seq = _as_seq(seq)
    T = seq.shape[0]
    length = max(1, int(length))
    if length >= T:
        return seq.copy()
    start = int(rng.integers(0, T - length + 1))
    return seq[start:start + length].astype(np.float32)


def spatial_jitter(seq: np.ndarray, rng: np.random.Generator,
                   max_shift: float = 0.05, max_scale: float = 0.1,
                   max_rot_deg: float = 10.0) -> np.ndarray:
    """Apply ONE 2D affine (shift+scale+rotation) to all 42 hand points, constant
    across the whole sequence (different camera framing / hand size / shake)."""
    seq = _as_seq(seq)
    T = seq.shape[0]
    pts = seq.reshape(T, N_HAND_POINTS, 2)

    scale = 1.0 + float(rng.uniform(-max_scale, max_scale))
    theta = np.deg2rad(float(rng.uniform(-max_rot_deg, max_rot_deg)))
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    shift = rng.uniform(-max_shift, max_shift, size=2).astype(np.float32)

    # Absence is encoded by all-zero hand blocks (a hand can drop out on any
    # frame). The shift would turn those zeros into a constant non-zero point,
    # breaking the absent/present distinction and injecting fake hand position.
    # Remember which of the two 21-point hand blocks are all-zero per frame, then
    # restore them to zero after the affine so absence stays absence.
    hand0 = pts[:, :N_HAND_POINTS // 2, :]      # (T,21,2) left hand
    hand1 = pts[:, N_HAND_POINTS // 2:, :]      # (T,21,2) right hand
    absent0 = ~np.any(hand0, axis=(1, 2))       # (T,) True where left hand all-zero
    absent1 = ~np.any(hand1, axis=(1, 2))       # (T,) True where right hand all-zero

    # (T,42,2) @ (2,2)^T then scale and shift (one affine for the whole sequence).
    out = (pts * scale) @ rot.T + shift
    out[absent0, :N_HAND_POINTS // 2, :] = 0.0
    out[absent1, N_HAND_POINTS // 2:, :] = 0.0
    return out.reshape(T, 84).astype(np.float32)


def add_noise(seq: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add light i.i.d. Gaussian coordinate noise."""
    seq = _as_seq(seq)
    if sigma <= 0.0:
        return seq.copy()
    noise = rng.normal(0.0, sigma, size=seq.shape).astype(np.float32)
    return (seq + noise).astype(np.float32)


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compose a random subset of the augmentations. Each op is applied with ~50%
    probability so the encoder sees a wide spread of tempos/framings. Guaranteed
    non-empty and NaN-free."""
    seq = _as_seq(seq)
    if rng.random() < 0.5:
        seq = time_warp(seq, float(rng.uniform(0.5, 2.0)))
    if rng.random() < 0.5:
        seq = frame_dropout(seq, float(rng.uniform(0.0, 0.2)), rng)
    if rng.random() < 0.5 and seq.shape[0] > 2:
        # crop to 50-100% of the current length
        length = int(round(seq.shape[0] * rng.uniform(0.5, 1.0)))
        seq = temporal_crop(seq, length, rng)
    if rng.random() < 0.5:
        seq = spatial_jitter(seq, rng)
    if rng.random() < 0.5:
        seq = add_noise(seq, float(rng.uniform(0.0, 0.02)), rng)
    # Safety: never return empty or NaN.
    if seq.shape[0] == 0:
        seq = _as_seq(seq).reshape(0, 84)
        seq = np.zeros((1, 84), dtype=np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return seq
