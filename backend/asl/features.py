import numpy as np

# Keypoint indices within a frame of shape (N, 3).
LEFT_SHOULDER = 0
RIGHT_SHOULDER = 1


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Re-center a frame on the shoulder midpoint and rescale by shoulder width.

    Mirrors pose-format's shoulder-reference normalization so live frames and the
    offline-built reference templates share one coordinate space.

    Args:
        frame: array of shape (N, 3) of (x, y, z) keypoints.

    Returns:
        A new (N, 3) array, translation- and scale-invariant.

    Raises:
        ValueError: if the shoulders coincide (near-zero width).
    """
    frame = np.asarray(frame, dtype=float)
    midpoint = (frame[LEFT_SHOULDER] + frame[RIGHT_SHOULDER]) / 2.0
    width = float(np.linalg.norm(frame[RIGHT_SHOULDER] - frame[LEFT_SHOULDER]))
    if width < 1e-9:
        raise ValueError("shoulder width is near zero; cannot normalize frame")
    return (frame - midpoint) / width
