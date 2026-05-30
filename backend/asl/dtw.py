import numpy as np
from dtaidistance import dtw_ndim


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Path-length-normalized multivariate DTW distance between two sequences.

    Args:
        a: array of shape (T1, D).
        b: array of shape (T2, D).

    Returns:
        The dtaidistance path cost (an L2 norm over per-step costs) divided by
        the square root of the warping-path length. This yields the RMS per-step
        cost, which is length-invariant so longer or slower sequences are not
        penalized for length.
    """
    a = np.ascontiguousarray(a, dtype=np.double)
    b = np.ascontiguousarray(b, dtype=np.double)
    raw = dtw_ndim.distance_fast(a, b)
    path = dtw_ndim.warping_path(a, b)
    return raw / np.sqrt(len(path)) if path else 0.0
