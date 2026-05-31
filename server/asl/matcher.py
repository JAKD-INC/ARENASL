import math
import numpy as np
from asl.dtw import dtw_distance


class Matcher:
    """Scores a live window against a target sign's reference templates.

    Stateless w.r.t. the game: it only turns (window, target) into a strength.
    Plan 2's embedding matcher can replace it behind this same interface.
    """

    def __init__(self, templates: dict[str, list[np.ndarray]], scale: float = 1.0):
        """
        Args:
            templates: {gloss: [sequence, ...]} of normalized landmark sequences,
                each shape (T, D).
            scale: DTW distance at which strength falls to 1/e. Larger = lenient.
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        self._templates = templates
        self._scale = scale

    def strength(self, window: np.ndarray, target: str) -> float:
        """Return match strength in [0, 1] of `window` against `target`."""
        exemplars = self._templates[target]  # KeyError if unknown target
        best = min(dtw_distance(window, t) for t in exemplars)
        return math.exp(-best / self._scale)
