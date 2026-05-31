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

    def best_distance(self, window: np.ndarray, target: str) -> float:
        """Smallest DTW distance from `window` to any exemplar of `target`."""
        exemplars = self._templates[target]  # KeyError if unknown target
        return min(dtw_distance(window, t) for t in exemplars)

    def strength(self, window: np.ndarray, target: str) -> float:
        """Return match strength in [0, 1] of `window` against `target`."""
        return math.exp(-self.best_distance(window, target) / self._scale)

    def rank(self, window: np.ndarray, k: int = 3) -> list[dict]:
        """Debug: the k closest glosses to `window` by best DTW distance.
        Reveals whether the sign being performed is actually the nearest match."""
        scored = sorted((self.best_distance(window, g), g) for g in self._templates)
        return [{"gloss": g, "distance": round(d, 3)} for d, g in scored[:k]]
