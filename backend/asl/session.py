from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional, Protocol
import numpy as np


class SupportsStrength(Protocol):
    def strength(self, window: np.ndarray, target: str) -> float: ...


@dataclass
class State:
    """Snapshot sent to the consumer after each frame."""
    current: str            # the sign the player must perform now
    queue: list[str]        # upcoming signs (lookahead preview)
    strength: float         # latest match strength for `current`, in [0, 1]
    score: int              # running score
    event: Optional[str]    # "get", "miss", or None this frame
    confirmed: Optional[str]  # gloss confirmed THIS frame (on "get"), else None
    distance: Optional[float] = None  # raw best DTW distance (debug/calibration)


class Session:
    """Server-authoritative typerace loop with sign-level peak segmentation."""

    def __init__(
        self,
        matcher: SupportsStrength,
        prompts: Iterator[str],
        *,
        get_threshold: float,
        confirm_drop: float,
        miss_budget: Optional[float],
        window_size: int,
        get_points: int = 20,
        miss_points: int = -10,
        lookahead: int = 3,
        overtake_frames: int = 2,
    ):
        if lookahead < 1:
            # push() probes queue[1] (the next target) every frame for the
            # overtake path; lookahead < 1 leaves the queue with < 2 entries.
            raise ValueError("lookahead must be >= 1")
        if overtake_frames < 1:
            # The overtake path needs at least one frame of evidence; requiring
            # >= 2 rejects single-frame next-target noise spikes.
            raise ValueError("overtake_frames must be >= 1")
        self._matcher = matcher
        self._prompts = prompts
        self._get_threshold = get_threshold
        self._confirm_drop = confirm_drop
        self._miss_budget = miss_budget
        self._get_points = get_points
        self._miss_points = miss_points
        self._overtake_frames = overtake_frames

        self._queue: deque[str] = deque(
            next(prompts) for _ in range(lookahead + 1)
        )
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._score = 0
        self._peak = 0.0
        self._overtake = 0
        self._target_start: Optional[float] = None

    def _advance(self) -> None:
        self._queue.popleft()
        self._queue.append(next(self._prompts))
        self._buffer.clear()
        self._peak = 0.0
        self._overtake = 0
        self._target_start = None

    def state(
        self,
        strength: float = 0.0,
        event: Optional[str] = None,
        confirmed: Optional[str] = None,
        distance: Optional[float] = None,
    ) -> State:
        q = list(self._queue)
        return State(
            current=q[0], queue=q[1:],
            strength=strength, score=self._score, event=event,
            confirmed=confirmed, distance=distance,
        )

    def push(self, frame: np.ndarray, t: float) -> State:
        """Feed one normalized frame at time `t` (seconds); return new state."""
        if self._target_start is None:
            self._target_start = t
        self._buffer.append(frame)

        window = np.array(self._buffer)
        strength = self._matcher.strength(window, self._queue[0])
        self._peak = max(self._peak, strength)
        # Raw distance for debug/calibration (real Matcher only; stubs omit it).
        distance = (self._matcher.best_distance(window, self._queue[0])
                    if hasattr(self._matcher, "best_distance") else None)

        # Strength of the NEXT target this frame: a fluent signer who never
        # pauses produces no dip on the current sign, but the next sign starts
        # out-matching the current one the moment they move on.
        next_strength = self._matcher.strength(window, self._queue[1])

        # Count CONSECUTIVE frames where the next target out-matches the
        # current one. A single-frame next-target noise spike (while a clean
        # current sign holds above threshold) must not confirm, so we require
        # `overtake_frames` consecutive overtaking frames; one good frame
        # below the bar resets the run.
        if next_strength > strength:
            self._overtake += 1
        else:
            self._overtake = 0
        sustained_overtake = self._overtake >= self._overtake_frames

        # Sign confirmed once it peaked above threshold AND the signer moved on,
        # detected EITHER by a dip from the peak OR by the next target SUSTAINED
        # out-matching the current target across consecutive frames (overtake).
        if self._peak >= self._get_threshold and (
            strength <= self._peak * self._confirm_drop
            or sustained_overtake
        ):
            gloss = self._queue[0]
            self._score += self._get_points
            self._advance()
            return self.state(strength=strength, event="get", confirmed=gloss,
                              distance=distance)

        # Timed auto-miss is opt-in: miss_budget=None disables it entirely, so a
        # prompt only advances when the sign is actually performed (no time limit).
        if self._miss_budget is not None and t - self._target_start >= self._miss_budget:
            self._score += self._miss_points
            self._advance()
            return self.state(strength=strength, event="miss", distance=distance)

        return self.state(strength=strength, event=None, distance=distance)
