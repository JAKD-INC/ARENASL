"""Server-side ASL recognition wired to OUR seeded word stream.

A process-wide `Matcher` is built once from the WLASL templates at startup. Each
player gets a `RecognitionSession` that runs the (vendored) `asl.Session`
peak/overtake segmentation against the words our stream emits, and reports a
`"get"`/`"miss"`/`None` outcome the duel engine turns into damage (phase 2c).

The Session's DTW work is CPU-bound; callers offload `push_*` to a thread pool
(phase 2c) so it never blocks the event loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app import words
from app.config import get_settings
from asl.features import normalize_frame
from asl.library import load_templates
from asl.matcher import Matcher
from asl.schema import assemble_frame
from asl.session import Session

logger = logging.getLogger("arenasl.recognition")

_matcher: Matcher | None = None
_glosses: tuple[str, ...] = ()


def init_matcher(templates_dir: str | None = None, scale: float | None = None) -> tuple[str, ...]:
    """Load templates and build the shared matcher. Returns the gloss set.
    Raises (FileNotFoundError/ValueError) if templates are missing — callers
    decide whether that's fatal."""
    global _matcher, _glosses
    s = get_settings()
    templates = load_templates(templates_dir or s.asl_templates_dir)
    _matcher = Matcher(templates, scale=scale if scale is not None else s.asl_scale)
    _glosses = tuple(sorted(templates))
    logger.info("ASL matcher loaded: %d glosses", len(_glosses))
    return _glosses


def get_matcher() -> Matcher:
    if _matcher is None:
        raise RuntimeError("ASL matcher not initialized")
    return _matcher


def available_glosses() -> tuple[str, ...]:
    return _glosses


def is_ready() -> bool:
    return _matcher is not None


def new_session(seed: int) -> "RecognitionSession":
    """Factory used by the duel to create a per-player recognizer (tests patch this)."""
    return RecognitionSession(seed)


def reset() -> None:
    """Test helper: forget the loaded matcher."""
    global _matcher, _glosses
    _matcher = None
    _glosses = ()


@dataclass
class Outcome:
    """Result of pushing one frame. On get/miss, `word`/`word_index` identify the
    word that just completed/expired; otherwise they're the current target."""

    event: str | None  # "get" | "miss" | None
    word: str
    word_index: int
    strength: float


class RecognitionSession:
    """Per-player recognition over the seeded stream `seed`. Two players sharing a
    seed see the same words but progress independently."""

    def __init__(self, seed: int, matcher: Matcher | None = None):
        s = get_settings()
        self._seed = seed
        self._index = 0
        self._session = Session(
            matcher or get_matcher(),
            words.word_iter(seed),
            get_threshold=s.asl_get_threshold,
            confirm_drop=s.asl_confirm_drop,
            miss_budget=s.asl_miss_budget,
            window_size=s.asl_window_size,
            overtake_frames=s.asl_overtake_frames,
        )

    @property
    def word_index(self) -> int:
        return self._index

    def push_frame(self, frame_row: np.ndarray, t: float) -> Outcome:
        """Feed one normalized, flattened (D,) frame at time `t` (seconds)."""
        state = self._session.push(np.asarray(frame_row, dtype=float), t)
        # The word that just completed/expired is at the pre-advance index; the
        # deterministic stream lets us name it without trusting Session internals.
        word = words.word_at(self._seed, self._index).word
        if state.event in ("get", "miss"):
            outcome = Outcome(state.event, word, self._index, state.strength)
            self._index += 1
            return outcome
        return Outcome(None, word, self._index, state.strength)

    def push_landmarks(self, pose, hand_left, hand_right, t: float) -> Outcome | None:
        """Assemble + normalize a MediaPipe landmark frame, then push it. Returns
        None for unusable frames (no pose / degenerate shoulders)."""
        try:
            frame = assemble_frame(pose, hand_left, hand_right)
            row = normalize_frame(frame).flatten()
        except (ValueError, TypeError):
            return None
        return self.push_frame(row, t)
