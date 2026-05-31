"""Server-side ASL recognition wired to OUR seeded word stream.

A process-wide `Matcher` is built once from the WLASL templates at startup. Each
player gets a `RecognitionSession` that runs the (vendored) `asl.Session`
peak/overtake segmentation against the words our stream emits, and reports a
`"get"`/`"miss"`/`None` outcome the duel engine turns into damage (phase 2c).

The Session's DTW work is CPU-bound; callers offload `push_*` to a thread pool
(phase 2c) so it never blocks the event loop.
"""

from __future__ import annotations

import inspect
import logging
import os
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

# The shared matcher is either the DTW `Matcher` (default) or the advanced
# `EmbeddingMatcher` (opt-in via asl_matcher_mode=="embedding"); both expose the
# same (window, target) -> strength interface, so everything downstream is typed
# against `Matcher` for back-compat.
_matcher: Matcher | None = None
_glosses: tuple[str, ...] = ()


def init_matcher(templates_dir: str | None = None, scale: float | None = None) -> tuple[str, ...]:
    """Load templates and build the shared matcher. Returns the gloss set.

    By default (asl_matcher_mode=="dtw") builds the DTW `Matcher` from the WLASL
    templates exactly as before. When asl_matcher_mode=="embedding" AND both the
    encoder and prototypes files exist, builds the learned `EmbeddingMatcher`
    instead (its glosses come from the prototypes file, not the templates dir).
    Raises (FileNotFoundError/ValueError) if templates are missing — callers
    decide whether that's fatal."""
    global _matcher, _glosses
    s = get_settings()

    if getattr(s, "asl_matcher_mode", "dtw") == "embedding":
        encoder_path = getattr(s, "asl_encoder_path", "")
        prototypes_path = getattr(s, "asl_prototypes_path", "")
        if encoder_path and prototypes_path and os.path.exists(encoder_path) and os.path.exists(prototypes_path):
            # Imported lazily so the default DTW path never requires onnxruntime
            # (nor the embedding_matcher module being present).
            from asl.embedding_matcher import EmbeddingMatcher

            _matcher = EmbeddingMatcher.from_files(encoder_path, prototypes_path)
            _glosses = tuple(sorted(_matcher._protos))
            logger.info("ASL embedding matcher loaded: %d glosses", len(_glosses))
            return _glosses
        logger.warning(
            "asl_matcher_mode=embedding but encoder/prototypes missing "
            "(%r / %r); falling back to DTW matcher",
            encoder_path,
            prototypes_path,
        )

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
        # "full" (default) feeds the full 147-dim frame; "hands" reduces the row
        # to the 84-dim hand-xy match features (see asl.schema.match_features).
        self._feature_mode = getattr(s, "asl_feature_mode", "full")

        kwargs = dict(
            get_threshold=s.asl_get_threshold,
            confirm_drop=s.asl_confirm_drop,
            miss_budget=s.asl_miss_budget,
            window_size=s.asl_window_size,
            overtake_frames=s.asl_overtake_frames,
        )
        # The advanced Session adds confirm_hold/warmup_frames/rank_every. The
        # fallbacks here equal the behavior-preserving config defaults
        # (confirm_hold effectively DISABLED via a large sentinel, warmup OFF,
        # ranking OFF), so today's dip/overtake/miss behavior is identical
        # whether or not config.py has been ported. They are passed only when
        # the installed Session accepts them, so this wiring is a no-op against
        # the older signature too.
        advanced = {
            "confirm_hold": getattr(s, "asl_confirm_hold", 100000),
            "warmup_frames": getattr(s, "asl_warmup_frames", 0),
            "rank_every": getattr(s, "asl_rank_every", None),
        }
        accepted = inspect.signature(Session.__init__).parameters
        for name, value in advanced.items():
            if name in accepted:
                kwargs[name] = value

        self._session = Session(
            matcher or get_matcher(),
            words.word_iter(seed),
            **kwargs,
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
        None for unusable frames (no pose / degenerate shoulders).

        In the default "full" feature mode the row is the full 147-dim frame. In
        "hands" mode it is reduced to the 84-dim hand-xy match features (matching
        what the offline-built embedding prototypes are scored against)."""
        try:
            frame = assemble_frame(pose, hand_left, hand_right)
            row = normalize_frame(frame).flatten()
            if self._feature_mode == "hands":
                # Imported lazily so the default 147-dim path never depends on
                # match_features being present in asl.schema.
                from asl.schema import match_features

                row = match_features(row)
        except (ValueError, TypeError):
            return None
        return self.push_frame(row, t)
