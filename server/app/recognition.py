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
_uses_embedding: bool = False  # True when the learned EmbeddingMatcher is active


def init_matcher(templates_dir: str | None = None, scale: float | None = None) -> tuple[str, ...]:
    """Load templates and build the shared matcher. Returns the gloss set.

    asl_matcher_mode: "auto" (default) uses the learned `EmbeddingMatcher` when its
    encoder + prototypes files exist, else the DTW `Matcher`; "embedding" forces the
    learned one (warns + falls back to DTW if artifacts are missing); "dtw" forces
    DTW. The embedding glosses come from the prototypes file, not the templates dir.
    Raises (FileNotFoundError/ValueError) if DTW templates are missing — callers
    decide whether that's fatal."""
    global _matcher, _glosses, _uses_embedding
    s = get_settings()

    mode = getattr(s, "asl_matcher_mode", "auto")
    encoder_path = getattr(s, "asl_encoder_path", "")
    prototypes_path = getattr(s, "asl_prototypes_path", "")
    have_artifacts = bool(
        encoder_path and prototypes_path
        and os.path.exists(encoder_path) and os.path.exists(prototypes_path)
    )

    if mode != "dtw" and have_artifacts:
        # Imported lazily so the DTW path never requires onnxruntime.
        from asl.embedding_matcher import EmbeddingMatcher

        _matcher = EmbeddingMatcher.from_files(encoder_path, prototypes_path)
        _glosses = tuple(sorted(_matcher._protos))
        _uses_embedding = True
        logger.info("ASL embedding matcher loaded: %d glosses", len(_glosses))
        return _glosses
    if mode == "embedding":
        logger.warning(
            "asl_matcher_mode=embedding but encoder/prototypes missing (%r / %r); "
            "falling back to DTW matcher", encoder_path, prototypes_path,
        )

    _uses_embedding = False
    templates = load_templates(templates_dir or s.asl_templates_dir)
    _matcher = Matcher(templates, scale=scale if scale is not None else s.asl_scale)
    _glosses = tuple(sorted(templates))
    logger.info("ASL matcher loaded: %d glosses", len(_glosses))
    return _glosses


def uses_embedding() -> bool:
    return _uses_embedding


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
        # Embedding prototypes are scored on hand-xy features, so couple the live
        # reduction to the active matcher: embedding -> "hands", DTW -> the config
        # default ("full" = the 147-dim frame).
        self._feature_mode = "hands" if _uses_embedding else getattr(s, "asl_feature_mode", "full")

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
            "rank_gate": getattr(s, "asl_rank_gate", 0) or None,
            "min_confirm_interval": getattr(s, "asl_min_confirm_interval", 0.0) or None,
        }
        accepted = inspect.signature(Session.__init__).parameters
        for name, value in advanced.items():
            if name in accepted:
                kwargs[name] = value

        if _uses_embedding:
            # The DTW-tuned confirm params are WRONG for the embedding matcher.
            # Its strength is (cosine + 1) / 2, which has a high positive baseline
            # (WRONG signs average ~0.63), so peak >= 0.6 is met the instant a hand
            # is in frame and the dip/overtake heuristics then fire on the noisy
            # baseline -> every prompt confirms immediately. Confirm on a SUSTAINED
            # HIGH HOLD instead: a high threshold with the warmup gate on, and the
            # dip + overtake paths neutralized (a held correct sign does not dip,
            # and a noisy next-target must not overtake). An 80-gloss held-out sweep
            # moved wrong-sign false-accept 81% -> ~12% while keeping true-accept
            # ~89%. Any param the operator set explicitly via env still wins.
            # The open-set rank gate (top-2) additionally requires the prompt to be
            # the best-matching gloss, so generic/arbitrary hand motion that merely
            # scores high on the prompt no longer passes.
            # window_size MUST match train.dataset.WINDOW_SIZE (16): the encoder and
            # its prototypes are built from 16-frame windows, so feeding the DTW-era
            # 48-frame buffer compares mismatched temporal extents (measured: 16 vs
            # 48 recovered true-accept 53% -> 69% at the same ~1% false-accept).
            # Hardcoded (not imported) because train.dataset pulls torch, absent from
            # the server image; keep in sync if WINDOW_SIZE changes (=> retrain).
            emb = {
                # Tuned for forgiveness (the encoder under-scores a real signer's
                # own signing vs WLASL prototypes): a lower strength bar, a wider
                # open-set rank window, and a shorter hold. Still gated by motion +
                # hand presence so arbitrary hands don't pass. Tighten via the
                # ASL_* env vars if it over-accepts.
                "asl_get_threshold": ("get_threshold", 0.82),
                "asl_confirm_hold": ("confirm_hold", 5),
                "asl_warmup_frames": ("warmup_frames", 10),
                "asl_confirm_drop": ("confirm_drop", -1.0),       # disable dip
                "asl_overtake_frames": ("overtake_frames", 10**9),  # disable overtake
                "asl_rank_gate": ("rank_gate", 6),                # prompt must be top-6
                "asl_window_size": ("window_size", 16),           # == train WINDOW_SIZE
                "asl_min_confirm_interval": ("min_confirm_interval", 2.0),  # >=2s apart
                # Auto-fail a word after 10s so an unrecognized sign doesn't block
                # the stream forever. Valid now that the live `t` is in SECONDS (the
                # landmark provider divides performance.now() by 1000); before that
                # fix a 6s budget mis-fired on the 2nd frame ("passes instantly").
                "asl_miss_budget": ("miss_budget", 10.0),
            }
            explicit = getattr(s, "model_fields_set", set())
            for field, (kw, value) in emb.items():
                if kw in kwargs and field not in explicit:
                    kwargs[kw] = value

        self._session = Session(
            matcher or get_matcher(),
            words.word_iter(seed),
            **kwargs,
        )
        # Opt-in backend tracing (ASL_DEBUG=1): logs the active params once, then a
        # per-frame line (hand presence, live feature range, strength, window
        # motion, peak/hold/warmup, top-3 open-set ranking, confirm event) so the
        # REAL live data — not cached clips — can be inspected.
        self._debug = os.environ.get("ASL_DEBUG") == "1"
        if self._debug:
            logger.info(
                "RECO new session seed=%d embedding=%s feature=%s window=%s "
                "get_threshold=%s confirm_hold=%s warmup=%s rank_gate=%s min_interval=%s",
                seed, _uses_embedding, self._feature_mode, kwargs.get("window_size"),
                kwargs.get("get_threshold"), kwargs.get("confirm_hold"),
                kwargs.get("warmup_frames"), kwargs.get("rank_gate"),
                kwargs.get("min_confirm_interval"),
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
        if self._debug:
            sess = self._session
            buf = sess._buffer
            win = np.array(buf) if len(buf) else None
            motion = (float(np.mean(np.std(win, axis=0)))
                      if win is not None and win.ndim == 2 and len(win) > 1 else 0.0)
            top = []
            mch = sess._matcher
            if win is not None and win.ndim == 2 and hasattr(mch, "rank"):
                try:
                    top = [(r["gloss"], r["distance"]) for r in mch.rank(win, 3)]
                except Exception as exc:  # never let tracing break recognition
                    top = [("rank_error", str(exc))]
            logger.info(
                "RECO t=%.2f win=%d motion=%.4f tgt=%r str=%.3f peak=%.3f hold=%d "
                "warm=%d/%d event=%s top3=%s",
                t, len(buf), motion, word, state.strength, sess._peak, sess._hold,
                sess._frames_since_advance, sess._warmup_frames, state.event, top,
            )
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
        # Hands-only matching is meaningless without a hand. An undetected hand
        # arrives as None; assemble_frame would zero-fill it, and a no-hand frame
        # normalizes (against the jittering shoulders) into spurious "motion" that
        # the encoder maps to a ~1.0 cosine -> the prompt confirms with no hands in
        # frame. Require at least one detected hand, checked on the RAW input so
        # shoulder jitter can't disguise an empty frame as movement. (The "full"
        # DTW mode still accepts pose-only frames, preserving its behavior.)
        if self._feature_mode == "hands" and not hand_left and not hand_right:
            if self._debug:
                logger.info("RECO t=%.2f REJECT no-hands (L=%s R=%s pose=%s)",
                            t, bool(hand_left), bool(hand_right), pose is not None)
            return None
        try:
            frame = assemble_frame(pose, hand_left, hand_right)
            row = normalize_frame(frame).flatten()
            if self._feature_mode == "hands":
                # Imported lazily so the default 147-dim path never depends on
                # match_features being present in asl.schema.
                from asl.schema import match_features

                row = match_features(row)
        except (ValueError, TypeError) as exc:
            if self._debug:
                logger.info("RECO t=%.2f REJECT unusable: %s", t, exc)
            return None
        if self._debug:
            logger.info("RECO t=%.2f frame L=%s R=%s feat[min=%.3f max=%.3f mean=%.3f]",
                        t, bool(hand_left), bool(hand_right),
                        float(row.min()), float(row.max()), float(row.mean()))
        return self.push_frame(row, t)
