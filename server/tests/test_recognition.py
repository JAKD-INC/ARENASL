"""RecognitionSession wiring (outcome mapping, word-index advance, landmark
assembly) via a scripted matcher, plus a real-DTW matcher load smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app import recognition, words
from app.config import get_settings
from app.recognition import RecognitionSession

D = 147  # 49 keypoints × 3
HANDS_D = 84  # 42 hand keypoints × (x, y); asl.schema.match_features output


class FakeMatcher:
    """Returns scheduled strengths in call order. Session calls strength() twice
    per frame (current target, then next target), so schedule in pairs."""

    def __init__(self, schedule):
        self._it = iter(schedule)

    def strength(self, window, target):
        return next(self._it, 0.0)


class WidthSpy:
    """Records the feature width of every window it is asked to score, so a test
    can assert which feature pipeline (147-dim full vs 84-dim hands) reached the
    matcher. Never confirms (always strength 0)."""

    def __init__(self):
        self.widths: list[int] = []

    def strength(self, window, target):
        self.widths.append(np.asarray(window).shape[-1])
        return 0.0


def _shoulders_pose():
    pose = [[0.0, 0.0, 0.0] for _ in range(33)]
    pose[11] = [0.0, 0.0, 0.0]  # left shoulder
    pose[12] = [1.0, 0.0, 0.0]  # right shoulder (width 1)
    return pose


def test_get_maps_to_word_and_advances(domain):
    # frame1/2 peak at 0.9, frame3 drops to 0.5 (< 0.9*0.8) -> "get".
    sess = RecognitionSession(seed=123, matcher=FakeMatcher([0.9, 0.0, 0.9, 0.0, 0.5, 0.0]))
    expected_word = words.word_at(123, 0).word

    o1 = sess.push_frame(np.zeros(D), t=0.0)
    o2 = sess.push_frame(np.zeros(D), t=0.1)
    o3 = sess.push_frame(np.zeros(D), t=0.2)

    assert o1.event is None and o2.event is None
    assert o3.event == "get"
    assert o3.word == expected_word
    assert o3.word_index == 0
    assert sess.word_index == 1  # advanced past the completed word


def test_miss_advances_with_no_get(domain):
    # Strength holds at 0.9 (never drops) but the word times out (miss_budget 6s).
    sess = RecognitionSession(seed=7, matcher=FakeMatcher([0.9, 0.0, 0.9, 0.0]))
    sess.push_frame(np.zeros(D), t=0.0)
    out = sess.push_frame(np.zeros(D), t=7.0)
    assert out.event == "miss"
    assert out.word_index == 0
    assert sess.word_index == 1


def test_push_landmarks_assembles_and_skips_unusable(domain):
    sess = RecognitionSession(seed=1, matcher=FakeMatcher([0.0] * 50))
    pose = [[0.0, 0.0, 0.0] for _ in range(33)]
    pose[11] = [0.0, 0.0, 0.0]  # left shoulder
    pose[12] = [1.0, 0.0, 0.0]  # right shoulder (width 1)

    out = sess.push_landmarks(pose, None, None, t=0.0)
    assert out is not None and out.event is None

    # No pose -> unusable frame -> None (degrades gracefully).
    assert sess.push_landmarks(None, None, None, t=0.1) is None


def test_init_matcher_loads_templates_and_scores(domain, tmp_path: Path):
    # A synthetic template; matching a window to itself => DTW 0 => strength ~1.
    template = np.random.RandomState(0).rand(8, D)
    np.save(tmp_path / "wave__0.npy", template)

    glosses = recognition.init_matcher(templates_dir=str(tmp_path))
    assert "wave" in glosses
    assert recognition.is_ready()

    strength = recognition.get_matcher().strength(template, "wave")
    assert strength > 0.99  # exact match


def test_default_feature_mode_pushes_full_147_dim(domain):
    # Behavior-preserving default: the full 147-dim frame reaches the matcher.
    spy = WidthSpy()
    sess = RecognitionSession(seed=1, matcher=spy)
    sess.push_landmarks(_shoulders_pose(), None, None, t=0.0)
    assert spy.widths and set(spy.widths) == {D}


def test_hands_feature_mode_reduces_to_84_dim(domain):
    # Opt-in: asl_feature_mode=="hands" applies match_features so the matcher
    # sees the 84-dim hand-xy features (what embedding prototypes are built on).
    get_settings.cache_clear()
    try:
        import os

        os.environ["ASL_FEATURE_MODE"] = "hands"
        get_settings.cache_clear()
        spy = WidthSpy()
        sess = RecognitionSession(seed=1, matcher=spy)
        # A detected left hand (21 pts) so the frame is usable: hands mode now
        # rejects hand-less frames before the matcher (see presence gate).
        hand = [[0.1 + 0.01 * i, 0.2, 0.0] for i in range(21)]
        sess.push_landmarks(_shoulders_pose(), hand, None, t=0.0)
        assert spy.widths and set(spy.widths) == {HANDS_D}
    finally:
        os.environ.pop("ASL_FEATURE_MODE", None)
        get_settings.cache_clear()


def _emb(monkeypatch):
    """Mark the active matcher as the learned EmbeddingMatcher for one test, so a
    RecognitionSession picks up the embedding-tuned confirm params."""
    monkeypatch.setattr(recognition, "_uses_embedding", True)


def test_embedding_mode_rejects_wrong_sign(domain, monkeypatch):
    """REGRESSION: with the embedding matcher active, a WRONG sign produces a
    moderate, noisy strength (~0.6-0.7 — wrong signs average ~0.63 on the (cos+1)/2
    scale) that clears the DTW-era 0.6 threshold and dips. The DTW dip/overtake
    confirm would pass the word instantly ("every word passes as soon as I'm in
    frame"). Embedding mode must confirm ONLY on a sustained HIGH hold, so this
    never reaches a 'get'."""
    _emb(monkeypatch)
    # (current-target, next-target) per frame: peaks at 0.70 then dips to 0.54
    # (<= 0.70*0.8) — a dip the DTW params would confirm. Never approaches a high
    # hold threshold.
    cur = [0.70, 0.62, 0.54, 0.50, 0.60, 0.66, 0.58, 0.52, 0.63, 0.59, 0.61, 0.57]
    sched = [v for c in cur for v in (c, 0.40)]  # next-target 0.40 (no overtake)
    sess = RecognitionSession(seed=123, matcher=FakeMatcher(sched))
    events = [sess.push_frame(np.zeros(HANDS_D), t=i * 0.1).event for i in range(len(cur))]
    assert "get" not in events
    assert sess.word_index == 0  # never advanced off the wrong sign


def test_embedding_mode_confirms_held_correct_sign(domain, monkeypatch):
    """True-accept preserved: a correctly-performed sign holds high strength
    (~0.92, like the ~0.94 self-match mean), so the sustained-hold path confirms it
    after the warmup gate."""
    _emb(monkeypatch)
    cur = [0.92] * 20
    sched = [v for c in cur for v in (c, 0.40)]
    sess = RecognitionSession(seed=123, matcher=FakeMatcher(sched))
    events = [sess.push_frame(np.zeros(HANDS_D), t=i * 0.1).event for i in range(len(cur))]
    assert "get" in events
    assert sess.word_index == 1  # advanced past the confirmed word


def test_embedding_mode_auto_miss_after_10s(domain, monkeypatch):
    """Embedding mode auto-fails a word after 10s (seconds — the live timestamp is
    now seconds, not browser ms), so an unrecognized sign doesn't block the stream
    forever while still giving the signer real time to perform it."""
    _emb(monkeypatch)
    sess = RecognitionSession(seed=1, matcher=FakeMatcher([0.0] * 10))
    assert sess._session._miss_budget == 10.0


def test_embedding_mode_window_matches_training_window(domain, monkeypatch):
    """Prototypes are enrolled from 16-frame windows (train.dataset.WINDOW_SIZE), so
    the live session must buffer 16 — not the DTW-era 48 — or it embeds a mismatched
    temporal extent. (Measured: 16 vs 48 recovered true-accept 53% -> 69%.)"""
    _emb(monkeypatch)
    sess = RecognitionSession(seed=1, matcher=FakeMatcher([0.0] * 10))
    assert sess._session._buffer.maxlen == 16


def test_embedding_mode_rejects_pose_only_frames(domain, monkeypatch):
    """REGRESSION: a person in frame with NO hands detected (handLeft/handRight
    None) must be an UNUSABLE frame in hands/embedding mode. assemble_frame
    zero-fills a missing hand, and that no-hand frame normalizes (against jittering
    shoulders) into spurious motion the encoder maps to ~1.0 cosine -> "passes every
    word with no hands in frame". The raw-input presence check rejects it regardless
    of pose jitter, so it never reaches the matcher and never confirms."""
    _emb(monkeypatch)
    sess = RecognitionSession(seed=123, matcher=FakeMatcher([0.99] * 100))
    outs = [sess.push_landmarks(_shoulders_pose(), None, None, t=i * 0.1) for i in range(20)]
    assert all(o is None for o in outs)
    assert sess.word_index == 0  # nothing confirmed off a hand-less stream


def test_init_matcher_embedding_mode_falls_back_to_dtw_when_files_missing(domain, tmp_path: Path):
    # Opt-in embedding mode without encoder/prototypes must NOT crash: it falls
    # back to the DTW Matcher built from templates (today's behavior preserved).
    import os

    from asl.matcher import Matcher

    np.save(tmp_path / "wave__0.npy", np.random.RandomState(0).rand(8, D))
    env = {
        "ASL_MATCHER_MODE": "embedding",
        "ASL_ENCODER_PATH": str(tmp_path / "missing-encoder.onnx"),
        "ASL_PROTOTYPES_PATH": str(tmp_path / "missing-protos.npz"),
    }
    os.environ.update(env)
    get_settings.cache_clear()
    try:
        glosses = recognition.init_matcher(templates_dir=str(tmp_path))
        assert "wave" in glosses
        assert isinstance(recognition.get_matcher(), Matcher)
    finally:
        for k in env:
            os.environ.pop(k, None)
        get_settings.cache_clear()


def test_init_matcher_embedding_mode_uses_from_files_when_present(domain, tmp_path: Path, monkeypatch):
    # Opt-in embedding mode WITH both files present builds the EmbeddingMatcher
    # via from_files (stubbed: no real onnx needed) and derives glosses from it.
    import os

    import asl.embedding_matcher as em

    enc = tmp_path / "encoder.onnx"
    pro = tmp_path / "protos.npz"
    enc.touch()
    pro.touch()

    class StubEM:
        def __init__(self):
            self._protos = {"wave": 1, "nod": 1}

        def strength(self, window, target):
            return 0.0

    seen = {}

    def fake_from_files(onnx_path, prototypes_path):
        seen["args"] = (str(onnx_path), str(prototypes_path))
        return StubEM()

    monkeypatch.setattr(em.EmbeddingMatcher, "from_files", staticmethod(fake_from_files))

    env = {
        "ASL_MATCHER_MODE": "embedding",
        "ASL_ENCODER_PATH": str(enc),
        "ASL_PROTOTYPES_PATH": str(pro),
    }
    os.environ.update(env)
    get_settings.cache_clear()
    try:
        glosses = recognition.init_matcher()
        assert seen["args"] == (str(enc), str(pro))
        assert isinstance(recognition.get_matcher(), StubEM)
        assert glosses == ("nod", "wave")  # sorted from the matcher's prototypes
    finally:
        for k in env:
            os.environ.pop(k, None)
        get_settings.cache_clear()
