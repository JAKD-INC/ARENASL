"""RecognitionSession wiring (outcome mapping, word-index advance, landmark
assembly) via a scripted matcher, plus a real-DTW matcher load smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app import recognition, words
from app.recognition import RecognitionSession

D = 147  # 49 keypoints × 3


class FakeMatcher:
    """Returns scheduled strengths in call order. Session calls strength() twice
    per frame (current target, then next target), so schedule in pairs."""

    def __init__(self, schedule):
        self._it = iter(schedule)

    def strength(self, window, target):
        return next(self._it, 0.0)


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
