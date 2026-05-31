import itertools

import numpy as np
import pytest
from asl.session import Session


class StubMatcher:
    """Returns scripted strengths so the segmentation logic is tested in
    isolation from DTW. Ignores window/target.

    The current target is always queue[0]; the session also probes queue[1]
    (the next target) each frame for overtake detection. Scripts here only care
    about the current target, so next-target probes return 0.0 and never
    trigger an overtake."""

    def __init__(self, strengths):
        self._it = iter(strengths)
        self._current_target = True  # toggles: current target then next target

    def strength(self, window, target):
        # Each frame the session calls strength twice: first for the current
        # target (queue[0]), then for the next target (queue[1], the overtake
        # probe). Scripts only describe the current target; the probe returns
        # 0.0 so it never overtakes.
        if self._current_target:
            self._current_target = False
            return next(self._it)
        self._current_target = True
        return 0.0


class TargetAwareStub:
    """Strength depends on the target argument: each target has its own
    scripted iterator of values, so the next target can rise and overtake the
    current one without any dip on the current target."""

    def __init__(self, per_target):
        self._its = {k: iter(v) for k, v in per_target.items()}
        self._last = {}

    def strength(self, window, target):
        it = self._its.get(target)
        if it is None:
            return 0.0
        try:
            self._last[target] = next(it)
        except StopIteration:
            pass
        return self._last.get(target, 0.0)


_FRAME = np.zeros((1, 1))  # content irrelevant; StubMatcher ignores it


def _session(prompts, strengths, **kwargs):
    defaults = dict(
        get_threshold=0.9, confirm_drop=0.8, miss_budget=2.0,
        window_size=8, get_points=20, miss_points=-10, lookahead=2,
    )
    defaults.update(kwargs)
    return Session(StubMatcher(strengths), itertools.cycle(prompts), **defaults)


def _session_with(matcher, prompts, **kwargs):
    defaults = dict(
        get_threshold=0.9, confirm_drop=0.8, miss_budget=2.0,
        window_size=8, get_points=20, miss_points=-10, lookahead=2,
    )
    defaults.update(kwargs)
    return Session(matcher, itertools.cycle(prompts), **defaults)


def test_lookahead_below_one_is_rejected():
    # push() probes queue[1] every frame for the overtake path, so the queue
    # must hold at least two entries; lookahead < 1 must fail fast.
    with pytest.raises(ValueError):
        _session(["A", "B"], [], lookahead=0)


def test_initial_state_exposes_current_and_queue():
    s = _session(["A", "B", "A", "B"], [])
    st = s.state()
    assert st.current == "A"
    assert st.queue[:2] == ["B", "A"]
    assert st.score == 0


def test_peak_then_decline_confirms_get_and_advances():
    # rises to 0.95, then falls to 0.7 (<= 0.95*0.8) -> confirm at the fall
    s = _session(["A", "B", "A"], [0.5, 0.95, 0.7])
    assert s.push(_FRAME, t=0.0).event is None      # 0.5 climbing
    assert s.push(_FRAME, t=0.1).event is None      # 0.95 new peak
    st = s.push(_FRAME, t=0.2)                       # 0.7 -> past the peak
    assert st.event == "get"
    assert st.score == 20
    assert st.current == "B"
    assert st.confirmed == "A"


def test_still_climbing_does_not_confirm():
    s = _session(["A", "B"], [0.5, 0.92, 0.95])
    s.push(_FRAME, t=0.0)
    s.push(_FRAME, t=0.1)
    st = s.push(_FRAME, t=0.2)   # still rising -> no premature get
    assert st.event is None
    assert st.current == "A"


def test_timeout_scores_miss_and_advances():
    s = _session(["A", "B"], [0.1, 0.1, 0.1])
    s.push(_FRAME, t=0.0)
    s.push(_FRAME, t=1.0)
    st = s.push(_FRAME, t=2.0)   # budget elapsed, never confirmed
    assert st.event == "miss"
    assert st.score == -10
    assert st.current == "B"


def test_fast_back_to_back_signs_both_score():
    # two clean peaks in quick succession -> two gets, keeps pace
    s = _session(
        ["A", "B", "C"],
        [0.95, 0.6,   # sign A: peak then fall -> get
         0.95, 0.6],  # sign B: peak then fall -> get
    )
    s.push(_FRAME, t=0.00)
    st1 = s.push(_FRAME, t=0.05)
    s.push(_FRAME, t=0.10)
    st2 = s.push(_FRAME, t=0.15)
    assert st1.event == "get" and st2.event == "get"
    assert st2.score == 40
    assert st2.current == "C"


def test_buffer_and_peak_reset_after_advance():
    # after a get, the low strength that confirmed it must not re-trigger:
    # peak resets to 0 on advance.
    s = _session(["A", "B", "C"], [0.95, 0.6, 0.5])
    s.push(_FRAME, t=0.0)
    s.push(_FRAME, t=0.1)        # get on A
    st = s.push(_FRAME, t=0.2)   # 0.5 against B, peak reset -> no event
    assert st.event is None
    assert st.current == "B"


def test_timer_resets_per_target():
    s = _session(["A", "B", "C"], [0.95, 0.6, 0.1])
    s.push(_FRAME, t=0.0)
    s.push(_FRAME, t=0.1)        # get on A at t=0.1 -> B starts at t=0.1
    st = s.push(_FRAME, t=1.5)   # only 1.4s into B's 2.0s budget
    assert st.event is None
    assert st.current == "B"


def test_continuous_no_dip_confirms_via_overtake():
    # Fluent signer: target A stays HIGH with no dip, while the next target B
    # rises and SUSTAINS out-matching A across consecutive frames. The
    # sustained overtake (not a dip) must confirm A; a single overtaking frame
    # is not enough (overtake_frames=2 default), guarding against noise spikes.
    matcher = TargetAwareStub({
        "A": [0.5, 0.95, 0.95, 0.94, 0.94],  # climbs above threshold, no dip
        "B": [0.0, 0.0,  0.90, 0.96, 0.97],  # overtakes for 2 frames (4 & 5)
    })
    s = _session_with(matcher, ["A", "B", "C"])
    assert s.push(_FRAME, t=0.0).event is None   # A 0.5,  B 0.0
    assert s.push(_FRAME, t=0.1).event is None   # A 0.95 peak, B 0.0
    assert s.push(_FRAME, t=0.2).event is None   # A 0.95, B 0.90 (not yet >)
    assert s.push(_FRAME, t=0.3).event is None   # A 0.94, B 0.96 (overtake #1)
    st = s.push(_FRAME, t=0.4)                    # A 0.94, B 0.97 (overtake #2)
    assert st.event == "get"
    assert st.confirmed == "A"
    assert st.current == "B"
    assert st.score == 20


def test_single_frame_overtake_spike_does_not_confirm():
    # A clean current sign holds high with NO dip; the next target B spikes for
    # exactly ONE frame (noise / frame-boundary artifact) then drops back. A
    # lone overtaking frame must NOT confirm; the miss budget eventually fires.
    matcher = TargetAwareStub({
        "A": [0.5, 0.95, 0.94, 0.94, 0.94],  # clean plateau, never dips
        "B": [0.0, 0.0,  0.96, 0.0,  0.0],   # one-frame spike at frame 3
    })
    s = _session_with(matcher, ["A", "B", "C"])
    assert s.push(_FRAME, t=0.0).event is None   # A 0.5,  B 0.0
    assert s.push(_FRAME, t=0.1).event is None   # A 0.95 peak, B 0.0
    assert s.push(_FRAME, t=0.2).event is None   # A 0.94, B 0.96 (spike #1)
    assert s.push(_FRAME, t=0.3).event is None   # A 0.94, B 0.0 (run reset)
    st = s.push(_FRAME, t=2.0)                    # budget elapsed -> miss
    assert st.event == "miss"
    assert st.confirmed is None
    assert st.current == "B"


def test_overtake_rejected_if_peak_below_threshold():
    # The current target A only climbs to 0.7 (below the 0.9 get_threshold) and
    # never peaks above it; B sustains out-matching A. Even a sustained overtake
    # must NOT confirm, because A was never performed well enough.
    matcher = TargetAwareStub({
        "A": [0.5, 0.6,  0.7,  0.7,  0.7],   # peak 0.7 < 0.9 threshold
        "B": [0.0, 0.0,  0.75, 0.76, 0.77],  # sustained overtake of A
    })
    s = _session_with(matcher, ["A", "B", "C"])
    assert s.push(_FRAME, t=0.0).event is None   # A 0.5,  B 0.0
    assert s.push(_FRAME, t=0.1).event is None   # A 0.6,  B 0.0
    assert s.push(_FRAME, t=0.2).event is None   # A 0.7,  B 0.75 (overtake #1)
    assert s.push(_FRAME, t=0.3).event is None   # A 0.7,  B 0.76 (overtake #2)
    st = s.push(_FRAME, t=2.0)                    # budget elapsed -> miss
    assert st.event == "miss"
    assert st.confirmed is None
    assert st.current == "B"


def test_high_plateau_no_overtake_no_dip_waits_for_miss_budget():
    # Lone high plateau on A with NO dip and NO overtake from B must NOT
    # confirm until the miss budget elapses, then it scores a miss.
    matcher = TargetAwareStub({
        "A": [0.95, 0.95, 0.95],   # plateaus high, never dips
        "B": [0.0, 0.0, 0.0],      # never overtakes
    })
    s = _session_with(matcher, ["A", "B", "C"])
    assert s.push(_FRAME, t=0.0).event is None   # 0s into budget
    assert s.push(_FRAME, t=1.0).event is None   # 1s, no dip, no overtake
    st = s.push(_FRAME, t=2.0)                    # budget elapsed -> miss
    assert st.event == "miss"
    assert st.confirmed is None
    assert st.current == "B"
    assert st.score == -10


def test_held_sign_confirms_without_dip_or_overtake():
    # Strength holds above threshold with no dip and no overtake; after
    # confirm_hold consecutive frames the held sign is accepted.
    s = _session(["A", "B", "C"], [0.7, 0.7, 0.7, 0.7],
                 get_threshold=0.5, confirm_hold=3)
    assert s.push(_FRAME, t=0.0).event is None   # hold 1
    assert s.push(_FRAME, t=0.1).event is None   # hold 2
    st = s.push(_FRAME, t=0.2)                    # hold 3 -> confirm
    assert st.event == "get"
    assert st.confirmed == "A"
    assert st.current == "B"
    assert st.score == 20


def test_no_auto_miss_when_budget_is_none():
    # miss_budget=None disables the timer entirely: however long the player
    # lingers on a weak sign, it never auto-misses and never advances.
    s = _session(["A", "B"], [0.1, 0.1, 0.1], miss_budget=None)
    s.push(_FRAME, t=0.0)
    s.push(_FRAME, t=1000.0)
    st = s.push(_FRAME, t=1_000_000.0)
    assert st.event is None
    assert st.score == 0
    assert st.current == "A"   # still on the first prompt; no advance
