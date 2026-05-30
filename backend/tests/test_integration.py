# Settled calibration for the RMS-normalized DTW (Plan 2 inherits these defaults):
#   scale=0.5, get_threshold=0.6, confirm_drop=0.8, window_size=6
# The matcher's DTW distance is RMS per-step (raw cost / sqrt(path length)),
# which is sequence-length-invariant. Under that metric the absolute match
# strengths land lower than a mean-per-step metric would produce: BYE only peaks
# ~0.63 here. 0.6 is therefore the settled gate -- it sits below that peak yet
# well above the cross-sign floor, so HELLO and BYE each segment cleanly. (These
# are the converged RMS constants, not a downgrade from an earlier value.)
# Replay is hello+bye+hello: the trailing HELLO frames push BYE's strength
# down past its peak so peak-segmentation confirms BYE (the second clean get).
# The prompt iterator is extended past the 3 init prompts (lookahead=2 consumes
# lookahead+1=3) so that each advance still has a prompt to dequeue.
import numpy as np
from asl.features import normalize_frame
from asl.matcher import Matcher
from asl.session import Session


def _sign(path_xs):
    # (3,3) frames: two shoulders + one moving hand point, normalized + flattened.
    frames = []
    for x in path_xs:
        f = np.zeros((3, 3))
        f[0] = [-1, 0, 0]   # left shoulder
        f[1] = [1, 0, 0]    # right shoulder
        f[2] = [x, 1, 0]    # hand traces a path
        frames.append(normalize_frame(f).flatten())
    return np.array(frames)


def test_two_fast_signs_each_score():
    hello = _sign([0.0, 0.5, 1.0])
    bye = _sign([0.0, -0.5, -1.0])
    templates = {"HELLO": [hello], "BYE": [bye]}

    matcher = Matcher(templates, scale=0.5)
    s = Session(
        matcher, iter(["HELLO", "BYE", "HELLO", "BYE", "HELLO", "BYE"]),
        get_threshold=0.6, confirm_drop=0.8, miss_budget=10.0,
        window_size=6, lookahead=2,
    )

    confirms = []  # (gloss confirmed this frame, score after the get)
    # Replay HELLO, BYE, then HELLO contiguously (a fast signer, no pause):
    # BYE's frames drive HELLO's strength down past its peak (confirming HELLO),
    # and the trailing HELLO frames do the same for BYE (confirming BYE).
    for i, frame in enumerate(np.concatenate([hello, bye, hello])):
        st = s.push(frame, t=i * 0.1)
        if st.event == "get":
            # confirmed names the gloss just completed THIS frame, captured
            # before the session advanced to the next prompt.
            confirms.append((st.confirmed, st.score))

    # EXACTLY two get events fire (no spurious or missing confirms).
    assert len(confirms) == 2
    # The confirmed glosses are HELLO then BYE, in that order, read from the
    # new State.confirmed field (the gloss completed THIS frame) rather than
    # state.current (which already names the NEXT prompt after a get advanced).
    assert [gloss for gloss, _ in confirms] == ["HELLO", "BYE"]
    # Each get awards 20 points: 20 after HELLO, 40 after BYE -> final score 40.
    assert [score for _, score in confirms] == [20, 40]
    assert confirms[-1][1] == 40
