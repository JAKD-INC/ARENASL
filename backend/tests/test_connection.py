import numpy as np
from server.connection import handle_message, state_to_dict
from asl.session import State


class StubSession:
    def __init__(self):
        self.pushed = []

    def push(self, frame, t):
        self.pushed.append((frame, t))
        return State(current="book", queue=["drink"], strength=0.5,
                     score=20, event="get", confirmed="book")


def _msg():
    pose = [[0.0, 0.0, 0.0] for _ in range(33)]
    pose[11] = [0.0, 0.0, 0.0]   # left shoulder
    pose[12] = [1.0, 0.0, 0.0]   # right shoulder (non-zero width)
    return {"t": 1.5, "pose": pose, "handLeft": None, "handRight": None}


def test_handle_message_pushes_normalized_frame():
    s = StubSession()
    out = handle_message(s, _msg())
    frame, t = s.pushed[0]
    assert t == 1.5
    assert frame.shape == (49 * 3,)        # normalized + flattened
    assert out["event"] == "get"
    assert out["confirmed"] == "book"


def test_handle_message_skips_frame_without_pose():
    s = StubSession()
    out = handle_message(s, {"t": 2.0, "pose": None, "handLeft": None, "handRight": None})
    assert s.pushed == []                   # nothing pushed
    assert out is None                       # caller sends nothing back


def test_state_to_dict_round_trips_fields():
    st = State(current="go", queue=["eat", "water"], strength=0.9,
               score=40, event=None, confirmed=None)
    d = state_to_dict(st)
    assert d == {"current": "go", "queue": ["eat", "water"], "strength": 0.9,
                 "score": 40, "event": None, "confirmed": None, "distance": None}
