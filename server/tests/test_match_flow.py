"""Handler-level duel flow: ready gate -> match.start -> landmark frames ->
server-side recognition -> match.state / recognition.update / match.over."""

from __future__ import annotations

import asyncio

import pytest

from app import recognition, state, words
from app.messages import Landmark, LobbyCreate, LobbyJoin, LobbyReady, QueueJoin, QueueLeave
from app.recognition import Outcome
from app.ws import handlers
from tests.conftest import register_players


class WinningRecognizer:
    """Every frame completes the current word at full strength."""

    def __init__(self, seed: int):
        self._seed = seed
        self._index = 0

    @property
    def word_index(self) -> int:
        return self._index

    def push_landmarks(self, pose, hand_left, hand_right, t):
        word = words.word_at(self._seed, self._index).word
        idx = self._index
        self._index += 1
        return Outcome(event="get", word=word, word_index=idx, strength=1.0)


class ProgressRecognizer:
    """Never completes — always reports in-progress strength."""

    def __init__(self, seed: int):
        self._seed = seed

    @property
    def word_index(self) -> int:
        return 0

    def push_landmarks(self, pose, hand_left, hand_right, t):
        return Outcome(event=None, word=words.word_at(self._seed, 0).word, word_index=0, strength=0.4)


@pytest.fixture
def asl(monkeypatch):
    """Recognition reports ready and hands out WinningRecognizers by default."""
    monkeypatch.setattr(recognition, "is_ready", lambda: True)
    monkeypatch.setattr(recognition, "new_session", lambda seed: WinningRecognizer(seed))


def _ready_lobby(fm) -> None:
    asyncio.run(handlers.dispatch(1, LobbyCreate(type="lobby.create")))
    code = fm.last(1, "lobby.update")["code"]
    asyncio.run(handlers.dispatch(2, LobbyJoin(type="lobby.join", code=code)))
    asyncio.run(handlers.dispatch(1, LobbyReady(type="lobby.ready", ready=True)))
    asyncio.run(handlers.dispatch(2, LobbyReady(type="lobby.ready", ready=True)))


def _landmark(t: float = 0.0) -> Landmark:
    return Landmark(type="landmark", t=t, pose=None)


def test_both_ready_starts_duel(fake, asl):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _ready_lobby(fake)
    start = fake.last(1, "match.start")
    assert isinstance(start["word_seed"], int)
    hp = {p["player_id"]: p["hp"] for p in fake.last(1, "match.state")["players"]}
    assert hp == {1: 100.0, 2: 100.0}


def test_duel_blocked_when_recognition_unavailable(fake, monkeypatch):
    monkeypatch.setattr(recognition, "is_ready", lambda: False)
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _ready_lobby(fake)
    assert "match.start" not in fake.types_for(1)
    assert fake.last(1, "error")["code"] == "recognition_unavailable"


def test_landmarks_drive_a_win(fake, asl):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _ready_lobby(fake)

    won = False
    for i in range(300):
        asyncio.run(handlers.dispatch(1, _landmark(t=float(i))))  # increasing t: not throttled
        if "match.over" in fake.types_for(1):
            won = True
            break
    assert won
    over = fake.last(1, "match.over")
    assert over["winner_id"] == 1 and over["reason"] == "win"
    assert state.players[1].status == "idle"  # torn down


def test_in_progress_frame_sends_recognition_update(fake, monkeypatch):
    monkeypatch.setattr(recognition, "is_ready", lambda: True)
    monkeypatch.setattr(recognition, "new_session", lambda seed: ProgressRecognizer(seed))
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _ready_lobby(fake)

    asyncio.run(handlers.dispatch(1, _landmark()))
    upd = fake.last(1, "recognition.update")
    assert upd["strength"] == 0.4
    assert upd["word_index"] == 0
    # no damage dealt
    hp = {p["player_id"]: p["hp"] for p in fake.last(1, "match.state")["players"]}
    assert hp[2] == 100.0


def test_warmup_landmarks_give_feedback_no_match(fake, monkeypatch):
    # A queued player practices: recognition.update, no match, no damage.
    monkeypatch.setattr(recognition, "is_ready", lambda: True)
    monkeypatch.setattr(recognition, "new_session", lambda seed: ProgressRecognizer(seed))
    register_players(fake, (1, "Solo", 1200))

    asyncio.run(handlers.dispatch(1, QueueJoin(type="queue.join")))
    assert state.players[1].warmup_session is not None
    asyncio.run(handlers.dispatch(1, _landmark(t=0.0)))
    assert fake.last(1, "recognition.update")["strength"] == 0.4
    assert not state.matches

    # Leaving the queue ends warmup.
    asyncio.run(handlers.dispatch(1, QueueLeave(type="queue.leave")))
    assert state.players[1].warmup_session is None


def test_landmark_frames_are_throttled_to_fps_cap(fake, asl, monkeypatch):
    # Two frames within 1/cap seconds: only the first is processed.
    monkeypatch.setattr(recognition, "new_session", lambda seed: ProgressRecognizer(seed))
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _ready_lobby(fake)

    cap = handlers.get_settings().recognition_fps_cap
    asyncio.run(handlers.dispatch(1, _landmark(t=0.0)))
    n_after_first = len(fake.for_(1))
    asyncio.run(handlers.dispatch(1, _landmark(t=(1.0 / cap) / 2)))  # too soon -> dropped
    assert len(fake.for_(1)) == n_after_first
    asyncio.run(handlers.dispatch(1, _landmark(t=1.0)))  # well past the interval -> processed
    assert len(fake.for_(1)) > n_after_first
