"""Handler-level practice flow: a SOLO recognizer (NO matchmaking) drives
recognition.update from landmark frames; practice.stop tears it down."""

from __future__ import annotations

import asyncio

import pytest

from app import matchmaking, recognition, state, words
from app.messages import Landmark, PracticeStartReq, PracticeStop
from app.recognition import Outcome
from app.ws import handlers
from tests.conftest import register_players


class ProgressRecognizer:
    """Mirrors RecognitionSession's index bookkeeping over the seeded stream:
    every other frame "gets" the current word (advancing _index, like a real
    session), so a sequence of frames yields the same incremental word indices a
    real recognizer would, never desyncing from words.word_at(seed, i)."""

    def __init__(self, seed: int):
        self._seed = seed
        self._index = 0
        self._frames = 0

    @property
    def word_index(self) -> int:
        return self._index

    def push_landmarks(self, pose, hand_left, hand_right, t):
        word = words.word_at(self._seed, self._index).word
        self._frames += 1
        # Confirm a sign on every other frame: report the just-completed word at
        # its pre-advance index (matching RecognitionSession.push_frame), then
        # advance. In-progress frames report the current target unchanged.
        if self._frames % 2 == 0:
            outcome = Outcome(event="get", word=word, word_index=self._index, strength=0.9)
            self._index += 1
            return outcome
        return Outcome(event=None, word=word, word_index=self._index, strength=0.4)


@pytest.fixture
def asl(monkeypatch):
    """Recognition reports ready and hands out non-scoring ProgressRecognizers."""
    monkeypatch.setattr(recognition, "is_ready", lambda: True)
    monkeypatch.setattr(recognition, "new_session", lambda seed: ProgressRecognizer(seed))


def _landmark(t: float = 0.0) -> Landmark:
    return Landmark(type="landmark", t=t, pose=None)


def test_practice_start_mints_session_without_matchmaking(fake, asl):
    register_players(fake, (1, "Solo", 1200))

    asyncio.run(handlers.dispatch(1, PracticeStartReq(type="practice.start")))

    player = state.players[1]
    assert player.practice_session is not None
    assert player.status == "practice"
    # SOLO: not enqueued for matchmaking, no lobby/match created.
    assert matchmaking.queue_position(1) == 0
    assert not state.queue
    assert not state.matches
    assert not state.lobbies
    # Ack carries the seeded word stream + dataset version.
    ack = fake.last(1, "practice.start")
    assert isinstance(ack["word_seed"], int)
    assert ack["dataset_version"]


def test_practice_landmark_yields_recognition_update(fake, asl):
    register_players(fake, (1, "Solo", 1200))
    asyncio.run(handlers.dispatch(1, PracticeStartReq(type="practice.start")))

    asyncio.run(handlers.dispatch(1, _landmark(t=0.0)))

    upd = fake.last(1, "recognition.update")
    assert upd["strength"] == 0.4
    assert upd["word_index"] == 0
    assert upd["word"]
    # The word's catalog word_strength rides along so the client gets the real
    # difficulty (not the wire default of 1.0).
    assert upd["difficulty"] == words.get_dataset().word_strength(upd["word"])
    # Solo: no match state / damage.
    assert not state.matches


def test_practice_word_index_advances_in_lockstep_with_stream(fake, asl):
    """Each confirmed word advances the index exactly by one, and the word the
    client is told matches words.word_at(seed, index) — so the HUD slug resolves
    to the built clip with no desync between server stream and client."""
    register_players(fake, (1, "Solo", 1200))
    asyncio.run(handlers.dispatch(1, PracticeStartReq(type="practice.start")))
    seed = fake.last(1, "practice.start")["word_seed"]

    # Push enough frames to confirm three words (a "get" lands every other frame).
    for i in range(6):
        asyncio.run(handlers.dispatch(1, _landmark(t=float(i))))

    updates = fake.for_(1)
    indices = [u["word_index"] for u in updates if u["type"] == "recognition.update"]
    # Indices only ever stay or step up by one — never skip or go backwards.
    assert indices == sorted(indices)
    for prev, cur in zip(indices, indices[1:]):
        assert cur - prev in (0, 1)
    # Every reported word equals the seeded stream's word at that index.
    for u in updates:
        if u["type"] == "recognition.update":
            assert u["word"] == words.word_at(seed, u["word_index"]).word
    # Three distinct words were drilled (indices 0, 1, 2 all appear).
    assert set(indices) >= {0, 1, 2}
    # Still solo: no matchmaking artifacts ever created.
    assert not state.matches and not state.queue and not state.lobbies


def test_practice_stop_clears_session(fake, asl):
    register_players(fake, (1, "Solo", 1200))
    asyncio.run(handlers.dispatch(1, PracticeStartReq(type="practice.start")))
    assert state.players[1].practice_session is not None

    asyncio.run(handlers.dispatch(1, PracticeStop(type="practice.stop")))

    player = state.players[1]
    assert player.practice_session is None
    assert player.status == "idle"

    # No further feedback once stopped.
    n_before = len(fake.for_(1))
    asyncio.run(handlers.dispatch(1, _landmark(t=1.0)))
    assert len(fake.for_(1)) == n_before


def test_practice_blocked_when_recognition_unavailable(fake, monkeypatch):
    monkeypatch.setattr(recognition, "is_ready", lambda: False)
    register_players(fake, (1, "Solo", 1200))

    asyncio.run(handlers.dispatch(1, PracticeStartReq(type="practice.start")))

    assert state.players[1].practice_session is None
    assert "practice.start" not in fake.types_for(1)
    assert fake.last(1, "error")["code"] == "recognition_unavailable"
