"""Authoritative match: damage math, tug-of-war, validation, win detection."""

from __future__ import annotations

import pytest

from app import match as match_engine
from app import state, words
from app.state import Match, Player


def _match(seed: int = 7) -> Match:
    state.players[1] = Player(id=1, display_name="A", elo=1200)
    state.players[2] = Player(id=2, display_name="B", elo=1200)
    m = Match(id="m1", player_ids=(1, 2), roles={1: "offerer", 2: "answerer"}, lobby_code="C")
    state.matches[m.id] = m
    match_engine.start_match(m, now=0.0)
    m.word_seed = seed  # fix the stream so damage is predictable
    return m


def test_start_match_initializes_state(domain):
    m = _match()
    assert m.state == "active"
    assert m.hp == {1: 100.0, 2: 100.0}
    assert m.word_index == {1: 0, 2: 0}
    assert m.word_seed is not None
    assert state.players[1].status == "in_match"
    assert m.event_log[0]["type"] == "start"


def test_attempt_applies_tug_of_war(domain):
    m = _match(seed=7)
    strength = words.word_at(m.word_seed, 0).difficulty  # DAMAGE_SCALE defaults to 1.0

    finished = match_engine.handle_attempt(m, pid=1, word_index=0, accuracy=1.0, now=0.1)
    assert finished is False
    assert m.hp[1] == 100.0  # gainer capped at 100
    assert m.hp[2] == pytest.approx(100.0 - strength)
    assert m.word_index[1] == 1  # only the signer's index advances
    assert m.word_index[2] == 0


def test_partial_accuracy_scales_damage(domain):
    m = _match(seed=7)
    strength = words.word_at(m.word_seed, 0).difficulty
    match_engine.handle_attempt(m, pid=1, word_index=0, accuracy=0.5, now=0.1)
    assert m.hp[2] == pytest.approx(100.0 - strength * 0.5)


def test_gainer_hp_capped_at_100(domain):
    m = _match(seed=7)
    m.hp[1] = 99.0
    match_engine.handle_attempt(m, pid=1, word_index=0, accuracy=1.0, now=0.1)
    assert m.hp[1] == 100.0


def test_bad_word_index_rejected(domain):
    m = _match()
    with pytest.raises(match_engine.MatchError) as exc:
        match_engine.handle_attempt(m, pid=1, word_index=5, accuracy=1.0, now=0.1)
    assert exc.value.code == "bad_word_index"
    assert m.hp[2] == 100.0  # nothing applied


@pytest.mark.parametrize("accuracy", [-0.1, 1.1, 2.0])
def test_bad_accuracy_rejected(domain, accuracy):
    m = _match()
    with pytest.raises(match_engine.MatchError) as exc:
        match_engine.handle_attempt(m, pid=1, word_index=0, accuracy=accuracy, now=0.1)
    assert exc.value.code == "bad_accuracy"


def test_attempt_on_inactive_match_rejected(domain):
    state.players[1] = Player(id=1, display_name="A", elo=1200)
    state.players[2] = Player(id=2, display_name="B", elo=1200)
    m = Match(id="m2", player_ids=(1, 2), roles={1: "offerer", 2: "answerer"}, lobby_code="C")
    # not started -> still "connecting"
    with pytest.raises(match_engine.MatchError) as exc:
        match_engine.handle_attempt(m, pid=1, word_index=0, accuracy=1.0, now=0.0)
    assert exc.value.code == "match_not_active"


def test_draining_opponent_ends_match_with_winner(domain):
    m = _match(seed=7)
    finished = False
    idx = 0
    for _ in range(200):  # each correct sign damages the opponent
        finished = match_engine.handle_attempt(m, pid=1, word_index=idx, accuracy=1.0, now=0.1)
        idx += 1
        if finished:
            break
    assert finished is True
    assert m.state == "finished"
    assert m.winner_id == 1
    assert m.hp[2] <= 0.0
    assert m.event_log[-1]["type"] == "over"
