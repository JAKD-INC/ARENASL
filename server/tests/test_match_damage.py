"""Authoritative match: damage math, tug-of-war, miss, win — now server-driven
(apply_completion / apply_miss fed by the RecognitionSession outcome)."""

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
    m.word_seed = seed  # fix the stream so words/difficulty are predictable
    return m


def _word_and_strength(m: Match, index: int):
    word = words.word_at(m.word_seed, index).word
    return word, words.get_dataset().word_strength(word)


def test_start_match_initializes_state(domain):
    m = _match()
    assert m.state == "active"
    assert m.hp == {1: 100.0, 2: 100.0}
    assert m.word_index == {1: 0, 2: 0}
    assert m.word_seed is not None
    assert state.players[1].status == "in_match"
    assert m.event_log[0]["type"] == "start"


def test_completion_applies_tug_of_war(domain):
    m = _match(seed=7)
    word, strength = _word_and_strength(m, 0)  # DAMAGE_SCALE defaults to 1.0

    finished = match_engine.apply_completion(m, pid=1, word=word, word_index=0, strength=1.0, now=0.1)
    assert finished is False
    assert m.hp[1] == 100.0                      # gainer capped at 100
    assert m.hp[2] == pytest.approx(100.0 - strength)
    assert m.word_index[1] == 1                  # only the signer advances
    assert m.word_index[2] == 0


def test_partial_strength_scales_damage(domain):
    m = _match(seed=7)
    word, strength = _word_and_strength(m, 0)
    match_engine.apply_completion(m, pid=1, word=word, word_index=0, strength=0.5, now=0.1)
    assert m.hp[2] == pytest.approx(100.0 - strength * 0.5)


def test_gainer_hp_capped_at_100(domain):
    m = _match(seed=7)
    word, _ = _word_and_strength(m, 0)
    m.hp[1] = 99.0
    match_engine.apply_completion(m, pid=1, word=word, word_index=0, strength=1.0, now=0.1)
    assert m.hp[1] == 100.0


def test_miss_advances_without_damage(domain):
    m = _match(seed=7)
    match_engine.apply_miss(m, pid=1, word_index=0, now=0.1)
    assert m.word_index[1] == 1
    assert m.hp[2] == 100.0
    assert m.event_log[-1]["type"] == "miss"


def test_completion_on_inactive_match_rejected(domain):
    state.players[1] = Player(id=1, display_name="A", elo=1200)
    state.players[2] = Player(id=2, display_name="B", elo=1200)
    m = Match(id="m2", player_ids=(1, 2), roles={1: "offerer", 2: "answerer"}, lobby_code="C")
    with pytest.raises(match_engine.MatchError) as exc:
        match_engine.apply_completion(m, pid=1, word="go", word_index=0, strength=1.0, now=0.0)
    assert exc.value.code == "match_not_active"


def test_draining_opponent_ends_match_with_winner(domain):
    m = _match(seed=7)
    finished = False
    for idx in range(300):
        word, _ = _word_and_strength(m, idx)
        finished = match_engine.apply_completion(m, pid=1, word=word, word_index=idx, strength=1.0, now=0.1)
        if finished:
            break
    assert finished is True
    assert m.state == "finished"
    assert m.winner_id == 1
    assert m.hp[2] <= 0.0
    assert m.event_log[-1]["type"] == "over"
