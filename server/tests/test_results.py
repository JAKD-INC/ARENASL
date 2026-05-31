"""ELO + MatchHistory persistence on match end."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app import results, state
from app.db import SessionLocal
from app.models_db import MatchHistory, User
from app.state import Match


def _two_users(elo_a: int = 1200, elo_b: int = 1200) -> tuple[int, int]:
    with SessionLocal() as db:
        a = User(email="w@e.com", password_hash="x", display_name="W", elo=elo_a, games_played=0)
        b = User(email="l@e.com", password_hash="x", display_name="L", elo=elo_b, games_played=0)
        db.add_all([a, b])
        db.commit()
        return a.id, b.id


def _finished_match(winner: int, players: tuple[int, int], reason: str = "win") -> Match:
    return Match(
        id="m1",
        player_ids=players,
        roles={players[0]: "offerer", players[1]: "answerer"},
        lobby_code="C",
        state="finished",
        winner_id=winner,
        started_wall=datetime.now(timezone.utc),
    )


def test_persist_updates_elo_and_writes_history(domain):
    w, l = _two_users()
    match = _finished_match(winner=w, players=(w, l))

    deltas, history_id = results.persist_match_result(match, reason="win")

    assert deltas[w][1] > 0  # winner gained
    assert deltas[l][1] < 0  # loser lost
    assert history_id is not None

    with SessionLocal() as db:
        assert db.get(User, w).elo > 1200
        assert db.get(User, l).elo < 1200
        assert db.get(User, w).games_played == 1
        assert db.get(User, l).games_played == 1
        hist = db.scalars(select(MatchHistory)).one()
        assert hist.id == history_id
        assert hist.winner_id == w
        assert hist.end_reason == "win"


def test_persist_records_forfeit_reason(domain):
    w, l = _two_users()
    match = _finished_match(winner=w, players=(w, l), reason="forfeit")
    results.persist_match_result(match, reason="forfeit")
    with SessionLocal() as db:
        assert db.scalars(select(MatchHistory)).one().end_reason == "forfeit"


def test_persist_no_users_returns_empty(domain):
    match = _finished_match(winner=999, players=(999, 1000))
    assert results.persist_match_result(match, reason="win") == ({}, None)
