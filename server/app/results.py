"""Durable match results: apply ELO deltas and write a MatchHistory row.

Synchronous (SQLite) — call via asyncio.to_thread from the event loop so the tiny
write never blocks it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app import elo, state
from app.db import SessionLocal
from app.models_db import MatchHistory, User


def persist_match_result(
    match: state.Match, reason: str
) -> tuple[dict[int, tuple[int, int]], int | None]:
    """Update both players' ratings, bump games_played, write history.

    Returns ({player_id: (new_elo, delta)}, match_history_id). Deltas are empty and
    the id is None if a user row is missing.
    """
    winner_id = match.winner_id
    a, b = match.player_ids
    if winner_id is None:
        return {}, None
    loser_id = a if winner_id == b else b

    with SessionLocal() as db:
        winner = db.get(User, winner_id)
        loser = db.get(User, loser_id)
        if winner is None or loser is None:
            db.rollback()
            return {}, None

        old_w, old_l = winner.elo, loser.elo
        new_w = elo.updated_rating(old_w, old_l, won=True, games_played=winner.games_played)
        new_l = elo.updated_rating(old_l, old_w, won=False, games_played=loser.games_played)

        winner.elo, loser.elo = new_w, new_l
        winner.games_played += 1
        loser.games_played += 1

        history = MatchHistory(
            player_a=a,
            player_b=b,
            winner_id=winner_id,
            elo_delta=new_w - old_w,
            started_at=match.started_wall or datetime.now(timezone.utc),
            end_reason=reason,
        )
        db.add(history)
        db.commit()
        history_id = history.id

    return {winner_id: (new_w, new_w - old_w), loser_id: (new_l, new_l - old_l)}, history_id
