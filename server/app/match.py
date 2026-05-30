"""Authoritative match logic: the server owns HP, the word stream position, and
the win decision. Clients only *propose* sign attempts.

Damage = word_strength (the sign's dataset difficulty) * accuracy * DAMAGE_SCALE,
applied tug-of-war: the signer gains it (capped at 100), the opponent loses it.
Every state change is appended to a timestamped event log (the replay backbone).
"""

from __future__ import annotations

from datetime import datetime, timezone

from app import state, words
from app.config import get_settings

MAX_HP = 100.0


class MatchError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _t_ms(match: state.Match, now: float) -> int:
    """Milliseconds since the match started (the shared record_start_ms origin)."""
    return int(round((now - (match.started_at or now)) * 1000))


def start_match(match: state.Match, now: float) -> None:
    """Begin the duel: seed the word stream, set HP, and start the clock."""
    match.word_seed = words.random_seed()
    match.hp = {pid: MAX_HP for pid in match.player_ids}
    match.word_index = {pid: 0 for pid in match.player_ids}
    match.started_at = now
    match.started_wall = datetime.now(timezone.utc)
    match.state = "active"
    match.event_log.append({"t": 0, "type": "start", "word_seed": match.word_seed})
    for pid in match.player_ids:
        player = state.players.get(pid)
        if player is not None:
            player.status = "in_match"


def handle_attempt(
    match: state.Match, pid: int, word_index: int, accuracy: float, now: float
) -> bool:
    """Apply a sign attempt. Returns True iff this attempt ended the match.
    Raises MatchError on anything the server won't accept."""
    if match.state != "active":
        raise MatchError("match_not_active", "Match is not active")
    if pid not in match.hp:
        raise MatchError("not_in_match", "You are not in this match")

    expected = match.word_index[pid]
    if word_index != expected:
        raise MatchError("bad_word_index", f"Expected word_index {expected}")
    if not (0.0 <= accuracy <= 1.0):
        raise MatchError("bad_accuracy", "accuracy must be within [0, 1]")

    entry = words.word_at(match.word_seed, word_index)
    damage = entry.difficulty * accuracy * get_settings().damage_scale
    opponent = next(p for p in match.player_ids if p != pid)

    match.hp[pid] = min(MAX_HP, match.hp[pid] + damage)
    match.hp[opponent] = match.hp[opponent] - damage
    match.word_index[pid] = word_index + 1

    match.event_log.append(
        {
            "t": _t_ms(match, now),
            "type": "attempt",
            "player_id": pid,
            "word_index": word_index,
            "word": entry.word,
            "accuracy": accuracy,
            "damage": damage,
            "hp": dict(match.hp),
        }
    )

    if match.hp[opponent] <= 0.0:
        match.state = "finished"
        match.winner_id = pid
        match.event_log.append({"t": _t_ms(match, now), "type": "over", "winner_id": pid})
        return True
    return False
