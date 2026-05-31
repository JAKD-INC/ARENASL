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


def apply_completion(
    match: state.Match, pid: int, word: str, word_index: int, strength: float, now: float
) -> bool:
    """Apply a server-recognized completed sign. `word`/`word_index`/`strength`
    come from the player's authoritative RecognitionSession, so there's nothing
    client-supplied to validate. Returns True iff this ended the match."""
    if match.state != "active":
        raise MatchError("match_not_active", "Match is not active")

    damage = words.get_dataset().word_strength(word) * strength * get_settings().damage_scale
    opponent = next(p for p in match.player_ids if p != pid)

    match.hp[pid] = min(MAX_HP, match.hp[pid] + damage)
    match.hp[opponent] = match.hp[opponent] - damage
    match.word_index[pid] = word_index + 1

    match.event_log.append(
        {
            "t": _t_ms(match, now),
            "type": "get",
            "player_id": pid,
            "word_index": word_index,
            "word": word,
            "strength": strength,
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


def apply_miss(match: state.Match, pid: int, word_index: int, now: float) -> None:
    """A word timed out for this player: advance past it, no damage."""
    if match.state != "active":
        return
    match.word_index[pid] = word_index + 1
    match.event_log.append(
        {"t": _t_ms(match, now), "type": "miss", "player_id": pid, "word_index": word_index}
    )
