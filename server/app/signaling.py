"""WebRTC signaling relay: a dumb pass-through.

The server never parses SDP or ICE — it forwards a `signal` message's opaque
`data` verbatim to the other player in the sender's match. There's exactly one
other socket per match, so no `target` field is needed.
"""

from __future__ import annotations

from app import state


class SignalingError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def peer_of(pid: int) -> int:
    """Return the other player's id in this player's current match.
    Raises SignalingError if the player isn't in a match."""
    player = state.players.get(pid)
    match_id = player.match_id if player is not None else None
    match = state.matches.get(match_id) if match_id else None
    if match is None:
        raise SignalingError("not_in_match", "You are not in a match")
    return next(p for p in match.player_ids if p != pid)
