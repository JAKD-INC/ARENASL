"""Lobby lifecycle: create / join-by-code / ready, for lobbies of exactly 2
players. Used by both entry paths — private (this module's `create_lobby`) and
quick match (the matchmaker creates a `queue`-origin lobby in phase 1c).

When a lobby fills, a Match stub is created and WebRTC roles are assigned (host =
offerer, joiner = answerer). The duel itself starts only when both ready
(phase 1e); this module just tracks readiness.
"""

from __future__ import annotations

import secrets
import string
import uuid

from app import state
from app.state import Lobby, LobbyMember, Match

CODE_ALPHABET = string.ascii_uppercase + string.digits
CODE_LENGTH = 6


class LobbyError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _generate_code() -> str:
    while True:
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        if code not in state.lobbies:
            return code


def create_lobby(pid: int, *, origin: str = "private") -> Lobby:
    _leave_current_lobby(pid)  # a player is only ever in one lobby
    lobby = Lobby(code=_generate_code(), origin=origin, members=[LobbyMember(player_id=pid)])
    state.lobbies[lobby.code] = lobby
    _mark_in_lobby(pid, lobby.code)
    return lobby


def join_lobby(pid: int, code: str) -> tuple[Lobby, Match | None]:
    """Join an existing lobby by code. Returns the lobby and, if this join filled
    it, the freshly created Match stub (else None)."""
    lobby = state.lobbies.get(code.upper())
    if lobby is None:
        raise LobbyError("lobby_not_found", "No lobby with that code")
    if lobby.member(pid) is not None:
        raise LobbyError("already_in_lobby", "You are already in this lobby")
    if lobby.is_full:
        raise LobbyError("lobby_full", "That lobby already has two players")

    _leave_current_lobby(pid)
    lobby.members.append(LobbyMember(player_id=pid))
    _mark_in_lobby(pid, lobby.code)

    match = _fill_to_match(lobby) if lobby.is_full else None
    return lobby, match


def set_ready(pid: int, ready: bool) -> Lobby:
    lobby = _lobby_of(pid)
    member = lobby.member(pid)
    assert member is not None  # _lobby_of guarantees membership
    member.ready = ready
    return lobby


def leave_lobby(pid: int) -> Lobby | None:
    """Remove a player from their lobby. Returns the lobby if one or more players
    remain in it (so callers can notify them), or None if the lobby was deleted."""
    code = state.players[pid].lobby_code if pid in state.players else None
    if code is None:
        return None
    lobby = state.lobbies.get(code)
    if lobby is None:
        return None

    lobby.members = [m for m in lobby.members if m.player_id != pid]
    _clear_lobby(pid)

    # A departure dissolves any pending match stub; survivors return to waiting.
    if lobby.match_id is not None:
        state.matches.pop(lobby.match_id, None)
        lobby.match_id = None
    for m in lobby.members:
        m.ready = False

    if not lobby.members:
        state.lobbies.pop(lobby.code, None)
        return None
    return lobby


def both_ready(lobby: Lobby) -> bool:
    return lobby.is_full and all(m.ready for m in lobby.members)


# --- internals --------------------------------------------------------------


def create_queue_lobby(pid_a: int, pid_b: int) -> tuple[Lobby, Match]:
    """Form a `queue`-origin lobby with both matchmade players already in it, then
    fill it to a Match. Both players then run the same ready-up flow as a private
    lobby."""
    _leave_current_lobby(pid_a)
    _leave_current_lobby(pid_b)
    lobby = Lobby(
        code=_generate_code(),
        origin="queue",
        members=[LobbyMember(player_id=pid_a), LobbyMember(player_id=pid_b)],
    )
    state.lobbies[lobby.code] = lobby
    _mark_in_lobby(pid_a, lobby.code)
    _mark_in_lobby(pid_b, lobby.code)
    match = _fill_to_match(lobby)
    return lobby, match


def _fill_to_match(lobby: Lobby) -> Match:
    host, joiner = lobby.members[0].player_id, lobby.members[1].player_id
    match = Match(
        id=uuid.uuid4().hex,
        player_ids=(host, joiner),
        roles={host: "offerer", joiner: "answerer"},
        lobby_code=lobby.code,
    )
    state.matches[match.id] = match
    lobby.match_id = match.id
    return match


def _lobby_of(pid: int) -> Lobby:
    code = state.players[pid].lobby_code if pid in state.players else None
    lobby = state.lobbies.get(code) if code else None
    if lobby is None or lobby.member(pid) is None:
        raise LobbyError("not_in_lobby", "You are not in a lobby")
    return lobby


def _mark_in_lobby(pid: int, code: str) -> None:
    p = state.players.get(pid)
    if p is not None:
        p.status = "in_lobby"
        p.lobby_code = code


def _clear_lobby(pid: int) -> None:
    p = state.players.get(pid)
    if p is not None:
        p.status = "idle"
        p.lobby_code = None


def _leave_current_lobby(pid: int) -> None:
    p = state.players.get(pid)
    if p is not None and p.lobby_code is not None:
        leave_lobby(pid)
