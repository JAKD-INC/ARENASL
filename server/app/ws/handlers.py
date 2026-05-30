"""Per-message dispatch for the WebSocket. Maps a validated client message to a
domain action and emits the resulting server messages.

Phase 1b handles lobby create/join/ready. Queue, signaling, and match messages
are accepted by the schema but answered with an `unsupported` error until their
phases (1c–1e) land.
"""

from __future__ import annotations

import logging

from app import lobby, state
from app.connection_manager import manager
from app.messages import (
    ClientMessage,
    LobbyCreate,
    LobbyJoin,
    LobbyMemberView,
    LobbyReady,
    LobbyUpdate,
    MatchFound,
    OpponentView,
    error,
)
from app.state import Lobby, Match, Player
from app.words import get_dataset

logger = logging.getLogger("arenasl.handlers")


def register_player(pid: int, display_name: str, elo: int) -> None:
    state.players[pid] = Player(id=pid, display_name=display_name, elo=elo)


async def dispatch(pid: int, msg: ClientMessage) -> None:
    if isinstance(msg, LobbyCreate):
        await _on_lobby_create(pid)
    elif isinstance(msg, LobbyJoin):
        await _on_lobby_join(pid, msg.code)
    elif isinstance(msg, LobbyReady):
        await _on_lobby_ready(pid, msg.ready)
    else:
        await manager.send(
            pid, error("unsupported", f"'{msg.type}' is not available yet")
        )


async def handle_disconnect(pid: int) -> None:
    """Clean up on socket close: leave any lobby (notifying a survivor), drop the
    connection, and forget the player."""
    remaining = lobby.leave_lobby(pid)
    if remaining is not None:
        await _broadcast_lobby_update(remaining)
    await manager.disconnect(pid)
    state.players.pop(pid, None)


# --- lobby handlers ---------------------------------------------------------


async def _on_lobby_create(pid: int) -> None:
    lob = lobby.create_lobby(pid)
    await _broadcast_lobby_update(lob)


async def _on_lobby_join(pid: int, code: str) -> None:
    try:
        lob, match = lobby.join_lobby(pid, code)
    except lobby.LobbyError as exc:
        await manager.send(pid, error(exc.code, exc.message))
        return
    await _broadcast_lobby_update(lob)
    if match is not None:
        await _send_match_found(match)


async def _on_lobby_ready(pid: int, ready: bool) -> None:
    try:
        lob = lobby.set_ready(pid, ready)
    except lobby.LobbyError as exc:
        await manager.send(pid, error(exc.code, exc.message))
        return
    await _broadcast_lobby_update(lob)
    # The match.start gate (both ready -> duel) is added in phase 1e.


# --- view builders ----------------------------------------------------------


def _lobby_update(lob: Lobby) -> LobbyUpdate:
    members = [
        LobbyMemberView(
            player_id=m.player_id,
            display_name=state.players[m.player_id].display_name,
            ready=m.ready,
            connected=manager.is_connected(m.player_id),
        )
        for m in lob.members
        if m.player_id in state.players
    ]
    return LobbyUpdate(code=lob.code, state="full" if lob.is_full else "waiting", members=members)


async def _broadcast_lobby_update(lob: Lobby) -> None:
    await manager.broadcast([m.player_id for m in lob.members], _lobby_update(lob))


async def _send_match_found(match: Match) -> None:
    version = get_dataset().version
    for pid in match.player_ids:
        other = next(p for p in match.player_ids if p != pid)
        opp = state.players[other]
        await manager.send(
            pid,
            MatchFound(
                match_id=match.id,
                role=match.roles[pid],
                dataset_version=version,
                opponent=OpponentView(
                    player_id=opp.id, display_name=opp.display_name, elo=opp.elo
                ),
            ),
        )
