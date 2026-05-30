"""Per-message dispatch for the WebSocket. Maps a validated client message to a
domain action and emits the resulting server messages.

Phase 1b handles lobby create/join/ready. Queue, signaling, and match messages
are accepted by the schema but answered with an `unsupported` error until their
phases (1c–1e) land.
"""

from __future__ import annotations

import asyncio
import logging

from app import lobby, matchmaking, signaling, state, turn, warmup
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
    QueueJoin,
    QueueLeave,
    QueueStatus,
    Signal,
    WarmupStart,
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
    elif isinstance(msg, QueueJoin):
        await _on_queue_join(pid)
    elif isinstance(msg, QueueLeave):
        await _on_queue_leave(pid)
    elif isinstance(msg, Signal):
        await _on_signal(pid, msg.data)
    else:
        await manager.send(
            pid, error("unsupported", f"'{msg.type}' is not available yet")
        )


async def announce_match(lob: Lobby, match: Match) -> None:
    """Tell both players a match formed: lobby is full, here are connect params."""
    await _broadcast_lobby_update(lob)
    await _send_match_found(match)


async def handle_disconnect(pid: int) -> None:
    """Clean up on socket close: leave the queue, leave any lobby (notifying a
    survivor), drop the connection, and forget the player."""
    matchmaking.leave_queue(pid)
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


# --- queue handlers ---------------------------------------------------------


async def _on_queue_join(pid: int) -> None:
    player = state.players.get(pid)
    if player is None:
        return
    if player.status not in ("idle", "queued"):
        await manager.send(pid, error("busy", "Leave your current lobby first"))
        return

    now = asyncio.get_running_loop().time()
    matchmaking.join_queue(pid, now)

    # Hand out a warmup stream and the queue position. The ticker pairs players.
    await manager.send(
        pid, WarmupStart(word_seed=warmup.new_seed(), dataset_version=get_dataset().version)
    )
    await manager.send(pid, QueueStatus(position=matchmaking.queue_position(pid)))


async def _on_queue_leave(pid: int) -> None:
    matchmaking.leave_queue(pid)
    await manager.send(pid, QueueStatus(position=0))


# --- signaling relay --------------------------------------------------------


async def _on_signal(pid: int, data: dict) -> None:
    """Forward opaque WebRTC signaling data to the other player in the match."""
    try:
        peer = signaling.peer_of(pid)
    except signaling.SignalingError as exc:
        await manager.send(pid, error(exc.code, exc.message))
        return
    await manager.send(peer, Signal(type="signal", data=data))


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
                ice_servers=turn.build_ice_servers(pid),
            ),
        )
