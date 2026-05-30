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
from app import match as match_engine
from app.connection_manager import manager
from app.messages import (
    ClientMessage,
    LobbyCreate,
    LobbyJoin,
    LobbyMemberView,
    LobbyReady,
    LobbyUpdate,
    MatchFound,
    MatchOver,
    MatchStart,
    MatchState,
    OpponentView,
    PlayerState,
    QueueJoin,
    QueueLeave,
    QueueStatus,
    Signal,
    SignAttempt,
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
    elif isinstance(msg, SignAttempt):
        await _on_sign_attempt(pid, msg.word_index, msg.accuracy)
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

    # Both joined and both ready -> start the duel.
    if lob.match_id and lobby.both_ready(lob):
        match = state.matches.get(lob.match_id)
        if match is not None and match.state == "connecting":
            await _start_duel(match)


async def _start_duel(match: Match) -> None:
    now = asyncio.get_running_loop().time()
    match_engine.start_match(match, now)
    await manager.broadcast(
        match.player_ids, MatchStart(match_id=match.id, word_seed=match.word_seed)
    )
    await _broadcast_match_state(match)


# --- sign attempts (authoritative duel) -------------------------------------


async def _on_sign_attempt(pid: int, word_index: int, accuracy: float) -> None:
    match = _current_match(pid)
    if match is None:
        await manager.send(pid, error("not_in_match", "You are not in a match"))
        return
    now = asyncio.get_running_loop().time()
    try:
        finished = match_engine.handle_attempt(match, pid, word_index, accuracy, now)
    except match_engine.MatchError as exc:
        await manager.send(pid, error(exc.code, exc.message))
        return

    await _broadcast_match_state(match)
    if finished:
        await manager.broadcast(
            match.player_ids,
            MatchOver(match_id=match.id, winner_id=match.winner_id, reason="win"),
        )
        _teardown_match(match)


def _current_match(pid: int) -> Match | None:
    player = state.players.get(pid)
    mid = player.match_id if player is not None else None
    return state.matches.get(mid) if mid else None


def _match_state(match: Match) -> MatchState:
    return MatchState(
        match_id=match.id,
        players=[
            PlayerState(player_id=pid, hp=round(match.hp[pid], 2), word_index=match.word_index[pid])
            for pid in match.player_ids
        ],
    )


async def _broadcast_match_state(match: Match) -> None:
    await manager.broadcast(match.player_ids, _match_state(match))


def _teardown_match(match: Match) -> None:
    """Clear live state after a match ends. (Persistence + ELO land in 1f;
    replay finalize in 1g.)"""
    for pid in match.player_ids:
        player = state.players.get(pid)
        if player is not None:
            player.status = "idle"
            player.match_id = None
            player.lobby_code = None
    state.matches.pop(match.id, None)
    state.lobbies.pop(match.lobby_code, None)


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
