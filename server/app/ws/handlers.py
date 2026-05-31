"""Per-message dispatch for the WebSocket. Maps a validated client message to a
domain action and emits the resulting server messages, and owns connect /
reconnect / disconnect orchestration for live state.
"""

from __future__ import annotations

import asyncio
import logging

from app import lobby, matchmaking, recognition, replay, results, signaling, state, turn, warmup
from app import match as match_engine
from app.config import get_settings
from app.connection_manager import manager
from app.messages import (
    ClientMessage,
    Landmark,
    LobbyCreate,
    LobbyJoin,
    LobbyMemberView,
    LobbyReady,
    LobbyUpdate,
    MatchFound,
    MatchOver,
    MatchStart,
    MatchState,
    OpponentStatus,
    OpponentView,
    PlayerState,
    QueueJoin,
    QueueLeave,
    QueueStatus,
    RecognitionUpdate,
    Signal,
    WarmupStart,
    error,
)
from app.state import Lobby, Match, Player
from app.words import get_dataset

logger = logging.getLogger("arenasl.handlers")

# Pending forfeit timers, keyed by the disconnected player's id. Holding the task
# here keeps a strong reference (the event loop only weak-refs tasks).
_forfeit_tasks: dict[int, asyncio.Task] = {}

# Last processed landmark timestamp (client clock) per player — used to downsample
# recognition to RECOGNITION_FPS_CAP frames per second of gameplay time.
_last_frame_t: dict[int, float] = {}


def register_player(pid: int, display_name: str, elo: int) -> None:
    state.players[pid] = Player(id=pid, display_name=display_name, elo=elo)


def reset() -> None:
    """Test helper: clear module-level per-connection bookkeeping."""
    _forfeit_tasks.clear()
    _last_frame_t.clear()


def register_or_reconnect(pid: int, display_name: str, elo: int) -> bool:
    """Called on WS connect. If the player has an in-flight active match, treat
    this as a reconnect (cancel the forfeit timer, keep live state) and return
    True. Otherwise register a fresh player and return False."""
    existing = state.players.get(pid)
    match = state.matches.get(existing.match_id) if existing and existing.match_id else None
    if existing is not None and match is not None and match.state == "active":
        existing.display_name = display_name
        task = _forfeit_tasks.pop(pid, None)
        if task is not None:
            task.cancel()
        return True
    register_player(pid, display_name, elo)
    return False


async def resume_after_reconnect(pid: int) -> None:
    """Re-sync a reconnected player: tell the opponent, replay match.start +
    current match.state."""
    match = _current_match(pid)
    if match is None or match.state != "active":
        return
    opp = _other(pid, match)
    await manager.send(opp, OpponentStatus(player_id=pid, connected=True))
    await manager.send(pid, MatchStart(match_id=match.id, word_seed=match.word_seed))
    await manager.send(pid, _match_state(match))


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
    elif isinstance(msg, Landmark):
        await _on_landmark(pid, msg)
    else:
        await manager.send(
            pid, error("unsupported", f"'{msg.type}' is not available yet")
        )


async def announce_match(lob: Lobby, match: Match) -> None:
    """Tell both players a match formed: lobby is full, here are connect params."""
    await _broadcast_lobby_update(lob)
    await _send_match_found(match)


async def handle_disconnect(pid: int) -> None:
    """On socket close: if mid-match, start the reconnection grace; otherwise
    leave the queue / lobby and forget the player."""
    match = _current_match(pid)
    if match is not None and match.state == "active":
        await _begin_reconnect_grace(pid, match)
        await manager.disconnect(pid)  # drop the socket but KEEP player + match
        return

    matchmaking.leave_queue(pid)
    _last_frame_t.pop(pid, None)
    remaining = lobby.leave_lobby(pid)
    if remaining is not None:
        await _broadcast_lobby_update(remaining)
    await manager.disconnect(pid)
    state.players.pop(pid, None)


# --- reconnection grace / forfeit -------------------------------------------


async def _begin_reconnect_grace(pid: int, match: Match) -> None:
    opp = _other(pid, match)
    await manager.send(opp, OpponentStatus(player_id=pid, connected=False))
    match.event_log.append({"type": "disconnect", "player_id": pid})
    _forfeit_tasks[pid] = asyncio.create_task(_forfeit_after(match.id, pid))


async def _forfeit_after(match_id: str, pid: int) -> None:
    grace = get_settings().reconnect_grace_seconds
    try:
        await asyncio.sleep(grace)
    except asyncio.CancelledError:
        raise  # reconnected in time
    _forfeit_tasks.pop(pid, None)
    match = state.matches.get(match_id)
    if match is None or match.state != "active":
        return  # match already ended some other way
    match.state = "finished"
    match.winner_id = _other(pid, match)
    match.event_log.append({"type": "forfeit", "player_id": pid})
    await _finalize_match(match, reason="forfeit")


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
    if not recognition.is_ready():
        await manager.broadcast(
            match.player_ids,
            error("recognition_unavailable", "Sign recognition is not available"),
        )
        return
    now = asyncio.get_running_loop().time()
    match_engine.start_match(match, now)
    for pid in match.player_ids:
        match.recognizers[pid] = recognition.new_session(match.word_seed)
    await manager.broadcast(
        match.player_ids, MatchStart(match_id=match.id, word_seed=match.word_seed)
    )
    await _broadcast_match_state(match)


# --- landmark frames -> server-side recognition (authoritative duel) --------


async def _on_landmark(pid: int, msg: Landmark) -> None:
    if _throttled(pid, msg.t):
        return  # downsample to the recognition fps cap

    match = _current_match(pid)
    if match is not None and match.state == "active":
        session = match.recognizers.get(pid)
        if session is not None:
            await _recognize_in_match(pid, match, session, msg)
        return

    # Warmup: a queued player practices; report strength, deal no damage.
    player = state.players.get(pid)
    if player is not None and player.status == "queued" and player.warmup_session is not None:
        outcome = await asyncio.to_thread(
            player.warmup_session.push_landmarks, msg.pose, msg.hand_left, msg.hand_right, msg.t
        )
        if outcome is not None:
            await manager.send(
                pid,
                RecognitionUpdate(
                    word_index=outcome.word_index, word=outcome.word, strength=outcome.strength
                ),
            )


def _throttled(pid: int, t: float) -> bool:
    """True if this frame should be dropped to honor the per-player fps cap."""
    cap = get_settings().recognition_fps_cap
    if cap <= 0:
        return False
    last = _last_frame_t.get(pid)
    if last is not None and t - last < 1.0 / cap:
        return True
    _last_frame_t[pid] = t
    return False


async def _recognize_in_match(pid: int, match: Match, session, msg: Landmark) -> None:
    # DTW is CPU-bound; offload so it never blocks the loop. Frames for one
    # player are processed serially (the receive loop awaits each dispatch).
    outcome = await asyncio.to_thread(
        session.push_landmarks, msg.pose, msg.hand_left, msg.hand_right, msg.t
    )
    if outcome is None:
        return  # unusable frame (no pose / degenerate shoulders)

    now = asyncio.get_running_loop().time()
    if outcome.event == "get":
        finished = match_engine.apply_completion(
            match, pid, outcome.word, outcome.word_index, outcome.strength, now
        )
        await _broadcast_match_state(match)
        if finished:
            await _finalize_match(match, reason="win")
    elif outcome.event == "miss":
        match_engine.apply_miss(match, pid, outcome.word_index, now)
        await _broadcast_match_state(match)
    else:
        await manager.send(
            pid,
            RecognitionUpdate(
                word_index=outcome.word_index, word=outcome.word, strength=outcome.strength
            ),
        )


async def _finalize_match(match: Match, reason: str) -> None:
    """End a match: apply ELO + write history (off the loop), tell both players
    their result, then tear down live state."""
    deltas, history_id = await asyncio.to_thread(results.persist_match_result, match, reason)
    try:
        await asyncio.to_thread(replay.finalize_match, match, history_id, reason)
    except Exception:  # replay is best-effort, never block the result
        logger.exception("replay finalize failed for match %s", match.id)
    for pid in match.player_ids:
        new_elo, delta = deltas.get(pid, (None, None))
        player = state.players.get(pid)
        if player is not None and new_elo is not None:
            player.elo = new_elo
        await manager.send(
            pid,
            MatchOver(
                match_id=match.id,
                winner_id=match.winner_id,
                reason=reason,
                elo=new_elo,
                elo_delta=delta,
            ),
        )
    _teardown_match(match)


def _current_match(pid: int) -> Match | None:
    player = state.players.get(pid)
    mid = player.match_id if player is not None else None
    return state.matches.get(mid) if mid else None


def _other(pid: int, match: Match) -> int:
    return next(p for p in match.player_ids if p != pid)


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
    """Clear live state after a match ends. Connected players return to idle;
    a player who is gone (forfeit) is forgotten. (Replay finalize lands in 1g.)"""
    for pid in match.player_ids:
        task = _forfeit_tasks.pop(pid, None)
        if task is not None:
            task.cancel()
        _last_frame_t.pop(pid, None)
        player = state.players.get(pid)
        if player is None:
            continue
        if manager.is_connected(pid):
            player.status = "idle"
            player.match_id = None
            player.lobby_code = None
        else:
            state.players.pop(pid, None)  # ghost from a disconnect/forfeit
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

    # Hand out a warmup stream + a server-side recognizer for live practice
    # feedback (no scoring). The ticker pairs players.
    seed = warmup.new_seed()
    if recognition.is_ready():
        player.warmup_session = recognition.new_session(seed)
    await manager.send(pid, WarmupStart(word_seed=seed, dataset_version=get_dataset().version))
    await manager.send(pid, QueueStatus(position=matchmaking.queue_position(pid)))


async def _on_queue_leave(pid: int) -> None:
    matchmaking.leave_queue(pid)
    player = state.players.get(pid)
    if player is not None:
        player.warmup_session = None
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
