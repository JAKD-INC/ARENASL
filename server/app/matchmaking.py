"""ELO matchmaking: a queue with a search window that widens over time, scanned
by a background ticker. A pair forms a `queue`-origin lobby (lobby.py), which then
runs the same ready-up flow as a private lobby.

`scan_queue` is a single synchronous pass (pairing + lobby creation, no awaits, so
no locking needed). The async ticker wraps it and announces each match.
"""

from __future__ import annotations

import asyncio
import logging

from app import lobby, state
from app.config import get_settings
from app.state import Lobby, Match, QueueEntry

logger = logging.getLogger("arenasl.matchmaking")


def is_queued(pid: int) -> bool:
    return any(e.player_id == pid for e in state.queue)


def queue_position(pid: int) -> int:
    for i, e in enumerate(state.queue):
        if e.player_id == pid:
            return i + 1
    return 0


def join_queue(pid: int, now: float) -> None:
    """Add a player to the queue (idempotent). Caller ensures the player is idle."""
    if is_queued(pid):
        return
    player = state.players.get(pid)
    if player is None:
        return
    player.status = "queued"
    state.queue.append(QueueEntry(player_id=pid, elo=player.elo, joined_at=now))


def leave_queue(pid: int) -> None:
    state.queue[:] = [e for e in state.queue if e.player_id != pid]
    player = state.players.get(pid)
    if player is not None and player.status == "queued":
        player.status = "idle"


def _window(entry: QueueEntry, now: float) -> int:
    """Acceptable ELO gap for this entry, widening by a step every 2 seconds."""
    s = get_settings()
    steps = max(0, int((now - entry.joined_at) // 2))
    return s.mm_window_start + s.mm_window_widen_per_2s * steps


def scan_queue(now: float) -> list[tuple[Lobby, Match]]:
    """Pair the closest-ELO players whose windows both span the gap. Removes paired
    players from the queue and forms a lobby+match for each pair."""
    entries = sorted(state.queue, key=lambda e: e.elo)
    paired: set[int] = set()
    results: list[tuple[Lobby, Match]] = []

    i = 0
    while i < len(entries) - 1:
        a = entries[i]
        if a.player_id in paired:
            i += 1
            continue
        b = entries[i + 1]
        if b.player_id in paired:
            i += 1
            continue
        gap = abs(a.elo - b.elo)
        if gap <= min(_window(a, now), _window(b, now)):
            paired.add(a.player_id)
            paired.add(b.player_id)
            results.append(lobby.create_queue_lobby(a.player_id, b.player_id))
            i += 2
        else:
            i += 1

    if paired:
        state.queue[:] = [e for e in state.queue if e.player_id not in paired]
    return results


async def matchmaking_loop(interval: float = 1.0) -> None:
    """Background ticker: scan the queue once per interval and announce matches.
    One bad tick is logged and skipped; cancellation propagates for clean shutdown."""
    from app.ws.handlers import announce_match  # lazy: avoid import cycle

    loop = asyncio.get_running_loop()
    try:
        while True:
            start = loop.time()
            try:
                for lob, match in scan_queue(loop.time()):
                    await announce_match(lob, match)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("matchmaking tick failed")
            await asyncio.sleep(max(0.0, interval - (loop.time() - start)))
    except asyncio.CancelledError:
        raise
    finally:
        logger.info("matchmaking loop stopped")
