"""Matchmaking: pure window/pairing logic, and the queue handlers (warmup +
pairing → match.found) driven with a FakeManager."""

from __future__ import annotations

import asyncio

from app import matchmaking, state
from app.messages import QueueJoin, QueueLeave
from app.state import Player
from app.ws import handlers
from tests.conftest import register_players


def _player(pid: int, elo: int) -> None:
    state.players[pid] = Player(id=pid, display_name=f"P{pid}", elo=elo)


# --- pure scan_queue --------------------------------------------------------


def test_close_elos_pair_immediately(domain):
    _player(1, 1200)
    _player(2, 1230)  # gap 30 <= window_start (50)
    matchmaking.join_queue(1, now=0.0)
    matchmaking.join_queue(2, now=0.0)

    pairs = matchmaking.scan_queue(now=0.0)
    assert len(pairs) == 1
    lob, match = pairs[0]
    assert lob.origin == "queue"
    assert set(match.player_ids) == {1, 2}
    assert state.queue == []  # both removed


def test_far_elos_do_not_pair_until_window_widens(domain):
    _player(1, 1000)
    _player(2, 1120)  # gap 120; start window 50, widen +25 / 2s
    matchmaking.join_queue(1, now=0.0)
    matchmaking.join_queue(2, now=0.0)

    assert matchmaking.scan_queue(now=0.0) == []      # 120 > 50
    assert matchmaking.scan_queue(now=2.0) == []      # 120 > 75
    assert matchmaking.scan_queue(now=4.0) == []      # 120 > 100
    pairs = matchmaking.scan_queue(now=6.0)           # window 125 >= 120
    assert len(pairs) == 1
    assert state.queue == []


def test_single_player_never_pairs(domain):
    _player(1, 1200)
    matchmaking.join_queue(1, now=0.0)
    assert matchmaking.scan_queue(now=100.0) == []
    assert len(state.queue) == 1


def test_join_is_idempotent_and_leave_removes(domain):
    _player(1, 1200)
    matchmaking.join_queue(1, now=0.0)
    matchmaking.join_queue(1, now=5.0)  # no duplicate
    assert len(state.queue) == 1
    assert state.players[1].status == "queued"

    matchmaking.leave_queue(1)
    assert state.queue == []
    assert state.players[1].status == "idle"


def test_closest_pair_chosen_among_three(domain):
    _player(1, 1000)
    _player(2, 1020)
    _player(3, 1400)
    for pid in (1, 2, 3):
        matchmaking.join_queue(pid, now=0.0)

    pairs = matchmaking.scan_queue(now=0.0)
    assert len(pairs) == 1
    _, match = pairs[0]
    assert set(match.player_ids) == {1, 2}  # 3 is too far, stays queued
    assert [e.player_id for e in state.queue] == [3]


# --- queue handlers (warmup + announce) ------------------------------------


def test_queue_join_emits_warmup_and_status(fake):
    register_players(fake, (1, "One", 1200))
    asyncio.run(handlers.dispatch(1, QueueJoin(type="queue.join")))

    warm = fake.last(1, "warmup.start")
    assert isinstance(warm["word_seed"], int)
    assert warm["dataset_version"]
    assert fake.last(1, "queue.status")["position"] == 1
    assert matchmaking.is_queued(1)


def test_queue_leave_clears(fake):
    register_players(fake, (1, "One", 1200))
    asyncio.run(handlers.dispatch(1, QueueJoin(type="queue.join")))
    asyncio.run(handlers.dispatch(1, QueueLeave(type="queue.leave")))
    assert fake.last(1, "queue.status")["position"] == 0
    assert not matchmaking.is_queued(1)


def test_two_queued_players_get_match_found_via_announce(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1210))
    asyncio.run(handlers.dispatch(1, QueueJoin(type="queue.join")))
    asyncio.run(handlers.dispatch(2, QueueJoin(type="queue.join")))

    # Simulate one ticker pass on a single loop.
    async def tick():
        for lob, match in matchmaking.scan_queue(now=0.0):
            await handlers.announce_match(lob, match)

    asyncio.run(tick())

    mf1 = fake.last(1, "match.found")
    mf2 = fake.last(2, "match.found")
    assert {mf1["role"], mf2["role"]} == {"offerer", "answerer"}
    assert mf1["match_id"] == mf2["match_id"]
    assert state.players[1].status == "in_lobby"
    assert matchmaking.is_queued(1) is False


def test_queue_join_rejected_when_in_lobby(fake):
    register_players(fake, (1, "One", 1200))
    # Put player 1 in a lobby first.
    from app import lobby

    lobby.create_lobby(1)
    asyncio.run(handlers.dispatch(1, QueueJoin(type="queue.join")))
    assert fake.last(1, "error")["code"] == "busy"
