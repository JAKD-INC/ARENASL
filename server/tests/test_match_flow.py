"""Handler-level duel flow: ready gate -> match.start -> match.state -> match.over."""

from __future__ import annotations

import asyncio

from app import state
from app.messages import LobbyCreate, LobbyJoin, LobbyReady, SignAttempt
from app.ws import handlers
from tests.conftest import register_players


def _ready_lobby(fm) -> str:
    """Both players join a lobby and ready up; returns the match id."""
    asyncio.run(handlers.dispatch(1, LobbyCreate(type="lobby.create")))
    code = fm.last(1, "lobby.update")["code"]
    asyncio.run(handlers.dispatch(2, LobbyJoin(type="lobby.join", code=code)))
    asyncio.run(handlers.dispatch(1, LobbyReady(type="lobby.ready", ready=True)))
    asyncio.run(handlers.dispatch(2, LobbyReady(type="lobby.ready", ready=True)))
    return fm.last(1, "match.found")["match_id"]


def test_both_ready_starts_duel(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    mid = _ready_lobby(fake)

    start = fake.last(1, "match.start")
    assert start["match_id"] == mid
    assert isinstance(start["word_seed"], int)
    assert start["record_start_ms"] == 0

    st = fake.last(1, "match.state")
    hp = {p["player_id"]: p["hp"] for p in st["players"]}
    assert hp == {1: 100.0, 2: 100.0}


def test_duel_does_not_start_until_both_ready(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    asyncio.run(handlers.dispatch(1, LobbyCreate(type="lobby.create")))
    code = fake.last(1, "lobby.update")["code"]
    asyncio.run(handlers.dispatch(2, LobbyJoin(type="lobby.join", code=code)))
    asyncio.run(handlers.dispatch(1, LobbyReady(type="lobby.ready", ready=True)))
    assert "match.start" not in fake.types_for(1)  # only one ready


def test_sign_attempt_updates_state_and_can_win(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _ready_lobby(fake)

    # Drive player 1's attempts until match.over is broadcast.
    won = False
    for idx in range(200):
        asyncio.run(handlers.dispatch(1, SignAttempt(type="sign.attempt", word_index=idx, accuracy=1.0)))
        if "match.over" in fake.types_for(1):
            won = True
            break
    assert won
    over = fake.last(1, "match.over")
    assert over["winner_id"] == 1
    assert over["reason"] == "win"
    # Live state torn down after the match.
    assert 1 not in [m for m in state.matches]
    assert state.players[1].status == "idle"


def test_bad_attempt_returns_error_without_ending(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _ready_lobby(fake)
    asyncio.run(handlers.dispatch(1, SignAttempt(type="sign.attempt", word_index=9, accuracy=1.0)))
    assert fake.last(1, "error")["code"] == "bad_word_index"
