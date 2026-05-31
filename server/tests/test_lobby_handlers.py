"""Tests for ws/handlers lobby dispatch + the server messages it emits, driven on
a single event loop via asyncio.run with a FakeManager (see conftest)."""

from __future__ import annotations

import asyncio

from app import state
from app.messages import LobbyCreate, LobbyJoin, LobbyReady, QueueJoin
from app.ws import handlers
from tests.conftest import register_players


def _make_full_lobby(fm) -> str:
    asyncio.run(handlers.dispatch(1, LobbyCreate(type="lobby.create")))
    code = fm.last(1, "lobby.update")["code"]
    asyncio.run(handlers.dispatch(2, LobbyJoin(type="lobby.join", code=code)))
    return code


def test_create_then_join_emits_update_and_match_found(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _make_full_lobby(fake)

    mf1 = fake.last(1, "match.found")
    mf2 = fake.last(2, "match.found")
    assert mf1["role"] == "offerer"
    assert mf2["role"] == "answerer"
    assert mf1["match_id"] == mf2["match_id"]
    assert mf1["dataset_version"]
    assert mf1["opponent"]["display_name"] == "Two"
    assert mf2["opponent"]["display_name"] == "One"

    upd = fake.last(1, "lobby.update")
    assert upd["state"] == "full" and len(upd["members"]) == 2


def test_ready_reflected_in_update(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _make_full_lobby(fake)

    asyncio.run(handlers.dispatch(1, LobbyReady(type="lobby.ready", ready=True)))
    ready_map = {m["player_id"]: m["ready"] for m in fake.last(1, "lobby.update")["members"]}
    assert ready_map[1] is True
    assert ready_map[2] is False


def test_unsupported_message_errors(fake):
    register_players(fake, (1, "One", 1200))
    asyncio.run(handlers.dispatch(1, LobbyReady(type="lobby.ready", ready=True)))
    err = fake.last(1, "error")
    assert err["code"] == "not_in_lobby"


def test_disconnect_notifies_survivor_and_forgets_player(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _make_full_lobby(fake)

    asyncio.run(handlers.handle_disconnect(2))

    upd = fake.last(1, "lobby.update")
    assert upd["state"] == "waiting" and len(upd["members"]) == 1
    assert 2 not in state.players
