"""Tests for ws/handlers dispatch + the server messages it emits.

Uses a FakeManager that records sends, so we exercise the real handler/broadcast
logic on a single event loop (via asyncio.run) without TestClient's per-socket
event loops — which is what production (single uvicorn loop) actually looks like.
"""

from __future__ import annotations

import asyncio

import pytest

from app import state
from app.messages import LobbyCreate, LobbyJoin, LobbyReady, QueueJoin
from app.ws import handlers


class FakeManager:
    def __init__(self) -> None:
        self.sent: list[tuple[int, dict]] = []
        self.connected: set[int] = set()

    def is_connected(self, pid: int) -> bool:
        return pid in self.connected

    async def connect(self, pid: int, ws) -> None:  # noqa: ANN001
        self.connected.add(pid)

    async def disconnect(self, pid: int) -> None:
        self.connected.discard(pid)

    async def send(self, pid: int, msg) -> None:  # noqa: ANN001
        self.sent.append((pid, msg.model_dump()))

    async def broadcast(self, pids, msg) -> None:  # noqa: ANN001
        for pid in pids:
            await self.send(pid, msg)

    def for_(self, pid: int) -> list[dict]:
        return [m for (p, m) in self.sent if p == pid]

    def types_for(self, pid: int) -> list[str]:
        return [m["type"] for m in self.for_(pid)]

    def last(self, pid: int, type_: str) -> dict:
        return [m for m in self.for_(pid) if m["type"] == type_][-1]


@pytest.fixture
def fake(domain, monkeypatch):
    fm = FakeManager()
    monkeypatch.setattr(handlers, "manager", fm)
    return fm


def _register(fm: FakeManager, pid: int, name: str, elo: int = 1200) -> None:
    handlers.register_player(pid, name, elo)
    fm.connected.add(pid)


def _make_full_lobby(fm: FakeManager) -> str:
    asyncio.run(handlers.dispatch(1, LobbyCreate(type="lobby.create")))
    code = fm.last(1, "lobby.update")["code"]
    asyncio.run(handlers.dispatch(2, LobbyJoin(type="lobby.join", code=code)))
    return code


def test_create_then_join_emits_update_and_match_found(fake):
    _register(fake, 1, "One")
    _register(fake, 2, "Two")
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
    _register(fake, 1, "One")
    _register(fake, 2, "Two")
    _make_full_lobby(fake)

    asyncio.run(handlers.dispatch(1, LobbyReady(type="lobby.ready", ready=True)))
    ready_map = {m["player_id"]: m["ready"] for m in fake.last(1, "lobby.update")["members"]}
    assert ready_map[1] is True
    assert ready_map[2] is False


def test_unsupported_message_errors(fake):
    _register(fake, 1, "One")
    asyncio.run(handlers.dispatch(1, QueueJoin(type="queue.join")))
    err = fake.last(1, "error")
    assert err["code"] == "unsupported"


def test_disconnect_notifies_survivor_and_forgets_player(fake):
    _register(fake, 1, "One")
    _register(fake, 2, "Two")
    _make_full_lobby(fake)

    asyncio.run(handlers.handle_disconnect(2))

    upd = fake.last(1, "lobby.update")
    assert upd["state"] == "waiting" and len(upd["members"]) == 1
    assert 2 not in state.players
