"""5-second reconnection grace: a mid-match drop notifies the opponent and starts
a forfeit timer; reconnecting in time cancels it, otherwise the opponent wins.

Each scenario runs in ONE asyncio.run so the forfeit task lives across steps; the
grace is monkeypatched tiny so tests are fast.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from app import state
from app.messages import LobbyCreate, LobbyJoin, LobbyReady
from app.ws import handlers
from tests.conftest import register_players


@pytest.fixture
def fast_grace(monkeypatch):
    monkeypatch.setattr(
        handlers, "get_settings", lambda: types.SimpleNamespace(reconnect_grace_seconds=0.05)
    )


async def _start_match(fm) -> None:
    await handlers.dispatch(1, LobbyCreate(type="lobby.create"))
    code = fm.last(1, "lobby.update")["code"]
    await handlers.dispatch(2, LobbyJoin(type="lobby.join", code=code))
    await handlers.dispatch(1, LobbyReady(type="lobby.ready", ready=True))
    await handlers.dispatch(2, LobbyReady(type="lobby.ready", ready=True))
    assert state.matches  # an active match exists


def test_disconnect_then_timeout_forfeits(fake, fast_grace):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))

    async def scenario():
        await _start_match(fake)
        await handlers.handle_disconnect(2)
        assert fake.last(1, "opponent.status")["connected"] is False
        await asyncio.sleep(0.12)  # exceed the 0.05 grace

    asyncio.run(scenario())

    over = fake.last(1, "match.over")
    assert over["winner_id"] == 1
    assert over["reason"] == "forfeit"
    assert state.players[1].status == "idle"
    assert 2 not in state.players  # the disconnected ghost is gone
    assert not state.matches


def test_reconnect_within_grace_cancels_forfeit(fake, fast_grace):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))

    async def scenario():
        await _start_match(fake)
        await handlers.handle_disconnect(2)
        await asyncio.sleep(0.01)  # still within grace

        # Player 2's socket comes back.
        await fake.connect(2, None)
        reconnected = handlers.register_or_reconnect(2, "Two", 1200)
        assert reconnected is True
        await handlers.resume_after_reconnect(2)

        await asyncio.sleep(0.1)  # past the old grace window

    asyncio.run(scenario())

    # Opponent told they're back; player 2 re-synced; no forfeit happened.
    assert fake.last(1, "opponent.status")["connected"] is True
    assert fake.last(2, "match.start")["match_id"]
    assert fake.last(2, "match.state")
    assert "match.over" not in fake.types_for(1)
    assert state.matches  # match still live


def test_non_match_disconnect_still_cleans_up(fake):
    # A player not in a match is simply forgotten on disconnect.
    register_players(fake, (1, "One", 1200))

    async def scenario():
        await handlers.dispatch(1, LobbyCreate(type="lobby.create"))
        await handlers.handle_disconnect(1)

    asyncio.run(scenario())
    assert 1 not in state.players
