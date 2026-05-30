"""Signaling relay: opaque forwarding to the other player in the match, and the
match.found ice_servers payload. Driven via handlers + FakeManager."""

from __future__ import annotations

import asyncio

import pytest

from app import signaling
from app.messages import LobbyCreate, LobbyJoin, Signal
from app.ws import handlers
from tests.conftest import register_players


def _full_lobby(fm) -> None:
    asyncio.run(handlers.dispatch(1, LobbyCreate(type="lobby.create")))
    code = fm.last(1, "lobby.update")["code"]
    asyncio.run(handlers.dispatch(2, LobbyJoin(type="lobby.join", code=code)))


def test_signal_relayed_verbatim_to_peer(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _full_lobby(fake)

    payload = {"sdp": {"type": "offer", "sdp": "v=0..."}, "extra": [1, 2, 3]}
    asyncio.run(handlers.dispatch(1, Signal(type="signal", data=payload)))

    relayed = fake.last(2, "signal")
    assert relayed["data"] == payload  # forwarded unchanged
    # Sender does not receive their own signal.
    assert "signal" not in fake.types_for(1)


def test_signal_without_match_errors(fake):
    register_players(fake, (1, "One", 1200))
    asyncio.run(handlers.dispatch(1, Signal(type="signal", data={"x": 1})))
    assert fake.last(1, "error")["code"] == "not_in_match"


def test_match_found_includes_ice_servers(fake):
    register_players(fake, (1, "One", 1200), (2, "Two", 1200))
    _full_lobby(fake)
    mf = fake.last(1, "match.found")
    assert mf["ice_servers"]  # at least STUN
    assert any("stun" in str(s["urls"]) for s in mf["ice_servers"])


def test_peer_of_helper(domain):
    from app import lobby, state
    from app.state import Player

    state.players[1] = Player(id=1, display_name="A", elo=1200)
    state.players[2] = Player(id=2, display_name="B", elo=1200)
    lob = lobby.create_lobby(1)
    lobby.join_lobby(2, lob.code)
    assert signaling.peer_of(1) == 2
    assert signaling.peer_of(2) == 1


def test_peer_of_raises_without_match(domain):
    from app import state
    from app.state import Player

    state.players[1] = Player(id=1, display_name="A", elo=1200)
    with pytest.raises(signaling.SignalingError):
        signaling.peer_of(1)
