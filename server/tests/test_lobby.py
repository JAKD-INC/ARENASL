"""Domain-level tests for app/lobby.py (no WS / no event loop)."""

from __future__ import annotations

import pytest

from app import lobby, state
from app.state import Player


def _player(pid: int, name: str = "P", elo: int = 1200) -> None:
    state.players[pid] = Player(id=pid, display_name=name, elo=elo)


def test_create_lobby(domain):
    _player(1)
    lob = lobby.create_lobby(1)
    assert lob.code in state.lobbies
    assert len(lob.members) == 1
    assert state.players[1].status == "in_lobby"
    assert state.players[1].lobby_code == lob.code


def test_join_fills_and_creates_match_with_roles(domain):
    _player(1, "One")
    _player(2, "Two")
    lob = lobby.create_lobby(1)
    lob2, match = lobby.join_lobby(2, lob.code)

    assert lob2 is lob and lob.is_full
    assert match is not None
    assert match.player_ids == (1, 2)
    assert match.roles == {1: "offerer", 2: "answerer"}  # host=offerer, joiner=answerer
    assert state.matches[match.id] is match
    assert lob.match_id == match.id


def test_join_is_case_insensitive_on_code(domain):
    _player(1)
    _player(2)
    lob = lobby.create_lobby(1)
    _, match = lobby.join_lobby(2, lob.code.lower())
    assert match is not None


def test_third_player_rejected(domain):
    for pid in (1, 2, 3):
        _player(pid)
    lob = lobby.create_lobby(1)
    lobby.join_lobby(2, lob.code)
    with pytest.raises(lobby.LobbyError) as exc:
        lobby.join_lobby(3, lob.code)
    assert exc.value.code == "lobby_full"


def test_join_unknown_code(domain):
    _player(1)
    with pytest.raises(lobby.LobbyError) as exc:
        lobby.join_lobby(1, "ZZZZZZ")
    assert exc.value.code == "lobby_not_found"


def test_join_same_lobby_twice(domain):
    _player(1)
    lob = lobby.create_lobby(1)
    with pytest.raises(lobby.LobbyError) as exc:
        lobby.join_lobby(1, lob.code)
    assert exc.value.code == "already_in_lobby"


def test_set_ready_and_both_ready(domain):
    _player(1)
    _player(2)
    lob = lobby.create_lobby(1)
    lobby.join_lobby(2, lob.code)
    assert not lobby.both_ready(lob)
    lobby.set_ready(1, True)
    assert not lobby.both_ready(lob)
    lobby.set_ready(2, True)
    assert lobby.both_ready(lob)


def test_set_ready_outside_lobby(domain):
    _player(1)
    with pytest.raises(lobby.LobbyError) as exc:
        lobby.set_ready(1, True)
    assert exc.value.code == "not_in_lobby"


def test_leave_returns_survivor_and_dissolves_match(domain):
    _player(1, "One")
    _player(2, "Two")
    lob = lobby.create_lobby(1)
    _, match = lobby.join_lobby(2, lob.code)

    survivor = lobby.leave_lobby(2)
    assert survivor is lob
    assert [m.player_id for m in survivor.members] == [1]
    assert match.id not in state.matches  # match dissolved
    assert survivor.match_id is None
    assert not survivor.members[0].ready  # readiness reset
    assert state.players[2].status == "idle"


def test_leave_last_member_deletes_lobby(domain):
    _player(1)
    lob = lobby.create_lobby(1)
    assert lobby.leave_lobby(1) is None
    assert lob.code not in state.lobbies


def test_creating_again_leaves_previous_lobby(domain):
    _player(1)
    first = lobby.create_lobby(1)
    second = lobby.create_lobby(1)
    assert first.code not in state.lobbies
    assert second.code in state.lobbies
