from __future__ import annotations

import pytest
from fastapi import WebSocketDisconnect

from app.elo import STARTING_ELO, Experience
from tests.conftest import register


def test_ws_auth_ok(client):
    token = register(client, email="ws@example.com")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": token})
        msg = ws.receive_json()
        assert msg["type"] == "auth.ok"
        assert msg["elo"] == STARTING_ELO[Experience.intermediate]
        assert isinstance(msg["player_id"], int)


def test_ws_unsupported_message_after_auth(client):
    # `signal` is a valid schema but not implemented until phase 1d.
    token = register(client, email="echo@example.com")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": token})
        assert ws.receive_json()["type"] == "auth.ok"
        ws.send_json({"type": "signal", "data": {}})
        resp = ws.receive_json()
        assert resp["type"] == "error"
        assert resp["code"] == "unsupported"


def test_ws_bad_token_is_closed(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": "garbage-token"})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_first_message_must_be_auth(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "queue.join"})  # not an auth message
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_create_lobby_end_to_end(client):
    # Single socket exercises the real ConnectionManager on one event loop.
    token = register(client, email="solo@example.com", display_name="Solo")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": token})
        assert ws.receive_json()["type"] == "auth.ok"
        ws.send_json({"type": "lobby.create"})
        upd = ws.receive_json()
        assert upd["type"] == "lobby.update"
        assert upd["state"] == "waiting"
        assert len(upd["members"]) == 1 and upd["members"][0]["display_name"] == "Solo"
        assert len(upd["code"]) == 6


def test_ws_queue_join_emits_warmup(client):
    # Single socket: the real ticker runs but won't pair a lone player.
    token = register(client, email="queue@example.com")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": token})
        assert ws.receive_json()["type"] == "auth.ok"
        ws.send_json({"type": "queue.join"})
        warm = ws.receive_json()
        assert warm["type"] == "warmup.start"
        assert isinstance(warm["word_seed"], int)
        assert warm["dataset_version"]
        status = ws.receive_json()
        assert status["type"] == "queue.status" and status["position"] == 1


def test_ws_bad_message_after_auth_gets_error_not_disconnect(client):
    token = register(client, email="badmsg@example.com")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "auth", "token": token})
        assert ws.receive_json()["type"] == "auth.ok"
        ws.send_json({"type": "totally.unknown"})  # invalid discriminator
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "bad_message"
