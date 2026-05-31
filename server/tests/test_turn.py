from __future__ import annotations

import base64
import hashlib
import hmac
import types

from app import turn


def test_make_turn_credentials_matches_coturn_hmac():
    secret, uid, ttl, now = "s3cr3t", "42", 3600, 1_000.0
    username, credential = turn.make_turn_credentials(secret, uid, ttl, now=now)

    assert username == f"{int(now) + ttl}:{uid}"  # "4600:42"
    expected = base64.b64encode(
        hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()
    ).decode()
    assert credential == expected


def test_credentials_expiry_advances_with_time():
    u1, _ = turn.make_turn_credentials("s", "1", 60, now=0.0)
    u2, _ = turn.make_turn_credentials("s", "1", 60, now=100.0)
    assert int(u1.split(":")[0]) == 60
    assert int(u2.split(":")[0]) == 160


def test_build_ice_servers_stun_only_by_default():
    # Default settings have no turn_host -> STUN only, no credentials leaked.
    servers = turn.build_ice_servers(user_id=7)
    assert servers and all("turn" not in str(s["urls"]).split(":")[0] for s in servers)
    assert all("credential" not in s for s in servers)


def test_build_ice_servers_includes_turn_when_configured(monkeypatch):
    fake = types.SimpleNamespace(
        stun_urls=["stun:stun.example.com:3478"],
        turn_host="turn.example.com",
        turn_udp_port=3478,
        turn_tls_port=5349,
        turn_secret="shared-secret",
        turn_ttl_seconds=3600,
    )
    monkeypatch.setattr(turn, "get_settings", lambda: fake)

    servers = turn.build_ice_servers(user_id=99, now=0.0)
    turn_entry = next(s for s in servers if "credential" in s)
    assert turn_entry["username"].endswith(":99")
    assert any(u.startswith("turn:turn.example.com:3478") for u in turn_entry["urls"])
    assert any(u.startswith("turns:turn.example.com:5349") for u in turn_entry["urls"])
