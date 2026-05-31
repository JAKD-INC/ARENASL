"""Session replay: chunk storage, manifest finalize, retention, and the HTTP
endpoints (auth + participant gating)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from app import replay, state
from app.state import Match
from tests.conftest import register


def _match(match_id: str = "mr1", players=(1, 2)) -> Match:
    m = Match(
        id=match_id,
        player_ids=players,
        roles={players[0]: "offerer", players[1]: "answerer"},
        lobby_code="C",
        state="finished",
        winner_id=players[0],
    )
    m.event_log = [{"t": 0, "type": "start"}, {"t": 4200, "type": "over", "winner_id": players[0]}]
    return m


# --- file logic -------------------------------------------------------------


def test_append_and_finalize_concatenates_in_seq_order(domain):
    m = _match()
    # Upload out of order to prove seq ordering on concat.
    replay.append_chunk(m.id, 1, seq=1, data=b"BBB")
    replay.append_chunk(m.id, 1, seq=0, data=b"AAA")
    replay.append_chunk(m.id, 2, seq=0, data=b"ZZZ")

    manifest_path = replay.finalize_match(m, history_id=None, reason="win")
    assert manifest_path.is_file()

    base = Path(os.environ["REPLAY_DIR"]) / m.id
    assert (base / "1.webm").read_bytes() == b"AAABBB"  # 0 then 1
    assert (base / "2.webm").read_bytes() == b"ZZZ"


def test_finalize_writes_manifest_with_feeds_and_event_log(domain):
    m = _match()
    replay.append_chunk(m.id, 1, seq=0, data=b"x")
    replay.finalize_match(m, history_id=None, reason="win")

    manifest = replay.load_manifest(m.id)
    assert manifest["match_id"] == m.id
    assert manifest["winner_id"] == 1
    assert manifest["duration_ms"] == 4200
    assert manifest["event_log"][-1]["type"] == "over"
    by_pid = {p["player_id"]: p for p in manifest["players"]}
    assert by_pid[1]["feed"] == "1.webm"
    assert by_pid[2]["feed"] is None  # player 2 uploaded nothing


def test_finalize_writes_replay_row_when_history_known(domain):
    from datetime import datetime, timezone

    from app.db import SessionLocal
    from app.models_db import MatchHistory, Replay, User

    with SessionLocal() as db:
        a = User(email="a@e.com", password_hash="x", display_name="A", elo=1200)
        b = User(email="b@e.com", password_hash="x", display_name="B", elo=1200)
        db.add_all([a, b])
        db.commit()
        aid, bid = a.id, b.id
        hist = MatchHistory(
            player_a=aid, player_b=bid, winner_id=aid, elo_delta=20,
            started_at=datetime.now(timezone.utc),
        )
        db.add(hist)
        db.commit()
        hid = hist.id

    m = _match(players=(aid, bid))
    replay.append_chunk(m.id, aid, seq=0, data=b"x")
    replay.finalize_match(m, history_id=hid, reason="win")

    with SessionLocal() as db:
        row = db.get(Replay, m.id)
        assert row is not None and row.match_history_id == hid


def test_is_participant():
    manifest = {"players": [{"player_id": 1}, {"player_id": 2}]}
    assert replay.is_participant(manifest, 1)
    assert not replay.is_participant(manifest, 99)


def test_sweep_removes_old_dirs_only(domain):
    root = Path(os.environ["REPLAY_DIR"])
    old = root / "old_match"
    new = root / "new_match"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    # Age the old dir by 30 days.
    old_time = time.time() - 30 * 86400
    os.utime(old, (old_time, old_time))

    removed = replay.sweep_retention(retention_days=14)
    assert removed == 1
    assert not old.exists()
    assert new.exists()


# --- HTTP endpoints ---------------------------------------------------------


def test_chunk_upload_and_manifest_fetch(client):
    token = register(client, email="rp1@example.com", display_name="One")
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    uid = me["player_id"]

    # Inject a live match the user participates in.
    mid = "httpmatch1"
    state.matches[mid] = Match(
        id=mid, player_ids=(uid, uid + 999),
        roles={uid: "offerer", uid + 999: "answerer"}, lobby_code="C", state="active",
    )

    h = {"Authorization": f"Bearer {token}"}
    r = client.post(f"/replay/{mid}/chunk?seq=0", headers=h, content=b"chunkdata")
    assert r.status_code == 204

    # Not finalized yet -> 404.
    assert client.get(f"/replay/{mid}", headers=h).status_code == 404

    # Finalize, then the participant can fetch the manifest.
    replay.finalize_match(state.matches[mid], history_id=None, reason="win")
    got = client.get(f"/replay/{mid}", headers=h)
    assert got.status_code == 200
    assert got.json()["match_id"] == mid


def test_chunk_upload_rejected_for_non_participant(client):
    token = register(client, email="rp2@example.com")
    uid = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["player_id"]
    mid = "httpmatch2"
    # Match exists but this user is NOT a participant.
    state.matches[mid] = Match(
        id=mid, player_ids=(uid + 1, uid + 2),
        roles={uid + 1: "offerer", uid + 2: "answerer"}, lobby_code="C", state="active",
    )
    r = client.post(f"/replay/{mid}/chunk?seq=0",
                    headers={"Authorization": f"Bearer {token}"}, content=b"x")
    assert r.status_code == 404


def test_manifest_fetch_forbidden_for_non_participant(client):
    owner = register(client, email="own@example.com")
    owner_id = client.get("/auth/me", headers={"Authorization": f"Bearer {owner}"}).json()["player_id"]
    other = register(client, email="other@example.com")

    mid = "httpmatch3"
    m = Match(id=mid, player_ids=(owner_id, owner_id + 500),
              roles={owner_id: "offerer", owner_id + 500: "answerer"}, lobby_code="C",
              state="finished", winner_id=owner_id)
    state.matches[mid] = m
    replay.finalize_match(m, history_id=None, reason="win")

    r = client.get(f"/replay/{mid}", headers={"Authorization": f"Bearer {other}"})
    assert r.status_code == 403
