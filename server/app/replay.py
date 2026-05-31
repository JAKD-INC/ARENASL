"""Server-side session-replay ingest.

Each client records its own webcam and uploads WebM chunks over HTTP (off the
gameplay WebSocket). Chunks land as per-player .part files; on match end they're
concatenated in seq order into one feed file and a manifest.json is written
(feeds + the timestamped event log). Capture is best-effort: a dropped chunk
degrades the replay, never the match.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from app import state
from app.config import get_settings
from app.db import SessionLocal
from app.models_db import Replay

logger = logging.getLogger("arenasl.replay")


def _match_dir(match_id: str) -> Path:
    return Path(get_settings().replay_dir) / match_id


def _player_dir(match_id: str, player_id: int) -> Path:
    return _match_dir(match_id) / str(player_id)


def append_chunk(match_id: str, player_id: int, seq: int, data: bytes) -> None:
    """Persist one uploaded chunk. seq orders chunks within a player's feed."""
    d = _player_dir(match_id, player_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{seq:08d}.part").write_bytes(data)


def finalize_match(match: state.Match, history_id: int | None, reason: str) -> Path | None:
    """Concatenate each player's chunks into a feed file and write manifest.json.
    Inserts a Replay row when history_id is known. Returns the manifest path."""
    mdir = _match_dir(match.id)
    mdir.mkdir(parents=True, exist_ok=True)

    feeds: dict[int, str | None] = {}
    for pid in match.player_ids:
        pdir = _player_dir(match.id, pid)
        parts = sorted(pdir.glob("*.part")) if pdir.exists() else []
        if not parts:
            feeds[pid] = None
            continue
        feed = mdir / f"{pid}.webm"
        with feed.open("wb") as out:
            for part in parts:
                out.write(part.read_bytes())
        feeds[pid] = feed.name

    duration_ms = max((e.get("t", 0) for e in match.event_log), default=0)
    manifest = {
        "match_id": match.id,
        "winner_id": match.winner_id,
        "reason": reason,
        "duration_ms": duration_ms,
        "players": [
            {"player_id": pid, "role": match.roles.get(pid), "feed": feeds[pid]}
            for pid in match.player_ids
        ],
        "event_log": match.event_log,
    }
    manifest_path = mdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    if history_id is not None:
        with SessionLocal() as db:
            db.add(
                Replay(
                    match_id=match.id,
                    match_history_id=history_id,
                    duration_ms=duration_ms,
                    manifest_path=str(manifest_path),
                )
            )
            db.commit()
    return manifest_path


def load_manifest(match_id: str) -> dict | None:
    p = _match_dir(match_id) / "manifest.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def is_participant(manifest: dict, user_id: int) -> bool:
    return any(pl["player_id"] == user_id for pl in manifest.get("players", []))


def sweep_retention(now: float | None = None, retention_days: int | None = None) -> int:
    """Delete replay match dirs older than the retention window. Returns the count."""
    s = get_settings()
    root = Path(s.replay_dir)
    if not root.exists():
        return 0
    days = s.replay_retention_days if retention_days is None else retention_days
    cutoff = (time.time() if now is None else now) - days * 86400
    removed = 0
    for child in root.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


async def retention_loop(interval_seconds: float = 3600.0) -> None:
    """Background sweep so replay storage stays bounded."""
    import asyncio

    try:
        while True:
            try:
                removed = sweep_retention()
                if removed:
                    logger.info("retention sweep removed %d replay dir(s)", removed)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("retention sweep failed")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        raise
