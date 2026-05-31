"""FastAPI application entrypoint.

Run: `uvicorn app.main:app --reload` (dev).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.staticfiles import StaticFiles

from app import replay, state
from app.auth.deps import get_current_user
from app.auth.routes import router as auth_router
from app.config import get_settings
from app.lifespan import lifespan
from app.models_db import User
from app.words import get_dataset
from app.ws.endpoint import router as ws_router

app = FastAPI(title="ARENASL", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(ws_router)


def mount_clips(app_: FastAPI) -> bool:
    """Mount the reference-clip directory at /clips, serving GET /clips/<gloss>.mp4.

    Only mounts when the configured dir exists so tests/CI without built clips
    still start. Returns whether the mount was added. Re-callable (tests clear
    the settings cache and re-invoke after pointing the setting at a tmp dir)."""
    clips_dir = Path(get_settings().asl_clips_dir)
    if not clips_dir.is_dir():
        return False
    app_.mount("/clips", StaticFiles(directory=str(clips_dir)), name="clips")
    return True


mount_clips(app)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/signs", tags=["meta"])
def signs() -> dict:
    """The sign dataset (word_strength source). Clients must hold this exact
    version to ready up for a match."""
    return get_dataset().to_payload()


@app.post("/replay/{match_id}/chunk", status_code=status.HTTP_204_NO_CONTENT, tags=["replay"])
async def upload_replay_chunk(
    match_id: str,
    seq: int,
    user: Annotated[User, Depends(get_current_user)],
    request: Request,
    t_offset_ms: int = 0,
) -> Response:
    """Append one recorded chunk (raw body) for the authed player. Only a
    participant in the live match may upload; the player_id is taken from the JWT."""
    match = state.matches.get(match_id)
    if match is None or user.id not in match.player_ids:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such active match")
    data = await request.body()
    await asyncio.to_thread(replay.append_chunk, match_id, user.id, seq, data)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/replay/{match_id}", tags=["replay"])
def get_replay(match_id: str, user: Annotated[User, Depends(get_current_user)]) -> dict:
    """Return a finalized replay manifest. Participants only."""
    manifest = replay.load_manifest(match_id)
    if manifest is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No replay for that match")
    if not replay.is_participant(manifest, user.id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not a participant in this match")
    return manifest
