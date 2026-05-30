"""FastAPI application entrypoint.

Run: `uvicorn app.main:app --reload` (dev).
"""

from __future__ import annotations

from fastapi import FastAPI

from app.auth.routes import router as auth_router
from app.lifespan import lifespan
from app.words import get_dataset
from app.ws.endpoint import router as ws_router

app = FastAPI(title="ARENASL", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(ws_router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/signs", tags=["meta"])
def signs() -> dict:
    """The sign dataset (word_strength source). Clients must hold this exact
    version to ready up for a match."""
    return get_dataset().to_payload()
