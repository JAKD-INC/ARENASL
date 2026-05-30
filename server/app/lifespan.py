"""Application lifespan: startup builds shared state, shutdown tears it down.

Phase 1a: create tables and load the sign dataset (fail fast if it's missing).
Later phases start the matchmaking background task here and cancel it on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import engine
from app.matchmaking import matchmaking_loop
from app.models_db import Base
from app.words import init_dataset

logger = logging.getLogger("arenasl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    Base.metadata.create_all(engine)

    dataset = init_dataset(settings.signs_dataset_path)
    logger.info(
        "sign dataset loaded: version=%s entries=%d",
        dataset.version,
        len(dataset.entries),
    )

    # Background matchmaking ticker. Keep a strong reference; cancel + await on
    # shutdown so its cleanup runs.
    matchmaker = asyncio.create_task(matchmaking_loop())
    app.state.matchmaker = matchmaker

    try:
        yield
    finally:
        matchmaker.cancel()
        try:
            await matchmaker
        except asyncio.CancelledError:
            pass
