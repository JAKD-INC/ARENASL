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
from app.replay import retention_loop
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

    # Background tasks. Keep strong references; cancel + await on shutdown.
    tasks = [
        asyncio.create_task(matchmaking_loop()),
        asyncio.create_task(retention_loop()),
    ]
    app.state.background_tasks = tasks

    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
