"""Application lifespan: startup builds shared state, shutdown tears it down.

Phase 1a: create tables and load the sign dataset (fail fast if it's missing).
Later phases start the matchmaking background task here and cancel it on shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import engine
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

    yield

    # (shutdown) later phases cancel background tasks / close sockets here.
