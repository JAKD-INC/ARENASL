"""Application lifespan: startup builds shared state, shutdown tears it down.

Phase 1a: create tables and load the sign dataset (fail fast if it's missing).
Later phases start the matchmaking background task here and cancel it on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import recognition, words
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

    # Best-effort: load ASL templates. If they're absent the server still boots
    # (recognition disabled); build them with `make templates`.
    try:
        glosses = recognition.init_matcher()
        # Restrict the word stream to glosses that actually have templates, so
        # every streamed word is recognizable.
        active = words.restrict_to(glosses)
        logger.info(
            "ASL recognition ready: %d glosses; word stream = %s",
            len(glosses),
            active.version,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "ASL templates unavailable at %s (%s); recognition disabled",
            settings.asl_templates_dir,
            exc,
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
