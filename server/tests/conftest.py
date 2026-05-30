"""Test fixtures. Environment is configured BEFORE importing the app so the
module-level settings/engine bind to a throwaway DB and test secrets."""

from __future__ import annotations

import os
import pathlib
import tempfile

# Must be set before any `app.*` import (app.config/app.db read these at import).
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-not-for-production")
os.environ.setdefault("TURN_SECRET", "test-turn-secret-not-for-production")

_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.environ.setdefault("DB_URL", f"sqlite:///{_DB_PATH}")

_DATASET = pathlib.Path(__file__).resolve().parent.parent / "app" / "data" / "signs.json"
os.environ.setdefault("SIGNS_DATASET_PATH", str(_DATASET))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import state  # noqa: E402
from app.connection_manager import manager  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models_db import Base  # noqa: E402
from app.words import init_dataset  # noqa: E402


@pytest.fixture
def client():
    """Fresh schema and live state per test, with the app lifespan running."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    state.reset()
    manager.reset()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def domain():
    """Reset live state and load the dataset for tests that drive the domain /
    handlers directly (no HTTP/WS server, single event loop)."""
    state.reset()
    manager.reset()
    init_dataset(str(_DATASET))
    yield
    state.reset()
    manager.reset()


def register(client: TestClient, *, email: str, password: str = "supersecret123",
             display_name: str = "Player", experience: str = "intermediate") -> str:
    """Register a user and return their access token."""
    resp = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
            "display_name": display_name,
            "experience": experience,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["access_token"]
