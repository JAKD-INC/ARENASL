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
# Force (not setdefault) so tests stay isolated even when the container/host sets
# DB_URL/REPLAY_DIR to real volume paths.
os.environ["DB_URL"] = f"sqlite:///{_DB_PATH}"

_DATASET = pathlib.Path(__file__).resolve().parent.parent / "app" / "data" / "signs.json"
os.environ.setdefault("SIGNS_DATASET_PATH", str(_DATASET))

_REPLAY_DIR = tempfile.mkdtemp(prefix="arenasl-replays-")
os.environ["REPLAY_DIR"] = _REPLAY_DIR

import shutil  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import recognition, state  # noqa: E402
from app.connection_manager import manager  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models_db import Base  # noqa: E402
from app.words import init_dataset  # noqa: E402
from app.ws import handlers  # noqa: E402


def _clear_replays() -> None:
    root = pathlib.Path(_REPLAY_DIR)
    if root.exists():
        for child in root.iterdir():
            shutil.rmtree(child, ignore_errors=True)


@pytest.fixture
def client():
    """Fresh schema and live state per test, with the app lifespan running."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    state.reset()
    manager.reset()
    handlers.reset()
    _clear_replays()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def domain():
    """Reset live state, fresh DB schema, and load the dataset for tests that
    drive the domain / handlers directly (no HTTP/WS server, single event loop)."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    state.reset()
    manager.reset()
    recognition.reset()
    handlers.reset()
    _clear_replays()
    init_dataset(str(_DATASET))
    yield
    state.reset()
    manager.reset()
    recognition.reset()


class FakeManager:
    """Records sends instead of touching real sockets, so handler/broadcast logic
    can be driven on a single event loop (asyncio.run) the way production runs."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, dict]] = []
        self.connected: set[int] = set()

    def is_connected(self, pid: int) -> bool:
        return pid in self.connected

    async def connect(self, pid: int, ws) -> None:  # noqa: ANN001
        self.connected.add(pid)

    async def disconnect(self, pid: int) -> None:
        self.connected.discard(pid)

    async def send(self, pid: int, msg) -> None:  # noqa: ANN001
        self.sent.append((pid, msg.model_dump()))

    async def broadcast(self, pids, msg) -> None:  # noqa: ANN001
        for pid in pids:
            await self.send(pid, msg)

    def for_(self, pid: int) -> list[dict]:
        return [m for (p, m) in self.sent if p == pid]

    def types_for(self, pid: int) -> list[str]:
        return [m["type"] for m in self.for_(pid)]

    def last(self, pid: int, type_: str) -> dict:
        return [m for m in self.for_(pid) if m["type"] == type_][-1]


@pytest.fixture
def fake(domain, monkeypatch):
    """A FakeManager monkeypatched into the handlers module (replaces the real
    ConnectionManager singleton for the duration of a test)."""
    from app.ws import handlers

    fm = FakeManager()
    monkeypatch.setattr(handlers, "manager", fm)
    return fm


def register_players(fm: FakeManager, *players: tuple[int, str, int]) -> None:
    """Register (player_id, display_name, elo) tuples and mark them connected."""
    from app.ws import handlers

    for pid, name, elo in players:
        handlers.register_player(pid, name, elo)
        fm.connected.add(pid)


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
