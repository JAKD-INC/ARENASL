"""In-memory live game state (single process). Nothing here survives a restart;
durable data lives in SQLite (app/models_db.py).

Phase 1b uses `players` and `lobbies`. `queue` (1c) and `matches` (1c/1e) are
declared here so later phases share the same module-level stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Player:
    id: int
    display_name: str
    elo: int
    status: str = "idle"  # idle | in_lobby | queued | in_match
    lobby_code: str | None = None
    match_id: str | None = None  # set when the lobby fills; used to route signaling


@dataclass
class LobbyMember:
    player_id: int
    ready: bool = False


@dataclass
class Lobby:
    code: str
    origin: str  # "private" | "queue"
    members: list[LobbyMember] = field(default_factory=list)
    match_id: str | None = None  # set once the lobby fills (2 players)

    @property
    def is_full(self) -> bool:
        return len(self.members) >= 2

    def member(self, pid: int) -> LobbyMember | None:
        return next((m for m in self.members if m.player_id == pid), None)


@dataclass
class QueueEntry:
    player_id: int
    elo: int
    joined_at: float  # monotonic clock (loop.time()); used to widen the search window


@dataclass
class Match:
    """Created when a lobby fills (connecting), activated when both players ready
    (the authoritative duel), then finished."""

    id: str
    player_ids: tuple[int, int]
    roles: dict[int, str]  # pid -> "offerer" | "answerer"
    lobby_code: str
    state: str = "connecting"  # connecting | active | finished

    # --- authoritative duel state (set at start_match, phase 1e) ---
    word_seed: int | None = None
    hp: dict[int, float] = field(default_factory=dict)          # pid -> remaining HP
    word_index: dict[int, int] = field(default_factory=dict)    # pid -> next word index
    started_at: float | None = None                             # monotonic origin (event-log offsets)
    started_wall: datetime | None = None                        # wall clock (history row)
    winner_id: int | None = None
    event_log: list[dict] = field(default_factory=list)         # timestamped, replay backbone


# --- module-level stores ----------------------------------------------------

players: dict[int, Player] = {}
lobbies: dict[str, Lobby] = {}
queue: list[QueueEntry] = []
matches: dict[str, Match] = {}


def reset() -> None:
    """Test helper: clear all live state."""
    players.clear()
    lobbies.clear()
    queue.clear()
    matches.clear()
