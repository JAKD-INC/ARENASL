"""In-memory live game state (single process). Nothing here survives a restart;
durable data lives in SQLite (app/models_db.py).

Phase 1b uses `players` and `lobbies`. `queue` (1c) and `matches` (1c/1e) are
declared here so later phases share the same module-level stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Player:
    id: int
    display_name: str
    elo: int
    status: str = "idle"  # idle | in_lobby | queued | in_match
    lobby_code: str | None = None


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
class Match:
    """Stub created when a lobby fills; the authoritative duel state (HP, word
    stream, event log) is layered on in phase 1e."""

    id: str
    player_ids: tuple[int, int]
    roles: dict[int, str]  # pid -> "offerer" | "answerer"
    lobby_code: str
    state: str = "connecting"  # connecting | active | finished


# --- module-level stores ----------------------------------------------------

players: dict[int, Player] = {}
lobbies: dict[str, Lobby] = {}
queue: list = []  # phase 1c
matches: dict[str, Match] = {}


def reset() -> None:
    """Test helper: clear all live state."""
    players.clear()
    lobbies.clear()
    queue.clear()
    matches.clear()
