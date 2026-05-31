"""WebSocket message envelope: Pydantic v2 discriminated unions on `type`.

Inbound messages are validated with a TypeAdapter (O(1) member selection, focused
errors, unknown types fail fast). Outbound messages are serialized with
`model_dump_json()`. Later phases add more members to each union; phase 1a needs
the auth handshake plus `auth.ok` / `error`.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# --- client -> server -------------------------------------------------------


class Auth(BaseModel):
    type: Literal["auth"]
    token: str


class QueueJoin(BaseModel):
    type: Literal["queue.join"]


class QueueLeave(BaseModel):
    type: Literal["queue.leave"]


class LobbyCreate(BaseModel):
    type: Literal["lobby.create"]
    private: bool = True


class LobbyJoin(BaseModel):
    type: Literal["lobby.join"]
    code: str


class LobbyReady(BaseModel):
    type: Literal["lobby.ready"]
    ready: bool  # ready => set up (dataset/recognizer/camera/peer) and want to start


class Signal(BaseModel):
    type: Literal["signal"]
    data: dict  # opaque WebRTC sdp/ice/bye; the server never parses this


class Landmark(BaseModel):
    """One frame of MediaPipe landmarks streamed during an active match. Kept
    loosely typed (lists, not per-point models) — this arrives many times/second
    and assemble_frame() validates the shapes. Browser sends handLeft/handRight."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["landmark"]
    t: float
    pose: list | None = None
    hand_left: list | None = Field(default=None, alias="handLeft")
    hand_right: list | None = Field(default=None, alias="handRight")


ClientMessage = Annotated[
    Union[Auth, QueueJoin, QueueLeave, LobbyCreate, LobbyJoin, LobbyReady, Signal, Landmark],
    Field(discriminator="type"),
]
client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


# --- server -> client -------------------------------------------------------


class AuthOk(BaseModel):
    type: Literal["auth.ok"] = "auth.ok"
    player_id: int
    elo: int


class Error(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


def error(code: str, message: str) -> Error:
    return Error(code=code, message=message)


class WarmupStart(BaseModel):
    type: Literal["warmup.start"] = "warmup.start"
    word_seed: int
    dataset_version: str


class QueueStatus(BaseModel):
    type: Literal["queue.status"] = "queue.status"
    position: int


class LobbyMemberView(BaseModel):
    player_id: int
    display_name: str
    ready: bool
    connected: bool


class LobbyUpdate(BaseModel):
    type: Literal["lobby.update"] = "lobby.update"
    code: str
    state: Literal["waiting", "full"]
    members: list[LobbyMemberView]


class OpponentView(BaseModel):
    player_id: int
    display_name: str
    elo: int


class MatchFound(BaseModel):
    """Sent when a lobby fills: WebRTC connect params. `ice_servers` is populated
    in phase 1d; the word seed arrives later in `match.start` (phase 1e)."""

    type: Literal["match.found"] = "match.found"
    match_id: str
    role: Literal["offerer", "answerer"]
    dataset_version: str
    opponent: OpponentView
    ice_servers: list[dict] = []


class MatchStart(BaseModel):
    """Both ready -> the duel begins. Carries the duel's deterministic params."""

    type: Literal["match.start"] = "match.start"
    match_id: str
    word_seed: int
    record_start_ms: int = 0  # shared clock origin for replay


class PlayerState(BaseModel):
    player_id: int
    hp: float
    word_index: int


class MatchState(BaseModel):
    type: Literal["match.state"] = "match.state"
    match_id: str
    players: list[PlayerState]


class RecognitionUpdate(BaseModel):
    """Per-frame feedback to the signing player (drives the strength UI). Sent
    while a sign is in progress (no get/miss yet)."""

    type: Literal["recognition.update"] = "recognition.update"
    word_index: int
    word: str
    strength: float


class MatchOver(BaseModel):
    type: Literal["match.over"] = "match.over"
    match_id: str
    winner_id: int | None
    reason: Literal["win", "forfeit"] = "win"
    elo: int | None = None        # recipient's new rating
    elo_delta: int | None = None  # recipient's change this match


class OpponentStatus(BaseModel):
    """Sent to the still-connected player when the opponent drops or returns."""

    type: Literal["opponent.status"] = "opponent.status"
    player_id: int
    connected: bool
