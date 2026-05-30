"""WebSocket message envelope: Pydantic v2 discriminated unions on `type`.

Inbound messages are validated with a TypeAdapter (O(1) member selection, focused
errors, unknown types fail fast). Outbound messages are serialized with
`model_dump_json()`. Later phases add more members to each union; phase 1a needs
the auth handshake plus `auth.ok` / `error`.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

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


class SignAttempt(BaseModel):
    type: Literal["sign.attempt"]
    word_index: int
    accuracy: float


ClientMessage = Annotated[
    Union[Auth, QueueJoin, QueueLeave, LobbyCreate, LobbyJoin, LobbyReady, Signal, SignAttempt],
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
