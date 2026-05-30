"""The single WebSocket endpoint.

Phase 1a: authenticate the socket via a first-message JWT (with a timeout), reply
with `auth.ok`, then echo subsequent validated messages. Lobby/matchmaking/match
dispatch is layered on in later phases; the auth handshake and receive-loop shape
established here are the foundation.
"""

from __future__ import annotations

import asyncio
import logging

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from app import state
from app.auth.security import decode_token
from app.connection_manager import manager
from app.db import SessionLocal
from app.messages import Auth, AuthOk, client_adapter, error
from app.models_db import User
from app.ws.handlers import (
    dispatch,
    handle_disconnect,
    register_or_reconnect,
    resume_after_reconnect,
)

logger = logging.getLogger("arenasl.ws")
router = APIRouter()

AUTH_TIMEOUT_SECONDS = 10


async def authenticate_ws(websocket: WebSocket) -> int | None:
    """Require a valid `auth` message as the first frame. Returns the user id, or
    closes the socket with 1008 and returns None on any failure."""
    try:
        raw = await asyncio.wait_for(
            websocket.receive_json(), timeout=AUTH_TIMEOUT_SECONDS
        )
        msg = client_adapter.validate_python(raw)
        if not isinstance(msg, Auth):
            raise ValueError("first message must be of type 'auth'")
        return decode_token(msg.token)
    except (
        asyncio.TimeoutError,
        ValidationError,
        ValueError,
        jwt.InvalidTokenError,
        WebSocketDisconnect,
    ):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None


def _load_user(user_id: int) -> User | None:
    with SessionLocal() as db:
        return db.get(User, user_id)


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    user_id = await authenticate_ws(websocket)
    if user_id is None:
        return

    user = _load_user(user_id)
    if user is None:  # valid token, but the account is gone
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(user.id, websocket)
    reconnected = register_or_reconnect(user.id, user.display_name, user.elo)
    live_elo = state.players[user.id].elo
    await manager.send(user.id, AuthOk(player_id=user.id, elo=live_elo))
    if reconnected:
        await resume_after_reconnect(user.id)

    try:
        while True:
            raw = await websocket.receive_json()
            try:
                msg = client_adapter.validate_python(raw)
            except ValidationError as exc:
                await manager.send(user.id, error("bad_message", str(exc)))
                continue
            try:
                await dispatch(user.id, msg)
            except Exception:  # a handler bug must not kill the socket
                logger.exception("handler failed for player %s", user.id)
                await manager.send(user.id, error("internal", "internal error"))
    except WebSocketDisconnect:
        pass
    finally:
        await handle_disconnect(user.id)
