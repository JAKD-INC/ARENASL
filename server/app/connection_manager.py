"""Maps player_id -> live WebSocket and serializes writes per socket.

A single Starlette WebSocket must not be written by two coroutines at once (the
receive loop and, later, the matchmaking/broadcast tasks). Each socket gets its
own asyncio.Lock; broadcast serializes the payload once and fans out with
gather(return_exceptions=True) so one dead socket can't break the others.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger("arenasl.conn")


class ConnectionManager:
    def __init__(self) -> None:
        self._conns: dict[int, WebSocket] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def is_connected(self, pid: int) -> bool:
        return pid in self._conns

    async def connect(self, pid: int, ws: WebSocket) -> None:
        """Register an already-accepted socket. If the player has a stale socket
        (reconnect / double login), close the old one first."""
        old = self._conns.get(pid)
        if old is not None:
            try:
                await old.close()
            except RuntimeError:
                pass
        self._conns[pid] = ws
        self._locks[pid] = asyncio.Lock()

    async def disconnect(self, pid: int) -> None:
        self._conns.pop(pid, None)
        self._locks.pop(pid, None)

    async def send(self, pid: int, msg: BaseModel) -> None:
        ws = self._conns.get(pid)
        lock = self._locks.get(pid)
        if ws is None or lock is None:
            return
        payload = msg.model_dump_json()
        async with lock:
            try:
                await ws.send_text(payload)
            except (WebSocketDisconnect, RuntimeError):
                await self.disconnect(pid)

    async def broadcast(self, pids: Iterable[int], msg: BaseModel) -> None:
        await asyncio.gather(
            *(self.send(pid, msg) for pid in pids), return_exceptions=True
        )

    def reset(self) -> None:
        """Test helper: drop all connections without touching live sockets."""
        self._conns.clear()
        self._locks.clear()


# Single-process singleton.
manager = ConnectionManager()
