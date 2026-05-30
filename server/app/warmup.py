"""Warmup while queued: a solo, non-authoritative practice stream.

When a player joins the matchmaking queue the server hands them a random word
seed; the client runs its recognizer against the resulting practice stream locally
to stay warm. The server does not score warmup or track attempts — it only mints
the seed. Warmup ends implicitly when `match.found` arrives or on `queue.leave`.
"""

from __future__ import annotations

import secrets


def new_seed() -> int:
    """A fresh, unpredictable seed for a solo practice stream."""
    return secrets.randbits(31)
