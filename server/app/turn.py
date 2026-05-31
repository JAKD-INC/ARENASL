"""TURN credentials and the iceServers payload delivered in `match.found`.

Uses coturn's REST-API / time-limited-credential scheme (`use-auth-secret`):
the app server and coturn share one static secret, and the server mints ephemeral
credentials per player. The username embeds an expiry so leaked creds die quickly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

from app.config import get_settings


def make_turn_credentials(
    secret: str, user_id: str, ttl: int, now: float | None = None
) -> tuple[str, str]:
    """Return (username, credential) matching coturn's HMAC-SHA1 scheme.

    username = "<unix-expiry>:<user_id>"; credential = base64(HMAC_SHA1(secret, username)).
    """
    seconds = time.time() if now is None else now
    expiry = int(seconds) + ttl
    username = f"{expiry}:{user_id}"
    digest = hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()
    credential = base64.b64encode(digest).decode()
    return username, credential


def build_ice_servers(user_id: int, now: float | None = None) -> list[dict]:
    """The iceServers list for one player: STUN always, TURN only if configured.
    TURN creds are minted fresh and scoped to this player's id."""
    s = get_settings()
    servers: list[dict] = [{"urls": url} for url in s.stun_urls]

    if s.turn_host:
        username, credential = make_turn_credentials(
            s.turn_secret, str(user_id), s.turn_ttl_seconds, now=now
        )
        servers.append(
            {
                "urls": [
                    f"turn:{s.turn_host}:{s.turn_udp_port}?transport=udp",
                    f"turn:{s.turn_host}:{s.turn_udp_port}?transport=tcp",
                    f"turns:{s.turn_host}:{s.turn_tls_port}?transport=tcp",
                ],
                "username": username,
                "credential": credential,
            }
        )
    return servers
