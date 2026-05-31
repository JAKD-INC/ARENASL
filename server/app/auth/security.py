"""Password hashing (argon2 via pwdlib) and JWT mint/verify (PyJWT).

`decode_token` raises `jwt.InvalidTokenError` (or a subclass, incl. expiry) on any
invalid token — callers catch that single type.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from app.config import get_settings

_password_hash = PasswordHash.recommended()  # argon2id with sane params

# Precomputed hash of a throwaway password. Verified against when an email is not
# found so login timing doesn't reveal whether an account exists.
DUMMY_HASH = _password_hash.hash("dummy-password-for-timing-defense")


def hash_password(plain: str) -> str:
    return _password_hash.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _password_hash.verify(plain, hashed)


def create_access_token(user_id: int) -> str:
    s = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=s.access_token_minutes)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> int:
    """Return the user id from a valid token. Raises InvalidTokenError otherwise."""
    s = get_settings()
    payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    sub = payload.get("sub")
    if sub is None:
        raise jwt.InvalidTokenError("token missing 'sub'")
    try:
        return int(sub)
    except (TypeError, ValueError) as exc:
        raise jwt.InvalidTokenError("token 'sub' is not an int") from exc
