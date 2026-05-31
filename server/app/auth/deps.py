"""Auth dependencies for REST routes: the OAuth2 bearer scheme and current-user
resolution. WebSocket auth is handled separately (it can't use this dependency —
see app/ws)."""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.auth.security import decode_token
from app.db import get_db
from app.models_db import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

CREDENTIALS_EXC = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    try:
        user_id = decode_token(token)
    except jwt.InvalidTokenError:
        raise CREDENTIALS_EXC
    user = db.get(User, user_id)
    if user is None:
        raise CREDENTIALS_EXC
    return user
