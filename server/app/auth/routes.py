"""Auth REST routes: register, token (login), me.

DB-touching handlers are `def` (not `async def`) so FastAPI runs them in its
threadpool and the sync SQLite calls don't block the event loop.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.auth.schemas import RegisterIn, TokenOut, UserOut
from app.auth.security import (
    DUMMY_HASH,
    create_access_token,
    hash_password,
    verify_password,
)
from app.db import get_db
from app.elo import starting_elo
from app.models_db import User

router = APIRouter(prefix="/auth", tags=["auth"])


def _normalize_email(email: str) -> str:
    return email.strip().lower()


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterIn, db: Annotated[Session, Depends(get_db)]) -> TokenOut:
    email = _normalize_email(payload.email)
    if db.scalar(select(User).where(User.email == email)) is not None:
        # Generic message; avoid leaking which emails are registered.
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
        elo=starting_elo(payload.experience),
    )
    db.add(user)
    db.flush()  # assigns user.id within this transaction (get_db commits on exit)
    return TokenOut(access_token=create_access_token(user.id))


@router.post("/token", response_model=TokenOut)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> TokenOut:
    # OAuth2 form uses `username`; we treat it as the email.
    email = _normalize_email(form.username)
    user = db.scalar(select(User).where(User.email == email))

    if user is None:
        verify_password(form.password, DUMMY_HASH)  # constant-time-ish; no user enumeration
        raise _bad_login()
    if not verify_password(form.password, user.password_hash):
        raise _bad_login()

    return TokenOut(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserOut)
def me(user: Annotated[User, Depends(get_current_user)]) -> UserOut:
    return UserOut(player_id=user.id, display_name=user.display_name, elo=user.elo)


def _bad_login() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
