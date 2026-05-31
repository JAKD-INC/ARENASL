"""Pydantic request/response schemas for auth. Output schemas never expose the
password hash."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field

from app.elo import Experience


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=50)
    experience: Experience  # seeds starting ELO


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    player_id: int
    display_name: str
    elo: int
