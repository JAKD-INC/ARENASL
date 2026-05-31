"""Durable SQLAlchemy 2.0 models. Only data that must survive a restart lives
here; all live game state is in-memory (see app/state.py, added in phase 1b)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)  # JWT `sub` = str(id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)  # stored lowercased
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(50))
    elo: Mapped[int] = mapped_column()  # seeded from experience at registration
    games_played: Mapped[int] = mapped_column(default=0)  # drives provisional K-factor
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MatchHistory(Base):
    __tablename__ = "match_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_a: Mapped[int] = mapped_column(ForeignKey("users.id"))
    player_b: Mapped[int] = mapped_column(ForeignKey("users.id"))
    winner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    elo_delta: Mapped[int]
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    end_reason: Mapped[str] = mapped_column(String(16), default="win")  # win | forfeit


class Replay(Base):
    __tablename__ = "replays"

    match_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    match_history_id: Mapped[int] = mapped_column(ForeignKey("match_history.id"))
    duration_ms: Mapped[int]
    manifest_path: Mapped[str] = mapped_column(String(255))  # replays/{match_id}/manifest.json
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
