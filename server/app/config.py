"""Application settings, loaded from environment / .env (pydantic-settings).

Secrets (JWT_SECRET, TURN_SECRET) are required and have no usable default — the
app fails fast at import if they're missing, rather than silently signing tokens
with a guessable key.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- secrets (required) ---
    jwt_secret: str
    turn_secret: str

    # --- auth ---
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60 * 24 * 7  # 7 days

    # --- persistence ---
    db_url: str = "sqlite:///./arena.db"

    # --- sign dataset (word_strength source) ---
    signs_dataset_path: str = "./app/data/signs.json"
    damage_scale: float = 1.0

    # --- elo (locked defaults) ---
    elo_k_provisional: int = 40
    elo_k: int = 24
    elo_provisional_games: int = 10

    # --- matchmaking ---
    mm_window_start: int = 50
    mm_window_widen_per_2s: int = 25

    # --- match lifecycle ---
    reconnect_grace_seconds: int = 5
    match_ready_timeout_seconds: int = 60

    # --- replay ---
    replay_dir: str = "./replays"
    replay_retention_days: int = 14


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. Tests can clear the cache via `get_settings.cache_clear()`."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
