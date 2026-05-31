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

    # --- webrtc / TURN ---
    # STUN is free + auth-less. TURN is only emitted when turn_host is set; its
    # credentials are short-lived and minted per player (see app/turn.py).
    stun_urls: list[str] = ["stun:stun.l.google.com:19302"]
    turn_host: str = ""  # empty => STUN-only (no TURN entries)
    turn_udp_port: int = 3478
    turn_tls_port: int = 5349
    turn_ttl_seconds: int = 3600

    # --- match lifecycle ---
    reconnect_grace_seconds: int = 5
    match_ready_timeout_seconds: int = 60

    # --- ASL recognition (server-side DTW; thresholds from the tuned backend) ---
    asl_templates_dir: str = "./data/templates"
    asl_clips_dir: str = "./data/clips"  # reference example clips served at /clips
    asl_scale: float = 0.5          # DTW distance at which strength = 1/e
    asl_get_threshold: float = 0.6  # strength must peak above this to confirm
    asl_confirm_drop: float = 0.8   # confirm once strength falls to this × peak
    asl_miss_budget: float = 6.0    # seconds before a word times out (miss)
    asl_window_size: int = 48       # rolling landmark window (~1.5-2s)
    asl_overtake_frames: int = 2    # consecutive next-target wins to confirm
    recognition_fps_cap: int = 15   # server-side per-player frame budget (phase 2e)

    # --- ASL recognition (ADVANCED recognizer; OPT-IN, defaults reproduce today) ---
    asl_matcher_mode: str = "auto"         # auto: embedding if artifacts exist, else dtw
    asl_feature_mode: str = "full"         # "full" (default) or "hands"
    asl_encoder_path: str = "./data/encoder.onnx"
    asl_prototypes_path: str = "./data/prototypes.npz"
    asl_warmup_frames: int = 0             # advanced cascade-guard OFF by default
    asl_confirm_hold: int = 100000         # effectively DISABLED (today's behavior)
    asl_rank_every: int = 0                # 0/None => server-side ranking OFF
    asl_rank_gate: int = 0                 # 0 => OFF; embedding mode defaults it to 2
    asl_min_confirm_interval: float = 0.0  # 0 => OFF; embedding defaults to 2.0s debounce

    # --- replay ---
    replay_dir: str = "./replays"
    replay_retention_days: int = 14


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. Tests can clear the cache via `get_settings.cache_clear()`."""
    return Settings()  # type: ignore[call-arg]  # values come from env/.env
