"""ELO: experience→starting-rating seeding, and the rating-update math.

The starting rating is seeded from a player's self-reported ASL experience at
registration so first matches land near the right skill band. A higher
provisional K-factor lets a mis-stated level self-correct quickly.
"""

from __future__ import annotations

from enum import Enum

from app.config import get_settings


class Experience(str, Enum):
    new = "new"
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"
    fluent = "fluent"


STARTING_ELO: dict[Experience, int] = {
    Experience.new: 800,
    Experience.beginner: 1000,
    Experience.intermediate: 1200,
    Experience.advanced: 1400,
    Experience.fluent: 1600,
}


def starting_elo(experience: Experience) -> int:
    return STARTING_ELO[experience]


def k_factor(games_played: int) -> int:
    """Provisional (high) K for the first N games, then the steady-state K."""
    s = get_settings()
    return s.elo_k_provisional if games_played < s.elo_provisional_games else s.elo_k


def expected_score(rating: int, opponent_rating: int) -> float:
    """Standard logistic expectation that `rating` beats `opponent_rating`."""
    return 1.0 / (1.0 + 10 ** ((opponent_rating - rating) / 400.0))


def updated_rating(rating: int, opponent_rating: int, won: bool, games_played: int) -> int:
    """New integer rating after a result. `won` is True for a win, False for a loss."""
    k = k_factor(games_played)
    score = 1.0 if won else 0.0
    return round(rating + k * (score - expected_score(rating, opponent_rating)))
