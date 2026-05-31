from __future__ import annotations

from app import elo


def test_expected_score_is_half_for_equal_ratings():
    assert elo.expected_score(1200, 1200) == 0.5


def test_expected_score_favors_higher_rating():
    assert elo.expected_score(1600, 1200) > 0.5
    assert elo.expected_score(1200, 1600) < 0.5


def test_provisional_k_factor():
    assert elo.k_factor(0) == 40       # first games: high K
    assert elo.k_factor(9) == 40
    assert elo.k_factor(10) == 24      # steady state


def test_updated_rating_winner_gains_loser_loses():
    assert elo.updated_rating(1200, 1200, won=True, games_played=0) > 1200
    assert elo.updated_rating(1200, 1200, won=False, games_played=0) < 1200


def test_starting_elo_mapping():
    assert elo.starting_elo(elo.Experience.new) == 800
    assert elo.starting_elo(elo.Experience.fluent) == 1600
