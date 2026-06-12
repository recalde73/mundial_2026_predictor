from __future__ import annotations

from dataclasses import dataclass
from math import log


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    friendly_k: float = 15.0
    competitive_k: float = 25.0
    home_advantage: float = 60.0


def expected_result(
    team_rating: float,
    opponent_rating: float,
    team_advantage: float = 0.0,
) -> float:
    """Return expected result probability from Elo ratings."""
    adjusted_rating = team_rating + team_advantage
    return 1 / (1 + 10 ** ((opponent_rating - adjusted_rating) / 400))


def actual_result(goals_for: int, goals_against: int) -> float:
    """Return actual Elo result: win=1, draw=0.5, loss=0."""
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def match_k_factor(tournament: str, config: EloConfig = EloConfig()) -> float:
    """Return a lower K for friendlies and higher K for competitive matches."""
    return config.friendly_k if tournament == "Friendly" else config.competitive_k


def goal_difference_multiplier(home_score: int, away_score: int) -> float:
    """Return a mild multiplier for larger wins."""
    goal_difference = abs(home_score - away_score)
    if goal_difference <= 1:
        return 1.0
    return 1.0 + log(goal_difference)


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool,
    config: EloConfig = EloConfig(),
) -> tuple[float, float]:
    """Return updated Elo ratings after one match."""
    home_advantage = 0.0 if neutral else config.home_advantage
    home_expected = expected_result(home_rating, away_rating, home_advantage)
    home_actual = actual_result(home_score, away_score)
    rating_change = (
        match_k_factor(tournament, config)
        * goal_difference_multiplier(home_score, away_score)
        * (home_actual - home_expected)
    )

    return home_rating + rating_change, away_rating - rating_change
