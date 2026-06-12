from __future__ import annotations

from pathlib import Path

from worldcup_predictor.ratings import (
    estimate_expected_goals,
    get_team_rating,
    load_team_ratings,
)
from worldcup_predictor.recommender import PickRecommendation, recommend_pick


def recommend_match_from_ratings(
    home_team: str,
    away_team: str,
    ratings_path: str | Path,
    base_goals: float = 1.35,
    rating_weight: float = 0.28,
    home_advantage: float = 0.0,
) -> PickRecommendation:
    """Load ratings, estimate expected goals, and recommend a scoreline."""
    ratings = load_team_ratings(ratings_path)
    home = get_team_rating(ratings, home_team)
    away = get_team_rating(ratings, away_team)
    estimate = estimate_expected_goals(
        home=home,
        away=away,
        base_goals=base_goals,
        rating_weight=rating_weight,
        home_advantage=home_advantage,
    )

    return recommend_pick(
        home_team=estimate.home_team,
        away_team=estimate.away_team,
        home_expected_goals=estimate.home_expected_goals,
        away_expected_goals=estimate.away_expected_goals,
    )
