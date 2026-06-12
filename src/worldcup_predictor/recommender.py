from __future__ import annotations

from dataclasses import dataclass

from worldcup_predictor.poisson import (
    most_likely_scorelines,
    result_probabilities,
    scoreline_probabilities,
)
from worldcup_predictor.scoring import (
    Scoreline,
    ScorelineProbabilities,
    candidate_scorelines,
    expected_points,
)


DEFAULT_MAX_TOTAL_CANDIDATE_GOALS = 2
DEFAULT_DRAW_PROBABILITY_MULTIPLIER = 1.23


@dataclass(frozen=True)
class PickRecommendation:
    home_team: str
    away_team: str
    home_expected_goals: float
    away_expected_goals: float
    recommended_scoreline: Scoreline
    expected_points: float
    result_probabilities: dict[str, float]
    most_likely_scorelines: list[tuple[Scoreline, float]]
    alternatives: list[tuple[Scoreline, float]]


def rank_predictions(
    probabilities: ScorelineProbabilities,
    max_candidate_goals: int = 6,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    limit: int = 10,
) -> list[tuple[Scoreline, float]]:
    """Rank candidate scorelines by expected game points."""
    if limit < 1:
        raise ValueError("limit must be greater than or equal to 1")
    if draw_probability_multiplier <= 0:
        raise ValueError("draw_probability_multiplier must be greater than 0")

    candidates = candidate_scorelines(max_candidate_goals)
    if max_total_candidate_goals is not None:
        candidates = tuple(
            scoreline
            for scoreline in candidates
            if sum(scoreline) <= max_total_candidate_goals
        )

    adjusted_probabilities = adjust_draw_probabilities(
        probabilities,
        draw_probability_multiplier=draw_probability_multiplier,
    )
    ranked = [
        (candidate, expected_points(candidate, adjusted_probabilities))
        for candidate in candidates
    ]
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]


def adjust_draw_probabilities(
    probabilities: ScorelineProbabilities,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
) -> dict[Scoreline, float]:
    """Return probabilities with a calibrated multiplier on draw scorelines."""
    if draw_probability_multiplier <= 0:
        raise ValueError("draw_probability_multiplier must be greater than 0")

    adjusted = {
        scoreline: probability * draw_probability_multiplier
        if scoreline[0] == scoreline[1]
        else probability
        for scoreline, probability in probabilities.items()
    }
    total_probability = sum(adjusted.values())
    if total_probability == 0:
        raise ValueError("total probability cannot be zero")

    return {
        scoreline: probability / total_probability
        for scoreline, probability in adjusted.items()
    }


def recommend_pick(
    home_team: str,
    away_team: str,
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 6,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    alternatives_limit: int = 10,
) -> PickRecommendation:
    """Recommend a scoreline by maximizing expected points under game rules."""
    probabilities = scoreline_probabilities(
        home_expected_goals=home_expected_goals,
        away_expected_goals=away_expected_goals,
        max_goals=max_goals,
    )
    alternatives = rank_predictions(
        probabilities=probabilities,
        max_candidate_goals=max_goals,
        max_total_candidate_goals=max_total_candidate_goals,
        draw_probability_multiplier=draw_probability_multiplier,
        limit=alternatives_limit,
    )
    recommended_scoreline, recommendation_value = alternatives[0]

    return PickRecommendation(
        home_team=home_team,
        away_team=away_team,
        home_expected_goals=home_expected_goals,
        away_expected_goals=away_expected_goals,
        recommended_scoreline=recommended_scoreline,
        expected_points=recommendation_value,
        result_probabilities=result_probabilities(probabilities),
        most_likely_scorelines=most_likely_scorelines(probabilities),
        alternatives=alternatives,
    )
