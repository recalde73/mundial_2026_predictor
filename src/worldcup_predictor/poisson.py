from __future__ import annotations

from math import exp, factorial

from worldcup_predictor.scoring import Scoreline, ScorelineProbabilities


def poisson_probability(goals: int, expected_goals: float) -> float:
    """Return Poisson probability for a goal count and expected goals."""
    if goals < 0:
        raise ValueError("goals must be greater than or equal to 0")
    if expected_goals < 0:
        raise ValueError("expected_goals must be greater than or equal to 0")

    return exp(-expected_goals) * expected_goals**goals / factorial(goals)


def scoreline_probabilities(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 6,
    normalize: bool = True,
) -> dict[Scoreline, float]:
    """Return scoreline probabilities from independent Poisson goal models."""
    if max_goals < 0:
        raise ValueError("max_goals must be greater than or equal to 0")

    probabilities = {
        (home_goals, away_goals): poisson_probability(home_goals, home_expected_goals)
        * poisson_probability(away_goals, away_expected_goals)
        for home_goals in range(max_goals + 1)
        for away_goals in range(max_goals + 1)
    }

    if normalize:
        total_probability = sum(probabilities.values())
        if total_probability == 0:
            raise ValueError("total probability cannot be zero")
        probabilities = {
            scoreline: probability / total_probability
            for scoreline, probability in probabilities.items()
        }

    return probabilities


def result_probabilities(probabilities: ScorelineProbabilities) -> dict[str, float]:
    """Return home/draw/away probabilities from scoreline probabilities."""
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for (home_goals, away_goals), probability in probabilities.items():
        if home_goals > away_goals:
            home_win += probability
        elif home_goals == away_goals:
            draw += probability
        else:
            away_win += probability

    return {"home_win": home_win, "draw": draw, "away_win": away_win}


def most_likely_scorelines(
    probabilities: ScorelineProbabilities,
    limit: int = 5,
) -> list[tuple[Scoreline, float]]:
    """Return the most likely scorelines sorted by probability."""
    if limit < 1:
        raise ValueError("limit must be greater than or equal to 1")

    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:limit]
