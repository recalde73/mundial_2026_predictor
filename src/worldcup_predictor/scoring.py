from __future__ import annotations

from collections.abc import Mapping, Sequence

Scoreline = tuple[int, int]
ScorelineProbabilities = Mapping[Scoreline, float]


EXACT_SCORE_POINTS = 10
WINNER_AND_DIFF_POINTS = 8
DRAW_POINTS = 6
WINNER_POINTS = 5


EXACT_SCORE = "exact_score"
WINNER_AND_DIFF = "winner_and_diff"
DRAW = "draw"
WINNER = "winner"
MISS = "miss"


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def points_for_prediction(prediction: Scoreline, actual: Scoreline) -> int:
    """Return game points for one predicted scoreline and one actual scoreline."""
    category = scoring_category(prediction, actual)

    if category == EXACT_SCORE:
        return EXACT_SCORE_POINTS
    if category == WINNER_AND_DIFF:
        return WINNER_AND_DIFF_POINTS
    if category == DRAW:
        return DRAW_POINTS
    if category == WINNER:
        return WINNER_POINTS
    return 0


def scoring_category(prediction: Scoreline, actual: Scoreline) -> str:
    """Return the scoring bucket for one prediction and actual scoreline."""
    pred_home, pred_away = prediction
    actual_home, actual_away = actual

    if prediction == actual:
        return EXACT_SCORE

    pred_diff = pred_home - pred_away
    actual_diff = actual_home - actual_away
    pred_result = _sign(pred_diff)
    actual_result = _sign(actual_diff)

    if pred_result == 0:
        return DRAW if actual_result == 0 else MISS

    if pred_result != actual_result:
        return MISS

    if pred_diff == actual_diff:
        return WINNER_AND_DIFF

    return WINNER


def expected_points(prediction: Scoreline, probabilities: ScorelineProbabilities) -> float:
    """Return expected points for a prediction over scoreline probabilities."""
    return sum(
        points_for_prediction(prediction, actual) * probability
        for actual, probability in probabilities.items()
    )


def best_prediction(
    probabilities: ScorelineProbabilities,
    candidates: Sequence[Scoreline] | None = None,
) -> tuple[Scoreline, float]:
    """Return the scoreline with highest expected points."""
    if candidates is None:
        candidates = tuple(probabilities.keys())

    if not candidates:
        raise ValueError("At least one candidate prediction is required")

    return max(
        ((candidate, expected_points(candidate, probabilities)) for candidate in candidates),
        key=lambda item: item[1],
    )


def candidate_scorelines(max_goals: int = 6) -> tuple[Scoreline, ...]:
    """Generate practical scoreline candidates from 0-0 up to max_goals-max_goals."""
    if max_goals < 0:
        raise ValueError("max_goals must be greater than or equal to 0")

    return tuple((home_goals, away_goals) for home_goals in range(max_goals + 1) for away_goals in range(max_goals + 1))
