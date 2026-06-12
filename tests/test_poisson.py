from pytest import approx

from worldcup_predictor.poisson import (
    most_likely_scorelines,
    poisson_probability,
    result_probabilities,
    scoreline_probabilities,
)


def test_poisson_probability_for_zero_goals() -> None:
    assert poisson_probability(0, 1.5) == approx(0.2231301601)


def test_scoreline_probabilities_are_normalized() -> None:
    probabilities = scoreline_probabilities(1.4, 0.9, max_goals=6)

    assert sum(probabilities.values()) == approx(1.0)


def test_result_probabilities_sum_to_one() -> None:
    probabilities = scoreline_probabilities(1.4, 0.9, max_goals=6)
    results = result_probabilities(probabilities)

    assert sum(results.values()) == approx(1.0)
    assert results["home_win"] > results["away_win"]


def test_most_likely_scorelines_returns_sorted_limit() -> None:
    probabilities = scoreline_probabilities(1.0, 1.0, max_goals=3)
    scorelines = most_likely_scorelines(probabilities, limit=3)

    assert len(scorelines) == 3
    assert scorelines[0][1] >= scorelines[1][1] >= scorelines[2][1]
