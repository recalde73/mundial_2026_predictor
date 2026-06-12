from worldcup_predictor.scoring import (
    best_prediction,
    candidate_scorelines,
    expected_points,
    points_for_prediction,
    scoring_category,
)


def test_exact_score_gets_10_points() -> None:
    assert points_for_prediction((2, 1), (2, 1)) == 10


def test_draw_prediction_gets_6_points_for_non_exact_draw() -> None:
    assert points_for_prediction((1, 1), (0, 0)) == 6


def test_winner_and_difference_gets_8_points() -> None:
    assert points_for_prediction((3, 1), (2, 0)) == 8


def test_winner_only_gets_5_points() -> None:
    assert points_for_prediction((1, 0), (3, 1)) == 5


def test_wrong_result_gets_0_points() -> None:
    assert points_for_prediction((1, 0), (0, 1)) == 0


def test_scoring_category_describes_points_bucket() -> None:
    assert scoring_category((2, 1), (2, 1)) == "exact_score"
    assert scoring_category((3, 1), (2, 0)) == "winner_and_diff"
    assert scoring_category((1, 1), (0, 0)) == "draw"
    assert scoring_category((1, 0), (3, 1)) == "winner"
    assert scoring_category((1, 0), (0, 1)) == "miss"


def test_expected_points_uses_scoreline_probabilities() -> None:
    probabilities = {
        (2, 1): 0.10,
        (1, 0): 0.20,
        (3, 1): 0.10,
        (1, 1): 0.60,
    }

    assert expected_points((2, 1), probabilities) == 10 * 0.10 + 8 * 0.20 + 5 * 0.10


def test_best_prediction_returns_highest_expected_points() -> None:
    probabilities = {
        (1, 1): 0.40,
        (1, 0): 0.25,
        (2, 0): 0.20,
        (0, 1): 0.15,
    }

    prediction, value = best_prediction(probabilities, candidates=((1, 1), (1, 0)))

    assert prediction == (1, 1)
    assert value == expected_points((1, 1), probabilities)


def test_candidate_scorelines_generates_grid() -> None:
    assert candidate_scorelines(max_goals=1) == ((0, 0), (0, 1), (1, 0), (1, 1))
