from pytest import approx

from worldcup_predictor.elo import actual_result, expected_result, update_ratings


def test_expected_result_is_balanced_for_equal_ratings() -> None:
    assert expected_result(1500, 1500) == approx(0.5)


def test_expected_result_increases_with_rating_advantage() -> None:
    assert expected_result(1600, 1500) > 0.5


def test_actual_result_maps_scoreline_to_elo_result() -> None:
    assert actual_result(2, 1) == 1.0
    assert actual_result(1, 1) == 0.5
    assert actual_result(0, 1) == 0.0


def test_update_ratings_rewards_winner() -> None:
    home_rating, away_rating = update_ratings(
        home_rating=1500,
        away_rating=1500,
        home_score=2,
        away_score=0,
        tournament="FIFA World Cup",
        neutral=True,
    )

    assert home_rating > 1500
    assert away_rating < 1500
