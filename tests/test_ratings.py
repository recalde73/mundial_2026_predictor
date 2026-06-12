from pytest import approx, raises

from worldcup_predictor.ratings import (
    TeamRating,
    estimate_expected_goals,
    get_team_rating,
    load_team_ratings,
)


def test_estimate_expected_goals_favors_stronger_team() -> None:
    home = TeamRating(team="A", rating=1850, attack=0.15, defense=0.12)
    away = TeamRating(team="B", rating=1700, attack=0.02, defense=0.04)

    estimate = estimate_expected_goals(home, away)

    assert estimate.home_expected_goals > estimate.away_expected_goals


def test_estimate_expected_goals_is_symmetric_for_equal_teams() -> None:
    home = TeamRating(team="A", rating=1800, attack=0.10, defense=0.10)
    away = TeamRating(team="B", rating=1800, attack=0.10, defense=0.10)

    estimate = estimate_expected_goals(home, away, home_advantage=0.0)

    assert estimate.home_expected_goals == approx(estimate.away_expected_goals)


def test_load_team_ratings_from_csv() -> None:
    ratings = load_team_ratings("data/raw/team_ratings_example.csv")

    assert ratings["Argentina"].rating == 1880
    assert ratings["Dinamarca"].defense == 0.10


def test_get_team_rating_raises_clear_error_for_missing_team() -> None:
    ratings = {"Argentina": TeamRating(team="Argentina", rating=1880)}

    with raises(KeyError, match="Team not found"):
        get_team_rating(ratings, "Atlantis")
