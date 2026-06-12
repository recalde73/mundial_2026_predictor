from worldcup_predictor.pipeline import recommend_match_from_ratings


def test_recommend_match_from_ratings_returns_recommendation() -> None:
    recommendation = recommend_match_from_ratings(
        home_team="Argentina",
        away_team="Dinamarca",
        ratings_path="data/raw/team_ratings_example.csv",
    )

    assert recommendation.home_team == "Argentina"
    assert recommendation.away_team == "Dinamarca"
    assert recommendation.home_expected_goals > recommendation.away_expected_goals
    assert recommendation.expected_points > 0
