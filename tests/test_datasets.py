import pandas as pd

from worldcup_predictor.datasets import build_match_features


def test_build_match_features_uses_only_prior_matches() -> None:
    results = pd.DataFrame(
        [
            ["2000-01-01", "A", "B", 1, 0, "Friendly", "X", "Y", True],
            ["2000-01-02", "A", "B", 3, 0, "Friendly", "X", "Y", True],
            ["2000-01-03", "A", "B", 0, 2, "Friendly", "X", "Y", True],
        ],
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ],
    )
    results["date"] = pd.to_datetime(results["date"])

    features = build_match_features(
        results,
        rolling_window=2,
        min_prior_matches=1,
        start_date=None,
    )

    assert len(features) == 2
    assert features.iloc[0].home_goals_for_avg == 1
    assert features.iloc[0].away_goals_for_avg == 0
    assert features.iloc[0].home_elo == 1507.5
    assert features.iloc[0].away_elo == 1492.5
    assert features.iloc[1].home_goals_for_avg == 2
    assert features.iloc[1].away_goals_for_avg == 0
    assert features.iloc[1].home_elo > features.iloc[1].away_elo


def test_build_match_features_can_filter_by_start_date() -> None:
    results = pd.DataFrame(
        [
            ["2000-01-01", "A", "B", 1, 0, "Friendly", "X", "Y", True],
            ["2000-01-02", "A", "B", 2, 0, "Friendly", "X", "Y", True],
            ["2000-01-03", "A", "B", 3, 0, "Friendly", "X", "Y", True],
        ],
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ],
    )
    results["date"] = pd.to_datetime(results["date"])

    features = build_match_features(
        results,
        min_prior_matches=1,
        start_date="2000-01-03",
    )

    assert len(features) == 1
    assert features.iloc[0].date == pd.Timestamp("2000-01-03")
