import pandas as pd
from pytest import raises

from worldcup_predictor.overrides import apply_result_overrides


def test_apply_result_overrides_updates_fixture_score() -> None:
    raw_results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-11"]),
            "home_team": ["Mexico"],
            "away_team": ["South Africa"],
            "home_score": [None],
            "away_score": [None],
            "tournament": ["FIFA World Cup"],
            "city": ["Mexico City"],
            "country": ["Mexico"],
            "neutral": [False],
        }
    )
    overrides = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-11"]),
            "home_team": ["Mexico"],
            "away_team": ["South Africa"],
            "home_score": [2],
            "away_score": [1],
        }
    )

    updated = apply_result_overrides(raw_results, overrides)

    assert updated.iloc[0].home_score == 2
    assert updated.iloc[0].away_score == 1


def test_apply_result_overrides_raises_for_missing_match() -> None:
    raw_results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-11"]),
            "home_team": ["Mexico"],
            "away_team": ["South Africa"],
            "home_score": [None],
            "away_score": [None],
        }
    )
    overrides = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-12"]),
            "home_team": ["Argentina"],
            "away_team": ["Algeria"],
            "home_score": [1],
            "away_score": [0],
        }
    )

    with raises(ValueError, match="Override match not found"):
        apply_result_overrides(raw_results, overrides)
