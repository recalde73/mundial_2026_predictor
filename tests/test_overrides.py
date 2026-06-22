import pandas as pd
from pytest import raises

from worldcup_predictor.overrides import apply_result_overrides, load_match_context_overrides


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


def test_load_match_context_overrides_fills_optional_defaults(tmp_path) -> None:
    path = tmp_path / "context.csv"
    path.write_text(
        "date,home_team,away_team,home_attack_multiplier,notes\n"
        "2026-06-22,France,Iraq,0.92,Striker questionable\n"
    )

    overrides = load_match_context_overrides(path)

    assert len(overrides) == 1
    assert overrides.iloc[0].home_attack_multiplier == 0.92
    assert overrides.iloc[0].away_attack_multiplier == 1.0
    assert overrides.iloc[0].draw_probability_multiplier == 1.0
    assert overrides.iloc[0].notes == "Striker questionable"


def test_load_match_context_overrides_rejects_non_positive_multiplier(tmp_path) -> None:
    path = tmp_path / "context.csv"
    path.write_text(
        "date,home_team,away_team,home_attack_multiplier\n"
        "2026-06-22,France,Iraq,0\n"
    )

    with raises(ValueError, match="greater than 0"):
        load_match_context_overrides(path)
