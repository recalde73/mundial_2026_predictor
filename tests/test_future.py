import pandas as pd

from worldcup_predictor.future import (
    build_fixture_features,
    build_team_states,
    completed_matches,
    future_fixtures,
)


def test_completed_matches_filters_missing_scores() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_score": [1.0, None],
            "away_score": [0.0, None],
            "tournament": ["Friendly", "FIFA World Cup"],
            "city": ["X", "Y"],
            "country": ["Z", "Z"],
            "neutral": [True, True],
        }
    )

    completed = completed_matches(raw)

    assert len(completed) == 1
    assert completed.iloc[0].home_score == 1


def test_future_fixtures_filters_tournament_and_date() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01", "2026-06-12"]),
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_score": [None, None],
            "away_score": [None, None],
            "tournament": ["Friendly", "FIFA World Cup"],
            "city": ["X", "Y"],
            "country": ["Z", "Z"],
            "neutral": [True, True],
        }
    )

    fixtures = future_fixtures(raw, tournament="FIFA World Cup", start_date="2026-06-01")

    assert len(fixtures) == 1
    assert fixtures.iloc[0].home_team == "C"


def test_build_fixture_features_uses_current_team_state() -> None:
    completed = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_score": [1, 2],
            "away_score": [0, 0],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
        }
    )
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-12"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "tournament": ["FIFA World Cup"],
            "neutral": [True],
        }
    )

    states = build_team_states(completed)
    features = build_fixture_features(fixtures, states)

    assert len(features) == 1
    assert features.iloc[0].home_team == "A"
    assert features.iloc[0].home_elo > features.iloc[0].away_elo
    assert features.iloc[0].home_goals_for_avg == 1.5
