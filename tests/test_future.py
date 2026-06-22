import pandas as pd
from pytest import approx

from worldcup_predictor.future import (
    build_fixture_features,
    build_team_states,
    completed_matches,
    future_fixtures,
    predict_fixture_picks,
)


class ConstantPredictor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: pd.DataFrame) -> list[float]:
        return [self.value] * len(features)


class ConstantGoalModel:
    feature_columns = ("neutral",)

    def __init__(self, home_goals: float, away_goals: float) -> None:
        self.home_model = ConstantPredictor(home_goals)
        self.away_model = ConstantPredictor(away_goals)


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


def test_predict_fixture_picks_applies_context_overrides_to_expected_goals() -> None:
    fixture_features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-22"]),
            "home_team": ["France"],
            "away_team": ["Iraq"],
            "tournament": ["FIFA World Cup"],
            "neutral": [True],
            "home_elo": [2100.0],
            "away_elo": [1700.0],
            "elo_diff": [400.0],
        }
    )
    context_overrides = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-22"]),
            "home_team": ["France"],
            "away_team": ["Iraq"],
            "home_attack_multiplier": [0.8],
            "home_defense_multiplier": [1.1],
            "away_attack_multiplier": [1.0],
            "away_defense_multiplier": [1.2],
            "draw_probability_multiplier": [0.9],
            "confidence": ["high"],
            "notes": ["France without key forward"],
        }
    )

    predictions = predict_fixture_picks(
        ConstantGoalModel(home_goals=2.0, away_goals=1.0),
        fixture_features,
        goal_inflation=1.0,
        context_overrides=context_overrides,
    )

    row = predictions.iloc[0]
    assert row.model_home_expected_goals == 2.0
    assert row.model_away_expected_goals == 1.0
    assert row.context_home_expected_goals == approx(1.92)
    assert row.context_away_expected_goals == approx(1.1)
    assert row.home_expected_goals == approx(1.92)
    assert row.away_expected_goals == approx(1.1)
    assert row.draw_probability_multiplier == 0.9
    assert bool(row.context_applied) is True
    assert row.context_notes == "France without key forward"
