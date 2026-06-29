import pandas as pd
from pytest import approx

from worldcup_predictor.future import (
    _apply_daily_scoreline_diversity,
    build_already_qualified_context_overrides,
    build_competitive_context,
    build_fixture_features,
    build_team_states,
    completed_matches,
    future_fixtures,
    predict_fixture_picks,
)


def _completed_group_stage() -> pd.DataFrame:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rows = []
    for group in range(12):
        for match_index, (home, away) in enumerate(pairs):
            rows.append(
                {
                    "date": pd.Timestamp("2026-06-11") + pd.Timedelta(days=match_index),
                    "home_team": f"T{group}_{home}",
                    "away_team": f"T{group}_{away}",
                    "home_score": 1,
                    "away_score": 0,
                    "tournament": "FIFA World Cup",
                }
            )
    return pd.DataFrame(rows)


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


def test_build_already_qualified_context_overrides_flags_last_group_match() -> None:
    completed = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-11", "2026-06-18", "2026-06-11", "2026-06-18"]),
            "home_team": ["A", "A", "C", "D"],
            "away_team": ["B", "C", "D", "B"],
            "home_score": [2, 1, 0, 0],
            "away_score": [0, 0, 0, 1],
            "tournament": ["FIFA World Cup"] * 4,
        }
    )
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-25", "2026-06-25"]),
            "home_team": ["A", "B"],
            "away_team": ["D", "C"],
            "home_score": [None, None],
            "away_score": [None, None],
            "tournament": ["FIFA World Cup"] * 2,
        }
    )

    overrides = build_already_qualified_context_overrides(completed, fixtures)

    assert len(overrides) == 1
    row = overrides.iloc[0]
    assert row.home_team == "A"
    assert row.away_team == "D"
    assert row.home_attack_multiplier == approx(0.88)
    assert row.home_defense_multiplier == approx(1.08)
    assert row.away_attack_multiplier == approx(1.0)
    assert row.draw_probability_multiplier == approx(1.08)
    assert "A (6 pts)" in row.notes


def test_build_already_qualified_context_overrides_avoids_unclinched_team() -> None:
    completed = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-11", "2026-06-11", "2026-06-18", "2026-06-18"]),
            "home_team": ["A", "C", "A", "B"],
            "away_team": ["B", "D", "C", "D"],
            "home_score": [1, 1, 2, 2],
            "away_score": [1, 0, 0, 0],
            "tournament": ["FIFA World Cup"] * 4,
        }
    )
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-25", "2026-06-25"]),
            "home_team": ["A", "B"],
            "away_team": ["D", "C"],
            "home_score": [None, None],
            "away_score": [None, None],
            "tournament": ["FIFA World Cup"] * 2,
        }
    )

    overrides = build_already_qualified_context_overrides(completed, fixtures, min_points=4)

    assert overrides.empty


def test_build_competitive_context_adds_group_need_columns() -> None:
    completed = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-11", "2026-06-11", "2026-06-18", "2026-06-18"]),
            "home_team": ["A", "C", "A", "B"],
            "away_team": ["B", "D", "C", "D"],
            "home_score": [2, 1, 2, 1],
            "away_score": [0, 0, 0, 0],
            "tournament": ["FIFA World Cup"] * 4,
        }
    )
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-25", "2026-06-25"]),
            "home_team": ["A", "B"],
            "away_team": ["D", "C"],
            "home_score": [None, None],
            "away_score": [None, None],
            "tournament": ["FIFA World Cup"] * 2,
        }
    )

    context = build_competitive_context(completed, fixtures)

    assert len(context) == 2
    assert "home_already_qualified" in context.columns
    assert "match_pressure_score" in context.columns
    assert context.loc[context["home_team"] == "A", "home_already_qualified"].iloc[0]


def test_predict_fixture_picks_uses_dynamic_goal_inflation_and_risk_profiles() -> None:
    fixture_features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-25", "2026-06-25"]),
            "home_team": ["Safe A", "Need A"],
            "away_team": ["Safe B", "Need B"],
            "tournament": ["FIFA World Cup", "FIFA World Cup"],
            "neutral": [True, True],
            "home_elo": [1800.0, 1800.0],
            "away_elo": [1800.0, 1800.0],
            "elo_diff": [0.0, 0.0],
        }
    )
    competitive_context = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-25", "2026-06-25"]),
            "home_team": ["Safe A", "Need A"],
            "away_team": ["Safe B", "Need B"],
            "group": ["Group 01", "Group 02"],
            "is_last_group_match": [True, True],
            "home_needs_win": [False, True],
            "away_needs_win": [False, True],
            "home_draw_is_enough": [False, False],
            "away_draw_is_enough": [False, False],
            "home_already_qualified": [True, False],
            "away_already_qualified": [True, False],
            "home_eliminated": [False, False],
            "away_eliminated": [False, False],
            "home_goal_difference_pressure": [False, True],
            "away_goal_difference_pressure": [False, True],
            "match_pressure_score": [0.0, 0.9],
            "group_scenario_volatility": [0.0, 0.8],
        }
    )

    predictions = predict_fixture_picks(
        ConstantGoalModel(home_goals=1.0, away_goals=1.0),
        fixture_features,
        goal_inflation=1.4,
        competitive_context=competitive_context,
        risk_profile="aggressive",
    )

    safe_row = predictions.loc[predictions["home_team"] == "Safe A"].iloc[0]
    need_row = predictions.loc[predictions["home_team"] == "Need A"].iloc[0]
    assert safe_row.dynamic_goal_inflation == approx(1.05)
    assert need_row.dynamic_goal_inflation == approx(1.45)
    assert safe_row.strategy_home_expected_goals == approx(1.05)
    assert need_row.strategy_home_expected_goals == approx(1.45)
    assert safe_row.recommended_scoreline_by_risk == safe_row.recommended_scoreline
    assert safe_row.strategic_pick_value > 0


def test_build_competitive_context_separates_knockout_from_group_logic() -> None:
    completed = _completed_group_stage()
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-29"]),
            "home_team": ["T0_0"],
            "away_team": ["T1_0"],
            "home_score": [None],
            "away_score": [None],
            "tournament": ["FIFA World Cup"],
        }
    )

    context = build_competitive_context(completed, fixtures)

    row = context.iloc[0]
    assert row.tournament_stage == "round_of_32"
    assert bool(row.is_knockout_stage) is True
    assert bool(row.is_last_group_match) is False
    assert bool(row.home_goal_difference_pressure) is False
    assert bool(row.away_goal_difference_pressure) is False
    assert row.group_scenario_volatility == 0.0
    assert bool(row.home_must_advance) is True
    assert bool(row.away_must_advance) is True


def test_predict_fixture_picks_caps_knockout_inflation_and_aggressive_goals() -> None:
    fixture_features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-29"]),
            "home_team": ["Netherlands"],
            "away_team": ["Morocco"],
            "tournament": ["FIFA World Cup"],
            "tournament_stage": ["round_of_32"],
            "is_group_stage": [False],
            "is_knockout_stage": [True],
            "is_extra_time_possible": [True],
            "is_penalty_possible": [True],
            "neutral": [True],
            "home_elo": [2016.0],
            "away_elo": [1978.0],
            "elo_diff": [38.0],
        }
    )
    competitive_context = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-29"]),
            "home_team": ["Netherlands"],
            "away_team": ["Morocco"],
            "tournament_stage": ["round_of_32"],
            "is_group_stage": [False],
            "is_knockout_stage": [True],
            "is_extra_time_possible": [True],
            "is_penalty_possible": [True],
        }
    )

    predictions = predict_fixture_picks(
        ConstantGoalModel(home_goals=1.42, away_goals=1.21),
        fixture_features,
        goal_inflation=1.4,
        competitive_context=competitive_context,
        risk_profile="aggressive",
    )

    row = predictions.iloc[0]
    assert row.tournament_stage == "round_of_32"
    assert row.dynamic_goal_inflation == approx(1.08)
    assert row.dynamic_goal_inflation <= 1.25
    assert row.aggressive_total_goal_cap == 4
    assert row.aggressive_scoreline not in {"3-2", "2-3"}
    assert row.recommended_home_score + row.recommended_away_score <= 4
    assert row.predicted_advancing_team in {"Netherlands", "Morocco"}
    assert row.extra_time_probability > 0


def test_daily_scoreline_diversity_penalizes_third_repeat() -> None:
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-29"] * 3),
            "recommended_scoreline": ["3-2", "3-2", "3-2"],
            "recommended_expected_points": [4.0, 3.8, 3.5],
            "strategic_pick_value": [7.0, 6.5, 6.0],
            "alert_flags": ["", "", ""],
            "model_warning_summary": ["", "", ""],
            "balanced_scoreline": ["2-1", "2-1", "2-1"],
            "balanced_expected_points": [3.9, 3.7, 3.4],
            "conservative_scoreline": ["2-0", "2-0", "2-0"],
            "conservative_expected_points": [3.5, 3.3, 3.1],
            "draw_alternative_scoreline": ["1-1", "1-1", "1-1"],
            "draw_alternative_expected_points": [2.0, 2.0, 2.0],
            "recommended_home_score": [3, 3, 3],
            "recommended_away_score": [2, 2, 2],
            "predicted_90min_scoreline": ["3-2", "3-2", "3-2"],
            "recommended_pick_explanation": ["", "", ""],
        }
    )

    diversified = _apply_daily_scoreline_diversity(rows)

    third = diversified.sort_values("recommended_expected_points", ascending=False).iloc[2]
    assert third.same_day_scoreline_count == 3
    assert third.scoreline_diversity_penalty > 0
    assert third.recommended_scoreline == "2-1"
    assert "daily_scoreline_diversity_penalty" in third.alert_flags
