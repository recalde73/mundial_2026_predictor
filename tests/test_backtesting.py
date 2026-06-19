import pandas as pd

from worldcup_predictor.backtesting import (
    backtest_model_on_frozen_period,
    backtest_model_on_period,
    evaluate_pick_strategies,
    summarize_picks,
)
from worldcup_predictor.datasets import build_match_features


def test_evaluate_pick_strategies_scores_optimized_pick() -> None:
    predictions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "home_expected_goals": 1.5,
                "away_expected_goals": 0.7,
            }
        ]
    )

    picks = evaluate_pick_strategies(predictions)

    assert len(picks) == 1
    assert picks.iloc[0].goal_inflation == 1.40
    assert picks.iloc[0].max_total_candidate_goals == 4
    assert picks.iloc[0].strategy_home_expected_goals > picks.iloc[0].home_expected_goals
    assert picks.iloc[0].optimized_points >= 0
    assert picks.iloc[0].most_likely_points >= 0


def test_summarize_picks_counts_categories() -> None:
    picks = pd.DataFrame(
        [
            {"optimized_points": 10, "optimized_category": "exact_score"},
            {"optimized_points": 5, "optimized_category": "winner"},
            {"optimized_points": 0, "optimized_category": "miss"},
        ]
    )

    summary = summarize_picks(picks, "optimized_points", "optimized_category")

    assert summary.matches == 3
    assert summary.total_points == 15
    assert summary.average_points == 5
    assert summary.exact_scores == 1
    assert summary.winners == 1
    assert summary.misses == 1
    assert summary.exact_score_rate == 1 / 3
    assert summary.result_accuracy_rate == 2 / 3
    assert summary.miss_rate == 1 / 3


def test_backtest_model_on_period_uses_fixed_validation_window() -> None:
    rows = []
    dates = pd.date_range("2023-01-01", periods=80, freq="30D")
    for index, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "home_team": "A" if index % 2 == 0 else "B",
                "away_team": "B" if index % 2 == 0 else "A",
                "home_score": 2 if index % 3 == 0 else 1,
                "away_score": 1 if index % 4 == 0 else 0,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": True,
            }
        )
    results = pd.DataFrame(rows)
    features = build_match_features(results, min_prior_matches=2, start_date=None)

    result = backtest_model_on_period(
        features,
        train_end_date="2024-12-31",
        validation_start_date="2025-01-01",
        validation_end_date="2025-12-31",
    )

    assert result.model.train_rows > 0
    assert result.model.validation_rows == len(result.picks)
    assert result.picks["date"].min() >= pd.Timestamp("2025-01-01")
    assert result.picks["date"].max() <= pd.Timestamp("2025-12-31")


def test_backtest_model_on_frozen_period_uses_train_end_team_state() -> None:
    rows = []
    dates = pd.date_range("2023-01-01", periods=80, freq="30D")
    for index, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "home_team": "A" if index % 2 == 0 else "B",
                "away_team": "B" if index % 2 == 0 else "A",
                "home_score": 2 if index % 3 == 0 else 1,
                "away_score": 1 if index % 4 == 0 else 0,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": True,
            }
        )
    results = pd.DataFrame(rows)
    features = build_match_features(results, min_prior_matches=2, start_date=None)

    result = backtest_model_on_frozen_period(
        features,
        results,
        train_end_date="2024-12-31",
        validation_start_date="2025-01-01",
        validation_end_date="2025-12-31",
    )

    assert result.model.train_rows > 0
    assert result.model.validation_rows == len(result.picks)
    assert result.picks["date"].min() >= pd.Timestamp("2025-01-01")
    assert result.picks["date"].max() <= pd.Timestamp("2025-12-31")
