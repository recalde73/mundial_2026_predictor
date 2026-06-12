import pandas as pd

from worldcup_predictor.backtesting import evaluate_pick_strategies, summarize_picks


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
