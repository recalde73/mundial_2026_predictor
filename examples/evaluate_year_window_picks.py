from pathlib import Path

import pandas as pd

from worldcup_predictor.backtesting import PickSummary, backtest_model_on_frozen_period
from worldcup_predictor.future import completed_matches, load_raw_results_with_fixtures
from worldcup_predictor.overrides import apply_result_overrides, load_result_overrides


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_END_DATE = "2024-12-31"
VALIDATION_START_DATE = "2025-01-01"
VALIDATION_END_DATE = "2025-12-31"


def print_summary(title: str, summary: PickSummary) -> None:
    print(title)
    print(f"Matches: {summary.matches}")
    print(f"Total points: {summary.total_points}")
    print(f"Average points: {summary.average_points:.3f}")
    print(f"Exact scores: {summary.exact_scores} ({summary.exact_score_rate:.1%})")
    print(f"Result accuracy: {summary.result_accuracy_rate:.1%}")
    print(f"Winner + diff: {summary.winner_and_diff}")
    print(f"Draws: {summary.draws}")
    print(f"Winners: {summary.winners}")
    print(f"Misses: {summary.misses} ({summary.miss_rate:.1%})")


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "processed" / "matches_with_features.csv"
    raw_results_path = PROJECT_ROOT / "data" / "raw" / "results.csv"
    overrides_path = PROJECT_ROOT / "data" / "manual" / "match_results_overrides.csv"
    processed_matches = pd.read_csv(dataset_path, parse_dates=["date"])
    raw_results = load_raw_results_with_fixtures(raw_results_path)
    overrides = load_result_overrides(overrides_path)
    completed_results = completed_matches(apply_result_overrides(raw_results, overrides))
    result = backtest_model_on_frozen_period(
        processed_matches,
        completed_results,
        train_end_date=TRAIN_END_DATE,
        validation_start_date=VALIDATION_START_DATE,
        validation_end_date=VALIDATION_END_DATE,
    )

    print("Year-window pick backtest")
    print(f"Training window: through {TRAIN_END_DATE}")
    print(f"Team state frozen at: {TRAIN_END_DATE}")
    print(f"Validation window: {VALIDATION_START_DATE} through {VALIDATION_END_DATE}")
    print(f"Training matches: {result.model.train_rows}")
    print(f"Validation matches: {result.model.validation_rows}")
    print(f"Home goals MAE: {result.model.home_mae:.3f}")
    print(f"Away goals MAE: {result.model.away_mae:.3f}")
    print("")
    print_summary("Optimized daily-points strategy", result.optimized_summary)
    print("")
    print_summary("Most likely scoreline", result.most_likely_summary)

    output_path = PROJECT_ROOT / "data" / "processed" / "backtest_2025_from_2024_picks.csv"
    result.picks.to_csv(output_path, index=False)
    print("")
    print(f"Saved validation picks: {output_path}")


if __name__ == "__main__":
    main()
