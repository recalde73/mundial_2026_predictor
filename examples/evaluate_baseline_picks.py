from pathlib import Path

import pandas as pd

from worldcup_predictor.backtesting import PickSummary, backtest_baseline_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    matches = pd.read_csv(dataset_path, parse_dates=["date"])
    result = backtest_baseline_model(matches)

    print("Baseline pick backtest")
    print(f"Home goals MAE: {result.model.home_mae:.3f}")
    print(f"Away goals MAE: {result.model.away_mae:.3f}")
    print("")
    print_summary("Optimized daily-points strategy", result.optimized_summary)
    print("")
    print_summary("Most likely scoreline", result.most_likely_summary)

    output_path = PROJECT_ROOT / "data" / "processed" / "baseline_validation_picks.csv"
    result.picks.to_csv(output_path, index=False)
    print("")
    print(f"Saved validation picks: {output_path}")


if __name__ == "__main__":
    main()
