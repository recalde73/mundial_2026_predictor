from pathlib import Path

import pandas as pd

from worldcup_predictor.future import (
    completed_matches,
    future_fixtures,
    load_raw_results_with_fixtures,
    train_final_model_and_predict_fixtures,
)
from worldcup_predictor.overrides import (
    apply_result_overrides,
    load_match_context_overrides,
    load_result_overrides,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw_results_path = PROJECT_ROOT / "data" / "raw" / "results.csv"
    overrides_path = PROJECT_ROOT / "data" / "manual" / "match_results_overrides.csv"
    context_overrides_path = PROJECT_ROOT / "data" / "manual" / "match_context_overrides.csv"
    processed_matches_path = PROJECT_ROOT / "data" / "processed" / "matches_with_features.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_predictions.csv"

    raw_results = load_raw_results_with_fixtures(raw_results_path)
    overrides = load_result_overrides(overrides_path)
    context_overrides = load_match_context_overrides(context_overrides_path)
    raw_results = apply_result_overrides(raw_results, overrides)
    processed_matches = pd.read_csv(processed_matches_path, parse_dates=["date"])
    completed = completed_matches(raw_results)
    fixtures = future_fixtures(raw_results, tournament="FIFA World Cup", start_date="2026-06-01")

    if fixtures.empty:
        raise SystemExit("No future FIFA World Cup fixtures found in data/raw/results.csv")

    predictions = train_final_model_and_predict_fixtures(
        processed_matches=processed_matches,
        completed_results=completed,
        fixtures=fixtures,
        context_overrides=context_overrides,
    )
    predictions.to_csv(output_path, index=False)

    print(f"Fixtures predicted: {len(predictions)}")
    print(f"Manual overrides applied: {len(overrides)}")
    print(f"Manual context overrides loaded: {len(context_overrides)}")
    print(f"Fixture context adjustments applied: {int(predictions['context_applied'].sum())}")
    print(f"Saved {output_path}")
    print("")
    print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
