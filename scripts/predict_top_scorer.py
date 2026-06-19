from pathlib import Path

import pandas as pd

from worldcup_predictor.future import completed_matches, load_raw_results_with_fixtures
from worldcup_predictor.overrides import apply_result_overrides, load_result_overrides
from worldcup_predictor.simulation import add_completed_results_for_simulation
from worldcup_predictor.top_scorer import estimate_top_scorers, load_top_scorer_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    candidates_path = PROJECT_ROOT / "data" / "raw" / "top_scorer_candidates.csv"
    predictions_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_predictions.csv"
    raw_results_path = PROJECT_ROOT / "data" / "raw" / "results.csv"
    overrides_path = PROJECT_ROOT / "data" / "manual" / "match_results_overrides.csv"
    tournament_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_tournament_simulation.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_top_scorer_predictions.csv"

    candidates = load_top_scorer_candidates(candidates_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    raw_results = load_raw_results_with_fixtures(raw_results_path)
    overrides = load_result_overrides(overrides_path)
    completed = completed_matches(apply_result_overrides(raw_results, overrides))
    predictions = add_completed_results_for_simulation(predictions, completed)
    tournament = pd.read_csv(tournament_path)
    top_scorers = estimate_top_scorers(candidates, predictions, tournament)
    top_scorers.to_csv(output_path, index=False)

    print(f"Candidates evaluated: {len(top_scorers)}")
    print(f"Saved {output_path}")
    print("")
    print(top_scorers.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
