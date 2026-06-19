from pathlib import Path

import pandas as pd

from worldcup_predictor.future import completed_matches, load_raw_results_with_fixtures
from worldcup_predictor.overrides import apply_result_overrides, load_result_overrides
from worldcup_predictor.simulation import add_completed_results_for_simulation, simulate_group_stage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP_SIMULATIONS = 25000
SIMULATION_SEED = 2026


def main() -> None:
    predictions_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_predictions.csv"
    raw_results_path = PROJECT_ROOT / "data" / "raw" / "results.csv"
    overrides_path = PROJECT_ROOT / "data" / "manual" / "match_results_overrides.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_group_simulation.csv"

    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    raw_results = load_raw_results_with_fixtures(raw_results_path)
    overrides = load_result_overrides(overrides_path)
    completed = completed_matches(apply_result_overrides(raw_results, overrides))
    simulation_input = add_completed_results_for_simulation(predictions, completed)
    simulation = simulate_group_stage(simulation_input, simulations=GROUP_SIMULATIONS, seed=SIMULATION_SEED)
    simulation.to_csv(output_path, index=False)

    print(f"Teams simulated: {len(simulation)}")
    print(f"Saved {output_path}")
    print("")
    print(simulation.to_string(index=False))


if __name__ == "__main__":
    main()
