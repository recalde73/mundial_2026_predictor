from pathlib import Path

import pandas as pd

from worldcup_predictor.simulation import simulate_group_stage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP_SIMULATIONS = 25000
SIMULATION_SEED = 2026


def main() -> None:
    predictions_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_predictions.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_group_simulation.csv"

    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    simulation = simulate_group_stage(predictions, simulations=GROUP_SIMULATIONS, seed=SIMULATION_SEED)
    simulation.to_csv(output_path, index=False)

    print(f"Teams simulated: {len(simulation)}")
    print(f"Saved {output_path}")
    print("")
    print(simulation.to_string(index=False))


if __name__ == "__main__":
    main()
