from pathlib import Path

import pandas as pd

from worldcup_predictor.simulation import simulate_tournament


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOURNAMENT_SIMULATIONS = 500000
SIMULATION_SEED = 2026


def main() -> None:
    predictions_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_predictions.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_tournament_simulation.csv"

    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    simulation = simulate_tournament(predictions, simulations=TOURNAMENT_SIMULATIONS, seed=SIMULATION_SEED)
    simulation.to_csv(output_path, index=False)

    print(f"Teams simulated: {len(simulation)}")
    print(f"Saved {output_path}")
    print("")
    print("Top champion probabilities")
    print(simulation.head(20).to_string(index=False))
    print("")
    champion = simulation.sort_values("champion_probability", ascending=False).iloc[0]
    runner_up_candidates = simulation[simulation["team"] != champion.team]
    runner_up = runner_up_candidates.sort_values("runner_up_probability", ascending=False).iloc[0]
    third_candidates = simulation[
        ~simulation["team"].isin({champion.team, runner_up.team})
    ]
    third = third_candidates.sort_values("third_place_probability", ascending=False).iloc[0]
    print("Recommended podium by marginal probabilities")
    print(f"Champion: {champion.team} ({champion.champion_probability:.1%})")
    print(f"Runner-up: {runner_up.team} ({runner_up.runner_up_probability:.1%})")
    print(f"Third place: {third.team} ({third.third_place_probability:.1%})")


if __name__ == "__main__":
    main()
