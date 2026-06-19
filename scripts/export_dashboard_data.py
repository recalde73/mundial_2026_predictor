from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "public" / "data"
GROUP_SIMULATIONS = 25000
TOURNAMENT_SIMULATIONS = 500000
SIMULATION_SEED = 2026


DATASETS = {
    "predictions": PROCESSED_DIR / "world_cup_2026_predictions.csv",
    "groups": PROCESSED_DIR / "world_cup_2026_group_simulation.csv",
    "tournament": PROCESSED_DIR / "world_cup_2026_tournament_simulation.csv",
    "top_scorers": PROCESSED_DIR / "world_cup_2026_top_scorer_predictions.csv",
    "qualification_backtest_strategies": PROCESSED_DIR / "backtest_2021_qualification_from_2018_cycle_strategies.csv",
    "qualification_backtest_picks": PROCESSED_DIR / "backtest_2021_qualification_from_2018_cycle_picks.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    frame = pd.read_csv(path)
    for column in frame.columns:
        if column == "date":
            frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    return frame


def export_dashboard_data() -> None:
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "data/processed CSV outputs",
        "group_simulations": GROUP_SIMULATIONS,
        "tournament_simulations": TOURNAMENT_SIMULATIONS,
        "simulation_seed": SIMULATION_SEED,
    }
    pd.Series(metadata).to_json(
        DASHBOARD_DATA_DIR / "metadata.json",
        force_ascii=False,
        indent=2,
    )

    for name, path in DATASETS.items():
        frame = _read_csv(path)
        frame.to_json(
            DASHBOARD_DATA_DIR / f"{name}.json",
            orient="records",
            force_ascii=False,
            indent=2,
        )
        print(f"Exported {name}: {len(frame)} rows")


if __name__ == "__main__":
    export_dashboard_data()
