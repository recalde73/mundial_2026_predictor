from pathlib import Path
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COMMANDS = [
    ["scripts/download_real_data.py"],
    ["scripts/build_training_dataset.py"],
    ["examples/evaluate_baseline_picks.py"],
    ["scripts/predict_world_cup_fixtures.py"],
    ["scripts/simulate_world_cup_groups.py"],
    ["scripts/simulate_world_cup_tournament.py"],
    ["scripts/predict_top_scorer.py"],
    ["scripts/export_dashboard_data.py"],
]


def main() -> None:
    python = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    for command in COMMANDS:
        print("")
        print(f"Running {' '.join(command)}")
        subprocess.run(
            [python, *command],
            cwd=PROJECT_ROOT,
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
