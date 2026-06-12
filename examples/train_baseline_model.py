from pathlib import Path

import pandas as pd

from worldcup_predictor.formatting import format_recommendation
from worldcup_predictor.modeling import predict_expected_goals, train_baseline_goal_model
from worldcup_predictor.recommender import recommend_pick


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "processed" / "matches_with_features.csv"
    matches = pd.read_csv(dataset_path, parse_dates=["date"])
    model = train_baseline_goal_model(matches)

    print("Baseline Poisson model")
    print(f"Train rows: {model.train_rows}")
    print(f"Validation rows: {model.validation_rows}")
    print(f"Home goals MAE: {model.home_mae:.3f}")
    print(f"Away goals MAE: {model.away_mae:.3f}")

    sample = matches.tail(1)
    prediction = predict_expected_goals(model, sample).iloc[0]
    recommendation = recommend_pick(
        home_team=prediction.home_team,
        away_team=prediction.away_team,
        home_expected_goals=float(prediction.home_expected_goals),
        away_expected_goals=float(prediction.away_expected_goals),
    )

    print("")
    print("Example recommendation from the latest processed match:")
    print(format_recommendation(recommendation))
    print("")
    print(
        "Actual result: "
        f"{prediction.home_team} {int(prediction.home_score)}-"
        f"{int(prediction.away_score)} {prediction.away_team}"
    )


if __name__ == "__main__":
    main()
