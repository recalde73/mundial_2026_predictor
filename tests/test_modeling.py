import pandas as pd

from worldcup_predictor.datasets import build_match_features
from worldcup_predictor.modeling import (
    predict_expected_goals,
    split_matches_chronologically,
    train_baseline_goal_model,
)


def test_train_baseline_goal_model_and_predict() -> None:
    rows = []
    for index in range(40):
        rows.append(
            [
                f"2000-01-{(index % 28) + 1:02d}",
                "A" if index % 2 == 0 else "B",
                "B" if index % 2 == 0 else "A",
                2 if index % 2 == 0 else 1,
                1 if index % 2 == 0 else 2,
                "Friendly",
                "X",
                "Y",
                True,
            ]
        )
    results = pd.DataFrame(
        rows,
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ],
    )
    results["date"] = pd.date_range("2000-01-01", periods=len(results), freq="D")
    features = build_match_features(results, min_prior_matches=2, start_date=None)

    model = train_baseline_goal_model(features, validation_fraction=0.25)
    predictions = predict_expected_goals(model, features.tail(2))

    assert model.train_rows > 0
    assert model.validation_rows > 0
    assert all(predictions["home_expected_goals"] > 0)
    assert all(predictions["away_expected_goals"] > 0)


def test_split_matches_chronologically_keeps_validation_after_training() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "home_score": [1] * 10,
            "away_score": [0] * 10,
        }
    )

    train, validation = split_matches_chronologically(matches, validation_fraction=0.3)

    assert len(train) == 7
    assert len(validation) == 3
    assert train.date.max() < validation.date.min()
