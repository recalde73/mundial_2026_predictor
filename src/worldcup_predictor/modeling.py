from __future__ import annotations

from dataclasses import dataclass

import math

import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error


DEFAULT_FEATURE_COLUMNS = (
    "neutral",
    "is_friendly",
    "elo_diff_scaled",
    "home_elo_expected_result",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "home_points_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
    "away_points_avg",
    "home_attack_minus_away_defense",
    "away_attack_minus_home_defense",
)


@dataclass(frozen=True)
class BaselineGoalModel:
    home_model: PoissonRegressor
    away_model: PoissonRegressor
    feature_columns: tuple[str, ...]
    train_rows: int
    validation_rows: int
    home_mae: float
    away_mae: float


def _prepare_features(
    matches: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    features = matches.loc[:, feature_columns].copy()
    for column in features.select_dtypes(include=["bool"]).columns:
        features[column] = features[column].astype(int)
    return features.astype(float)


def fit_goal_model(
    matches: pd.DataFrame,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
) -> BaselineGoalModel:
    """Fit goal models on every available row for future predictions."""
    if len(matches) < 20:
        raise ValueError("At least 20 matches are required to fit the goal model")

    matches = matches.sort_values("date").reset_index(drop=True)
    features = _prepare_features(matches, feature_columns)

    home_model = PoissonRegressor(alpha=0.001, max_iter=1000)
    away_model = PoissonRegressor(alpha=0.001, max_iter=1000)
    home_model.fit(features, matches["home_score"])
    away_model.fit(features, matches["away_score"])

    return BaselineGoalModel(
        home_model=home_model,
        away_model=away_model,
        feature_columns=feature_columns,
        train_rows=len(matches),
        validation_rows=0,
        home_mae=float("nan"),
        away_mae=float("nan"),
    )


def train_baseline_goal_model(
    matches: pd.DataFrame,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    validation_fraction: float = 0.2,
) -> BaselineGoalModel:
    """Train separate Poisson regressors for home and away goals."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(matches) < 20:
        raise ValueError("At least 20 matches are required to train the baseline model")

    train, validation = split_matches_chronologically(matches, validation_fraction)

    train_x = _prepare_features(train, feature_columns)
    validation_x = _prepare_features(validation, feature_columns)

    home_model = PoissonRegressor(alpha=0.001, max_iter=1000)
    away_model = PoissonRegressor(alpha=0.001, max_iter=1000)
    home_model.fit(train_x, train["home_score"])
    away_model.fit(train_x, train["away_score"])

    home_predictions = home_model.predict(validation_x)
    away_predictions = away_model.predict(validation_x)

    return BaselineGoalModel(
        home_model=home_model,
        away_model=away_model,
        feature_columns=feature_columns,
        train_rows=len(train),
        validation_rows=len(validation),
        home_mae=float(mean_absolute_error(validation["home_score"], home_predictions)),
        away_mae=float(mean_absolute_error(validation["away_score"], away_predictions)),
    )


def split_matches_chronologically(
    matches: pd.DataFrame,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split matches by date so validation always happens after training."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(matches) < 2:
        raise ValueError("At least 2 matches are required to split the dataset")

    matches = matches.sort_values("date").reset_index(drop=True)
    split_index = max(1, math.floor(len(matches) * (1 - validation_fraction)))
    return matches.iloc[:split_index].copy(), matches.iloc[split_index:].copy()


def predict_expected_goals(model: BaselineGoalModel, matches: pd.DataFrame) -> pd.DataFrame:
    """Predict expected goals for match feature rows."""
    features = _prepare_features(matches, model.feature_columns)
    predictions = matches.copy()
    predictions["home_expected_goals"] = model.home_model.predict(features)
    predictions["away_expected_goals"] = model.away_model.predict(features)
    return predictions
