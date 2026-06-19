from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from worldcup_predictor.modeling import (
    BaselineGoalModel,
    fit_goal_model,
    predict_expected_goals,
    split_matches_chronologically,
    train_baseline_goal_model,
)
from worldcup_predictor.poisson import most_likely_scorelines, scoreline_probabilities
from worldcup_predictor.recommender import (
    DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    DEFAULT_GOAL_INFLATION,
    DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    apply_goal_inflation,
    rank_predictions,
)
from worldcup_predictor.scoring import (
    DRAW,
    EXACT_SCORE,
    MISS,
    WINNER,
    WINNER_AND_DIFF,
    Scoreline,
    points_for_prediction,
    scoring_category,
)

@dataclass(frozen=True)
class PickSummary:
    matches: int
    total_points: int
    average_points: float
    exact_scores: int
    winner_and_diff: int
    draws: int
    winners: int
    misses: int

    @property
    def exact_score_rate(self) -> float:
        return self.exact_scores / self.matches if self.matches else 0.0

    @property
    def result_accuracy_rate(self) -> float:
        return (self.matches - self.misses) / self.matches if self.matches else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.matches if self.matches else 0.0


@dataclass(frozen=True)
class BacktestResult:
    model: BaselineGoalModel
    picks: pd.DataFrame
    optimized_summary: PickSummary
    most_likely_summary: PickSummary


def evaluate_pick_strategies(
    predictions: pd.DataFrame,
    max_goals: int = 6,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    goal_inflation: float = DEFAULT_GOAL_INFLATION,
) -> pd.DataFrame:
    """Evaluate optimized and most-likely scoreline picks against actual results."""
    rows: list[dict[str, object]] = []

    for match in predictions.itertuples(index=False):
        strategy_home_expected_goals, strategy_away_expected_goals = apply_goal_inflation(
            float(match.home_expected_goals),
            float(match.away_expected_goals),
            goal_inflation=goal_inflation,
        )
        probabilities = scoreline_probabilities(
            home_expected_goals=strategy_home_expected_goals,
            away_expected_goals=strategy_away_expected_goals,
            max_goals=max_goals,
        )
        optimized_scoreline, optimized_expected_points = rank_predictions(
            probabilities,
            max_candidate_goals=max_goals,
            max_total_candidate_goals=max_total_candidate_goals,
            draw_probability_multiplier=draw_probability_multiplier,
            limit=1,
        )[0]
        most_likely_scoreline = most_likely_scorelines(probabilities, limit=1)[0][0]
        actual_scoreline = (int(match.home_score), int(match.away_score))

        rows.append(
            {
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_score": int(match.home_score),
                "away_score": int(match.away_score),
                "home_expected_goals": float(match.home_expected_goals),
                "away_expected_goals": float(match.away_expected_goals),
                "strategy_home_expected_goals": strategy_home_expected_goals,
                "strategy_away_expected_goals": strategy_away_expected_goals,
                "goal_inflation": goal_inflation,
                "max_total_candidate_goals": max_total_candidate_goals,
                "draw_probability_multiplier": draw_probability_multiplier,
                "optimized_home_score": optimized_scoreline[0],
                "optimized_away_score": optimized_scoreline[1],
                "optimized_expected_points": float(optimized_expected_points),
                "optimized_points": points_for_prediction(optimized_scoreline, actual_scoreline),
                "optimized_category": scoring_category(optimized_scoreline, actual_scoreline),
                "most_likely_home_score": most_likely_scoreline[0],
                "most_likely_away_score": most_likely_scoreline[1],
                "most_likely_points": points_for_prediction(most_likely_scoreline, actual_scoreline),
                "most_likely_category": scoring_category(most_likely_scoreline, actual_scoreline),
            }
        )

    return pd.DataFrame(rows)


def summarize_picks(
    picks: pd.DataFrame,
    points_column: str,
    category_column: str,
) -> PickSummary:
    """Summarize historical points for one pick strategy."""
    matches = len(picks)
    total_points = int(picks[points_column].sum())
    categories = picks[category_column].value_counts().to_dict()

    return PickSummary(
        matches=matches,
        total_points=total_points,
        average_points=total_points / matches if matches else 0.0,
        exact_scores=int(categories.get(EXACT_SCORE, 0)),
        winner_and_diff=int(categories.get(WINNER_AND_DIFF, 0)),
        draws=int(categories.get(DRAW, 0)),
        winners=int(categories.get(WINNER, 0)),
        misses=int(categories.get(MISS, 0)),
    )


def _build_backtest_result(model: BaselineGoalModel, picks: pd.DataFrame) -> BacktestResult:
    return BacktestResult(
        model=model,
        picks=picks,
        optimized_summary=summarize_picks(
            picks,
            points_column="optimized_points",
            category_column="optimized_category",
        ),
        most_likely_summary=summarize_picks(
            picks,
            points_column="most_likely_points",
            category_column="most_likely_category",
        ),
    )


def backtest_baseline_model(
    matches: pd.DataFrame,
    validation_fraction: float = 0.2,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    goal_inflation: float = DEFAULT_GOAL_INFLATION,
) -> BacktestResult:
    """Train the baseline model and backtest picks on chronological validation."""
    model = train_baseline_goal_model(matches, validation_fraction=validation_fraction)
    _train, validation = split_matches_chronologically(matches, validation_fraction)
    predictions = predict_expected_goals(model, validation)
    picks = evaluate_pick_strategies(
        predictions,
        max_total_candidate_goals=max_total_candidate_goals,
        draw_probability_multiplier=draw_probability_multiplier,
        goal_inflation=goal_inflation,
    )

    return _build_backtest_result(model, picks)


def backtest_model_on_frozen_period(
    processed_matches: pd.DataFrame,
    completed_results: pd.DataFrame,
    train_end_date: str | pd.Timestamp,
    validation_start_date: str | pd.Timestamp,
    validation_end_date: str | pd.Timestamp,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    goal_inflation: float = DEFAULT_GOAL_INFLATION,
) -> BacktestResult:
    """Train through one date and predict a future window from frozen team state."""
    from worldcup_predictor.future import build_fixture_features, build_team_states

    train_end = pd.Timestamp(train_end_date)
    validation_start = pd.Timestamp(validation_start_date)
    validation_end = pd.Timestamp(validation_end_date)
    if validation_start > validation_end:
        raise ValueError("validation_start_date must be before validation_end_date")

    train = processed_matches[processed_matches["date"] <= train_end].copy()
    if len(train) < 20:
        raise ValueError("At least 20 training matches are required")

    completed = completed_results.sort_values("date").reset_index(drop=True)
    history = completed[completed["date"] <= train_end].copy()
    validation = completed[
        (completed["date"] >= validation_start)
        & (completed["date"] <= validation_end)
    ].copy()
    if validation.empty:
        raise ValueError("At least one validation match is required")

    model = fit_goal_model(train)
    team_states = build_team_states(history)
    fixture_features = build_fixture_features(validation, team_states)
    predictions = predict_expected_goals(model, fixture_features)
    predictions["home_score"] = validation["home_score"].to_numpy(dtype=int)
    predictions["away_score"] = validation["away_score"].to_numpy(dtype=int)

    model = replace(
        model,
        validation_rows=len(validation),
        home_mae=float((predictions["home_score"] - predictions["home_expected_goals"]).abs().mean()),
        away_mae=float((predictions["away_score"] - predictions["away_expected_goals"]).abs().mean()),
    )
    picks = evaluate_pick_strategies(
        predictions,
        max_total_candidate_goals=max_total_candidate_goals,
        draw_probability_multiplier=draw_probability_multiplier,
        goal_inflation=goal_inflation,
    )

    return _build_backtest_result(model, picks)


def backtest_model_on_period(
    matches: pd.DataFrame,
    train_end_date: str | pd.Timestamp,
    validation_start_date: str | pd.Timestamp,
    validation_end_date: str | pd.Timestamp,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    goal_inflation: float = DEFAULT_GOAL_INFLATION,
) -> BacktestResult:
    """Train through one date and evaluate picks on a later fixed window."""
    train_end = pd.Timestamp(train_end_date)
    validation_start = pd.Timestamp(validation_start_date)
    validation_end = pd.Timestamp(validation_end_date)
    if validation_start > validation_end:
        raise ValueError("validation_start_date must be before validation_end_date")

    sorted_matches = matches.sort_values("date").reset_index(drop=True)
    train = sorted_matches[sorted_matches["date"] <= train_end].copy()
    validation = sorted_matches[
        (sorted_matches["date"] >= validation_start)
        & (sorted_matches["date"] <= validation_end)
    ].copy()
    if len(train) < 20:
        raise ValueError("At least 20 training matches are required")
    if validation.empty:
        raise ValueError("At least one validation match is required")

    model = fit_goal_model(train)
    predictions = predict_expected_goals(model, validation)
    model = replace(
        model,
        validation_rows=len(validation),
        home_mae=float((predictions["home_score"] - predictions["home_expected_goals"]).abs().mean()),
        away_mae=float((predictions["away_score"] - predictions["away_expected_goals"]).abs().mean()),
    )
    picks = evaluate_pick_strategies(
        predictions,
        max_total_candidate_goals=max_total_candidate_goals,
        draw_probability_multiplier=draw_probability_multiplier,
        goal_inflation=goal_inflation,
    )

    return _build_backtest_result(model, picks)


def scoreline_from_columns(row: pd.Series, home_column: str, away_column: str) -> Scoreline:
    """Return a scoreline tuple from two dataframe columns."""
    return int(row[home_column]), int(row[away_column])
