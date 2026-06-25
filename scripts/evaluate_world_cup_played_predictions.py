from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup_predictor.future import (
    build_fixture_features,
    build_team_states,
    completed_matches,
    load_raw_results_with_fixtures,
    predict_fixture_picks,
)
from worldcup_predictor.modeling import fit_goal_model
from worldcup_predictor.overrides import (
    apply_result_overrides,
    load_match_context_overrides,
    load_result_overrides,
)
from worldcup_predictor.scoring import points_for_prediction, scoring_category


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOURNAMENT = "FIFA World Cup"
START_DATE = "2026-06-01"
MAX_POINTS_PER_MATCH = 10

OUTPUT_COLUMNS = [
    "date",
    "tournament",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "model_home_expected_goals",
    "model_away_expected_goals",
    "home_expected_goals",
    "away_expected_goals",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "recommended_home_score",
    "recommended_away_score",
    "recommended_scoreline",
    "recommended_expected_points",
    "expected_effectiveness",
    "actual_points",
    "actual_effectiveness",
    "actual_category",
    "result_correct",
    "exact_score",
]


def evaluate_played_predictions(
    processed_matches: pd.DataFrame,
    completed_results: pd.DataFrame,
    context_overrides: pd.DataFrame | None = None,
    tournament: str = TOURNAMENT,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """Evaluate model picks for played World Cup matches without same-day leakage."""
    processed_matches = processed_matches.copy()
    completed_results = completed_results.copy()
    processed_matches["date"] = pd.to_datetime(processed_matches["date"])
    completed_results["date"] = pd.to_datetime(completed_results["date"])

    played = completed_results[
        (completed_results["tournament"] == tournament)
        & (completed_results["date"] >= pd.Timestamp(start_date))
    ].copy()
    if played.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows: list[pd.DataFrame] = []
    for match_date in sorted(played["date"].unique()):
        match_date = pd.Timestamp(match_date)
        train = processed_matches[processed_matches["date"] < match_date].copy()
        history = completed_results[completed_results["date"] < match_date].copy()
        fixtures = played[played["date"] == match_date].copy()
        if len(train) < 20 or fixtures.empty:
            continue

        model = fit_goal_model(train)
        team_states = build_team_states(history)
        fixture_features = build_fixture_features(fixtures, team_states)
        predictions = predict_fixture_picks(
            model,
            fixture_features,
            context_overrides=context_overrides,
        )

        predictions["home_score"] = fixtures["home_score"].to_numpy(dtype=int)
        predictions["away_score"] = fixtures["away_score"].to_numpy(dtype=int)
        rows.append(predictions)

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    audit = pd.concat(rows, ignore_index=True)
    actual_points: list[int] = []
    categories: list[str] = []
    for row in audit.itertuples(index=False):
        prediction = (int(row.recommended_home_score), int(row.recommended_away_score))
        actual = (int(row.home_score), int(row.away_score))
        actual_points.append(points_for_prediction(prediction, actual))
        categories.append(scoring_category(prediction, actual))

    audit["actual_points"] = actual_points
    audit["actual_category"] = categories
    audit["expected_effectiveness"] = audit["recommended_expected_points"] / MAX_POINTS_PER_MATCH
    audit["actual_effectiveness"] = audit["actual_points"] / MAX_POINTS_PER_MATCH
    audit["result_correct"] = audit["actual_category"] != "miss"
    audit["exact_score"] = audit["actual_category"] == "exact_score"

    return audit.loc[:, OUTPUT_COLUMNS].sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def main() -> None:
    raw_results_path = PROJECT_ROOT / "data" / "raw" / "results.csv"
    overrides_path = PROJECT_ROOT / "data" / "manual" / "match_results_overrides.csv"
    context_overrides_path = PROJECT_ROOT / "data" / "manual" / "match_context_overrides.csv"
    processed_matches_path = PROJECT_ROOT / "data" / "processed" / "matches_with_features.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_prediction_audit.csv"

    raw_results = load_raw_results_with_fixtures(raw_results_path)
    overrides = load_result_overrides(overrides_path)
    context_overrides = load_match_context_overrides(context_overrides_path)
    completed = completed_matches(apply_result_overrides(raw_results, overrides))
    processed_matches = pd.read_csv(processed_matches_path, parse_dates=["date"])

    audit = evaluate_played_predictions(
        processed_matches=processed_matches,
        completed_results=completed,
        context_overrides=context_overrides,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    print(f"Played matches evaluated: {len(audit)}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
