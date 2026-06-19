from pathlib import Path

import pandas as pd

from worldcup_predictor.backtesting import evaluate_pick_strategies, summarize_picks
from worldcup_predictor.datasets import build_match_features
from worldcup_predictor.future import (
    build_fixture_features,
    build_team_states,
    completed_matches,
    load_raw_results_with_fixtures,
)
from worldcup_predictor.modeling import fit_goal_model, predict_expected_goals
from worldcup_predictor.overrides import apply_result_overrides, load_result_overrides


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_QUALIFICATION_START_DATE = "2015-01-01"
WORLD_CUP_2018_START_DATE = "2018-06-14"
WORLD_CUP_2018_END_DATE = "2018-07-15"
VALIDATION_START_DATE = "2021-01-01"
VALIDATION_END_DATE = "2021-12-31"
TOURNAMENT = "FIFA World Cup qualification"


def strategy_grid() -> list[dict[str, float | int | None]]:
    configs: list[dict[str, float | int | None]] = []
    for goal_multiplier in (0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20):
        for max_total_goals in (2, 3, 4, None):
            for draw_multiplier in (0.90, 1.00, 1.10, 1.21, 1.35):
                configs.append(
                    {
                        "goal_inflation": goal_multiplier,
                        "max_total_candidate_goals": max_total_goals,
                        "draw_probability_multiplier": draw_multiplier,
                    }
                )
    return configs


def select_training_results(results: pd.DataFrame) -> pd.DataFrame:
    qualification = (
        (results["tournament"] == TOURNAMENT)
        & (results["date"] >= pd.Timestamp(TRAIN_QUALIFICATION_START_DATE))
        & (results["date"] < pd.Timestamp(WORLD_CUP_2018_START_DATE))
    )
    world_cup = (
        (results["tournament"] == "FIFA World Cup")
        & (results["date"] >= pd.Timestamp(WORLD_CUP_2018_START_DATE))
        & (results["date"] <= pd.Timestamp(WORLD_CUP_2018_END_DATE))
    )
    return results[qualification | world_cup].sort_values("date").reset_index(drop=True)


def select_validation_results(results: pd.DataFrame) -> pd.DataFrame:
    validation = (
        (results["tournament"] == TOURNAMENT)
        & (results["date"] >= pd.Timestamp(VALIDATION_START_DATE))
        & (results["date"] <= pd.Timestamp(VALIDATION_END_DATE))
    )
    return results[validation].sort_values("date").reset_index(drop=True)


def evaluate_strategies(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_rows: list[dict[str, object]] = []
    pick_frames: list[pd.DataFrame] = []

    for config in strategy_grid():
        picks = evaluate_pick_strategies(
            predictions,
            max_total_candidate_goals=config["max_total_candidate_goals"],
            draw_probability_multiplier=float(config["draw_probability_multiplier"]),
            goal_inflation=float(config["goal_inflation"]),
        )
        summary = summarize_picks(picks, "optimized_points", "optimized_category")
        average_pick_goals = picks["optimized_home_score"].add(picks["optimized_away_score"]).mean()
        high_total_pick_rate = picks["optimized_home_score"].add(picks["optimized_away_score"]).ge(3).mean()
        strategy_rows.append(
            {
                **config,
                "matches": summary.matches,
                "total_points": summary.total_points,
                "average_points": summary.average_points,
                "exact_scores": summary.exact_scores,
                "exact_score_rate": summary.exact_score_rate,
                "result_accuracy_rate": summary.result_accuracy_rate,
                "winner_and_diff": summary.winner_and_diff,
                "draws": summary.draws,
                "winners": summary.winners,
                "misses": summary.misses,
                "miss_rate": summary.miss_rate,
                "average_pick_goals": average_pick_goals,
                "high_total_pick_rate": high_total_pick_rate,
            }
        )
        picks = picks.copy()
        picks["strategy_goal_inflation"] = config["goal_inflation"]
        picks["strategy_max_total_candidate_goals"] = config["max_total_candidate_goals"]
        picks["strategy_draw_probability_multiplier"] = config["draw_probability_multiplier"]
        pick_frames.append(picks)

    strategies = pd.DataFrame(strategy_rows).sort_values(
        ["average_points", "exact_score_rate", "result_accuracy_rate"],
        ascending=False,
    ).reset_index(drop=True)
    strategies.insert(0, "rank", range(1, len(strategies) + 1))
    strategies["selected_strategy"] = strategies["rank"] == 1

    best = strategies.iloc[0]
    best_picks = next(
        frame
        for frame in pick_frames
        if frame["strategy_goal_inflation"].iloc[0] == best["goal_inflation"]
        and frame["strategy_max_total_candidate_goals"].iloc[0] == best["max_total_candidate_goals"]
        and frame["strategy_draw_probability_multiplier"].iloc[0] == best["draw_probability_multiplier"]
    )
    return strategies, best_picks


def main() -> None:
    raw_results_path = PROJECT_ROOT / "data" / "raw" / "results.csv"
    overrides_path = PROJECT_ROOT / "data" / "manual" / "match_results_overrides.csv"
    picks_output_path = PROJECT_ROOT / "data" / "processed" / "backtest_2021_qualification_from_2018_cycle_picks.csv"
    strategies_output_path = PROJECT_ROOT / "data" / "processed" / "backtest_2021_qualification_from_2018_cycle_strategies.csv"

    raw_results = load_raw_results_with_fixtures(raw_results_path)
    overrides = load_result_overrides(overrides_path)
    results = completed_matches(apply_result_overrides(raw_results, overrides))
    training_results = select_training_results(results)
    validation_results = select_validation_results(results)

    training_features = build_match_features(training_results, min_prior_matches=0, start_date=None)
    model = fit_goal_model(training_features)
    team_states = build_team_states(training_results)
    validation_features = build_fixture_features(validation_results, team_states)
    predictions = predict_expected_goals(model, validation_features)
    predictions["home_score"] = validation_results["home_score"].to_numpy(dtype=int)
    predictions["away_score"] = validation_results["away_score"].to_numpy(dtype=int)

    strategies, best_picks = evaluate_strategies(predictions)
    home_mae = float((predictions["home_score"] - predictions["home_expected_goals"]).abs().mean())
    away_mae = float((predictions["away_score"] - predictions["away_expected_goals"]).abs().mean())

    for frame in (strategies, best_picks):
        frame["training_window"] = "2018 qualification cycle plus 2018 World Cup"
        frame["validation_window"] = "2021 FIFA World Cup qualification"
        frame["training_matches"] = len(training_results)
        frame["training_feature_rows"] = len(training_features)
        frame["validation_matches"] = len(validation_results)
        frame["home_mae"] = home_mae
        frame["away_mae"] = away_mae

    picks_output_path.parent.mkdir(parents=True, exist_ok=True)
    strategies.to_csv(strategies_output_path, index=False)
    best_picks.to_csv(picks_output_path, index=False)

    best = strategies.iloc[0]
    print("2018 cycle -> 2021 World Cup qualification backtest")
    print(f"Training matches: {len(training_results)}")
    print(f"Training feature rows: {len(training_features)}")
    print(f"Validation matches: {len(validation_results)}")
    print(f"Home goals MAE: {home_mae:.3f}")
    print(f"Away goals MAE: {away_mae:.3f}")
    print("")
    print("Best daily-points strategy")
    print(f"Goal multiplier: {best.goal_inflation:.2f}")
    print(f"Max total candidate goals: {best.max_total_candidate_goals}")
    print(f"Draw multiplier: {best.draw_probability_multiplier:.2f}")
    print(f"Average points: {best.average_points:.3f}")
    print(f"Exact scores: {int(best.exact_scores)} ({best.exact_score_rate:.1%})")
    print(f"Result accuracy: {best.result_accuracy_rate:.1%}")
    print(f"Misses: {int(best.misses)} ({best.miss_rate:.1%})")
    print("")
    print(f"Saved picks: {picks_output_path}")
    print(f"Saved strategies: {strategies_output_path}")


if __name__ == "__main__":
    main()
