from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_top_scorer_candidates(path: str | Path) -> pd.DataFrame:
    """Load editable top scorer candidate assumptions."""
    candidates = pd.read_csv(path)
    required_columns = {"player", "team", "goal_share", "minutes_share", "penalty_boost"}
    missing_columns = required_columns.difference(candidates.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    return candidates


def estimate_team_tournament_goals(predictions: pd.DataFrame, tournament_simulation: pd.DataFrame) -> pd.DataFrame:
    """Estimate team tournament goals from group xG and advancement probabilities."""
    group_goals: dict[str, float] = {}
    for match in predictions.itertuples(index=False):
        group_goals[match.home_team] = group_goals.get(match.home_team, 0.0) + float(match.home_expected_goals)
        group_goals[match.away_team] = group_goals.get(match.away_team, 0.0) + float(match.away_expected_goals)

    rows = []
    for team, goals in group_goals.items():
        team_row = tournament_simulation[tournament_simulation["team"] == team]
        if team_row.empty:
            knockout_bonus = 0.0
        else:
            row = team_row.iloc[0]
            knockout_bonus = (
                row.round_of_16_probability
                + row.quarterfinal_probability
                + row.semifinal_probability
                + row.final_probability
            )
        rows.append(
            {
                "team": team,
                "group_expected_goals": goals,
                "expected_knockout_matches": knockout_bonus,
                "estimated_team_tournament_goals": goals + knockout_bonus * max(0.8, goals / 3),
            }
        )

    return pd.DataFrame(rows)


def estimate_top_scorers(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    tournament_simulation: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate top scorer candidates from team goals and player assumptions."""
    team_goals = estimate_team_tournament_goals(predictions, tournament_simulation)
    merged = candidates.merge(team_goals, on="team", how="left")
    merged["estimated_team_tournament_goals"] = merged["estimated_team_tournament_goals"].fillna(0.0)
    merged["expected_goals"] = merged["estimated_team_tournament_goals"] * (
        merged["goal_share"] * merged["minutes_share"] + merged["penalty_boost"]
    )
    merged["top_scorer_score"] = merged["expected_goals"]
    return merged.sort_values("top_scorer_score", ascending=False).reset_index(drop=True)
