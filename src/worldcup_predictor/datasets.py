from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from worldcup_predictor.elo import EloConfig, expected_result, update_ratings


@dataclass(frozen=True)
class MatchRecord:
    goals_for: int
    goals_against: int
    points: int


def load_results(path: str | Path) -> pd.DataFrame:
    """Load raw international match results."""
    results = pd.read_csv(path, parse_dates=["date"])
    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    results = results.dropna(subset=["home_score", "away_score"]).copy()
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results["neutral"] = results["neutral"].astype(bool)
    return results.sort_values("date").reset_index(drop=True)


def _average(records: Iterable[MatchRecord], attribute: str, default: float) -> float:
    values = [getattr(record, attribute) for record in records]
    if not values:
        return default
    return sum(values) / len(values)


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def build_match_features(
    results: pd.DataFrame,
    rolling_window: int = 10,
    min_prior_matches: int = 5,
    start_date: str | None = "1993-01-01",
    elo_config: EloConfig = EloConfig(),
) -> pd.DataFrame:
    """Build pre-match rolling features without using future information."""
    if rolling_window < 1:
        raise ValueError("rolling_window must be greater than or equal to 1")
    if min_prior_matches < 0:
        raise ValueError("min_prior_matches must be greater than or equal to 0")

    sorted_results = results.sort_values("date").reset_index(drop=True)
    histories: dict[str, deque[MatchRecord]] = defaultdict(lambda: deque(maxlen=rolling_window))
    elo_ratings: defaultdict[str, float] = defaultdict(lambda: elo_config.initial_rating)
    rows: list[dict[str, object]] = []
    start_timestamp = pd.Timestamp(start_date) if start_date else None

    for match in sorted_results.itertuples(index=False):
        home_history = histories[match.home_team]
        away_history = histories[match.away_team]
        home_elo = elo_ratings[match.home_team]
        away_elo = elo_ratings[match.away_team]
        home_advantage = 0.0 if bool(match.neutral) else elo_config.home_advantage
        home_elo_expected_result = expected_result(home_elo, away_elo, home_advantage)

        include_match = (
            (start_timestamp is None or match.date >= start_timestamp)
            and len(home_history) >= min_prior_matches
            and len(away_history) >= min_prior_matches
        )

        if include_match:
            rows.append(
                {
                    "date": match.date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "tournament": match.tournament,
                    "neutral": bool(match.neutral),
                    "is_friendly": match.tournament == "Friendly",
                    "year": match.date.year,
                    "home_prior_matches": len(home_history),
                    "away_prior_matches": len(away_history),
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_diff": home_elo - away_elo,
                    "elo_diff_scaled": (home_elo - away_elo) / 400,
                    "home_elo_expected_result": home_elo_expected_result,
                    "home_goals_for_avg": _average(home_history, "goals_for", 1.0),
                    "home_goals_against_avg": _average(home_history, "goals_against", 1.0),
                    "home_points_avg": _average(home_history, "points", 1.0),
                    "away_goals_for_avg": _average(away_history, "goals_for", 1.0),
                    "away_goals_against_avg": _average(away_history, "goals_against", 1.0),
                    "away_points_avg": _average(away_history, "points", 1.0),
                    "home_attack_minus_away_defense": _average(home_history, "goals_for", 1.0)
                    - _average(away_history, "goals_against", 1.0),
                    "away_attack_minus_home_defense": _average(away_history, "goals_for", 1.0)
                    - _average(home_history, "goals_against", 1.0),
                    "home_score": int(match.home_score),
                    "away_score": int(match.away_score),
                    "total_goals": int(match.home_score + match.away_score),
                    "goal_difference": int(match.home_score - match.away_score),
                }
            )

        home_history.append(
            MatchRecord(
                goals_for=int(match.home_score),
                goals_against=int(match.away_score),
                points=_points_for(int(match.home_score), int(match.away_score)),
            )
        )
        away_history.append(
            MatchRecord(
                goals_for=int(match.away_score),
                goals_against=int(match.home_score),
                points=_points_for(int(match.away_score), int(match.home_score)),
            )
        )

        updated_home_elo, updated_away_elo = update_ratings(
            home_rating=home_elo,
            away_rating=away_elo,
            home_score=int(match.home_score),
            away_score=int(match.away_score),
            tournament=match.tournament,
            neutral=bool(match.neutral),
            config=elo_config,
        )
        elo_ratings[match.home_team] = updated_home_elo
        elo_ratings[match.away_team] = updated_away_elo

    return pd.DataFrame(rows)


def save_match_features(features: pd.DataFrame, destination: str | Path) -> Path:
    """Save processed match features to CSV."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)
    return destination
