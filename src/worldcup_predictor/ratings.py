from __future__ import annotations

import csv
from dataclasses import dataclass
from math import exp, log
from pathlib import Path


@dataclass(frozen=True)
class TeamRating:
    team: str
    rating: float
    attack: float = 0.0
    defense: float = 0.0


@dataclass(frozen=True)
class ExpectedGoalsEstimate:
    home_team: str
    away_team: str
    home_expected_goals: float
    away_expected_goals: float


def estimate_expected_goals(
    home: TeamRating,
    away: TeamRating,
    base_goals: float = 1.35,
    rating_weight: float = 0.28,
    home_advantage: float = 0.0,
) -> ExpectedGoalsEstimate:
    """Estimate expected goals from team ratings, attack, and defense.

    The model is intentionally simple and explainable. Attack and defense are
    centered around zero: positive attack improves scoring, positive defense
    reduces the opponent's scoring.
    """
    if base_goals <= 0:
        raise ValueError("base_goals must be greater than 0")

    rating_gap = (home.rating - away.rating) / 400
    base_log_goals = log(base_goals)

    home_log_goals = (
        base_log_goals
        + home.attack
        - away.defense
        + rating_weight * rating_gap
        + home_advantage
    )
    away_log_goals = (
        base_log_goals
        + away.attack
        - home.defense
        - rating_weight * rating_gap
    )

    return ExpectedGoalsEstimate(
        home_team=home.team,
        away_team=away.team,
        home_expected_goals=exp(home_log_goals),
        away_expected_goals=exp(away_log_goals),
    )


def load_team_ratings(path: str | Path) -> dict[str, TeamRating]:
    """Load team ratings from a CSV file."""
    ratings: dict[str, TeamRating] = {}

    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"team", "rating", "attack", "defense"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {sorted(required_columns)}")

        for row in reader:
            team = row["team"].strip()
            if not team:
                raise ValueError("team cannot be empty")

            ratings[team] = TeamRating(
                team=team,
                rating=float(row["rating"]),
                attack=float(row["attack"]),
                defense=float(row["defense"]),
            )

    return ratings


def get_team_rating(ratings: dict[str, TeamRating], team: str) -> TeamRating:
    """Return one team rating with a clear error when the team is missing."""
    try:
        return ratings[team]
    except KeyError as error:
        available = ", ".join(sorted(ratings))
        raise KeyError(f"Team not found: {team}. Available teams: {available}") from error
