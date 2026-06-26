from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from math import comb
from pathlib import Path

import pandas as pd

from worldcup_predictor.datasets import MatchRecord
from worldcup_predictor.elo import EloConfig, expected_result, update_ratings
from worldcup_predictor.modeling import BaselineGoalModel, fit_goal_model, predict_expected_goals
from worldcup_predictor.overrides import CONTEXT_MULTIPLIER_COLUMNS, CONTEXT_TEXT_COLUMNS
from worldcup_predictor.poisson import scoreline_probabilities
from worldcup_predictor.recommender import (
    DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    DEFAULT_GOAL_INFLATION,
    DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    adjust_draw_probabilities,
    apply_goal_inflation,
    rank_predictions,
)
from worldcup_predictor.scoring import candidate_scorelines, expected_points


COMPETITIVE_DRAW_PROBABILITY_THRESHOLD = 0.26
COMPETITIVE_DRAW_RESULT_MARGIN = 0.03
MIN_CONTEXT_EXPECTED_GOALS = 0.05
ALREADY_QUALIFIED_MIN_POINTS = 6
ALREADY_QUALIFIED_ADVANCING_SLOTS = 2
ALREADY_QUALIFIED_ATTACK_MULTIPLIER = 0.88
ALREADY_QUALIFIED_DEFENSE_MULTIPLIER = 1.08
ALREADY_QUALIFIED_DRAW_PROBABILITY_MULTIPLIER = 1.08
THIRD_PLACE_ADVANCING_SLOTS_FOR_ELIMINATION = 3
DEFAULT_RISK_PROFILE = "aggressive"
VALID_RISK_PROFILES = {"conservative", "balanced", "aggressive", "desperation"}
MIN_DYNAMIC_GOAL_INFLATION = 0.85
MAX_DYNAMIC_GOAL_INFLATION = 1.50

COMPETITIVE_CONTEXT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "group",
    "is_last_group_match",
    "home_needs_win",
    "away_needs_win",
    "home_draw_is_enough",
    "away_draw_is_enough",
    "home_already_qualified",
    "away_already_qualified",
    "home_eliminated",
    "away_eliminated",
    "home_goal_difference_pressure",
    "away_goal_difference_pressure",
    "match_pressure_score",
    "group_scenario_volatility",
]


@dataclass(frozen=True)
class TeamState:
    team: str
    history: tuple[MatchRecord, ...]
    elo: float


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _average(records: tuple[MatchRecord, ...], attribute: str, default: float = 1.0) -> float:
    if not records:
        return default
    return sum(getattr(record, attribute) for record in records) / len(records)


def _best_draw_alternative(
    probabilities: dict[tuple[int, int], float],
    max_total_candidate_goals: int = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
) -> tuple[tuple[int, int], float]:
    adjusted_probabilities = adjust_draw_probabilities(
        probabilities,
        draw_probability_multiplier=draw_probability_multiplier,
    )
    draw_candidates = tuple(
        scoreline
        for scoreline in candidate_scorelines()
        if scoreline[0] == scoreline[1] and sum(scoreline) <= max_total_candidate_goals
    )
    return max(
        ((candidate, expected_points(candidate, adjusted_probabilities)) for candidate in draw_candidates),
        key=lambda item: item[1],
    )


def _fixture_key(date: object, home_team: str, away_team: str) -> tuple[str, str, str]:
    return (pd.Timestamp(date).strftime("%Y-%m-%d"), home_team, away_team)


def _build_context_override_map(
    context_overrides: pd.DataFrame | None,
) -> dict[tuple[str, str, str], dict[str, object]]:
    if context_overrides is None or context_overrides.empty:
        return {}

    overrides: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in context_overrides.itertuples(index=False):
        key = _fixture_key(row.date, str(row.home_team), str(row.away_team))
        overrides[key] = {
            "home_attack_multiplier": float(getattr(row, "home_attack_multiplier", 1.0)),
            "home_defense_multiplier": float(getattr(row, "home_defense_multiplier", 1.0)),
            "away_attack_multiplier": float(getattr(row, "away_attack_multiplier", 1.0)),
            "away_defense_multiplier": float(getattr(row, "away_defense_multiplier", 1.0)),
            "draw_probability_multiplier": float(getattr(row, "draw_probability_multiplier", 1.0)),
            "confidence": "" if pd.isna(getattr(row, "confidence", "")) else str(getattr(row, "confidence", "")),
            "notes": "" if pd.isna(getattr(row, "notes", "")) else str(getattr(row, "notes", "")),
        }
    return overrides


def _empty_context_overrides() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "home_team",
            "away_team",
            *CONTEXT_MULTIPLIER_COLUMNS,
            *CONTEXT_TEXT_COLUMNS,
        ]
    )


def _combine_context_overrides(*context_frames: pd.DataFrame | None) -> pd.DataFrame:
    combined: dict[tuple[str, str, str], dict[str, object]] = {}

    for frame in context_frames:
        if frame is None or frame.empty:
            continue
        for row in frame.itertuples(index=False):
            key = _fixture_key(row.date, str(row.home_team), str(row.away_team))
            context = combined.setdefault(
                key,
                {
                    "date": row.date,
                    "home_team": str(row.home_team),
                    "away_team": str(row.away_team),
                    "home_attack_multiplier": 1.0,
                    "home_defense_multiplier": 1.0,
                    "away_attack_multiplier": 1.0,
                    "away_defense_multiplier": 1.0,
                    "draw_probability_multiplier": 1.0,
                    "confidence": [],
                    "notes": [],
                },
            )

            for column in CONTEXT_MULTIPLIER_COLUMNS:
                context[column] = float(context[column]) * float(getattr(row, column, 1.0))

            for column in CONTEXT_TEXT_COLUMNS:
                value = getattr(row, column, "")
                if pd.notna(value) and str(value).strip():
                    context[column].append(str(value).strip())

    if not combined:
        return _empty_context_overrides()

    rows: list[dict[str, object]] = []
    for context in combined.values():
        row = dict(context)
        row["confidence"] = "; ".join(row["confidence"])
        row["notes"] = " | ".join(row["notes"])
        rows.append(row)

    columns = ["date", "home_team", "away_team", *CONTEXT_MULTIPLIER_COLUMNS, *CONTEXT_TEXT_COLUMNS]
    return pd.DataFrame(rows).loc[:, columns].sort_values("date").reset_index(drop=True)


def _infer_group_map(matches: pd.DataFrame) -> dict[str, str]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, pd.Timestamp] = {}

    for match in matches.sort_values("date").itertuples(index=False):
        adjacency[str(match.home_team)].add(str(match.away_team))
        adjacency[str(match.away_team)].add(str(match.home_team))
        first_seen.setdefault(str(match.home_team), match.date)
        first_seen.setdefault(str(match.away_team), match.date)

    visited: set[str] = set()
    groups: list[list[str]] = []
    for team in sorted(adjacency, key=lambda item: (first_seen[item], item)):
        if team in visited:
            continue
        stack = [team]
        component: list[str] = []
        visited.add(team)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        groups.append(sorted(component, key=lambda item: (first_seen[item], item)))

    groups.sort(key=lambda group: min(first_seen[team] for team in group))
    return {team: f"Group {index:02d}" for index, group in enumerate(groups, start=1) for team in group}


def _add_standing_result(
    standings: dict[str, dict[str, int]],
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> None:
    standings[home_team]["played"] += 1
    standings[away_team]["played"] += 1
    standings[home_team]["goals_for"] += home_score
    standings[home_team]["goals_against"] += away_score
    standings[home_team]["goal_difference"] += home_score - away_score
    standings[away_team]["goals_for"] += away_score
    standings[away_team]["goals_against"] += home_score
    standings[away_team]["goal_difference"] += away_score - home_score
    if home_score > away_score:
        standings[home_team]["points"] += 3
    elif home_score < away_score:
        standings[away_team]["points"] += 3
    else:
        standings[home_team]["points"] += 1
        standings[away_team]["points"] += 1


def _empty_standing_record() -> dict[str, int]:
    return {"played": 0, "points": 0, "goals_for": 0, "goals_against": 0, "goal_difference": 0}


def _has_clinched_advancing_slot(
    team: str,
    group_teams: set[str],
    standings: dict[str, dict[str, int]],
    remaining_matches: list[tuple[str, str]],
    advancing_slots: int,
) -> bool:
    current_points = {candidate: standings[candidate]["points"] for candidate in group_teams}
    outcomes = ((3, 0), (1, 1), (0, 3))

    for scenario in product(outcomes, repeat=len(remaining_matches)):
        scenario_points = current_points.copy()
        for (home_team, away_team), (home_points, away_points) in zip(remaining_matches, scenario):
            scenario_points[home_team] += home_points
            scenario_points[away_team] += away_points

        team_points = scenario_points[team]
        teams_level_or_above = sum(points >= team_points for points in scenario_points.values())
        if teams_level_or_above > advancing_slots:
            return False

    return True


def _can_reach_advancing_slot(
    team: str,
    group_teams: set[str],
    standings: dict[str, dict[str, int]],
    remaining_matches: list[tuple[str, str]],
    advancing_slots: int,
) -> bool:
    current_points = {candidate: standings[candidate]["points"] for candidate in group_teams}
    outcomes = ((3, 0), (1, 1), (0, 3))

    for scenario in product(outcomes, repeat=len(remaining_matches)):
        scenario_points = current_points.copy()
        for (home_team, away_team), (home_points, away_points) in zip(remaining_matches, scenario):
            scenario_points[home_team] += home_points
            scenario_points[away_team] += away_points

        team_points = scenario_points[team]
        teams_strictly_above = sum(points > team_points for points in scenario_points.values())
        if teams_strictly_above < advancing_slots:
            return True

    return False


def _with_current_match_points(
    standings: dict[str, dict[str, int]],
    home_team: str,
    away_team: str,
    home_points: int,
    away_points: int,
) -> dict[str, dict[str, int]]:
    adjusted = {team: dict(record) for team, record in standings.items()}
    adjusted[home_team]["points"] += home_points
    adjusted[away_team]["points"] += away_points
    return adjusted


def _remaining_after_current_match(
    remaining_matches: list[tuple[str, str]],
    current_match: tuple[str, str],
) -> list[tuple[str, str]]:
    remaining = list(remaining_matches)
    try:
        remaining.remove(current_match)
    except ValueError:
        return remaining
    return remaining


def _result_clinches_advancing_slot(
    team: str,
    group_teams: set[str],
    standings: dict[str, dict[str, int]],
    current_match: tuple[str, str],
    remaining_matches: list[tuple[str, str]],
    result_points: tuple[int, int],
    advancing_slots: int,
) -> bool:
    home_team, away_team = current_match
    adjusted = _with_current_match_points(
        standings,
        home_team,
        away_team,
        result_points[0],
        result_points[1],
    )
    return _has_clinched_advancing_slot(
        team,
        group_teams,
        adjusted,
        _remaining_after_current_match(remaining_matches, current_match),
        advancing_slots,
    )


def _result_can_reach_slot(
    team: str,
    group_teams: set[str],
    standings: dict[str, dict[str, int]],
    current_match: tuple[str, str],
    remaining_matches: list[tuple[str, str]],
    result_points: tuple[int, int],
    advancing_slots: int,
) -> bool:
    home_team, away_team = current_match
    adjusted = _with_current_match_points(
        standings,
        home_team,
        away_team,
        result_points[0],
        result_points[1],
    )
    return _can_reach_advancing_slot(
        team,
        group_teams,
        adjusted,
        _remaining_after_current_match(remaining_matches, current_match),
        advancing_slots,
    )


def _group_scenario_volatility(
    group_teams: set[str],
    standings: dict[str, dict[str, int]],
    remaining_matches: list[tuple[str, str]],
    advancing_slots: int,
) -> float:
    if len(group_teams) <= advancing_slots:
        return 0.0

    current_points = {team: standings[team]["points"] for team in group_teams}
    outcomes = ((3, 0), (1, 1), (0, 3))
    top_sets: set[tuple[str, ...]] = set()

    for scenario in product(outcomes, repeat=len(remaining_matches)):
        scenario_points = current_points.copy()
        for (home_team, away_team), (home_points, away_points) in zip(remaining_matches, scenario):
            scenario_points[home_team] += home_points
            scenario_points[away_team] += away_points
        ranked = sorted(
            group_teams,
            key=lambda team: (
                scenario_points[team],
                standings[team]["goal_difference"],
                standings[team]["goals_for"],
                team,
            ),
            reverse=True,
        )
        top_sets.add(tuple(sorted(ranked[:advancing_slots])))

    max_sets = max(1, comb(len(group_teams), advancing_slots))
    if max_sets == 1:
        return 0.0
    return min(1.0, (len(top_sets) - 1) / (max_sets - 1))


def _team_goal_difference_pressure(
    team: str,
    group_teams: set[str],
    standings: dict[str, dict[str, int]],
    already_qualified: bool,
    eliminated: bool,
) -> bool:
    if already_qualified or eliminated:
        return False

    team_points = standings[team]["points"]
    nearby_teams = [
        other
        for other in group_teams
        if other != team and abs(standings[other]["points"] - team_points) <= 3
    ]
    tied_or_near_cutoff = len(nearby_teams) >= 2
    weak_tiebreaker = standings[team]["goal_difference"] <= 1
    return tied_or_near_cutoff and weak_tiebreaker


def _match_pressure_score(context: dict[str, object]) -> float:
    pressure = 0.0
    pressure += 0.28 if bool(context["home_needs_win"]) else 0.0
    pressure += 0.28 if bool(context["away_needs_win"]) else 0.0
    pressure += 0.14 if bool(context["home_goal_difference_pressure"]) else 0.0
    pressure += 0.14 if bool(context["away_goal_difference_pressure"]) else 0.0
    pressure += 0.16 * float(context["group_scenario_volatility"])
    if bool(context["home_already_qualified"]) and bool(context["away_already_qualified"]):
        pressure -= 0.40
    if bool(context["home_draw_is_enough"]) or bool(context["away_draw_is_enough"]):
        pressure -= 0.18
    if bool(context["home_eliminated"]) and bool(context["away_eliminated"]):
        pressure -= 0.24
    return min(1.0, max(0.0, pressure))


def _empty_competitive_context() -> pd.DataFrame:
    return pd.DataFrame(columns=COMPETITIVE_CONTEXT_COLUMNS)


def _build_competitive_context_map(
    competitive_context: pd.DataFrame | None,
) -> dict[tuple[str, str, str], dict[str, object]]:
    if competitive_context is None or competitive_context.empty:
        return {}

    contexts: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in competitive_context.itertuples(index=False):
        key = _fixture_key(row.date, str(row.home_team), str(row.away_team))
        contexts[key] = {column: getattr(row, column) for column in COMPETITIVE_CONTEXT_COLUMNS}
    return contexts


def _default_competitive_context(date: object, home_team: str, away_team: str) -> dict[str, object]:
    return {
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "group": "",
        "is_last_group_match": False,
        "home_needs_win": False,
        "away_needs_win": False,
        "home_draw_is_enough": False,
        "away_draw_is_enough": False,
        "home_already_qualified": False,
        "away_already_qualified": False,
        "home_eliminated": False,
        "away_eliminated": False,
        "home_goal_difference_pressure": False,
        "away_goal_difference_pressure": False,
        "match_pressure_score": 0.0,
        "group_scenario_volatility": 0.0,
    }


def build_competitive_context(
    completed_results: pd.DataFrame,
    fixtures: pd.DataFrame,
    tournament: str = "FIFA World Cup",
    start_date: str = "2026-06-01",
    advancing_slots: int = ALREADY_QUALIFIED_ADVANCING_SLOTS,
) -> pd.DataFrame:
    """Infer group incentives before each pending fixture."""
    if fixtures.empty:
        return _empty_competitive_context()

    completed = completed_results[
        (completed_results["tournament"] == tournament)
        & (completed_results["date"] >= pd.Timestamp(start_date))
    ].copy()
    pending = fixtures[
        (fixtures["tournament"] == tournament)
        & (fixtures["date"] >= pd.Timestamp(start_date))
    ].copy()
    if pending.empty:
        return _empty_competitive_context()

    columns = ["date", "home_team", "away_team", "tournament", "home_score", "away_score"]
    schedule = pd.concat([completed.loc[:, columns], pending.loc[:, columns]], ignore_index=True)
    group_by_team = _infer_group_map(schedule)
    teams_by_group: defaultdict[str, set[str]] = defaultdict(set)
    remaining_matches_by_group: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    total_matches_by_team: defaultdict[str, int] = defaultdict(int)

    for match in schedule.itertuples(index=False):
        home_team = str(match.home_team)
        away_team = str(match.away_team)
        group = group_by_team.get(home_team)
        if group is None or group != group_by_team.get(away_team):
            continue
        teams_by_group[group].update((home_team, away_team))
        total_matches_by_team[home_team] += 1
        total_matches_by_team[away_team] += 1

    for match in pending.itertuples(index=False):
        home_team = str(match.home_team)
        away_team = str(match.away_team)
        group = group_by_team.get(home_team)
        if group is None or group != group_by_team.get(away_team):
            continue
        remaining_matches_by_group[group].append((home_team, away_team))

    standings: defaultdict[str, dict[str, int]] = defaultdict(_empty_standing_record)
    for match in completed.sort_values("date").itertuples(index=False):
        if group_by_team.get(str(match.home_team)) != group_by_team.get(str(match.away_team)):
            continue
        _add_standing_result(
            standings,
            str(match.home_team),
            str(match.away_team),
            int(match.home_score),
            int(match.away_score),
        )

    rows: list[dict[str, object]] = []
    for match in pending.sort_values("date").itertuples(index=False):
        home_team = str(match.home_team)
        away_team = str(match.away_team)
        current_match = (home_team, away_team)
        group = group_by_team.get(home_team, "")
        group_teams = teams_by_group[group]
        remaining_matches = remaining_matches_by_group[group]
        volatility = _group_scenario_volatility(group_teams, standings, remaining_matches, advancing_slots)

        row = _default_competitive_context(match.date, home_team, away_team)
        row["group"] = group
        row["is_last_group_match"] = (
            standings[home_team]["played"] == total_matches_by_team[home_team] - 1
            and standings[away_team]["played"] == total_matches_by_team[away_team] - 1
        )
        row["group_scenario_volatility"] = volatility

        for side, team, draw_points, loss_points in (
            ("home", home_team, (1, 1), (0, 3)),
            ("away", away_team, (1, 1), (3, 0)),
        ):
            already_qualified = _has_clinched_advancing_slot(
                team,
                group_teams,
                standings,
                remaining_matches,
                advancing_slots,
            )
            eliminated = not _can_reach_advancing_slot(
                team,
                group_teams,
                standings,
                remaining_matches,
                THIRD_PLACE_ADVANCING_SLOTS_FOR_ELIMINATION,
            )
            draw_is_enough = False
            if not already_qualified and not eliminated:
                draw_is_enough = _result_clinches_advancing_slot(
                    team,
                    group_teams,
                    standings,
                    current_match,
                    remaining_matches,
                    draw_points,
                    advancing_slots,
                )
            can_reach_after_loss = _result_can_reach_slot(
                team,
                group_teams,
                standings,
                current_match,
                remaining_matches,
                loss_points,
                THIRD_PLACE_ADVANCING_SLOTS_FOR_ELIMINATION,
            )
            needs_win = (
                bool(row["is_last_group_match"])
                and not already_qualified
                and not eliminated
                and not draw_is_enough
                and not can_reach_after_loss
            )
            goal_difference_pressure = _team_goal_difference_pressure(
                team,
                group_teams,
                standings,
                already_qualified,
                eliminated,
            ) or (bool(row["is_last_group_match"]) and not already_qualified and not eliminated and not draw_is_enough)

            row[f"{side}_needs_win"] = needs_win
            row[f"{side}_draw_is_enough"] = draw_is_enough
            row[f"{side}_already_qualified"] = already_qualified
            row[f"{side}_eliminated"] = eliminated
            row[f"{side}_goal_difference_pressure"] = goal_difference_pressure

        row["match_pressure_score"] = _match_pressure_score(row)
        rows.append(row)

    if not rows:
        return _empty_competitive_context()
    return pd.DataFrame(rows).loc[:, COMPETITIVE_CONTEXT_COLUMNS].reset_index(drop=True)


def _dynamic_goal_inflation(
    base_goal_inflation: float,
    context: dict[str, object],
    elo_diff: float,
) -> tuple[float, str]:
    value = base_goal_inflation
    reasons: list[str] = []

    both_already_qualified = bool(context["home_already_qualified"]) and bool(context["away_already_qualified"])
    both_eliminated = bool(context["home_eliminated"]) and bool(context["away_eliminated"])
    any_draw_is_enough = bool(context["home_draw_is_enough"]) or bool(context["away_draw_is_enough"])
    both_need_win = bool(context["home_needs_win"]) and bool(context["away_needs_win"])
    any_need_win = bool(context["home_needs_win"]) or bool(context["away_needs_win"])
    any_goal_difference_pressure = bool(context["home_goal_difference_pressure"]) or bool(
        context["away_goal_difference_pressure"]
    )
    volatility = float(context["group_scenario_volatility"])
    strong_favorite = abs(elo_diff) >= 250

    closed_context = both_already_qualified or any_draw_is_enough or both_eliminated
    if both_already_qualified:
        value = min(value, 1.05)
        reasons.append("both qualified: lower tempo")
    elif both_eliminated:
        value = min(value, 1.12)
        reasons.append("both eliminated: lower competitive edge")
    elif any_draw_is_enough:
        value = min(value, 1.10)
        reasons.append("draw useful: closed incentives")

    if both_need_win and not closed_context:
        value = max(value, 1.45)
        reasons.append("both need win")
    elif any_need_win and not closed_context:
        value = max(value, 1.32)
        reasons.append("one side needs win")

    if any_goal_difference_pressure and not closed_context:
        value = max(value, 1.35)
        reasons.append("goal-difference pressure")
    if volatility >= 0.65 and not closed_context:
        value = max(value, 1.35)
        reasons.append("volatile group scenario")
    if strong_favorite and not closed_context:
        value = max(value, 1.30)
        reasons.append("strong favorite mismatch")
    if bool(context["is_last_group_match"]) and float(context["match_pressure_score"]) <= 0.20:
        value = min(value, 1.15)
        reasons.append("low-pressure final group match")

    value = min(MAX_DYNAMIC_GOAL_INFLATION, max(MIN_DYNAMIC_GOAL_INFLATION, value))
    if not reasons:
        reasons.append("baseline live scoring pace")
    return value, "; ".join(reasons)


def _scoreline_result(scoreline: tuple[int, int]) -> str:
    if scoreline[0] > scoreline[1]:
        return "home"
    if scoreline[0] < scoreline[1]:
        return "away"
    return "draw"


def _scoreline_text(scoreline: tuple[int, int]) -> str:
    return f"{scoreline[0]}-{scoreline[1]}"


def _result_probabilities_from_scorelines(probabilities: dict[tuple[int, int], float]) -> dict[str, float]:
    return {
        "home": sum(probability for scoreline, probability in probabilities.items() if _scoreline_result(scoreline) == "home"),
        "draw": sum(probability for scoreline, probability in probabilities.items() if _scoreline_result(scoreline) == "draw"),
        "away": sum(probability for scoreline, probability in probabilities.items() if _scoreline_result(scoreline) == "away"),
    }


def _estimate_pick_popularity(
    scoreline: tuple[int, int],
    adjusted_probabilities: dict[tuple[int, int], float],
    result_probabilities: dict[str, float],
) -> float:
    common_scoreline_prior = {
        (1, 0): 0.78,
        (2, 0): 0.72,
        (2, 1): 0.70,
        (0, 1): 0.66,
        (0, 2): 0.62,
        (1, 2): 0.62,
        (1, 1): 0.68,
        (0, 0): 0.40,
    }
    max_scoreline_probability = max(adjusted_probabilities.values())
    relative_scoreline_probability = adjusted_probabilities.get(scoreline, 0.0) / max_scoreline_probability
    result = _scoreline_result(scoreline)
    favorite_result_probability = max(result_probabilities.values())
    result_component = result_probabilities[result]
    if result_probabilities[result] < favorite_result_probability:
        result_component *= 0.70
    total_goals = sum(scoreline)
    high_total_penalty = max(0.35, 1.0 - 0.08 * max(0, total_goals - 2))
    popularity = (
        0.45 * relative_scoreline_probability
        + 0.35 * result_component
        + 0.20 * common_scoreline_prior.get(scoreline, 0.24)
    ) * high_total_penalty
    return min(0.90, max(0.05, popularity))


def _upside_multiplier(scoreline: tuple[int, int], result_probabilities: dict[str, float]) -> float:
    total_goals = sum(scoreline)
    result = _scoreline_result(scoreline)
    favorite_result = max(result_probabilities, key=result_probabilities.get)
    underdog_bonus = 0.12 if result != favorite_result else 0.0
    draw_bonus = 0.08 if result == "draw" and result_probabilities["draw"] >= 0.22 else 0.0
    value = 1.0 + 0.06 * total_goals + 0.04 * abs(scoreline[0] - scoreline[1]) + underdog_bonus + draw_bonus
    return min(1.55, max(1.0, value))


def _pick_metric(
    scoreline: tuple[int, int],
    adjusted_probabilities: dict[tuple[int, int], float],
    result_probabilities: dict[str, float],
) -> dict[str, float]:
    pick_expected_points = expected_points(scoreline, adjusted_probabilities)
    estimated_popularity = _estimate_pick_popularity(scoreline, adjusted_probabilities, result_probabilities)
    differential_multiplier = min(2.00, 1.0 + (1.0 - estimated_popularity) * 0.85)
    upside_multiplier = _upside_multiplier(scoreline, result_probabilities)
    return {
        "expected_points": pick_expected_points,
        "estimated_pick_popularity": estimated_popularity,
        "differential_multiplier": differential_multiplier,
        "upside_multiplier": upside_multiplier,
        "strategic_pick_value": pick_expected_points * differential_multiplier * upside_multiplier,
    }


def _candidate_pool(
    max_candidate_goals: int = 6,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
) -> tuple[tuple[int, int], ...]:
    candidates = candidate_scorelines(max_candidate_goals)
    if max_total_candidate_goals is None:
        return candidates
    return tuple(candidate for candidate in candidates if sum(candidate) <= max_total_candidate_goals)


def _risk_profile_picks(
    probabilities: dict[tuple[int, int], float],
    max_total_candidate_goals: int | None,
    draw_probability_multiplier: float,
    closed_context: bool = False,
) -> dict[str, dict[str, object]]:
    adjusted_probabilities = adjust_draw_probabilities(
        probabilities,
        draw_probability_multiplier=draw_probability_multiplier,
    )
    result_probabilities = _result_probabilities_from_scorelines(adjusted_probabilities)
    favorite_result = max(result_probabilities, key=result_probabilities.get)

    conservative_candidates = [
        candidate
        for candidate in _candidate_pool(max_total_candidate_goals=max_total_candidate_goals)
        if _scoreline_result(candidate) == favorite_result
    ]
    conservative_pick = max(
        conservative_candidates,
        key=lambda candidate: adjusted_probabilities.get(candidate, 0.0),
    )
    balanced_pick, _balanced_value = rank_predictions(
        probabilities,
        max_total_candidate_goals=max_total_candidate_goals,
        draw_probability_multiplier=draw_probability_multiplier,
        limit=1,
    )[0]

    aggressive_candidates = _candidate_pool(
        max_total_candidate_goals=max_total_candidate_goals if closed_context else 5
    )
    desperation_candidates = _candidate_pool(max_total_candidate_goals=5 if closed_context else None)
    aggressive_pick = max(
        aggressive_candidates,
        key=lambda candidate: (
            _pick_metric(candidate, adjusted_probabilities, result_probabilities)["expected_points"]
            * _pick_metric(candidate, adjusted_probabilities, result_probabilities)["upside_multiplier"]
            * (_pick_metric(candidate, adjusted_probabilities, result_probabilities)["differential_multiplier"] ** 0.45)
        ),
    )
    desperation_pick = max(
        desperation_candidates,
        key=lambda candidate: _pick_metric(candidate, adjusted_probabilities, result_probabilities)["strategic_pick_value"],
    )

    return {
        "conservative": {"scoreline": conservative_pick, **_pick_metric(conservative_pick, adjusted_probabilities, result_probabilities)},
        "balanced": {"scoreline": balanced_pick, **_pick_metric(balanced_pick, adjusted_probabilities, result_probabilities)},
        "aggressive": {"scoreline": aggressive_pick, **_pick_metric(aggressive_pick, adjusted_probabilities, result_probabilities)},
        "desperation": {"scoreline": desperation_pick, **_pick_metric(desperation_pick, adjusted_probabilities, result_probabilities)},
    }


def _confidence_level(result_probabilities: dict[str, float], context: dict[str, object]) -> str:
    favorite_probability = max(result_probabilities.values())
    volatility = float(context["group_scenario_volatility"])
    if favorite_probability >= 0.66 and volatility <= 0.35:
        return "high"
    if favorite_probability >= 0.52 and volatility <= 0.65:
        return "medium"
    return "low"


def _error_risk(result_probabilities: dict[str, float], context: dict[str, object]) -> str:
    favorite_probability = max(result_probabilities.values())
    if float(context["group_scenario_volatility"]) >= 0.70 or favorite_probability < 0.45:
        return "high"
    if favorite_probability < 0.58 or bool(context["home_draw_is_enough"]) or bool(context["away_draw_is_enough"]):
        return "medium"
    return "low"


def _context_reasons(context: dict[str, object], dynamic_goal_inflation_reason: str) -> tuple[str, str]:
    reasons = [dynamic_goal_inflation_reason]
    alerts: list[str] = []
    if bool(context["home_already_qualified"]) and bool(context["away_already_qualified"]):
        alerts.append("both_already_qualified")
    elif bool(context["home_already_qualified"]) or bool(context["away_already_qualified"]):
        alerts.append("one_team_already_qualified")
    if bool(context["home_draw_is_enough"]) or bool(context["away_draw_is_enough"]):
        alerts.append("draw_enough_for_one_team")
    if bool(context["home_goal_difference_pressure"]) or bool(context["away_goal_difference_pressure"]):
        alerts.append("goal_difference_pressure")
    if float(context["group_scenario_volatility"]) >= 0.65:
        alerts.append("high_group_volatility")
    if bool(context["home_eliminated"]) or bool(context["away_eliminated"]):
        alerts.append("elimination_context")
    return " | ".join(reasons), ",".join(alerts)


def build_already_qualified_context_overrides(
    completed_results: pd.DataFrame,
    fixtures: pd.DataFrame,
    tournament: str = "FIFA World Cup",
    start_date: str = "2026-06-01",
    min_points: int = ALREADY_QUALIFIED_MIN_POINTS,
    advancing_slots: int = ALREADY_QUALIFIED_ADVANCING_SLOTS,
) -> pd.DataFrame:
    """Derive passive/rotation context for teams safe before their last group match."""
    if fixtures.empty:
        return _empty_context_overrides()

    completed = completed_results[
        (completed_results["tournament"] == tournament)
        & (completed_results["date"] >= pd.Timestamp(start_date))
    ].copy()
    pending = fixtures[
        (fixtures["tournament"] == tournament)
        & (fixtures["date"] >= pd.Timestamp(start_date))
    ].copy()
    if pending.empty:
        return _empty_context_overrides()

    columns = ["date", "home_team", "away_team", "tournament", "home_score", "away_score"]
    schedule = pd.concat([completed.loc[:, columns], pending.loc[:, columns]], ignore_index=True)
    group_by_team = _infer_group_map(schedule)
    teams_by_group: defaultdict[str, set[str]] = defaultdict(set)
    remaining_matches_by_group: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    total_matches_by_team: defaultdict[str, int] = defaultdict(int)
    for match in schedule.itertuples(index=False):
        home_team = str(match.home_team)
        away_team = str(match.away_team)
        group = group_by_team.get(home_team)
        if group is None or group != group_by_team.get(away_team):
            continue
        teams_by_group[group].update((home_team, away_team))
        total_matches_by_team[home_team] += 1
        total_matches_by_team[away_team] += 1

    for match in pending.itertuples(index=False):
        home_team = str(match.home_team)
        away_team = str(match.away_team)
        group = group_by_team.get(home_team)
        if group is None or group != group_by_team.get(away_team):
            continue
        remaining_matches_by_group[group].append((home_team, away_team))

    standings: defaultdict[str, dict[str, int]] = defaultdict(_empty_standing_record)
    for match in completed.sort_values("date").itertuples(index=False):
        if group_by_team.get(str(match.home_team)) != group_by_team.get(str(match.away_team)):
            continue
        _add_standing_result(
            standings,
            str(match.home_team),
            str(match.away_team),
            int(match.home_score),
            int(match.away_score),
        )

    rows: list[dict[str, object]] = []
    for match in pending.sort_values("date").itertuples(index=False):
        home_team = str(match.home_team)
        away_team = str(match.away_team)
        qualified_teams: list[str] = []
        row = {
            "date": match.date,
            "home_team": home_team,
            "away_team": away_team,
            "home_attack_multiplier": 1.0,
            "home_defense_multiplier": 1.0,
            "away_attack_multiplier": 1.0,
            "away_defense_multiplier": 1.0,
            "draw_probability_multiplier": 1.0,
            "confidence": "medium",
            "notes": "",
        }

        for side, team in (("home", home_team), ("away", away_team)):
            team_standing = standings[team]
            group = group_by_team.get(team)
            is_last_group_match = team_standing["played"] == total_matches_by_team[team] - 1
            has_clinched = (
                group is not None
                and team_standing["points"] >= min_points
                and _has_clinched_advancing_slot(
                    team,
                    teams_by_group[group],
                    standings,
                    remaining_matches_by_group[group],
                    advancing_slots,
                )
            )
            if is_last_group_match and has_clinched:
                row[f"{side}_attack_multiplier"] = ALREADY_QUALIFIED_ATTACK_MULTIPLIER
                row[f"{side}_defense_multiplier"] = ALREADY_QUALIFIED_DEFENSE_MULTIPLIER
                row["draw_probability_multiplier"] = ALREADY_QUALIFIED_DRAW_PROBABILITY_MULTIPLIER
                qualified_teams.append(f"{team} ({team_standing['points']} pts)")

        if qualified_teams:
            row["notes"] = (
                "Automatic qualified-team context: "
                f"{', '.join(qualified_teams)} before final group match; "
                "rotation and passive tempo expected."
            )
            rows.append(row)

    if not rows:
        return _empty_context_overrides()
    columns = ["date", "home_team", "away_team", *CONTEXT_MULTIPLIER_COLUMNS, *CONTEXT_TEXT_COLUMNS]
    return pd.DataFrame(rows).loc[:, columns].reset_index(drop=True)


def _context_adjustment_for(
    context_overrides: dict[tuple[str, str, str], dict[str, object]],
    date: object,
    home_team: str,
    away_team: str,
) -> dict[str, object]:
    return context_overrides.get(
        _fixture_key(date, home_team, away_team),
        {
            "home_attack_multiplier": 1.0,
            "home_defense_multiplier": 1.0,
            "away_attack_multiplier": 1.0,
            "away_defense_multiplier": 1.0,
            "draw_probability_multiplier": 1.0,
            "confidence": "",
            "notes": "",
        },
    )


def load_raw_results_with_fixtures(path: str | Path) -> pd.DataFrame:
    """Load raw results including future fixtures with missing scores."""
    results = pd.read_csv(path, parse_dates=["date"])
    results["neutral"] = results["neutral"].astype(bool)
    return results.sort_values("date").reset_index(drop=True)


def completed_matches(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Return only matches with known final scores."""
    completed = raw_results.dropna(subset=["home_score", "away_score"]).copy()
    completed["home_score"] = completed["home_score"].astype(int)
    completed["away_score"] = completed["away_score"].astype(int)
    return completed.sort_values("date").reset_index(drop=True)


def future_fixtures(
    raw_results: pd.DataFrame,
    tournament: str | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """Return fixtures with missing scores, optionally filtered by tournament/date."""
    fixtures = raw_results[
        raw_results["home_score"].isna() | raw_results["away_score"].isna()
    ].copy()
    if tournament is not None:
        fixtures = fixtures[fixtures["tournament"] == tournament]
    if start_date is not None:
        fixtures = fixtures[fixtures["date"] >= pd.Timestamp(start_date)]
    return fixtures.sort_values("date").reset_index(drop=True)


def build_team_states(
    results: pd.DataFrame,
    rolling_window: int = 10,
    elo_config: EloConfig = EloConfig(),
) -> dict[str, TeamState]:
    """Build latest rolling and Elo state for each team from completed results."""
    histories: dict[str, deque[MatchRecord]] = defaultdict(lambda: deque(maxlen=rolling_window))
    elo_ratings: defaultdict[str, float] = defaultdict(lambda: elo_config.initial_rating)

    for match in results.sort_values("date").itertuples(index=False):
        home_elo = elo_ratings[match.home_team]
        away_elo = elo_ratings[match.away_team]

        histories[match.home_team].append(
            MatchRecord(
                goals_for=int(match.home_score),
                goals_against=int(match.away_score),
                points=_points_for(int(match.home_score), int(match.away_score)),
            )
        )
        histories[match.away_team].append(
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

    teams = set(histories) | set(elo_ratings)
    return {
        team: TeamState(
            team=team,
            history=tuple(histories[team]),
            elo=elo_ratings[team],
        )
        for team in teams
    }


def build_fixture_features(
    fixtures: pd.DataFrame,
    team_states: dict[str, TeamState],
    elo_config: EloConfig = EloConfig(),
) -> pd.DataFrame:
    """Convert future fixtures into model-ready features."""
    rows: list[dict[str, object]] = []

    for fixture in fixtures.itertuples(index=False):
        home_state = team_states.get(
            fixture.home_team,
            TeamState(fixture.home_team, tuple(), elo_config.initial_rating),
        )
        away_state = team_states.get(
            fixture.away_team,
            TeamState(fixture.away_team, tuple(), elo_config.initial_rating),
        )
        home_advantage = 0.0 if bool(fixture.neutral) else elo_config.home_advantage
        home_elo_expected_result = expected_result(home_state.elo, away_state.elo, home_advantage)

        rows.append(
            {
                "date": fixture.date,
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "tournament": fixture.tournament,
                "neutral": bool(fixture.neutral),
                "is_friendly": fixture.tournament == "Friendly",
                "year": fixture.date.year,
                "home_prior_matches": len(home_state.history),
                "away_prior_matches": len(away_state.history),
                "home_elo": home_state.elo,
                "away_elo": away_state.elo,
                "elo_diff": home_state.elo - away_state.elo,
                "elo_diff_scaled": (home_state.elo - away_state.elo) / 400,
                "home_elo_expected_result": home_elo_expected_result,
                "home_goals_for_avg": _average(home_state.history, "goals_for"),
                "home_goals_against_avg": _average(home_state.history, "goals_against"),
                "home_points_avg": _average(home_state.history, "points"),
                "away_goals_for_avg": _average(away_state.history, "goals_for"),
                "away_goals_against_avg": _average(away_state.history, "goals_against"),
                "away_points_avg": _average(away_state.history, "points"),
                "home_attack_minus_away_defense": _average(home_state.history, "goals_for")
                - _average(away_state.history, "goals_against"),
                "away_attack_minus_home_defense": _average(away_state.history, "goals_for")
                - _average(home_state.history, "goals_against"),
            }
        )

    return pd.DataFrame(rows)


def predict_fixture_picks(
    model: BaselineGoalModel,
    fixture_features: pd.DataFrame,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    goal_inflation: float = DEFAULT_GOAL_INFLATION,
    context_overrides: pd.DataFrame | None = None,
    competitive_context: pd.DataFrame | None = None,
    dynamic_goal_inflation_enabled: bool = True,
    risk_profile: str = DEFAULT_RISK_PROFILE,
) -> pd.DataFrame:
    """Predict expected goals and recommended picks for future fixtures."""
    if risk_profile not in VALID_RISK_PROFILES:
        raise ValueError(f"risk_profile must be one of {sorted(VALID_RISK_PROFILES)}")

    predictions = predict_expected_goals(model, fixture_features)
    context_by_fixture = _build_context_override_map(context_overrides)
    competitive_context_by_fixture = _build_competitive_context_map(competitive_context)
    rows: list[dict[str, object]] = []

    for match in predictions.itertuples(index=False):
        key = _fixture_key(match.date, str(match.home_team), str(match.away_team))
        context = _context_adjustment_for(
            context_by_fixture,
            match.date,
            str(match.home_team),
            str(match.away_team),
        )
        competitive = competitive_context_by_fixture.get(
            key,
            _default_competitive_context(match.date, str(match.home_team), str(match.away_team)),
        )
        competitive_context_applied = key in competitive_context_by_fixture
        context_applied = key in context_by_fixture
        model_home_expected_goals = float(match.home_expected_goals)
        model_away_expected_goals = float(match.away_expected_goals)
        context_home_expected_goals = max(
            MIN_CONTEXT_EXPECTED_GOALS,
            model_home_expected_goals
            * float(context["home_attack_multiplier"])
            * float(context["away_defense_multiplier"]),
        )
        context_away_expected_goals = max(
            MIN_CONTEXT_EXPECTED_GOALS,
            model_away_expected_goals
            * float(context["away_attack_multiplier"])
            * float(context["home_defense_multiplier"]),
        )
        effective_draw_probability_multiplier = (
            draw_probability_multiplier * float(context["draw_probability_multiplier"])
        )
        dynamic_goal_inflation, dynamic_goal_inflation_reason = (
            _dynamic_goal_inflation(goal_inflation, competitive, float(match.elo_diff))
            if dynamic_goal_inflation_enabled and competitive_context_applied
            else (goal_inflation, "fixed goal inflation")
        )
        strategy_home_expected_goals, strategy_away_expected_goals = apply_goal_inflation(
            context_home_expected_goals,
            context_away_expected_goals,
            goal_inflation=dynamic_goal_inflation,
        )
        market_adjusted_home_expected_goals = strategy_home_expected_goals
        market_adjusted_away_expected_goals = strategy_away_expected_goals
        probabilities = scoreline_probabilities(
            home_expected_goals=market_adjusted_home_expected_goals,
            away_expected_goals=market_adjusted_away_expected_goals,
            max_goals=6,
        )
        closed_pick_context = (
            (bool(competitive["home_already_qualified"]) and bool(competitive["away_already_qualified"]))
            or bool(competitive["home_draw_is_enough"])
            or bool(competitive["away_draw_is_enough"])
            or (bool(competitive["home_eliminated"]) and bool(competitive["away_eliminated"]))
        )
        risk_picks = _risk_profile_picks(
            probabilities,
            max_total_candidate_goals=max_total_candidate_goals,
            draw_probability_multiplier=effective_draw_probability_multiplier,
            closed_context=closed_pick_context,
        )
        selected_pick = risk_picks[risk_profile]
        pick = selected_pick["scoreline"]
        pick_expected_points = float(selected_pick["expected_points"])
        home_win_probability = sum(
            probability
            for (home_goals, away_goals), probability in probabilities.items()
            if home_goals > away_goals
        )
        draw_probability = sum(
            probability
            for (home_goals, away_goals), probability in probabilities.items()
            if home_goals == away_goals
        )
        away_win_probability = sum(
            probability
            for (home_goals, away_goals), probability in probabilities.items()
            if home_goals < away_goals
        )
        draw_pick, draw_expected_points = _best_draw_alternative(
            probabilities,
            draw_probability_multiplier=effective_draw_probability_multiplier,
        )
        draw_is_competitive = (
            draw_probability >= COMPETITIVE_DRAW_PROBABILITY_THRESHOLD
            and abs(home_win_probability - away_win_probability) <= COMPETITIVE_DRAW_RESULT_MARGIN
        )
        result_probabilities = {
            "home": home_win_probability,
            "draw": draw_probability,
            "away": away_win_probability,
        }
        confidence_level = _confidence_level(result_probabilities, competitive)
        error_risk = _error_risk(result_probabilities, competitive)
        main_reasons, alert_flags = _context_reasons(competitive, dynamic_goal_inflation_reason)

        rows.append(
            {
                "date": match.date,
                "tournament": match.tournament,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "group": str(competitive["group"]),
                "model_home_expected_goals": model_home_expected_goals,
                "model_away_expected_goals": model_away_expected_goals,
                "context_home_expected_goals": context_home_expected_goals,
                "context_away_expected_goals": context_away_expected_goals,
                "context_home_attack_multiplier": float(context["home_attack_multiplier"]),
                "context_home_defense_multiplier": float(context["home_defense_multiplier"]),
                "context_away_attack_multiplier": float(context["away_attack_multiplier"]),
                "context_away_defense_multiplier": float(context["away_defense_multiplier"]),
                "context_draw_probability_multiplier": float(context["draw_probability_multiplier"]),
                "context_applied": context_applied,
                "context_confidence": str(context["confidence"]),
                "context_notes": str(context["notes"]),
                "strategy_home_expected_goals": strategy_home_expected_goals,
                "strategy_away_expected_goals": strategy_away_expected_goals,
                "market_adjusted_home_expected_goals": market_adjusted_home_expected_goals,
                "market_adjusted_away_expected_goals": market_adjusted_away_expected_goals,
                "home_expected_goals": strategy_home_expected_goals,
                "away_expected_goals": strategy_away_expected_goals,
                "goal_inflation": goal_inflation,
                "dynamic_goal_inflation": dynamic_goal_inflation,
                "dynamic_goal_inflation_reason": dynamic_goal_inflation_reason,
                "max_total_candidate_goals": max_total_candidate_goals,
                "draw_probability_multiplier": effective_draw_probability_multiplier,
                "is_last_group_match": bool(competitive["is_last_group_match"]),
                "home_needs_win": bool(competitive["home_needs_win"]),
                "away_needs_win": bool(competitive["away_needs_win"]),
                "home_draw_is_enough": bool(competitive["home_draw_is_enough"]),
                "away_draw_is_enough": bool(competitive["away_draw_is_enough"]),
                "competitive_context_applied": competitive_context_applied,
                "home_already_qualified": bool(competitive["home_already_qualified"]),
                "away_already_qualified": bool(competitive["away_already_qualified"]),
                "home_eliminated": bool(competitive["home_eliminated"]),
                "away_eliminated": bool(competitive["away_eliminated"]),
                "home_goal_difference_pressure": bool(competitive["home_goal_difference_pressure"]),
                "away_goal_difference_pressure": bool(competitive["away_goal_difference_pressure"]),
                "match_pressure_score": float(competitive["match_pressure_score"]),
                "group_scenario_volatility": float(competitive["group_scenario_volatility"]),
                "home_win_probability": home_win_probability,
                "draw_probability": draw_probability,
                "away_win_probability": away_win_probability,
                "market_home_win_probability": pd.NA,
                "market_draw_probability": pd.NA,
                "market_away_win_probability": pd.NA,
                "market_over_2_5_probability": pd.NA,
                "market_under_2_5_probability": pd.NA,
                "market_btts_yes_probability": pd.NA,
                "market_btts_no_probability": pd.NA,
                "home_market_edge": pd.NA,
                "draw_market_edge": pd.NA,
                "away_market_edge": pd.NA,
                "market_disagreement_score": pd.NA,
                "market_warning_flag": False,
                "market_warning_reason": "",
                "risk_profile": risk_profile,
                "conservative_scoreline": _scoreline_text(risk_picks["conservative"]["scoreline"]),
                "conservative_expected_points": float(risk_picks["conservative"]["expected_points"]),
                "balanced_scoreline": _scoreline_text(risk_picks["balanced"]["scoreline"]),
                "balanced_expected_points": float(risk_picks["balanced"]["expected_points"]),
                "aggressive_scoreline": _scoreline_text(risk_picks["aggressive"]["scoreline"]),
                "aggressive_expected_points": float(risk_picks["aggressive"]["expected_points"]),
                "desperation_scoreline": _scoreline_text(risk_picks["desperation"]["scoreline"]),
                "desperation_expected_points": float(risk_picks["desperation"]["expected_points"]),
                "recommended_scoreline_by_risk": _scoreline_text(pick),
                "estimated_pick_popularity": float(selected_pick["estimated_pick_popularity"]),
                "differential_multiplier": float(selected_pick["differential_multiplier"]),
                "upside_multiplier": float(selected_pick["upside_multiplier"]),
                "strategic_pick_value": float(selected_pick["strategic_pick_value"]),
                "confidence_level": confidence_level,
                "error_risk": error_risk,
                "main_reasons": main_reasons,
                "alert_flags": alert_flags,
                "recommended_home_score": pick[0],
                "recommended_away_score": pick[1],
                "recommended_scoreline": f"{pick[0]}-{pick[1]}",
                "recommended_expected_points": float(pick_expected_points),
                "draw_alternative_home_score": draw_pick[0],
                "draw_alternative_away_score": draw_pick[1],
                "draw_alternative_scoreline": f"{draw_pick[0]}-{draw_pick[1]}",
                "draw_alternative_expected_points": float(draw_expected_points),
                "draw_alternative_is_competitive": draw_is_competitive,
                "home_elo": float(match.home_elo),
                "away_elo": float(match.away_elo),
                "elo_diff": float(match.elo_diff),
            }
        )

    return pd.DataFrame(rows)


def train_final_model_and_predict_fixtures(
    processed_matches: pd.DataFrame,
    completed_results: pd.DataFrame,
    fixtures: pd.DataFrame,
    max_total_candidate_goals: int | None = DEFAULT_MAX_TOTAL_CANDIDATE_GOALS,
    draw_probability_multiplier: float = DEFAULT_DRAW_PROBABILITY_MULTIPLIER,
    goal_inflation: float = DEFAULT_GOAL_INFLATION,
    context_overrides: pd.DataFrame | None = None,
    dynamic_goal_inflation_enabled: bool = True,
    risk_profile: str = DEFAULT_RISK_PROFILE,
    qualified_context_enabled: bool = True,
    qualified_context_min_points: int = ALREADY_QUALIFIED_MIN_POINTS,
    qualified_context_advancing_slots: int = ALREADY_QUALIFIED_ADVANCING_SLOTS,
) -> pd.DataFrame:
    """Fit on all processed history and predict future fixture picks."""
    model = fit_goal_model(processed_matches)
    team_states = build_team_states(completed_results)
    fixture_features = build_fixture_features(fixtures, team_states)
    automatic_context = (
        build_already_qualified_context_overrides(
            completed_results,
            fixtures,
            min_points=qualified_context_min_points,
            advancing_slots=qualified_context_advancing_slots,
        )
        if qualified_context_enabled
        else None
    )
    combined_context = _combine_context_overrides(automatic_context, context_overrides)
    competitive_context = build_competitive_context(
        completed_results,
        fixtures,
        advancing_slots=qualified_context_advancing_slots,
    )
    return predict_fixture_picks(
        model,
        fixture_features,
        max_total_candidate_goals=max_total_candidate_goals,
        draw_probability_multiplier=draw_probability_multiplier,
        goal_inflation=goal_inflation,
        context_overrides=combined_context,
        competitive_context=competitive_context,
        dynamic_goal_inflation_enabled=dynamic_goal_inflation_enabled,
        risk_profile=risk_profile,
    )
