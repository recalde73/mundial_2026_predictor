from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
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
    if home_score > away_score:
        standings[home_team]["points"] += 3
    elif home_score < away_score:
        standings[away_team]["points"] += 3
    else:
        standings[home_team]["points"] += 1
        standings[away_team]["points"] += 1


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

    standings: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"played": 0, "points": 0})
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
) -> pd.DataFrame:
    """Predict expected goals and recommended picks for future fixtures."""
    predictions = predict_expected_goals(model, fixture_features)
    context_by_fixture = _build_context_override_map(context_overrides)
    rows: list[dict[str, object]] = []

    for match in predictions.itertuples(index=False):
        context = _context_adjustment_for(
            context_by_fixture,
            match.date,
            str(match.home_team),
            str(match.away_team),
        )
        context_applied = _fixture_key(match.date, str(match.home_team), str(match.away_team)) in context_by_fixture
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
        strategy_home_expected_goals, strategy_away_expected_goals = apply_goal_inflation(
            context_home_expected_goals,
            context_away_expected_goals,
            goal_inflation=goal_inflation,
        )
        probabilities = scoreline_probabilities(
            home_expected_goals=strategy_home_expected_goals,
            away_expected_goals=strategy_away_expected_goals,
            max_goals=6,
        )
        pick, pick_expected_points = rank_predictions(
            probabilities,
            max_total_candidate_goals=max_total_candidate_goals,
            draw_probability_multiplier=effective_draw_probability_multiplier,
            limit=1,
        )[0]
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

        rows.append(
            {
                "date": match.date,
                "tournament": match.tournament,
                "home_team": match.home_team,
                "away_team": match.away_team,
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
                "home_expected_goals": strategy_home_expected_goals,
                "away_expected_goals": strategy_away_expected_goals,
                "goal_inflation": goal_inflation,
                "max_total_candidate_goals": max_total_candidate_goals,
                "draw_probability_multiplier": effective_draw_probability_multiplier,
                "home_win_probability": home_win_probability,
                "draw_probability": draw_probability,
                "away_win_probability": away_win_probability,
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
    return predict_fixture_picks(
        model,
        fixture_features,
        max_total_candidate_goals=max_total_candidate_goals,
        draw_probability_multiplier=draw_probability_multiplier,
        goal_inflation=goal_inflation,
        context_overrides=combined_context,
    )
