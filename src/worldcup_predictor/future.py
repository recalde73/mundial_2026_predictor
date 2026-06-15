from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from worldcup_predictor.datasets import MatchRecord
from worldcup_predictor.elo import EloConfig, expected_result, update_ratings
from worldcup_predictor.modeling import BaselineGoalModel, fit_goal_model, predict_expected_goals
from worldcup_predictor.poisson import scoreline_probabilities
from worldcup_predictor.recommender import adjust_draw_probabilities, rank_predictions
from worldcup_predictor.scoring import candidate_scorelines, expected_points


COMPETITIVE_DRAW_PROBABILITY_THRESHOLD = 0.26
COMPETITIVE_DRAW_RESULT_MARGIN = 0.03


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
    max_total_candidate_goals: int = 2,
) -> tuple[tuple[int, int], float]:
    adjusted_probabilities = adjust_draw_probabilities(probabilities)
    draw_candidates = tuple(
        scoreline
        for scoreline in candidate_scorelines()
        if scoreline[0] == scoreline[1] and sum(scoreline) <= max_total_candidate_goals
    )
    return max(
        ((candidate, expected_points(candidate, adjusted_probabilities)) for candidate in draw_candidates),
        key=lambda item: item[1],
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
) -> pd.DataFrame:
    """Predict expected goals and recommended picks for future fixtures."""
    predictions = predict_expected_goals(model, fixture_features)
    rows: list[dict[str, object]] = []

    for match in predictions.itertuples(index=False):
        probabilities = scoreline_probabilities(
            home_expected_goals=float(match.home_expected_goals),
            away_expected_goals=float(match.away_expected_goals),
            max_goals=6,
        )
        pick, pick_expected_points = rank_predictions(probabilities, limit=1)[0]
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
        draw_pick, draw_expected_points = _best_draw_alternative(probabilities)
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
                "home_expected_goals": float(match.home_expected_goals),
                "away_expected_goals": float(match.away_expected_goals),
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
) -> pd.DataFrame:
    """Fit on all processed history and predict future fixture picks."""
    model = fit_goal_model(processed_matches)
    team_states = build_team_states(completed_results)
    fixture_features = build_fixture_features(fixtures, team_states)
    return predict_fixture_picks(model, fixture_features)
