from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulatedTeamStanding:
    team: str
    group: str
    position: int
    points: int
    goals_for: int
    goals_against: int
    goal_difference: int


@dataclass(frozen=True)
class TeamTournamentProfile:
    team: str
    elo: float
    attack: float
    defense: float


@dataclass(frozen=True)
class FastGroupData:
    group: str
    team_indices: tuple[int, ...]
    home_local_indices: np.ndarray
    away_local_indices: np.ndarray
    home_expected_goals: np.ndarray
    away_expected_goals: np.ndarray


@dataclass(frozen=True)
class FastTournamentData:
    teams: tuple[str, ...]
    team_groups: tuple[str, ...]
    groups: tuple[FastGroupData, ...]
    elo: np.ndarray
    attack: np.ndarray
    defense: np.ndarray


def infer_groups_from_fixtures(fixtures: pd.DataFrame) -> dict[str, str]:
    """Infer group labels from the fixture graph."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, pd.Timestamp] = {}

    for fixture in fixtures.sort_values("date").itertuples(index=False):
        adjacency[fixture.home_team].add(fixture.away_team)
        adjacency[fixture.away_team].add(fixture.home_team)
        first_seen.setdefault(fixture.home_team, fixture.date)
        first_seen.setdefault(fixture.away_team, fixture.date)

    visited: set[str] = set()
    components: list[list[str]] = []
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
        components.append(sorted(component, key=lambda item: (first_seen[item], item)))

    components.sort(key=lambda component: min(first_seen[team] for team in component))
    team_to_group: dict[str, str] = {}
    for index, component in enumerate(components, start=1):
        group = f"Group {index:02d}"
        for team in component:
            team_to_group[team] = group

    return team_to_group


def add_inferred_groups(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return fixtures with inferred group labels."""
    groups = infer_groups_from_fixtures(fixtures)
    enriched = fixtures.copy()
    enriched["home_group"] = enriched["home_team"].map(groups)
    enriched["away_group"] = enriched["away_team"].map(groups)
    enriched["group"] = enriched["home_group"]
    return enriched


def _empty_table(teams: list[str]) -> dict[str, dict[str, int]]:
    return {
        team: {"points": 0, "goals_for": 0, "goals_against": 0, "goal_difference": 0}
        for team in teams
    }


def _add_match(
    table: dict[str, dict[str, int]],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    table[home_team]["goals_for"] += home_goals
    table[home_team]["goals_against"] += away_goals
    table[away_team]["goals_for"] += away_goals
    table[away_team]["goals_against"] += home_goals
    table[home_team]["goal_difference"] += home_goals - away_goals
    table[away_team]["goal_difference"] += away_goals - home_goals

    if home_goals > away_goals:
        table[home_team]["points"] += 3
    elif home_goals < away_goals:
        table[away_team]["points"] += 3
    else:
        table[home_team]["points"] += 1
        table[away_team]["points"] += 1


def _rank_table(table: dict[str, dict[str, int]], rng: np.random.Generator) -> list[tuple[str, dict[str, int]]]:
    return sorted(
        table.items(),
        key=lambda item: (
            item[1]["points"],
            item[1]["goal_difference"],
            item[1]["goals_for"],
            rng.random(),
        ),
        reverse=True,
    )


def _standing_from_ranked_group(
    group: str,
    ranked: list[tuple[str, dict[str, int]]],
) -> list[SimulatedTeamStanding]:
    return [
        SimulatedTeamStanding(
            team=team,
            group=group,
            position=position,
            points=stats["points"],
            goals_for=stats["goals_for"],
            goals_against=stats["goals_against"],
            goal_difference=stats["goal_difference"],
        )
        for position, (team, stats) in enumerate(ranked, start=1)
    ]


def simulate_group_tables_once(
    predictions: pd.DataFrame,
    rng: np.random.Generator,
) -> list[SimulatedTeamStanding]:
    """Simulate one full group stage and return standings."""
    fixtures = add_inferred_groups(predictions)
    group_teams = {
        group: sorted(set(group_rows["home_team"]) | set(group_rows["away_team"]))
        for group, group_rows in fixtures.groupby("group")
    }
    standings: list[SimulatedTeamStanding] = []

    for group, group_rows in fixtures.groupby("group"):
        table = _empty_table(group_teams[group])
        for match in group_rows.itertuples(index=False):
            home_goals = int(rng.poisson(float(match.home_expected_goals)))
            away_goals = int(rng.poisson(float(match.away_expected_goals)))
            _add_match(table, match.home_team, match.away_team, home_goals, away_goals)

        standings.extend(_standing_from_ranked_group(group, _rank_table(table, rng)))

    return standings


def select_round_of_32_qualifiers(
    standings: list[SimulatedTeamStanding],
    rng: np.random.Generator,
) -> list[SimulatedTeamStanding]:
    """Select top two from each group plus the eight best third-placed teams."""
    automatic = [standing for standing in standings if standing.position <= 2]
    thirds = [standing for standing in standings if standing.position == 3]
    best_thirds = sorted(
        thirds,
        key=lambda standing: (
            standing.points,
            standing.goal_difference,
            standing.goals_for,
            rng.random(),
        ),
        reverse=True,
    )[:8]
    qualifiers = automatic + best_thirds
    return sorted(
        qualifiers,
        key=lambda standing: (
            standing.position,
            -standing.points,
            -standing.goal_difference,
            -standing.goals_for,
            rng.random(),
        ),
    )


def team_profiles_from_predictions(predictions: pd.DataFrame) -> dict[str, TeamTournamentProfile]:
    """Build neutral-team profiles for knockout simulations from group predictions."""
    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"elo": [], "attack": [], "defense": []})

    for match in predictions.itertuples(index=False):
        rows[match.home_team]["elo"].append(float(match.home_elo))
        rows[match.home_team]["attack"].append(float(match.home_expected_goals))
        rows[match.home_team]["defense"].append(float(match.away_expected_goals))
        rows[match.away_team]["elo"].append(float(match.away_elo))
        rows[match.away_team]["attack"].append(float(match.away_expected_goals))
        rows[match.away_team]["defense"].append(float(match.home_expected_goals))

    return {
        team: TeamTournamentProfile(
            team=team,
            elo=float(np.mean(values["elo"])),
            attack=max(0.2, float(np.mean(values["attack"]))),
            defense=max(0.2, float(np.mean(values["defense"]))),
        )
        for team, values in rows.items()
    }


def expected_knockout_goals(
    team: TeamTournamentProfile,
    opponent: TeamTournamentProfile,
) -> tuple[float, float]:
    """Estimate neutral knockout expected goals between two arbitrary teams."""
    elo_factor = (team.elo - opponent.elo) / 400
    team_goals = ((team.attack + opponent.defense) / 2) * np.exp(0.25 * elo_factor)
    opponent_goals = ((opponent.attack + team.defense) / 2) * np.exp(-0.25 * elo_factor)
    return max(0.2, float(team_goals)), max(0.2, float(opponent_goals))


def simulate_knockout_winner(
    team: str,
    opponent: str,
    profiles: dict[str, TeamTournamentProfile],
    rng: np.random.Generator,
) -> str:
    """Simulate one knockout match and return the advancing team."""
    team_goals_expected, opponent_goals_expected = expected_knockout_goals(
        profiles[team],
        profiles[opponent],
    )
    team_goals = int(rng.poisson(team_goals_expected))
    opponent_goals = int(rng.poisson(opponent_goals_expected))

    if team_goals > opponent_goals:
        return team
    if opponent_goals > team_goals:
        return opponent

    team_penalty_probability = 1 / (1 + np.exp(-(profiles[team].elo - profiles[opponent].elo) / 300))
    return team if rng.random() < team_penalty_probability else opponent


def pair_seeded_qualifiers(qualifiers: list[SimulatedTeamStanding]) -> list[tuple[str, str]]:
    """Pair qualifiers as strongest vs weakest in an approximate bracket."""
    seeded = sorted(
        qualifiers,
        key=lambda standing: (
            standing.position,
            -standing.points,
            -standing.goal_difference,
            -standing.goals_for,
            standing.group,
            standing.team,
        ),
    )
    return [
        (seeded[index].team, seeded[-index - 1].team)
        for index in range(len(seeded) // 2)
    ]


def simulate_tournament(
    predictions: pd.DataFrame,
    simulations: int = 10000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Simulate group stage plus an approximate knockout bracket."""
    if simulations < 1:
        raise ValueError("simulations must be greater than or equal to 1")

    data = _prepare_fast_tournament_data(predictions)
    rng = np.random.default_rng(seed)
    metric_names = (
        "round_of_32",
        "round_of_16",
        "quarterfinal",
        "semifinal",
        "final",
        "champion",
        "runner_up",
        "third_place",
    )
    metric_index = {metric: index for index, metric in enumerate(metric_names)}
    counts = np.zeros((len(data.teams), len(metric_names)), dtype=np.int64)

    for _simulation in range(simulations):
        qualifiers = _simulate_fast_group_qualifiers(data, rng)
        counts[qualifiers, metric_index["round_of_32"]] += 1

        current = _pair_seeded_indices(qualifiers)
        winners, _losers = _simulate_knockout_round(current, data, rng)
        counts[winners, metric_index["round_of_16"]] += 1

        current = _pair_adjacent_indices(winners)
        winners, _losers = _simulate_knockout_round(current, data, rng)
        counts[winners, metric_index["quarterfinal"]] += 1

        current = _pair_adjacent_indices(winners)
        winners, _losers = _simulate_knockout_round(current, data, rng)
        counts[winners, metric_index["semifinal"]] += 1

        current = _pair_adjacent_indices(winners)
        finalists, semifinal_losers = _simulate_knockout_round(current, data, rng)
        counts[finalists, metric_index["final"]] += 1

        final_pair = np.array([[finalists[0], finalists[1]]], dtype=np.int64)
        champion_array, runner_up_array = _simulate_knockout_round(final_pair, data, rng)
        counts[champion_array[0], metric_index["champion"]] += 1
        counts[runner_up_array[0], metric_index["runner_up"]] += 1

        third_pair = np.array([[semifinal_losers[0], semifinal_losers[1]]], dtype=np.int64)
        third_array, _fourth_array = _simulate_knockout_round(third_pair, data, rng)
        counts[third_array[0], metric_index["third_place"]] += 1

    rows = []
    for team_index, team in enumerate(data.teams):
        rows.append(
            {
                "team": team,
                "group": data.team_groups[team_index],
                "round_of_32_probability": counts[team_index, metric_index["round_of_32"]] / simulations,
                "round_of_16_probability": counts[team_index, metric_index["round_of_16"]] / simulations,
                "quarterfinal_probability": counts[team_index, metric_index["quarterfinal"]] / simulations,
                "semifinal_probability": counts[team_index, metric_index["semifinal"]] / simulations,
                "final_probability": counts[team_index, metric_index["final"]] / simulations,
                "champion_probability": counts[team_index, metric_index["champion"]] / simulations,
                "runner_up_probability": counts[team_index, metric_index["runner_up"]] / simulations,
                "third_place_probability": counts[team_index, metric_index["third_place"]] / simulations,
                "podium_expected_points": counts[team_index, metric_index["champion"]] / simulations * 10
                + counts[team_index, metric_index["runner_up"]] / simulations * 8
                + counts[team_index, metric_index["third_place"]] / simulations * 5,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["champion_probability", "final_probability", "semifinal_probability"],
        ascending=False,
    ).reset_index(drop=True)


def _prepare_fast_tournament_data(predictions: pd.DataFrame) -> FastTournamentData:
    fixtures = add_inferred_groups(predictions).sort_values(["group", "date"]).reset_index(drop=True)
    profiles = team_profiles_from_predictions(fixtures)
    teams = tuple(sorted(profiles))
    team_to_index = {team: index for index, team in enumerate(teams)}
    groups_by_team = infer_groups_from_fixtures(fixtures)
    team_groups = tuple(groups_by_team[team] for team in teams)

    fast_groups: list[FastGroupData] = []
    for group, group_rows in fixtures.groupby("group", sort=True):
        group_team_indices = tuple(
            sorted(
                {team_to_index[team] for team in group_rows["home_team"]}
                | {team_to_index[team] for team in group_rows["away_team"]}
            )
        )
        local_index = {team_index: index for index, team_index in enumerate(group_team_indices)}
        home_indices = np.array(
            [local_index[team_to_index[team]] for team in group_rows["home_team"]],
            dtype=np.int64,
        )
        away_indices = np.array(
            [local_index[team_to_index[team]] for team in group_rows["away_team"]],
            dtype=np.int64,
        )
        fast_groups.append(
            FastGroupData(
                group=group,
                team_indices=group_team_indices,
                home_local_indices=home_indices,
                away_local_indices=away_indices,
                home_expected_goals=group_rows["home_expected_goals"].to_numpy(dtype=float),
                away_expected_goals=group_rows["away_expected_goals"].to_numpy(dtype=float),
            )
        )

    return FastTournamentData(
        teams=teams,
        team_groups=team_groups,
        groups=tuple(fast_groups),
        elo=np.array([profiles[team].elo for team in teams], dtype=float),
        attack=np.array([profiles[team].attack for team in teams], dtype=float),
        defense=np.array([profiles[team].defense for team in teams], dtype=float),
    )


def _rank_group_arrays(
    team_indices: tuple[int, ...],
    points: np.ndarray,
    goals_for: np.ndarray,
    goals_against: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int, int]]:
    goal_difference = goals_for - goals_against
    order = sorted(
        range(len(team_indices)),
        key=lambda index: (
            int(points[index]),
            int(goal_difference[index]),
            int(goals_for[index]),
            float(rng.random()),
        ),
        reverse=True,
    )
    return [
        (
            team_indices[index],
            int(points[index]),
            int(goal_difference[index]),
            int(goals_for[index]),
            position,
        )
        for position, index in enumerate(order, start=1)
    ]


def _simulate_fast_group_qualifiers(
    data: FastTournamentData,
    rng: np.random.Generator,
) -> np.ndarray:
    automatic: list[tuple[int, int, int, int, int]] = []
    thirds: list[tuple[int, int, int, int, int]] = []

    for group in data.groups:
        team_count = len(group.team_indices)
        points = np.zeros(team_count, dtype=np.int64)
        goals_for = np.zeros(team_count, dtype=np.int64)
        goals_against = np.zeros(team_count, dtype=np.int64)
        home_goals = rng.poisson(group.home_expected_goals)
        away_goals = rng.poisson(group.away_expected_goals)

        for match_index, home_local_index in enumerate(group.home_local_indices):
            away_local_index = group.away_local_indices[match_index]
            home_score = int(home_goals[match_index])
            away_score = int(away_goals[match_index])
            goals_for[home_local_index] += home_score
            goals_against[home_local_index] += away_score
            goals_for[away_local_index] += away_score
            goals_against[away_local_index] += home_score

            if home_score > away_score:
                points[home_local_index] += 3
            elif away_score > home_score:
                points[away_local_index] += 3
            else:
                points[home_local_index] += 1
                points[away_local_index] += 1

        ranked = _rank_group_arrays(group.team_indices, points, goals_for, goals_against, rng)
        automatic.extend(ranked[:2])
        thirds.append(ranked[2])

    best_thirds = sorted(
        thirds,
        key=lambda item: (item[1], item[2], item[3], float(rng.random())),
        reverse=True,
    )[:8]
    qualifiers = automatic + best_thirds
    seeded = sorted(
        qualifiers,
        key=lambda item: (item[4], -item[1], -item[2], -item[3], item[0]),
    )
    return np.array([item[0] for item in seeded], dtype=np.int64)


def _pair_seeded_indices(team_indices: np.ndarray) -> np.ndarray:
    return np.array(
        [[int(team_indices[index]), int(team_indices[-index - 1])] for index in range(len(team_indices) // 2)],
        dtype=np.int64,
    )


def _pair_adjacent_indices(team_indices: np.ndarray) -> np.ndarray:
    return np.array(
        [[int(team_indices[index]), int(team_indices[index + 1])] for index in range(0, len(team_indices), 2)],
        dtype=np.int64,
    )


def _simulate_knockout_round(
    pairs: np.ndarray,
    data: FastTournamentData,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    winners = np.empty(len(pairs), dtype=np.int64)
    losers = np.empty(len(pairs), dtype=np.int64)

    for index, (team, opponent) in enumerate(pairs):
        elo_factor = (data.elo[team] - data.elo[opponent]) / 400
        team_expected_goals = max(
            0.2,
            float(((data.attack[team] + data.defense[opponent]) / 2) * np.exp(0.25 * elo_factor)),
        )
        opponent_expected_goals = max(
            0.2,
            float(((data.attack[opponent] + data.defense[team]) / 2) * np.exp(-0.25 * elo_factor)),
        )
        team_goals = int(rng.poisson(team_expected_goals))
        opponent_goals = int(rng.poisson(opponent_expected_goals))

        if team_goals > opponent_goals:
            winner = team
        elif opponent_goals > team_goals:
            winner = opponent
        else:
            team_penalty_probability = 1 / (1 + np.exp(-(data.elo[team] - data.elo[opponent]) / 300))
            winner = team if rng.random() < team_penalty_probability else opponent

        winners[index] = winner
        losers[index] = opponent if winner == team else team

    return winners, losers


def simulate_group_stage(
    predictions: pd.DataFrame,
    simulations: int = 10000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Simulate group stage standings and advancement probabilities."""
    if simulations < 1:
        raise ValueError("simulations must be greater than or equal to 1")

    fixtures = add_inferred_groups(predictions)
    rng = np.random.default_rng(seed)
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    counts = {
        team: {"first": 0, "second": 0, "third": 0, "fourth": 0, "advance": 0}
        for team in teams
    }
    group_teams = {
        group: sorted(set(group_rows["home_team"]) | set(group_rows["away_team"]))
        for group, group_rows in fixtures.groupby("group")
    }

    for _simulation in range(simulations):
        third_place: list[tuple[str, dict[str, int]]] = []

        for group, group_rows in fixtures.groupby("group"):
            table = _empty_table(group_teams[group])
            for match in group_rows.itertuples(index=False):
                home_goals = int(rng.poisson(float(match.home_expected_goals)))
                away_goals = int(rng.poisson(float(match.away_expected_goals)))
                _add_match(table, match.home_team, match.away_team, home_goals, away_goals)

            ranked = _rank_table(table, rng)
            third_place.append(ranked[2])

            for position, (team, _stats) in enumerate(ranked, start=1):
                if position == 1:
                    counts[team]["first"] += 1
                    counts[team]["advance"] += 1
                elif position == 2:
                    counts[team]["second"] += 1
                    counts[team]["advance"] += 1
                elif position == 3:
                    counts[team]["third"] += 1
                else:
                    counts[team]["fourth"] += 1

        for team, _stats in _rank_table(dict(third_place), rng)[:8]:
            counts[team]["advance"] += 1

    groups = infer_groups_from_fixtures(predictions)
    rows = []
    for team, values in counts.items():
        rows.append(
            {
                "team": team,
                "group": groups[team],
                "first_probability": values["first"] / simulations,
                "second_probability": values["second"] / simulations,
                "third_probability": values["third"] / simulations,
                "fourth_probability": values["fourth"] / simulations,
                "advance_probability": values["advance"] / simulations,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["group", "advance_probability", "first_probability"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
