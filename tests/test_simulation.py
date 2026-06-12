import pandas as pd

from worldcup_predictor.simulation import (
    infer_groups_from_fixtures,
    select_round_of_32_qualifiers,
    simulate_group_stage,
    simulate_group_tables_once,
    simulate_tournament,
)


def test_infer_groups_from_fixtures_uses_connected_components() -> None:
    fixtures = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "home_team": ["A", "A", "C"],
            "away_team": ["B", "C", "D"],
        }
    )

    groups = infer_groups_from_fixtures(fixtures)

    assert groups["A"] == groups["D"]


def test_simulate_group_stage_returns_probabilities() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6),
            "home_team": ["A", "A", "A", "B", "B", "C"],
            "away_team": ["B", "C", "D", "C", "D", "D"],
            "home_expected_goals": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            "away_expected_goals": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        }
    )

    simulation = simulate_group_stage(predictions, simulations=20, seed=1)

    assert len(simulation) == 4
    assert set(simulation["team"]) == {"A", "B", "C", "D"}
    assert all(simulation["advance_probability"].between(0, 1))


def test_select_round_of_32_qualifiers_selects_32_teams() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=72),
            "home_team": [f"T{group}_{home}" for group in range(12) for home, _away in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]],
            "away_team": [f"T{group}_{away}" for group in range(12) for _home, away in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]],
            "home_expected_goals": [1.2] * 72,
            "away_expected_goals": [1.0] * 72,
        }
    )
    rng = __import__("numpy").random.default_rng(1)
    standings = simulate_group_tables_once(predictions, rng)
    qualifiers = select_round_of_32_qualifiers(standings, rng)

    assert len(qualifiers) == 32


def test_simulate_tournament_returns_podium_probabilities() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=72),
            "home_team": [f"T{group}_{home}" for group in range(12) for home, _away in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]],
            "away_team": [f"T{group}_{away}" for group in range(12) for _home, away in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]],
            "home_expected_goals": [1.3] * 72,
            "away_expected_goals": [1.0] * 72,
            "home_elo": [1600.0] * 72,
            "away_elo": [1500.0] * 72,
        }
    )

    simulation = simulate_tournament(predictions, simulations=5, seed=1)

    assert len(simulation) == 48
    assert all(simulation["champion_probability"].between(0, 1))
    assert simulation["champion_probability"].sum() == 1
