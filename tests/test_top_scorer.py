import pandas as pd

from worldcup_predictor.top_scorer import estimate_top_scorers


def test_estimate_top_scorers_ranks_candidates() -> None:
    candidates = pd.DataFrame(
        {
            "player": ["A", "B"],
            "team": ["Team A", "Team B"],
            "goal_share": [0.4, 0.2],
            "minutes_share": [0.9, 0.9],
            "penalty_boost": [0.1, 0.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "home_team": ["Team A"],
            "away_team": ["Team B"],
            "home_expected_goals": [2.0],
            "away_expected_goals": [1.0],
        }
    )
    tournament = pd.DataFrame(
        {
            "team": ["Team A", "Team B"],
            "round_of_16_probability": [1.0, 0.0],
            "quarterfinal_probability": [0.5, 0.0],
            "semifinal_probability": [0.2, 0.0],
            "final_probability": [0.1, 0.0],
        }
    )

    top_scorers = estimate_top_scorers(candidates, predictions, tournament)

    assert top_scorers.iloc[0].player == "A"
    assert top_scorers.iloc[0].expected_goals > top_scorers.iloc[1].expected_goals
