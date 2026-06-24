import csv

from scripts import update_world_cup_live as live


def test_sync_played_matches_history_creates_and_deduplicates_file(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "world_cup_2026_played_matches.csv"
    monkeypatch.setattr(live, "PLAYED_MATCHES_HISTORY_PATH", history_path)
    results = {
        ("2026-06-11", "Mexico", "South Africa"): {
            "date": "2026-06-11",
            "home_team": "Mexico",
            "away_team": "South Africa",
            "home_score": "2",
            "away_score": "0",
        },
        ("2026-06-12", "Canada", "Bosnia and Herzegovina"): {
            "date": "2026-06-12",
            "home_team": "Canada",
            "away_team": "Bosnia and Herzegovina",
            "home_score": "1",
            "away_score": "1",
        },
    }

    assert live.sync_played_matches_history(results) == 2
    assert live.sync_played_matches_history(results) == 0

    with history_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "date": "2026-06-11",
            "tournament": "FIFA World Cup",
            "home_team": "Mexico",
            "away_team": "South Africa",
            "home_score": "2",
            "away_score": "0",
        },
        {
            "date": "2026-06-12",
            "tournament": "FIFA World Cup",
            "home_team": "Canada",
            "away_team": "Bosnia and Herzegovina",
            "home_score": "1",
            "away_score": "1",
        },
    ]


def test_sync_played_matches_history_updates_corrected_score(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "world_cup_2026_played_matches.csv"
    monkeypatch.setattr(live, "PLAYED_MATCHES_HISTORY_PATH", history_path)
    original = {
        ("2026-06-11", "Mexico", "South Africa"): {
            "date": "2026-06-11",
            "home_team": "Mexico",
            "away_team": "South Africa",
            "home_score": "2",
            "away_score": "0",
        }
    }
    corrected = {
        ("2026-06-11", "Mexico", "South Africa"): {
            "date": "2026-06-11",
            "home_team": "Mexico",
            "away_team": "South Africa",
            "home_score": "2",
            "away_score": "1",
        }
    }

    live.sync_played_matches_history(original)

    assert live.sync_played_matches_history(corrected) == 1

    with history_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["away_score"] == "1"
