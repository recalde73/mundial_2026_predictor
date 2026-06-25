from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_RESULTS_PATH = PROJECT_ROOT / "data" / "raw" / "results.csv"
OVERRIDES_PATH = PROJECT_ROOT / "data" / "manual" / "match_results_overrides.csv"
PLAYED_MATCHES_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "world_cup_2026_played_matches.csv"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
PREVIEW_PID_PATH = DASHBOARD_DIR / ".vite-preview.pid"
PREVIEW_LOG_PATH = DASHBOARD_DIR / "vite-preview.log"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date}"
)
RESULT_FIELDNAMES = ["date", "home_team", "away_team", "home_score", "away_score"]
PLAYED_MATCHES_HISTORY_FIELDNAMES = [
    "date",
    "tournament",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]


def run(command: list[str], cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None) -> None:
    print(f"\nRunning {' '.join(command)}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update World Cup results, regenerate predictions, and deploy the dashboard locally."
    )
    parser.add_argument("--from-date", default="2026-06-11")
    parser.add_argument("--to-date", default=date.today().isoformat())
    parser.add_argument("--no-deploy", action="store_true", help="Build data only; do not start Vite preview.")
    parser.add_argument("--port", default="4173", help="Vite preview port.")
    return parser.parse_args()


def normalize_team_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().replace("&", "and")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    aliases = {
        "bosnia herzegovina": "bosnia and herzegovina",
        "congo dr": "dr congo",
        "czechia": "czech republic",
        "turkiye": "turkey",
        "usa": "united states",
    }
    return aliases.get(normalized, normalized)


def find_fixture(
    fixtures: dict[tuple[str, str, str], tuple[str, str, str]],
    match_date: date,
    home_team: str,
    away_team: str,
) -> tuple[tuple[str, str, str], bool] | None:
    for date_offset in (0, -1, 1):
        fixture_date = (match_date + timedelta(days=date_offset)).isoformat()
        fixture = fixtures.get((fixture_date, home_team, away_team))
        if fixture:
            return fixture, False
        fixture = fixtures.get((fixture_date, away_team, home_team))
        if fixture:
            return fixture, True
    return None


def daterange(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def load_world_cup_fixtures() -> dict[tuple[str, str, str], tuple[str, str, str]]:
    fixtures: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    with RAW_RESULTS_PATH.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["tournament"] != "FIFA World Cup":
                continue
            key = (
                row["date"],
                normalize_team_name(row["home_team"]),
                normalize_team_name(row["away_team"]),
            )
            fixtures[key] = (row["date"], row["home_team"], row["away_team"])
    return fixtures


def fetch_completed_results(fixtures: dict[tuple[str, str, str], tuple[str, str, str]], days: list[date]) -> list[dict[str, str]]:
    completed_results: list[dict[str, str]] = []
    for match_date in days:
        url = ESPN_SCOREBOARD_URL.format(date=match_date.strftime("%Y%m%d"))
        with urlopen(url, timeout=30) as response:
            payload = json.load(response)

        for event in payload.get("events", []):
            competition = event.get("competitions", [{}])[0]
            status = competition.get("status", {}).get("type", {})
            if not status.get("completed"):
                continue

            teams = competition.get("competitors", [])
            home = next((team for team in teams if team.get("homeAway") == "home"), None)
            away = next((team for team in teams if team.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            home_team = normalize_team_name(home["team"]["displayName"])
            away_team = normalize_team_name(away["team"]["displayName"])
            fixture_match = find_fixture(fixtures, match_date, home_team, away_team)
            if not fixture_match:
                print(
                    "Skipping ESPN result not found in raw fixtures: "
                    f"{match_date.isoformat()} {home['team']['displayName']} vs {away['team']['displayName']}"
                )
                continue
            fixture, reverse_score = fixture_match
            home_score = int(home["score"])
            away_score = int(away["score"])
            if reverse_score:
                home_score, away_score = away_score, home_score

            completed_results.append(
                {
                    "date": fixture[0],
                    "home_team": fixture[1],
                    "away_team": fixture[2],
                    "home_score": str(home_score),
                    "away_score": str(away_score),
                }
            )
    return completed_results


def read_overrides() -> dict[tuple[str, str, str], dict[str, str]]:
    if not OVERRIDES_PATH.exists():
        return {}
    with OVERRIDES_PATH.open(newline="") as file:
        reader = csv.DictReader(file)
        return {
            (row["date"], row["home_team"], row["away_team"]): row
            for row in reader
        }


def write_overrides(rows: dict[tuple[str, str, str], dict[str, str]]) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows.values(), key=lambda row: (row["date"], row["home_team"], row["away_team"]))
    with OVERRIDES_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered_rows)


def read_played_matches_history() -> dict[tuple[str, str, str], dict[str, str]]:
    if not PLAYED_MATCHES_HISTORY_PATH.exists():
        return {}
    with PLAYED_MATCHES_HISTORY_PATH.open(newline="") as file:
        reader = csv.DictReader(file)
        return {
            (row["date"], row["home_team"], row["away_team"]): row
            for row in reader
        }


def write_played_matches_history(rows: dict[tuple[str, str, str], dict[str, str]]) -> None:
    PLAYED_MATCHES_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows.values(), key=lambda row: (row["date"], row["home_team"], row["away_team"]))
    with PLAYED_MATCHES_HISTORY_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLAYED_MATCHES_HISTORY_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered_rows)


def sync_played_matches_history(results: dict[tuple[str, str, str], dict[str, str]]) -> int:
    history = read_played_matches_history()
    changed = 0
    for result in results.values():
        row = {
            "date": result["date"],
            "tournament": "FIFA World Cup",
            "home_team": result["home_team"],
            "away_team": result["away_team"],
            "home_score": result["home_score"],
            "away_score": result["away_score"],
        }
        key = (row["date"], row["home_team"], row["away_team"])
        if history.get(key) != row:
            history[key] = row
            changed += 1

    write_played_matches_history(history)
    return changed


def sync_result_overrides(start: date, end: date) -> int:
    fixtures = load_world_cup_fixtures()
    espn_results = fetch_completed_results(fixtures, daterange(start, end))
    overrides = read_overrides()

    changed = 0
    for result in espn_results:
        key = (result["date"], result["home_team"], result["away_team"])
        if overrides.get(key) != result:
            overrides[key] = result
            changed += 1

    write_overrides(overrides)
    history_changed = sync_played_matches_history(overrides)
    print(f"Completed ESPN results found: {len(espn_results)}")
    print(f"Manual overrides changed: {changed}")
    print(f"Played matches history changed: {history_changed}")
    return changed


def stop_existing_preview() -> None:
    if not PREVIEW_PID_PATH.exists():
        return
    pid_text = PREVIEW_PID_PATH.read_text().strip()
    if not pid_text:
        PREVIEW_PID_PATH.unlink(missing_ok=True)
        return

    pid = int(pid_text)
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
    except ProcessLookupError:
        pass
    PREVIEW_PID_PATH.unlink(missing_ok=True)


def build_and_deploy_dashboard(port: str) -> None:
    run(["npm", "run", "build"], cwd=DASHBOARD_DIR)
    stop_existing_preview()

    log = PREVIEW_LOG_PATH.open("a")
    process = subprocess.Popen(
        ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", port],
        cwd=DASHBOARD_DIR,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PREVIEW_PID_PATH.write_text(str(process.pid))
    print(f"Dashboard deployed locally: http://localhost:{port}")
    print(f"Preview log: {PREVIEW_LOG_PATH}")


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.to_date, "%Y-%m-%d").date()
    if end < start:
        raise SystemExit("--to-date must be equal to or later than --from-date")

    python = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    run([python, "scripts/download_real_data.py"], env=env)
    sync_result_overrides(start, end)
    run([python, "scripts/run_full_pipeline.py"], env=env)

    if not args.no_deploy:
        build_and_deploy_dashboard(args.port)


if __name__ == "__main__":
    main()
