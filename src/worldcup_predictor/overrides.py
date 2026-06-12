from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_OVERRIDE_COLUMNS = {"date", "home_team", "away_team", "home_score", "away_score"}


def load_result_overrides(path: str | Path) -> pd.DataFrame:
    """Load manually entered match results, returning an empty frame when absent."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=sorted(REQUIRED_OVERRIDE_COLUMNS))

    overrides = pd.read_csv(path, parse_dates=["date"])
    missing_columns = REQUIRED_OVERRIDE_COLUMNS.difference(overrides.columns)
    if missing_columns:
        raise ValueError(f"Missing override columns: {sorted(missing_columns)}")

    overrides = overrides.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    if overrides.empty:
        return overrides

    overrides["home_score"] = overrides["home_score"].astype(int)
    overrides["away_score"] = overrides["away_score"].astype(int)
    overrides["home_team"] = overrides["home_team"].astype(str).str.strip()
    overrides["away_team"] = overrides["away_team"].astype(str).str.strip()
    return overrides.sort_values("date").reset_index(drop=True)


def apply_result_overrides(raw_results: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Apply manual score overrides to raw results/fixtures."""
    if overrides.empty:
        return raw_results.copy()

    results = raw_results.copy()
    results["date"] = pd.to_datetime(results["date"])

    for override in overrides.itertuples(index=False):
        match_mask = (
            (results["date"] == override.date)
            & (results["home_team"] == override.home_team)
            & (results["away_team"] == override.away_team)
        )
        if not match_mask.any():
            raise ValueError(
                "Override match not found in raw results: "
                f"{override.date.date()} {override.home_team} vs {override.away_team}"
            )

        results.loc[match_mask, "home_score"] = int(override.home_score)
        results.loc[match_mask, "away_score"] = int(override.away_score)

    return results.sort_values("date").reset_index(drop=True)
