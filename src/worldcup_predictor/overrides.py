from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_OVERRIDE_COLUMNS = {"date", "home_team", "away_team", "home_score", "away_score"}
REQUIRED_CONTEXT_OVERRIDE_COLUMNS = {"date", "home_team", "away_team"}
CONTEXT_MULTIPLIER_COLUMNS = (
    "home_attack_multiplier",
    "home_defense_multiplier",
    "away_attack_multiplier",
    "away_defense_multiplier",
    "draw_probability_multiplier",
)
CONTEXT_TEXT_COLUMNS = ("confidence", "notes")


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


def load_match_context_overrides(path: str | Path) -> pd.DataFrame:
    """Load manual fixture context adjustments, returning an empty frame when absent."""
    path = Path(path)
    if not path.exists():
        return _empty_context_overrides()

    overrides = pd.read_csv(path, parse_dates=["date"])
    missing_columns = REQUIRED_CONTEXT_OVERRIDE_COLUMNS.difference(overrides.columns)
    if missing_columns:
        raise ValueError(f"Missing context override columns: {sorted(missing_columns)}")

    overrides = overrides.dropna(subset=["date", "home_team", "away_team"]).copy()
    if overrides.empty:
        return _empty_context_overrides()

    for column in CONTEXT_MULTIPLIER_COLUMNS:
        if column not in overrides:
            overrides[column] = 1.0
        overrides[column] = pd.to_numeric(overrides[column], errors="coerce").fillna(1.0).astype(float)
        if (overrides[column] <= 0).any():
            raise ValueError(f"Context override multiplier must be greater than 0: {column}")

    for column in CONTEXT_TEXT_COLUMNS:
        if column not in overrides:
            overrides[column] = ""
        overrides[column] = overrides[column].fillna("").astype(str).str.strip()

    overrides["home_team"] = overrides["home_team"].astype(str).str.strip()
    overrides["away_team"] = overrides["away_team"].astype(str).str.strip()

    duplicated = overrides.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    if duplicated.any():
        duplicate = overrides.loc[duplicated].iloc[0]
        raise ValueError(
            "Duplicate context override for fixture: "
            f"{duplicate.date.date()} {duplicate.home_team} vs {duplicate.away_team}"
        )

    columns = [
        "date",
        "home_team",
        "away_team",
        *CONTEXT_MULTIPLIER_COLUMNS,
        *CONTEXT_TEXT_COLUMNS,
    ]
    return overrides.loc[:, columns].sort_values("date").reset_index(drop=True)


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
