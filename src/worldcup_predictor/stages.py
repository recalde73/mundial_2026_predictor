from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


GROUP_STAGE_MATCHES = 72
ROUND_OF_32_MATCHES = 16
ROUND_OF_16_MATCHES = 8
QUARTERFINAL_MATCHES = 4
SEMIFINAL_MATCHES = 2
THIRD_PLACE_MATCHES = 1

GROUP_STAGE = "group_stage"
ROUND_OF_32 = "round_of_32"
ROUND_OF_16 = "round_of_16"
QUARTERFINAL = "quarterfinal"
SEMIFINAL = "semifinal"
THIRD_PLACE = "third_place"
FINAL = "final"

KNOCKOUT_STAGES = {
    ROUND_OF_32,
    ROUND_OF_16,
    QUARTERFINAL,
    SEMIFINAL,
    THIRD_PLACE,
    FINAL,
}

STAGE_ORDER = (
    (GROUP_STAGE_MATCHES, GROUP_STAGE),
    (ROUND_OF_32_MATCHES, ROUND_OF_32),
    (ROUND_OF_16_MATCHES, ROUND_OF_16),
    (QUARTERFINAL_MATCHES, QUARTERFINAL),
    (SEMIFINAL_MATCHES, SEMIFINAL),
    (THIRD_PLACE_MATCHES, THIRD_PLACE),
)


def tournament_stage_for_match_number(match_number: int) -> str:
    """Return the 2026 World Cup stage for a 1-based match number."""
    if match_number < 1:
        raise ValueError("match_number must be greater than or equal to 1")

    upper_bound = 0
    for match_count, stage in STAGE_ORDER:
        upper_bound += match_count
        if match_number <= upper_bound:
            return stage
    return FINAL


def is_knockout_stage(stage: str) -> bool:
    return stage in KNOCKOUT_STAGES


def add_tournament_stages(
    matches: pd.DataFrame,
    tournament: str = "FIFA World Cup",
    start_date: str = "2026-06-01",
    order_columns: Iterable[str] = ("date", "home_team", "away_team"),
) -> pd.DataFrame:
    """Add tournament-stage columns without removing existing data."""
    staged = matches.copy()
    staged["tournament_stage"] = ""
    staged["is_group_stage"] = False
    staged["is_knockout_stage"] = False
    staged["is_extra_time_possible"] = False
    staged["is_penalty_possible"] = False

    if staged.empty:
        return staged

    stage_mask = (staged["tournament"] == tournament) & (staged["date"] >= pd.Timestamp(start_date))
    ordered_index = staged.loc[stage_mask].sort_values(list(order_columns)).index
    for ordinal, index in enumerate(ordered_index, start=1):
        stage = tournament_stage_for_match_number(ordinal)
        knockout = is_knockout_stage(stage)
        staged.at[index, "tournament_stage"] = stage
        staged.at[index, "is_group_stage"] = stage == GROUP_STAGE
        staged.at[index, "is_knockout_stage"] = knockout
        staged.at[index, "is_extra_time_possible"] = knockout
        staged.at[index, "is_penalty_possible"] = knockout

    return staged
