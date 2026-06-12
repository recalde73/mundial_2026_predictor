from __future__ import annotations

from worldcup_predictor.recommender import PickRecommendation
from worldcup_predictor.scoring import Scoreline


def format_scoreline(scoreline: Scoreline) -> str:
    return f"{scoreline[0]}-{scoreline[1]}"


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_recommendation(recommendation: PickRecommendation) -> str:
    result_probs = recommendation.result_probabilities
    lines = [
        f"{recommendation.home_team} vs {recommendation.away_team}",
        "",
        "Goles esperados:",
        f"- {recommendation.home_team}: {recommendation.home_expected_goals:.2f}",
        f"- {recommendation.away_team}: {recommendation.away_expected_goals:.2f}",
        "",
        "Probabilidades 1X2:",
        f"- Gana {recommendation.home_team}: {format_percentage(result_probs['home_win'])}",
        f"- Empate: {format_percentage(result_probs['draw'])}",
        f"- Gana {recommendation.away_team}: {format_percentage(result_probs['away_win'])}",
        "",
        "Marcadores mas probables:",
    ]

    lines.extend(
        f"- {format_scoreline(scoreline)}: {format_percentage(probability)}"
        for scoreline, probability in recommendation.most_likely_scorelines
    )

    lines.extend(
        [
            "",
            "Pick recomendado por valor esperado:",
            f"- {format_scoreline(recommendation.recommended_scoreline)} "
            f"({recommendation.expected_points:.2f} pts esperados)",
            "",
            "Alternativas:",
        ]
    )

    lines.extend(
        f"- {format_scoreline(scoreline)}: {value:.2f} pts esperados"
        for scoreline, value in recommendation.alternatives[:5]
    )

    return "\n".join(lines)
