from worldcup_predictor.recommender import adjust_draw_probabilities, rank_predictions, recommend_pick


def test_rank_predictions_prefers_draw_when_draw_probability_is_high() -> None:
    probabilities = {
        (0, 0): 0.20,
        (1, 1): 0.30,
        (2, 2): 0.10,
        (1, 0): 0.20,
        (0, 1): 0.20,
    }

    ranked = rank_predictions(
        probabilities,
        max_candidate_goals=2,
        max_total_candidate_goals=4,
        draw_probability_multiplier=1.0,
    )

    assert ranked[0][0] == (1, 1)


def test_rank_predictions_can_filter_high_total_goal_candidates() -> None:
    probabilities = {
        (1, 0): 0.30,
        (2, 1): 0.20,
        (4, 3): 0.10,
        (0, 0): 0.40,
    }

    ranked = rank_predictions(
        probabilities,
        max_candidate_goals=4,
        max_total_candidate_goals=5,
        draw_probability_multiplier=1.0,
    )

    assert (4, 3) not in [scoreline for scoreline, _value in ranked]


def test_adjust_draw_probabilities_boosts_draw_scorelines() -> None:
    probabilities = {(1, 1): 0.25, (1, 0): 0.75}

    adjusted = adjust_draw_probabilities(probabilities, draw_probability_multiplier=2.0)

    assert adjusted[(1, 1)] > probabilities[(1, 1)]
    assert round(sum(adjusted.values()), 10) == 1.0


def test_recommend_pick_returns_expected_fields() -> None:
    recommendation = recommend_pick(
        home_team="Argentina",
        away_team="Dinamarca",
        home_expected_goals=1.72,
        away_expected_goals=0.91,
    )

    assert recommendation.home_team == "Argentina"
    assert recommendation.away_team == "Dinamarca"
    assert recommendation.expected_points > 0
    assert recommendation.alternatives[0][0] == recommendation.recommended_scoreline
    assert set(recommendation.result_probabilities) == {"home_win", "draw", "away_win"}
