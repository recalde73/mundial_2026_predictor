from worldcup_predictor.formatting import format_recommendation
from worldcup_predictor.recommender import recommend_pick


def main() -> None:
    recommendation = recommend_pick(
        home_team="Argentina",
        away_team="Dinamarca",
        home_expected_goals=1.72,
        away_expected_goals=0.91,
    )

    print(format_recommendation(recommendation))


if __name__ == "__main__":
    main()
