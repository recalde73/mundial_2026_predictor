from pathlib import Path

from worldcup_predictor.formatting import format_recommendation
from worldcup_predictor.pipeline import recommend_match_from_ratings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    recommendation = recommend_match_from_ratings(
        home_team="Argentina",
        away_team="Dinamarca",
        ratings_path=PROJECT_ROOT / "data" / "raw" / "team_ratings_example.csv",
    )

    print(format_recommendation(recommendation))


if __name__ == "__main__":
    main()
