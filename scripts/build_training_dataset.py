from pathlib import Path

from worldcup_predictor.datasets import build_match_features, save_match_features
from worldcup_predictor.future import completed_matches, load_raw_results_with_fixtures
from worldcup_predictor.overrides import apply_result_overrides, load_result_overrides


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results_path = PROJECT_ROOT / "data" / "raw" / "results.csv"
    overrides_path = PROJECT_ROOT / "data" / "manual" / "match_results_overrides.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "matches_with_features.csv"

    raw_results = load_raw_results_with_fixtures(results_path)
    overrides = load_result_overrides(overrides_path)
    results = completed_matches(apply_result_overrides(raw_results, overrides))
    features = build_match_features(results)
    save_match_features(features, output_path)

    print(f"Raw matches: {len(results)}")
    print(f"Manual overrides applied: {len(overrides)}")
    print(f"Processed matches: {len(features)}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
