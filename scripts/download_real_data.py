from pathlib import Path

from worldcup_predictor.data_sources import download_international_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = download_international_results(PROJECT_ROOT / "data" / "raw", overwrite=True)
    for path in paths:
        print(f"Downloaded {path}")


if __name__ == "__main__":
    main()
