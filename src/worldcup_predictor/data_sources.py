from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class DataSource:
    filename: str
    url: str
    description: str


INTERNATIONAL_RESULTS_SOURCES = (
    DataSource(
        filename="results.csv",
        url="https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        description="Historical international football match results",
    ),
    DataSource(
        filename="goalscorers.csv",
        url="https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
        description="Historical international goal scorers",
    ),
    DataSource(
        filename="shootouts.csv",
        url="https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
        description="Historical international penalty shootouts",
    ),
)


def download_file(url: str, destination: str | Path, overwrite: bool = False) -> Path:
    """Download one URL to destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        return destination

    with urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())

    return destination


def download_international_results(
    raw_data_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Download the public international results dataset."""
    raw_data_dir = Path(raw_data_dir)
    return [
        download_file(source.url, raw_data_dir / source.filename, overwrite=overwrite)
        for source in INTERNATIONAL_RESULTS_SOURCES
    ]
