"""Download the credit-card fraud dataset into ``data/creditcard.csv``.

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PATH  # noqa: E402
from src.data import download_data  # noqa: E402


def main() -> None:
    print(f"Downloading dataset to {DATA_PATH} ...")
    download_data()
    size_mb = DATA_PATH.stat().st_size / 1e6
    print(f"Done. {DATA_PATH.name} is {size_mb:.1f} MB.")


if __name__ == "__main__":
    main()
