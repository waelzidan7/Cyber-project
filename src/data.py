"""Dataset download, loading, and cleaning."""

from __future__ import annotations

import urllib.request

import pandas as pd

from .config import DATA_PATH, DATA_URL


def download_data(url: str = DATA_URL, dest=DATA_PATH) -> None:
    """Download the dataset to ``dest`` if it is not already present.

    The CSV is ~102 MB and is therefore not stored in the Git repository; this
    function fetches it on demand from a public, no-authentication mirror.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urllib.request.urlretrieve(url, dest)


def load_data(path=DATA_PATH, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load the dataset, optionally removing exact duplicate rows.

    The raw file contains 1,081 fully duplicated rows. Because the features are
    anonymised principal components, two rows being identical across all 30
    columns means the same transaction was recorded twice; keeping them would
    leak identical records across the train/test boundary and inflate scores,
    so they are dropped by default.
    """
    frame = pd.read_csv(path)
    if drop_duplicates:
        frame = frame.drop_duplicates().reset_index(drop=True)
    return frame


def split_features_target(frame: pd.DataFrame, target: str = "Class"):
    """Return ``(X, y)`` with the target column separated from the features."""
    feature_matrix = frame.drop(columns=[target])
    target_vector = frame[target]
    return feature_matrix, target_vector
