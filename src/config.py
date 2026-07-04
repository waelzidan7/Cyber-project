"""Shared constants used across the project.

Keeping these in one place guarantees that every module, the notebook, and the
report all use exactly the same random seed and column conventions, which is a
prerequisite for reproducibility.
"""

from pathlib import Path

# Fixed seed for every stochastic step (split, SMOTE, model init) so results
# are identical on every run and on every machine.
RANDOM_STATE = 42

# Project paths (resolved relative to the repository root, not the caller).
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "creditcard.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Direct (no-authentication) mirror of the Kaggle / ULB dataset.
DATA_URL = (
    "https://raw.githubusercontent.com/nsethi31/"
    "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
)

# Column conventions for the dataset.
TARGET = "Class"
AMOUNT = "Amount"
TIME = "Time"
# The 28 anonymised principal components produced by the data publisher.
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

# Fraud is the positive (minority) class.
POSITIVE_LABEL = 1
