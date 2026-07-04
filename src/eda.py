"""Exploratory data analysis helpers and plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AMOUNT, FIGURES_DIR, TARGET, TIME


def class_balance(frame: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Return per-class counts and prevalence (share of total)."""
    counts = frame[target].value_counts().sort_index()
    prevalence = counts / counts.sum()
    return pd.DataFrame({"count": counts, "prevalence": prevalence})


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the count and percentage of missing values per column."""
    missing = frame.isnull().sum()
    return pd.DataFrame(
        {"missing": missing, "pct": 100 * missing / len(frame)}
    ).query("missing > 0")


def outlier_share_iqr(series: pd.Series) -> float:
    """Return the fraction of points outside the 1.5*IQR Tukey fences."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((series < lower) | (series > upper)).mean())


def fraud_rate_by_hour(frame: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Return transaction count and fraud rate for each hour of day.

    Requires an ``Hour`` column (see ``features.add_hour_of_day``).
    """
    grouped = frame.groupby("Hour")[target]
    return pd.DataFrame({"n": grouped.size(), "fraud_rate": grouped.mean()})


def correlation_with_target(
    frame: pd.DataFrame, target: str = TARGET, method: str = "spearman"
) -> pd.Series:
    """Return features ranked by absolute correlation with the target.

    Spearman is the default: it measures monotonic association via ranks, so it
    is robust to the heavy skew of ``Amount`` and to the non-Gaussian, outlier-
    rich principal components, and it does not assume linearity the way Pearson
    does.
    """
    correlations = frame.corr(method=method)[target].drop(target)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------------- #
# Plotting helpers. Each returns the saved figure path so the notebook and the
# report can reference the same artefacts.
# --------------------------------------------------------------------------- #

def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return str(path)


def plot_class_balance(frame: pd.DataFrame, target: str = TARGET) -> str:
    balance = class_balance(frame, target)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legit (0)", "Fraud (1)"], balance["count"], color=["#4c72b0", "#c44e52"])
    ax.set_yscale("log")
    ax.set_ylabel("Number of transactions (log scale)")
    ax.set_title(
        f"Class imbalance: fraud is {100 * balance['prevalence'].iloc[1]:.3f}% of data"
    )
    for i, value in enumerate(balance["count"]):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    return _save(fig, "class_balance.png")


def plot_amount_distribution(frame: pd.DataFrame, target: str = TARGET) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(frame[AMOUNT], bins=80, color="#4c72b0")
    axes[0].set_title("Amount (raw, right-skewed)")
    axes[0].set_xlabel("Amount")
    axes[1].hist(np.log1p(frame[AMOUNT]), bins=80, color="#55a868")
    axes[1].set_title("log1p(Amount) (compressed tail)")
    axes[1].set_xlabel("log1p(Amount)")
    return _save(fig, "amount_distribution.png")


def plot_fraud_rate_by_hour(frame: pd.DataFrame, target: str = TARGET) -> str:
    table = fraud_rate_by_hour(frame, target)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(table.index, table["n"], color="#c7c7c7", label="transactions")
    ax1.set_xlabel("Hour of day (derived from Time)")
    ax1.set_ylabel("Transaction count")
    ax2 = ax1.twinx()
    ax2.plot(table.index, 100 * table["fraud_rate"], color="#c44e52", marker="o", label="fraud rate")
    ax2.set_ylabel("Fraud rate (%)", color="#c44e52")
    ax1.set_title("Transaction volume and fraud rate by hour of day")
    return _save(fig, "fraud_rate_by_hour.png")


def plot_correlation_with_target(frame: pd.DataFrame, target: str = TARGET, top: int = 12) -> str:
    correlations = correlation_with_target(frame, target).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#c44e52" if v < 0 else "#4c72b0" for v in correlations.values]
    ax.barh(correlations.index, correlations.values, color=colors)
    ax.set_title(f"Top {top} features by |Spearman correlation| with fraud")
    ax.set_xlabel("Spearman correlation with Class")
    return _save(fig, "correlation_with_target.png")
