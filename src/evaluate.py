"""Evaluation metrics, curves, and threshold/cost analysis.

Metric choices are driven by the extreme class imbalance: accuracy and even
ROC-AUC are over-optimistic when the negative class dominates, so the headline
metrics here are Average Precision (PR-AUC), MCC, and the recall-weighted F2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .eda import _save


def classification_metrics(y_true, y_pred, y_score) -> dict:
    """Return the full suite of metrics for one model.

    ``accuracy`` is reported deliberately so the report can show that it is
    misleadingly high; the decision-quality metrics are the ones that matter.
    """
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    """Turn ``{model_name: metrics_dict}`` into a tidy, rounded DataFrame."""
    return pd.DataFrame(results).T.round(4)


def plot_confusion(y_true, y_pred, title: str, fname: str) -> str:
    """Plot a 2x2 confusion matrix with raw counts."""
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred legit", "Pred fraud"])
    ax.set_yticks([0, 1], labels=["True legit", "True fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="black", fontsize=11)
    ax.set_title(title)
    return _save(fig, fname)


def plot_pr_curves(curves: dict, fname: str = "pr_curves.png") -> str:
    """Overlay precision-recall curves for several models.

    ``curves`` maps model name -> (y_true, y_score).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    baseline = float(np.mean(list(curves.values())[0][0]))
    ax.axhline(baseline, ls="--", color="grey", label=f"baseline (prevalence={baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (corrected pipeline)")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, fname)


def plot_roc_curves(curves: dict, fname: str = "roc_curves.png") -> str:
    """Overlay ROC curves for several models. ``curves`` maps name -> (y_true, y_score)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (corrected pipeline)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, fname)


def threshold_cost_analysis(
    y_true, y_score, fn_cost: float = 100.0, fp_cost: float = 1.0
) -> pd.DataFrame:
    """Sweep decision thresholds and compute confusion counts plus a cost.

    The asymmetric default cost (a missed fraud is 100x worse than a false
    alarm) encodes the cybersecurity reality and lets us locate the threshold
    that minimises expected business cost rather than maximising accuracy.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    y_true = np.asarray(y_true)
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        rows.append(
            {
                "threshold": threshold,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "cost": fn_cost * fn + fp_cost * fp,
            }
        )
    return pd.DataFrame(rows)


def plot_cost_curve(cost_table: pd.DataFrame, fname: str = "cost_curve.png") -> str:
    """Plot expected cost against decision threshold and mark the minimum."""
    best = cost_table.loc[cost_table["cost"].idxmin()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cost_table["threshold"], cost_table["cost"], color="#4c72b0")
    ax.axvline(best["threshold"], ls="--", color="#c44e52",
               label=f"min cost @ threshold={best['threshold']:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost (100*FN + 1*FP)")
    ax.set_title("Cost-sensitive threshold selection")
    ax.legend()
    return _save(fig, fname)
