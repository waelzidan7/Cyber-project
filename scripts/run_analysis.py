"""End-to-end driver used during development to validate the pipeline and to
capture the real numbers that the notebook and report rely on.

Run: python scripts/run_analysis.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402

from src import data, eda, evaluate, features, models  # noqa: E402
from src.config import PCA_FEATURES, RANDOM_STATE, TARGET  # noqa: E402

pd.set_option("display.width", 140)


def main() -> None:
    summary = {}

    # ---- Load & clean ---------------------------------------------------- #
    raw = data.load_data(drop_duplicates=False)
    clean = data.load_data(drop_duplicates=True)
    summary["rows_raw"] = len(raw)
    summary["rows_clean"] = len(clean)
    summary["duplicates_removed"] = len(raw) - len(clean)
    summary["n_features"] = clean.shape[1] - 1
    summary["n_missing"] = int(clean.isnull().sum().sum())

    balance = eda.class_balance(clean)
    summary["fraud_count"] = int(balance.loc[1, "count"])
    summary["fraud_prevalence_pct"] = round(100 * balance.loc[1, "prevalence"], 4)

    # ---- Feature engineering (row-wise only) ----------------------------- #
    engineered = features.add_hour_of_day(clean)
    engineered = features.add_log_amount(engineered)

    # Outlier shares for a couple of representative columns.
    summary["outlier_share_amount"] = round(eda.outlier_share_iqr(clean["Amount"]), 4)

    # Correlation with target (Spearman).
    corr = eda.correlation_with_target(clean, method="spearman")
    summary["top_spearman_corr"] = corr.head(5).round(3).to_dict()

    # Use the same explicit feature ordering as the notebook so the two agree
    # exactly (tree-ensemble feature subsampling depends on column order).
    feature_columns = PCA_FEATURES + ["Amount", "Time", "Hour", "LogAmount"]
    X = engineered[feature_columns]
    y = engineered[TARGET]

    # ---- Flawed reproduction (SMOTE before split) ------------------------ #
    yt_f, yp_f, ys_f = models.flawed_reproduction(X, y)
    summary["flawed_metrics"] = {
        k: round(v, 4) for k, v in evaluate.classification_metrics(yt_f, yp_f, ys_f).items()
    }

    # ---- Corrected pipeline --------------------------------------------- #
    X_train, X_test, y_train, y_test = models.stratified_split(X, y)
    summary["train_size"] = len(X_train)
    summary["test_size"] = len(X_test)
    summary["test_fraud"] = int(y_test.sum())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["average_precision", "roc_auc", "recall", "precision"]

    corrected_results = {}
    pr_curves = {}
    cv_summary = {}
    for name, estimator in models.supervised_estimators().items():
        pipeline = models.build_corrected_pipeline(estimator)
        # Cross-validated scores on the training data (leakage-free).
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_summary[name] = {
            "cv_average_precision": round(scores["test_average_precision"].mean(), 4),
            "cv_roc_auc": round(scores["test_roc_auc"].mean(), 4),
        }
        # Refit on full training set, evaluate on untouched test set.
        y_pred, y_score = models.fit_predict_proba(pipeline, X_train, y_train, X_test)
        corrected_results[name] = evaluate.classification_metrics(y_test, y_pred, y_score)
        pr_curves[name] = (y_test.values, y_score)

    # Isolation Forest (unsupervised) on the same test set.
    prevalence = float(y_train.mean())
    yp_if, ys_if = models.isolation_forest_scores(X_train, X_test, contamination=prevalence)
    corrected_results["Isolation Forest"] = evaluate.classification_metrics(y_test, yp_if, ys_if)
    pr_curves["Isolation Forest"] = (y_test.values, ys_if)

    summary["cv_summary"] = cv_summary
    summary["corrected_metrics"] = {
        name: {k: round(v, 4) for k, v in m.items()} for name, m in corrected_results.items()
    }

    # ---- Cost / threshold analysis on best model (Random Forest) -------- #
    rf_pipeline = models.build_corrected_pipeline(models.supervised_estimators()["Random Forest"])
    _, rf_score = models.fit_predict_proba(rf_pipeline, X_train, y_train, X_test)
    cost_table = evaluate.threshold_cost_analysis(y_test, rf_score)
    best = cost_table.loc[cost_table["cost"].idxmin()]
    summary["rf_default_threshold_0.5"] = {
        "recall": round(float(cost_table.iloc[49]["recall"]), 4),
        "precision": round(float(cost_table.iloc[49]["precision"]), 4),
    }
    summary["rf_cost_optimal"] = {
        "threshold": round(float(best["threshold"]), 2),
        "recall": round(float(best["recall"]), 4),
        "precision": round(float(best["precision"]), 4),
        "fn": int(best["fn"]), "fp": int(best["fp"]),
    }

    print(json.dumps(summary, indent=2, default=str))

    # Persist for the report.
    out = Path(__file__).resolve().parents[1] / "reports" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()
