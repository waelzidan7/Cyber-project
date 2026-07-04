"""Modelling pipelines: the intentionally flawed reproduction and the corrected,
leakage-free pipeline.

The contrast between these two is the empirical core of the project. The flawed
path mirrors the common tutorial mistake of resampling before splitting; the
corrected path applies SMOTE strictly inside cross-validation folds and never
touches the held-out test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .config import RANDOM_STATE


def supervised_estimators() -> dict:
    """Return the supervised estimators compared in the corrected pipeline.

    ``class_weight='balanced'`` is intentionally omitted because SMOTE already
    rebalances the training folds; combining both would double-count the
    minority class.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def build_corrected_pipeline(estimator) -> ImbPipeline:
    """Wrap an estimator in a leakage-free pipeline.

    Order matters: RobustScaler and SMOTE are fitted on the training portion
    only (whether that is a CV fold or the final train split), so no information
    from the validation/test data is ever used to transform the training data.
    RobustScaler is chosen over StandardScaler because it centres on the median
    and scales by the IQR, which resists the heavy outliers in ``Amount``.
    """
    return ImbPipeline(
        steps=[
            ("scaler", RobustScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", estimator),
        ]
    )


def stratified_split(X, y, test_size: float = 0.2):
    """Stratified train/test split that preserves the fraud prevalence."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def fit_predict_proba(pipeline, X_train, y_train, X_test):
    """Fit a probabilistic pipeline and return ``(y_pred, y_score)``."""
    pipeline.fit(X_train, y_train)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    return y_pred, y_score


def flawed_reproduction(X, y, estimator=None):
    """Reproduce the tutorial's mistake: SMOTE the whole dataset *before*
    splitting, then evaluate on the resampled (synthetic-contaminated) test set.

    This is the configuration that produces the headline "near-perfect" scores.
    It is included precisely so we can measure how much of the reported skill is
    an artefact. Returns ``(y_test, y_pred, y_score)`` on the leaked test set.
    """
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        )
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_scaled, y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=RANDOM_STATE
    )
    estimator.fit(X_tr, y_tr)
    y_score = estimator.predict_proba(X_te)[:, 1]
    y_pred = estimator.predict(X_te)
    return np.asarray(y_te), y_pred, y_score


def isolation_forest_scores(X_train, X_test, contamination: float):
    """Train an unsupervised Isolation Forest and score the test set.

    Isolation Forest is trained without labels (it isolates anomalies via random
    partitioning). We return an anomaly score in [0, 1]-ish orientation where a
    higher value means *more anomalous*, so it can feed the same metric helpers
    as the supervised models. ``contamination`` is set to the observed fraud
    prevalence so the default decision threshold is sensible.
    """
    detector = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    detector.fit(X_train)
    # score_samples: higher = more normal. Negate so higher = more anomalous.
    anomaly_score = -detector.score_samples(X_test)
    # predict: -1 anomaly, 1 normal -> map to {1 fraud, 0 legit}.
    y_pred = (detector.predict(X_test) == -1).astype(int)
    return y_pred, anomaly_score
