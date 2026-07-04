# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Critical Review of a Credit-Card Fraud Detection Tutorial
#
# **Course:** Data Science in Cyber — Final Project
#
# **Selected source:** *Enhancing Credit Card Fraud Detection Models using Machine
# Learning and PCA* (public GitHub project,
# `BalaElangovan/Enhancing-Credit-Card-Fraud-Detection-Models-using-Machine-Learning-and-PCA`).
# The project balances the data with **SMOTE**, reduces dimensionality with **PCA**,
# and compares Logistic Regression, Random Forest, and a Neural Network. It reports
# **AUC-ROC ≈ 1.00** for Random Forest and the Neural Network and 0.9923 for Logistic
# Regression.
#
# **Dataset:** Kaggle / ULB *Credit Card Fraud Detection* (Dal Pozzolo et al., 2015):
# 284,807 European card transactions, 492 fraudulent (0.172%). Features `Time`,
# `V1`–`V28` (PCA-anonymised), `Amount`, and the binary target `Class`.
#
# **Central question of this notebook.** Is the reported near-perfect performance a
# genuine property of the models, or an artefact of methodology? Our thesis is the
# latter: resampling applied **before** the train/test split leaks information, and
# accuracy/ROC-AUC are the wrong lenses for a 0.172%-prevalence problem. We
# (1) reproduce the flawed pipeline to show the inflated scores, then (2) build a
# leakage-free pipeline and report honest metrics (PR-AUC, MCC, F2) and the
# false-positive / false-negative trade-off that matters operationally.
#
# All randomness uses a fixed seed (`RANDOM_STATE = 42`). Heavy logic lives in the
# documented helper modules under `src/`, keeping this notebook a readable narrative.

# %%
import sys
from pathlib import Path

# Make the project root importable whether the notebook is run from the repo root
# or from a subfolder.
ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import data, eda, evaluate, features, models
from src.config import RANDOM_STATE, TARGET, PCA_FEATURES

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
np.random.seed(RANDOM_STATE)
print("Environment ready. RANDOM_STATE =", RANDOM_STATE)

# %% [markdown]
# ## 1. Data Loading and Inspection
#
# We first download the dataset on demand (it is too large to store in Git), then
# load it and inspect its structure, types, and integrity.

# %%
data.download_data()  # no-op if data/creditcard.csv already exists

raw = data.load_data(drop_duplicates=False)
print("Raw shape:", raw.shape)
raw.head()

# %% [markdown]
# ### Feature types, index, and column names
#
# The index is a default `RangeIndex` (no meaningful key such as a transaction id),
# and every column is numeric. The column names are uninformative by design: the
# publisher applied PCA and released only the components `V1`–`V28` to protect
# cardholder privacy. The only two human-readable features are `Time` (seconds since
# the first transaction) and `Amount`. This is sensible for a privacy-preserving
# release, but it severely limits semantic feature engineering — a point we return to
# in the report.

# %%
print("Dtypes:\n", raw.dtypes.value_counts())
print("\nColumns:", list(raw.columns))
raw.describe().T[["mean", "std", "min", "50%", "max"]]

# %% [markdown]
# ### Missing values, duplicates, and single-value features
#
# There are **no missing values**. There are, however, **1,081 fully duplicated
# rows**. Because the features are anonymised principal components, two rows being
# identical across all 30 columns indicates the same transaction recorded twice.
# Keeping duplicates risks the identical record appearing on both sides of the
# train/test split (a subtle leak), so we drop them. No column is constant
# (single-valued), so none is dropped on that basis.

# %%
print("Missing values total:", int(raw.isnull().sum().sum()))
print("Exact duplicate rows:", int(raw.duplicated().sum()))
print("Constant (single-value) columns:", [c for c in raw.columns if raw[c].nunique() == 1])

df = data.load_data(drop_duplicates=True)
print("\nShape after dropping duplicates:", df.shape)

# %% [markdown]
# ### Temporal coverage
#
# `Time` runs from 0 to 172,792 seconds — almost exactly **48 hours**. The dataset is
# therefore a two-day window of one bank's traffic. This matters: any temporal pattern
# we find (e.g. a day/night cycle) is estimated from only two days and should be
# treated cautiously.

# %%
print("Time min/max (seconds):", df["Time"].min(), df["Time"].max())
print("Span in hours:", round(df["Time"].max() / 3600, 1))
print("Span in days:", round(df["Time"].max() / 86400, 2))

# %% [markdown]
# ## 2. Exploratory Data Analysis
#
# ### 2.1 Class imbalance (prevalence)
#
# Fraud is **0.17%** of all transactions. In real terms a naive classifier that always
# predicts "legitimate" would be 99.83% accurate while catching zero fraud — which is
# exactly why accuracy is the wrong headline metric here. The imbalance also means the
# *positive* class is data-starved: with so few frauds, variance of any fraud-specific
# estimate is high. The original tutorial addresses this with SMOTE; we will show that
# *where* SMOTE is applied determines whether it helps or silently cheats.

# %%
balance = eda.class_balance(df)
print(balance)
print("\nFraud prevalence: %.4f%%" % (100 * balance.loc[1, "prevalence"]))
eda.plot_class_balance(df)

# %% [markdown]
# ![class balance](reports/figures/class_balance.png)

# %% [markdown]
# ### 2.2 Feature distributions and outliers
#
# `Amount` is strongly right-skewed (median 22, max ~25,700). A `log1p` transform
# tames the tail, which helps linear/distance-based models and makes the distribution
# legible. Under the 1.5×IQR Tukey rule, ~11% of `Amount` values are "outliers" — but
# in fraud detection extreme values are signal, not noise, so we must scale robustly
# rather than clip.

# %%
print("Amount summary:\n", df["Amount"].describe())
print("\nIQR-outlier share of Amount: %.3f" % eda.outlier_share_iqr(df["Amount"]))
eda.plot_amount_distribution(df)

# %% [markdown]
# ![amount distribution](reports/figures/amount_distribution.png)

# %% [markdown]
# ### 2.3 Temporal analysis: fraud rate by hour of day
#
# Converting `Time` to hour-of-day reveals that legitimate traffic follows a clear
# day/night cycle, while fraud is comparatively flatter — so the *fraud rate* spikes
# during the quiet overnight hours. This aligns with world knowledge: attackers
# operate when monitoring is thin and victims are asleep. (Caveat: only two days of
# data, so this is suggestive, not conclusive.)

# %%
df = features.add_hour_of_day(df)
df = features.add_log_amount(df)
print(eda.fraud_rate_by_hour(df))
eda.plot_fraud_rate_by_hour(df)

# %% [markdown]
# ![fraud rate by hour](reports/figures/fraud_rate_by_hour.png)

# %% [markdown]
# ### 2.4 Correlation analysis — and why Spearman
#
# We studied Pearson, Spearman, and Kendall correlation. The choice matters:
#
# - **Pearson** measures *linear* association and assumes roughly normal, outlier-free
#   data. `Amount` is heavily skewed and the `V` features are not Gaussian, so Pearson
#   is a poor fit.
# - **Spearman** measures *monotonic* association via ranks. It is robust to skew and
#   outliers and makes no linearity assumption — the best general-purpose choice here.
# - **Kendall** is also rank-based and preferable for very small samples or many ties;
#   with 280k rows it mostly agrees with Spearman but is far more expensive to compute.
#
# We therefore use **Spearman** as the primary measure. The correlations of individual
# features with `Class` are *small in absolute terms* (top |ρ| ≈ 0.06). This is an
# important, practically significant finding: **no single feature separates fraud**,
# so the task is inherently multivariate and linear models will struggle relative to
# tree ensembles that capture interactions. Statistical significance (large *n* makes
# almost everything "significant") is not the same as practical significance.

# %%
spearman = eda.correlation_with_target(df.drop(columns=["Hour", "LogAmount"]), method="spearman")
print("Top features by |Spearman correlation| with Class:")
print(spearman.head(10).round(4))
eda.plot_correlation_with_target(df.drop(columns=["Hour", "LogAmount"]))

# %% [markdown]
# ![correlation with target](reports/figures/correlation_with_target.png)

# %% [markdown]
# ### 2.5 Redundancy check
#
# Because `V1`–`V28` are principal components, they are **mutually orthogonal by
# construction**, so we expect almost no inter-feature redundancy among them. We verify
# this: the largest absolute off-diagonal Spearman correlation among the `V` features
# is tiny. The only redundancy we introduce ourselves is `Amount` vs `LogAmount` (a
# deterministic transform) and `Time` vs `Hour`; we keep one representation per concept
# in modelling to avoid duplicated signal.

# %%
v_corr = df[PCA_FEATURES].corr(method="spearman").abs()
np.fill_diagonal(v_corr.values, 0)
print("Max |Spearman| correlation between any two V-features: %.4f" % v_corr.values.max())

# %% [markdown]
# ## 3. Feature Engineering
#
# The dataset is already heavily engineered by the publisher (PCA on the original,
# undisclosed features), which constrains what we can add:
#
# - **Scaling.** `Amount` and `Time` are on very different scales from the unit-variance
#   `V` components. We scale with **RobustScaler** (median/IQR) because it resists the
#   `Amount` outliers that a StandardScaler would let dominate. Crucially, scaling is
#   placed *inside* the modelling pipeline so it is fitted on training folds only.
# - **`LogAmount`** = `log1p(Amount)` to compress the skew.
# - **`Hour`** from `Time` to expose the daily cycle (Section 2.3).
# - **Encoding.** There are no categorical features, so one-hot / target encoding are
#   not applicable; we explain this in the report rather than forcing an encoder.
# - **Dimensionality reduction.** PCA is already applied; re-applying it (as the source
#   does) is redundant and, when fitted on the full data, is another leakage vector.

# %%
feature_columns = PCA_FEATURES + ["Amount", "Time", "Hour", "LogAmount"]
X = df[feature_columns]
y = df[TARGET]
print("Feature matrix shape:", X.shape)
print("Engineered columns added:", ["Hour", "LogAmount"])

# %% [markdown]
# ## 4. Model Training
#
# ### 4.1 Flawed reproduction — SMOTE applied *before* the split
#
# We first reproduce the methodology that yields the headline numbers: scale and
# **SMOTE the entire dataset**, *then* split into train/test and evaluate. Because
# SMOTE interpolates new minority points from existing ones across the whole dataset,
# synthetic neighbours of test-set frauds end up in training, and the test set is no
# longer an honest held-out sample.

# %%
y_true_flawed, y_pred_flawed, y_score_flawed = models.flawed_reproduction(X, y)
flawed_metrics = evaluate.classification_metrics(y_true_flawed, y_pred_flawed, y_score_flawed)
print("Flawed pipeline metrics (SMOTE before split, evaluated on resampled test set):")
pd.Series(flawed_metrics).round(4)

# %% [markdown]
# Every metric is ≈ **1.00**, faithfully reproducing the source's claim. This is *not*
# evidence of a good model — it is the signature of data leakage on a synthetically
# balanced test set.

# %% [markdown]
# ### 4.2 Corrected, leakage-free pipeline
#
# Now the honest setup:
# 1. **Stratified split first** (80/20), preserving the 0.17% prevalence in both parts.
# 2. A single `imblearn` **Pipeline** `RobustScaler → SMOTE → estimator`, evaluated with
#    **StratifiedKFold(5)** cross-validation so SMOTE is fitted on each training fold
#    only.
# 3. The 20% test set is **never resampled** and is touched once, at the end.
#
# We compare Logistic Regression and Random Forest (matching the source) and add an
# **Isolation Forest** — an unsupervised anomaly detector — as a supervised-vs-
# unsupervised contrast.

# %%
X_train, X_test, y_train, y_test = models.stratified_split(X, y)
print("Train:", X_train.shape, "| Test:", X_test.shape, "| Test frauds:", int(y_test.sum()))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["average_precision", "roc_auc", "recall", "precision"]

cv_rows = {}
test_results = {}
pr_curves = {}
roc_curves = {}

for name, estimator in models.supervised_estimators().items():
    pipeline = models.build_corrected_pipeline(estimator)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_rows[name] = {
        "cv_PR_AUC": scores["test_average_precision"].mean(),
        "cv_ROC_AUC": scores["test_roc_auc"].mean(),
        "cv_recall": scores["test_recall"].mean(),
    }
    y_pred, y_score = models.fit_predict_proba(pipeline, X_train, y_train, X_test)
    test_results[name] = evaluate.classification_metrics(y_test, y_pred, y_score)
    pr_curves[name] = (y_test.values, y_score)
    roc_curves[name] = (y_test.values, y_score)

print("Cross-validated scores on the training set (leakage-free):")
pd.DataFrame(cv_rows).T.round(4)

# %% [markdown]
# ### 4.3 Isolation Forest (unsupervised)
#
# Isolation Forest learns nothing from labels; it isolates points that are easy to
# separate by random splits. We set `contamination` to the training fraud prevalence
# so its default threshold is reasonable, then score the same untouched test set.

# %%
prevalence = float(y_train.mean())
y_pred_if, y_score_if = models.isolation_forest_scores(X_train, X_test, contamination=prevalence)
test_results["Isolation Forest"] = evaluate.classification_metrics(y_test, y_pred_if, y_score_if)
pr_curves["Isolation Forest"] = (y_test.values, y_score_if)
roc_curves["Isolation Forest"] = (y_test.values, y_score_if)
print("Isolation Forest scored on the untouched test set.")

# %% [markdown]
# ## 5. Evaluation
#
# ### 5.1 Why these metrics
#
# - **Accuracy** — fraction correct. Reported only to expose its uselessness here:
#   the all-legit baseline already scores 99.83%.
# - **Precision** = TP/(TP+FP) — of the transactions we flag, how many are truly fraud.
#   Low precision means analysts drown in false alarms.
# - **Recall** = TP/(TP+FN) — of all frauds, how many we catch. Low recall means money
#   lost.
# - **F1 / F2** — harmonic mean of precision and recall; **F2** weights recall higher,
#   matching the fraud reality that a missed fraud (FN) costs more than a false alarm.
# - **MCC** (Matthews) — a balanced correlation between predictions and truth that stays
#   honest under extreme imbalance; arguably the single best scalar summary here.
# - **ROC-AUC** — ranking quality across all thresholds, but optimistic under imbalance
#   because the huge true-negative count flatters the false-positive rate.
# - **PR-AUC / Average Precision** — area under the precision-recall curve; the most
#   informative threshold-free metric for rare positives.

# %%
summary_table = evaluate.metrics_table(test_results)
print("Corrected-pipeline metrics on the untouched test set:")
summary_table

# %% [markdown]
# ### 5.2 Flawed vs corrected, side by side
#
# The contrast is the whole point: the flawed pipeline reports ≈1.00 across the board,
# while the honest Random Forest achieves a strong-but-imperfect PR-AUC ≈ 0.81 and
# MCC ≈ 0.84. Note that Random Forest's *accuracy* is 0.9995 even when honest — and
# Logistic Regression's accuracy is 0.97 despite catching fraud with only ~5% precision.
# Accuracy simply cannot distinguish these very different models.

# %%
comparison = pd.DataFrame(
    {
        "Flawed RF (leaked)": pd.Series(flawed_metrics),
        "Corrected RF": pd.Series(test_results["Random Forest"]),
        "Corrected LogReg": pd.Series(test_results["Logistic Regression"]),
    }
).round(4)
comparison

# %% [markdown]
# ### 5.3 Curves

# %%
evaluate.plot_pr_curves(pr_curves)
evaluate.plot_roc_curves(roc_curves)
evaluate.plot_confusion(y_test, models.fit_predict_proba(
    models.build_corrected_pipeline(models.supervised_estimators()["Random Forest"]),
    X_train, y_train, X_test)[0], "Random Forest (corrected) confusion matrix", "confusion_rf.png")
print("Saved PR, ROC, and confusion-matrix figures to reports/figures/.")

# %% [markdown]
# ![pr curves](reports/figures/pr_curves.png)
#
# ![roc curves](reports/figures/roc_curves.png)
#
# ![confusion rf](reports/figures/confusion_rf.png)
#
# The ROC curves look almost equally excellent for every model (all AUC > 0.93),
# whereas the PR curves separate them sharply — visual proof that ROC-AUC hides the
# differences that matter under imbalance.

# %% [markdown]
# ## 6. Error Analysis
#
# ### 6.1 The false-positive / false-negative trade-off
#
# At the default 0.5 threshold the corrected Random Forest is precision-heavy (few false
# alarms, but it misses ~23% of frauds). In fraud, a missed fraud usually costs far more
# than a false alarm, so we sweep the threshold with an asymmetric cost (FN = 100×FP) and
# pick the cost-minimising operating point.

# %%
rf_pipeline = models.build_corrected_pipeline(models.supervised_estimators()["Random Forest"])
_, rf_score = models.fit_predict_proba(rf_pipeline, X_train, y_train, X_test)
cost_table = evaluate.threshold_cost_analysis(y_test, rf_score, fn_cost=100, fp_cost=1)
best = cost_table.loc[cost_table["cost"].idxmin()]
print("Cost-optimal operating point (FN=100x FP):")
print(best[["threshold", "recall", "precision", "tp", "fp", "fn"]])
evaluate.plot_cost_curve(cost_table)

# %% [markdown]
# ![cost curve](reports/figures/cost_curve.png)
#
# Lowering the threshold from 0.5 to ≈0.09 trades a handful of extra false alarms for
# several additional caught frauds — the right call under the stated cost model. This is
# the kind of deliberate, cost-aware decision the original tutorial never surfaces,
# because its ≈1.00 metrics leave no apparent trade-off to make.

# %% [markdown]
# ### 6.2 What kind of frauds slip through
#
# We inspect the false negatives of the corrected Random Forest. Missed frauds skew
# toward **small amounts** that look like ordinary spending — economically the least
# damaging individually, but a known tactic ("card testing") for probing stolen cards
# with tiny purchases before a large one.

# %%
rf_pred_default = (rf_score >= 0.5).astype(int)
test_view = X_test.copy()
test_view["true"] = y_test.values
test_view["pred"] = rf_pred_default
false_neg = test_view[(test_view["true"] == 1) & (test_view["pred"] == 0)]
true_pos = test_view[(test_view["true"] == 1) & (test_view["pred"] == 1)]
print("False negatives:", len(false_neg), "| True positives:", len(true_pos))
print("Median Amount — missed frauds: %.2f | caught frauds: %.2f"
      % (false_neg["Amount"].median(), true_pos["Amount"].median()))

# %% [markdown]
# ## 7. Conclusions
#
# - **The source's central claim is not supported.** Its near-perfect AUC-ROC ≈ 1.00 is
#   reproducible only when SMOTE is applied before the split and the model is evaluated
#   on a synthetically balanced test set — i.e. it measures leakage, not skill.
# - **Honest performance is strong but bounded.** A leakage-free Random Forest reaches
#   PR-AUC ≈ 0.81 and MCC ≈ 0.84; it catches ~77% of frauds at 91% precision out of the
#   box, and can be tuned for higher recall at a controlled cost.
# - **Metric choice changes the story.** Accuracy and ROC-AUC are near-perfect for every
#   model; only PR-AUC, MCC, and F2 reveal the real, large gaps between them.
# - **No single feature detects fraud** (top |Spearman| ≈ 0.06); the problem is
#   irreducibly multivariate, which is why tree ensembles beat the linear model.
# - **Future work:** cost-sensitive learning, gradient boosting (XGBoost/LightGBM),
#   threshold calibration per business cost, and a temporally-aware split if more than
#   two days of data become available.
#
# **Recommendation:** the *modelling idea* (tree ensemble on these features) is sound and
# worth reusing; the *evaluation methodology* of the source is not, and its headline
# numbers should not be trusted or repeated.
