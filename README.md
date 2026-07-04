# Critical Review of a Credit-Card Fraud Detection Tutorial

Final project for **Data Science in Cyber**. This project critically reproduces and
evaluates a published credit-card fraud detection tutorial, testing whether its
headline claim — **AUC-ROC ≈ 1.00** — holds up under sound methodology.

## Project description

The selected source balances an extremely imbalanced dataset with SMOTE, applies PCA,
and reports near-perfect detection. We show that this near-perfect score is a
**data-leakage artefact**: SMOTE is applied before the train/test split and the model
is judged on a synthetically balanced test set with metrics (accuracy, ROC-AUC) that
are over-optimistic under 0.17% prevalence.

We then rebuild the analysis as a **leakage-free pipeline** (SMOTE applied inside
cross-validation folds only; the test set is never resampled) and report honest metrics
— Precision-Recall AUC, Matthews Correlation Coefficient, and the recall-weighted F2 —
together with a cost-based analysis of the false-positive / false-negative trade-off.

**Key result:** the flawed pipeline reproduces ≈1.00 across every metric, while the
honest Random Forest achieves PR-AUC ≈ 0.81 and MCC ≈ 0.84 — strong, but far from
perfect. The source's claims are **not supported** by sound methodology.

## Links

- **Selected source (reproduced & critiqued):**
  https://github.com/BalaElangovan/Enhancing-Credit-Card-Fraud-Detection-Models-using-Machine-Learning-and-PCA
- **Original GitHub repository:** same as above (the source is itself a GitHub project).
- **Dataset source:** Kaggle / ULB *Credit Card Fraud Detection* (Dal Pozzolo et al.,
  2015): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  Downloaded in this project from a public no-authentication mirror:
  https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection

## Repository layout

```
.
├── README.md
├── requirements.txt
├── fraud_detection_critical_review.ipynb   # the main, executed notebook
├── report/
│   └── report.pdf                          # the written report (8 sections)
├── src/                                    # documented helper modules
│   ├── config.py        # shared constants and the fixed random seed
│   ├── data.py          # download, load, de-duplicate
│   ├── eda.py           # EDA summaries and plots
│   ├── features.py      # temporal + log transforms
│   ├── models.py        # flawed and leakage-free pipelines
│   └── evaluate.py      # metrics, curves, cost analysis
├── scripts/
│   ├── download_data.py # fetch the dataset into data/
│   └── run_analysis.py  # end-to-end driver (prints/saves all metrics)
├── reports/
│   ├── figures/         # generated plots
│   └── metrics.json     # captured numerical results
└── data/                # dataset lives here (git-ignored, downloaded on demand)
```

## How to run

```bash
# 1. Create an environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the dataset (~102 MB) into data/creditcard.csv
python scripts/download_data.py

# 3a. Reproduce all numbers from the command line
python scripts/run_analysis.py

# 3b. ...or run the full notebook
jupyter notebook fraud_detection_critical_review.ipynb
```

All stochastic steps use a fixed seed (`RANDOM_STATE = 42`), so results are
reproducible across runs and machines.
