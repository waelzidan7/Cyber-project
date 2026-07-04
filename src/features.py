"""Feature engineering: temporal features and scaling transforms.

Design note: scaling is *not* applied here as a global, fitted step. Fitting a
scaler on the full dataset before splitting is itself a (mild) form of leakage.
Instead, scaling lives inside the modelling pipeline (see ``models.py``) so that
the scaler is fitted on training folds only. The helpers here add columns that
are pure row-wise functions and therefore safe to compute before splitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AMOUNT, TIME


def add_hour_of_day(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an ``Hour`` column (0-23) derived from the ``Time`` feature.

    ``Time`` counts seconds since the first transaction and spans ~48 hours.
    Converting to hour-of-day exposes the daily rhythm of activity, which is the
    cyber-relevant signal (fraud and legitimate traffic follow different daily
    patterns). This is a deterministic per-row transform, so it cannot leak.
    """
    out = frame.copy()
    out["Hour"] = (out[TIME] // 3600 % 24).astype(int)
    return out


def add_log_amount(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``LogAmount = log1p(Amount)``.

    ``Amount`` is strongly right-skewed (median 22, max ~25,700). ``log1p``
    compresses the long tail towards a more symmetric distribution, which helps
    distance- and gradient-based models and makes visual EDA legible. ``log1p``
    (not ``log``) is used because the minimum amount is 0.
    """
    out = frame.copy()
    out["LogAmount"] = np.log1p(out[AMOUNT])
    return out
