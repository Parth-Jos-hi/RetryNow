"""Load + light-cleaning of the failed-payment table; train/test splits.

Two split strategies are provided:

  * ``train_test_split_failed`` — random split (headline policy evaluation).
  * ``temporal_train_test_split`` — cut at a date (leak-free; used for the
    drift experiment). History features must be fit *inside the train window*
    only — the caller builds features on the training slice, never the whole
    table — otherwise future outcomes leak into test-row features.

Both splits keep the censor-aware discipline: the label is the observed outcome
``retry_success_obs`` (NaN = censored), and the oracle / exposure columns
(``p_success_*``, ``retry_propensity``, ``was_retried_historically``) are never
feature columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.generate_synthetic_data import Config

# Columns that must never become features: identity, time, the censored label,
# the evaluator-only oracle, and the historical exposure/selection columns.
_FORBIDDEN_FEATURES = [
    "retry_success_obs", "txn_id", "timestamp", "customer_id", "merchant_id",
    "retry_propensity", "was_retried_historically", "risk_score",
    "p_success_retry_soon", "p_success_retry_later",
    "p_success_switch_method", "p_success_send_link",
]


def load_failed_payments(path: str | Path) -> pd.DataFrame:
    """Load the generated CSV and coerce dtypes (dates, numerics, categories)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/run_pipeline.py --step generate` first.")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure date-derived columns exist (idempotent if the generator already set them)."""
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df.timestamp.dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df.timestamp.dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df.timestamp.dt.dayofweek >= 5).astype(int)
    if "amount_log" not in df.columns:
        df["amount_log"] = pd.Series(df.amount).map(lambda a: __import__("math").log1p(a))
    return df


def _feature_matrix(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    """Strip forbidden columns; returns (X, y) aligned to ``df``'s row order."""
    if label not in df.columns:
        raise KeyError(f"label column '{label}' missing")
    drop = [c for c in _FORBIDDEN_FEATURES if c in df.columns and c != label]
    X = df.drop(columns=drop, errors="ignore")
    y = df[label]
    return X, y


def train_test_split_failed(df: pd.DataFrame,
                            label: str = "retry_success_obs",
                            test_size: float = 0.20,
                            seed: int = 42,
                            stratify: bool = True):
    """Random split for headline evaluation (features already engineered).

    Note: with censored labels the splitter is label-aware but exposure-aware
    stratification is intentionally NOT applied — the IPW weighting happens at
    evaluation time.
    """
    X, y = _feature_matrix(df, label)
    stratify_col = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=seed,
                            stratify=stratify_col)


def temporal_train_test_split(df: pd.DataFrame,
                              split_date: str,
                              label: str = "retry_success_obs"):
    """Leak-free temporal cut: train = all rows before ``split_date``.

    The returned test frame keeps identity/oracle columns — the caller may need
    them for counterfactual evaluation; feature building should be done on the
    train slice only, then applied to the test slice by column alignment.
    """
    cutoff = pd.Timestamp(split_date)
    train = df[df.timestamp < cutoff].reset_index(drop=True)
    test = df[df.timestamp >= cutoff].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"temporal split at {split_date}: empty train ({len(train)}) "
                         f"or test ({len(test)}) side")
    X_train, y_train = _feature_matrix(train, label)
    X_test, y_test = _feature_matrix(test, label)
    return X_train, X_test, y_train, y_test, train, test


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    """Load the main config yaml into an attribute-style object."""
    with open(path) as fh:
        return Config(yaml.safe_load(fh))