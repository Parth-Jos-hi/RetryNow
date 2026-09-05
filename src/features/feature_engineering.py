"""Feature engineering: from raw failed rows to the predictor's design matrix.

Two feature archetypes:
  * event-level: failure reason, amount, method, merchant vertical, time-of-day…
  * history-level: customer / merchant / instrument history that the *opportunity*
    retry needs (computed from the same table, in a leak-free temporal way).

Censored-outcome discipline (the important rule this module enforces):
  * The rolling history features are computed ONLY from rows with an *observed*
    outcome (``retry_success_obs`` not NaN) — never from oracle columns, and
    never by treating un-retried rows as failures.
  * The historical exposure/selection columns (``was_retried_historically``,
    ``retry_propensity``) are NEVER model features — they encode the past
    policy, so a model that sees them would learn "not retried = unrecoverable".
  * Row order is preserved: ``feature_matrix[i]`` always corresponds to ``df[i]``
    (the temporal sort used for leak-free history is applied internally, then
    the added columns are reindexed back to the input order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "amount", "amount_log", "hour_of_day", "day_of_week", "is_weekend",
    "customer_instruments", "customers_previous_success_rate",
    "merchant_previous_success_rate", "days_since_failure_seen_for_instrument",
]

CATEGORICAL_FEATURES = [
    "payment_method", "failure_reason_code", "merchant_vertical", "bank",
]

# Columns added by :func:`build_history_features`, mapped to the Series they are
# computed from — used to reindex them back to the input row order.
_HISTORY_COLUMNS = [
    "customers_previous_success_rate",
    "merchant_previous_success_rate",
    "days_since_failure_seen_for_instrument",
]


def build_history_features(df: pd.DataFrame,
                           outcome_col: str = "retry_success_obs") -> pd.DataFrame:
    """Compute leak-free rolling history features per customer / merchant / instrument.

    For a failed row at time ``t`` we want, e.g., “how often did this customer's
    past failures recover?” — computed only from failures strictly before ``t``
    **with an observed outcome**.  The input row order is preserved.

    ``outcome_col`` defaults to the censored observed label; rows with NaN
    outcomes contribute nothing to history (they are censored, not failed).
    """
    out = df.copy()
    original_index = out.index
    ordered = df.sort_values("timestamp")        # leak-free: “seen before” only

    # observed indicator: only rows with a real outcome enter the history feed
    observed = ordered[outcome_col].notna() if outcome_col in ordered.columns \
        else pd.Series(True, index=ordered.index)
    eff_col = outcome_col if outcome_col in ordered.columns else "retry_success_obs"
    if eff_col not in ordered.columns:
        raise KeyError(f"observed-outcome column '{eff_col}' missing from data")

    # customer-level rolling recovery rate (expanding, observed rows only)
    cus_rate = (
        ordered.assign(_observed_masked=np.where(observed, ordered[eff_col], np.nan))
        .groupby("customer_id")["_observed_masked"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0.5)  # no observed history → neutral prior
    )
    out["customers_previous_success_rate"] = cus_rate.reindex(original_index).to_numpy()

    # merchant-level rolling recovery rate (observed rows only)
    mer_rate = (
        ordered.assign(_observed_masked=np.where(observed, ordered[eff_col], np.nan))
        .groupby("merchant_id")["_observed_masked"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0.5)
    )
    out["merchant_previous_success_rate"] = mer_rate.reindex(original_index).to_numpy()

    # instrument (method+bank) lockup proxy: days since this (method,bank) combo
    # last *failed a retry* (i.e. produced an observed failure) — “locked” while
    # it keeps failing. Censored rows do not count as failures.
    instrument = ordered["payment_method"] + "|" + ordered["bank"]
    # .mask keeps datetime64 dtype (np.where would fail promotion with NaN)
    failed_obs = ordered["timestamp"].mask(
        ~(observed & (ordered[eff_col] == 0)))
    last_fail = pd.Series(failed_obs, index=ordered.index).groupby(
        instrument).shift(1)  # strictly prior rows only
    days_since = (ordered["timestamp"] - last_fail).dt.total_seconds() / 86400
    out["days_since_failure_seen_for_instrument"] = (
        days_since.reindex(original_index).to_numpy().clip(0, 90))

    # unseen / no-observed-failure instruments → neutral 90-day proxy
    out["days_since_failure_seen_for_instrument"] = (
        out["days_since_failure_seen_for_instrument"].fillna(90.0))

    return out


def build_feature_matrix(df: pd.DataFrame,
                         numeric: list[str] | None = None,
                         categorical: list[str] | None = None,
                         feature_janitor: bool = True) -> pd.DataFrame:
    """Assemble the model-ready matrix: numerics kept, categoricals one-hot encoded.

    Returns a dense float DataFrame — the safe, portable representation that
    works across XGBoost / HistGB / RandomForest.  Row order matches ``df``.
    """
    numeric = numeric or NUMERIC_FEATURES
    categorical = categorical or CATEGORICAL_FEATURES

    if feature_janitor:
        df = build_history_features(df.copy())

    # order the numeric subset
    present_numeric = [c for c in numeric if c in df.columns]
    X_num = df[present_numeric].astype(float)

    # one-hot the categoricals
    present_cat = [c for c in categorical if c in df.columns]
    X_cat = pd.get_dummies(df[present_cat], prefix=present_cat, dtype=float)

    X = pd.concat([X_num, X_cat], axis=1)
    return X


def expected_feature_columns() -> list[str]:
    """Crude but honest list of the columns the pipeline produces (for tests)."""
    return NUMERIC_FEATURES + [
        c + "_" for c in CATEGORICAL_FEATURES]  # prefix trick; refined in tests