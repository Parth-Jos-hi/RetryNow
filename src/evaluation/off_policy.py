"""Off-policy / selective-label evaluation — the honest part of the report.

Censored outcomes: we only observe the outcome of actions that were taken.
Three estimates are reported per policy so the reader can see the gap:

  * ``observed`` — outcomes of executed actions only (naive; biased by which
    actions policy chose to take).
  * ``ipw`` — inverse-propensity weighted, clipped (default cap 10) to control
    variance: recovers the *would-have-recovered* estimate by up-weighting
    unfashionable actions. Uses the generator's ``retry_propensity`` — for the
    historical record — or the empirical action-propensity for agent runs.
  * ``oracle`` — the simulator's latent per-action truth (labeled
    "simulator-only"): what an omniscient policy's *expected* recovery would be
    (Σ value × max-action p). The gap observed→oracle is the recoverable headroom.

We do NOT fit propensity models (the environment provides them) and we do NOT
claim these estimates prove real-world performance — the whole module is about
*showing the selection-bias gap* honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUCCESS = "SUCCESS"
ORACLE_ACTIONS = ["p_success_retry_soon", "p_success_retry_later",
                  "p_success_switch_method", "p_success_send_link"]


def observed_recovery(failed: pd.DataFrame, value_col: str = "amount") -> float:
    """₹ (and count) recovered from executed actions only."""
    rec = failed[failed["outcome"] == SUCCESS]
    if "recovered_value" in failed.columns:
        return float(rec["recovered_value"].sum() or 0.0)
    return float(rec[value_col].astype(float).sum() or 0.0)


def ipw_recovery(failed: pd.DataFrame,
                 propensity_col: str = "retry_propensity",
                 outcome_col: str = "retry_success_obs",
                 value_col: str = "amount",
                 clip: float = 10.0) -> float:
    """Hájek-style IPW estimate of recoverable value under full exposure.

    Weights = 1 / max(propensity, 1/clip). Rows with no observed outcome
    (censored) contribute value × 0 — they are *unknown*, not failed; the
    estimate therefore leans on the exposed rows' outcomes, weighted up.
    """
    df = failed.copy()
    if outcome_col not in df.columns or propensity_col not in df.columns:
        return float("nan")
    w = 1.0 / np.clip(df[propensity_col].astype(float), 1.0 / clip, 1.0)
    y = df[outcome_col].fillna(0.0).astype(float)
    v = df[value_col].astype(float)
    if w.sum() == 0:
        return float("nan")
    return float((y * v * w).sum() / w.sum())


def oracle_recovery(failed: pd.DataFrame, value_col: str = "amount") -> float:
    """Expected value an omniscient policy would recover (simulator-only).

    Uses the latent per-action probabilities — NEVER available to the agent.
    """
    present = [c for c in ORACLE_ACTIONS if c in failed.columns]
    if not present:
        return float("nan")
    p_best = failed[present].max(axis=1)
    return float((p_best * failed[value_col].astype(float)).sum())


def selective_label_summary(failed: pd.DataFrame) -> dict:
    """The headline numbers of the censoring story, per policy run frame.

    ``ipw_recovery`` is a Hájek *rate* (weighted mean recovered value per
    exposed txn); it is scaled by ``n`` here so all three estimates are on the
    same footing: total ₹ the historical record implies for the batch.
    """
    n = len(failed)
    executed = int(failed["outcome"].notna().sum()) if "outcome" in failed.columns else 0
    ipw_rate = ipw_recovery(failed)
    ipw_total = ipw_rate * n if ipw_rate == ipw_rate else float("nan")  # NaN-safe
    oracle = oracle_recovery(failed)
    observed = observed_recovery(failed)
    return {
        "n_failed": n,
        "n_executed": executed,
        "n_censored": n - executed,
        "observed_recovered_value": round(observed, 2),
        "ipw_estimated_recovered_value": round(ipw_total, 2),
        "oracle_expected_recovered_value": round(oracle, 2),
        "headroom_vs_oracle": round(oracle - observed, 2),
        "note": ("oracle = simulator latent truth (counterfactual); "
                 "ipw = clipped inverse-propensity estimate, scaled to a "
                 "batch total (rate × n); censored rows are UNKNOWN "
                 "outcomes, never 'failures'"),
    }



                                                            



