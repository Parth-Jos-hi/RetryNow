"""Recoverability: per-action P(success | txn) estimates for the decision engine.

The historical record only supervised ONE action (dumb retries ≈ retry_soon), so
a single ML model cannot be trained for every action without hallucinating labels.
This layer therefore uses a two-part estimator (per the design review):

  * ``retry_soon``     — the trained ML predictor (all engineered features;
                         this is where the learned signal lives).
  * ``retry_later`` / ``switch_method`` / ``send_link`` — empirical per-reason
                         tables fit **only on rows with an observed outcome**
                         (never on censored rows), shrunk toward acknowledged
                         priors with the config's ``table_prior_alpha``
                         (empirical-Bayes-style smoothing: data dominates as
                         observations accumulate).

Context modulations applied at estimate time (config-driven, explainable):
  * ``retry_later``: salary-window boost for funding failures (1st–5th).
  * ``switch_method``: requires an alternative instrument on file.
  * ``send_link``: mild everywhere.

Honesty notes: priors are documented assumptions for actions with thin historic
supervision — they shrink as the agent explores and observes; nothing here reads
the oracle (``p_success_*``) columns.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.decision_engine import (RETRY_LATER, RETRY_SOON, SEND_LINK,
                                        SWITCH_METHOD)

log = logging.getLogger(__name__)

PRIOR_ALPHA_DEFAULT = 5.0
SALARY_BOOST_LATER_DEFAULT = 1.30
SWITCH_NO_ALT_DEFAULT = 0.05


class RecoverabilityEstimator:
    """Fitted multi-action estimator: ``estimate(row) -> {action: p}``."""

    def __init__(self, predictor=None,
                 prior_alpha: float = PRIOR_ALPHA_DEFAULT,
                 priors: Optional[dict] = None,
                 salary_boost_later: float = SALARY_BOOST_LATER_DEFAULT,
                 switch_no_alt: float = SWITCH_NO_ALT_DEFAULT):
        self.predictor = predictor
        self.prior_alpha = max(1.0, float(prior_alpha))
        self.priors = priors or {}          # {reason: {action: p}} — acknowledged
        self.salary_boost_later = salary_boost_later
        self.switch_no_alt = switch_no_alt
        self.tables: dict[str, dict[str, float]] = {}   # action -> {reason: p}

    # ---------------------------------------------------------------- fitting
    def fit_tables(self, df: pd.DataFrame,
                   reason_col: str = "failure_reason_code",
                   outcome_col: str = "retry_success_obs",
                   prior_flat: float = 0.30,
                   observed_actions: Optional[tuple] = None) -> "RecoverabilityEstimator":
        """Fit per-reason tables from observed outcomes only.

        THE SUPERVISION CAVEAT: the historical record observed essentially one
        arm — retries (≈ retry_soon). Its labels (``outcome_col``) therefore
        describe retry_soon outcomes, and must NOT be credited to other arms;
        otherwise every action's table silently inherits the retry-success rate
        and the EV engine treats switch/link as if they were retried-and-observed.
        So:

          * ``retry_soon``  → empirical-Bayes blend of the observed rate toward
            the prior:  (k·p̂ + α·p0) / (k + α).
          * other actions   → priors only (documented assumptions; these arms
            were never exposed). They shrink toward reality as the agent runs
            and observes its own executions (feedback/learning layer).

        Censored rows (NaN outcome) contribute nothing anywhere.
        """
        observed_actions = observed_actions or (RETRY_SOON,)
        obs = df[df[outcome_col].notna()].copy()
        grouped = obs.groupby(reason_col)[outcome_col].agg(["mean", "count"])
        reasons = set(grouped.index) | {r for r in self.priors if r not in grouped.index}
        self.exposure_counts: dict[str, dict[str, int]] = {
            action: {} for action in (RETRY_SOON, RETRY_LATER, SWITCH_METHOD, SEND_LINK)}
        for action in (RETRY_SOON, RETRY_LATER, SWITCH_METHOD, SEND_LINK):
            table: dict[str, float] = {}
            for reason in sorted(reasons):
                prior = self.priors.get(reason, {}).get(action, prior_flat)
                if action in observed_actions and reason in grouped.index:
                    rate, count = grouped.loc[reason]
                    smoothed = (count * rate + self.prior_alpha * prior) / \
                               (count + self.prior_alpha)
                    table[reason] = float(np.clip(smoothed, 0.0, 1.0))
                    self.exposure_counts[action][reason] = int(count)
                else:                       # unobserved arm → prior stands
                    table[reason] = float(np.clip(prior, 0.0, 1.0))
                    self.exposure_counts[action][reason] = 0
            self.tables[action] = table
        log.info("recoverability tables fit on %d observed rows (observed arms: %s, "
                 "priors alpha=%.1f)", len(obs), observed_actions, self.prior_alpha)
        return self

    # ------------------------------------------------------------- estimation
    def estimate(self, row: dict | pd.Series) -> dict[str, float]:
        """Return {action: P(success | row)} for the four candidate actions."""
        reason = row.get("failure_reason_code")
        tables = self.tables
        p_soon = self._ml_soon(row)
        p_later = self._table(reason, tables.get(RETRY_LATER, {}),
                              row, boost=self.salary_boost_later)
        p_switch = self._table(reason, tables.get(SWITCH_METHOD, {}), row)
        if (row.get("customer_instruments") or 1) <= 1:
            p_switch *= self.switch_no_alt   # no alternative on file → dead end
        p_link = self._table(reason, tables.get(SEND_LINK, {}), row)
        return {
            RETRY_SOON: float(np.clip(p_soon, 0.0, 1.0)),
            RETRY_LATER: float(np.clip(p_later, 0.0, 1.0)),
            SWITCH_METHOD: float(np.clip(p_switch, 0.0, 1.0)),
            SEND_LINK: float(np.clip(p_link, 0.0, 1.0)),
        }

    # -------------------------------------------------------------- internals
    def _ml_soon(self, row: dict | pd.Series) -> float:
        """Hierarchical blend for retry_soon: ML ← reason-group table ← prior.

        The ML learns within-group structure but overfits thin groups (e.g.
        account_blocked had ~5 exposed train rows → ~0.27 vs true ~0.05). The
        per-reason table (observed rate shrunk toward the documented prior) is
        the group anchor; the ML's weight w = count/(count+α) means the anchor
        takes over only where supervision is thin. Without a predictor the
        table (or the flat default) stands alone.
        """
        if self.predictor is None:
            return 0.3
        reason = row.get("failure_reason_code")
        table_p = self.tables.get(RETRY_SOON, {}).get(reason)
        if table_p is None:
            return 0.3
        count = self.exposure_counts.get(RETRY_SOON, {}).get(reason, 0)
        w = count / (count + self.prior_alpha)
        feat = _row_to_matrix(row, self.predictor)
        p = self.predictor.predict_proba(feat)[0][1] if hasattr(
            self.predictor, "predict_proba") else float(self.predictor.predict(feat)[0])
        ml_p = float(np.clip(p, 0.0, 1.0))
        return float(w * ml_p + (1.0 - w) * table_p)

    def _table(self, reason, table: dict, row, boost: float = 1.0) -> float:
        p = table.get(reason, 0.30)
        if boost != 1.0 and reason == "insufficient_funds" \
                and (row.get("day_of_month") or 0) <= 5:
            p *= boost
        return float(np.clip(p, 0.0, 1.0))

    def describe(self) -> dict:
        return {
            "prior_alpha": self.prior_alpha,
            "n_actions_tables": len(self.tables),
            "reason_rate_samples": {r: round(float(t.get("retry_soon", 0.0)), 3)
                                    for r, t in self.tables.get("retry_soon", {}).items()},
        }


def _row_to_matrix(row: dict | pd.Series, model) -> pd.DataFrame:
    """Build a one-row feature matrix from a row/dict, matching training columns.

    Training uses the engineered matrix columns; when the caller hands a raw row
    with fewer columns, missing ones are zero-filled — acceptable for the
    website/single-txn path (the pipeline path passes the already-engineered
    feature row).
    """
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    else:
        cols = list(getattr(model, "classes_", None) or [])
    data = {c: (row.get(c, 0.0) if isinstance(row, dict) else
                row[c] if c in row.index else 0.0) for c in cols}
    return pd.DataFrame([data])