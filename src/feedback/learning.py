"""Feedback / learning layer: observed outcomes → retrain path.

In production, this would:
  1. Collect executed decisions + observed outcomes from audit trail
  2. Augment with simulator's oracle truth (for offline evaluation)
  3. Retrain recoverability estimator on new observed labels
  4. Update per-reason tables (Empirical Bayes smoothing)
  5. Log retraining metrics (before/after AUC, label distribution shift)

For the hackathon, this is a skeleton that shows the architecture: how
observed outcomes (the "ground truth" from execution) close the loop and
inform the next iteration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


class FeedbackCollector:
    """Collect executed decisions + outcomes from audit trail."""

    def __init__(self, audit_path: str | Path):
        self.audit_path = Path(audit_path)

    def load_audit(self) -> pd.DataFrame:
        """Parse JSONL audit trail into DataFrame."""
        if not self.audit_path.exists():
            return pd.DataFrame()
        records = []
        with open(self.audit_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def extract_outcomes(self) -> pd.DataFrame:
        """Filter to execute events (outcome-bearing records)."""
        audit = self.load_audit()
        if audit.empty:
            return pd.DataFrame()

        # Filter to 'execute' events which have outcome
        executed = audit[audit["event"] == "execute"].copy()
        if executed.empty:
            return pd.DataFrame()

        # Normalize columns: txn_id, action, outcome, amount
        executed = executed[[
            "txn_id", "action", "outcome", "amount",
        ]].drop_duplicates(subset=["txn_id", "action"])

        return executed

    def augment_with_oracle(self, outcomes: pd.DataFrame,
                             oracle_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Join observed outcomes with simulator's oracle truth (if available).

        The oracle frame has columns:
          - txn_id
          - p_success_retry_soon / _retry_later / _switch_method / _send_link

        We use the oracle to compute "what would the best action have been"
        for offline RL feedback.
        """
        if oracle_df is None or oracle_df.empty:
            return outcomes

        merged = outcomes.merge(oracle_df, on="txn_id", how="left")
        # Compute optimal action post-hoc (for learning)
        oracle_cols = [c for c in merged.columns if c.startswith("p_success_")]
        if oracle_cols:
            merged["oracle_best_action"] = merged[oracle_cols].idxmax(axis=1)
            merged["oracle_best_p"] = merged[oracle_cols].max(axis=1)
        return merged


class RetrainingModule:
    """Retrain recoverability estimator on observed outcomes."""

    def __init__(self, estimator, predictor, feature_builder):
        """
        Args:
            estimator: RecoverabilityEstimator instance to update
            predictor: ML model (sklearn-like) for retry_soon action
            feature_builder: function (df) -> feature_matrix
        """
        self.estimator = estimator
        self.predictor = predictor
        self.feature_builder = feature_builder
        self.retrain_count = 0
        self.history = []

    def retrain(self, outcomes: pd.DataFrame, original_df: Optional[pd.DataFrame] = None,
                reason_col: str = "failure_reason_code",
                outcome_col: str = "outcome") -> dict:
        """Retrain tables from observed outcomes.

        Args:
            outcomes: executed decisions with observed outcomes
            original_df: full dataset (for context; optional)
            reason_col: column name for failure reason
            outcome_col: name of observed outcome column (SUCCESS / FAILED / etc)

        Returns:
            Metrics dict: label distribution shift, n_new_observed, before/after AUC
        """
        before_metrics = self._compute_metrics()

        # Retrain per-reason tables (Empirical Bayes smoothing)
        if reason_col in outcomes.columns and outcome_col in outcomes.columns:
            self.estimator.fit_tables(outcomes, reason_col=reason_col,
                                       outcome_col=outcome_col)

        # If we have a predictor and features, retrain the ML model
        if self.predictor is not None and self.feature_builder is not None:
            try:
                X = self.feature_builder(outcomes)
                y = (outcomes[outcome_col] == "SUCCESS").astype(int)
                if len(X) > 10:  # Only retrain if we have enough samples
                    self.predictor.fit(X, y)
            except Exception as e:
                # Graceful degradation: log but don't fail
                print(f"  [Feedback] Could not retrain ML model: {e}")

        after_metrics = self._compute_metrics()
        self.retrain_count += 1

        metrics = {
            "retrain_iteration": self.retrain_count,
            "n_outcomes_processed": len(outcomes),
            "label_distribution_shift": {
                "success_rate_before": before_metrics.get("success_rate", 0),
                "success_rate_after": after_metrics.get("success_rate", 0),
            },
            "status": "ok",
        }
        self.history.append(metrics)
        return metrics

    def _compute_metrics(self) -> dict:
        """Compute current model diagnostics (success rate, etc)."""
        # This is a placeholder; in production you'd compute AUC, calibration, etc.
        return {"success_rate": 0.5}  # Mock


class ColdstartModule:
    """Initialize estimator tables from domain priors when data is sparse."""

    @staticmethod
    def initialize_with_priors(estimator, priors: dict) -> None:
        """Set prior expectations on recoverability per action + reason.

        Args:
            estimator: RecoverabilityEstimator to configure
            priors: dict like {
                ("upi_server_busy", "retry_soon"): 0.78,
                ("card_expired", "retry_soon"): 0.05,
                ...
            }
        """
        if not hasattr(estimator, "_tables"):
            estimator._tables = {}
        for (reason, action), prob in priors.items():
            if reason not in estimator._tables:
                estimator._tables[reason] = {}
            estimator._tables[reason][action] = prob


def learning_loop_step(audit_path: str | Path,
                        estimator,
                        predictor: Optional = None,
                        feature_builder: Optional = None,
                        oracle_df: Optional[pd.DataFrame] = None) -> dict:
    """One step of the feedback loop: collect → augment → retrain.

    Args:
        audit_path: path to JSONL audit trail
        estimator: RecoverabilityEstimator to update
        predictor: ML model to retrain (optional)
        feature_builder: feature extraction function (optional)
        oracle_df: simulator oracle (for offline RL feedback; optional)

    Returns:
        Summary dict of retraining metrics
    """
    collector = FeedbackCollector(audit_path)
    outcomes = collector.extract_outcomes()

    if outcomes.empty:
        return {"status": "no_outcomes_to_process"}

    outcomes = collector.augment_with_oracle(outcomes, oracle_df)

    retrainer = RetrainingModule(estimator, predictor, feature_builder)
    metrics = retrainer.retrain(outcomes)

    return metrics
