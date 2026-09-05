"""Evaluation package: ₹-centric policy metrics + off-policy (IPW) estimates."""

from src.evaluation.metrics import (breakdowns, compare_policies,
                                    metrics_to_markdown, policy_metrics)
from src.evaluation.off_policy import (ipw_recovery, observed_recovery,
                                       oracle_recovery,
                                       selective_label_summary)

__all__ = ["breakdowns", "compare_policies", "metrics_to_markdown",
           "policy_metrics", "ipw_recovery", "observed_recovery",
           "oracle_recovery", "selective_label_summary"]