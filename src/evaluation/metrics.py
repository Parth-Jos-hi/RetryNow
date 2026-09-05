"""₹-centric policy metrics — the primary bar of this project.

Accuracy/F1 are diagnostics here; the headline is **% of failed-transaction
VALUE recovered**, broken down by reason / instrument / time window / value
bucket, plus the economics: retry cost, net recovered value, touchpoints,
recovery-to-bother ratio, risk-blocked count, give-up rate.

Inputs are the annotated frames produced by the agent / baselines (columns:
``amount``, ``outcome``, ``recovered_value``, optional ``decision`` dict /
``attempts_taken`` / ``risk_blocked``).
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from src.models.decision_engine import ALL_ACTIONS, GIVE_UP, SEND_LINK, SWITCH_METHOD


def _decision_field(row, field: str, default=None):
    d = row.get("decision") if isinstance(row, dict) else getattr(row, "decision", None)
    if d is None or pd.isna(d):
        return default
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except (ValueError, TypeError):
            return default
    return d.get(field, default) if isinstance(d, dict) else default


def policy_metrics(failed: pd.DataFrame,
                   retry_cost_per_attempt: float = 2.0,
                   action_col: str = "decision",
                   attempts_col: str = "attempts_taken") -> dict:
    """One dict of metrics for one policy's annotated outcome frame."""
    n = len(failed)
    total_failed_value = float(failed["amount"].sum())
    recovered = failed[failed["outcome"] == "SUCCESS"]
    recovered_value = float(recovered["recovered_value"].sum() or 0.0)
    succeeded_count = len(recovered)

    if action_col in failed.columns:
        attempts = int(failed[attempts_col].sum()) if attempts_col in failed.columns else 0
        # touchpoints: count *visible* actions (switch offer / payment link) —
        # silent background retries do not bother the customer
        touches = int(failed.apply(
            lambda r: 1 if _decision_field(r, "action") in (SWITCH_METHOD, SEND_LINK)
            else 0, axis=1).sum())
    else:
        attempts = 0
        touches = 0

    unnecessary = int((failed[failed["outcome"] == "FAILED"].shape[0])
                      if "outcome" in failed.columns else 0)
    retry_cost = attempts * retry_cost_per_attempt
    return {
        "n_failed": n,
        "total_failed_value": round(total_failed_value, 2),
        "recovered_value": round(recovered_value, 2),
        "recovery_rate_value": round(recovered_value / total_failed_value, 4)
                              if total_failed_value else 0.0,
        "recovered_count": succeeded_count,
        "recovery_rate_count": round(succeeded_count / n, 4) if n else 0.0,
        "retry_cost": round(retry_cost, 2),
        "net_recovered_value": round(recovered_value - retry_cost, 2),
        "attempts": int(attempts),
        "unnecessary_retries": int(unnecessary),
        "touchpoints": int(touches),
        "recovery_to_bother": round(recovered_value / touches, 2) if touches else 0.0,
        "risk_blocked": int(failed.apply(
            lambda r: 1 if _decision_field(r, "risk_blocked") else 0, axis=1).sum())
                          if action_col in failed.columns else 0,
        "risk_suppressed_value": round(float(failed.apply(
            lambda r: r["amount"]
            if _decision_field(r, "risk_blocked") else 0.0, axis=1).sum()), 2)
                          if action_col in failed.columns else 0.0,
        "give_up_rate": round(float(failed.apply(
            lambda r: 1 if _decision_field(r, "action") == GIVE_UP else 0,
            axis=1).sum()) / n, 4) if action_col in failed.columns else None,
        "action_counts": _action_table(failed, action_col),
    }


def _action_table(failed: pd.DataFrame, action_col: str) -> dict:
    if action_col not in failed.columns:
        return {}
    counts = {a: 0 for a in ALL_ACTIONS}
    for _, row in failed.iterrows():
        a = _decision_field(row, "action")
        if a in counts:
            counts[a] += 1
    return counts


def breakdowns(failed: pd.DataFrame, value_col: str = "recovered_value") -> dict:
    """Recovered value % by reason / method / value bucket / calendar month."""
    def _rate(frame):
        total = frame["amount"].sum()
        rec = frame[value_col].sum()
        return round(rec / total, 4) if total else 0.0

    out = {}
    for dim, col in [("by_reason", "failure_reason_code"),
                     ("by_instrument", "payment_method"),
                     ("by_value_bucket", "txn_value_bucket")]:
        if col in failed.columns:
            out[dim] = failed.groupby(col).apply(
                _rate, include_groups=False).round(4).to_dict() \
                if hasattr(failed, "groupby") else {}
    if "timestamp" in failed.columns:
        ts = pd.to_datetime(failed["timestamp"])
        failed = failed.assign(_month=ts.dt.to_period("M").astype(str))
        out["by_month"] = failed.groupby("_month").apply(
            _rate, include_groups=False).round(4).to_dict()
    return out


def compare_policies(results: dict[str, pd.DataFrame],
                     retry_cost_per_attempt: float = 2.0) -> pd.DataFrame:
    """Comparison table across policies (rows = policies, cols = headline metrics)."""
    rows = {}
    for name, frame in results.items():
        m = policy_metrics(frame, retry_cost_per_attempt)
        rows[name] = {
            "recovered_value": m["recovered_value"],
            "recovery_rate_value": m["recovery_rate_value"],
            "recovery_rate_count": m["recovery_rate_count"],
            "attempts": m["attempts"],
            "touchpoints": m["touchpoints"],
            "retry_cost": m["retry_cost"],
            "net_recovered_value": m["net_recovered_value"],
            "recovery_to_bother": m["recovery_to_bother"],
            "risk_blocked": m["risk_blocked"],
            "risk_suppressed_value": m["risk_suppressed_value"],
            "give_up_rate": m["give_up_rate"],
        }
    table = pd.DataFrame(rows).T
    table["incremental_vs_dumb"] = (
        table["net_recovered_value"] - table.loc["dumb_retry",
                                                 "net_recovered_value"]
        if "dumb_retry" in table.index else 0.0)
    table["incremental_vs_rule"] = (
        table["net_recovered_value"] - table.loc["rule_retry",
                                                 "net_recovered_value"]
        if "rule_retry" in table.index else 0.0)
    return table.round(2)


def metrics_to_markdown(comparison: pd.DataFrame,
                        breakdowns_by_policy: Optional[dict] = None) -> str:
    """Render the comparison as a README-ready markdown section."""
    lines = ["## Policy comparison (same test transactions)",
             "", comparison.to_markdown() if hasattr(comparison, "to_markdown")
             else comparison.to_string(), ""]
    if breakdowns_by_policy:
        for pol, bd in breakdowns_by_policy.items():
            lines.append(f"### {pol} — recovery by reason")
            lines.append("")
            lines.append(pd.Series(bd.get("by_reason", {})).to_string())
            lines.append("")
    return "\n".join(lines)