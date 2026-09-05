"""Baseline policies for the fair comparison — ALL executed through the SAME
simulator as the AI agent (no flat recovery-rate shortcuts).

  * ``do_nothing``    — no action, no touchpoints, no cost. The floor.
  * ``dumb_retry``    — retry EVERY eligible failure once after a fixed window
                        (4h). This is the over-retry baseline: it also retries
                        card_expired and account_blocked, burning cost+touches.
  * ``rule_retry``    — failure-reason→(action, hours) table (the heuristic
                        "Smart Retry" style). Better than dumb, still blind to
                        context, value, risk and customer budgets.

Both retry policies respect the scheduler bandwidth and the AttemptTracker's
budgets, exactly like the agent does.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.models.decision_engine import (GIVE_UP, RETRY_LATER, RETRY_SOON,
                                        SEND_LINK, SWITCH_METHOD)
from src.recovery.attempt_tracker import AttemptTracker
from src.recovery.scheduler import (RetryJob, RetryScheduler,
                                    schedule_from_decisions)

log = logging.getLogger(__name__)

# Rule policy defaults (mirror configs/config.yaml -> rule_policy)
RULE_DEFAULTS = {
    "upi_server_busy": {"action": RETRY_SOON, "hours": 1},
    "bank_server_timeout": {"action": RETRY_SOON, "hours": 2},
    "network_timeout": {"action": RETRY_SOON, "hours": 2},
    "otp_expired": {"action": SEND_LINK, "hours": 6},
    "insufficient_funds": {"action": RETRY_LATER, "hours": 72},
    "issuer_decline": {"action": RETRY_LATER, "hours": 48},
    "duplicate_payment_block": {"action": RETRY_LATER, "hours": 24},
    "card_expired": {"action": SEND_LINK, "hours": 12},
    "account_blocked": {"action": GIVE_UP, "hours": None},
}


def _run_policy(txns: pd.DataFrame, simulator, jobs: list[dict],
                scheduler_hours: list[float]) -> pd.DataFrame:
    """Shared driver: schedule -> clocked simulation -> annotate outcomes.

    ``jobs``: list of {txn_id, customer_id, payment_method, action,
    retry_in_hours, probability}. ``scheduler_hours``: per-job due offsets
    (mirrors the agent's schedule_from_decisions path).

    Annotations mirror the agent's output frame so the shared ₹-metrics treat
    every policy the same: ``outcome``, ``recovered_value``, ``attempts_taken``
    and a minimal ``decision`` dict (action + scheduling context) so retry
    costs, touchpoints and give_up share the same accounting.
    """
    sched = RetryScheduler()
    tracker = AttemptTracker()
    base = txns["timestamp"].min()
    n = schedule_from_decisions(jobs, base, sched)
    log.info("policy run: %d jobs scheduled", n)
    results = simulator.run_simulation(sched, tracker)

    out = txns.copy()
    # per-txn: first (earliest) outcome wins; attempts = number of executed
    # attempts (denied-by-budget attempts are not executed, so not counted)
    attempts: dict[str, int] = {}
    outcome_map: dict[str, str] = {}
    for r in results:
        outcome_map.setdefault(r.txn_id, r.outcome)
        attempts[r.txn_id] = attempts.get(r.txn_id, 0) + 1
    out["outcome"] = out["txn_id"].map(outcome_map)
    out["attempts_taken"] = out["txn_id"].map(attempts).fillna(0).astype(int)
    out["recovered_value"] = out.apply(
        lambda r: r["amount"] if r["outcome"] == "SUCCESS" else 0.0, axis=1)

    # minimal decision record per txn for shared metrics (action attribution,
    # touchpoints, give_up accounting)
    rule_by_txn = {j["txn_id"]: j for j in jobs}
    out["decision"] = out["txn_id"].apply(
        lambda t: {"action": rule_by_txn[t]["action"]} if t in rule_by_txn
        else {"action": GIVE_UP})
    return out


def do_nothing(txns: pd.DataFrame, simulator) -> pd.DataFrame:
    """Policy 1: no recovery attempted — the floor for the comparison."""
    out = txns.copy()
    out["outcome"] = None
    out["recovered_value"] = 0.0
    out["attempts_taken"] = 0
    out["decision"] = [{"action": GIVE_UP, "probability": 0.0} for _ in range(len(out))]
    return out


def dumb_retry(txns: pd.DataFrame, simulator,
               retry_hours: float = 4.0,
               max_attempts: int = 1) -> pd.DataFrame:
    """Policy 2: every failure retried once after a fixed delay.

    No eligibility filtering — the point is that brutishness: expired and
    blocked cards get a useless retry too (cost + touch burned).
    """
    jobs = [
        {"txn_id": r.txn_id, "customer_id": r.customer_id,
         "payment_method": r.payment_method,
         "action": RETRY_SOON, "retry_in_hours": retry_hours, "probability": 0.5}
        for r in txns.itertuples()
    ]
    out = _run_policy(txns, simulator, jobs, [retry_hours] * len(jobs))
    out["policy"] = "dumb_retry"
    return out


def rule_retry(txns: pd.DataFrame, simulator,
               rules: Optional[dict] = None,
               max_attempts: int = 2) -> pd.DataFrame:
    """Policy 3: heuristic reason→(action, hours) Smart-Retry style."""
    rules = {**RULE_DEFAULTS, **(rules or {})}
    jobs = []
    for r in txns.itertuples():
        spec = rules.get(r.failure_reason_code, {"action": SEND_LINK,
                                                 "hours": 6})
        if spec.get("action") == GIVE_UP or spec.get("hours") is None:
            continue
        jobs.append({"txn_id": r.txn_id, "customer_id": r.customer_id,
                     "payment_method": r.payment_method,
                     "action": spec["action"],
                     "retry_in_hours": spec["hours"], "probability": 0.5})
    out = _run_policy(txns, simulator, jobs,
                      [j["retry_in_hours"] for j in jobs])
    out["policy"] = "rule_retry"
    return out