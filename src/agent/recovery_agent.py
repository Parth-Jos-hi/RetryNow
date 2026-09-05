"""The Recovery Agent: stateless decide() + a clocked, bounded batch driver.

Per the design review, the agent is split in two:

  * :meth:`RecoveryAgent.decide` — stateless per-attempt decision: estimate
    per-action recoverability → ask the risk gate → expected-utility selection.
    Produces exactly ONE action + an audit event.
  * :meth:`RecoveryAgent.run` — the batch driver: schedules decided retries on
    a :class:`RetryScheduler`, advances the clock, executes due jobs through
    the :class:`PaymentSimulator` (respecting the AttemptTracker's attempt and
    customer-daily budgets), cancels payments on SUCCESS, and writes the audit
    trail. This is where "stop when the budget says stop" actually lives.

The audit trail is one JSONL file — one line per decision event and per
execution event, keyed by ``txn_id`` — the reproducible record behind every
website number.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.decision_engine import (Decision, DecisionEngine, GIVE_UP,
                                        RETRY_SOON, SWITCH_METHOD)
from src.models.recoverability import RecoverabilityEstimator
from src.recovery.attempt_tracker import AttemptTracker
from src.recovery.scheduler import RetryJob, RetryScheduler, schedule_from_decisions
from src.risk.risk_gate import MockRiskGate, RiskGate

log = logging.getLogger(__name__)


class RecoveryAgent:
    """Bounded recovery agent: decision policy + clocked batch execution."""

    def __init__(self, estimator: RecoverabilityEstimator,
                 engine: DecisionEngine,
                 risk_gate: Optional[RiskGate] = None,
                 audit_path: Optional[str | Path] = None,
                 seed: int = 42):
        self.estimator = estimator
        self.engine = engine
        self.risk_gate = risk_gate or MockRiskGate()
        self._audit_path = Path(audit_path) if audit_path else None
        self._audit_handle = None
        self.seed = seed

    # -------------------------------------------------------------- lifecycle
    def open_audit(self, path: Optional[str | Path] = None) -> None:
        if self._audit_handle is not None:
            self.close_audit()
        path = Path(path) if path else self._audit_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # one trail per run — reopen truncates earlier runs' stale lines
        self._audit_handle = open(path, "w", encoding="utf-8")
        self._audit_path = path

    def close_audit(self) -> None:
        if self._audit_handle is not None:
            self._audit_handle.close()
            self._audit_handle = None

    def _log_event(self, event: dict) -> None:
        if self._audit_handle is not None:
            self._audit_handle.write(json.dumps(event, ensure_ascii=False,
                                                default=str) + "\n")
            self._audit_handle.flush()

    # ------------------------------------------------------------- stateless
    def decide(self, txn: dict, attempts: int = 0) -> Decision:
        """One decision: estimate → risk gate → expected-utility argmax.

        ``txn`` carries the features/context the estimator needs (reason,
        amount, day_of_month, customer_instruments, feature columns…) plus
        ``txn_id``/``customer_id``/``payment_method``/``risk_score``.
        """
        p_actions = self.estimator.estimate(txn)
        verdict = self.risk_gate.check(txn)
        decision = self.engine.decide(
            amount=float(txn.get("amount", 0.0)),
            p_actions=p_actions,
            attempts=attempts,
            current_method=txn.get("payment_method"),
            risk_allowed=verdict.allowed,
        )
        self._log_event({
            "event": "decide",
            "txn_id": txn.get("txn_id"),
            "customer_id": txn.get("customer_id"),
            "ts": str(txn.get("timestamp", datetime.now())),
            "amount": float(txn.get("amount", 0.0)),
            "risk_score": float(txn.get("risk_score", 0.0)),
            "risk_verdict": {k: bool(v) for k, v in verdict.allowed.items()},
            "probabilities": {k: round(float(v), 4) for k, v in p_actions.items()},
            "decision": decision.as_dict(),
        })
        return decision

    # ------------------------------------------------------------ batch driver
    def run(self, txns: pd.DataFrame,
            simulator,
            scheduler: Optional[RetryScheduler] = None,
            tracker: Optional[AttemptTracker] = None,
            retry_soon_hours: Optional[float] = None,
            retry_later_hours: Optional[float] = None) -> pd.DataFrame:
        """Bounded batch run: decide → schedule → clock → execute → record.

        Returns the transaction frame annotated with ``decision`` and
        ``outcome`` columns (outcome = SUCCESS/FAILED/… or None when nothing
        was executed for that txn).
        """
        s = self.engine.settings
        if self._audit_handle is None and self._audit_path is not None:
            self.open_audit()  # lazy: run() owns the trail when given a path
        scheduler = scheduler or RetryScheduler()
        tracker = tracker or AttemptTracker(
            max_attempts_per_payment=s["max_attempts_per_payment"],
            max_retries_per_customer_day=s["max_retries_per_customer_day"])
        base_time = txns["timestamp"].min()
        out = txns.copy()

        # 1) decide every transaction in the batch (one decision each — the
        #    attempt budget is held by the tracker for retries)
        decisions: list[dict] = []
        for idx, row in out.iterrows():
            d = self.decide(row.to_dict(), attempts=0).as_dict()
            decisions.append({**d, "txn_id": row["txn_id"],
                              "customer_id": row["customer_id"],
                              "payment_method": row["payment_method"]})
            out.loc[idx, "decision"] = json.dumps(d, default=str)

        # 2) schedule retryable actions
        n_jobs = schedule_from_decisions(decisions, base_time, scheduler)
        log.info("agent: %d decisions, %d retry jobs scheduled", len(decisions), n_jobs)

        # 3) clocked execution through the simulator (budgets enforced inside)
        results = simulator.run_simulation(scheduler, tracker)

        # 4) annotate outcomes (per txn: first SUCCESS wins; else latest attempt)
        outcome_map: dict[str, str] = {}
        attempt_taken: dict[str, int] = {}
        for r in results:
            outcome_map.setdefault(r.txn_id, r.outcome)   # first (earliest) outcome
            attempt_taken[r.txn_id] = r.attempt_number
            self._log_event({
                "event": "execute",
                "txn_id": r.txn_id,
                "ts": str(datetime.now()),
                "action": r.action,
                "attempt": r.attempt_number,
                "outcome": r.outcome,
                "recovered_value": r.recovered_value,
                "idempotent_replay": r.idempotent_replay,
            })
        out["outcome"] = out["txn_id"].map(outcome_map)
        out["attempts_taken"] = out["txn_id"].map(attempt_taken).fillna(0).astype(int)
        out["recovered_value"] = out.apply(
            lambda r: r["amount"] if r["outcome"] == "SUCCESS" else 0.0, axis=1)
        self.close_audit()
        return out