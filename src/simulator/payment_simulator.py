"""Payment simulator: the ONLY way "execution" happens in this project.

We have no permission to execute real financial transactions — this module makes
that impossible by construction. It simulates outcomes for recovery actions
against the latent per-action ground truth the generator stored:

  * ``SUCCESS``        — the payment went through, ₹ recovered.
  * ``FAILED``         — executed, declined / failed again.
  * ``USER_DECLINED``  — the customer (simulated) declined the visible prompt.
  * ``EXPIRED``        — a payment link expired before use.
  * ``TIMED_OUT``      — the attempt timed out at the PSP without completing.

Idempotency: executing the same (txn, action, attempt_number) twice returns the
stored outcome — mirroring how a real payment system must behave, without ever
touching real rails.

Counterfactual honesty: the oracle (``p_success_*`` columns) lives here and is
used ONLY to sample outcomes for the *chosen* action. The agent never sees it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.models.decision_engine import (RETRY_SOON, SEND_LINK)

log = logging.getLogger(__name__)

SUCCESS = "SUCCESS"
FAILED = "FAILED"
EXPIRED = "EXPIRED"
USER_DECLINED = "USER_DECLINED"
TIMED_OUT = "TIMED_OUT"

ALL_OUTCOMES = [SUCCESS, FAILED, EXPIRED, USER_DECLINED, TIMED_OUT]

# failure sub-state weights once success is missed (per action family)
_FAIL_SPLIT_RETRY = {FAILED: 0.75, TIMED_OUT: 0.25}          # background retries
_FAIL_SPLIT_VISIBLE = {FAILED: 0.55, USER_DECLINED: 0.30, EXPIRED: 0.15}  # link/switch


@dataclass(frozen=True)
class AttemptResult:
    """The outcome of one simulated execution."""

    txn_id: str
    action: str
    attempt_number: int
    outcome: str                       # one of ALL_OUTCOMES
    amount: float
    recovered_value: float             # amount if SUCCESS else 0
    idempotent_replay: bool = False    # True when served from the store


class PaymentSimulator:
    """Outcome engine over the latent oracle; deterministic per (seed, action order)."""

    def __init__(self, oracle: pd.DataFrame,
                 seed: int = 42,
                 idempotent: bool = True):
        """
        ``oracle``: frame with ``txn_id``, ``amount`` and the ``p_success_*``
        columns (generated truth). The simulator is the only component allowed
        to read them.
        """
        needed = {"txn_id", "amount",
                  "p_success_retry_soon", "p_success_retry_later",
                  "p_success_switch_method", "p_success_send_link"}
        missing = needed - set(oracle.columns)
        if missing:
            raise KeyError(f"oracle missing columns: {sorted(missing)}")
        self._oracle = oracle.set_index("txn_id")
        self._rng = np.random.default_rng(seed)
        self._store: dict[tuple[str, str, int], AttemptResult] = {}
        self.idempotent = idempotent

    # ---------------------------------------------------------------- oracle
    def p_success(self, txn_id: str, action: str) -> float:
        """Latent P(success | txn, action) — evaluator-only access."""
        row = self._oracle.loc[txn_id]
        return float(row.get(f"p_success_{action}", 0.0))

    # -------------------------------------------------------------- execution
    def execute(self, txn_id: str, action: str, attempt_number: int = 1) -> AttemptResult:
        """Simulate one execution of ``action`` on ``txn_id`` (idempotent)."""
        key = (txn_id, action, attempt_number)
        if self.idempotent and key in self._store:
            prior = self._store[key]
            return AttemptResult(prior.txn_id, prior.action, prior.attempt_number,
                                 prior.outcome, prior.amount, prior.recovered_value,
                                 idempotent_replay=True)

        row = self._oracle.loc[txn_id]
        amount = float(row["amount"])
        p = self.p_success(txn_id, action)
        outcome = self._sample_outcome(p, action)
        value = amount if outcome == SUCCESS else 0.0
        result = AttemptResult(txn_id, action, attempt_number, outcome, amount, value)
        if self.idempotent:
            self._store[key] = result
        return result

    def _sample_outcome(self, p: float, action: str) -> str:
        if self._rng.random() < float(p):
            return SUCCESS
        split = _FAIL_SPLIT_RETRY if action == RETRY_SOON or action == "retry_later" \
            else _FAIL_SPLIT_VISIBLE
        states = list(split.keys())
        probs = list(split.values())
        leftover = 1.0 - sum(probs)
        probs = [x * (1.0 - leftover) for x in probs]  # normalize defensively
        return self._rng.choice(states, p=probs)

    # ---------------------------------------------------------- batch helpers
    def execute_batch(self, jobs: list[tuple[str, str, int]]) -> list[AttemptResult]:
        """Execute a list of (txn_id, action, attempt_number) jobs in order."""
        return [self.execute(*j) for j in jobs]

    def recovered_value(self, results: list[AttemptResult]) -> float:
        return float(sum(r.recovered_value for r in results))

    # ------------------------------------------------------------ simulation
    def run_simulation(self, schedule, tracker, rng_override=None) -> list[AttemptResult]:
        """Full clocked simulation over a RetryScheduler (used by the driver).

        Steps each due job when its time arrives, respects the tracker's attempt
        budget, cancels the payment on SUCCESS, and books the outcome. Returns
        all executed AttemptResults.

        Virtual clock: this is a *batch* simulation (no real waiting), so when
        more jobs are due at ``now`` than the bandwidth bucket allows in one
        minute, the clock advances one bandwidth period so the bucket refills —
        the backlog drains at exactly ``bandwidth_per_minute`` per batch.
        """
        results: list[AttemptResult] = []
        virtual_now = schedule.next_due_at()
        while virtual_now is not None:
            for job in schedule.drain_due(virtual_now):
                if not tracker.can_attempt(job.txn_id, job.customer_id, virtual_now):
                    continue
                result = self.execute(job.txn_id, job.action, job.attempt_number)
                tracker.record(job.txn_id, job.customer_id, virtual_now,
                               success=(result.outcome == SUCCESS))
                if result.outcome == SUCCESS:
                    schedule.cancel(job.txn_id)   # recovered → no more attempts
                results.append(result)
            # throttled: jobs still due at the same timestamp, bucket dry →
            # wait one bandwidth period (60s) so the bucket refills
            if schedule.next_due_at() is not None and schedule.next_due_at() <= virtual_now:
                virtual_now += 60.0
            else:
                virtual_now = schedule.next_due_at()  # jump to next due window
        return results