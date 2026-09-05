"""RetryScheduler: a clocked, bandwidth-throttled job queue for retries.

The decision engine answers *what* to do ("retry_soon in 2h"); the scheduler
answers *when*: it holds pending retry jobs ordered by due time, releases them
when the clock passes their due time, and never emits more than
``bandwidth_per_minute`` jobs in a minute (payment-provider throttling).

Cancellation: when a retry actually recovers the payment, ``cancel(txn_id)``
removes all its *future* jobs — the scheduler treats a recovered payment as
done rather than retrying it again.

All times are unix epoch seconds (float) so the scheduler has no timezone
opinion — the caller converts ``pd.Timestamp`` / ``datetime`` as needed.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

from typing import Optional


@dataclass
class RetryJob:
    """One scheduled retry attempt for one failed payment."""

    txn_id: str
    customer_id: str
    due_at: float                # unix epoch seconds — when this retry should fire
    p_success: float             # P(retry succeeds) at scheduling time
    method: str                  # method to retry (same as original, or the target)
    action: str                  # the decision that produced this job
    attempt_number: int = 1      # which attempt of this payment this job is (1-based)
    extra: dict = field(default_factory=dict)


def _epoch(ts) -> float:
    """Accept datetime / pd.Timestamp / float and return epoch seconds."""
    if hasattr(ts, "timestamp"):           # datetime & pd.Timestamp
        return float(ts.timestamp())
    return float(ts)


class RetryScheduler:
    """Min-heap of retry jobs; drain due jobs subject to a bandwidth cap."""

    def __init__(self, bandwidth_per_minute: int = 300):
        self._bandwidth_per_minute = max(1, int(bandwidth_per_minute))
        self._heap: list[tuple[float, int, RetryJob]] = []
        self._seq = itertools.count()
        self._cancelled: set[str] = set()
        # token bucket: capacity == one minute's worth, refills at the same rate
        self._tokens: Optional[float] = None
        self._last_refill_ts: Optional[float] = None

    # ------------------------------------------------------------- scheduling
    def add(self, job: RetryJob) -> None:
        heapq.heappush(self._heap, (job.due_at, next(self._seq), job))

    def cancel(self, txn_id: str) -> None:
        """Drop all future jobs for ``txn_id`` (payment recovered / terminal)."""
        self._cancelled.add(txn_id)

    @property
    def has_pending(self) -> bool:
        """True if any non-cancelled job remains (skips cancelled lazily)."""
        return any(job.txn_id not in self._cancelled
                   for _, _, job in self._heap)

    def next_due_at(self) -> Optional[float]:
        """Epoch time of the earliest non-cancelled pending job, or None."""
        for due, _, job in self._heap:
            if job.txn_id not in self._cancelled:
                return due
        return None

    # ---------------------------------------------------------------- draining
    def drain_due(self, now, capacity: Optional[int] = None) -> list[RetryJob]:
        """Release jobs due at or before ``now``, up to the bandwidth cap.

        ``capacity`` overrides the per-minute bandwidth (used by tests to run a
        whole simulation without artificial throttling).  Cancelled jobs are
        dropped permanently.  Returns released jobs in due order.
        """
        now = _epoch(now)
        limit = self._take_tokens(now) if capacity is None else max(1, int(capacity))
        released: list[RetryJob] = []
        kept: list[tuple[float, int, RetryJob]] = []

        for item in self._heap:
            due, seq, job = item
            if job.txn_id in self._cancelled:
                continue                       # cancelled: drop permanently
            if due <= now and len(released) < limit:
                released.append(job)
            else:
                kept.append(item)

        self._heap = kept
        return released

    # ------------------------------------------------------------- bandwidth
    def _take_tokens(self, now: float) -> int:
        """Token bucket: capacity = rate (one minute's worth), refill at rate.

        A backlog of late-due jobs therefore drains at exactly the per-minute
        ceiling — no burst can exceed ``bandwidth_per_minute`` in one release.
        """
        if self._tokens is None:
            self._tokens = float(self._bandwidth_per_minute)
            self._last_refill_ts = now
        else:
            elapsed = max(0.0, (now - self._last_refill_ts) / 60.0)
            self._tokens = min(
                float(self._bandwidth_per_minute),
                self._tokens + self._bandwidth_per_minute * elapsed)
            self._last_refill_ts = now
        taken = min(int(self._tokens), self._bandwidth_per_minute)
        self._tokens -= taken
        return taken

    # ------------------------------------------------------------- reporting
    def pending_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, job in self._heap:
            if job.txn_id not in self._cancelled:
                counts[job.action] = counts.get(job.action, 0) + 1
        return counts


def schedule_from_decisions(decisions: list[dict],
                            base_time,
                            scheduler: RetryScheduler,
                            id_fields: tuple[str, str] = ("txn_id", "customer_id"),
                            p_field: str = "probability",
                            method_field: str = "payment_method") -> int:
    """Convenience: turn decision dicts into scheduled jobs, skipping
    non-retryable actions (send_link / give_up have no retry slot).

    Decision dicts must carry the row context fields — ``txn_id``,
    ``customer_id`` and ``payment_method`` — e.g. by merging
    ``DecisionEngine.decide_batch`` output row fields with the ``decision``
    dict: ``{**row.decision, "txn_id": row.txn_id, ...}``.  ``retry_in_hours``
    comes from the decision itself (engine already resolved action→hours).
    Returns the number of jobs scheduled.
    """
    n = 0
    for d in decisions:
        action = d.get("action", "")
        retry_in_hours = d.get("retry_in_hours")
        if not action or retry_in_hours is None:
            continue  # send_link / give_up — nothing to schedule
        due = _epoch(base_time) + float(retry_in_hours) * 3600.0
        scheduler.add(RetryJob(
            txn_id=d[id_fields[0]],
            customer_id=d[id_fields[1]],
            due_at=due,
            p_success=float(d.get(p_field, 0.5)),
            method=d.get(method_field) or "same",
            action=action,
        ))
        n += 1
    return n