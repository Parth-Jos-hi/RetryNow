"""AttemptTracker: the budget keeper — how many times may we try, and how often.

Two independent caps (mirroring ``configs/config.yaml -> decision_engine``):

  * ``max_attempts_per_payment``   — a payment is retried at most N times total.
  * ``max_retries_per_customer_day``— a customer is contacted at most M times/day.

The tracker is pure in-memory state; during a simulation the scheduler asks
``can_attempt`` before executing a job and ``record`` after.  In a live system
this class would be backed by a KV store with the same interface.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Optional


def _day(ts) -> date:
    """Normalize a timestamp into the calendar day used for daily caps."""
    if isinstance(ts, (datetime, date)):
        return ts.date() if isinstance(ts, datetime) else ts
    if hasattr(ts, "date"):            # pd.Timestamp
        return ts.date()
    return datetime.fromtimestamp(float(ts)).date()


class AttemptTracker:
    def __init__(self, max_attempts_per_payment: int = 3,
                 max_retries_per_customer_day: int = 5):
        self.max_attempts_per_payment = max(1, int(max_attempts_per_payment))
        self.max_retries_per_customer_day = max(1, int(max_retries_per_customer_day))
        self._per_payment: dict[str, int] = {}          # txn_id -> attempts spent
        self._per_customer_day: dict[tuple[str, date], int] = {}

    # ----------------------------------------------------------------- state
    def attempts(self, txn_id: str) -> int:
        """Number of retries already spent on this payment."""
        return self._per_payment.get(txn_id, 0)

    def customer_retries_on(self, customer_id: str, ts) -> int:
        """Retries already spent contacting this customer on that calendar day."""
        return self._per_customer_day.get((customer_id, _day(ts)), 0)

    # ------------------------------------------------------------------ gates
    def can_attempt(self, txn_id: str, customer_id: str, ts) -> bool:
        """Is another retry allowed under both budgets, right now?"""
        if self.attempts(txn_id) >= self.max_attempts_per_payment:
            return False
        if self.customer_retries_on(customer_id, ts) >= self.max_retries_per_customer_day:
            return False
        return True

    def record(self, txn_id: str, customer_id: str, ts,
               success: Optional[bool] = None) -> None:
        """Atomically check-and-record: raises ``AttemptLimit`` if out of budget.

        ``success=True`` also freezes the payment (a recovered payment is never
        retried again, even if a stale job slips through the scheduler).
        """
        if not self.can_attempt(txn_id, customer_id, ts):
            raise AttemptLimit(
                f"txn {txn_id} / customer {customer_id}: retry budget exhausted")
        self._per_payment[txn_id] = self.attempts(txn_id) + 1
        day = _day(ts)
        key = (customer_id, day)
        self._per_customer_day[key] = self._per_customer_day.get(key, 0) + 1
        if success:
            self._per_payment[txn_id] = self.max_attempts_per_payment  # freeze


class AttemptLimit(Exception):
    """Raised when a retry attempt would exceed the configured budgets."""