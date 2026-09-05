"""Tests for the recovery layer: scheduler, attempt tracker, messaging."""

import pandas as pd
import pytest

from src.models.decision_engine import DecisionEngine
from src.recovery.attempt_tracker import AttemptLimit, AttemptTracker
from src.recovery.messaging import channel_for, compose_notification
from src.recovery.scheduler import (RetryJob, RetryScheduler,
                                    schedule_from_decisions)

T0 = 1_700_000_000.0  # fixed epoch for tests


def job(txn="pay1", due=T0, action="retry_soon", method="upi", p=0.8, n=1):
    return RetryJob(txn_id=txn, customer_id="cus1", due_at=due, p_success=p,
                    method=method, action=action, attempt_number=n)


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------

def test_drain_releases_due_jobs_in_order():
    s = RetryScheduler()
    s.add(job("pay2", due=T0 + 100))
    s.add(job("pay1", due=T0))            # earlier — must come out first
    out = s.drain_due(T0 + 1000, capacity=100)
    assert [j.txn_id for j in out] == ["pay1", "pay2"]


def test_drain_keeps_future_jobs_pending():
    s = RetryScheduler()
    s.add(job("pay1", due=T0))
    s.add(job("pay2", due=T0 + 3600))
    out = s.drain_due(T0, capacity=100)
    assert [j.txn_id for j in out] == ["pay1"]
    assert s.has_pending
    assert s.next_due_at() == T0 + 3600


def test_bandwidth_cap_limits_release():
    s = RetryScheduler()
    for i in range(5):
        s.add(job(f"pay{i}", due=T0 + i))
    out = s.drain_due(T0 + 10, capacity=2)   # only 2 slots
    assert len(out) == 2
    assert s.has_pending                     # the other 3 wait for the next drain


def test_bandwidth_over_time_smoothes_bursts():
    # a 100-job backlog arriving at once must drain at ≤30/min, never bursting
    s = RetryScheduler(bandwidth_per_minute=30)
    for i in range(100):
        s.add(job(f"pay{i}", due=T0))
    batches = [s.drain_due(T0 + 60 * m) for m in range(6)]  # drain once a minute
    total = sum(len(b) for b in batches)
    assert total == 100                          # everything eventually drains
    assert all(len(b) <= 30 for b in batches)    # per-minute ceiling never broken


def test_run_simulation_drains_throttled_backlog():
    """A backlog larger than the per-minute bandwidth must fully drain through
    PaymentSimulator.run_simulation (the virtual-clock path) — regression for
    the infinite loop where a throttled batch never advanced the clock."""
    import numpy as np
    import pandas as pd
    from src.recovery.attempt_tracker import AttemptTracker
    from src.simulator.payment_simulator import PaymentSimulator

    rng = np.random.default_rng(0)
    oracle = pd.DataFrame({
        "txn_id": [f"pay{i}" for i in range(70)],
        "amount": rng.uniform(100, 5000, 70),
        "p_success_retry_soon": 0.4,
        "p_success_retry_later": 0.3,
        "p_success_switch_method": 0.2,
        "p_success_send_link": 0.1,
    })
    sim = PaymentSimulator(oracle=oracle, seed=1)
    sched = RetryScheduler(bandwidth_per_minute=30)
    for i in range(70):
        sched.add(RetryJob(txn_id=f"pay{i}", customer_id=f"cus{i}",
                           due_at=T0, p_success=0.4, method="upi",
                           action="retry_soon", attempt_number=1))
    tracker = AttemptTracker(max_attempts_per_payment=3,
                             max_retries_per_customer_day=5)

    results = sim.run_simulation(sched, tracker)   # must terminate

    assert len(results) == 70                       # every job executed once
    assert not sched.has_pending                    # backlog fully drained


def test_cancel_drops_future_jobs_and_has_pending_respects_it():
    s = RetryScheduler()
    s.add(job("pay1", due=T0))
    s.cancel("pay1")
    assert not s.has_pending
    assert s.drain_due(T0 + 1000, capacity=100) == []
    assert s.next_due_at() is None


def test_schedule_from_decisions_skips_non_retryable():
    engine = DecisionEngine()
    decisions = [
        {"txn_id": "a", "customer_id": "c1", "action": "retry_soon",
         "retry_in_hours": 2, "probability": 0.7, "payment_method": "upi"},
        {"txn_id": "b", "customer_id": "c1", "action": "send_link",
         "retry_in_hours": None, "probability": 0.1, "payment_method": "upi"},
        {"txn_id": "c", "customer_id": "c2", "action": "give_up",
         "retry_in_hours": None, "probability": 0.01, "payment_method": "upi"},
    ]
    s = RetryScheduler()
    n = schedule_from_decisions(decisions, pd.Timestamp("2025-01-01 10:00:00"), s)
    assert n == 1
    assert s.next_due_at() == pd.Timestamp("2025-01-01 10:00:00").timestamp() + 2 * 3600


# ---------------------------------------------------------------------------
# attempt tracker
# ---------------------------------------------------------------------------

def test_tracker_enforces_payment_attempt_budget():
    t = AttemptTracker(max_attempts_per_payment=2, max_retries_per_customer_day=5)
    assert t.can_attempt("pay1", "cus1", T0)
    t.record("pay1", "cus1", T0)
    t.record("pay1", "cus1", T0 + 3600)
    assert t.attempts("pay1") == 2
    assert not t.can_attempt("pay1", "cus1", T0 + 7200)   # budget spent
    with pytest.raises(AttemptLimit):
        t.record("pay1", "cus1", T0 + 7200)


def test_tracker_enforces_customer_daily_budget():
    t = AttemptTracker(max_attempts_per_payment=10, max_retries_per_customer_day=2)
    t.record("pay1", "cus1", pd.Timestamp("2025-01-01 10:00"))
    t.record("pay2", "cus1", pd.Timestamp("2025-01-01 11:00"))
    assert not t.can_attempt("pay3", "cus1", pd.Timestamp("2025-01-01 12:00"))
    # next day the budget resets
    assert t.can_attempt("pay4", "cus1", pd.Timestamp("2025-01-02 09:00"))


def test_tracker_budgets_are_per_customer():
    t = AttemptTracker(max_attempts_per_payment=10, max_retries_per_customer_day=2)
    t.record("pay1", "cus1", T0)
    t.record("pay2", "cus1", T0 + 60)
    assert not t.can_attempt("pay3", "cus1", T0 + 120)  # cus1 capped
    assert t.can_attempt("pay4", "cus2", T0 + 120)      # cus2 untouched


def test_tracker_success_freeze():
    t = AttemptTracker(max_attempts_per_payment=3, max_retries_per_customer_day=5)
    t.record("pay1", "cus1", T0, success=True)
    assert not t.can_attempt("pay1", "cus1", T0 + 60)   # recovered: never retry


# ---------------------------------------------------------------------------
# messaging
# ---------------------------------------------------------------------------

def test_channel_mapping():
    assert channel_for("retry_soon") == "push"
    assert channel_for("retry_later") == "push"
    assert channel_for("switch_method") == "push"
    assert channel_for("send_link") == "email_or_sms"
    assert channel_for("give_up") is None               # silent by design


def test_compose_notification_carries_job_details():
    n = compose_notification(job(), txn={"amount": 500}, message="hi")
    assert n["txn_id"] == "pay1" and n["channel"] == "push"
    assert n["attempt_number"] == 1 and n["p_success"] == 0.8
    assert n["txn"] == {"amount": 500} and n["message"] == "hi"