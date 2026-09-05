"""Recovery layer: when to retry (scheduler), how much (tracker), on what channel (messaging).

The three modules work together during simulations and live operation:

    RetryScheduler  — time+bandwidth: turns "retry in 2h" decisions into clocked
                      jobs, releases them when due, throttled by bandwidth.
    AttemptTracker  — budgets: caps retries per payment and per customer-day so
                      the engine never nags a customer into churn.
    messaging.py    — channel routing + notification templates per action.
"""

from src.recovery.attempt_tracker import AttemptTracker
from src.recovery.messaging import channel_for, compose_notification
from src.recovery.scheduler import (RetryJob, RetryScheduler,
                                    schedule_from_decisions)

__all__ = ["AttemptTracker", "RetryJob", "RetryScheduler", "channel_for",
           "compose_notification", "schedule_from_decisions"]