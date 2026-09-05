"""Messaging: which channel a recovery action uses, and the notification payloads.

The explainer composes the *customer-facing copy*; this module owns the routing
— channel per action — and the operational notification that accompanies each
attempt (sent to the merchant ops feed / logs).  ``GIVE_UP`` routes nowhere:
that is the point of the engine — stop nudging after the budget is spent.
"""

from __future__ import annotations

import json
from typing import Optional

from src.models.decision_engine import (GIVE_UP, RETRY_LATER, RETRY_SOON,
                                        SEND_LINK, SWITCH_METHOD)

# Channel per action.  Silent push for silent retries; a channel with a message
# for anything that asks the customer to do something.
CHANNELS = {
    RETRY_SOON: "push",            # silent background retry, device notification
    RETRY_LATER: "push",           # silent scheduled retry
    SWITCH_METHOD: "push",         # "want to try this bank card instead?"
    SEND_LINK: "email_or_sms",     # explicit payment link
    GIVE_UP: None,                 # no contact — respect the customer
}


def channel_for(action: str) -> Optional[str]:
    """Return the contact channel for an action, or None if we should stay silent."""
    return CHANNELS.get(action)


def compose_notification(job, txn: Optional[dict] = None,
                         message: Optional[str] = None) -> dict:
    """Operational notification for one executed retry job.

    ``job`` is a ``RetryJob``; ``txn`` (optional) carries amount/merchant details
    for the ops log; ``message`` (optional) is the customer copy from the
    explainer.  Returns a JSON-serializable dict, intended for an ops feed.
    """
    payload = {
        "txn_id": job.txn_id,
        "customer_id": job.customer_id,
        "action": job.action,
        "channel": channel_for(job.action),
        "attempt_number": job.attempt_number,
        "p_success": round(float(job.p_success), 4),
    }
    if txn is not None:
        payload["txn"] = txn
    if message is not None:
        payload["message"] = message
    return payload


def to_json_line(payload: dict) -> str:
    """Serialize a notification as one line of a JSON-lines ops log."""
    return json.dumps(payload, ensure_ascii=False, default=str)