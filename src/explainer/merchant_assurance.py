"""Merchant-facing assurance: 'will this failed payment still recover?'

A failed payment is a moment of panic for the merchant — their mental model
is "lost order" → cancel / refund / re-issue. In reality most failures are
transient and already being recovered. This module turns the decision engine's
single chosen action into an honest, confidence-bearing note for the merchant
dashboard: whether the payment is EXPECTED to complete, roughly WHEN, and how
confident we are — so they can keep the order open and not break trust.

This mirrors the customer-facing ``explainer.customer_message`` but pointed at
the merchant. It is deterministic (no LLM) so assurance can never go down with
an external service.

``outcome`` values (the honesty contract with the merchant):
  * ``will``       — a retry is already scheduled; the payment is expected to
                     complete on its own (high P, silent auto-retry).
  * ``may``        — a customer action is needed (pay the link / pick another
                     instrument); recovery is likely but not automatic.
  * ``won't``      — the agent deliberately stopped; this payment will not
                     complete, so the merchant should treat the order as failed.
  * ``blocked``    — suppressed by the risk gate, not "unrecoverable"; goes to
                     manual compliance review, not a blanket failure.
"""
from __future__ import annotations

from src.models.decision_engine import (GIVE_UP, RETRY_LATER, RETRY_SOON,
                                        SEND_LINK, SWITCH_METHOD)

_INR = "₹{amount:,.0f}"
# rough Razorpay fee share shown in the benefit story (documented proxy, not a
# contract) — only for the demo's joint-benefit framing.
FEE_RATE = 0.02


def merchant_assurance(decision: dict) -> dict:
    """Map one decision dict -> {outcome, headline, message, benefit...}.

    ``decision`` is the per-row decision produced by the recovery pipeline
    (keys: action, amount, probability, ev, risk_blocked, reason,
    retry_in_hours, failure_reason_code, ...).
    """
    action = decision.get("action")
    amount = float(decision.get("amount", 0.0))
    p = float(decision.get("probability", 0.0))
    blocked = bool(decision.get("risk_blocked"))
    retry_in = decision.get("retry_in_hours")
    reason = decision.get("failure_reason_code") or decision.get("reason") or ""
    amt = _INR.format(amount=amount)

    if action == GIVE_UP and blocked:
        outcome = "blocked"
        headline = "Held for risk review — not failed"
        message = (f"{amt} was suppressed by the risk gate before any recovery "
                   f"attempt. Treat it as paused for compliance review, not lost.")
    elif action == GIVE_UP:
        outcome = "won't"
        headline = "Won't recover — release the order"
        message = (f"{amt} is {_never_reason(reason)}. The agent stopped rather "
                   f"than spend more on a hopeless retry — mark the order failed.")
    elif action == RETRY_SOON:
        when = f" in about {int(retry_in) if retry_in else 2} hours"
        outcome = "will"
        headline = f"Will recover — auto-retrying{when}"
        message = (f"{amt} is expected to complete (≈{p:.0%}). A retry is already "
                   f"scheduled{when} — no action needed, keep the order open.")
    elif action == RETRY_LATER:
        when = (f" in about {int(retry_in) if retry_in else 72} hours"
                if retry_in else " after the funded/off-peak window")
        outcome = "will"
        headline = f"Will recover — auto-retrying later{when}"
        message = (f"{amt} is expected to complete (≈{p:.0%}). Recovery is "
                   f"scheduled to run{when}, usually right after the customer's "
                   f"funding lands. No action needed.")
    elif action == SWITCH_METHOD:
        outcome = "may"
        headline = "Recovering via an alternative instrument"
        message = (f"{amt} failed on the current method but another instrument is "
                   f"available (≈{p:.0%}). A switch offer was made — likely to "
                   f"complete if the customer opts in.")
    elif action == SEND_LINK:
        outcome = "may"
        headline = "Recovery link sent to the customer"
        message = (f"{amt} is below the auto-retry bar but not hopeless (≈{p:.0%}). "
                   f"A payment link was sent so the customer can pay at their "
                   f"convenience — likely to complete.")
    else:  # safety net
        outcome = "won't"
        headline = "No recovery action"
        message = f"No recovery action was available for {amt}."
    return {
        "outcome": outcome,
        "headline": headline,
        "message": message,
        "merchant_benefit": round(amount * p, 2),       # expected ₹ recovered
        "razorpay_fee": round(amount * p * FEE_RATE, 2),  # expected fee on it
    }


def _never_reason(reason: str) -> str:
    r = (reason or "").lower()
    if "card_expired" in r or "expired" in r:
        return "failing on an expired card"
    if "account_blocked" in r or "blocked" in r:
        return "failing on a blocked account"
    if "limit" in r:
        return "above the permitted limit for that instrument"
    return "a cause the engine judged not economically worth pursuing"


def benefit_row(decision: dict) -> dict:
    """One row of the joint-benefit story (merchant CSV → both parties win)."""
    a = merchant_assurance(decision)
    return {
        "txn_id": decision.get("txn_id"),
        "amount": decision.get("amount"),
        "action": decision.get("action"),
        "outcome": a["outcome"],
        "assurance": a["headline"],
        "merchant_benefit": a["merchant_benefit"],
        "razorpay_fee": a["razorpay_fee"],
    }