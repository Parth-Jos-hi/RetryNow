"""Explainer: plain-language 'why' + customer-facing recovery messages.

Two outputs per decision:
  * ``explain_decision`` — an internal, ops-facing explanation of *why* the
    decision engine chose this action for this payment (used in the website
    and evaluation reports).
  * ``customer_message`` — the actual copy sent to the customer on the action's
    channel (per ``recovery.messaging`` routing).

Provider resolution (``configs/config.yaml -> explainer.provider``):
  * ``auto``     — use OpenAI if ``OPENAI_API_KEY`` is set, else Groq if
                   ``GROQ_API_KEY`` is set, else deterministic templates.
  * ``openai`` / ``groq`` — force a provider; missing key falls back to template.
  * ``template`` — always deterministic (fully offline).

If the LLM call fails for any reason (timeout, rate limit, bad model name) the
explainer degrades to templates rather than failing the pipeline — recovery
operations must not depend on an external service being up.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from src.models.decision_engine import (GIVE_UP, RETRY_LATER, RETRY_SOON,
                                        SEND_LINK, SWITCH_METHOD)

log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv optional — env vars may already be set
    pass

_MODEL_ENV = "EXPLAINER_MODEL"
_INR_FORMAT = "₹{amount:,.0f}"


class Explainer:
    def __init__(self, provider: str = "auto", model: str = "gpt-4o-mini",
                 temperature: float = 0.4, max_tokens: int = 220):
        self.requested_provider = provider
        self.model = os.environ.get(_MODEL_ENV, model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self.provider = self._resolve_provider(provider)

    # ------------------------------------------------------------- resolution
    def _resolve_provider(self, requested: str) -> str:
        """Pick openai / groq / template; always returns a usable provider."""
        if requested == "template":
            return "template"
        if requested in ("openai", "groq"):
            key = os.environ.get("OPENAI_API_KEY" if requested == "openai"
                                 else "GROQ_API_KEY")
            if key and self._try_client(requested, key):
                return requested
            log.warning("explainer provider '%s' requested but unavailable — "
                        "using template mode", requested)
            return "template"
        # auto
        for candidate, key_name in (("openai", "OPENAI_API_KEY"),
                                    ("groq", "GROQ_API_KEY")):
            key = os.environ.get(key_name)
            if key and self._try_client(candidate, key):
                return candidate
        return "template"

    def _try_client(self, provider: str, api_key: str) -> bool:
        client = self._make_client(provider, api_key)
        if client is not None:
            self._client = client
            return True
        return False

    def _make_client(self, provider: str, api_key: str):
        try:
            from openai import OpenAI
        except ImportError:
            log.warning("openai package not installed — using template mode")
            return None
        if provider == "groq":
            return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        return OpenAI(api_key=api_key)

    @property
    def is_live(self) -> bool:
        """True when generating with an LLM (vs deterministic templates)."""
        return self._client is not None

    # ---------------------------------------------------------------- API
    def explain_decision(self, decision: dict, txn: Optional[dict] = None) -> str:
        """Why did the engine pick this action? (ops-facing, plain language)."""
        if self._client is not None:
            try:
                return self._llm(decision, txn, purpose="explain")
            except Exception as exc:
                log.warning("LLM explain failed, using template: %s", exc)
        return self._explain_template(decision, txn or {})

    def customer_message(self, decision: dict,
                         txn: Optional[dict] = None) -> Optional[str]:
        """Customer-facing copy for the decision's channel; None for give_up."""
        if decision.get("action") == GIVE_UP:
            return None  # we do not contact the customer at all
        if self._client is not None:
            try:
                return self._llm(decision, txn, purpose="message")
            except Exception as exc:
                log.warning("LLM message failed, using template: %s", exc)
        return self._message_template(decision, txn or {})

    # ---------------------------------------------------------------- LLM path
    def _llm(self, decision: dict, txn: Optional[dict], purpose: str) -> str:
        if purpose == "explain":
            system = ("You are the explanation engine of a payment-failure "
                      "recovery system. Explain the next-best action in one "
                      "plain paragraph (2-3 sentences). No markdown, no "
                      "bullet points.")
            user = ("Decision: {decision}\nTransaction: {txn}\n"
                    "Why was {action} chosen for this payment?").format(
                decision=json.dumps(decision, default=str),
                txn=json.dumps(txn or {}, default=str),
                action=decision.get("action"))
        else:
            system = ("You write customer-facing SMS/push messages for a "
                      "payment-failure recovery service (Razorpay-style tone: "
                      "short, warm, honest, no jargon, no scolding). One or two "
                      "sentences max. Mention amount and merchant. No markdown.")
            user = ("Decision: {decision}\nTransaction: {txn}\n"
                    "Write the customer message.").format(
                decision=json.dumps(decision, default=str),
                txn=json.dumps(txn or {}, default=str))
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # --------------------------------------------------------- template paths
    @staticmethod
    def _fmt_amount(amount: Any) -> str:
        try:
            return _INR_FORMAT.format(amount=float(amount))
        except (TypeError, ValueError):
            return f"₹{amount}"

    def _explain_template(self, d: dict, txn: dict) -> str:
        action = d.get("action")
        p = d.get("probability", 0.0)
        method = d.get("target_method") or txn.get("payment_method") or "the same method"
        hours = d.get("retry_in_hours")
        reason = txn.get("failure_reason_code", "the failure")
        amount = self._fmt_amount(txn.get("amount", "the amount"))
        merchant = txn.get("merchant_vertical", "the merchant")

        when = "shortly" if hours is None else f"in ~{hours:.0f}h"

        if action == RETRY_SOON:
            return (f"High confidence retry: ~{p:.0%} estimated success on a quick "
                    f"retry — “{reason}” failures usually clear within hours. "
                    f"Retry {method} {when}.")
        if action == RETRY_LATER:
            return (f"Moderate confidence ({p:.0%}) — “{reason}” often resolves "
                    f"after a day or two, so the retry is deferred {when} "
                    f"rather than burning a fast attempt.")
        if action == SWITCH_METHOD:
            return (f"Current instrument is sticky (~{p:.0%} on {method}); an "
                    f"alternative instrument scores higher, so switch the attempt "
                    f"instead of retrying the same method.")
        if action == SEND_LINK:
            return (f"Below the retry bar ({p:.0%}) but not hopeless — a payment "
                    f"link lets the customer complete {amount} to {merchant} at "
                    f"their own convenience, cheaper than another failed retry.")
        if action == GIVE_UP:
            return (f"Estimated success ~{p:.0%} — below the contact threshold. "
                    f"No further attempts; we stop here to respect the customer.")
        return f"Action '{action}' (p~{p:.0%})."

    def _message_template(self, d: dict, txn: dict) -> str:
        action = d.get("action")
        amount = self._fmt_amount(txn.get("amount", "your payment"))
        merchant = txn.get("merchant_vertical", "your merchant")
        hours = d.get("retry_in_hours")
        method = d.get("target_method") or txn.get("payment_method", "card")

        if action in (RETRY_SOON, RETRY_LATER):
            when = ("in a few hours" if hours is None
                    else f"in about {hours:.0f} hours")
            return (f"Your {amount} payment to {merchant} didn't go through — "
                    f"no action needed. We'll retry automatically {when}.")
        if action == SWITCH_METHOD:
            return (f"Your {amount} payment to {merchant} was declined. Want to "
                    f"retry with your {method} instead? Tap to confirm.")
        if action == SEND_LINK:
            return (f"Your {amount} payment to {merchant} is waiting. Complete it "
                    f"anytime via your payment link — it stays valid for 7 days.")
        return None  # give_up / unknown -> no customer contact