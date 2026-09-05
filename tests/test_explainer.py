"""Tests for the explainer: template mode (deterministic, offline) and
LLM-mode behavior when the openai client is unavailable/failing."""

import pytest

from src.explainer.explainer import Explainer
from src.models.decision_engine import (GIVE_UP, RETRY_LATER, RETRY_SOON,
                                        SEND_LINK, SWITCH_METHOD)

TXN = {"amount": 1250, "merchant_vertical": "ecommerce",
       "payment_method": "upi", "failure_reason_code": "upi_server_busy"}


def make_explainer(provider="template", **kwargs):
    return Explainer(provider=provider, **kwargs)


# ---------------------------------------------------------------------------
# template mode — deterministic and offline
# ---------------------------------------------------------------------------

def test_template_mode_is_offline_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    exp = Explainer(provider="auto")
    assert exp.provider == "template"
    assert not exp.is_live


def test_explain_every_action_has_template():
    exp = make_explainer()
    cases = [
        ({"action": RETRY_SOON, "probability": 0.7, "retry_in_hours": 2},
         "high confidence"),
        ({"action": RETRY_LATER, "probability": 0.3, "retry_in_hours": 72},
         "moderate confidence"),
        ({"action": SWITCH_METHOD, "probability": 0.5}, "switch"),
        ({"action": SEND_LINK, "probability": 0.1}, "payment link"),
        ({"action": GIVE_UP, "probability": 0.01}, "threshold"),
    ]
    for decision, fragment in cases:
        text = exp.explain_decision(decision, TXN)
        assert isinstance(text, str) and len(text) > 10
        assert fragment in text.lower()


def test_template_explanation_includes_engine_artifacts():
    exp = make_explainer()
    text = exp.explain_decision(
        {"action": RETRY_SOON, "probability": 0.7, "retry_in_hours": 2}, TXN)
    assert "70%" in text and "2h" in text and "upi_server_busy" in text


def test_template_customer_message_mentions_amount_and_merchant():
    exp = make_explainer()
    msg = exp.customer_message(
        {"action": RETRY_SOON, "probability": 0.7, "retry_in_hours": 2}, TXN)
    assert "₹1,250" in msg and "ecommerce" in msg
    assert "no action needed" in msg.lower()


def test_customer_message_none_for_give_up():
    exp = make_explainer()
    assert exp.customer_message({"action": GIVE_UP, "probability": 0.01}, TXN) is None


def test_switch_method_message_names_target():
    exp = make_explainer()
    msg = exp.customer_message(
        {"action": SWITCH_METHOD, "probability": 0.5, "target_method": "credit_card"},
        TXN)
    assert "credit_card" in msg


# ---------------------------------------------------------------------------
# LLM mode — safe degradation when the API is unavailable
# ---------------------------------------------------------------------------

def test_llm_failure_falls_back_to_template(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    class _Broken:
        chat = None  # will raise AttributeError on .completions
    exp = Explainer(provider="openai")  # no key set -> template immediately
    assert exp.provider == "template"

    # simulate: a client exists but calls fail -> must still produce output
    exp._client = _Broken()
    exp.provider = "openai"
    assert exp.is_live
    text = exp.explain_decision({"action": RETRY_SOON, "probability": 0.7}, TXN)
    assert "retry" in text.lower()          # degraded gracefully


def test_missing_openai_package_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no openai installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    exp = Explainer(provider="openai")
    assert exp.provider == "template"       # package missing -> template mode
    assert not exp.is_live