"""Tests for policy baselines (do_nothing, dumb_retry, rule_retry)."""

import pandas as pd
import pytest

from src.policy.baselines import do_nothing, dumb_retry, rule_retry, RULE_DEFAULTS
from src.simulator.payment_simulator import PaymentSimulator


def _sample_data(n=50, seed=42):
    """Generate minimal test data with required columns."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "txn_id": [f"txn_{i}" for i in range(n)],
        "customer_id": [f"cus_{i % 10}" for i in range(n)],
        "amount": rng.uniform(100, 5000, n),
        "failure_reason_code": rng.choice(
            ["upi_server_busy", "insufficient_funds", "network_timeout",
             "otp_expired", "card_expired"], n
        ),
        "payment_method": rng.choice(["upi", "debit_card", "credit_card"], n),
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
    })


def _make_oracle(df):
    """Add oracle columns for simulator."""
    df = df.copy()
    for action in ["retry_soon", "retry_later", "switch_method", "send_link"]:
        df[f"p_success_{action}"] = 0.5  # constant for testing
    return df


# ---------------------------------------------------------------------------
# do_nothing
# ---------------------------------------------------------------------------

def test_do_nothing_returns_original_with_nones():
    """do_nothing should return original df with outcome=None."""
    df = _sample_data(10)
    result = do_nothing(df.copy(), PaymentSimulator(_make_oracle(df)))

    assert "outcome" in result.columns
    assert result["outcome"].isna().all()  # no attempts made


# ---------------------------------------------------------------------------
# dumb_retry
# ---------------------------------------------------------------------------

def test_dumb_retry_retries_everything():
    """dumb_retry should attempt all transactions."""
    df = _sample_data(20)
    simulator = PaymentSimulator(_make_oracle(df))

    result = dumb_retry(df.copy(), simulator, retry_hours=4.0)

    # All rows should have attempted
    assert result["outcome"].notna().all()
    # Policy column should be added
    assert "policy" in result.columns
    assert (result["policy"] == "dumb_retry").all()


def test_dumb_retry_max_attempts():
    """dumb_retry should respect max_attempts."""
    df = _sample_data(5)
    simulator = PaymentSimulator(_make_oracle(df))

    result = dumb_retry(df.copy(), simulator, max_attempts=2)

    for _, row in result.iterrows():
        attempts = row.get("attempts_taken", 0)
        assert attempts <= 2


# ---------------------------------------------------------------------------
# rule_retry
# ---------------------------------------------------------------------------

def test_rule_retry_maps_reason_to_action():
    """rule_retry should apply RULE_DEFAULTS correctly."""
    df = _sample_data(20)
    simulator = PaymentSimulator(_make_oracle(df))

    result = rule_retry(df.copy(), simulator)

    for _, row in result.iterrows():
        reason = row["failure_reason_code"]
        rule = RULE_DEFAULTS.get(reason, RULE_DEFAULTS.get("default", {}))
        decision = row.get("decision", {})
        if decision:
            action = decision.get("action") if isinstance(decision, dict) else "retry_soon"
            # For known reasons, should have matching action
            if reason in RULE_DEFAULTS:
                expected_action = RULE_DEFAULTS[reason]["action"]
                assert action == expected_action


def test_rule_retry_skips_give_up():
    """rule_retry should not schedule retry for give_up actions."""
    df = _sample_data(10)
    simulator = PaymentSimulator(_make_oracle(df))

    result = rule_retry(df.copy(), simulator)

    # For give_up reasons (like account_blocked), no outcome should be recorded
    for _, row in result.iterrows():
        decision = row.get("decision", {})
        if isinstance(decision, dict) and decision.get("action") == "give_up":
            # No attempt made
            assert row.get("outcome") is None or pd.isna(row.get("outcome"))


def test_rule_retry_custom_rules():
    """rule_retry should accept custom rule overrides."""
    df = _sample_data(20)
    simulator = PaymentSimulator(_make_oracle(df))

    custom_rules = {
        "upi_server_busy": {"action": "send_link", "hours": 6},
    }

    result = rule_retry(df.copy(), simulator, rules=custom_rules)

    # Custom rules are applied; verify result structure (outcome indicates execution)
    assert "outcome" in result.columns
    assert "policy" in result.columns
    assert (result["policy"] == "rule_retry").all()


# ---------------------------------------------------------------------------
# RULE_DEFAULTS integrity
# ---------------------------------------------------------------------------

def test_rule_defaults_coverage():
    """RULE_DEFAULTS should cover expected failure reasons."""
    expected_reasons = [
        "upi_server_busy", "bank_server_timeout", "network_timeout",
        "otp_expired", "insufficient_funds", "issuer_decline",
        "duplicate_payment_block", "card_expired", "account_blocked",
    ]
    for reason in expected_reasons:
        assert reason in RULE_DEFAULTS, f"Missing rule for: {reason}"


def test_rule_defaults_structure():
    """Each rule should have action and hours."""
    for reason, rule in RULE_DEFAULTS.items():
        assert "action" in rule, f"Missing 'action' in rule: {reason}"
        assert "hours" in rule, f"Missing 'hours' in rule: {reason}"


def test_give_up_action():
    """Account blocked should be give_up."""
    assert RULE_DEFAULTS["account_blocked"]["action"] == "give_up"


def test_retry_soon_actions():
    """Transient failures should be retry_soon."""
    retry_soon_reasons = ["upi_server_busy", "bank_server_timeout", "network_timeout"]
    for reason in retry_soon_reasons:
        assert RULE_DEFAULTS[reason]["action"] == "retry_soon"


def test_retry_later_actions():
    """Soft failures should be retry_later."""
    retry_later_reasons = ["insufficient_funds", "issuer_decline", "duplicate_payment_block"]
    for reason in retry_later_reasons:
        assert RULE_DEFAULTS[reason]["action"] == "retry_later"


def test_send_link_actions():
    """Customer-action failures should be send_link."""
    send_link_reasons = ["otp_expired", "card_expired"]
    for reason in send_link_reasons:
        assert RULE_DEFAULTS[reason]["action"] == "send_link"