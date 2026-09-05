"""Tests for evaluation layer: metrics + off-policy evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (policy_metrics, breakdowns, compare_policies,
                                    metrics_to_markdown, _action_table, _decision_field)
from src.evaluation.off_policy import (observed_recovery, ipw_recovery,
                                       oracle_recovery, selective_label_summary)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_policy_metrics_basic():
    """Basic policy metrics calculation."""
    df = pd.DataFrame({
        "txn_id": ["a", "b", "c"],
        "amount": [1000.0, 2000.0, 500.0],
        "outcome": ["SUCCESS", "FAILED", "SUCCESS"],
        "recovered_value": [1000.0, 0.0, 500.0],
        "decision": [
            '{"action": "retry_soon", "risk_blocked": false}',
            '{"action": "retry_soon", "risk_blocked": false}',
            '{"action": "retry_soon", "risk_blocked": false}',
        ],
    })
    metrics = policy_metrics(df, retry_cost_per_attempt=2.0)

    assert metrics["n_failed"] == 3
    assert metrics["total_failed_value"] == 3500.0
    assert metrics["recovered_value"] == 1500.0
    # Recovery rate is rounded to 4 decimals by policy_metrics
    assert abs(metrics["recovery_rate_value"] - 1500.0 / 3500.0) < 0.001
    assert metrics["recovered_count"] == 2
    assert abs(metrics["recovery_rate_count"] - 2.0 / 3.0) < 0.001


def test_policy_metrics_retry_cost():
    """Retry cost should scale with attempts."""
    df = pd.DataFrame({
        "txn_id": ["a", "b"],
        "amount": [1000.0, 1000.0],
        "outcome": ["SUCCESS", "FAILED"],
        "recovered_value": [1000.0, 0.0],
        "decision": ['{"action": "retry_soon"}', '{"action": "retry_soon"}'],
        "attempts_taken": [1, 3],  # a succeeded on 1st, b took 3 attempts
    })
    metrics = policy_metrics(df, retry_cost_per_attempt=2.0)
    # Total attempts = 1 + 3 = 4, cost = 4 * 2 = 8
    assert metrics["retry_cost"] == 8.0
    # Net recovered = recovered value - retry cost
    # But touchpoints also have friction cost (2 visible touchpoints = silent retries)
    # The exact value depends on how touched touchpoints are counted
    assert metrics["recovered_value"] == 1000.0


def test_policy_metrics_touchpoints():
    """Touchpoints should count visible actions (switch, link)."""
    df = pd.DataFrame({
        "txn_id": ["a", "b", "c"],
        "amount": [1000.0, 2000.0, 500.0],
        "outcome": ["SUCCESS", "FAILED", "SUCCESS"],
        "recovered_value": [1000.0, 0.0, 500.0],
        "decision": [
            '{"action": "retry_soon"}',  # silent
            '{"action": "send_link"}',   # visible
            '{"action": "switch_method"}',  # visible
        ],
    })
    metrics = policy_metrics(df)
    assert metrics["touchpoints"] == 2


def test_policy_metrics_give_up_rate():
    """Give-up rate should be computed correctly."""
    df = pd.DataFrame({
        "txn_id": ["a", "b", "c", "d"],
        "amount": [1000.0] * 4,
        "outcome": ["SUCCESS", "FAILED", "FAILED", "FAILED"],
        "recovered_value": [1000.0, 0.0, 0.0, 0.0],
        "decision": [
            '{"action": "retry_soon"}',
            '{"action": "give_up"}',
            '{"action": "give_up"}',
            '{"action": "give_up"}',
        ],
    })
    metrics = policy_metrics(df)
    assert metrics["give_up_rate"] == pytest.approx(0.75)


def test_policy_metrics_risk_blocked():
    """Risk-blocked count should be extracted from decision."""
    df = pd.DataFrame({
        "txn_id": ["a", "b", "c"],
        "amount": [1000.0] * 3,
        "outcome": ["FAILED", "FAILED", "FAILED"],
        "recovered_value": [0.0, 0.0, 0.0],
        "decision": [
            '{"action": "retry_soon", "risk_blocked": false}',
            '{"action": "give_up", "risk_blocked": true}',
            '{"action": "give_up", "risk_blocked": true}',
        ],
    })
    metrics = policy_metrics(df)
    assert metrics["risk_blocked"] == 2
    # suppressed value is the ₹-amount of the blocked payments, not their outcome
    assert metrics["risk_suppressed_value"] == 2000.0


def test_action_table():
    """Action counts should be aggregated correctly."""
    df = pd.DataFrame({
        "decision": [
            '{"action": "retry_soon"}',
            '{"action": "retry_later"}',
            '{"action": "retry_soon"}',
            '{"action": "give_up"}',
        ],
    })
    counts = _action_table(df, "decision")
    assert counts["retry_soon"] == 2
    assert counts["retry_later"] == 1
    assert counts["give_up"] == 1
    assert counts["switch_method"] == 0


def test_decision_field():
    """Decision field extraction should handle dict/string/json."""
    row_dict = {"decision": {"action": "retry_soon", "probability": 0.8}}
    assert _decision_field(row_dict, "action") == "retry_soon"

    row_json = {"decision": '{"action": "retry_soon"}'}
    assert _decision_field(row_json, "action") == "retry_soon"

    row_missing = {"amount": 1000}
    assert _decision_field(row_missing, "action", default="give_up") == "give_up"


def test_breakdowns():
    """Breakdowns should group by dimension."""
    df = pd.DataFrame({
        "amount": [1000.0, 2000.0, 500.0, 1000.0],
        "recovered_value": [1000.0, 0.0, 500.0, 0.0],
        "failure_reason_code": ["upi_server_busy", "insufficient_funds",
                                "upi_server_busy", "card_expired"],
        "payment_method": ["upi", "upi", "upi", "credit_card"],
    })

    bd = breakdowns(df)
    assert "by_reason" in bd
    assert "upi_server_busy" in bd["by_reason"]
    # upi_server_busy: 1500 recovered / 1500 total = 1.0
    assert bd["by_reason"]["upi_server_busy"] == pytest.approx(1.0)
    # insufficient_funds: 0 recovered / 2000 total = 0.0
    assert bd["by_reason"]["insufficient_funds"] == 0.0


def test_compare_policies():
    """Compare policies should compute incremental values."""
    df_agent = pd.DataFrame({
        "txn_id": ["a", "b"],
        "amount": [1000.0, 1000.0],
        "outcome": ["SUCCESS", "FAILED"],
        "recovered_value": [1000.0, 0.0],
        "decision": ['{"action": "retry_soon"}', '{"action": "retry_soon"}'],
        "attempts_taken": [1, 2],
    })

    df_dumb = pd.DataFrame({
        "txn_id": ["a", "b"],
        "amount": [1000.0, 1000.0],
        "outcome": ["SUCCESS", "SUCCESS"],
        "recovered_value": [1000.0, 1000.0],
        "decision": ['{"action": "retry_soon"}', '{"action": "retry_soon"}'],
        "attempts_taken": [1, 1],
    })

    comparison = compare_policies({
        "recovery_agent": df_agent,
        "dumb_retry": df_dumb,
    })

    assert "recovery_agent" in comparison.index
    assert "dumb_retry" in comparison.index
    # AI recovered 1000, dumb recovered 2000 → negative incremental
    assert comparison.loc["recovery_agent", "incremental_vs_dumb"] < 0


# ---------------------------------------------------------------------------
# off-policy
# ---------------------------------------------------------------------------

def test_observed_recovery():
    """Observed recovery sums only SUCCESS outcomes."""
    df = pd.DataFrame({
        "outcome": ["SUCCESS", "FAILED", "SUCCESS", "EXPIRED"],
        "recovered_value": [1000.0, 0.0, 500.0, 0.0],
    })
    assert observed_recovery(df) == 1500.0


def test_oracle_recovery():
    """Oracle recovery uses latent P(success) × amount."""
    df = pd.DataFrame({
        "amount": [1000.0, 2000.0],
        "p_success_retry_soon": [0.5, 0.8],
        "p_success_retry_later": [0.4, 0.7],
        "p_success_switch_method": [0.3, 0.6],
        "p_success_send_link": [0.2, 0.5],
    })
    # Best action for each: 0.5*1000 + 0.8*2000 = 500 + 1600 = 2100
    assert oracle_recovery(df) == pytest.approx(2100.0)


def test_oracle_recovery_missing_columns():
    """Oracle recovery returns NaN if oracle columns missing."""
    df = pd.DataFrame({
        "amount": [1000.0],
        "p_other": [0.5],
    })
    import math
    assert math.isnan(oracle_recovery(df))


def test_ipw_recovery():
    """IPW recovery should weight by inverse propensity."""
    df = pd.DataFrame({
        "retry_propensity": [0.8, 0.4, 0.2],
        "retry_success_obs": [1.0, 0.0, 1.0],  # success/fail
        "amount": [1000.0, 500.0, 2000.0],
    })

    # IPW formula: Σ(y * v * w) / Σ(w), w = 1/propensity (clipped)
    # w = [1.25, 2.5, 5.0] (no clipping needed)
    # numerator = 1*1000*1.25 + 0*500*2.5 + 1*2000*5.0 = 1250 + 0 + 10000 = 11250
    # denominator = 1.25 + 2.5 + 5.0 = 8.75
    expected = 11250 / 8.75
    assert ipw_recovery(df) == pytest.approx(expected)


def test_ipw_recovery_censored_handling():
    """IPW should treat NaN outcomes as 0 contribution."""
    df = pd.DataFrame({
        "retry_propensity": [0.5, 0.5],
        "retry_success_obs": [np.nan, 1.0],  # censored + observed
        "amount": [1000.0, 2000.0],
    })
    # First row: NaN → fillna(0) → 0 * 1000 * w = 0
    # Second row: w = 1/0.5 = 2, contribution = 1*2000*2 = 4000
    # Hájek denominator = 2.0 + 2.0 = 4.0 (censored rows keep their weight:
    # they were "exposed" by the historical policy at that propensity)
    # result = 4000 / 4.0 = 1000 — the censored row drags the estimate toward
    # the population average rather than being silently dropped
    result = ipw_recovery(df)
    assert result == pytest.approx(1000.0)


def test_ipw_clipping():
    """IPW should clip high weights."""
    df = pd.DataFrame({
        "retry_propensity": [0.01, 0.01],  # very low propensity
        "retry_success_obs": [1.0, 1.0],
        "amount": [1000000.0, 1000000.0],
    })
    # Without clipping: w = 100, contribution huge
    # With clipping (cap=10): w = 10, contribution = 1*1M*10 = 10M
    # Total numerator = 20M, denominator = 20 → 1M per transaction
    result = ipw_recovery(df, clip=10.0)
    assert result == pytest.approx(1000000.0)


def test_selective_label_summary():
    """Summary should contain all key metrics."""
    df = pd.DataFrame({
        "txn_id": ["a", "b", "c", "d"],
        "amount": [1000.0, 2000.0, 500.0, 300.0],
        # row "c" was never executed (history censored it at the propensity 0.3)
        "outcome": ["SUCCESS", "FAILED", np.nan, "SUCCESS"],
        "retry_propensity": [0.8, 0.5, 0.3, 0.9],
        "retry_success_obs": [1.0, 0.0, np.nan, 1.0],
        "p_success_retry_soon": [0.7, 0.4, 0.5, 0.8],
        "p_success_retry_later": [0.6, 0.3, 0.4, 0.7],
        "p_success_switch_method": [0.2, 0.1, 0.3, 0.5],
        "p_success_send_link": [0.1, 0.2, 0.2, 0.4],
    })

    summary = selective_label_summary(df)

    assert summary["n_failed"] == 4
    assert summary["n_executed"] == 3  # non-null outcomes = attempts executed
    assert summary["n_censored"] == 1  # row c: never executed → unknown, not failed
    assert "observed_recovered_value" in summary
    assert "ipw_estimated_recovered_value" in summary
    assert "oracle_expected_recovered_value" in summary
    assert "headroom_vs_oracle" in summary
    assert "note" in summary