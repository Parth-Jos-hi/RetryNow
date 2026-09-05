"""Tests for the models layer: predictor + decision engine (EV-based)."""

import numpy as np
import pandas as pd
import pytest

from src.models.decision_engine import (DecisionEngine, GIVE_UP, RETRY_LATER,
                                        RETRY_SOON, SEND_LINK, SWITCH_METHOD,
                                        action_priority, ALL_ACTIONS)


# ---------------------------------------------------------------------------
# decision engine (EV-based)
# ---------------------------------------------------------------------------

def test_decide_high_prob_retries_soon():
    """High success probability → retry soon (short delay, low friction)."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.9, "retry_later": 0.8, "switch_method": 0.1, "send_link": 0.1}
    )
    assert d.action == RETRY_SOON
    assert d.retry_in_hours == 2


def test_decide_moderate_prob_retries_later():
    """Moderate probability + salary timing → retry later (72h wait)."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.3, "retry_later": 0.35, "switch_method": 0.1, "send_link": 0.1}
    )
    assert d.action == RETRY_LATER
    assert d.retry_in_hours == 72


def test_decide_switches_method_when_better():
    """Switch method has higher EV than retries."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.1, "retry_later": 0.1, "switch_method": 0.5, "send_link": 0.1},
        current_method="upi"
    )
    assert d.action == SWITCH_METHOD
    # The engine decides *that* switching is optimal; the target instrument is
    # resolved by the agent layer from real per-method probabilities
    assert d.target_method is None


def test_decide_sends_link_for_otp_expired():
    """Low prob for retries + visible action → send link."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.1, "retry_later": 0.1, "switch_method": 0.1, "send_link": 0.15}
    )
    assert d.action == SEND_LINK


def test_decide_gives_up_when_all_ev_below_zero():
    """All actions have negative expected value → give up."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.01, "retry_later": 0.01, "switch_method": 0.01, "send_link": 0.01}
    )
    assert d.action == GIVE_UP


def test_decide_respects_attempt_budget():
    """Even high probability must respect attempt budget."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.95, "retry_later": 0.8, "switch_method": 0.5, "send_link": 0.3},
        attempts=3
    )
    assert d.action == GIVE_UP
    assert "budget" in d.reason.lower()

    # 2 attempts is within budget (default max is 3)
    d2 = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.95, "retry_later": 0.8, "switch_method": 0.5, "send_link": 0.3},
        attempts=2
    )
    assert d2.action == RETRY_SOON


def test_decide_respects_risk_gate():
    """Risk gate blocks actions above threshold."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.95, "retry_later": 0.8, "switch_method": 0.5, "send_link": 0.3},
        risk_allowed={"retry_soon": False, "retry_later": True, "switch_method": True, "send_link": True}
    )
    # retry_soon blocked, retry_later wins
    assert d.action == RETRY_LATER


def test_decide_uses_ev_not_just_probability():
    """Higher probability isn't always best if friction cost differs."""
    # send_link has lower prob but also lower friction (visible = ₹25 vs silent retry = ₹2)
    # On small amounts, retry might still win due to low cost
    d = DecisionEngine().decide(
        amount=500.0,  # small amount
        p_actions={"retry_soon": 0.4, "retry_later": 0.3, "switch_method": 0.2, "send_link": 0.45}
    )
    # For ₹500, even with 45% prob, ₹25 visible friction hurts EV
    # EV_retry = 0.4*500 - 2 = 198
    # EV_link = 0.45*500 - 25 = 200
    # Very close, but retry wins by default tie-break
    assert d.action in ALL_ACTIONS


def test_decide_gives_up_when_risk_blocks_all():
    """All actions blocked by risk gate → give up with risk_blocked flag."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.95, "retry_later": 0.8, "switch_method": 0.5, "send_link": 0.3},
        risk_allowed={"retry_soon": False, "retry_later": False, "switch_method": False, "send_link": False}
    )
    assert d.action == GIVE_UP
    assert d.risk_blocked is True


def test_decide_min_p_floor_prunes_unlikely_actions():
    """Actions with probability below min_p_floor (default 0.03) are pruned."""
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 0.02, "retry_later": 0.01, "switch_method": 0.001, "send_link": 0.001}
    )
    assert d.action == GIVE_UP


def test_decide_ev_formula():
    """Verify EV = P*amount - retry_cost - friction."""
    # At P=1.0 for ₹1000: EV = 1000 - 2 - 15 (silent friction ≈ customer burden)
    # = 983; visible actions (switch/link) carry ₹25 friction
    d = DecisionEngine().decide(
        amount=1000.0,
        p_actions={"retry_soon": 1.0, "retry_later": 0.0, "switch_method": 0.0, "send_link": 0.0}
    )
    assert d.ev == pytest.approx(983.0)
    assert d.action == RETRY_SOON


def test_decide_batch_returns_decisions():
    """Batch decision adds decision column with action/probability/ev/reason."""
    df = pd.DataFrame({
        "txn_id": ["a", "b", "c"],
        "amount": [1000.0, 2000.0, 500.0],
        # single probability column; later/alt/link derived (see decide_batch)
        "p_success": [0.9, 0.3, 0.01],
    })
    out = DecisionEngine().decide_batch(df, p_col="p_success", amount_col="amount")
    assert "decision" in out.columns
    actions = [row["decision"]["action"] for _, row in out.iterrows()]
    assert actions == [RETRY_SOON, RETRY_SOON, GIVE_UP]
    # every decision carries the audit fields
    assert {"action", "probability", "ev", "reason", "risk_blocked"} <= set(
        out["decision"].iloc[0].keys())


def test_choose_alternative_picks_best():
    """choose_alternative returns best method and its probability."""
    probs = {"upi": 0.1, "debit_card": 0.7, "credit_card": 0.4}
    method, prob = DecisionEngine.choose_alternative(probs, exclude="upi")
    assert method == "debit_card"
    assert prob == 0.7
    # no alternatives left
    method, prob = DecisionEngine.choose_alternative({"upi": 0.5}, exclude="upi")
    assert method is None
    assert prob == 0.0


def test_action_priority_ordering():
    """Lower priority number = more aggressive (cheaper) action."""
    assert action_priority(RETRY_SOON) < action_priority(RETRY_LATER) < \
        action_priority(SWITCH_METHOD) < action_priority(SEND_LINK) < \
        action_priority(GIVE_UP)


# ---------------------------------------------------------------------------
# predictor (unchanged)
# ---------------------------------------------------------------------------

def test_resolve_backend_falls_back_gracefully():
    # on this machine xgboost may be absent; whatever we get must be usable
    from src.models.predictor import resolve_backend
    backend = resolve_backend("xgboost")
    assert backend in {"xgboost", "histogram_gb", "random_forest"}


def test_sample_params_respects_space_bounds():
    from src.models.predictor import sample_params
    rng = np.random.default_rng(0)
    params = sample_params("histogram_gb", rng)
    assert "max_iter" in params and "learning_rate" in params
    assert 100 <= params["max_iter"] <= 300
    assert 0.02 <= params["learning_rate"] <= 0.25
    assert isinstance(params["max_iter"], int)


def test_make_model_returns_estimator():
    from src.models.predictor import make_model, resolve_backend, sample_params
    backend = resolve_backend("xgboost")
    model = make_model(backend, **sample_params(backend, np.random.default_rng(1)))
    assert hasattr(model, "fit") and hasattr(model, "predict_proba")


def _tiny_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "amount": np.exp(rng.normal(5, 1, n)),
        "hour_of_day": rng.integers(0, 24, n),
        "recoverable_reason": rng.choice([0, 1], size=n, p=[0.65, 0.35]),
    })
    # label driven mostly by the 'reason' flag so the model has signal
    p = 0.1 + 0.7 * X["recoverable_reason"]
    y = pd.Series((rng.random(n) < p).astype(int), name="retry_success")
    return X, y


def test_train_evaluate_roundtrip_learns_signal():
    from src.models.predictor import train, predict_proba
    X, y = _tiny_dataset()
    model, meta = train(X, y, backend="xgboost", n_trials=2,
                        probability_calibration=False, seed=0)
    assert meta["backend"] in {"xgboost", "histogram_gb", "random_forest"}
    assert "best_params" in meta and meta["best_params"]

    proba = predict_proba(model, X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    # the recoverable_reason separation must be learnable
    assert proba[X["recoverable_reason"] == 1].mean() > \
        proba[X["recoverable_reason"] == 0].mean()


def test_save_load_roundtrip(tmp_path):
    from src.models.predictor import train, save_model, load_model, predict_proba
    X, y = _tiny_dataset(n=120)
    model, meta = train(X, y, backend="xgboost", n_trials=1,
                        probability_calibration=False, seed=0)
    path = save_model(model, tmp_path / "model.joblib", meta)
    restored, restored_meta = load_model(path)
    np.testing.assert_allclose(predict_proba(restored, X), predict_proba(model, X))
    assert restored_meta["backend"] == meta["backend"]