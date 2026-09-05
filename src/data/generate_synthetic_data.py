"""Deterministic synthetic generation of failed-payment transactions.

Models the real-world distributions that matter to payment recovery:
  * instrument mix (UPI-heavy, cards, netbanking residual)
  * failure-reason mix (11 codes with different *transient recoverability*)
  * salary-cycle effects (insufficient_funds resolves after the 1st–5th)
  * peak-hour UPI/bank load ("server busy" clusters)
  * merchant verticals and banks
  * a drift switch (the review-mandated distribution shift: one bank's timeout
    recovery rate improves mid-period, from `config.data.drift.drift_date`)

Every failed transaction carries a **latent per-action ground truth**:
``p_success_retry_soon`` / ``p_success_retry_later`` / ``p_success_switch_method`` /
``p_success_send_link`` — the probability that each candidate action would recover
the payment. These oracle columns are used ONLY by the evaluator (counterfactual
honesty); the agent observes only the outcome of the action it actually executed.

The generator also simulates **historical policy bias**: each row gets a
``retry_propensity`` (probability that the *past* policy would have retried it,
high for frequently-retried reasons like ``upi_server_busy``, near-zero for
``card_expired``) and an exposure flag ``was_retried_historically``. The observed
label ``retry_success_obs`` is present only for exposed rows (NaN otherwise) —
non-retried payments are censored, NEVER labelled "failure".

Deterministic: every random source is seeded from config (``app.random_seed``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Fixed categorical worlds (kept in code for readability; shares merge with config)
BANKS = ["bank_a", "bank_b", "bank_c", "bank_d", "bank_e",
         "bank_f", "bank_g", "bank_h", "bank_i", "bank_j"]
VERTICALS = ["food_delivery", "ecommerce", "education_fees",
             "utility_bills", "travel", "subscription_saas"]
METHODS = ["upi", "debit_card", "credit_card", "netbanking"]

ACTIONS = ["retry_soon", "retry_later", "switch_method", "send_link"]

# Per-action recoverability shifts by reason: how well does THIS action recover
# THIS failure (relative to the reason's base addressability).  Learnable-but-
# not-trivial: transient reasons recover via retry_soon, funding issues via
# retry_later (salary-boosted), instrument-level problems via switch_method,
# customer-side issues best left to send_link.  card_expired / account_blocked
# are effectively dead for every same-instrument action.
ACTION_SHIFTS_DEFAULT = {
    "retry_soon": {
        "upi_server_busy": 1.00, "bank_server_timeout": 1.00, "network_timeout": 1.00,
        "otp_expired": 0.70, "issuer_decline": 0.75, "duplicate_payment_block": 0.50,
        "insufficient_funds": 0.45, "card_expired": 0.05, "account_blocked": 0.03,
    },
    "retry_later": {
        "upi_server_busy": 0.50, "bank_server_timeout": 0.65, "network_timeout": 0.40,
        "otp_expired": 0.75, "issuer_decline": 0.70, "duplicate_payment_block": 0.90,
        "insufficient_funds": 1.15, "card_expired": 0.04, "account_blocked": 0.03,
    },
    "switch_method": {
        "upi_server_busy": 0.80, "bank_server_timeout": 0.90, "network_timeout": 0.60,
        "otp_expired": 0.30, "issuer_decline": 0.65, "duplicate_payment_block": 0.30,
        "insufficient_funds": 0.35, "card_expired": 0.55, "account_blocked": 0.20,
    },
    "send_link": {
        "upi_server_busy": 0.50, "bank_server_timeout": 0.45, "network_timeout": 0.45,
        "otp_expired": 0.55, "issuer_decline": 0.40, "duplicate_payment_block": 0.40,
        "insufficient_funds": 0.70, "card_expired": 0.30, "account_blocked": 0.10,
    },
}

# Salary-cycle boost per action (insufficient_funds, days 1–5 of the month):
# retry_later is where the money arrives; retry_soon barely benefits.
SALARY_ACTION_FACTORS = {"retry_soon": 1.05, "retry_later": 1.40,
                         "switch_method": 1.00, "send_link": 1.10}

# Historical retry propensity by reason — the *bias* the past policy baked in.
RETRY_PROPENSITY_DEFAULT = {
    "upi_server_busy": 0.85, "bank_server_timeout": 0.80, "network_timeout": 0.75,
    "otp_expired": 0.60, "insufficient_funds": 0.50, "issuer_decline": 0.45,
    "duplicate_payment_block": 0.30, "card_expired": 0.05, "account_blocked": 0.03,
}


class Config:
    """Tiny attribute-style wrapper so the generator reads from YAML-ish dicts."""

    def __init__(self, mapping: dict):
        for key, value in mapping.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __getattr__(self, item):
        raise AttributeError(f"missing config key: {item}")


def _to_config(cfg) -> Config:
    if isinstance(cfg, Config):
        return cfg
    return Config(cfg)


def _coerce_dict(value, default: dict) -> dict:
    """Accept None, a plain dict, or an attribute-style Config -> plain dict.

    The generator is called from two places: ``run_pipeline`` (plain yaml
    dicts) and its own CLI ``main()`` (Config wrappers). Coercing here keeps
    both entry points honest about the same values.
    """
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    out = {}
    for key in vars(value):
        v = getattr(value, key)
        if isinstance(v, Config):
            v = _coerce_dict(v, {})
        out[key] = v
    return out


# ---------------------------------------------------------------------------
# modular penalty stack (reused per action)
# ---------------------------------------------------------------------------

def _hour_penalty(hour: np.ndarray, method: np.ndarray) -> np.ndarray:
    """Peak-load penalty: bank/UPI "server busy" worse in busy windows."""
    busy = (((hour >= 10) & (hour <= 13)) | ((hour >= 17) & (hour <= 21))).astype(float)
    method_pen = np.where(method == "upi", 1.0, 0.7)
    penalty = 1.0 - 0.35 * busy * method_pen
    return np.clip(penalty, 0.35, 1.0)


def _salary_day_factor(day_of_month: np.ndarray,
                       reason: np.ndarray,
                       action_factor: float) -> np.ndarray:
    """Salary-window multiplier for funding-caused failures (1st–5th)."""
    salary_window = (day_of_month >= 1) & (day_of_month <= 5)
    is_funding = reason == "insufficient_funds"
    factor = np.where(is_funding & salary_window, action_factor, 1.0)
    return factor


def _vertical_penalty(vertical: np.ndarray) -> np.ndarray:
    factor = np.where(vertical == "education_fees", 0.9, 1.0)
    return factor


def _amount_penalty(log_amount: np.ndarray) -> np.ndarray:
    penalty = 1.0 - 0.12 * np.clip((log_amount - 8.5) / 3.0, 0.0, 1.0)
    return np.clip(penalty, 0.55, 1.0)


def _method_penalty(method: np.ndarray) -> np.ndarray:
    penalty = np.where(method == "netbanking", 0.8, 1.0)
    return penalty


def generate_failed_payments(
    n_transactions: int = 20000,
    failure_share: float = 0.20,
    start: str = "2025-01-01",
    end: str = "2025-06-30",
    n_merchants: int = 120,
    n_customers: int = 4000,
    method_mix: Optional[dict] = None,
    reason_mix: Optional[dict] = None,
    recovery_seed_probs: Optional[dict] = None,
    vertical_mix: Optional[dict] = None,
    action_shifts: Optional[dict] = None,
    retry_propensity: Optional[dict] = None,
    drift_date: Optional[str] = "2025-05-01",
    drift_bank_timeout_lift: float = 1.2,
    risk_config: Optional[dict] = None,
    latent_noise: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the failed-payment table with per-action latent truth + censored obs."""
    rng = np.random.default_rng(seed)

    # ----------------------------- defaults (mirror configs/*.yaml) ---------
    method_mix = _coerce_dict(method_mix, {
        "upi": 0.52, "debit_card": 0.22, "credit_card": 0.16, "netbanking": 0.10})
    reason_mix = _coerce_dict(reason_mix, {
        "insufficient_funds": 0.22, "bank_server_timeout": 0.18,
        "issuer_decline": 0.16, "upi_server_busy": 0.14, "network_timeout": 0.10,
        "otp_expired": 0.08, "card_expired": 0.05, "account_blocked": 0.04,
        "duplicate_payment_block": 0.03})
    recovery_seed_probs = _coerce_dict(recovery_seed_probs, {
        "insufficient_funds": 0.55, "bank_server_timeout": 0.75,
        "issuer_decline": 0.30, "upi_server_busy": 0.80, "network_timeout": 0.70,
        "otp_expired": 0.60, "card_expired": 0.10, "account_blocked": 0.05,
        "duplicate_payment_block": 0.20})
    vertical_mix = _coerce_dict(vertical_mix, {
        "food_delivery": 0.28, "ecommerce": 0.22, "education_fees": 0.16,
        "utility_bills": 0.14, "travel": 0.10, "subscription_saas": 0.10})
    action_shifts = _coerce_dict(action_shifts, ACTION_SHIFTS_DEFAULT)
    retry_propensity = _coerce_dict(retry_propensity, RETRY_PROPENSITY_DEFAULT)
    risk_config = _coerce_dict(risk_config, {
        "base": 0.15, "high_risk_banks": ["bank_d", "bank_j"],
        "high_risk_share": 0.03, "high_risk_lift": 0.25,
        "bank_risk_lift": 0.0, "amount_lift": 0.10, "noise": 0.06})

    for name, mix in [("method", method_mix), ("reason", reason_mix),
                      ("vertical", vertical_mix)]:
        total = sum(mix.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mix '{name}' sums to {total:.3f}, expected 1.0")

    # ----------------------------- base row columns -------------------------
    all_ts = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="min")
    if len(all_ts) == 0:
        raise ValueError(f"empty time range {start} -> {end}")

    ts_idx = rng.integers(0, len(all_ts), size=n_transactions)
    timestamps = all_ts[ts_idx]
    hours = timestamps.hour.to_numpy()
    days_of_month = timestamps.day.to_numpy()

    methods = rng.choice(list(method_mix.keys()), size=n_transactions,
                         p=list(method_mix.values()))
    reasons = rng.choice(list(reason_mix.keys()), size=n_transactions,
                         p=list(reason_mix.values()))
    banks = rng.choice(BANKS, size=n_transactions)
    verticals = rng.choice(list(vertical_mix.keys()), size=n_transactions,
                           p=list(vertical_mix.values()))
    merchant_ids = rng.integers(1, n_merchants + 1, size=n_transactions)
    customer_ids = rng.integers(1, n_customers + 1, size=n_transactions)
    customer_instruments = rng.integers(1, 5, size=n_transactions)

    amounts = np.exp(rng.normal(loc=6.5, scale=1.3, size=n_transactions))
    amounts = np.clip(amounts, 50.0, 500_000.0).round(2)
    log_amt = np.log1p(amounts)

    df = pd.DataFrame({
        "txn_id": [f"pay{100000 + i}" for i in range(n_transactions)],
        "customer_id": [f"cus{int(c)}" for c in customer_ids],
        "merchant_id": [f"mer{int(m)}" for m in merchant_ids],
        "timestamp": timestamps,
        "amount": amounts,
        "payment_method": methods,
        "bank": banks,
        "merchant_vertical": verticals,
        "failure_reason_code": reasons,
        "customer_instruments": customer_instruments,
    })

    # ----------------------------- per-action latent truth (oracle) ----------
    base = np.array([recovery_seed_probs[r] for r in reasons])
    is_upi = (methods == "upi").astype(int)

    # drift: one bank's timeout recovery improves after drift_date
    if drift_date is not None:
        drifted = (timestamps >= pd.Timestamp(drift_date)) & \
                  (reasons == "bank_server_timeout") & \
                  (banks == "bank_c")  # arbitrary but fixed: bank_c's outages improve
        base = np.where(drifted, base * drift_bank_timeout_lift, base)

    shared = (_hour_penalty(hours, is_upi)
              * _vertical_penalty(verticals)
              * _amount_penalty(log_amt)
              * _method_penalty(methods))

    p_actions: dict[str, np.ndarray] = {}
    for action in ACTIONS:
        shift = np.array([action_shifts.get(action, {}).get(r, 0.5) for r in reasons])
        salary = _salary_day_factor(days_of_month, reasons,
                                    SALARY_ACTION_FACTORS.get(action, 1.0))
        if action in ("retry_soon", "retry_later"):
            p = base * shift * shared * salary
        else:
            # customer-side actions are less time/load sensitive
            p = base * shift * salary * _amount_penalty(log_amt) * _vertical_penalty(verticals)
        if action == "switch_method":
            p = p * np.where(customer_instruments > 1, 1.0, 0.05)  # need an alt instrument
        # one noise draw per (txn, action) — the "true" probability the evaluator uses
        p_actions[action] = np.clip(p + rng.normal(0.0, latent_noise, n_transactions),
                                    0.0, 1.0)

    # ----------------------------- historical policy bias --------------------
    prop_base = np.array([retry_propensity.get(r, 0.3) for r in reasons])
    propensity = np.clip(prop_base + rng.normal(0.0, 0.05, n_transactions), 0.0, 1.0)
    was_retried = (rng.random(n_transactions) < propensity).astype(int)

    # historical policy retried "dumb": roughly a retry_soon window (~4h)
    obs_success = (rng.random(n_transactions) < p_actions["retry_soon"]).astype(int)
    retry_success_obs = np.where(was_retried == 1, obs_success, np.nan)

    # ----------------------------- synthetic risk score ----------------------
    # A small fraud-flagged segment (high_risk_share) carries the strong lift —
    # that's the segment whose recoveries the mock gate must demonstrably
    # block, otherwise "compliant escalation" never fires. The segment flag
    # comes from a dedicated RNG so the MAIN simulation stream (and therefore
    # every downstream number) is untouched by adding the risk system.
    segment_rng = np.random.default_rng(seed + 0x7A11_7E9)  # fixed, independent
    flagged = segment_rng.random(n_transactions) < risk_config.get(
        "high_risk_share", 0.03)
    risk_score = (risk_config["base"]
                  + risk_config["amount_lift"] * (log_amt - 6.0).clip(0, 3)
                  + flagged.astype(float)
                  * risk_config.get("high_risk_lift", 0.25)
                  + np.isin(banks, risk_config["high_risk_banks"]).astype(float)
                  * risk_config.get("bank_risk_lift", 0.0)
                  + rng.normal(0.0, risk_config["noise"], n_transactions))
    risk_score = np.clip(risk_score, 0.0, 1.0)

    df["retry_propensity"] = propensity.round(4)
    df["was_retried_historically"] = was_retried
    df["retry_success_obs"] = retry_success_obs
    df["risk_score"] = risk_score.round(4)
    for action in ACTIONS:
        df[f"p_success_{action}"] = p_actions[action].round(4)

    # ----------------------------- failure-sized sample ----------------------
    n_fail_target = int(round(n_transactions * failure_share))
    failed = _select_failures(df, n_fail_target, rng)

    failed = failed.assign(
        is_weekend=(failed.timestamp.dt.dayofweek >= 5).astype(int),
        hour_of_day=failed.timestamp.dt.hour,
        day_of_week=failed.timestamp.dt.dayofweek,
        day_of_month=failed.timestamp.dt.day,
        amount_log=np.log1p(failed.amount),
    )
    failed["txn_value_bucket"] = pd.cut(
        failed.amount, bins=[0, 500, 5000, 50000, np.inf],
        labels=["small", "medium", "large", "huge"])
    failed = failed.reset_index(drop=True)

    log.info("generated %d transactions; kept %d failed rows (share=%.2f)",
             n_transactions, len(failed), len(failed) / n_transactions)
    return failed


def _select_failures(df: pd.DataFrame, n_fail_target: int,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Keep exactly ``n_fail_target`` rows as the failed-payment feed."""
    if len(df) <= n_fail_target:
        return df
    keep = rng.choice(df.index, size=n_fail_target, replace=False)
    return df.loc[keep]


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry: write generated CSV to data/synthetic/failed_payments.csv."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Generate synthetic failed-payment data.")
    parser.add_argument("--out", default="data/synthetic/failed_payments.csv",
                        help="output CSV path")
    parser.add_argument("--config", default="configs/config.yaml", help="main config yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml",
                        help="data generator yaml")
    args = parser.parse_args(argv)

    cfg = Config(yaml.safe_load(Path(args.config).read_text()))
    dcfg = Config(yaml.safe_load(Path(args.data_config).read_text()))

    g = dcfg.generator
    data_cfg = cfg.data
    drift = getattr(data_cfg, "drift", None)
    df = generate_failed_payments(
        n_transactions=data_cfg.n_transactions,
        failure_share=data_cfg.failure_share,
        start=data_cfg.start_date,
        end=data_cfg.end_date,
        n_merchants=g.n_merchants,
        n_customers=g.n_customers,
        method_mix=g.payment_methods,
        reason_mix=g.failure_reasons,
        recovery_seed_probs=g.recovery_seed_probs,
        vertical_mix=g.merchants.verticals,
        action_shifts=(g.action_shifts if hasattr(g, "action_shifts") else None),
        retry_propensity=(g.retry_propensity if hasattr(g, "retry_propensity") else None),
        drift_date=(drift.drift_date if drift else "2025-05-01"),
        drift_bank_timeout_lift=(drift.bank_timeout_recovery_lift if drift else 1.2),
        risk_config=(g.risk if hasattr(g, "risk") else None),
        latent_noise=(g.latent_noise if hasattr(g, "latent_noise") else 0.12),
        seed=cfg.app.random_seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} failed-payment rows -> {out}")
    print(df["failure_reason_code"].value_counts(normalize=True).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())