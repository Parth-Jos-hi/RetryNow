#!/usr/bin/env python3
"""Deep live verification of every AI feature against the real dataset.

Run: PYTHONIOENCODING=utf-8 python -u scripts/verify_ai.py

Checks, in order:
  1. ML predictor quality on held-out test (AUC, Brier, ECE calibration)
  2. Multi-action estimator behavior (ranges, salary boost, switch dead-end,
     model-vs-latent correlation)
  3. EV engine math (independent recompute of chosen-action EV on real rows)
  4. Risk gate coverage (does the mock ever actually block in this data?)
  5. Drift experiment on the real dataset
  6. Off-policy IPW summary consistency
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import temporal_train_test_split
from src.features.feature_engineering import build_feature_matrix
from src.models.predictor import train as train_model_fn
from src.models.recoverability import RecoverabilityEstimator
from src.risk.risk_gate import MockRiskGate
from src.models.decision_engine import DecisionEngine


def build_priors(dcfg):
    seeds = dcfg.get("generator", {}).get("recovery_seed_probs", {})
    shifts = dcfg.get("generator", {}).get("action_shifts", {})
    return {r: {a: float(np.clip(seed * shift.get(r, 0.3), 0, 1))
                for a, shift in shifts.items()} for r, seed in seeds.items()}


def main():
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "config.yaml", encoding="utf-8"))
    dcfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "data_config.yaml", encoding="utf-8"))
    df = pd.read_csv(PROJECT_ROOT / "outputs" / "generated_data.csv", parse_dates=["timestamp"])

    # ---------- same train path as the pipeline ----------
    split_date = cfg["data"]["drift"]["drift_date"]
    X_tr, X_te, y_tr, y_te, tr_df, te_df = temporal_train_test_split(
        df, split_date=split_date, label="retry_success_obs")
    X_tr_m = build_feature_matrix(tr_df)
    X_te_m = build_feature_matrix(te_df)
    exposed = y_tr.notna().to_numpy()
    X_exposed, y_exposed = X_tr_m[exposed], y_tr[exposed]

    model, meta = train_model_fn(
        X_exposed, y_exposed, backend="auto",
        n_trials=int(cfg["model"]["n_trials"]),
        probability_calibration=cfg["model"]["probability_calibration"],
        seed=42)
    print(f"\n[ML CHECK] backend={meta['backend']} calibrated={meta.get('calibrated')}")

    # ---------- 1. AUC / Brier / ECE on held-out test (exposed subset) ----------
    test_exposed = y_te.notna().to_numpy()
    if test_exposed.sum() > 10:
        from sklearn.metrics import brier_score_loss, roc_auc_score
        proba = model.predict_proba(X_te_m[test_exposed])[:, 1]
        auc = roc_auc_score(y_te[test_exposed], proba)
        brier = brier_score_loss(y_te[test_exposed], proba)
        print(f"[ML CHECK] test AUC (exposed subset, n={test_exposed.sum()}) = {auc:.4f}")
        print(f"[ML CHECK] Brier = {brier:.4f} (naive coin ~0.25, constant-0.3 ~0.21)")
        bins = np.linspace(0, 1, 6)
        idx = np.digitize(proba, bins) - 1
        ece = 0.0
        for b in range(5):
            m = idx == b
            if m.sum() == 0:
                continue
            ece += (m.mean() * abs(proba[m].mean() - y_te[test_exposed][m].mean()))
        print(f"[ML CHECK] ECE (5 bins) = {ece:.4f} (lower = more trustworthy probabilities)")

    # ---------- 2. estimator on real test rows per reason ----------
    est = RecoverabilityEstimator(
        predictor=model, prior_alpha=cfg["recoverability"]["table_prior_alpha"],
        priors=build_priors(dcfg),
        salary_boost_later=cfg["recoverability"]["salary_boost_later"],
        switch_no_alt=cfg["recoverability"]["switch_no_alt_multiplier"])
    est.fit_tables(tr_df)
    X_new = [c for c in X_te_m.columns if c not in te_df.columns]
    te2 = pd.concat([te_df.reset_index(drop=True),
                     X_te_m[X_new].reset_index(drop=True)], axis=1)
    rows = te2.sample(250, random_state=7)
    p_df = pd.DataFrame(list(rows.apply(lambda r: est.estimate(r.to_dict()), axis=1)))
    print("\n[ESTIMATOR] p ranges: " + ", ".join(
        f"{a}: [{p_df[a].min():.3f},{p_df[a].max():.3f}]" for a in p_df.columns))

    sf = rows[(rows.failure_reason_code == "insufficient_funds") & (rows.day_of_month <= 5)]
    sp = rows[(rows.failure_reason_code == "insufficient_funds") & (rows.day_of_month >= 20)]
    if len(sf) and len(sp):
        est_sf = sf.apply(lambda r: est.estimate(r.to_dict())["retry_later"], axis=1).mean()
        est_sp = sp.apply(lambda r: est.estimate(r.to_dict())["retry_later"], axis=1).mean()
        print(f"[ESTIMATOR] salary boost retry_later: day1-5 {est_sf:.3f} vs day20+ {est_sp:.3f}"
              f" (latent: {sf.p_success_retry_later.mean():.3f} vs {sp.p_success_retry_later.mean():.3f})")

    one_inst = rows[rows.customer_instruments <= 1]
    if len(one_inst):
        sw = one_inst.apply(lambda r: est.estimate(r.to_dict())["switch_method"], axis=1).mean()
        print(f"[ESTIMATOR] switch_method w/ 1 instrument → mean {sw:.3f} (prior·0.05 ≈ {0.05 * 0.3:.3f}-class)")

    mp = rows.apply(lambda r: est.estimate(r.to_dict())["retry_soon"], axis=1)
    lat = rows["p_success_retry_soon"]
    print(f"[ESTIMATOR] model-vs-latent correlation (retry_soon) = {mp.corr(lat):.3f}")

    # ---------- 3. EV engine math ----------
    de = DecisionEngine(settings={
        "retry_soon_hours": cfg["decision_engine"]["retry_soon_hours"],
        "retry_later_hours": cfg["decision_engine"]["retry_later_hours"],
        "retry_cost_per_attempt": cfg["decision_engine"]["retry_cost_per_attempt"],
        "friction_cost_per_touch_silent": cfg["decision_engine"]["friction_cost_per_touch_silent"],
        "friction_cost_per_touch_visible": cfg["decision_engine"]["friction_cost_per_touch_visible"],
        "min_p_floor": cfg["decision_engine"]["min_p_floor"],
        "max_attempts_per_payment": cfg["decision_engine"]["max_attempts_per_payment"],
        "max_retries_per_customer_day": cfg["decision_engine"]["max_retries_per_customer_day"],
    })
    bad = 0
    for _, r in rows.head(100).iterrows():
        pa = est.estimate(r.to_dict())
        d = de.decide(float(r["amount"]), pa, 0, r["payment_method"])
        cost = 0.0 if d.action == "give_up" else (
            cfg["decision_engine"]["retry_cost_per_attempt"] +
            (15.0 if d.action in ("retry_soon", "retry_later") else 25.0))
        ev_manual = pa.get(d.action, 0.0) * float(r["amount"]) - cost
        if abs(d.ev - ev_manual) > 0.51:
            bad += 1
    print(f"\n[EV ENGINE] independent EV recompute mismatches in 100 rows: {bad}")

    # ---------- 4. risk gate coverage ----------
    gate = MockRiskGate(block_threshold=cfg["risk_gate"]["block_threshold"],
                        per_action=cfg["risk_gate"]["per_action_thresholds"])
    risk = df["risk_score"]
    min_thr = min(cfg["risk_gate"]["per_action_thresholds"].values())
    n_block_any = int((risk >= min_thr).sum())
    n_block_silent = int((risk >= cfg["risk_gate"]["per_action_thresholds"]["retry_soon"]).sum())
    print(f"\n[RISK GATE] risk_score range [{risk.min():.3f}, {risk.max():.3f}] "
          f"mean {risk.mean():.3f} p99 {risk.quantile(0.99):.3f}")
    print(f"[RISK GATE] rows ≥ min threshold ({min_thr}): {n_block_any} "
          f"({n_block_any / len(df):.1%}) — rows ≥ retry_soon thr: {n_block_silent}")
    verdicts = [gate.check({"risk_score": s}).blocked_all for s in risk.sample(500, random_state=1)]
    print(f"[RISK GATE] all-blocked share in 500-row sample: {sum(verdicts) / 500:.1%}")

    # ---------- 5. drift ----------
    from src.experiments.drift import run_drift_experiment
    dres = run_drift_experiment(df, drift_date=split_date, seed=42)
    print(f"\n[DRIFT] pre {dres['pre_drift']['mean_p_success']:.1%} → post "
          f"{dres['post_drift']['mean_p_success']:.1%} (lift {dres['lift']:.1%} on "
          f"n={dres['pre_drift']['n_transactions']}/{dres['post_drift']['n_transactions']})")

    # ---------- 6. off-policy ----------
    from src.evaluation.off_policy import selective_label_summary
    hist = df.copy()
    hist["outcome"] = hist["retry_success_obs"].map(
        {1.0: "SUCCESS", 0.0: "FAILED", 1: "SUCCESS", 0: "FAILED"})
    s = selective_label_summary(hist)
    print(f"\n[OFF-POLICY] observed {s['observed_recovered_value']:,.0f} / "
          f"IPW {s['ipw_estimated_recovered_value']:,.0f} / "
          f"oracle {s['oracle_expected_recovered_value']:,.0f} / "
          f"censored {s['n_censored']}")

    print("\n✓ Deep verification complete.")


if __name__ == "__main__":
    main()