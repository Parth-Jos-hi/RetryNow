#!/usr/bin/env python3
"""End-to-end pipeline: generate → train → policies → agent → evaluate → report.

Usage:
    python scripts/run_pipeline.py --mode all
    python scripts/run_pipeline.py --mode generate
    python scripts/run_pipeline.py --mode train
    python scripts/run_pipeline.py --mode evaluate

The pipeline runs four policies (do_nothing, dumb_retry, rule_retry,
RecoveryAgent) on the *same* test transactions and produces a markdown report
comparing ₹-centric metrics: recovered value, recovery rate, net recovered,
incremental vs baselines, touchpoints, risk-blocked, give-up rate.

All randomness is seeded for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# --- Paths (relative to project root) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AUDIT_DIR = OUTPUT_DIR / "audit"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.md"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_data_config() -> dict:
    with open(DATA_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def generate_data(cfg: dict, data_cfg: dict, seed: int = 42) -> pd.DataFrame:
    """Step 1: Generate synthetic failed payments with latent counterfactuals."""
    from src.data.generate_synthetic_data import generate_failed_payments

    print("\n[1/5] Generating synthetic data...")
    df = generate_failed_payments(
        n_transactions=cfg["data"]["n_transactions"],
        seed=seed,
        action_shifts=data_cfg.get("generator", {}).get("action_shifts"),
        retry_propensity=data_cfg.get("generator", {}).get("retry_propensity"),
        drift_date=cfg["data"]["drift"]["drift_date"],
        drift_bank_timeout_lift=cfg["data"]["drift"]["bank_timeout_recovery_lift"],
        risk_config=data_cfg.get("generator", {}).get("risk"),
        latent_noise=data_cfg.get("generator", {}).get("latent_noise", 0.12),
    )
    print(f"  → {len(df)} failed payments generated")
    print(f"  → Censored rows: {df['retry_success_obs'].isna().sum()}")
    return df


def build_priors(data_cfg: dict) -> dict:
    """Domain-knowledge priors for the weakly-observed actions.

    ``action_shifts`` (documented recovery-playbook assumptions per reason) ×
    the base seed probabilities give absolute prior P(success) per
    (reason, action). These are ACKNOWLEDGED priors — they shrink toward the
    observed record as data accumulates (empirical-Bayes smoothing) and are
    never derived from the oracle columns. This is the "documented assumption"
    the design doc reserves for actions with thin historical supervision.
    """
    seeds = data_cfg.get("generator", {}).get("recovery_seed_probs", {})
    shifts = data_cfg.get("generator", {}).get("action_shifts", {})
    priors: dict[str, dict[str, float]] = {}
    for reason, seed in seeds.items():
        priors[reason] = {
            action: float(np.clip(seed * shift.get(reason, 0.3), 0.0, 1.0))
            for action, shift in shifts.items()
        }
    return priors


def train_model(df: pd.DataFrame, cfg: dict, data_cfg: dict, seed: int = 42):
    """Step 2: Train recoverability estimator (ML + per-reason tables)."""
    from src.data.preprocess import temporal_train_test_split
    from src.features.feature_engineering import build_feature_matrix
    from src.models.predictor import train as train_sklearn_model
    from src.models.recoverability import RecoverabilityEstimator

    print("\n[2/5] Training recoverability estimator...")

    # Temporal split (leak-free: train strictly before drift date)
    split_date = cfg["data"]["drift"]["drift_date"]
    X_train, X_test, y_train, y_test, train_df, test_df = temporal_train_test_split(
        df, split_date=split_date, label="retry_success_obs"
    )
    print(f"  → Train: {len(train_df)} | Test: {len(test_df)}")

    # Build features (observed-only history)
    X_train_matrix = build_feature_matrix(train_df)
    X_test_matrix = build_feature_matrix(test_df)

    # Censored-safe training (config.model.train_on_exposed): ML labels are the
    # observed outcomes only — NaN-label rows are UNKNOWN, never 0/failure
    exposed = y_train.notna().to_numpy()
    X_exposed, y_exposed = X_train_matrix[exposed], y_train[exposed]
    print(f"  → Exposed (labeled) train rows: {len(X_exposed)}/{len(X_train_matrix)}"
          f" — the rest are censored, excluded from ML supervision")

    # Train ML model for retry_soon (the only action with historical supervision)
    model_cfg = cfg.get("model", {})
    predictor, _model_meta = train_sklearn_model(
        X_exposed, y_exposed,
        backend=model_cfg.get("backend", "auto"),
        n_trials=int(model_cfg.get("n_trials", 30)),
        probability_calibration=model_cfg.get("probability_calibration", True),
        seed=seed,
    )

    # Wrap in recoverability estimator (acknowledged priors → EB-smoothed tables)
    estimator = RecoverabilityEstimator(
        predictor=predictor,
        prior_alpha=cfg["recoverability"]["table_prior_alpha"],
        priors=build_priors(data_cfg),
        salary_boost_later=cfg["recoverability"]["salary_boost_later"],
        switch_no_alt=cfg["recoverability"]["switch_no_alt_multiplier"],
    )
    estimator.fit_tables(train_df)

    # Attach engineered features to the test rows so the ML path sees real
    # columns at decision time (leak-free: built from observed history only);
    # keep only matrix columns the raw rows do not already carry (amount & co.)
    new_cols = [c for c in X_test_matrix.columns if c not in test_df.columns]
    test_df = pd.concat(
        [test_df.reset_index(drop=True),
         X_test_matrix[new_cols].reset_index(drop=True)],
        axis=1)

    print(f"  → Model trained on {X_train_matrix.shape[1]} features")

    # Save estimator for the upload API (so merchants can process their own data)
    import joblib
    estimator_path = OUTPUT_DIR / "estimator.joblib"
    joblib.dump(estimator, estimator_path)
    print(f"  → Estimator saved to {estimator_path}")

    return estimator, test_df


def run_policies(test_df: pd.DataFrame, cfg: dict, seed: int = 42,
                 estimator=None, data_cfg=None) -> dict[str, pd.DataFrame]:
    """Step 3: Run all four policies on the *same* test transactions."""
    from src.policy.baselines import do_nothing, dumb_retry, rule_retry
    from src.agent.recovery_agent import RecoveryAgent
    from src.models.decision_engine import DecisionEngine
    from src.models.recoverability import RecoverabilityEstimator
    from src.risk.risk_gate import MockRiskGate
    from src.simulator.payment_simulator import PaymentSimulator

    print("\n[3/5] Running policies on same test transactions...")

    # Same oracle for every policy, separate simulator instance each time
    oracle_df = test_df[["txn_id", "amount",
                         "p_success_retry_soon", "p_success_retry_later",
                         "p_success_switch_method", "p_success_send_link"]].copy()

    results = {}

    # 1. do_nothing
    print("  → do_nothing...")
    results["do_nothing"] = do_nothing(
        test_df.copy(), PaymentSimulator(oracle_df, seed=seed))

    # 2. dumb_retry — same transactions, same simulator semantics
    print("  → dumb_retry...")
    results["dumb_retry"] = dumb_retry(
        test_df.copy(), PaymentSimulator(oracle_df, seed=seed),
        retry_hours=cfg["simulation"]["baseline_retry_hours"])

    # 3. rule_retry
    print("  → rule_retry...")
    results["rule_retry"] = rule_retry(
        test_df.copy(), PaymentSimulator(oracle_df, seed=seed))

    # 4. RecoveryAgent (AI) — estimator + EV engine + risk gate + audit trail
    print("  → RecoveryAgent (AI)...")
    recoverability = cfg["recoverability"]
    de = cfg["decision_engine"]
    # Prefer the leak-free trained estimator from step 2 (ML retry_soon + tables
    # fit on the PRE-drift train split); fall back to a tables-only estimator.
    if estimator is None:
        estimator = RecoverabilityEstimator(
            predictor=None,
            prior_alpha=recoverability["table_prior_alpha"],
            priors=build_priors(data_cfg) if data_cfg else None,
            salary_boost_later=recoverability["salary_boost_later"],
            switch_no_alt=recoverability["switch_no_alt_multiplier"],
        )
        estimator.fit_tables(test_df)

    engine = DecisionEngine(settings={
        "retry_soon_hours": de["retry_soon_hours"],
        "retry_later_hours": de["retry_later_hours"],
        "retry_cost_per_attempt": de["retry_cost_per_attempt"],
        "friction_cost_per_touch_silent": de["friction_cost_per_touch_silent"],
        "friction_cost_per_touch_visible": de["friction_cost_per_touch_visible"],
        "min_p_floor": de["min_p_floor"],
        "max_attempts_per_payment": de["max_attempts_per_payment"],
        "max_retries_per_customer_day": de["max_retries_per_customer_day"],
    })
    risk_gate = MockRiskGate(
        block_threshold=cfg["risk_gate"]["block_threshold"],
        per_action=cfg["risk_gate"]["per_action_thresholds"],
    )

    audit_path = AUDIT_DIR / "agent_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    agent = RecoveryAgent(
        estimator=estimator,
        engine=engine,
        risk_gate=risk_gate,
        audit_path=str(audit_path),
        seed=seed,
    )
    results["recovery_agent"] = agent.run(test_df.copy(),
                                          PaymentSimulator(oracle_df, seed=seed))

    print(f"  → 4 policies evaluated on {len(test_df)} transactions")
    return results


def historical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Adapter: the historical record speaks retry_success_obs (1/0/NaN) —
    map it to the policy-run outcome vocabulary (SUCCESS/FAILED/NaN) for the
    off-policy summary. Censored rows stay NaN = unknown, never 'failed'."""
    out = df.copy()
    # .map keeps NaN as float-NA (np.where would fail dtype promotion: str vs NaN)
    out["outcome"] = out["retry_success_obs"].map(
        {1.0: "SUCCESS", 0.0: "FAILED", 1: "SUCCESS", 0: "FAILED"})
    return out


def evaluate(results: dict[str, pd.DataFrame], cfg: dict, historical_df=None):
    """Step 4: Compute ₹-centric metrics + off-policy estimates.

    ``historical_df`` carries the censored historical record (retry_propensity,
    retry_success_obs NaN for non-retried rows). The off-policy section belongs
    THERE — the policy-run frames' outcomes are fresh simulator samples, so the
    historical propensity/censoring story does not apply to them.
    """
    from src.evaluation.metrics import compare_policies, breakdowns
    from src.evaluation.off_policy import selective_label_summary

    print("\n[4/5] Evaluating policies...")

    comparison = compare_policies(
        results,
        retry_cost_per_attempt=cfg["decision_engine"]["retry_cost_per_attempt"],
    )

    # Off-policy summary describes the HISTORICAL data (selection bias of the
    # record we learned from), not a policy run
    off_policy_summary = {}
    if historical_df is not None:
        off_policy_summary["historical_record"] = selective_label_summary(
            historical_df)

    breakdowns_by_policy = {name: breakdowns(df) for name, df in results.items()}

    print(f"  → Comparison table: {len(comparison)} policies × {len(comparison.columns)} metrics")
    return comparison, breakdowns_by_policy, off_policy_summary


def write_report(comparison: pd.DataFrame,
                 breakdowns_by_policy: dict,
                 off_policy_summary: dict,
                 drift_result: Optional[dict] = None,
                 output_path: Path = REPORT_PATH):
    """Step 5: Write markdown evaluation report."""
    from src.evaluation.metrics import metrics_to_markdown

    print(f"\n[5/5] Writing report to {output_path}...")

    lines = [
        "# Evaluation Report: RetryNow Revenue Recovery Agent\n",
        f"**Generated**: {pd.Timestamp.now().isoformat()}\n",
        "## Policy Comparison (₹-centric metrics)\n",
        comparison.to_markdown(floatfmt=".2f"), "\n",
        "### Key Findings\n",
    ]

    if "recovery_agent" in comparison.index:
        ai_row = comparison.loc["recovery_agent"]
        if "incremental_vs_dumb" in comparison.columns:
            lines.append(
                f"- **AI vs Dumb Retry**: +₹{ai_row['incremental_vs_dumb']:.0f} "
                f"incremental net recovered value\n")
        if "incremental_vs_rule" in comparison.columns:
            lines.append(
                f"- **AI vs Rule Retry**: +₹{ai_row['incremental_vs_rule']:.0f} "
                f"incremental net recovered value\n")
        lines.append(
            f"- **Recovery rate**: {ai_row['recovery_rate_value']:.1%} of failed value recovered\n")
        lines.append(
            f"- **Customer touchpoints**: {int(ai_row['touchpoints'])} "
            f"(recovery-to-bother: ₹{ai_row['recovery_to_bother']:.0f}/touch)\n")
        if int(ai_row.get("risk_blocked", 0)) > 0:
            # the compliance trade-off is a feature, not a footnote: the gate
            # suppressed real recoverable value on purpose
            lines.append(
                f"- **Risk gate**: {int(ai_row['risk_blocked'])} high-risk payments "
                f"suppressed (₹{ai_row.get('risk_suppressed_value', 0):,.0f} in "
                f"failed value deliberately NOT pursued — compliance before revenue)\n")

    if off_policy_summary:
        lines.append("\n## Off-Policy Evaluation — Selection Bias of the Historical Record\n")
        lines.append("This section is about the DATA WE LEARNED FROM, not a policy run:\n")
        lines.append("- **Observed**: outcomes of historically executed retries only (naive, biased)\n")
        lines.append("- **IPW**: inverse-propensity weighted estimate (clipped at 10)\n")
        lines.append("- **Oracle**: simulator's latent truth (counterfactual, evaluator-only)\n")
        lines.append("- **Censored**: non-retried rows are UNKNOWN outcomes, never 'failures'\n\n")

        for pol, summary in off_policy_summary.items():
            lines.append(f"### {pol}\n")
            lines.append("| Metric | Value |\n|--------|-------|\n")
            lines.append(f"| Observed recovered value | ₹{summary['observed_recovered_value']:,.0f} |\n")
            lines.append(f"| IPW estimated recovered value | ₹{summary['ipw_estimated_recovered_value']:,.0f} |\n")
            lines.append(f"| Oracle expected recovered value | ₹{summary['oracle_expected_recovered_value']:,.0f} |\n")
            lines.append(f"| Headroom vs oracle | ₹{summary['headroom_vs_oracle']:,.0f} |\n")
            lines.append(f"| Censored rows | {summary['n_censored']:,} (out of {summary['n_failed']:,}) |\n")

    # Drift section — explains the agent's largest loss segment and the
    # pre-drift-training conservatism behind it
    if drift_result:
        from src.experiments.drift import report_drift_metrics
        lines.append("\n## Drift — Why the Agent Trails Rule on bank_server_timeout\n")
        lines.append(
            "The rule baseline's largest edge sits in the drifted segment "
            "(bank_c + bank_server_timeout after the infrastructure fix): "
            "the agent's ML was trained strictly on PRE-drift data, so its "
            "estimates shrink toward pre-drift recoverability there, while "
            "the rule fires blind. The drift experiment quantifies the shift:\n")
        lines.append(report_drift_metrics(drift_result))

    lines.append("\n## Recovery Rate by Failure Reason\n")
    for pol in ["recovery_agent", "rule_retry", "dumb_retry"]:
        if pol in breakdowns_by_policy:
            by_reason = breakdowns_by_policy[pol].get("by_reason", {})
            if by_reason:
                lines.append(f"\n### {pol}\n")
                lines.append("| Reason | Recovery Rate |\n|--------|---------------|\n")
                for reason, rate in sorted(by_reason.items(), key=lambda x: -x[1]):
                    lines.append(f"| {reason} | {rate:.1%} |\n")

    lines.append("\n## Limitations\n")
    lines.append("- Results are on synthetic data; real-world performance may differ.\n")
    lines.append("- IPW estimates depend on propensity model accuracy (we use generator-injected values).\n")
    lines.append("- Oracle metrics are NOT available to the agent; they expose counterfactual headroom.\n")
    lines.append("- Risk gate is a mock interface; production would need real fraud/risk integration.\n")
    lines.append("- Censored outcomes (non-retried transactions) are unknown, not labeled as failures.\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RetryNow evaluation pipeline")
    parser.add_argument("--mode", choices=["all", "generate", "train", "evaluate"],
                        default="all", help="Pipeline mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = load_config()
    data_cfg = load_data_config()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "all":
        df = generate_data(cfg, data_cfg, seed=args.seed)
        estimator, test_df = train_model(df, cfg, data_cfg, seed=args.seed)
        results = run_policies(test_df, cfg, seed=args.seed, estimator=estimator,
                               data_cfg=data_cfg)
        comparison, breakdowns_by_policy, off_policy = evaluate(
            results, cfg, historical_df=historical_frame(df))
        from src.experiments.drift import run_drift_experiment
        drift_result = run_drift_experiment(
            df, drift_date=cfg["data"]["drift"]["drift_date"], seed=args.seed)
        write_report(comparison, breakdowns_by_policy, off_policy, drift_result)
        drift_csv = OUTPUT_DIR / "generated_data.csv"
        df.to_csv(drift_csv, index=False)
        print(f"\n✓ Data saved to {drift_csv}")
        print("\n✓ Pipeline complete. See outputs/evaluation_report.md + outputs/audit/agent_audit.jsonl")
    elif args.mode == "generate":
        df = generate_data(cfg, data_cfg, seed=args.seed)
        df.to_csv(OUTPUT_DIR / "generated_data.csv", index=False)
        print(f"\n✓ Data saved to {OUTPUT_DIR / 'generated_data.csv'}")
    elif args.mode == "train":
        data_path = OUTPUT_DIR / "generated_data.csv"
        if data_path.exists():
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
        else:
            df = generate_data(cfg, data_cfg, seed=args.seed)
        estimator, test_df = train_model(df, cfg, data_cfg, seed=args.seed)
        print("\n✓ Model trained")
    elif args.mode == "evaluate":
        data_path = OUTPUT_DIR / "generated_data.csv"
        if not data_path.exists():
            print("ERROR: Run --mode generate first")
            return
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        # Retrain the same leak-free estimator from the saved data so evaluate
        # matches `--mode all` exactly (tables-only fallback would differ).
        estimator, test_df = train_model(df, cfg, data_cfg, seed=args.seed)
        results = run_policies(test_df, cfg, seed=args.seed, estimator=estimator,
                               data_cfg=data_cfg)
        comparison, breakdowns_by_policy, off_policy = evaluate(
            results, cfg, historical_df=historical_frame(df))
        from src.experiments.drift import run_drift_experiment as _run_drift
        drift_result = _run_drift(
            df, drift_date=cfg["data"]["drift"]["drift_date"], seed=args.seed)
        write_report(comparison, breakdowns_by_policy, off_policy, drift_result)
        print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()