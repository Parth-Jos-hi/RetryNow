# Evaluation: Reproducibility Pin

One command, bit-for-bit reproducible results. This pins what a single run
should produce so anyone (including the reviewers) can verify independently.

## The command

```bash
cd /d/razorpay
pip install -r requirements.txt        # install deps (~1–2 min)
PYTHONIOENCODING=utf-8 python -u scripts/run_pipeline.py --mode all --seed 42
```

Windows note: `PYTHONIOENCODING=utf-8` keeps the ₹ symbol clean in the
terminal. On Linux/macOS it is harmless and optional.

Runtime: ~2 min (dominated by model training + calibration).

## What the run does

| Step | Output |
|---|---|
| 1. Generate | 4,000 failed transactions (of 20,000 generated), 1,645 of them **censored** (never retried historically) |
| 2. Train | Recoverability estimator on **pre-drift** data only (train: 2,664 rows, 1,584 exposed/labeled) — tempally split at the drift date, so the test set is genuinely held-out |
| 3. Policies | Same 1,336 test transactions run through do_nothing / dumb_retry / rule_retry / **RecoveryAgent** |
| 4. Evaluate | ₹-centric metrics + off-policy (IPW) on the historical record + drift experiment |
| 5. Report | `outputs/evaluation_report.md` + `outputs/audit/agent_audit.jsonl` + `outputs/generated_data.csv` |

## Pinned numbers (seed 42)

### Policy comparison — the headline table

| Policy | Recovered ₹ | Recovery rate (value) | Attempts | Touchpoints | Net recovered ₹ |
|---|---|---|---|---|---|
| do_nothing | ₹0 | 0% | 0 | 0 | ₹0 |
| dumb_retry | ₹854,766 | 38.0% | 1,336 | 0 | ₹852,094 |
| rule_retry | ₹1,061,383 | 48.1% | 1,289 | 179 | ₹1,058,805 |
| **recovery_agent** | **₹859,546** | **38.6%** | **1,219** | **82** | **₹857,108** |

Key deltas:
- **AI vs dumb_retry: +₹5,014** net recovered value
- **AI vs rule_retry: −₹201,697** — concentrated in the drifted
  `bank_server_timeout`/`bank_c` segment (post-fix recoverability improved
  66.4% → 78.2% *after* the agent's training cutoff). The rule fires blind
  and therefore wins there; the drift section quantifies it.
- **Touchpoints**: 82 vs 179 (rule) — 45% fewer visible customer touches for
  most of the rule's recovery.

### Off-policy summary (the historical record — the data we learned from)

| Metric | Value |
|---|---|
| Observed recovered value (naïve) | ₹1,897,581 |
| IPW estimated recovered value | ₹1,122,773 |
| Oracle expected recovered value (counterfactual) | ₹3,304,375 |
| Censored rows | 1,645 / 4,000 |

The gap between observed and IPW is **selection bias made visible**: the
historical policy retried favorable reasons more often, so naive rates
overstate recoverability. The agent's evaluation is net of that fact.

### Audit trail consistency

- 1,336 `decide` events (one per test transaction)
- 1,219 `execute` events (only non-give_up decisions execute)
- Every execute's txn appears in decide; audit file truncates per run so it
  always matches the report.

### Drift experiment (why the agent trails the rule in one segment)

- `bank_c` + `bank_server_timeout` pre-drift: 66.4% mean P(success)
- post-drift (≥ 2025-05-01): 78.2% → **+17.8% lift**
- Implication: the ML is a pre-drift artifact; periodic retraining or a
  feature-flagged model update closes the gap.

## Verification checklist (for the reviewer)

1. `python -m pytest tests/ -q` → **86 passed**.
2. Run the pinned command; compare the report's comparison table to the
   numbers above (same seed → same values).
3. `outputs/audit/agent_audit.jsonl` exists with 2,555 records, 1,336 decide
   + 1,219 execute.
4. Optional: `python web/server.py` → http://localhost:8000 — hero stats show
   ₹859,546 recovered / 82 touchpoints / 2.6% give-up, all from the audit trail.
5. The report's Limitations section is present and honest (synthetic data,
   IPW dependence, oracle unavailable to the agent, mock risk gate,
   censored = unknown).

## Mutation checks (same command, new seed)

Changing `--seed` re-initializes the synthetic world — numbers shift because
the world shifts, not because of nondeterminism. If two runs with the *same*
seed ever differ, that is a bug worth reporting.