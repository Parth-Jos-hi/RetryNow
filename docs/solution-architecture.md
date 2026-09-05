# Solution Architecture — RetryNow

## Overview

RetryNow is an **expected-value recovery agent**. For every failed payment it runs a
bounded loop — *diagnose → estimate recoverability → score candidate actions by expected
value → pass the risk gate → execute one action → observe → record → stop when the
budget says stop* — and reports **measured ₹ recovered across the batch**, with an audit
trail per transaction.

This is deliberately **not** a binary classifier ("will a retry succeed? yes/no") and
**not** a fixed retry policy. It is a decision agent that answers: *should we attempt
recovery at all, what action, when, on which instrument, and when should we stop?*

```
Failed Payment
      ↓
Transaction Context ────────────── features (leak-free, order-preserving)
      ↓
Root Cause / Failure Analysis      ← failure reason + context diagnostics
      ↓
Recoverability Estimation          ← P(success | txn, candidate action, timing)
      ↓
Candidate Action Evaluation        ← EV(action) = P(success)·value
                                       − retry_cost
                                       − friction_penalty
                                       − risk_penalty
      ↓
Risk Gate (interface + mock)      ← blocked ⇒ give_up, recorded, compliant
      ↓
Expected-Utility Policy Engine    ← argmax EV subject to budgets
                                       {retry_soon, retry_later, switch_method,
                                        send_link, give_up}
      ↓
Bounded Workflow Execution        ← SAFE simulator: SUCCESS / FAILED / EXPIRED /
                                       USER_DECLINED / TIMED_OUT (idempotent)
      ↓
Outcome ──► Audit record (JSONL, one per txn) ──► Metrics + Feedback ──► retraining
```

## Module map

| Component | Module | Purpose |
|---|---|---|
| Synthetic transaction generator | `src/data/generate_synthetic_data.py` | deterministic, seeded ~20k failed txns; *latent per-action ground truth*; historical retry-propensity; drift switch |
| Feature/context builder | `src/features/feature_engineering.py` | leak-free rolling histories; preserves input row order |
| Root-cause / recoverability layer | `src/models/recoverability.py` | per-action recoverability estimators (same-method, alt-method, link) |
| Prediction model | `src/models/predictor.py` | XGBoost→HistGB→RF + calibration; learned probabilities, not rules |
| Expected-utility decision engine | `src/models/decision_engine.py` | argmax EV over the five actions with cost/friction/risk terms |
| Risk gate | `src/risk/risk_gate.py` | clean interface + mock score; **NOT** fraud detection (out of scope) |
| Recovery agent | `src/agent/recovery_agent.py` | the bounded 13-step loop + audit JSONL |
| Payment simulator | `src/simulator/payment_simulator.py` | safe simulated execution; latent oracle for counterfactuals |
| Recovery primitives | `src/recovery/` | retry scheduler (bandwidth bucket), attempt tracker (budgets), messaging |
| Baseline policies | `src/policy/` | do-nothing, dumb retry, rule-based smart retry — same transactions |
| Off-policy evaluation | `src/evaluation/off_policy.py` | propensity / IPW; OBSERVED vs COUNTERFACTUAL reporting |
| ₹-centric metrics | `src/evaluation/metrics.py` | recovered value %, incremental ₹ vs baselines, per-reason/instrument/time/value buckets, recovery-to-bother |
| Explainer | `src/explainer/` | per-decision "why" from actual features (LLM with template fallback) |
| Website | `web/server.py` + `web/` | stdlib HTTP server + hand-built frontend; KPIs, policy comparison, decision trail, explainer, drift — reads real outputs via JSON API |

## Step 1 — Diagnose & estimate recoverability

The predictor (Layer 1, `src/models/predictor.py`) learns

```
P(success | txn, candidate_action, timing)
```

from the engineered context (failure reason, amount, method, bank, merchant vertical,
hour/day/week, customer & merchant rolling recovery history, instrument lockup proxy,
salary-cycle window). *Learning* this matters: the generator encodes causal structure
(e.g. `insufficient_funds` recovers around salary windows; `card_expired` almost never
recovers on the same card), but the model must rediscover it — we do not hard-code
reason→action rules into the engine.

Calibration (default isotonic, `CalibratedClassifierCV`) keeps the probabilities honest
enough to feed expected-value arithmetic.

## Step 2 — Expected-value decision engine

For each candidate action `a` the engine computes

```
EV(a) = P(success | txn, a, timing) × value(txn) − cost(a) − friction_penalty(a) − risk_penalty(a)
```

All terms come from `configs/config.yaml` (retry cost per call, friction per touch
with per-action burden weights, risk multiplier) — the engine is config-driven and
explainable, and thresholds in the classic sense re-appear only as *defaults* for
exposition. `give_up` is a real candidate with `EV = 0`, not a fallback.

Constraints applied before argmax:

- `max_attempts_per_payment` — hard stop after N attempts on one payment.
- `max_retries_per_customer_day` — politeness cap at the customer level.
- Risk gate: if blocked → `give_up` (recorded with the reason; the payment escalates
  to a safe terminal state, never to an unrecorded void).

## Step 3 — Risk gate (interface + mock)

```python
class RiskGate:
    def check(self, txn, action) -> RiskVerdict: ...   # ALLOW / BLOCK
```

- **Scope note:** fraud detection is *outside* this project's scope. `RiskGate` is an
  interface prepared for integration with a production risk system; the shipped
  implementation scores from synthetic risk fields carried in the generated data.
- A blocked decision is stored in the audit trail with the verdict — the agent
  demonstrates *compliant escalation*, not silent dropping.

## Step 4 — The bounded agent loop

`src/agent/recovery_agent.py` implements the brief's loop per failed transaction:

1. Read transaction context → 2. diagnose failure → 3. estimate recoverability →
4. generate candidate actions → 5. estimate EV per candidate → 6. check risk gate →
7. apply retry/touch budgets → 8. select exactly ONE action → 9. execute via the
   SAFE simulator → 10. observe result → 11. record decision + outcome → 12. update
   metrics/feedback → 13. stop when budget or policy says stop.

Execution is **idempotent** (idempotency keys; re-executing the same attempt returns the
same outcome), mirroring how a real payment system would behave without ever touching
real rails.

## Step 5 — Censored outcomes, historical bias, honest evaluation

The two hardest parts of this problem are data problems, and we model them explicitly:

- **Latent truth / oracle.** The generator assigns every failed transaction a latent
  `P(success | action)` for *all* candidate actions. The agent observes only the outcome
  of the action it chose — the others remain counterfactuals. The oracle is used **only**
  by the evaluator, never by the agent.
- **Selective labels.** Non-retried payments are *not* labelled "failure". Evaluation
  distinguishes OBSERVED outcomes from COUNTERFACTUAL/UNKNOWN outcomes.
- **Historical policy bias.** Generated history includes a retry-propensity per reason
  (e.g. `server_busy` retried often, `expired_card` rarely), so a naive model would learn
  "not retried = hopeless". We train on the same data the agent generates, use propensity
  information in evaluation (IPW), and say plainly: observational retry outcomes do not
  prove causal recovery performance.

### Policies compared (same test transactions)

| Policy | Behaviour | Expected |
|---|---|---|
| Do Nothing | no action, no touch, no cost | floor |
| Dumb Retry | every eligible failure once after fixed delay | over-retries, burns touches |
| Rule Retry | failure-reason + fixed time windows | better, still blind to context |
| **AI Agent** | learned P, EV argmax, risk gate, budgets, timing | ↑ recovered ₹, ↓ touches, ↓ wasted retries |

## Step 6 — Metrics: money first

Primary: **% of failed-transaction VALUE recovered** (and net of retry cost).
Also reported: incremental ₹ vs Dumb and vs Rule baselines, recovery rate by count and
value, unnecessary-retry count, customer touchpoints, recovery-to-bother ratio (₹/touch),
retry cost, risk-blocked count, give-up rate, and recoveries broken down by reason,
instrument, time window, and value bucket. Accuracy/F1 are reported as diagnostics only.

A one-line-per-transaction audit/JSONL backs every aggregate, so any number in the
website can be traced to the transactions that produced it.

## Data flow & artifacts

```
configs/config.yaml        → every knob (costs, budgets, thresholds, simulation, explainer)
data/synthetic/failed_payments.csv      (generated, with latent oracle + propensity)
data/synthetic/audit/*.jsonl            (per-policy audit trails)
models/recovery_model.joblib            (serialized predictor)
reports/evaluation.json                 (metrics + policy comparison)
reports/evaluation_report.md            (human-readable comparison report)
outputs/audit/agent_audit.jsonl         (transaction-level decision log, the website feeds on it)
```

## Reproducibility & honesty

- Seeded at every randomness source (data, split, simulation) — `seed=42` reproduces
  the whole experiment.
- Every report carries `seed`, `config_hash`, model version.
- The pipeline never *requires* a network call (LLM degrades to templates); it never
  executes real payments, period.

## Scope boundaries (explicitly out of scope)

- Real Razorpay transaction data (private; synthetic stand-in with the same schema contract).
- **Real payment execution** — the simulator is the only "execution" layer (safe by design).
- Fraud/risk scoring itself — we consume a risk gate interface.
- Chargeback reduction (Track 2 territory).
- Production infrastructure (queues, Kafka) — scheduling is modeled in-process.

## Future extensions

1. Time-aware bandit for retry timing (explore windows per vertical, instead of fixed).
2. Instrument transition graph (learn "A failed ⇒ B likely OK" pairs).
3. Doubly-robust off-policy estimators once real logs exist (propensity models, DR/SNIPS).
4. Merchant-margin-aware scaling of cost/friction terms.

---

*Back to [README](../README.md).*