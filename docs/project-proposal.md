# One-Page Project Proposal

## RetryNow — An Expected-Value Revenue Recovery Agent (Track 3)

### Problem
In India, **3–8% of digital payment transactions fail** — mostly for transient reasons
(`insufficient_funds`, `bank_server_timeout`, `upi_server_busy`, `issuer_decline`).
Most of those failures are recoverable: retried at the right time, or routed to a
different instrument, the money goes through. Today platforms run fixed rules (retry
everything after 4 hours), which over-retries hopeless cases, under-retries recoverable
ones, and never personalizes. **Result: millions of ₹ of recoverable revenue is written
off every year.**

### Solution
An **AI recovery agent** that, for every failed payment, runs a bounded loop:
diagnose the failure, estimate each candidate action's **expected value**
(`P(success)·value − retry_cost − friction − risk`), pass the risk gate, execute
**exactly one** action, observe the outcome, record an audit trail, and stop when
the budget says stop.

- **retry_soon** (transient glitch, ~2h) · **retry_later** (funding issue, ~72h)
- **switch_method** (try the customer's other instrument) · **send_link** (customer retries)
- **give_up** (EV ≤ 0 — don't waste cost & goodwill)

A lightweight LLM layer explains each decision and drafts a friendly customer recovery
message. **The bar is measured money: recovered ₹ across a batch, comparisons against
do-nothing / dumb retry / rule-based retry on the same transactions, stopping rules,
and an audit trail for every decision.**

### Why now / Why Razorpay
Razorpay already has **Smart Retry** — a rule-based recovery product. RetryNow is the
data-driven evolution: learned timing, instrument switching, expected-value selection,
and stop-annoying-the-customer logic. Tabular ML, CPU-only, deterministic and seeded —
built for a reproducible hackathon demo that slots into existing infrastructure with a
schema exchange.

### What makes the evaluation honest
The synthetic environment encodes **latent per-action ground truth** (which action
would have succeeded, counterfactually) while the agent only ever observes the outcome
of the action it **chose**. Historical retry policy leaves a **retry-propensity bias**
(server_busy retried often, expired_card never) — we make that explicit and evaluate
propensity-aware (IPW), reporting OBSERVED vs COUNTERFACTUAL outcomes separately.
No causal claims from observational data.

### Approach & timeline
| Phase | Deliverable |
|---|---|
| 1. Data | Deterministic synthetic generator: latent per-action truths, retry propensity, drift switch (~20k txns) |
| 2. Predictor | Calibrated `P(success \| txn, action, timing)` (XGBoost, HistGB fallback) |
| 3. Decision engine | Expected-utility argmax + risk gate + budgets |
| 4. Agent + simulator | Bounded 13-step loop; safe simulated execution (idempotent) |
| 5. Policy comparison | Do-nothing / dumb / rule / AI agent on same txns → ₹-centric report |
| 6. Demo | Real website (zero-dependency web app): KPI cards, policy comparison, decision trail, per-txn explainer, drift — all from real outputs |

### Expected impact (simulated)
| Baseline (dumb retry) | RetryNow | Lift |
|---|---|---|
| ~15% of failures recovered | **20–30%** of failures recovered | +5–15 pts 📈 |
| Same fixed delay for everyone | Personalized per instrument/merchant/time | — |
| Every failure touched once | Fewer touches, higher ₹/touch (recovery-to-bother) | — |

### Differentiators
1. **Decision agent, not classifier** — expected-value selection, not threshold scoring.
2. **Censored-outcome honesty** — counterfactuals modeled, propensity-aware evaluation.
3. **Instrument switching** — most demo projects ignore the *how*.
4. **Politeness + risk constraints** — budgets and a risk gate are first-class.
5. **Explainable + auditable** — every decision ships with a reason and an audit record.
6. **Fully offline-runnable** — no API key needed to reproduce the full result.

### One-page verdict
RetryNow is a *small, precise, measurable* piece of revenue intelligence — the kind of
project that becomes a real product feature, not a research poster.