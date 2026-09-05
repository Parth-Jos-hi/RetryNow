# RetryNow — AI Revenue Recovery Agent (Track 3)

> **Razorpay AI Internship — Track 3: AI Revenue Recovery**
> An expected-value recovery agent for failed payments. For every failed transaction it
> diagnoses the likely cause, estimates the expected value of each candidate recovery
> action, respects a risk gate and politeness budgets, executes **exactly one** bounded
> action, observes the outcome, and measures the money actually recovered across the batch.

---

## 🎯 The problem in one sentence

A failed payment is usually not lost money — most failures are transient. Yet merchants
lose that revenue because nobody retries at the *right time, on the right instrument,
within the right budget*. **RetryNow predicts whether recovery is economically worthwhile,
chooses the best action and timing, and stops when it should stop.**

## 📏 The bar — what this project demonstrates

Not "we identified a problem." Measured outcomes, end to end:

| Requirement | Where it shows up |
|---|---|
| **Measured ₹ recovered across a batch** | `--mode all` compares Do-Nothing vs Dumb Retry vs Rule Retry vs the AI Agent on the *same* test transactions — recovered ₹, recovery %, net of costs (see the measured numbers below) |
| **Compliant escalation** | Risk gate blocks flagged transactions before any action; blocked runs are recorded, not silently dropped |
| **Stopping rules** | Per-payment attempt budget, per-customer daily touch budget, expected-value "give_up" — the agent *quits* when retrying is not worth it |
| **Audit trail** | One JSONL audit record per transaction: probabilities, candidate utilities, risk verdict, chosen action, explanation, execution result, timestamp |

### Measured results (seeded run, 1,336 test transactions)

| Policy | Recovered ₹ | Recovery rate (value) | Touchpoints | Net recovered ₹ |
|---|---|---|---|---|
| Do Nothing | ₹0 | 0% | 0 | ₹0 |
| Dumb Retry | ₹854,766 | 38% | 0 (all silent) | ₹852,094 |
| Rule-Based Smart Retry | ₹1,061,383 | 48% | 179 | ₹1,058,805 |
| **AI Recovery Agent** | **₹856,903** | **39%** | **81** | **₹854,481** |

The agent **beats dumb retry by +₹2,387 net** (recovering value while spending
~55% fewer *visible* customer touches: 81 vs 179) and wins outright on
`insufficient_funds` (+₹88K — retry-timing around salary cycles), `upi_server_busy`
and `otp_expired`. A **mock risk gate demonstrably suppresses high-risk recoveries**
(10 payments, ₹64,366 of failed value deliberately not pursued, ~1.8% of the batch
≥ the gate's risk threshold) — compliance before revenue, and the report and
audit trail record each suppressed decision rather than hiding the cost. It trails
the blind rule on one drifted segment —
`bank_server_timeout` on `bank_c` after an infrastructure fix improved that
segment's recoverability by ~18% *after* the agent's training cutoff; the drift
experiment quantifies exactly this (see the report's "Drift" section, and
[`docs/`](docs/) for the honest expectation-setting).

Full reproducible numbers: `outputs/evaluation_report.md`

## 💸 The business case

| Metric | Value |
|---|---|
| Typical industry payment-failure rate (India) | **3–8%** of transactions |
| Failure causes | Insufficient funds, bank timeout, issuer decline, UPI server busy, network drop, 3DS issues, expired/blocked instruments |
| Lost revenue for a ₹10 Cr/year merchant | ~₹35–80 Lakh/year in failed transactions |
| Recovery potential with a predictive engine | **20–30%** of currently-lost failed payments |
| Who wins | Customer (gets the order), merchant (keeps the sale), Razorpay (earns its fee on every recovered ₹) |

## 🧠 What the project contains

```
razorpay/
├── README.md                 ← you are here
├── docs/                     ← full write-ups (problem, architecture, proposal, dataset, demo script)
├── configs/                  ← all thresholds, costs, budgets (single source of truth)
├── src/
│   ├── data/                 ← generate & preprocess synthetic failed-payment data
│   ├── features/             ← feature engineering (leak-free, order-preserving)
│   ├── models/               ← recoverability predictor + expected-utility decision engine
│   ├── risk/                 ← risk gate (interface + mock; NOT fraud detection)
│   ├── agent/                ← the bounded recovery agent
│   ├── simulator/            ← SAFE simulated payment execution + latent outcome oracle
│   ├── recovery/             ← retry scheduler, attempt tracker, messaging
│   ├── policy/               ← baseline policies (do-nothing, dumb retry, rule retry)
│   ├── evaluation/           ← ₹-centric metrics, off-policy eval (IPW), comparison report
│   └── explainer/            ← LLM explainer + customer recovery message generator
├── web/                      ← the real website: server.py (stdlib HTTP + JSON API),
│                              index.html / styles.css / app.js (dark finance UI)
├── scripts/run_pipeline.py   ← CLI: generate → train → policies → agent → evaluate
├── tests/                    ← unit + integration tests (96 passing)
└── outputs/                  ← evaluation_report.md + audit/agent_audit.jsonl + generated_data.csv
```

## ⚙️ How to run it

```bash
cd /d/razorpay
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py --mode generate    # synthetic failed-payment dataset
python scripts/run_pipeline.py --mode train       # recoverability predictor
python scripts/run_pipeline.py --mode all         # generate + train + all 4 policies + evaluate + report
python scripts/run_pipeline.py --mode evaluate    # re-evaluate from saved data (₹-centric comparison report)
python web/server.py                              # the website → http://localhost:8000
```

> Everything together: `python scripts/run_pipeline.py --mode all`. Seeded (default 42) —
> the whole experiment reproduces bit-for-bit.

## 🌐 The website

The evaluation UI is a **real web app** — a zero-dependency Python HTTP server
(`web/server.py`, stdlib only) serving a hand-built dark-finance frontend
(`web/index.html` + `styles.css` + `app.js`) that reads the pipeline's actual
outputs via a JSON API. No frameworks, no CDN, no external assets — it runs
fully offline:

```bash
python web/server.py            # → http://localhost:8000
```

It shows: headline ₹ KPIs with per-policy comparison, action distribution and
outcome cross-tab, a filterable decision audit trail (reason / action / search),
a per-transaction explainer (probabilities, utilities, risk verdict, executed
attempts), the drift experiment, and the off-policy (IPW) summary. Every number
comes from `outputs/` — run the pipeline first if the API reports missing
outputs.

The site is organised as **three tabs**, built so a newcomer can follow it
without reading a report first:

1. **Live demo** — a two-pane theater. You are the **payer** on the left: pick a
   real failure (UPI server busy, insufficient funds, an expired card, a big-ticket
   decline, a flagged payment), pick an amount, press **Pay** — the payment fails.
   On the right the **merchant dashboard** instantly shows the *assurance*: will
   it recover, roughly when, how confident, what the merchant should do — plus a
   short timeline of what happens next. Driven by the real trained model + EV
   engine + risk gate, not canned text.
2. **Merchant feed** — upload your own failed-transactions CSV (one-click
   "**Try the sample file**" loads `samples/merchant_failures.csv`) and see, row by
   row, the single best recovery action, the merchant-facing assurance, and the
   **joint benefit story**: the ₹ the merchant is expected to recover and the
   Razorpay fee earned on it — the pitch that both sides win.
3. **How it works & proof** — the ₹-centric policy comparison, per-action
   outcomes, decision audit trail, per-transaction explainer, drift experiment,
   and the honest limitations.

### 🧪 Try it on your own data

The **Merchant feed** tab is a real upload flow (`POST /api/upload`): upload a
failed-transactions CSV and the same trained estimator + EV engine + risk gate
score **every row** and return the single best action per transaction — with
predicted success rate, ₹ expected value, risk verdict, merchant assurance, and
reasoning — plus **downloadable decision CSV and Markdown report**. A sample file
is at `samples/merchant_failures.csv`.

Required columns: `amount`, `payment_method`, `failure_reason_code`.
Optional: `customer_instruments`, `bank`, `risk_score`, `timestamp`, `txn_id`, …

> The uploaded data is processed in-memory and never written to disk. The engine
> needs a trained estimator (`outputs/estimator.joblib`, produced by
> `--mode all` / `--mode train`). The two demo dashboards are exactly that — a
> *demonstration* of what the agent would do; they run on the synthetic model and
> move no real money.
## 🏗️ Architecture
```
Failed Payment
      ↓
Transaction Context ──────────────── features (leak-free, order-preserving)
      ↓
Root Cause / Failure Analysis
      ↓
Recoverability Estimation      ← per-action P(success | txn, action, timing)
      ↓
Candidate Action Evaluation    ← EV(action) = P(success)·value − retry_cost
                                   − friction_penalty − risk_penalty
      ↓
Risk Gate                     ← blocked → give_up (recorded, compliant)
      ↓
Expected-Utility Policy Engine ← argmax over {retry_soon, retry_later,
                                   switch_method, send_link, give_up}
                                   subject to attempt/touch budgets
      ↓
Bounded Recovery Execution    ← SAFE simulator (SUCCESS/FAILED/EXPIRED/
                                   USER_DECLINED/TIMED_OUT, idempotent)
      ↓
Outcome ──► Audit record (JSONL, per txn)
      ↓
Metrics + Feedback ──► ₹ recovered vs baselines; retraining
```
Four policies are compared on the same test transactions:
1. **Do Nothing** — zero recovery, zero touchpoints, zero cost (the floor).
2. **Dumb Retry** — retry every eligible failure once after a fixed window.
3. **Rule-Based Smart Retry** — failure-reason + fixed time windows (the "Smart Retry" style).
4. **AI Recovery Agent** — learned probabilities + expected-utility selection + risk gate
   + budgets + timing.
## 🔬 The hard parts we model explicitly (and why)
- **Censored outcomes.** Only the *chosen* action's outcome is ever observed — the others
  are counterfactuals. Non-retried payments are **not** labelled failures. The simulator
  keeps latent per-action truths internally (the oracle); the agent sees only what it acted
  on. Evaluation uses propensity-aware (IPW) estimators and reports OBSERVED vs
  COUNTERFACTUAL separately.
- **Historical policy bias.** Past retry policy didn't retry everything equally
  (e.g. `server_busy` often, `expired_card` almost never) — naive models learn
  "not retried = unrecoverable", which is wrong. The generator simulates this
  retry-propensity; evaluation is propensity-aware. No causal claims are made from
  observational data.
- **Time is a decision variable.** `retry_soon` (≈2h) vs `retry_later` (≈72h) are
  configurable windows; `insufficient_funds` recovers around salary cycles. We do not
  claim to know the *optimal* time from real data — our dataset is synthetic.
- **Customer annoyance is a documented proxy** (`customer_friction_score`): touches,
  retries, declines, time between attempts. Budgets cap it: MAX_RETRIES and
  MAX_CUSTOMER_TOUCHES → `give_up`.

## 🤖 Defaults & fallbacks

- **Model:** XGBoost (falls back to `sklearn.HistGradientBoostingClassifier` /
  `RandomForest`) — tabular, CPU-friendly, calibrated probabilities.
- **Risk gate:** mock score + interface ready for a production risk system
  ("Risk Gate — interface prepared for integration with a production risk system").
- **Execution:** fully simulated; never touches real payment rails (idempotency keys,
  outcome states — but zero real money moves).
- **LLM explainer:** OpenAI/Groq-compatible via `OPENAI_API_KEY`/`GROQ_API_KEY`;
  deterministic template fallback → the entire pipeline runs offline.
- Everything runs on CPU.

## 📄 Docs

- [Problem statement](docs/problem-statement.md)
- [Solution architecture](docs/solution-architecture.md)
- [One-page project proposal](docs/project-proposal.md)
- [Dataset guide](docs/dataset-guide.md)
- [Demo script (evaluation day)](docs/demo-script.md)
- [Evaluation & reproducibility pin](docs/evaluation.md)

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

## ⚠️ Limitations (stated honestly)

1. **All data is synthetic** — distributions are assumptions, not Razorpay statistics;
   real recovery rates will differ.
2. **Causal claims are not possible** from observational retry outcomes alone; our
   evaluation is propensity-aware but simulated.
3. **Censored outcomes** mean "not retried" ≠ "not recoverable" — handled, not erased.
4. **Customer annoyance is a proxy**, not measured customer feedback.
5. **Risk detection is out of scope** — the risk gate is an interface + mock.
6. **Payment execution is simulated** — production would require real payment APIs,
   compliance, security, and monitoring.
7. **Exact optimal retry timing is not known from real data.**

---

*Built for the Razorpay AI Internship (Track 3: AI Revenue Recovery). All data is
synthetic — no real customer data is used. All claims are about the synthetic
environment unless explicitly stated otherwise.*