# Build Challenges & Technical Obstacles

## Censored Outcomes: Learning from Actions Never Taken

**The problem:** In a real payment recovery system, only the *chosen* action's outcome is observed. If we retry a payment and it fails, we know that. If we *don't* retry it, we learn nothing about whether it would have recovered. The model only sees what happened, not counterfactuals.

**How we solved it:**
- Built a **latent outcome oracle** into the simulator: every action has a hidden ground truth, but only the chosen action's result is revealed to the agent.
- Implemented **inverse-propensity weighting (IPW)** evaluation to estimate what would have recovered under full exposure, accounting for the historical bias in which actions were actually taken.
- Report three estimates per policy: observed (naive, biased), IPW (debiased estimate), and oracle (simulator ground truth) — so readers see the selection-bias gap honestly.
- Code: [src/evaluation/off_policy.py](../src/evaluation/off_policy.py) (Hájek-style IPW with clipping to cap variance).

## Historical Policy Bias: "Not Retried" ≠ "Unrecoverable"

**The problem:** Past retry decisions were skewed — `server_busy` errors were retried often, but `expired_card` almost never. A naive model trained on this data learns the historical *policy*, not the truth: "not retried = unrecoverable," which is backwards.

**How we solved it:**
- The synthetic data generator simulates **retry propensity** per failure reason (how likely the historical system was to even attempt recovery).
- Features that encode policy (e.g., `was_retried_historically`, `retry_propensity`) are *never* fed to the model — they're audit columns only.
- Feature engineering enforces **leak-free history**: rolling customer/merchant success rates are computed only from rows with *observed* outcomes, never from oracle columns or censored rows.
- Code: [src/features/feature_engineering.py](../src/features/feature_engineering.py) (explicit censoring discipline with masking).

## Time as a Decision Variable: Retry Timing Matters

**The problem:** When to retry is as important as *whether* to retry. Insufficient-funds failures often recover around salary cycles (2–3 days later). But we don't know the exact optimal window from observational data alone.

**How we solved it:**
- Made time a **controllable decision**: the engine chooses between `retry_soon` (~2h) and `retry_later` (~72h).
- Per-action probabilities adjust: `p_success_retry_later = p_success_retry_soon * 0.7` as a built-in time-sensitivity multiplier (conservative, config-driven).
- Acknowledged in docs: we don't claim to know the *true* optimal window; the dataset is synthetic, and the assumption is testable with real data.

## Customer Annoyance as a Proxy: Budgets Over Friction

**The problem:** We can't measure real customer satisfaction, but every retry attempt is friction: a notification, a failed charge, time waiting. We need a stop rule that isn't arbitrary.

**How we solved it:**
- Introduced **customer friction scores** as a proxy: touches, retries, declines, and time between attempts contribute to a cost model.
- Implemented hard **budgets**: `max_retries_per_customer_day` (5) and `max_attempts_per_payment` (3) enforce politeness rules before any attempt is made.
- The decision engine includes friction in expected-utility math: `EV = P(success)·value − retry_cost − friction_penalty`.
- Code: [src/models/decision_engine.py](../src/models/decision_engine.py) (silent vs. visible touch costs, budget enforcement).

## Synthetic Data Limitations: Distributions are Assumptions

**The problem:** Real Razorpay failure distributions, recovery rates, and cost structures are proprietary. We generate synthetic data that is *realistic* but not real, and model performance on synthetic data may not transfer to production.

**How we solved it:**
- Built a **realistic generator** with failure reasons, payment methods, banks, and customer instruments that reflect real payment ecosystems.
- Simulated **per-action outcomes** with plausible recovery rates (e.g., `insufficient_funds` ~65% when retried later, `server_busy` ~80% soon).
- Measured everything on *the same synthetic test set* so all four policies are compared fairly; absolute ₹ recovered is a proxy, not a claim about production.
- Documented in README: "All data is synthetic — distributions are assumptions, not Razorpay statistics; real recovery rates will differ."

## Risk Gate Out of Scope: Interface-Ready, Mock Implementation

**The problem:** A production recovery agent must block high-risk payments (fraud, chargebacks, disputes) before attempting recovery. But building a real fraud detector is outside the scope of this internship.

**How we solved it:**
- Created a **risk gate interface** ([src/risk/risk_gate.py](../src/risk/risk_gate.py)) with `block_threshold` and per-action allow/block decisions.
- Implemented a **mock gate** that flags payments above a risk score (0.8) for suppression — not fraud detection, but a compliance checkpoint.
- Recorded every blocked decision in the audit trail, so the report shows the cost of suppression (₹64K in the measured run, ~1.8% of the batch).
- The architecture is ready for a real risk system to drop in; no causal claims are made about what constitutes "risk."

## Simulated Payment Execution: Safe but Limited

**The problem:** We can't actually retry real payments in a demo; we need safe, deterministic simulation that never touches production rails but still models outcomes accurately.

**How we solved it:**
- Built a **payment simulator** ([src/simulator/payment_simulator.py](../src/simulator/payment_simulator.py)) that executes actions in memory with idempotency keys, outcome states (SUCCESS, FAILED, EXPIRED, USER_DECLINED, TIMED_OUT), and latent per-action ground truth.
- The simulator is deterministic and repeatable (seed-based), so the entire pipeline (generate → train → evaluate) reproduces bit-for-bit.
- Acknowledged limitation: simulated execution is not real; production would require actual payment APIs, monitoring, and compliance infrastructure.

## Model Fallbacks: XGBoost → HistGB / RandomForest

**The problem:** XGBoost is excellent for tabular data and calibrated probabilities, but it requires the xgboost package. We wanted the core pipeline to work even if that dependency fails or in constrained environments.

**How we solved it:**
- Used **XGBoost by default** (fast, calibrated, proven).
- Implemented fallbacks: if XGBoost is unavailable, the pipeline automatically uses `sklearn.HistGradientBoostingClassifier` (gradient boosting, competitive), then `RandomForest` (reliable baseline).
- All three models produce calibrated probability estimates via `predict_proba()`, so the decision engine is model-agnostic.
- Code: [src/models/recoverability.py](../src/models/recoverability.py) (try/except fallback chain).

## LLM Explainer Graceful Degradation

**The problem:** The web UI includes per-transaction explanations ("why is this payment worth recovering?") powered by an LLM (OpenAI or Groq). But LLM APIs are optional and may not be available in all environments.

**How we solved it:**
- Made LLM integration **optional via environment variables** (`OPENAI_API_KEY` or `GROQ_API_KEY`).
- Built a **deterministic template fallback** that generates human-readable explanations (probability, EV, budgets) even without an LLM.
- The entire pipeline runs **offline** if no API key is set; the web UI shows the template explanation instead.
- Code: [src/explainer/explainer.py](../src/explainer/explainer.py) (environment-aware with fallback).

## Off-Policy Evaluation: Rigor Over Convenience

**The problem:** A naive comparison of policies on the same test set could favor one that takes all the easy recoveries and ignores hard ones. How do we account for the fact that each policy chose *different* actions?

**How we solved it:**
- Implemented **off-policy (propensity-aware) evaluation** that doesn't just count successes, but weights them by how likely each policy was to have taken that action.
- Three metrics are computed: observed (only executed actions), IPW estimate (debiased), and oracle (simulator ground truth).
- The report shows the gap between each policy's observed and oracle performance — that gap is the recoverable headroom.
- Explicitly states: "We do NOT claim these estimates prove real-world performance — the whole module is about showing the selection-bias gap honestly."

## 96 Passing Tests: Coverage Over Claims

**How we validated:**
- Wrote unit tests for feature engineering (leak-free history, censoring discipline), decision engine (EV math, budgets, risk gating), off-policy metrics, and the simulator.
- All tests run on a minimal synthetic dataset to keep CI fast.
- Ran the full pipeline end-to-end with `--mode all` to verify the audit trail, report generation, and web API.
- Tests pass; no claim of 100% correctness, but confidence that core logic holds.

---

**Bottom line:** Every hard part is documented, not hidden. Censored outcomes, policy bias, time, friction, synthetic data, risk gating, simulation, model selection, and LLM fallbacks are all addressed with explicit trade-offs and honest limitations.
