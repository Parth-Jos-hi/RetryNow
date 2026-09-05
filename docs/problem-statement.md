# Problem Statement — AI Revenue Recovery

## Track
**Track 3: AI Revenue Recovery** — Razorpay AI Internship.

## The business problem, in plain language

Every day, digital-payment platforms process millions of transactions. A meaningful
slice of those transactions **fail** — the customer's card is declined, the UPI server
is busy, the bank times out, or the customer's balance is short at that exact moment.

Industry data consistently shows an **overall payment-failure rate of 3–8%** in India
(depending on instrument and merchant vertical). For UPI alone, which carries the bulk
of Indian digital payments, "system busy" and "bank timeout" failures are among the most
common decline reasons and are **largely transient** — the money was there, and the
network/issuer was simply not cooperative at that instant.

### What happens to failed payments today?

Most platforms/merchants handle failures in one of three ways:

1. **Do nothing.** The failure is recorded, the customer may or may not get a nudge, and
   the revenue is written off. The merchant lost the sale. The platform lost the fee.
2. **Dumb retry.** Retry *every* failure once after a fixed window (e.g., 4 hours).
   - Over-retries the hopeless cases (stolen-card blocks, expired cards) → wasted network
     calls, bank penalty fees, worse issuer relationships.
   - Under-retries the recoverable cases (insufficient funds that resolve after payday,
     UPI "server busy" that clears in 20 minutes).
   - Has **zero personalization** — every customer and every payment is treated identically.
3. **Smart retry (rules).** Razorpay's current "Smart Retry" uses heuristics like failure
   reason codes and simple time windows. This is a real product and a real improvement,
   but the decision logic is **rule-based, not learned**:
   - It does not *predict* the probability that a given retry succeeds.
   - It does not choose *between* instruments (retry UPI vs. switch to card vs. send link).
   - It does not learn merchant-, customer-, or time-specific patterns.

### The cost of lost recoverable payments

For a merchant doing **₹10 crore/year** of payments:

| Line | Estimate |
|---|---|
| Failed payments (5% of volume) | ₹50 Lakh / year |
| Reasonable share that is *transiently recoverable* | ~35–50% |
| Recovered by a *dumb* fixed retry | ~15% of failures |
| Recovered by a *predictive* engine | 20–30% of failures → **₹10–15 Lakh / year more** |

Every recovered rupee is revenue for the merchant, keeps the customer happy (they actually
get their order), and earns Razorpay its transaction fee. Revenue recovery is therefore not
an optimization — **it is a revenue line item**.

## The problem we solve

> **Given a failed payment transaction, decide the *single best recovery action* —
> whether to retry, when to retry, and on which instrument — so that total recovered
> revenue is maximized while customer annoyance and retry cost are minimized.**

Concretely, for every failed payment, the system must choose one of:

| Action | Meaning |
|---|---|
| `retry_soon` | Retry the *same* instrument within hours (transient technical failure). |
| `retry_later` | Retry the *same* instrument after a few days (funding/session issue). |
| `switch_method` | Try a *different* instrument the customer has on file. |
| `send_link` | Send the customer a payment link / push notification to retry themselves. |
| `give_up` | Do not retry — the failure is effectively permanent and retrying wastes money & goodwill. |

## The expected-value framing

A recovery decision is a **money decision**. For each candidate action the agent
estimates:

```
EV(action) = P(success | txn, action, timing) × transaction_value
             − retry_cost − customer_friction − risk_penalty
```

and executes the single action with the highest expected value — including
`give_up` (EV = 0) — subject to budgets. This is why a binary classifier is
insufficient: two payments can have the same "recoverable?" probability and still
deserve different actions because their *value* and *friction cost* differ.

## Why this is hard (and why AI helps)

1. **Censored outcomes.** We only observe retry success *if we retried*. Failed payments
   that were never retried are counterfactuals — never "failures". The environment keeps
   latent per-action truths so policies can be evaluated honestly (propensity / IPW).
2. **Historical policy bias.** Past retry policy retried some reasons aggressively and
   almost never touched others (`expired_card`), so "not retried" does *not* mean
   "unrecoverable". Naive models trained on biased historical decisions learn the wrong
   thing; evaluation must be propensity-aware. Observational data cannot prove causal
   recovery performance.
3. **Heterogeneous recoverability.** "insufficient_funds" is recoverable after a
   salary cycle; "account_blocked" is near-permanent; "issuer_decline" sits in between.
4. **Time-of-day / calendar effects.** Bank-server issues cluster around macro events
   and load spikes. Retry timing is a *continuous decision*, not a checkbox.
5. **Instrument switching.** Sometimes the correct answer is not *when* but *how* —
   the customer's UPI failed because of a short-lived issue, but their card
   would have gone through.
6. **Politeness constraint.** Recovery is not free. Customer annoyance is a documented
   proxy (`customer_friction_score`: touches, retries, declines, time between attempts),
   and budgets (`MAX_RETRIES`, `MAX_CUSTOMER_TOUCHES`) end the loop. The engine must
   learn to *stop*.

## Success criteria

- **Primary:** % of failed-transaction **value** recovered — and the **measured ₹
  recovered across a batch** — compared against `dumb retry` and rule-based `smart retry`
  baselines run on the **same transactions**.
- **Secondary:** recovery-to-bother ratio (₹ recovered per customer touchpoint);
  per-instrument success-rate lift from smart instrument switching;
  unnecessary-retry reduction; per-reason/instrument/time/value-bucket breakdowns.
- **Process constraints:** fully explainable decisions (every action ships with a
  human-readable reason derived from the actual decision features), a risk gate that
  blocks flagged transactions into a recorded safe state (compliant escalation), stopping
  rules enforced by budgets, and a per-transaction audit trail backing every aggregate.
- **Honesty:** counterfactual/unknown outcomes are reported separately from observed
  ones; limitations are documented, not hidden.

## Data we use

Because real transaction data is private to Razorpay, this project ships a **synthetic
data generator** that models the real-world distributions (instrument mix, failure-reason
mix, transient-vs-permanent recoverability, salary-cycle effects). The generator is
deterministic (seeded) and produces ~20,000 realistic failed-payment transactions. The ML
pipeline is designed so the *same* code path plugs into real Razorpay event streams by
exchanging only the data source.

## Related real-world context

- India's payments stack: UPI dominant, card less common than in the West, netbanking
  residual. UPI "SERVER BUSY" and bank-timeout failures are endemic.
- Platforms with mature recovery engines report **20–30%** recoverable-failure recovery.
- Recovery interacts with **risk**: retrying an instrument flagged risky is dangerous;
  the decision engine must respect a risk gate (clearly scoped as future work in this
  project, with an interface ready for it).

---

*Back to [README](../README.md).*