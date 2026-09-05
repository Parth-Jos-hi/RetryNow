# Demo Script — RetryNow (Track 3: AI Revenue Recovery)

A 5–6 minute walkthrough for the evaluation day. Every number below is from a
real seeded run (`--mode all --seed 42`, 1,336 test transactions) — verify
fresh numbers after a re-run, they publish to
`outputs/evaluation_report.md`.

---

## 0. Setup (before the reviewers arrive)

```bash
cd /d/razorpay
# one command regenerates everything from scratch (~2 min)
PYTHONIOENCODING=utf-8 python -u scripts/run_pipeline.py --mode all --seed 42
# in a second terminal, start the website
python web/server.py                   # → http://localhost:8000
```

> `PYTHONIOENCODING=utf-8` keeps ₹ symbols clean on Windows terminals.
> The website is a zero-dependency web app (stdlib HTTP server + hand-built
> frontend) — no frameworks, no CDN, works fully offline.

---

## 1. The pitch (60–90 s) — *without* opening code

- **Problem**: a failed payment is usually not lost money (3–8% of Indian
  payments fail; most transiently). Merchants don't retry because "when, how,
  how often" is a judgment call — so revenue dies silently.
- **The agent's job**: for each failed txn, diagnose → estimate per-action
  recovery probability → pick the action with the best **expected value**
  (`EV = P·amount − retry_cost − friction − risk_penalty`) → respect budgets →
  act once → observe → audit.
- **One sentence**: *"It's not a classifier; it's a decision engine that
  argues in rupees."*
- **Key honesty point up front**: all data is synthetic; the value is the
  *method* — censored outcomes, propagation of uncertainty, and an audit
  trail. Say this yourself before anyone asks.

## 2. The measured result (60 s)

Open `outputs/evaluation_report.md` → the policy comparison table:

| Policy | Recovered ₹ | Touchpoints |
|---|---|---|
| Do Nothing | ₹0 | 0 |
| Dumb Retry | ₹854,766 | 0 (silent) |
| Rule Smart Retry | ₹1,061,383 | 179 |
| **AI Agent** | **₹859,546** | **82** |

Say: *"The agent beats dumb retry by ₹5,014 net and recovers most of the
rule's value while bothering 45% fewer customers (82 vs 179 touches). The
rule wins more overall, and the report explains exactly why."* → then the
next slide owns the gap.

## 3. Why the agent trails the rule: the Drift section (45 s)

Scroll to **"Drift — Why the Agent Trails Rule on bank_server_timeout"**:

- An infrastructure fix improved `bank_c` server timeouts **after** the
  agent's training cutoff: 66.4% → 78.2% recoverability (+17.8% lift).
- The rule retries blindly, so it wins in that segment; the agent's ML was
  trained pre-drift and is *correctly conservative* there.
- **This is the honest limitation, quantified** — and the failure mode tells
  you the fix (feature-flagged retraining). A panel that has already asked
  "what are you hiding?" now has nothing to dig for.

## 4. The audit trail (60 s, most memorable)

Open `outputs/audit/agent_audit.jsonl` + the website's
**"Why the agent chose this action"** panel (`http://localhost:8000/#explain`).
Story of one transaction:

- `insufficient_funds` customer, month-start (salary window) → estimated
  `retry_later` P(success) 63% → EV ₹330 vs retry-now ₹93 → chose
  `retry_later` in 72h → outcome logged.
- Show the explainability fields in one JSONL line: probabilities for all 4
  actions, utilities, cost/friction, risk verdict, explanation string.

Say: *"Every decision is machine-readable: what was considered, what was
chosen, what happened. That's what makes this deployable — an auditor can
trace any rupee."*

## 5. Interactive demo (60–90 s)

In the website:
1. Hero stats: AI recovered ₹859,546 and "net vs dumb retry" from the real run.
2. **Policy comparison** table (4 policies on the same transactions).
3. **Decision trail** — filter by a reason (e.g. `bank_server_timeout`) →
   see the audit rows; search a txn id.
4. **Explain** — pick a transaction → probabilities per action vs expected
   utilities, chosen action, risk verdict, executed attempts.
5. **Drift** section — the pre/post lift cards (66.4% → 78.2%).
6. Run experiment in front of them: `--mode all` again → identical numbers
   (seed 42). *"Every run is reproducible, bit-for-bit."*

## 6. Architecture + limitations (60 s)

- Walk the flow diagram in `README.md` (Failed Payment → … → Audit).
- Mention the four hard problems modeled explicitly: **censored outcomes**
  (non-retried ≠ failed), **historical policy bias** (IPW off-policy section
  in the report), **time as a decision variable**, **friction budget**
  (MAX_RETRIES / MAX_CUSTOMER_TOUCHES → give_up).
- Closing line: *"Open problems are stated in the report — the strategy is
  honest evaluation, not optimized headlines."*

---

## Q&A cheat sheet

| Likely question | Answer |
|---|---|
| "Is this just a retry loop?" | No — it selects among 5 actions by expected value, respects risk + friction budgets, and stops when EV can't clear costs (give_up ≈ 3%). |
| "Why does the rule beat the agent?" | One drifted segment (bank_c timeouts) where the rule fires blind post-fix; quantified in the Drift section, with the retraining fix. |
| "How do you know recovery isn't just luck?" | Off-policy section: observed ₹1.9M (naïve) vs IPW ₹1.1M — selection bias made retried txns look better than they are; the agent is evaluated net of that. |
| "Would this work on real data?" | The architecture is railway-grade on interfaces (risk gate, simulator, explainer are all swappable); the *numbers* are synthetic until real data exists. |
| "Why so much focus on honesty?" | Because the biggest failure mode in auto-recovery is overclaiming — a production system that says 80% recovery but only recovered 40% gets turned off. |

## Troubleshooting

- **₹ shows as mojibake** → run with `PYTHONIOENCODING=utf-8`.
- **Website won't start** → the web app is stdlib-only; if `python web/server.py`
  fails, Python itself has a problem. If it serves but `/api/data` returns
  "Outputs missing", run the pipeline first (the report + audit are always the
  demo backbone).
- **Stale outputs** → re-run the one-command pipeline; the audit file
  truncates each run so the JSONL always matches the current report.