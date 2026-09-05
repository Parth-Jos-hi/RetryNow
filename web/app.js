/* RetryNow — client-side rendering.
   Three tabs: Live demo (payer <-> merchant theater), Merchant feed (your CSV,
   the joint benefit story), How it works & proof (the analytics). */
"use strict";

const fmt = {
  inr: (v) => "₹" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 }),
  inr2: (v) => "₹" + Number(v).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  pct: (v) => (Number(v) * 100).toFixed(1) + "%",
  num: (v) => Number(v).toLocaleString("en-IN"),
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };

let DATA = null;
let SCENARIOS = null;
let CURRENT_SCENARIO = null;
let DEMO_LOCK = false;

/* ───────────────────────────── tabs ───────────────────────────── */

const TABS = [
  ["demo", "Live demo"],
  ["merchant", "Merchant feed"],
  ["proof", "How it works & proof"],
];

function switchView(key) {
  document.querySelectorAll(".tab-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === key));
  document.querySelectorAll(".view").forEach((v) =>
    v.classList.toggle("active", v.id === "view-" + key));
}

function initTabs() {
  const wrap = $("tabs");
  wrap.innerHTML = "";
  for (const [key, label] of TABS) {
    const b = el("button", "tab-btn");
    b.textContent = label;
    b.dataset.view = key;
    b.onclick = () => switchView(key);
    wrap.appendChild(b);
  }
  switchView("demo");
}

/* ───────────────────────────── data ───────────────────────────── */

function actionKind(a) {
  if (a === "give_up") return { label: "give_up", cls: "warn" };
  if (a === "retry_soon" || a === "retry_later") return { label: "retry (silent)", cls: "" };
  return { label: "visible touch", cls: "good" };
}

async function init() {
  let res;
  try {
    res = await fetch("/api/data");
  } catch {
    document.body.innerHTML =
      `<div class="status"><h1>Server unreachable</h1><p>Start it with <code>python web/server.py</code>.</p></div>`;
    return;
  }
  const payload = await res.json();
  if (!payload.ok) {
    document.body.innerHTML =
      `<div class="status"><h1>Outputs missing</h1><p>${payload.hint || "Run the pipeline first."}</p></div>`;
    return;
  }
  DATA = payload;
  initTabs();
  render();
  initDemo();
  initUpload();
  initSample();
}

/* ───────────────────────────── sections ───────────────────────────── */

function render() {
  const { meta, agent, comparison, actions, off_policy, drift, trail } = DATA;
  void off_policy;

  $("run-badge").textContent =
    `${meta.n_test_transactions} txns · seed ${meta.seed} · ${meta.n_executions} executions`;

  // hero stats
  $("stat-recovered").textContent = fmt.inr(agent.recovered_value);
  $("stat-delta").textContent = "+" + fmt.inr(agent.net_recovered_value - comparison.dumb_retry.net_recovered_value);
  $("stat-rate").textContent = fmt.pct(agent.recovery_rate_value);
  $("stat-touches").textContent = agent.touchpoints;

  // hero takeaways
  const take = $("hero-takeaways");
  take.innerHTML = "";
  const items = [
    ["good", `Beats dumb retry by <b>+${fmt.inr(agent.net_recovered_value - comparison.dumb_retry.net_recovered_value)} net</b> on the same 1,336 transactions.`],
    ["good", `Recovers most of the rule's value with <b>${agent.touchpoints} vs ${comparison.rule_retry.touchpoints} customer touches</b> (${Math.round((1 - agent.touchpoints / comparison.rule_retry.touchpoints) * 100)}% fewer).`],
    [agent.risk_blocked ? "warn" : "", agent.risk_blocked
      ? `Risk gate suppressed <b>${agent.risk_blocked} high-risk payments</b> (₹${fmt.inr(agent.risk_suppressed_value)} deliberately not pursued) — compliance before revenue, suppressed ≠ lost.`
      : `EV selection uses <b>${fmt.pct(agent.give_up_rate)} give-ups</b> — the stopping rule working, not a blanket retry.`],
    ["warn", `Trails the rule by ${fmt.inr(Math.abs(agent.net_recovered_value - comparison.rule_retry.net_recovered_value))} in one drifted segment — <b>quantified below</b>, fix = flagged retraining.`],
  ];
  for (const [cls, html] of items) {
    const d = el("div", "takeaway" + (cls ? " " + cls : ""));
    d.innerHTML = html;
    take.appendChild(d);
  }

  renderComparison(comparison);
  renderActions(actions);
  renderTrail(trail);
  renderExplainer(trail);
  renderDrift(drift);
  renderOffPolicy(off_policy);
  renderDemoNote();
}

/* ───────────────────────────── comparison ───────────────────────────── */

function renderComparison(comparison) {
  const table = $("comparison-table");
  const metricCols = [
    ["recovered_value", "Recovered ₹", "inr", "num"],
    ["recovery_rate_value", "Rate (value)", "pct", ""],
    ["attempts", "Attempts", "num", ""],
    ["touchpoints", "Touches", "num", ""],
    ["retry_cost", "Retry cost", "inr", ""],
    ["net_recovered_value", "Net recovered ₹", "inr", "num"],
    ["give_up_rate", "Give-up", "pct", ""],
  ];
  const order = ["do_nothing", "dumb_retry", "rule_retry", "recovery_agent"];
  const labels = {
    do_nothing: "Do Nothing", dumb_retry: "Dumb Retry (baseline)",
    rule_retry: "Rule-Based Smart Retry", recovery_agent: "AI Recovery Agent",
  };

  const thead = el("thead");
  const trh = el("tr");
  trh.appendChild(el("th"));
  for (const [key, label] of metricCols) {
    const th = el("th");
    th.textContent = label;
    trh.appendChild(th);
  }
  void metricCols;
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const name of order) {
    if (!comparison[name]) continue;
    const tr = el("tr");
    if (name === "recovery_agent") tr.className = "hl";
    const td0 = el("td");
    td0.innerHTML = labels[name] || name;
    tr.appendChild(td0);
    for (const [key, , kind, mood] of metricCols) {
      const td = el("td");
      const v = comparison[name][key];
      if (v === undefined || v === null) { td.textContent = "—"; }
      else {
        td.textContent = kind === "inr" ? fmt.inr(v) : kind === "pct" ? fmt.pct(v) : fmt.num(v);
        if (mood) td.className = mood;
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  // metric strip under the table (agent economics)
  const strip = $("agent-metrics");
  strip.innerHTML = "";
  const m = comparison.recovery_agent;
  for (const [label, val] of [
    ["Net recovered ₹", fmt.inr(m.net_recovered_value)],
    ["Recovery-to-bother ₹/touch", fmt.inr(m.recovered_value / m.touchpoints)],
    ["Risk-blocked", fmt.num(m.risk_blocked || 0)],
    ["Incremental net vs dumb", (m.net_recovered_value - comparison.dumb_retry.net_recovered_value >= 0 ? "+" : "") + fmt.inr(m.net_recovered_value - comparison.dumb_retry.net_recovered_value)],
  ]) {
    const c = el("div", "stat-card");
    const n = el("div", "stat-num"); n.textContent = val;
    const l = el("div", "stat-label"); l.textContent = label;
    c.append(n, l);
    strip.appendChild(c);
  }
}

/* ───────────────────────────── actions ───────────────────────────── */

function renderActions(actions) {
  const chart = $("actions-chart");
  chart.innerHTML = "";
  const rows = Object.entries(actions).sort((a, b) => b[1].count - a[1].count);
  const max = rows.length ? rows[0][1].count : 1;

  for (const [action, s] of rows) {
    const row = el("div", "bar-row");
    const name = el("div", "bar-name"); name.textContent = action;
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = Math.round((s.count / max) * 100) + "%";
    track.appendChild(fill);
    const val = el("div", "bar-value"); val.textContent = s.count;
    row.append(name, track, val);
    chart.appendChild(row);
  }

  // outcome cross-tab
  const tbody = el("tbody");
  for (const [action, s] of rows) {
    const tr = el("tr");
    const a = el("td"); a.textContent = action;
    const k = actionKind(action);
    a.innerHTML += ` <span class="badge ${k.cls}">${k.label}</span>`;
    const succ = el("td", "good"); succ.textContent = s.SUCCESS || 0;
    const fail = el("td"); fail.textContent = s.FAILED || 0;
    const rate = s.SUCCESS + s.FAILED ? (s.SUCCESS / (s.SUCCESS + s.FAILED)) : null;
    const r = el("td");
    r.textContent = rate === null ? "—" : fmt.pct(rate);
    r.className = rate !== null && rate > 0.5 ? "good" : "";
    tr.append(a, succ, fail, r);
    tbody.appendChild(tr);
  }
  const head = el("thead");
  const hr = el("tr");
  for (const h of ["Executed action", "Success", "Failed", "Success rate"]) {
    const th = el("th"); th.textContent = h; hr.appendChild(th);
  }
  head.appendChild(hr);
  const table = $("outcome-table");
  table.appendChild(head);
  table.appendChild(tbody);
}

/* ───────────────────────────── trail ───────────────────────────── */

function renderTrail(trail) {
  const reasonSel = $("f-reason");
  const actionSel = $("f-action");
  const reasons = [...new Set(trail.map((t) => t.reason).filter(Boolean))].sort();
  const actions = [...new Set(trail.map((t) => t.action))].sort();
  for (const r of reasons) {
    const o = el("option"); o.value = r; o.textContent = r;
    reasonSel.appendChild(o);
  }
  for (const a of actions) {
    const o = el("option"); o.value = a; o.textContent = a;
    actionSel.appendChild(o);
  }

  const apply = () => {
    const reason = reasonSel.value, action = actionSel.value, q = $("f-q").value.trim();
    const limit = Number($("f-limit").value);
    let rows = trail;
    if (reason) rows = rows.filter((t) => t.reason === reason);
    if (action) rows = rows.filter((t) => t.action === action);
    if (q) rows = rows.filter((t) => (t.txn_id || "").includes(q) || JSON.stringify(t).includes(q));

    const table = $("trail-table");
    table.innerHTML = "";
    const head = el("thead");
    const hr = el("tr");
    for (const h of ["Txn", "Amount", "Reason", "Action", "P(success)", "EV", "Risk", "Outcome"]) {
      const th = el("th"); th.textContent = h; hr.appendChild(th);
    }
    head.appendChild(hr);
    table.appendChild(head);

    const body = el("tbody");
    for (const t of rows.slice(0, limit)) {
      const tr = el("tr");
      const td0 = el("td"); td0.textContent = t.txn_id;
      const o = t.outcomes[0];
      const td1 = el("td"); td1.textContent = fmt.inr(t.amount);
      const td2 = el("td"); td2.textContent = t.reason || "—";
      const td3 = el("td"); td3.textContent = t.action || "give_up";
      const td4 = el("td"); td4.textContent = t.probability !== undefined ? fmt.pct(t.probability) : "—";
      const td5 = el("td"); td5.textContent = t.ev !== undefined ? fmt.inr2(t.ev) : "—";
      const td6 = el("td"); td6.textContent = t.risk_score.toFixed(2);
      const td7 = el("td");
      if (t.risk_blocked) { td7.textContent = "BLOCKED"; td7.className = "danger"; }
      else if (o) {
        td7.textContent = o.outcome;
        if (o.outcome === "SUCCESS") td7.className = "good";
        else if (o.outcome === "FAILED") td7.className = "danger";
        else td7.className = "warn";
      } else { td7.textContent = "no-op"; td7.className = "muted"; }
      tr.append(td0, td1, td2, td3, td4, td5, td6, td7);
      body.appendChild(tr);
    }
    table.appendChild(body);
    $("trail-count").textContent =
      `Showing ${Math.min(rows.length, limit)} of ${rows.length.toLocaleString("en-IN")} decisions`
      + (rows.length !== trail.length ? ` (filtered from ${trail.length.toLocaleString("en-IN")})` : "");
  };

  ["change", "input"].forEach((evt) => {
    reasonSel.addEventListener(evt, apply);
    actionSel.addEventListener(evt, apply);
    $("f-q").addEventListener(evt, apply);
    $("f-limit").addEventListener(evt, apply);
  });
  apply();
}

/* ───────────────────────────── explainer ───────────────────────────── */

function renderExplainer(trail) {
  const sel = $("e-txn");
  const shown = trail.slice(0, 400);
  for (const t of shown) {
    const o = el("option");
    o.value = t.txn_id;
    o.textContent = `${t.txn_id} — ${t.reason || "?"} — ${fmt.inr(t.amount)} → ${t.action || "give_up"}`;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => showExplain(shown.find((t) => t.txn_id === sel.value)));
  showExplain(shown[0]);
}

function showExplain(t) {
  const body = $("explain-body");
  if (!t) { body.innerHTML = "<p class='muted'>Select a transaction.</p>"; return; }
  body.innerHTML = "";

  const hero = el("div", "action-hero");
  const big = el("div", "big");
  big.textContent = (t.action || "give_up").replaceAll("_", " ") + (t.risk_blocked ? " — BLOCKED BY RISK GATE" : "");
  hero.appendChild(big);

  const metaLine = el("div");
  metaLine.innerHTML =
    `Chosen probability <b>${fmt.pct(t.probability)}</b> · ` +
    `EV <b>${fmt.inr2(t.ev)}</b> · cost ${fmt.inr2(t.cost)} · friction ${fmt.inr2(t.friction)} · ` +
    `risk score ${t.risk_score.toFixed(2)} · ${t.bank ? t.bank + " · " : ""}${t.payment_method || ""}`;
  hero.appendChild(metaLine);

  const reason = el("div", "explain-reason");
  reason.textContent = t.reasoning || "No explanation string recorded.";
  hero.appendChild(reason);

  const o = t.outcomes[0];
  if (o) {
    const badge = el("span", "badge " + (o.outcome === "SUCCESS" ? "good" : o.outcome === "FAILED" ? "danger" : ""));
    badge.textContent = o.outcome + (o.recovered_value ? ` · +${fmt.inr2(o.recovered_value)}` : "");
    hero.appendChild(badge);
    if (t.outcomes.length > 1) {
      for (const extra of t.outcomes.slice(1)) {
        const b = el("span", "badge");
        b.textContent = `attempt ${extra.attempt} → ${extra.outcome}`;
        hero.appendChild(b);
      }
    }
  }

  body.appendChild(hero);

  // grid: probabilities vs utilities
  const grid = el("div", "explain-grid");
  const probs = el("div", "explain-panel");
  probs.appendChild(Object.assign(el("h4"), { textContent: "Estimated P(success) per action" }));
  const maxP = Math.max(...Object.values(t.probabilities || {}), 1e-9);
  for (const [a, p] of Object.entries(t.probabilities || {})) {
    const row = el("div", "prob-row");
    const name = el("span"); name.textContent = a;
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = Math.round((p / maxP) * 100) + "%";
    track.appendChild(fill);
    const val = el("span"); val.textContent = fmt.pct(p);
    row.append(name, track, val);
    probs.appendChild(row);
  }

  const utils = el("div", "explain-panel");
  utils.appendChild(Object.assign(el("h4"), { textContent: "Expected utilities (₹-equivalent)" }));
  const maxU = Math.max(...Object.values(t.utilities || { 0: 0 }), 1e-9);
  for (const [a, u] of Object.entries(t.utilities || {})) {
    const row = el("div", "prob-row");
    const name = el("span"); name.textContent = a;
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = Math.round((Math.max(u, 0) / maxU) * 100) + "%";
    if (u <= 0) fill.style.background = "linear-gradient(90deg,#5a2340,var(--danger))";
    track.appendChild(fill);
    const val = el("span"); val.textContent = fmt.inr2(u);
    row.append(name, track, val);
    utils.appendChild(row);
  }
  grid.append(probs, utils);
  body.appendChild(grid);
}

/* ───────────────────────────── drift ───────────────────────────── */

function renderDrift(drift) {
  if (!drift) { $("drift-cards").innerHTML = "<p class='muted'>No drift data.</p>"; return; }
  const cards = $("drift-cards");
  cards.classList.add("drift-cards");
  cards.innerHTML = "";
  const mk = (period, n, p, rec, extra) => {
    const c = el("div", "drift-card");
    const pe = el("div", "period"); pe.textContent = period;
    const num = el("div", "num"); num.textContent = fmt.pct(p);
    const sub = el("div", "muted small");
    sub.textContent = `${n} transactions · mean expected ₹${fmt.inr(rec)}`;
    c.append(pe, num, extra || sub);
    return c;
  };
  cards.appendChild(mk(drift.pre_drift.period, drift.pre_drift.n_transactions,
    drift.pre_drift.mean_p_success, drift.pre_drift.mean_expected_recovery));
  const post = mk(drift.post_drift.period, drift.post_drift.n_transactions,
    drift.post_drift.mean_p_success, drift.post_drift.mean_expected_recovery);
  const lift = el("div", "drift-lift");
  lift.textContent = "+" + fmt.pct(drift.lift) + " lift";
  post.appendChild(lift);
  cards.appendChild(post);

  // monthly table
  const tb = $("drift-monthly");
  const head = el("thead");
  const hr = el("tr");
  for (const h of ["Month", "P(success)", "Transactions", "Value"]) {
    const th = el("th"); th.textContent = h; hr.appendChild(th);
  }
  head.appendChild(hr);
  tb.appendChild(head);
  const body = el("tbody");
  for (const m of drift.monthly_breakdown) {
    const tr = el("tr");
    for (const [k, f] of [["month", null], ["p_success_mean", "pct"], ["n_transactions", "num"], ["total_value", "inr"]]) {
      const td = el("td");
      td.textContent = f ? fmt[f](m[k]) : m[k];
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  tb.appendChild(body);
}

/* ───────────────────────────── off-policy ───────────────────────────── */

function renderOffPolicy(off) {
  const tb = $("offpolicy-table");
  const head = el("thead");
  const hr = el("tr");
  for (const h of ["Metric", "Value"]) { const th = el("th"); th.textContent = h; hr.appendChild(th); }
  head.appendChild(hr);
  tb.appendChild(head);
  const body = el("tbody");
  for (const [k, v] of Object.entries(off)) {
    const tr = el("tr");
    const a = el("td"); a.textContent = k;
    const b = el("td"); b.textContent = v;
    tr.append(a, b);
    body.appendChild(tr);
  }
  tb.appendChild(body);
}

/* ══════════════════ LIVE DEMO · payer <-> merchant ══════════════════ */

function initDemo() {
  const list = $("scenario-list");
  fetch("/api/scenarios")
    .then((r) => r.json())
    .then((sc) => {
      SCENARIOS = sc;
      list.innerHTML = "";
      for (const k of Object.keys(sc)) {
        const b = el("button", "scenario-chip");
        b.textContent = sc[k].label;
        b.dataset.key = k;
        b.onclick = () => selectScenario(k);
        list.appendChild(b);
      }
      selectScenario(Object.keys(sc)[0]);
    });
  $("pay-btn").onclick = pay;
}

function selectScenario(key) {
  CURRENT_SCENARIO = key;
  document.querySelectorAll(".scenario-chip").forEach((c) =>
    c.classList.toggle("active", c.dataset.key === key));
  if (SCENARIOS && SCENARIOS[key] && SCENARIOS[key].amount) {
    $("demo-amount").value = SCENARIOS[key].amount;
  }
  $("payer-stage").innerHTML =
    `<div class="txn-line dim">${SCENARIOS ? SCENARIOS[key].hook : ""}</div>`;
  $("merchant-stage").innerHTML =
    `<div class="txn-line dim">New failed payments will appear here as the payer pays.</div>`;
}

async function pay() {
  if (DEMO_LOCK) return;
  DEMO_LOCK = true;
  const btn = $("pay-btn");
  btn.disabled = true;
  btn.textContent = "Processing…";

  const key = CURRENT_SCENARIO || "server_busy";
  const sc = (SCENARIOS && SCENARIOS[key]) || {};
  const amount = Number($("demo-amount").value) || 1;
  const reason = (sc.reason || key).replaceAll("_", " ");

  const payer = $("payer-stage");
  payer.innerHTML = "";
  const t1 = el("div", "txn-line");
  t1.innerHTML = `Initiating payment of <b>${fmt.inr(amount)}</b>…`;
  payer.appendChild(t1);

  await sleep(850);
  const t2 = el("div", "txn-line");
  t2.innerHTML = `Payment <span class="fail">failed</span> — ${reason}.`;
  payer.appendChild(t2);
  await sleep(300);

  try {
    const res = await fetch("/api/demo/pay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: key, amount }),
    });
    const j = await res.json();
    if (!j.ok) throw new Error(j.error || "demo failed");
    await sleep(450);
    showAssurance(j.decision);
  } catch (e) {
    const t3 = el("div", "txn-line");
    t3.innerHTML = `<span class="fail">Engine error:</span> ${String(e).replaceAll("<", "&lt;")}`;
    payer.appendChild(t3);
  } finally {
    DEMO_LOCK = false;
    btn.disabled = false;
    btn.textContent = "Pay now";
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function showAssurance(decision) {
  const a = decision.assurance;
  const stage = $("merchant-stage");
  stage.innerHTML = "";
  $("merchant-head").textContent = "Recovery & assurance · live";

  // TWO-WAY exchange: the customer spots their SMS in the left (payer) pane
  // while the merchant sees the assurance in the right pane — both get a message.
  const payer = $("payer-stage");
  const gotMsg = decision.customer_message;   // string, or null for give_up/blocked
  const actor = el("div", "cx-caption");
  actor.innerHTML = gotMsg
    ? "<span class='cx-to'>You</span> &nbsp;— SMS/push from your payment provider:"
    : "You — no customer contact (silent follow-up / held).";
  payer.appendChild(actor);

  if (gotMsg) {
    const bubble = el("div", "cx-bubble payer-bubble");
    bubble.innerHTML =
      `<svg class="phone" viewBox="0 0 24 24" width="13" height="13" aria-hidden="true"><rect x="6" y="2" width="12" height="20" rx="2.5" fill="none" stroke="currentColor" stroke-width="1.6"/><line x1="10.5" y1="18.5" x2="13.5" y2="18.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg> ${escapeHtml(gotMsg)}`;
    payer.appendChild(bubble);
  } else {
    const quiet = el("div", "cx-bubble payer-bubble muted-bubble");
    quiet.textContent = "No message was sent — we don't contact you unless it's worth it.";
    payer.appendChild(quiet);
  }

  const mc = el("div", "cx-caption");
  mc.innerHTML = "<span class='cx-to'>Merchant</span> &nbsp;— message sent to you:";
  stage.appendChild(mc);

  // the merchant's delivered message/notification (symmetric with the payer's SMS)
  const mBubble = el("div", "cx-bubble merchant-bubble");
  mBubble.innerHTML =
    `<svg class="bell" viewBox="0 0 24 24" width="13" height="13" aria-hidden="true"><path d="M12 3a5.5 5.5 0 0 0-5.5 5.5v4.7L5 16v1h14v-1l-1.5-2.8V8.5A5.5 5.5 0 0 0 12 3zM9.5 19a2.5 2.5 0 0 0 5 0" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg> ${escapeHtml(a.message)}`;
  stage.appendChild(mBubble);

  const card = el("div", "assure-card os-" + a.outcome);
  const title = el("div", "assure-title");
  title.textContent = a.headline;
  const body = el("div", "assure-body");
  body.textContent = a.message;
  card.append(title, body);

  const benefit = el("div", "benefit-line");
  const c1 = el("span", "chip");
  c1.innerHTML = `Merchant recovers <b>${fmt.inr2(a.merchant_benefit)}</b> expected`;
  const c2 = el("span", "chip");
  c2.innerHTML = `Razorpay fee <b>${fmt.inr2(a.razorpay_fee)}</b>`;
  benefit.append(c1, c2);
  card.appendChild(benefit);
  stage.appendChild(card);

  // what happens next — a small timeline
  const tl = el("div", "timeline");
  const steps = [];
  const act = decision.action;
  const p = fmt.pct(decision.probability);
  const gain = fmt.inr2(a.merchant_benefit);
  const when = decision.retry_in_hours ? `~${Math.round(decision.retry_in_hours)}h` : "scheduled";
  if (act === "retry_soon") {
    steps.push(["done", "Payment failed", "just now"]);
    steps.push(["", "Silent auto-retry queued", when]);
    steps.push(["", `Expected to complete (${p}) · +${gain}`, "no customer contact"]);
  } else if (act === "retry_later") {
    steps.push(["done", "Payment failed", "just now"]);
    steps.push(["", "Retry queued for funding / off-peak window", when]);
    steps.push(["", `Expected to complete (${p}) · +${gain}`, "no customer contact"]);
  } else if (act === "switch_method") {
    steps.push(["done", "Payment failed", "just now"]);
    steps.push(["", "Alternative-instrument offer sent", "visible touch"]);
    steps.push(["", `Likely to complete (${p}) · +${gain}`, "if customer opts in"]);
  } else if (act === "send_link") {
    steps.push(["done", "Payment failed", "just now"]);
    steps.push(["", "Payment link sent to customer", "visible touch"]);
    steps.push(["", `Likely to complete (${p}) · +${gain}`, "customer pays at leisure"]);
  } else {
    steps.push(["done", "Payment failed", "just now"]);
    steps.push(["fail", decision.risk_blocked ? "Held for risk review — not failed" : "Stopped — not worth pursuing",
      decision.risk_blocked ? "manual compliance review" : "expected value < cost"]);
  }
  for (const [cls, text, meta] of steps) {
    const s = el("div", "tl-step" + (cls ? " " + cls : ""));
    const dot = el("div", "dot");
    const tx = el("div");
    const txt = el("div", "tl-text"); txt.textContent = text;
    const m = el("div", "tl-meta"); m.textContent = meta;
    tx.append(txt, m);
    s.append(dot, tx);
    tl.appendChild(s);
  }
  stage.appendChild(tl);
}

function renderDemoNote() {
  const list = $("demo-explainer-note");
  list.innerHTML = "";
  const items = [
    "The merchant is told <b>what happens next</b> in plain words — not a raw decline code. Their trust isn't broken, so the order isn't cancelled.",
    "The <b>payer is never spammed</b>: most recoveries are silent auto-retries; visible touches (switch offer, link) are used only when they're the best expected value.",
    "Every number here — probability, ₹ expected, retry window, risk verdict — is produced by the <b>same trained model and EV engine</b> that scored the 1,336 test transactions.",
    "The <b>risk gate runs first</b>: a flagged payment is held for manual review, not retried. Compliance before revenue, and the decision is recorded, not hidden.",
  ];
  for (const html of items) {
    const li = el("li"); li.innerHTML = html; list.appendChild(li);
  }
}

/* ══════════════════ MERCHANT FEED · your CSV ══════════════════ */

function buildUploadMarkdown(j) {
  const act = j.by_action || {};
  const L = [];
  L.push("# RetryNow — AI Recovery Decisions (your data)");
  L.push("");
  L.push(`**Transactions scored**: ${fmt.num(j.n_transactions)}  `);
  L.push(`**Failed value**: ${fmt.inr(j.potential_value)}  `);
  L.push("");
  L.push("## Per-action summary");
  L.push("");
  L.push("| Action | Count | ₹ value | Risk-blocked |");
  L.push("|---|---:|---:|---:|");
  for (const [a, m] of Object.entries(act)) {
    L.push(`| ${a.replaceAll("_", " ")} | ${fmt.num(m.count)} | ${fmt.inr(m.value)} | ${m.risk_blocked} |`);
  }
  L.push("");
  L.push("## Per-transaction decisions");
  L.push("");
  L.push("| txn | action | p(success) | EV ₹ | risk | reason |");
  L.push("|---|---:|---:|---:|---:|---|");
  for (const r of j.decisions || []) {
    L.push(`| ${r["txn_id"]} | ${r["action"].replaceAll("_", " ")} | ${(r["probability"] * 100).toFixed(0)}% | ${r["ev"].toFixed(0)} | ${r["risk_score"].toFixed(2)} | ${(r["reason"] || "").replace(/\|/g, "/")} |`);
  }
  L.push("");
  L.push("*Decision trail: recovery agent, expected-value engine, risk gate — RetryNow.*");
  return L.join("\n");
}

function jointBenefit(decisions) {
  let merchant = 0, fee = 0, will = 0, may = 0, wont = 0, blocked = 0;
  for (const d of decisions) {
    const a = d.assurance || {};
    merchant += a.merchant_benefit || 0;
    fee += a.razorpay_fee || 0;
    if (a.outcome === "will") will++;
    else if (a.outcome === "may") may++;
    else if (a.outcome === "won't") wont++;
    else if (a.outcome === "blocked") blocked++;
  }
  return { merchant, fee, will, may, wont, blocked };
}

function renderUpload(fileInput, progress, results) {
  const file = fileInput.files[0];
  if (!file) { alert("Choose a CSV file first."); return; }

  progress.hidden = false;
  progress.textContent = `Scoring ${file.name} …`;

  const data = new FormData();
  data.append("file", file);

  fetch("/api/upload", { method: "POST", body: data })
    .then(r => r.json().then(j => ({ ok: r.ok, j })))
    .then(({ ok, j }) => {
      progress.hidden = true;
      if (!ok) {
        results.hidden = false;
        results.innerHTML =
          `<h3 style="color:var(--danger)">Could not process your file</h3>
           <p class="table-footer">${j.error || "Unknown error"}${j.required ? "<br>Required: " + j.required.join(", ") : ""}</p>`;
        return;
      }
      renderUploadResults(results, j);
    })
    .catch(e => {
      progress.hidden = true;
      results.hidden = false;
      results.innerHTML = `<h3 style="color:var(--danger)">Upload failed</h3><p>${e}</p>`;
    });
}

function renderUploadResults(node, j) {
  node.hidden = false;
  node.innerHTML = "";
  const act = j.by_action || {};
  const decisions = j.decisions || [];
  const jb = jointBenefit(decisions);

  // headline: the joint-benefit story
  const story = el("div", "card");
  story.appendChild(Object.assign(el("h3"), { textContent: "The joint benefit — if this was real money" }));
  const feed = el("div", "feed-meta");
  const mk = (label, val, cls) => {
    const c = el("div", "stat-card" + (cls ? " " + cls : ""));
    const n = el("div", "stat-num"); n.textContent = val;
    const l = el("div", "stat-label"); l.textContent = label;
    c.append(n, l); return c;
  };
  feed.append(
    mk("Transactions scored", fmt.num(j.n_transactions)),
    mk("Failed value recovered (expected)", fmt.inr(jb.merchant), "mine"),
    mk("Razorpay fee earned on it", fmt.inr(jb.fee)),
    mk("Expected to recover", `${jb.will} will + ${jb.may} may`),
    mk("Risk-blocked (manual review)", fmt.num(jb.blocked)),
  );
  story.appendChild(feed);
  story.appendChild(Object.assign(el("p", "muted small"), {
    textContent: `Expected ₹ per row = amount × predicted P(success). Razorpay fee shown at a documented 2% proxy of the recovered value (demo framing, not a contract). The order is kept open and the customer isn't spammed — recovery happens on the single best action.`,
  }));
  node.appendChild(story);

  // benefit-by-row table
  node.appendChild(Object.assign(el("h3"), { textContent: "Row-by-row: the best action + the assurance sent to the merchant" }));
  const table = el("table", "data-table");
  const head = el("thead");
  const hr = el("tr");
  for (const h of ["Txn", "Amount", "Best action", "Outcome", "Assurance to merchant", "Merchant recovers", "Razorpay fee"]) {
    const th = el("th"); th.textContent = h; hr.appendChild(th);
  }
  head.appendChild(hr); table.appendChild(head);
  const body = el("tbody");
  for (const d of decisions) {
    const a = d.assurance || {};
    const tr = el("tr");
    const c0 = el("td"); c0.textContent = d.txn_id;
    const c1 = el("td"); c1.textContent = fmt.inr(d.amount);
    const c2 = el("td"); c2.textContent = (d.action || "").replaceAll("_", " ");
    const c3 = el("td"); c3.textContent = a.outcome || "—";
    c3.className = a.outcome === "will" ? "good" : a.outcome === "won't" || a.outcome === "blocked" ? "danger" : "warn";
    const c4 = el("td"); c4.textContent = a.headline || "—";
    const c5 = el("td"); c5.textContent = fmt.inr2(a.merchant_benefit || 0);
    const c6 = el("td"); c6.textContent = fmt.inr2(a.razorpay_fee || 0);
    tr.append(c0, c1, c2, c3, c4, c5, c6);
    body.appendChild(tr);
  }
  table.appendChild(body);
  const wrap = el("div", "table-scroll"); wrap.appendChild(table);
  node.appendChild(wrap);

  // per-action summary
  node.appendChild(Object.assign(el("h3"), { textContent: "Actions chosen across your batch" }));
  const at = el("table", "data-table");
  const ah = el("thead");
  const ar = el("tr");
  for (const h of ["Action", "Count", "₹ value", "Risk-blocked"]) {
    const th = el("th"); th.textContent = h; ar.appendChild(th);
  }
  ah.appendChild(ar); at.appendChild(ah);
  const ab = el("tbody");
  for (const [a, m] of Object.entries(act)) {
    const tr = el("tr");
    const c1 = el("td"); c1.textContent = a.replaceAll("_", " ");
    const c2 = el("td"); c2.textContent = fmt.num(m.count);
    const c3 = el("td"); c3.textContent = fmt.inr(m.value);
    const c4 = el("td"); c4.textContent = fmt.num(m.risk_blocked);
    tr.append(c1, c2, c3, c4); ab.appendChild(tr);
  }
  at.appendChild(ab);
  node.appendChild(at);

  // per-row editable trail
  const editable = el("pre", "upload-editable");
  editable.textContent = "txn_id,customer_id,amount,payment_method,failure_reason_code,risk_score,action,probability,ev,risk_blocked,reason\n" +
    decisions.map(r =>
      [r["txn_id"], r["customer_id"], r["amount"], r["payment_method"], r["failure_reason_code"],
       r["risk_score"], r["action"], r["probability"], r["ev"], r["risk_blocked"] ? "yes" : "no",
       '"' + (r["reason"] || "").replace(/"/g, "'") + '"'].join(",")).join("\n");
  node.appendChild(Object.assign(el("h3"), { textContent: "Per-transaction decisions (copyable)" }));
  node.appendChild(editable);

  // download buttons
  const dl = el("div", "upload-dl");
  const dlCsv = el("button", "btn");
  dlCsv.textContent = "Download decisions (CSV)";
  dlCsv.onclick = () => {
    const blob = new Blob([editable.textContent], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "retrynow_decisions.csv"; a.click();
  };
  const dlMd = el("button", "btn ghost");
  dlMd.textContent = "Download report (Markdown)";
  dlMd.onclick = () => {
    const md = buildUploadMarkdown(j);
    const blob = new Blob([md], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "retrynow_report.md"; a.click();
  };
  dl.append(dlCsv, dlMd);
  node.appendChild(dl);
}

function initUpload() {
  const form = $("upload-form");
  if (!form) return;
  const file = $("upload-file");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    renderUpload(file, $("upload-progress"), $("upload-results"));
  });
}

/* One click on the sample file: fetch it and run it through the same flow. */
function initSample() {
  const btn = $("sample-btn");
  if (!btn) return;
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = "Loading sample…";
    try {
      const res = await fetch("samples/merchant_failures.csv");
      const text = await res.text();
      const file = new File([text], "merchant_failures.csv", { type: "text/csv" });
      const dt = new DataTransfer();
      dt.items.add(file);
      const fileInput = $("upload-file");
      fileInput.files = dt.files;
      renderUpload(fileInput, $("upload-progress"), $("upload-results"));
    } catch (e) {
      alert("Could not load the sample file: " + e);
    } finally {
      btn.disabled = false;
      btn.textContent = "Try the sample file";
    }
  };
}

init();
