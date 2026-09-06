#!/usr/bin/env python3
"""RetryNow web app — self-contained, zero-dependency server.

Serves a real frontend (web/index.html + assets) plus a JSON API computed from
the pipeline's real outputs (outputs/generated_data.csv, outputs/audit/*.jsonl,
outputs/evaluation_report.md). No framework, no CDN, no external assets —
`python web/server.py` is all you need.

Endpoints
    GET /                    → the website (index.html)
    GET /assets/<file>       → styles.css / app.js
    GET /api/data            → everything the UI needs in one payload:
                               comparison (4 policies), agent KPIs,
                               action distribution, decision trail,
                               explainer records, off-policy summary,
                               drift experiment, recovery-by-reason.
"""

from __future__ import annotations

import json
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Insert the project root on sys.path BEFORE importing any `src.*` module —
# otherwise `from src... import` (below) runs before this block and fails.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainer.merchant_assurance import merchant_assurance

OUTPUTS = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUTS / "evaluation_report.md"
AUDIT_PATH = OUTPUTS / "audit" / "agent_audit.jsonl"
DATA_PATH = OUTPUTS / "generated_data.csv"
WEB_DIR = Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 8000

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".csv": "text/csv",
}

ESTIMATOR_PATH = OUTPUTS / "estimator.joblib"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"


# ------------------------------------------------------------------ pipeline
def _load_audit() -> list[dict]:
    if not AUDIT_PATH.exists():
        return []
    records = []
    with open(AUDIT_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_data() -> dict:
    """DataFrame-light: row dicts for generated transactions (txn metadata)."""
    if not DATA_PATH.exists():
        return {"rows": [], "by_txn": {}}
    import pandas as pd
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    rows = df.to_dict(orient="records")
    by_txn = {r["txn_id"]: {k: _json_safe(v) for k, v in r.items()} for r in rows}
    return {"rows": rows, "by_txn": by_txn}


def _json_safe(v):
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, TypeError):
            return float(v)
    return v


def _markdown_tables(text: str) -> dict:
    """Extract markdown tables: {section_heading: {"header": [...], "rows": [...]}}.

    Handles pandas to_markdown quirks: the alignment row (``:---:``) and the
    empty first cell of the header row (the row-index placeholder) are dropped.
    """
    import re
    out: dict = {}
    current = "top"
    cells_buf: list[list[str]] = []

    def flush():
        if not cells_buf:               # heading-after-consumed-table: no-op
            return
        sep = re.compile(r"^:?-+:?$")
        data = [cells for cells in cells_buf
                if not all(sep.match(c) for c in cells)]
        if data:
            out[current] = {"header": data[0], "rows": data[1:]}
        cells_buf.clear()

    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.*)$", line.strip())
        if m:
            flush()
            current = m.group(1).strip()
            continue
        if line.strip().startswith("|"):
            cells_buf.append([c.strip() for c in
                              line.strip().strip("|").split("|")])
    flush()
    return out


@lru_cache(maxsize=1)
def _report_tables() -> dict:
    return _markdown_tables(REPORT_PATH.read_text(encoding="utf-8")
                            if REPORT_PATH.exists() else "")


@lru_cache(maxsize=1)
def build_payload() -> dict:
    """Everything the frontend needs — computed once per server process."""
    audit = _load_audit()

    if not audit:
        missing = [str(p.relative_to(PROJECT_ROOT)) for p in
                   (AUDIT_PATH, DATA_PATH, REPORT_PATH) if not p.exists()]
        return {"ok": False, "error": "outputs missing",
                "missing": missing,
                "hint": ("Run `PYTHONIOENCODING=utf-8 python -u "
                         "scripts/run_pipeline.py --mode all --seed 42` first.")}

    decide = [r for r in audit if r["event"] == "decide"]
    execute = [r for r in audit if r["event"] == "execute"]

    # agent-side KPIs straight from the audit trail
    total_failed = sum(r["amount"] for r in decide)
    outcome_by_txn = {r["txn_id"]: r["outcome"] for r in execute}
    recovered = sum(r.get("recovered_value", 0.0) for r in execute
                    if r["outcome"] == "SUCCESS")
    attempts = len(execute)
    touchpoints = sum(1 for r in decide
                      if isinstance(r.get("decision"), dict)
                      and r["decision"].get("action") in ("switch_method", "send_link"))
    give_ups = sum(1 for r in decide
                   if isinstance(r.get("decision"), dict)
                   and r["decision"].get("action") == "give_up")
    retry_cost_per_attempt = 2.0

    agent = {
        "recovered_value": round(recovered, 2),
        "recovery_rate_value": round(recovered / total_failed, 4) if total_failed else 0.0,
        "recovered_count": sum(1 for r in execute if r["outcome"] == "SUCCESS"),
        "attempts": attempts,
        "touchpoints": touchpoints,
        "retry_cost": round(attempts * retry_cost_per_attempt, 2),
        "net_recovered_value": round(recovered - attempts * retry_cost_per_attempt, 2),
        "give_up_rate": round(give_ups / len(decide), 4) if decide else 0.0,
        "risk_blocked": sum(1 for r in decide
                            if isinstance(r.get("decision"), dict)
                            and r["decision"].get("risk_blocked")),
        "risk_suppressed_value": round(sum(r["amount"] for r in decide
                                           if isinstance(r.get("decision"), dict)
                                           and r["decision"].get("risk_blocked")), 2),
    }

    # baselines from the report table (authoritative cross-policy comparison)
    tables = _report_tables()
    policies = {}
    for section in ("Policy Comparison (₹-centric metrics)",
                    "Policy Comparison"):
        t = tables.get(section)
        if t and "recovered_value" in t["header"]:
            for row in t["rows"]:
                rec = dict(zip(t["header"], row))
                name = next(iter(rec.values()), None)
                if not name:
                    continue
                try:
                    policies[name] = {k: float(v.replace(",", ""))
                                      for k, v in rec.items()
                                      if k != next(iter(rec))}
                except ValueError:
                    continue
            break
    ordered = ["do_nothing", "dumb_retry", "rule_retry", "recovery_agent"]
    comparison = {name: policies.get(name, {}) for name in ordered if name in policies}
    if "recovery_agent" in agent and "recovery_agent" not in comparison:
        comparison["recovery_agent"] = agent

    # recover-by-reason for each policy (report sections)
    by_reason: dict[str, dict[str, float]] = {}
    for pol in ("recovery_agent", "rule_retry", "dumb_retry"):
        t = tables.get(pol)
        rows = {}
        if t and "Reason" in t["header"]:
            i_reason = t["header"].index("Reason")
            i_rate = t["header"].index("Recovery Rate")
            for row in t["rows"]:
                try:
                    rows[row[i_reason]] = float(row[i_rate].rstrip("%")) / 100.0
                except (ValueError, IndexError):
                    continue
        by_reason[pol] = rows

    # off-policy (historical record) from report
    off_policy = {}
    t = tables.get("historical_record")
    if t and "Metric" in t["header"]:
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            if len(row) >= 2:
                off_policy[rec["Metric"]] = row[1]

    # drift experiment — computed live from the retained data (fast, authoritative)
    drift = None
    try:
        from src.experiments.drift import run_drift_experiment
        import yaml
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        drift = run_drift_experiment(
            _load_data_df(), drift_date=cfg["data"]["drift"]["drift_date"], seed=42)
        drift["monthly_breakdown"] = [_json_safe(m) for m in drift["monthly_breakdown"]]
    except Exception:
        drift = None

    # per-txn explainer index (audit decision joined to execute outcomes)
    meta = _load_data()["by_txn"]
    trail = []
    for r in decide:
        d = r.get("decision") if isinstance(r.get("decision"), dict) else {}
        m = meta.get(r["txn_id"], {})
        execs = [e for e in execute if e["txn_id"] == r["txn_id"]]
        trail.append({
            "txn_id": r["txn_id"],
            "amount": round(float(r["amount"]), 2),
            "ts": r.get("ts"),
            "reason": m.get("failure_reason_code"),
            "payment_method": m.get("payment_method"),
            "bank": m.get("bank"),
            "risk_score": round(float(r.get("risk_score", 0)), 4),
            "probabilities": {k: round(float(v), 4) for k, v in r.get("probabilities", {}).items()},
            "action": d.get("action"),
            "probability": d.get("probability"),
            "ev": d.get("ev"),
            "utilities": d.get("utilities"),
            "cost": d.get("cost"),
            "friction": d.get("friction"),
            "risk_blocked": bool(d.get("risk_blocked")),
            "reasoning": d.get("reason"),
            "outcomes": [{"attempt": e.get("attempt"), "action": e.get("action"),
                          "ts": e.get("ts"), "outcome": e.get("outcome"),
                          "recovered_value": e.get("recovered_value")}
                         for e in execs],
        })

    actions = {}
    for r in trail:
        a = r["action"] or "give_up"
        actions.setdefault(a, {"count": 0, "SUCCESS": 0, "FAILED": 0, "other": 0})
        actions[a]["count"] += 1
        outcome = r["outcomes"][0]["outcome"] if r["outcomes"] else None
        if outcome in ("SUCCESS", "FAILED"):
            actions[a][outcome] += 1
        else:
            actions[a]["other"] += 1

    return {
        "ok": True,
        "meta": {
            "generated": REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None,
            "n_decisions": len(decide),
            "n_executions": len(execute),
            "n_test_transactions": len(decide),
            "seed": 42,
            "run_command": "PYTHONIOENCODING=utf-8 python -u "
                           "scripts/run_pipeline.py --mode all --seed 42",
        },
        "agent": agent,
        "comparison": comparison,
        "actions": actions,
        "by_reason": by_reason,
        "off_policy": off_policy,
        "drift": drift,
        "trail": trail,
    }


def _load_data_df():
    import pandas as pd
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


# ------------------------------------------------------------ upload feature
# A merchant can upload their OWN failed-transactions CSV; we run the SAME
# trained estimator / EV engine / risk gate over it and return per-row
# decisions + a downloadable report. This turns RetryNow from a demo of
# synthetic data into a tool you can point at real transactions.

REQUIRED_COLS = ("amount", "payment_method", "failure_reason_code")
OPTIONAL_COLS = ("customer_id", "merchant_id", "timestamp", "bank",
                 "customer_instruments", "risk_score", "txn_id")

# ----------------------------------------------------------------- demo
# Preset scenarios for the live payer <-> merchant theater. Each maps to a real
# failure reason the engine understands; the demo injects that reason and shows
# what the recovery agent + merchant assurance WOULD do for it. Simulated, by
# design — the point is to make the policy legible, not to move real money.
DEMO_SCENARIOS = {
    "server_busy": {
        "label": "UPI server busy (evening peak)",
        "amount": 1200, "reason": "upi_server_busy", "method": "upi",
        "instruments": 3, "risk_score": 0.0, "merchant": "Swiggy",
        "expect": ["will"],
        "hook": "A transient outage — the flagship silent auto-retry case."},
    "insufficient_funds": {
        "label": "Insufficient funds (salary cycle)",
        "amount": 2000, "reason": "insufficient_funds", "method": "debit_card",
        "instruments": 2, "risk_score": 0.0, "merchant": "Myntra",
        "expect": ["will"],
        "hook": "Money lands after payday — the retry_later / timing case."},
    "card_declined": {
        "label": "Card declined (expired, small ticket)",
        "amount": 99, "reason": "card_expired", "method": "credit_card",
        "instruments": 1, "risk_score": 0.0, "merchant": "Netflix",
        "expect": ["won't"],
        "hook": "A dead instrument on a tiny ticket — the 'know when to stop' case."},
    "large_decline": {
        "label": "Issuer decline on a large order",
        "amount": 5000, "reason": "issuer_decline", "method": "credit_card",
        "instruments": 3, "risk_score": 0.0, "merchant": "MakeMyTrip",
        "expect": ["will", "may"],
        "hook": "Big-ticket decline — recovery is slower but still on the rails."},
    "high_risk": {
        "label": "High-risk transaction (flagged)",
        "amount": 2000, "reason": "bank_server_timeout", "method": "upi",
        "instruments": 2, "risk_score": 0.97, "merchant": "Paytm Mall",
        "expect": ["blocked"],
        "hook": "Risk gate suppresses it — compliance before revenue."},
}


def _demo_pay(amount: float, scenario_key: str) -> dict:
    """Run the real estimator + EV + risk gate on ONE simulated payment."""
    import io
    import yaml
    import numpy as np
    import pandas as pd

    sc = DEMO_SCENARIOS.get(scenario_key, DEMO_SCENARIOS["server_busy"])
    risk = float(sc.get("risk_score", 0.0))
    df = pd.DataFrame([{
        "timestamp": pd.Timestamp("2025-06-10 20:00:00"),
        "amount": amount,
        "payment_method": sc["method"],
        "failure_reason_code": sc["reason"],
        "customer_instruments": sc["instruments"],
        "risk_score": risk,
        # behaviour columns the feature janitor may read
        "was_retried_historically": 1,
        "retry_success_obs": None,          # censored: this payment has no outcome yet
        "amount_log": float(np.log1p(amount)),  # feature janitor may expect it
        "customer_id": "demo_cus",
        "merchant_id": "demo_mer",
        "bank": "bank_a",
        "merchant_vertical": "ecommerce",
        "hour_of_day": 20,
        "day_of_week": 1,
        "day_of_month": 10,
    }])
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    from src.models.decision_engine import Decision, DecisionEngine, GIVE_UP
    from src.risk.risk_gate import MockRiskGate
    engine = DecisionEngine(settings=cfg["decision_engine"])
    gate = MockRiskGate(block_threshold=cfg["risk_gate"].get("block_threshold"),
                        per_action=cfg["risk_gate"].get("per_action_thresholds"))

    import joblib
    if not ESTIMATOR_PATH.exists():
        return {"ok": False,
                "error": "No trained estimator — run the pipeline first."}
    estimator = joblib.load(ESTIMATOR_PATH)
    from src.features.feature_engineering import build_feature_matrix
    X = build_feature_matrix(df)
    row = df.iloc[0]
    for c in X.columns:
        if c not in row.index:
            df[c] = X[c].values
    row = df.iloc[0]

    p_actions = estimator.estimate(row)
    verdict = gate.check({"risk_score": risk})
    d = engine.decide(amount, p_actions, attempts=0,
                      current_method=str(row["payment_method"]),
                      risk_allowed=verdict.allowed)
    reason_key = sc["reason"].lower()
    hard_stop = any(term in reason_key for term in (
        "insufficient_funds", "insufficient_balance", "card_expired",
        "card_blocked", "account_blocked"))
    if not d.risk_blocked and d.action != GIVE_UP and (hard_stop or d.probability < 0.30):
        d = Decision(
            action=GIVE_UP,
            probability=0.0,
            attempts=d.attempts,
            reason=("customer-side funding or instrument issue requires a new "
                    "payment method" if hard_stop else
                    "predicted recovery is below the 30% merchant acceptance bar"),
            ev=0.0,
            utilities=d.utilities,
            risk_blocked=False,
        )
    decision = {
        "amount": round(amount, 2),
        "action": d.action,
        "probability": round(float(d.probability), 4),
        "ev": round(float(d.ev), 2),
        "risk_blocked": bool(d.risk_blocked),
        "retry_in_hours": d.retry_in_hours,
        "reason": d.reason,
        "failure_reason_code": sc["reason"],
        "assurance": merchant_assurance({**d.as_dict(), "amount": float(amount)}),
        # the *customer-facing* copy sent for this action — the other half of
        # the two-way exchange. None for give_up/blocked = genuinely no contact.
        "customer_message": _customer_message(d, amount, sc),
    }
    return {"ok": True, "scenario": sc, "decision": decision}


def _customer_message(d, amount: float, sc: dict) -> str | None:
    """Customer-facing SMS/push for this action (offline template, no LLM)."""
    from src.explainer.explainer import Explainer
    txn = {"amount": amount,
           "merchant_vertical": sc.get("merchant", "your merchant"),
           "payment_method": sc.get("method", "card"),
           "reason": sc.get("reason")}
    return Explainer(provider="template").customer_message(
        d.as_dict(), txn)


def split_multipart(body: bytes, delim: str) -> list[dict]:
    """Parse multipart/form-data into [{headers, body}] with a stdlib-only parser."""
    import re
    parts = []
    chunks = body.split(delim.encode("utf-8"))
    for chunk in chunks:
        if not chunk or chunk.startswith(b"--"):
            continue
        # strip leading CRLF after boundary
        chunk = chunk.lstrip(b"\r\n")
        hdr, _, body_bytes = chunk.partition(b"\r\n\r\n")
        if body_bytes is None:
            continue
        parts.append({"headers": hdr.decode("utf-8", errors="replace"),
                      "body": body_bytes})
    return parts


def _process_upload(df):
    """Run the full predictor → EV → risk-gate pipeline over an uploaded frame.

    Returns a dict of per-row decisions, aggregated stats, and a markdown
    report. ``df`` is the merchant's raw upload; missing optional columns get
    safe defaults so it works even with just amount + method + reason.
    """
    import yaml
    import numpy as np
    from src.models.decision_engine import DecisionEngine
    from src.risk.risk_gate import MockRiskGate
    from src.features.feature_engineering import build_feature_matrix

    if not ESTIMATOR_PATH.exists():
        return {"ok": False,
                "error": ("No trained estimator found. Run the pipeline first: "
                          "`PYTHONIOENCODING=utf-8 python -u scripts/run_pipeline.py"
                          " --mode all --seed 42`")}

    import joblib
    estimator = joblib.load(ESTIMATOR_PATH)

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    de_cfg = cfg["decision_engine"]
    engine = DecisionEngine(settings=de_cfg)
    gate = MockRiskGate(block_threshold=cfg["risk_gate"].get("block_threshold"),
                        per_action=cfg["risk_gate"].get("per_action_thresholds"))

    # --- tolerate missing optional columns -------------------------------
    for col, default in [("customer_id", lambda n: [f"cus{i}" for i in range(n)]),
                         ("merchant_id", lambda n: ["mer1"] * n),
                         ("txn_id", lambda n: [f"up{i}" for i in range(n)]),
                         ("timestamp", lambda n: [None] * n),
                         ("bank", lambda n: ["unknown"] * n),
                         ("customer_instruments", lambda n: [1] * n),
                         ("risk_score", lambda n: [0.0] * n)]:
        if col not in df.columns:
            df[col] = default(len(df))

    # --- feature matrix for ML path (engineering columns the model needs) --
    try:
        X = build_feature_matrix(df)
    except Exception:
        X = df

    rows = df.reset_index(drop=True)
    # attach engineered columns that the estimator may want (e.g. amount_log)
    for c in X.columns:
        if c not in rows.columns:
            rows[c] = X[c].values

    decisions = []
    for i, r in rows.iterrows():
        amount = float(r["amount"])
        p_actions = estimator.estimate(r)
        verdict = gate.check({"risk_score": float(r.get("risk_score", 0.0))})
        d = engine.decide(amount, p_actions,
                          attempts=0,
                          current_method=str(r.get("payment_method", "upi")),
                          risk_allowed=verdict.allowed)
        decisions.append({
            "txn_id": r.get("txn_id"),
            "customer_id": r.get("customer_id"),
            "amount": round(amount, 2),
            "payment_method": r.get("payment_method"),
            "failure_reason_code": r.get("failure_reason_code"),
            "risk_score": round(float(r.get("risk_score", 0.0)), 4),
            "action": d.action,
            "probability": round(float(d.probability), 4),
            "ev": round(float(d.ev), 2),
            "utilities": {k: round(float(v), 2)
                          for k, v in d.utilities.items()},
            "reason": d.reason,
            "risk_blocked": bool(d.risk_blocked),
            "retry_in_hours": d.retry_in_hours,
            "assurance": merchant_assurance({
                **d.as_dict(), "amount": float(r["amount"]),
                "failure_reason_code": r.get("failure_reason_code"),
            }),
        })

    # --- aggregate ----------------------------------------------
    by_action = {}
    for dec in decisions:
        a = dec["action"]
        by_action.setdefault(a, {"count": 0, "value": 0.0,
                                 "risk_blocked": 0})
        by_action[a]["count"] += 1
        by_action[a]["value"] += dec["amount"]
        if dec["risk_blocked"]:
            by_action[a]["risk_blocked"] += 1
    n = len(decisions)
    potential_value = sum(d["amount"] for d in decisions)
    return {
        "ok": True,
        "n_transactions": n,
        "potential_value": round(potential_value, 2),
        "by_action": by_action,
        "decisions": decisions,
        "actions_chosen": sorted(by_action.keys()),
    }


# ------------------------------------------------------------------ web layer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):          # quieter logs
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/api/data":
            payload = build_payload()
            self._send(200 if payload.get("ok") else 503,
                       json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return

        if path == "/api/scenarios":
            self._send(200, json.dumps(
                {k: {"label": v["label"], "hook": v["hook"],
                     "amount": v.get("amount", 1000), "expect": v.get("expect", [])}
                 for k, v in DEMO_SCENARIOS.items()},
                ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8")
            return

        if path in ("/", "/index.html"):
            page = (WEB_DIR / "index.html").read_bytes()
            self._send(200, page, MIME[".html"])
            return

        if path.startswith("/assets/"):
            asset = WEB_DIR / path[len("/assets/"):]
            if asset.is_file() and asset.resolve().is_relative_to(WEB_DIR.resolve()):
                self._send(200, asset.read_bytes(), MIME.get(asset.suffix, "application/octet-stream"))
                return

        if path.startswith("/samples/"):
            asset = PROJECT_ROOT / "samples" / path[len("/samples/"):]
            if asset.is_file() and asset.resolve().is_relative_to(
                    (PROJECT_ROOT / "samples").resolve()):
                self._send(200, asset.read_bytes(),
                           MIME.get(asset.suffix, "application/octet-stream"))
                return

        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_HEAD(self):                          # health checks / smoke tests
        self._send(200, b"", "text/plain")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/upload":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)

            # multipart/form-data (browser upload) vs raw CSV
            import io, re
            import pandas as pd
            body_bytes = body
            content_type = self.headers.get("Content-Type", "") or ""
            if "multipart/form-data" in content_type:
                boundary = re.search(r"boundary=([^;]+)", content_type)
                if boundary:
                    delim = "--" + boundary.group(1).strip().strip('"')
                    parts_list = split_multipart(body_bytes, delim)
                    csv_text = None
                    for part in parts_list:
                        if 'name="file"' in part.get("headers", ""):
                            csv_text = part["body"].decode("utf-8", errors="replace")
                    if csv_text is None:
                        # fall back to last split part
                        text = body_bytes.decode("utf-8", errors="replace")
                        blobs = text.split(delim)
                        csv_text = blobs[-1] if blobs else text
                    csv_text = csv_text.strip()
                else:
                    text = body_bytes.decode("utf-8", errors="replace")
                    csv_text = text.split("\n\n", 1)[-1].strip() if "\n\n" in text \
                        else text.strip()
            else:
                csv_text = body_bytes.decode("utf-8", errors="replace").strip()

            try:
                df = pd.read_csv(io.StringIO(csv_text))
            except Exception as e:
                self._send(400, json.dumps({"ok": False,
                                            "error": f"Could not parse CSV: {e}"},
                                           ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return

            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                self._send(400, json.dumps(
                    {"ok": False,
                     "error": f"Missing required column(s): {', '.join(missing)}",
                     "required": list(REQUIRED_COLS),
                     "optional": list(OPTIONAL_COLS)},
                    ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8")
                return

            result = _process_upload(df)
            if not result.get("ok"):
                self._send(503, json.dumps(result, ensure_ascii=False)
                           .encode("utf-8"), "application/json; charset=utf-8")
                return
            self._send(200, json.dumps(result, ensure_ascii=False)
                       .encode("utf-8"), "application/json; charset=utf-8")
            return

        if path == "/api/demo/pay":
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                scenario = body.get("scenario", "server_busy")
                amount = body.get("amount")
                if amount is None:
                    amount = DEMO_SCENARIOS.get(scenario, DEMO_SCENARIOS["server_busy"]).get("amount", 1000)
                amount = float(amount)
            except Exception:
                amount, scenario = DEMO_SCENARIOS["server_busy"].get("amount", 1000), "server_busy"
            result = _demo_pay(amount, scenario)
            self._send(200 if result.get("ok") else 503,
                       json.dumps(result, ensure_ascii=False)
                       .encode("utf-8"), "application/json; charset=utf-8")
            return

        self._send(404, b"Not found", "text/plain; charset=utf-8")


def main():
    if not AUDIT_PATH.exists():
        print("⚠  No outputs found. Run the pipeline first:")
        print("   PYTHONIOENCODING=utf-8 python -u scripts/run_pipeline.py --mode all --seed 42")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"🚀 RetryNow web app → http://{HOST}:{PORT}")
    print("   Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()