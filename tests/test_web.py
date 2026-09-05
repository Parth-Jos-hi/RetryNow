"""Tests for the standalone web app (web/server.py).

The app has no framework dependencies: a stdlib http.server serving static
assets + a JSON API computed from the pipeline's real outputs. These tests
(1) validate the API payload structure/semantics against the audit trail and
(2) smoke-test the HTTP layer on an ephemeral port.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web import server  # noqa: E402


@pytest.fixture(scope="module")
def payload():
    return server.build_payload()


def test_payload_ok(payload):
    assert payload["ok"] is True


def test_metadata(payload):
    meta = payload["meta"]
    assert meta["n_decisions"] == meta["n_test_transactions"] > 0
    assert meta["n_executions"] > 0


def test_comparison_has_four_policies(payload):
    comp = payload["comparison"]
    assert set(comp) == {"do_nothing", "dumb_retry", "rule_retry", "recovery_agent"}
    assert comp["do_nothing"]["recovered_value"] == 0.0
    assert comp["recovery_agent"]["recovered_value"] > comp["dumb_retry"]["recovered_value"]


def test_agent_metrics_agree_with_trail(payload):
    """Agent KPIs derived from the audit trail must reconcile."""
    from web import server as srv
    audit = srv._load_audit()
    decide = [r for r in audit if r["event"] == "decide"]
    execute = [r for r in audit if r["event"] == "execute"]
    total = sum(r["amount"] for r in decide)
    recovered = sum(r.get("recovered_value", 0.0) for r in execute
                    if r["outcome"] == "SUCCESS")
    agent = payload["agent"]
    assert agent["recovered_value"] == round(recovered, 2)
    assert len(payload["trail"]) == len(decide)


def test_actions_cover_actions(payload):
    acts = payload["actions"]
    assert "give_up" in acts and "retry_soon" in acts
    total = sum(v["count"] for v in acts.values())
    assert total == payload["meta"]["n_decisions"]


def test_off_policy_present(payload):
    op = payload["off_policy"]
    assert "Observed recovered value" in op
    assert "Censored rows" in op


@pytest.fixture(scope="module")
def http_base():
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def test_http_index(http_base):
    import urllib.request
    with urllib.request.urlopen(http_base + "/", timeout=5) as r:
        assert r.status == 200
        body = r.read().decode("utf-8")
        assert "RetryNow" in body


def test_http_assets(http_base):
    import urllib.request
    for asset in ("styles.css", "app.js"):
        with urllib.request.urlopen(f"{http_base}/assets/{asset}", timeout=5) as r:
            assert r.status == 200
            assert r.read()


def test_http_api(http_base):
    import urllib.request
    with urllib.request.urlopen(http_base + "/api/data", timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
        assert data["ok"] is True
        assert data["meta"]["n_decisions"] > 0


def test_http_404_and_traversal_guard(http_base):
    import urllib.error
    import urllib.request
    for path in ("/nope", "/assets/../server.py", "/assets/../../configs/config.yaml"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(http_base + path, timeout=5)
        assert exc.value.code == 404


def test_http_upload(http_base):
    """Upload a small failed-transactions CSV → per-row decisions from the AI."""
    import urllib.request
    if not server.ESTIMATOR_PATH.exists():
        pytest.skip("no trained estimator (run the pipeline first)")

    csv = ("amount,payment_method,failure_reason_code,customer_instruments\n"
           "1200.50,upi,insufficient_funds,3\n"
           "875,debit_card,bank_server_timeout,2\n")
    req = urllib.request.Request(http_base + "/api/upload", data=csv.encode("utf-8"),
                                 method="POST")
    req.add_header("Content-Type", "text/csv")
    with urllib.request.urlopen(req, timeout=60) as r:
        assert r.status == 200
        data = json.loads(r.read().decode("utf-8"))
    assert data["ok"] is True
    assert data["n_transactions"] == 2
    assert len(data["decisions"]) == 2
    for dec in data["decisions"]:
        assert dec["action"] in ("retry_soon", "retry_later", "switch_method",
                                 "send_link", "give_up")
        assert 0.0 <= dec["probability"] <= 1.0