"""Risk gate: the interface where recovery meets risk controls.

**Scope note (important):** fraud/risk detection is OUT OF SCOPE for this
project. This module provides the *interface* a production risk system would
implement, plus a deterministic mock for the demo/hackathon. The mock scores
from the synthetic ``risk_score`` column in the generated data.

Semantics: the gate returns a per-action ``allowed`` verdict
(``check(txn, action) -> bool``). Recovery must NEVER override it:

  * allowed action  → proceeds to expected-value selection.
  * blocked action  → excluded from the argmax.
  * all blocked     → the payment resolves to ``give_up`` with
                      ``risk_blocked: true`` — a *suppressed* payment,
                      recorded in the audit trail. Not "unrecoverable",
                      and never silently dropped.

Production integration point: implement :class:`RiskGate` against the real
risk system; nothing else in the codebase changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.models.decision_engine import (RETRY_LATER, RETRY_SOON, SEND_LINK,
                                        SWITCH_METHOD)

log = logging.getLogger(__name__)

DEFAULT_BLOCK_THRESHOLD = 0.8
# Per-action thresholds: silent background retries may pass at higher risk than
# visible customer-facing actions (a link to a flagged customer is riskiest).
DEFAULT_PER_ACTION = {
    RETRY_SOON: 0.95,
    RETRY_LATER: 0.90,
    SWITCH_METHOD: 0.85,
    SEND_LINK: 0.85,
}


@dataclass(frozen=True)
class RiskVerdict:
    """One gate decision for one transaction."""

    allowed: dict[str, bool]      # {action: may_we_do_it}
    score: float                  # the risk score that drove the verdict
    source: str = "mock"          # which risk implementation produced this

    @property
    def blocked_all(self) -> bool:
        return not any(self.allowed.values())


class RiskGate(ABC):
    """Interface prepared for integration with a production risk system."""

    name = "abstract"

    @abstractmethod
    def check(self, txn: dict) -> RiskVerdict:
        """Return per-action allow verdicts for a failed transaction row.

        ``txn`` is the raw transaction context (must carry ``risk_score`` for
        the mock; a production implementation would consult the risk service).
        """

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"<RiskGate:{self.name}>"


class MockRiskGate(RiskGate):
    """Deterministic mock driven by the synthetic ``risk_score`` column.

    Not a fraud model — a stand-in that demonstrates the interface and the
    compliant-escalation behaviour (blocked → recorded give_up).
    """

    name = "mock"

    def __init__(self, block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
                 per_action: Optional[dict] = None):
        self.block_threshold = float(block_threshold)
        self.per_action = {**DEFAULT_PER_ACTION, **(per_action or {})}

    def check(self, txn: dict) -> RiskVerdict:
        score = float(txn.get("risk_score", 0.0))
        allowed = {
            action: score < self.per_action.get(action, self.block_threshold)
            for action in (RETRY_SOON, RETRY_LATER, SWITCH_METHOD, SEND_LINK)
        }
        return RiskVerdict(allowed=allowed, score=score, source=self.name)


class AllowAllRiskGate(RiskGate):
    """Dry-run gate (used by baseline policies / ablation: risk off)."""

    name = "allow_all"

    def check(self, txn: dict) -> RiskVerdict:
        return RiskVerdict(allowed={a: True for a in
                                    (RETRY_SOON, RETRY_LATER, SWITCH_METHOD, SEND_LINK)},
                           score=float(txn.get("risk_score", 0.0)), source=self.name)