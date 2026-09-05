"""Decision engine: expected-value selection of the single next-best action.

For a failed payment we hold per-action recoverability estimates
(``p_actions``: retry_soon, retry_later, switch_method, send_link) and compute,
for every candidate:

    EV(action) = P(success | action) × amount − retry_cost − friction(action)

  * ``retry_cost``      — fixed ₹-cost of executing one attempt (network call,
                          bank API, SMS for links; config: retry_cost_per_attempt).
  * ``friction``        — customer-burden proxy in ₹: silent touches (retry_soon,
                          retry_later) are cheaper than visible ones (switch
                          offers, payment links); config: friction_cost_per_touch_*.
  * ``give_up``         — EV 0. It is a real candidate in the argmax, so the
                          engine "gives up" whenever every action's EV ≤ 0 —
                          no forced retries, no threshold gymnastics.

Constraints, in order: risk gate (blocked actions are excluded from the argmax;
a fully-blocked payment resolves to a *risk-suppressed* give_up, recorded) and
the attempt budget (max_attempts_per_payment → give_up). A `min_p_floor` below
which a probability is noise and its EV not trusted also prunes candidates.

Everything is config-driven (``decision_engine`` in configs/config.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Actions — canonical string constants used across the codebase.
RETRY_SOON = "retry_soon"
RETRY_LATER = "retry_later"
SWITCH_METHOD = "switch_method"
SEND_LINK = "send_link"
GIVE_UP = "give_up"

ALL_ACTIONS = [RETRY_SOON, RETRY_LATER, SWITCH_METHOD, SEND_LINK, GIVE_UP]

# Mirrors configs/config.yaml -> decision_engine (kept in code so the engine is
# usable standalone, e.g. in tests, without a yaml file on disk).
DEFAULTS = {
    "retry_soon_hours": 2,
    "retry_later_hours": 72,
    "retry_cost_per_attempt": 2.0,
    "friction_cost_per_touch_silent": 15.0,
    "friction_cost_per_touch_visible": 25.0,
    "min_p_floor": 0.03,
    "max_attempts_per_payment": 3,
    "max_retries_per_customer_day": 5,
}

# Friction tier per action: silent (background retries) vs visible (asks the
# customer to do something).
_FRICTION_TIER = {
    RETRY_SOON: "silent",
    RETRY_LATER: "silent",
    SWITCH_METHOD: "visible",
    SEND_LINK: "visible",
    GIVE_UP: None,
}


def _as_dict(cfg) -> dict:
    """Accept a yaml ``Config`` object, a plain dict, or None → defaults dict."""
    if cfg is None:
        return dict(DEFAULTS)
    if isinstance(cfg, dict):
        return {**DEFAULTS, **cfg}
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        try:
            out[key] = getattr(cfg, key)
        except AttributeError:
            pass
    return out


@dataclass
class Decision:
    """One next-best action for one failed payment, with its economics."""

    action: str                                            # one of ALL_ACTIONS
    probability: float                                     # P(success) that drove this
    attempts: int = 0                                      # attempts already spent
    retry_in_hours: Optional[float] = None                 # delay for retry_* actions
    target_method: Optional[str] = None                    # method for switch_method
    reason: str = field(default="", repr=False)            # human explanation
    # economics + audit fields (stable keys in as_dict, additive only)
    ev: float = 0.0                                        # EV of the chosen action
    utilities: dict = field(default_factory=dict, repr=False)  # {action: EV}
    cost: float = 0.0                                      # execution cost of the action
    friction: float = 0.0                                  # friction cost of the action
    risk_blocked: bool = False                             # gate suppressed this payment

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "probability": round(float(self.probability), 4),
            "attempts": self.attempts,
            "retry_in_hours": self.retry_in_hours,
            "target_method": self.target_method,
            "reason": self.reason,
            "ev": round(float(self.ev), 2),
            "utilities": {k: round(float(v), 2) for k, v in self.utilities.items()},
            "cost": round(float(self.cost), 2),
            "friction": round(float(self.friction), 2),
            "risk_blocked": self.risk_blocked,
        }


class DecisionEngine:
    """Stateless-by-design policy: all knobs live on ``self.settings``."""

    def __init__(self, settings=None):
        self.settings = _as_dict(settings)

    # ------------------------------------------------------------------ core
    def expected_utility(self, p: float, amount: float, action: str) -> float:
        """EV(action) = P·value − retry_cost − friction(action)."""
        s = self.settings
        if action == GIVE_UP:
            return 0.0
        friction = 0.0
        tier = _FRICTION_TIER.get(action)
        if tier == "silent":
            friction = s["friction_cost_per_touch_silent"]
        elif tier == "visible":
            friction = s["friction_cost_per_touch_visible"]
        return float(np.clip(p, 0.0, 1.0)) * float(amount) \
            - s["retry_cost_per_attempt"] - friction

    def decide(self, amount: float,
               p_actions: dict[str, float],
               attempts: int = 0,
               current_method: Optional[str] = None,
               risk_allowed: Optional[dict[str, bool]] = None) -> Decision:
        """Select the action with the highest expected value.

        ``p_actions``: {action: P(success)} for the four candidates (give_up has
        EV 0 by construction). ``risk_allowed``: {action: bool} from the risk
        gate; blocked actions are excluded (recorded as risk_blocked).
        """
        s = self.settings
        p_actions = {a: float(np.clip(p_actions.get(a, 0.0), 0.0, 1.0))
                     for a in (RETRY_SOON, RETRY_LATER, SWITCH_METHOD, SEND_LINK)}

        # attempt budget — the hard stopping rule
        if attempts >= s["max_attempts_per_payment"]:
            return Decision(GIVE_UP, 0.0, attempts,
                            reason="attempt budget exhausted "
                                   f"({s['max_attempts_per_payment']})")

        # risk gate: drop blocked actions from consideration
        allowed = {a: True for a in p_actions}
        if risk_allowed:
            allowed.update(risk_allowed)
        any_blocked = any(not allowed[a] for a in p_actions)
        p_actions = {a: p for a, p in p_actions.items() if allowed.get(a, True)}

        # min_p_floor: prune candidates whose probability is noise
        p_actions = {a: p for a, p in p_actions.items() if p >= s["min_p_floor"]}

        # argmax over candidate EV, with give_up (EV=0) always a candidate
        utilities = {a: self.expected_utility(p, amount, a)
                     for a, p in p_actions.items()}
        utilities[GIVE_UP] = 0.0
        best = max(utilities, key=utilities.get)

        if best == GIVE_UP:
            if any_blocked and not p_actions:
                reason = ("risk gate blocked all recovery actions — "
                          "payment suppressed, not unrecoverable")
                return Decision(GIVE_UP, 0.0, attempts,
                                reason=reason, utilities=utilities,
                                risk_blocked=True)
            if not p_actions:
                return Decision(GIVE_UP, 0.0, attempts,
                                reason="all actions below the min_p_floor "
                                       f"({s['min_p_floor']})",
                                utilities=utilities)
            return Decision(GIVE_UP, 0.0, attempts,
                            reason="no action has positive expected value",
                            utilities=utilities)

        p_chosen = p_actions[best]
        friction = (s["friction_cost_per_touch_silent"]
                    if _FRICTION_TIER[best] == "silent"
                    else s["friction_cost_per_touch_visible"])
        target_method = None
        if best == SWITCH_METHOD:
            target_method = self._best_alt_target(p_actions, current_method)
        retry_in = (s["retry_soon_hours"] if best == RETRY_SOON
                    else s["retry_later_hours"] if best == RETRY_LATER else None)
        return Decision(
            best, p_chosen, attempts,
            retry_in_hours=retry_in, target_method=target_method,
            reason=self._reason(best, p_chosen, amount, current_method,
                                target_method, retry_in),
            ev=utilities[best], utilities=utilities,
            cost=0.0 if best == GIVE_UP else s["retry_cost_per_attempt"],
            friction=friction, risk_blocked=False,
        )

    @staticmethod
    def _best_alt_target(p_actions: dict[str, float],
                         current_method: Optional[str]) -> Optional[str]:
        """Best 'other method' for the switch decision (stored on the decision).

        The engine itself decides *that* switching is optimal; the agent layer
        supplies the actual method-level probabilities via ``choose_alternative``.
        """
        return None  # resolved by the agent with real instrument scores

    def _reason(self, action, p, amount, current_method, target_method, retry_in) -> str:
        if action == RETRY_SOON:
            return (f"transient failure with high predicted success "
                    f"({p:.0%}) — retry soon")
        if action == RETRY_LATER:
            return (f"recoverable with time (e.g. funding issues around salary "
                    f"cycle, {p:.0%}) — retry later")
        if action == SWITCH_METHOD:
            return (f"current instrument sticky ({p:.0%}) — switch to an "
                    f"alternative with better expected value")
        if action == SEND_LINK:
            return (f"below the retry bar but not hopeless ({p:.0%}) — a "
                    f"payment link lets the customer pay at their convenience")
        return f"give up ({action})"

    # --------------------------------------------------------------- helpers
    @staticmethod
    def choose_alternative(probabilities: dict[str, float],
                           exclude: str | None = None) -> tuple[str, float]:
        """Return ``(method, p)`` of the highest-scoring alternative method."""
        candidates = {m: p for m, p in probabilities.items() if m != exclude}
        if not candidates:
            return None, 0.0
        method = max(candidates, key=candidates.get)
        return method, candidates[method]

    def decide_batch(self, df: pd.DataFrame,
                     p_col: str = "p_success",
                     p_alt_col: str | None = "p_alt",
                     attempts_col: str | None = "attempts",
                     method_col: str | None = "payment_method",
                     alt_method_col: str | None = "best_alt_method",
                     amount_col: str = "amount",
                     risk_col: str | None = None,
                     ) -> pd.DataFrame:
        """Vectorized convenience: apply :meth:`decide` row-wise to a frame.

        Expected columns: ``p_col`` and ``amount_col`` (required).  When
        ``risk_col`` is given, rows with risk above the gate's block threshold
        are risk-blocked per action via ``risk_allowed``.  Returns a copy of the
        input (minus helper probability columns) plus a ``decision`` dict column.
        """
        if p_col not in df.columns:
            raise KeyError(f"missing probability column '{p_col}'")
        if amount_col not in df.columns:
            raise KeyError(f"missing amount column '{amount_col}'")
        out = df.copy()
        rows = []
        for idx, row in out.iterrows():
            r_allowed = None
            if risk_col and risk_col in out.columns:
                gate = _mock_gate_lookup(row[risk_col])
                r_allowed = {a: gate.get(a, True) for a in
                             (RETRY_SOON, RETRY_LATER, SWITCH_METHOD, SEND_LINK)}
            rows.append(self.decide(
                row[amount_col],
                {RETRY_SOON: row[p_col],
                 RETRY_LATER: row[p_col] * 0.7,
                 SWITCH_METHOD: row.get(p_alt_col, 0.0) if p_alt_col else 0.0,
                 SEND_LINK: row[p_col] * 0.5},
                attempts=int(row.get(attempts_col, 0)) if attempts_col else 0,
                current_method=row.get(method_col) if method_col else None,
                risk_allowed=r_allowed,
            ))
        out["decision"] = [d.as_dict() for d in rows]
        drop = [c for c in (p_alt_col, attempts_col, alt_method_col) if c]
        return out.drop(columns=[c for c in drop if c in out.columns])

    def action_recovery_value(self, decisions: list[Decision]) -> dict[str, int]:
        """Count actions for attribution."""
        counts = {a: 0 for a in ALL_ACTIONS}
        for d in decisions:
            counts[d.action] += 1
        return counts


def _mock_gate_lookup(risk_score: float) -> dict[str, bool]:
    """Tiny local default for decide_batch; the real gate lives in src/risk.

    Thresholds mirror ``risk_gate.block_threshold`` (0.8) so batch decisions and
    agent-run decisions agree in the demo.
    """
    blocked = float(risk_score) >= 0.8
    return {a: not blocked for a in (RETRY_SOON, RETRY_LATER, SWITCH_METHOD, SEND_LINK)}


def action_priority(action: str) -> int:
    """Ranking for reporting: higher = more aggressive recovery effort."""
    order = {RETRY_SOON: 0, RETRY_LATER: 1, SWITCH_METHOD: 2, SEND_LINK: 3, GIVE_UP: 4}
    return order.get(action, 99)