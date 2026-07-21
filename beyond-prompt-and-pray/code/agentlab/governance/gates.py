"""Governance gates.

The three default gates (Syntax, Policy, Plausibility) live in
agentlab.tools.executor because they are reused at the tool-execution
layer. This module re-exports them and adds governance-specific gates
that operate on broader state.
"""

from __future__ import annotations

from typing import Callable

from agentlab.tools.executor import (
    Gate,
    GateDecision,
    GateResult,
    PlausibilityGate,
    PolicyCheck,
    PolicyGate,
    SyntaxGate,
)

__all__ = [
    "Gate",
    "GateDecision",
    "GateResult",
    "PlausibilityGate",
    "PolicyCheck",
    "PolicyGate",
    "StateInvariantGate",
    "SyntaxGate",
]


class StateInvariantGate:
    """Escalate when the state violates a configured invariant.

    Invariants are callables taking AgentState and returning bool. A failing
    invariant produces an ESCALATE decision rather than DENY so a human can
    review unexpected state shapes.
    """

    name = "state_invariant"

    def __init__(self, invariants: list[Callable]) -> None:
        self._invariants = invariants

    def check(self, action, state, registry) -> GateResult:
        if state is None:
            return GateResult(GateDecision.ALLOW, self.name)
        for inv in self._invariants:
            try:
                ok = bool(inv(state))
            except Exception:
                ok = False
            if not ok:
                name = getattr(inv, "__name__", "<lambda>")
                return GateResult(GateDecision.ESCALATE, self.name, f"invariant violated: {name}")
        return GateResult(GateDecision.ALLOW, self.name)
