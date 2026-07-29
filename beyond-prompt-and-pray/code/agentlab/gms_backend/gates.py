"""GMSPlausibilityGate: a Gate that uses GMS geodesic scoring.

Drop-in replacement for Chapter 6's placeholder `PlausibilityGate`. The
heuristic gate from Ch 6 just rejected oversized arguments. This gate
asks the GMS substrate: *is this (context, tool_name) pair geometrically
plausible given what the store knows?* Tool calls scoring above the
calibrated threshold are denied.

The adapter does not import the GMS library at module load. The caller
constructs a GMS store and passes it in. Tests pass a mock store with a
``score_triple`` method.
"""

from __future__ import annotations

from typing import Any, Protocol

from agentlab.core.action import ToolCall
from agentlab.core.state import AgentState
from agentlab.tools.executor import GateDecision, GateResult
from agentlab.tools.registry import ToolRegistry


class _ScoreTripleStore(Protocol):
    """Minimal protocol the gate needs from a GMS-shaped store."""

    def score_triple(self, head: str, relation: str, tail: str) -> float | None: ...


class GMSPlausibilityGate:
    """Reject tool calls whose (context, tool) pair scores above theta.

    Parameters
    ----------
    store : object with `score_triple(head, relation, tail) -> float | None`
        A GMS-shaped store. In production this is a `GMSExpertStore` from
        the `gms`/`docgms` library. In tests it is a mock.
    theta : float
        Maximum admissible geodesic distance. Calibrated per domain in
        Appendix C; the GMS library's default is 1.5.
    context : str
        The context label for the plausibility check. Real systems pass
        the agent's current phase (`"training"`, `"diagnosis"`, etc.).
    relation : str
        The relation used to score. Defaults to `"should_call"`. The store
        must have been trained on triples using this relation.
    on_missing : {"allow", "deny", "escalate"}
        Behavior when the store returns None (no information). Default is
        ``"allow"`` (benefit of the doubt).
    """

    name = "gms_plausibility"

    def __init__(
        self,
        store: _ScoreTripleStore,
        theta: float = 1.5,
        context: str | Any = "default",
        relation: str = "should_call",
        on_missing: str = "allow",
        tool_node_map: dict[str, str] | None = None,
    ) -> None:
        if on_missing not in ("allow", "deny", "escalate"):
            raise ValueError(f"on_missing must be allow/deny/escalate, got {on_missing!r}")
        self._store = store
        self._theta = float(theta)
        # context may be a fixed string or a callable(action, state) -> str, so a
        # fixed-workflow agent can score each transition (prev_step -> tool).
        self._context = context
        self._relation = relation
        self._on_missing = on_missing
        self._tool_node_map = tool_node_map or {}

    def check(
        self,
        action: ToolCall,
        state: AgentState | None,
        registry: ToolRegistry,
    ) -> GateResult:
        """Score the (context, relation, tool) triple and deny above theta.

        Args:
            action: The proposed tool call; non-tool-call actions are allowed.
            state: The current state, passed to a callable context if configured.
            registry: The tool registry (unused; present for the Gate interface).

        Returns:
            DENY when the store's score exceeds theta, ALLOW when at or below it,
            and the configured on_missing decision when the store returns None.
        """
        if action.kind != "tool_call":
            return GateResult(GateDecision.ALLOW, self.name)
        context = self._context(action, state) if callable(self._context) else self._context
        tool_node = self._tool_node_map.get(action.tool_name, action.tool_name)
        score = self._store.score_triple(context, self._relation, tool_node)
        if score is None:
            return self._missing_result()
        if score > self._theta:
            return GateResult(
                GateDecision.DENY,
                self.name,
                f"plausibility {score:.3f} exceeds theta={self._theta:.3f} "
                f"for ({context!r}, {self._relation!r}, {tool_node!r})",
            )
        return GateResult(GateDecision.ALLOW, self.name, f"plausibility {score:.3f}")

    def _missing_result(self) -> GateResult:
        reason = (
            f"store has no plausibility data for ({self._context!r}, "
            f"{self._relation!r}, ...)"
        )
        if self._on_missing == "allow":
            return GateResult(GateDecision.ALLOW, self.name, reason)
        if self._on_missing == "deny":
            return GateResult(GateDecision.DENY, self.name, reason)
        return GateResult(GateDecision.ESCALATE, self.name, reason)
