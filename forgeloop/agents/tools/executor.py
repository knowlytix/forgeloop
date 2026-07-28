"""GovernedToolExecutor: runs gates, then calls the tool.

Three default gates: syntax (schema validation), policy (pluggable rules)
and plausibility (argument-size sanity). Chapter 12 builds richer gates
that conform to the same protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

from forgeloop.agents.core.action import ToolCall
from forgeloop.agents.core.state import AgentState
from forgeloop.agents.tools.registry import ToolRegistry


class GateDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class GateResult:
    """Outcome of a single gate check.

    Attributes:
        decision: Whether the gate allows, denies or escalates the call.
        gate_name: Name of the gate that produced the result.
        reason: Explanation for a deny or escalate decision.
    """

    decision: GateDecision
    gate_name: str
    reason: str = ""


@dataclass
class ToolHooks:
    """Observability/control callbacks around tool execution.

    Mirrors the knowlytix ``HarnessHooks`` contract so an external testing
    gateway (e.g. ``knowlytix.harness.testing.ToolGateway``) can install on a
    ``GovernedToolExecutor`` and intercept tool calls for fault injection. All
    hooks default to ``None`` -- when unset the executor behaves exactly as
    before, so this is purely additive.

      * ``before_tool(name, args) -> bool``: return ``False`` to block the call
        (the executor records it as a failed tool result --- an injected fault).
      * ``intercept_tool(name, args) -> tuple[bool, Any]``: return
        ``(True, response)`` to bypass real execution and substitute ``response``
        (a mock/stale result); ``(False, None)`` to run the tool normally.
      * ``on_tool_call`` / ``on_tool_result``: observability only.

    Attributes:
        on_tool_call: Observability callback invoked with ``(name, args)`` before
            the tool runs.
        before_tool: Predicate on ``(name, args)``; returning ``False`` blocks the
            call and records an injected fault.
        intercept_tool: Callback on ``(name, args)`` returning ``(intercepted,
            response)`` to substitute a response and skip real execution.
        on_tool_result: Observability callback invoked with ``(name, text_output,
            output)`` after the tool returns.
    """

    on_tool_call: Callable[[str, dict], None] | None = None
    before_tool: Callable[[str, dict], bool] | None = None
    intercept_tool: Callable[[str, dict], "tuple[bool, Any]"] | None = None
    on_tool_result: Callable[[str, str, Any], None] | None = None


@runtime_checkable
class Gate(Protocol):
    name: str

    def check(
        self,
        action: ToolCall,
        state: AgentState | None,
        registry: ToolRegistry,
    ) -> GateResult:
        """Evaluate a tool call and return an allow, deny or escalate result."""
        ...


@dataclass
class ToolResult:
    """Result of executing a tool call through the governed executor.

    Attributes:
        tool_name: Name of the tool that was called.
        arguments: Arguments passed to the tool.
        output: The tool's output when the call succeeded.
        error: Error message when the call was denied, escalated or failed.
        gate_results: Results of each gate that evaluated the call.
        success: True when the tool ran and returned without error.
    """

    tool_name: str
    arguments: dict[str, Any]
    output: Any = None
    error: str | None = None
    gate_results: list[GateResult] = field(default_factory=list)
    success: bool = False


class SyntaxGate:
    """Gate that denies unknown tools and arguments failing schema validation."""

    name = "syntax"

    def check(
        self,
        action: ToolCall,
        state: AgentState | None,
        registry: ToolRegistry,
    ) -> GateResult:
        """Deny if the tool is unknown or the arguments fail schema validation, else allow."""
        try:
            registry.get(action.tool_name)
        except KeyError:
            return GateResult(GateDecision.DENY, self.name, f"unknown tool {action.tool_name!r}")
        try:
            registry.validate(action.tool_name, action.arguments)
        except Exception as e:
            return GateResult(GateDecision.DENY, self.name, f"schema: {e}")
        return GateResult(GateDecision.ALLOW, self.name)


PolicyCheck = Callable[[ToolCall, AgentState | None], GateResult]


class PolicyGate:
    """Aggregates a list of policy checks. Chapter 12 fills the list with banking policies."""

    name = "policy"

    def __init__(self, policies: list[PolicyCheck] | None = None) -> None:
        self._policies = policies or []

    def check(
        self,
        action: ToolCall,
        state: AgentState | None,
        registry: ToolRegistry,
    ) -> GateResult:
        """Return the first non-allow policy result, or allow if every policy allows."""
        for p in self._policies:
            r = p(action, state)
            if r.decision != GateDecision.ALLOW:
                return r
        return GateResult(GateDecision.ALLOW, self.name)


class PlausibilityGate:
    """Reject implausibly large arguments. A high-school-physics filter."""

    name = "plausibility"

    def __init__(self, max_args_size: int = 10000) -> None:
        self._max_args_size = max_args_size

    def check(
        self,
        action: ToolCall,
        state: AgentState | None,
        registry: ToolRegistry,
    ) -> GateResult:
        """Deny arguments that are not JSON-serializable or exceed the size limit, else allow."""
        try:
            s = json.dumps(action.arguments, default=str)
        except Exception:
            return GateResult(GateDecision.DENY, self.name, "arguments are not JSON-serializable")
        if len(s) > self._max_args_size:
            return GateResult(
                GateDecision.DENY,
                self.name,
                f"argument size {len(s)} exceeds limit {self._max_args_size}",
            )
        return GateResult(GateDecision.ALLOW, self.name)


class GovernedToolExecutor:
    """Runs a tool call through an ordered list of gates, then executes the tool."""

    def __init__(
        self,
        registry: ToolRegistry,
        gates: list[Gate] | None = None,
        max_retries: int = 0,
        hooks: ToolHooks | None = None,
    ) -> None:
        self._registry = registry
        self._gates = gates if gates is not None else [SyntaxGate(), PolicyGate(), PlausibilityGate()]
        self._max_retries = max_retries
        # Mutable holder so an external gateway can attach via `executor.hooks`.
        self.hooks = hooks if hooks is not None else ToolHooks()

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def force_execute(
        self,
        action: ToolCall,
        state: AgentState | None = None,
        reason: str = "override",
    ) -> ToolResult:
        """Skip gates and run the tool directly. Used by reviewer overrides.

        Audit logs should still record that gates were bypassed and why.
        """
        try:
            tool = self._registry.get(action.tool_name)
        except KeyError:
            return ToolResult(
                tool_name=action.tool_name,
                arguments=dict(action.arguments),
                error=f"unknown tool {action.tool_name!r}",
                success=False,
            )
        if tool.fn is None:
            return ToolResult(
                tool_name=action.tool_name,
                arguments=dict(action.arguments),
                error="tool has no implementation",
                success=False,
            )
        try:
            parsed = self._registry.validate(action.tool_name, action.arguments)
            output = tool.fn(**parsed.model_dump())
            return ToolResult(
                tool_name=action.tool_name,
                arguments=dict(action.arguments),
                output=output,
                gate_results=[GateResult(GateDecision.ALLOW, "human_override", reason)],
                success=True,
            )
        except Exception as e:
            return ToolResult(
                tool_name=action.tool_name,
                arguments=dict(action.arguments),
                error=f"{type(e).__name__}: {e}",
                gate_results=[GateResult(GateDecision.ALLOW, "human_override", reason)],
                success=False,
            )

    def execute(
        self,
        action: ToolCall,
        state: AgentState | None = None,
    ) -> ToolResult:
        """Run the gates then execute the tool, returning a ToolResult.

        Each gate is checked in order; a deny or escalate returns immediately
        with the accumulated gate results. Testing hooks run after the gates and
        may block or intercept the call. On execution the tool is retried up to
        max_retries times before returning the last error.

        Args:
            action: The tool call to execute.
            state: Optional agent state passed to the gates.

        Returns:
            A ToolResult recording the output or error, the gate results and
            whether the call succeeded.
        """
        gate_results: list[GateResult] = []
        for gate in self._gates:
            r = gate.check(action, state, self._registry)
            gate_results.append(r)
            if r.decision == GateDecision.DENY:
                return ToolResult(
                    tool_name=action.tool_name,
                    arguments=dict(action.arguments),
                    error=f"denied by {r.gate_name}: {r.reason}",
                    gate_results=gate_results,
                    success=False,
                )
            if r.decision == GateDecision.ESCALATE:
                return ToolResult(
                    tool_name=action.tool_name,
                    arguments=dict(action.arguments),
                    error=f"escalated by {r.gate_name}: {r.reason}",
                    gate_results=gate_results,
                    success=False,
                )

        tool = self._registry.get(action.tool_name)
        if tool.fn is None:
            return ToolResult(
                tool_name=action.tool_name,
                arguments=dict(action.arguments),
                error="tool has no implementation",
                gate_results=gate_results,
                success=False,
            )

        # Testing hooks (no-ops unless a gateway is attached). These run AFTER the
        # governance gates so fault injection exercises the agent's recovery, not
        # the gates themselves.
        args = dict(action.arguments)
        if self.hooks.on_tool_call is not None:
            self.hooks.on_tool_call(action.tool_name, args)
        if self.hooks.before_tool is not None and not self.hooks.before_tool(action.tool_name, args):
            return ToolResult(
                tool_name=action.tool_name,
                arguments=args,
                error="tool call blocked/faulted by testing gateway",
                gate_results=gate_results + [GateResult(GateDecision.DENY, "gateway", "fault injected")],
                success=False,
            )
        if self.hooks.intercept_tool is not None:
            intercepted, response = self.hooks.intercept_tool(action.tool_name, args)
            if intercepted:
                if self.hooks.on_tool_result is not None:
                    self.hooks.on_tool_result(action.tool_name, str(response), response)
                return ToolResult(
                    tool_name=action.tool_name,
                    arguments=args,
                    output=response,
                    gate_results=gate_results + [GateResult(GateDecision.ALLOW, "gateway", "response intercepted")],
                    success=True,
                )

        last_error: str | None = None
        for _ in range(self._max_retries + 1):
            try:
                parsed = self._registry.validate(action.tool_name, action.arguments)
                output = tool.fn(**parsed.model_dump())
                if self.hooks.on_tool_result is not None:
                    self.hooks.on_tool_result(action.tool_name, str(output), output)
                return ToolResult(
                    tool_name=action.tool_name,
                    arguments=dict(action.arguments),
                    output=output,
                    gate_results=gate_results,
                    success=True,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

        return ToolResult(
            tool_name=action.tool_name,
            arguments=dict(action.arguments),
            error=last_error,
            gate_results=gate_results,
            success=False,
        )
