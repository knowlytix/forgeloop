from pydantic import BaseModel

from agentlab.core import ToolCall
from agentlab.tools import (
    GateDecision,
    GovernedToolExecutor,
    PlausibilityGate,
    PolicyGate,
    RiskLevel,
    SyntaxGate,
    Tool,
    ToolRegistry,
)
from agentlab.tools.executor import GateResult


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


def _adder(x: int) -> dict:
    return {"y": x + 1}


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(
        Tool(
            name="adder",
            description="add one",
            input_schema=_In,
            output_schema=_Out,
            risk=RiskLevel.LOW,
            fn=_adder,
        )
    )
    return r


def test_execute_success():
    ex = GovernedToolExecutor(_registry())
    result = ex.execute(ToolCall(tool_name="adder", arguments={"x": 1}))
    assert result.success
    assert result.output == {"y": 2}


def test_unknown_tool_denied_by_syntax_gate():
    ex = GovernedToolExecutor(_registry())
    result = ex.execute(ToolCall(tool_name="nope", arguments={}))
    assert not result.success
    assert any(
        g.gate_name == "syntax" and g.decision == GateDecision.DENY for g in result.gate_results
    )


def test_invalid_args_denied_by_syntax_gate():
    ex = GovernedToolExecutor(_registry())
    result = ex.execute(ToolCall(tool_name="adder", arguments={"x": "not-an-int"}))
    assert not result.success
    assert any(
        g.gate_name == "syntax" and g.decision == GateDecision.DENY for g in result.gate_results
    )


def test_plausibility_gate_blocks_large_args():
    ex = GovernedToolExecutor(
        _registry(),
        gates=[SyntaxGate(), PlausibilityGate(max_args_size=20)],
    )
    big = {"x": "x" * 1000}
    result = ex.execute(ToolCall(tool_name="adder", arguments=big))
    assert not result.success


def test_policy_gate_denies_via_policy():
    def deny_all(action, state):
        return GateResult(GateDecision.DENY, "test-policy", "blocked")

    ex = GovernedToolExecutor(_registry(), gates=[PolicyGate(policies=[deny_all])])
    result = ex.execute(ToolCall(tool_name="adder", arguments={"x": 1}))
    assert not result.success
    assert "test-policy" in result.error


def test_policy_gate_allows_when_empty():
    ex = GovernedToolExecutor(_registry(), gates=[PolicyGate()])
    result = ex.execute(ToolCall(tool_name="adder", arguments={"x": 1}))
    # PolicyGate alone allows, but tool will still run because no other gate denies
    assert result.success
