from pydantic import BaseModel

from agentlab.core import BaseAgent, Finish, TaskSpec, ToolCall
from agentlab.governance import (
    EscalationRequest,
    GovernanceHarness,
    HumanDecision,
    HumanResponse,
    PlausibilityGate,
    PolicyEngine,
    ScriptedReviewer,
    SyntaxGate,
    pii_policy,
)
from agentlab.tools import GovernedToolExecutor, RiskLevel, Tool, ToolRegistry


class _EmailIn(BaseModel):
    to: str
    body: str


class _EmailOut(BaseModel):
    success: bool


def _send_email_impl(to: str, body: str) -> dict:
    return {"sent_to": to, "len": len(body)}


_SEND_EMAIL = Tool(
    name="send_email",
    description="send an email",
    input_schema=_EmailIn,
    output_schema=_EmailOut,
    risk=RiskLevel.HIGH,
    fn=_send_email_impl,
)


class _PIIAgent(BaseAgent):
    """One call with PII, then finish."""

    def propose_action(self, state):
        if state.step > 0:
            return Finish(output="done")
        return ToolCall(
            tool_name="send_email",
            arguments={"to": "x@x.com", "body": "contact me at a@b.com"},
        )


def _harness():
    registry = ToolRegistry()
    registry.register(_SEND_EMAIL)
    engine = PolicyEngine([pii_policy])
    gates = [SyntaxGate(), engine.as_gate(), PlausibilityGate()]
    return GovernanceHarness(_PIIAgent(), GovernedToolExecutor(registry, gates=gates))


def test_escalation_request_roundtrip():
    r = EscalationRequest(
        run_id="r1", step=2, reason="PII", proposed_action={"kind": "tool_call"}, gate_results=[]
    )
    r2 = EscalationRequest.from_json(r.to_json())
    assert r2.run_id == "r1"
    assert r2.step == 2


def test_scripted_reviewer_records_requests():
    rev = ScriptedReviewer(HumanResponse(decision=HumanDecision.DENY))
    h = _harness()
    h.run(TaskSpec(goal="test"), human_reviewer=rev)
    assert len(rev.requests) == 1
    assert "pii" in rev.requests[0].reason.lower()


def test_approve_overrides_pii_gate():
    rev = ScriptedReviewer(HumanResponse(decision=HumanDecision.APPROVE, note="confirmed safe"))
    h = _harness()
    traj = h.run(TaskSpec(goal="test"), human_reviewer=rev)
    tool_steps = [r for r in traj.records if r.action.kind == "tool_call"]
    assert tool_steps[0].observation["success"] is True
    gate_names = [g["gate"] for g in tool_steps[0].observation["gate_results"]]
    assert "human_override" in gate_names


def test_deny_keeps_failure():
    rev = ScriptedReviewer(HumanResponse(decision=HumanDecision.DENY))
    h = _harness()
    traj = h.run(TaskSpec(goal="test"), human_reviewer=rev)
    tool_steps = [r for r in traj.records if r.action.kind == "tool_call"]
    assert tool_steps[0].observation["success"] is False


def test_defer_ends_loop_as_escalated():
    rev = ScriptedReviewer(HumanResponse(decision=HumanDecision.DEFER))
    h = _harness()
    traj = h.run(TaskSpec(goal="test"), human_reviewer=rev)
    assert traj.final_state.status == "escalated"
    # The loop should not have continued to the Finish action
    assert not any(r.action.kind == "finish" for r in traj.records)


def test_no_reviewer_keeps_old_behavior():
    """When no reviewer is passed, gate escalation simply causes a failed result."""
    h = _harness()
    traj = h.run(TaskSpec(goal="test"))
    tool_steps = [r for r in traj.records if r.action.kind == "tool_call"]
    assert tool_steps[0].observation["success"] is False
