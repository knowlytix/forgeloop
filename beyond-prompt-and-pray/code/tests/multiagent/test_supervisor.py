import pytest

from agentlab.core import BaseAgent, Finish, TaskSpec
from agentlab.governance import GovernanceHarness
from agentlab.multiagent import MessageBus, Supervisor, Worker
from agentlab.tools import GovernedToolExecutor, ToolRegistry


class _FinishAgent(BaseAgent):
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def propose_action(self, state):
        return Finish(output=self._answer)


def _make_worker(name: str, capability: str, answer: str) -> Worker:
    executor = GovernedToolExecutor(ToolRegistry(), gates=[])
    harness = GovernanceHarness(_FinishAgent(answer), executor)
    return Worker(name=name, capability=capability, harness=harness)


def test_supervisor_delegates_and_returns_response():
    bus = MessageBus()
    sup = Supervisor(
        name="sup",
        workers=[_make_worker("classifier", "classify", "complaint")],
        bus=bus,
    )
    response = sup.delegate("classifier", goal="classify the message")
    assert response.message_type == "response"
    assert response.sender == "classifier"
    assert response.payload["final_output"] == "complaint"


def test_supervisor_records_request_and_response_on_bus():
    bus = MessageBus()
    sup = Supervisor(
        name="sup",
        workers=[_make_worker("w1", "x", "ok")],
        bus=bus,
    )
    sup.delegate("w1", goal="g")
    assert len(bus) == 2


def test_unknown_worker_raises():
    sup = Supervisor(name="sup", workers=[_make_worker("only", "x", "")])
    with pytest.raises(KeyError):
        sup.delegate("missing", goal="g")


def test_worker_rejects_unsupported_message_type():
    w = _make_worker("w1", "x", "")
    from agentlab.multiagent import AgentMessage
    response = w.handle(AgentMessage(sender="sup", receiver="w1", message_type="chitchat", payload={}))
    assert response.message_type == "rejected"


def test_supervisor_workers_listed():
    sup = Supervisor(
        name="sup",
        workers=[_make_worker("a", "x", ""), _make_worker("b", "y", "")],
    )
    assert set(sup.workers()) == {"a", "b"}


def test_each_worker_has_independent_audit_log():
    """The 'governance per worker' guarantee: each worker's harness has its own audit chain."""
    w1 = _make_worker("w1", "x", "out1")
    w2 = _make_worker("w2", "y", "out2")
    sup = Supervisor(name="sup", workers=[w1, w2])
    sup.delegate("w1", goal="g1")
    sup.delegate("w2", goal="g2")
    assert w1.harness.audit is not w2.harness.audit
    assert w1.harness.audit.verify()
    assert w2.harness.audit.verify()
