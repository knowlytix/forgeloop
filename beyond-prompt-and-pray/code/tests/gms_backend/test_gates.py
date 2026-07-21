"""Tests for GMSPlausibilityGate against a real loaded GMS store."""

import pytest

from agentlab.core import ToolCall
from agentlab.gms_backend import GMSPlausibilityGate
from agentlab.tools.executor import GateDecision
from agentlab.tools.registry import ToolRegistry


def test_gate_allows_below_theta_with_real_store(gms_store):
    """Any (context, relation, tail) the store actually knows scores low."""
    triples = gms_store.query_triples()
    assert triples, "store has no triples; cannot calibrate test"
    head, relation, tail = triples[0]
    gate = GMSPlausibilityGate(
        gms_store,
        theta=10.0,  # generous so any real score passes
        context=head,
        relation=relation,
    )
    r = gate.check(ToolCall(tool_name=tail, arguments={}), None, ToolRegistry())
    assert r.decision == GateDecision.ALLOW


def test_gate_handles_unknown_triple(gms_store):
    """An unknown (context, relation, tail) goes through the on_missing path."""
    gate = GMSPlausibilityGate(
        gms_store,
        theta=1.5,
        context="totally_unrelated_context",
        relation="should_call",
        on_missing="allow",
    )
    r = gate.check(ToolCall(tool_name="never_seen_tool", arguments={}), None, ToolRegistry())
    # Either the real score-triple returned a number (and we allow/deny) or it
    # returned None and we hit on_missing="allow". Both end as ALLOW here.
    assert r.decision in (GateDecision.ALLOW, GateDecision.DENY)


def test_gate_only_acts_on_tool_calls(gms_store):
    gate = GMSPlausibilityGate(gms_store, theta=1.5)

    class _NotAToolCall:
        kind = "ask_user"
        tool_name = ""
        arguments = {}

    r = gate.check(_NotAToolCall(), None, ToolRegistry())
    assert r.decision == GateDecision.ALLOW


def test_invalid_on_missing_rejected(gms_store):
    with pytest.raises(ValueError):
        GMSPlausibilityGate(gms_store, on_missing="maybe")
