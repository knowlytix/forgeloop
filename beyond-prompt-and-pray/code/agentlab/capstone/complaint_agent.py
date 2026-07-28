"""ComplaintAgent: a fixed-workflow agent for the capstone.

Workflow: classify → extract → search policy → flag regulatory → draft
or escalate → finish. The agent escalates on any prior tool failure
(which is how injected-PII or prompt-injection cases short-circuit). It
also escalates directly when flag_regulatory marks UDAAP or Reg X risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentlab.capstone.banking_tools import register_all
from agentlab.core import BaseAgent, Escalate, Finish, ToolCall
from agentlab.core.action import Action
from agentlab.core.state import AgentState
from agentlab.gms_backend import GMSPlausibilityGate
from agentlab.governance import (
    GovernanceHarness,
    PolicyEngine,
    SyntaxGate,
    pii_policy,
)
from agentlab.governance.semantic_guard import (
    semantic_prohibited_advice_policy,
    semantic_prompt_injection_policy,
)
from agentlab.reasoning import Scratchpad, TrustLevel
from agentlab.tools import GovernedToolExecutor, ToolRegistry

# Workflow node names as the GMS banking store knows them (its has_enables DAG),
# in execution order. classify_complaint/extract_facts map to the store's short
# node names; the rest already match.
_WORKFLOW_NODES = ["classify", "extract", "search_policy", "flag_regulatory", "draft_response"]
_TOOL_NODE_MAP = {"classify_complaint": "classify", "extract_facts": "extract"}


def _prev_workflow_node(action, state) -> str:
    """Previous workflow node for the plausibility transition check: 'start'
    before the first tool, else the node of the tool that ran last."""
    n = 0 if state is None else len(state.tool_results)
    if n <= 0:
        return "start"
    return _WORKFLOW_NODES[min(n - 1, len(_WORKFLOW_NODES) - 1)]


class ComplaintAgent(BaseAgent):
    def propose_action(self, state: AgentState) -> Action:
        for i, result in enumerate(state.tool_results):
            if isinstance(result, dict) and result.get("success") is False:
                return Escalate(
                    reason=f"tool failure at step {i}: {result.get('error', '')}",
                    context={"failed_step": i},
                )

        msg = str(state.task.inputs.get("message", ""))

        if state.step == 0:
            return ToolCall(tool_name="classify_complaint", arguments={"message": msg})

        if state.step == 1:
            return ToolCall(tool_name="extract_facts", arguments={"message": msg})

        if state.step == 2:
            facts = self._output(state, 1)
            # Reuse the grounded extraction from step 1 so the message is parsed
            # exactly once: extract_facts already ran the GEODE parse⇄bind loop,
            # and the RAG retrieves/generates from that same extraction.
            return ToolCall(
                tool_name="search_policy",
                arguments={"query": msg, "extraction": facts.get("extraction")},
            )

        if state.step == 3:
            facts = self._output(state, 1)
            # Pass the bound policy triples so flag_regulatory grounds
            # (product, issue) on what the message actually resolved to.
            return ToolCall(
                tool_name="flag_regulatory",
                arguments={
                    "product": str(facts.get("product", "unknown")),
                    "issue": str(facts.get("issue", "general")),
                    "message": msg,
                    "query_facts": facts.get("query_facts", []),
                },
            )

        if state.step == 4:
            flags = self._output(state, 3)
            if flags.get("escalate"):
                return Escalate(
                    reason=f"regulatory risk flagged: {flags.get('flags', [])}",
                    context={"flags": flags.get("flags", [])},
                )
            facts = self._output(state, 1)
            policy_results = self._output(state, 2).get("results", [])
            classification = self._output(state, 0)
            # Chapter 3 discipline made operational: before the agent lets the
            # model write the customer-facing draft, it asserts that every claim
            # the reply is grounded on carries an evidence pointer. A claim with
            # no evidence escalates to a human instead of reaching the draft step
            # --- the structural check the harness runs before output leaves the
            # system, not a model judging itself.
            pad = self._scratchpad(state)
            try:
                pad.assert_all_claims_have_evidence()
            except AssertionError as exc:
                return Escalate(
                    reason=f"ungrounded claim before drafting: {exc}",
                    context={"unsupported": [e.text for e in pad.unsupported_claims()]},
                )
            return ToolCall(
                tool_name="draft_response",
                arguments={
                    "category": str(classification.get("category", "unknown")),
                    "issue": str(facts.get("issue", "general")),
                    "policy_evidence": policy_results,
                    "message": msg,
                },
            )

        # The GMS draft verifier may demand escalation (an unauthorized
        # fee-waiver promise it could not safely auto-correct). Honor it before
        # finishing.
        draft = self._output(state, 4)
        if draft.get("requires_escalation"):
            return Escalate(
                reason=f"draft verification: {draft.get('reason', 'unsafe draft')}",
                context={"draft_reason": draft.get("reason", "")},
            )

        return Finish(output=self._compile_output(state))

    @staticmethod
    def _output(state: AgentState, step_idx: int) -> dict[str, Any]:
        if step_idx >= len(state.tool_results):
            return {}
        out = state.tool_results[step_idx].get("output")
        return out or {}

    def _scratchpad(self, state: AgentState) -> Scratchpad:
        """Reconstruct the case's structured reasoning record (Chapter 3) from
        the tool results so far. It is a pure function of the trajectory --- the
        same trajectory always yields the same scratchpad --- so it replays from
        the audit log without rerunning the agent. Each model output enters as a
        typed entry with an explicit trust level and an evidence pointer, never
        as free-form prose; fields the model merely inferred without textual
        support (urgency, sentiment) are recorded as assumptions, not claims, so
        they can never pass an evidence check they have not earned."""
        pad = Scratchpad()

        classification = self._output(state, 0)
        if classification:
            cat = classification.get("category", "unknown")
            conf = float(classification.get("confidence", 0.0))
            pad.add_observation(
                f"message classified as {cat}",
                source="classify_complaint",
                evidence=f"classifier confidence={conf:.2f}",
                trust=TrustLevel.MEDIUM,  # a tool result
            )

        facts = self._output(state, 1)
        if facts:
            # When the extraction grounded the message to real policy entities,
            # the issue is a cited, independently-checked claim (the bound policy
            # head) -> HIGH trust. When grounding abstained, it is the hybrid
            # classifier's guess over the message -> MEDIUM.
            bound = facts.get("query_facts") or []
            if bound:
                heads = ", ".join(sorted({str(h) for h, _r, _t in bound}))
                pad.add_claim(
                    f"issue is {facts.get('issue', 'general')}",
                    evidence=f"extract_facts grounded to policy entities [{heads}]",
                    trust=TrustLevel.HIGH,
                )
            else:
                pad.add_claim(
                    f"issue is {facts.get('issue', 'general')}",
                    evidence="extract_facts (over message)",
                    trust=TrustLevel.MEDIUM,
                )
            # Inferred, uncited -> an assumption, not a claim.
            pad.add_assumption(
                f"urgency {facts.get('urgency', 'normal')}, "
                f"sentiment {facts.get('sentiment', 'neutral')}"
            )

        for r in self._output(state, 2).get("results", []):
            pid = r.get("id")
            if pid:
                pad.add_observation(
                    f"governing policy retrieved: {pid}",
                    source="search_policy",
                    evidence=pid,  # a cited source
                    trust=TrustLevel.HIGH,
                )

        for flag in self._output(state, 3).get("flags", []):
            pad.add_claim(
                f"regulation implicated: {flag}",
                evidence=f"GMS regulatory graph ({flag})",
                trust=TrustLevel.HIGHEST,  # independently checked against the graph
            )

        return pad

    def _compile_output(self, state: AgentState) -> dict[str, Any]:
        classification = self._output(state, 0)
        facts = self._output(state, 1)
        policy_results = self._output(state, 2).get("results", [])
        flags = self._output(state, 3)
        draft = self._output(state, 4) if state.step >= 5 else {}
        return {
            "classification": classification.get("category"),
            "summary": facts.get("summary"),
            "product": facts.get("product"),
            "issue": facts.get("issue"),
            "risk_flags": flags.get("flags", []),
            "policy_evidence": policy_results,
            "recommended_action": "escalate" if flags.get("escalate") else "respond",
            "draft_response": draft.get("text", ""),
            "reasoning": self._scratchpad(state).render_table(),
        }


def build_complaint_harness(
    policies_dir: Path | str | None = None,
    extra_policies: list | None = None,
) -> tuple[GovernanceHarness, ToolRegistry]:
    registry = ToolRegistry()
    register_all(registry, policies_dir=policies_dir)
    # PII stays a regex format check (the right tool for SSN/card/email). The
    # semantic intents — prompt injection, prohibited advice — move to the
    # Qwen-backed semantic guard. Fee-waiver promises are caught downstream by
    # the GMS draft verifier (on the generated draft), not as an input policy.
    default_policies = [
        pii_policy,
        semantic_prompt_injection_policy,
        semantic_prohibited_advice_policy,
    ]
    if extra_policies:
        default_policies = default_policies + list(extra_policies)
    engine = PolicyEngine(default_policies)

    # Plausibility is the trained GMS geometric gate: it scores each workflow
    # transition (prev_node -> tool) against the banking store's has_enables DAG
    # at the calibrated threshold. Required — no placeholder fallback.
    plausibility_gate = _build_gms_plausibility_gate()
    gates = [SyntaxGate(), engine.as_gate(), plausibility_gate]
    executor = GovernedToolExecutor(registry, gates=gates)
    return GovernanceHarness(ComplaintAgent(), executor), registry


def _build_gms_plausibility_gate() -> GMSPlausibilityGate:
    import json

    import torch
    from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

    root = Path(__file__).resolve().parents[2]
    store_path = root / "data" / "gms_banking_store"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GMSExpertStore(DocGMSConfig(store_path=str(store_path)), device=device)
    if not store.load():
        raise RuntimeError(
            f"failed to load GMS banking store at {store_path!s} for the "
            "plausibility gate; run the store build first."
        )
    theta = float(
        json.loads((store_path / "calibration.json").read_text())["plausibility_gate"]["threshold"]
    )
    return GMSPlausibilityGate(
        store,
        theta=theta,
        context=_prev_workflow_node,
        relation="has_enables",
        on_missing="allow",
        tool_node_map=_TOOL_NODE_MAP,
    )
