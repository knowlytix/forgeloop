Human-in-the-Loop and Escalation UX
===================================

When a gate escalates a call the agent cannot clear on its own, the decision
passes to a person, and the shape of the handoff determines whether that person
can act on it. This page shows how a
:class:`~forgeloop.agents.governance.GovernanceHarness` routes an escalating gate
result to a :class:`~forgeloop.agents.governance.HumanReviewer`, what the reviewer
receives as an :class:`~forgeloop.agents.governance.EscalationRequest` and how the
three decisions in :class:`~forgeloop.agents.governance.HumanDecision` resume the
run. The request is a small fixed contract rather than the full transcript, so it
is reviewable without reconstructing the run.

The escalation surface
----------------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Role
   * - :class:`~forgeloop.agents.governance.HumanReviewer`
     - Protocol: ``review(request) -> HumanResponse``.
   * - :class:`~forgeloop.agents.governance.ScriptedReviewer`
     - Returns preset responses; used in tests.
   * - :class:`~forgeloop.agents.governance.CLIReviewer`
     - Reads a decision from stdin; used in notebooks.
   * - :class:`~forgeloop.agents.governance.EscalationRequest`
     - The context handed to the reviewer.
   * - :class:`~forgeloop.agents.governance.HumanResponse`
     - A decision and an optional note.
   * - :class:`~forgeloop.agents.governance.HumanDecision`
     - The three answers: APPROVE, DENY, DEFER.

Setup: a call that escalates
----------------------------

The agent proposes one high-risk email whose body contains an address, so
:func:`~forgeloop.agents.governance.pii_policy` escalates it. Passing a reviewer
to :meth:`~forgeloop.agents.governance.GovernanceHarness.run` routes the
escalating result to that reviewer.

.. code-block:: python

   from pydantic import BaseModel
   from forgeloop.agents.core import BaseAgent, Finish, TaskSpec, ToolCall
   from forgeloop.agents.governance import (
       GovernanceHarness, HumanDecision, HumanResponse, PlausibilityGate,
       PolicyEngine, ScriptedReviewer, SyntaxGate, pii_policy,
   )
   from forgeloop.agents.tools import GovernedToolExecutor, RiskLevel, Tool, ToolRegistry

   class EmailIn(BaseModel):
       to: str
       body: str
   class EmailOut(BaseModel):
       success: bool

   tool = Tool(name="send_email", description="send email",
               input_schema=EmailIn, output_schema=EmailOut,
               risk=RiskLevel.HIGH, fn=lambda to, body: {"success": True})

   class OneShotAgent(BaseAgent):
       def propose_action(self, state):
           if state.step > 0:
               return Finish(output="done")
           return ToolCall(tool_name="send_email",
                           arguments={"to": "x@x.com", "body": "contact a@b.com"})

   def harness():
       r = ToolRegistry(); r.register(tool)
       engine = PolicyEngine([pii_policy])
       ex = GovernedToolExecutor(r, gates=[SyntaxGate(), engine.as_gate(), PlausibilityGate()])
       return GovernanceHarness(OneShotAgent(), ex)

The request the reviewer sees
-----------------------------

An :class:`~forgeloop.agents.governance.EscalationRequest` carries the proposed
action, the gate results and a short reason, and its ``to_json`` renders that for
a reviewer. The conversation history and reasoning trace are not part of the
request; they remain in the audit log, so the request is the summary and the log
is the archive.

.. code-block:: python

   rev_approve = ScriptedReviewer(HumanResponse(decision=HumanDecision.APPROVE, note="confirmed safe"))
   h = harness()
   traj = h.run(TaskSpec(goal="test"), human_reviewer=rev_approve)

   print(rev_approve.requests[0].to_json(indent=2))

A :class:`~forgeloop.agents.governance.ScriptedReviewer` records every request it
receives on ``requests``, which is how a test asserts what the reviewer was
asked.

The three decisions
-------------------

APPROVE overrides the gate: the harness calls
:meth:`~forgeloop.agents.tools.GovernedToolExecutor.force_execute`, so the tool
runs, and the audit record keeps both the gate that escalated and the override
that authorized the bypass.

.. code-block:: python

   tool_step = next(r for r in traj.records if r.action.kind == "tool_call")
   print("success:", tool_step.observation["success"])   # True
   print("gates: ", [(g["gate"], g["decision"]) for g in tool_step.observation["gate_results"]])

DENY confirms the gate. The call fails and its failed observation is returned to
the loop, so the agent stays alive to observe the failure and proceed to its next
action.

.. code-block:: python

   rev_deny = ScriptedReviewer(HumanResponse(decision=HumanDecision.DENY))
   traj = harness().run(TaskSpec(goal="test"), human_reviewer=rev_deny)
   tool_step = next(r for r in traj.records if r.action.kind == "tool_call")
   print("success:", tool_step.observation["success"])   # False

DEFER ends the loop. The harness writes ``human_decision = "defer"`` into the
observation, sets the terminal status to ``escalated`` and stops, leaving the
trajectory intact for a reviewer with more authority to resume.

.. code-block:: python

   rev_defer = ScriptedReviewer(HumanResponse(decision=HumanDecision.DEFER, note="need legal"))
   traj = harness().run(TaskSpec(goal="test"), human_reviewer=rev_defer)
   print("final status:", traj.final_state.status)   # escalated

Why three, and resuming
-----------------------

The contract is held to three decisions because each additional answer needs its
own resume path, and a request that seems to need a fourth usually decomposes
into the three. Human review costs time set by the reviewer rather than a tool
call, so a deployment batches escalations and sets timeouts. Resuming a deferred
run re-validates state against the same gates, because a fact that held when the
call escalated can be stale by the time the decision lands, which is why the
harness runs the gates on resume rather than trusting the earlier verdict.

See also
--------

- :doc:`/3-api-reference/modules/governance/index` — signatures for the reviewers, the
  request and response types and the harness.
- :doc:`12_runtime_governance` — the gates and the audit chain that produce and
  record the escalation.
- :doc:`/4-notebook-examples/agents/index` — the approve, deny and defer paths
  run end to end.
