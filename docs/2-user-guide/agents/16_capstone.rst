Chapter 16 — Capstone: Governed Banking Complaint Agent
=======================================================

This page shows how to assemble and run the governed complaint agent from
:mod:`forgeloop.agents.capstone`. The agent reads a customer message, finds the
governing policy and either drafts a reply or hands the case to a human, under a
constraint that it never invents a fact, promises a remedy it cannot authorize or
processes a message it should have refused.
:func:`~forgeloop.agents.capstone.complaint_agent.build_complaint_harness`
registers five tools and the gate stack;
:class:`~forgeloop.agents.capstone.complaint_agent.ComplaintAgent` drives the
fixed workflow. The workflow is fixed by choice: a regulated setting has a small
enumerable set of states whose failure modes are countable.

The tools
---------

Each model-backed tool pairs the shared Qwen3-4B model with a deterministic
backstop, so a regulated decision never rests on an unverifiable inference. The
five tools run in a fixed order.

.. list-table::
   :header-rows: 1

   * - Tool
     - What it does
   * - ``classify_complaint``
     - Labels the message with a frozen Qwen encoder and a trained logit head.
   * - ``extract_facts``
     - Binds the message to real policy entities, returning a grounded fact and a reusable query.
   * - ``search_policy`` (:func:`~forgeloop.agents.capstone.banking_tools.make_search_policy_tool`)
     - Retrieves through the relation operators under a calibrated bind floor and synthesizes a self-verified answer.
   * - ``flag_regulatory``
     - Lets Qwen propose regulations while a trained GMS store verifies, corrects and decides escalation.
   * - ``draft_response``
     - Writes the reply with a LoRA adapter, then verifies it against the store.

Build the harness
-----------------

:func:`~forgeloop.agents.capstone.complaint_agent.build_complaint_harness` calls
:func:`~forgeloop.agents.capstone.banking_tools.register_all` to populate a tool
registry and stacks three gates: a syntax gate, a policy engine and the trained
GMS plausibility gate that scores each workflow transition against the banking
store's ``has_enables`` DAG at a calibrated threshold.

.. code-block:: python

   import json
   from pathlib import Path
   from forgeloop.agents.capstone import build_complaint_harness
   from forgeloop.agents.core import Budget, BudgetTracker, TaskSpec

   cases = json.loads(Path("data/eval_cases/cases.json").read_text())
   harness, registry = build_complaint_harness(policies_dir="data/policies")
   print("registered tools:", [t.name for t in registry.all()])

Run a case and read the trajectory
-----------------------------------

:meth:`~forgeloop.agents.governance.harness.GovernanceHarness.run` takes a
``TaskSpec`` and returns a full trajectory. A routine inquiry runs the workflow to
a drafted reply and finishes with ``status`` ``done``.

.. code-block:: python

   case = next(c for c in cases if c["id"] == "case-002")
   task = TaskSpec(goal="handle complaint", inputs={"message": case["message"]})
   traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))
   print("status:      ", traj.final_state.status)
   print("final output:", traj.final_state.final_output)

An adversarial overdraft case escalates instead of drafting. The GMS regulatory
guard flags UDAAP by traversing the severity path, and the agent returns an
``Escalate`` action carrying the reason.

.. code-block:: python

   case = next(c for c in cases if c["id"] == "case-016")
   task = TaskSpec(goal="handle complaint", inputs={"message": case["message"]})
   traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))
   esc = next(r for r in traj.records if r.action.kind == "escalate")
   print("status:    ", traj.final_state.status)
   print("escalation:", esc.action.reason)

A message carrying PII is refused at the first tool call. The policy gate denies
the call, the observation reports failure and the agent escalates rather than
proceeding.

.. code-block:: python

   case = next(c for c in cases if c["id"] == "case-011")
   task = TaskSpec(goal="handle complaint", inputs={"message": case["message"]})
   traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))
   failed = next(r for r in traj.records
                 if r.action.kind == "tool_call" and not r.observation.get("success"))
   print("failing gate:", failed.observation["error"])

Extraction as a grounded query
------------------------------

``extract_facts`` runs the GEODE parse-bind loop once and threads its result into
``search_policy`` and ``flag_regulatory``, so the message is parsed a single time.
A fee complaint binds to the ``overdraft`` and ``fee_reversal`` entities; a vague
message binds nothing and the extractor falls back to a coarse default rather than
a fabricated label.

.. code-block:: python

   from forgeloop.agents.capstone.banking_tools import _extract_impl

   for msg in [
       "I was charged a $35 overdraft fee on my checking account, reverse it!",
       "I'm going to sue you unless you give me my money back today.",  # vague
   ]:
       facts = _extract_impl(msg)
       print(facts["product"], "/", facts["issue"],
             "| query_facts:", facts["query_facts"],
             "| grounded:", bool(facts["extraction"]))

Aggregate over the benchmark
----------------------------

Running the twenty labeled cases reports classification and escalation accuracy,
and confirms the hash-chained audit log verifies after the run.

.. code-block:: python

   escalation_correct = 0
   for case in cases:
       task = TaskSpec(goal="handle complaint", inputs={"message": case["message"]})
       traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))
       out = traj.final_state.final_output or {}
       escalated = (traj.final_state.status in ("escalated", "failed")
                    or (isinstance(out, dict) and out.get("recommended_action") == "escalate"))
       escalation_correct += escalated == case.get("expected_escalation")
   print(f"escalation accuracy: {escalation_correct}/{len(cases)}")
   print("audit verifies:     ", harness.audit.verify())

The gate stack and the escalation triggers
-------------------------------------------

Every tool call passes three gates in order: syntax, policy, then GMS
plausibility. Three conditions divert the run to a human. A gate refusing a call
escalates from the tool failure, which is how a PII or prompt-injection message
short-circuits at the input. A high-severity regulatory flag escalates from the
``flag_regulatory`` verdict. A draft the store's verifier objects to escalates
before the reply is sent: the verifier corrects a drifted fee against exact
numerical memory and escalates an unauthorized fee-waiver promise rather than
sending it. The regulatory guard and the draft verifier are required backstops
that fail loud, so a load error raises and the agent escalates rather than
downgrading silently to a weaker check.

See also
--------

- :doc:`/3-api-reference/modules/capstone/index` — the agent, the tools and
  :func:`~forgeloop.agents.capstone.complaint_agent.build_complaint_harness`.
- :doc:`/3-api-reference/modules/capstone_internals/index` — the retriever, the
  regulatory guard and the draft verifier the tools call.
- :doc:`13_governed_retrieval` and :doc:`17_testing_agents` — the retrieval layer
  and the test stand for this agent.
- :doc:`/4-notebook-examples/agents/index` — the capstone run end to end.
