Chapter 1 — The Governed Agent Loop
===================================

This page shows how to assemble a working agent from the primitives in
:mod:`forgeloop.agents.core` and drive it with :func:`~forgeloop.agents.core.run_loop`,
then where a governance gate attaches to the same loop. The loop is a generator
of :class:`~forgeloop.agents.core.StepRecord` values, so a run is inspectable,
replayable and auditable by construction — the property the evaluation
(:doc:`10_trajectory_evaluation`) and governance (:doc:`12_runtime_governance`)
chapters build on.

The pieces
----------

A run is assembled from six types. Actions are a discriminated union: the
``kind`` field is a ``Literal`` on each subclass, so an ill-formed action raises
at construction rather than at execution.

.. list-table::
   :header-rows: 1

   * - Type
     - Role
   * - :class:`~forgeloop.agents.core.TaskSpec`
     - The task: goal, inputs, expected outputs, constraints.
   * - :class:`~forgeloop.agents.core.AgentState`
     - Serializable state carried across steps.
   * - :class:`~forgeloop.agents.core.ToolCall`,
     - The typed actions an agent may propose
   * - :class:`~forgeloop.agents.core.Finish`,
     - (with :class:`~forgeloop.agents.core.AskUser`,
   * - :class:`~forgeloop.agents.core.Escalate`
     - :class:`~forgeloop.agents.core.ActionKind`).
   * - :class:`~forgeloop.agents.core.BaseAgent`
     - The proposer: ``propose_action(state) -> Action``.
   * - :class:`~forgeloop.agents.core.Environment`
     - The effector: ``step(action) -> dict`` observation.
   * - :func:`~forgeloop.agents.core.run_loop`
     - The driver, yielding one ``StepRecord`` per step.

Define a task and initial state
-------------------------------

:class:`~forgeloop.agents.core.TaskSpec` validates that ``goal`` is non-empty;
:class:`~forgeloop.agents.core.AgentState` starts with ``status="running"`` and
empty result lists.

.. code-block:: python

   from forgeloop.agents.core import TaskSpec, AgentState

   task = TaskSpec(
       goal="classify a customer complaint message",
       inputs={"message": "I was charged a $35 overdraft fee I did not authorize."},
       expected_outputs=["category"],
   )
   state = AgentState(task=task)

Implement the agent
-------------------

Subclass :class:`~forgeloop.agents.core.BaseAgent` and implement
``propose_action``. It reads the current state and returns the next typed
action; the loop, not the agent, acts on :class:`~forgeloop.agents.core.Finish`
and :class:`~forgeloop.agents.core.Escalate`. The default ``update`` advances the
step and appends non-empty observations; override it to control how an
observation folds into state.

.. code-block:: python

   from forgeloop.agents.core import BaseAgent, ToolCall, Finish

   class ClassifyThenFinish(BaseAgent):
       def propose_action(self, state):
           if not state.tool_results:
               return ToolCall(
                   tool_name="classify_complaint",
                   arguments={"message": state.task.inputs["message"]},
               )
           return Finish(output=state.tool_results[-1])

       def update(self, state, action, observation):
           new_state = state.model_copy(deep=True)
           new_state.step += 1
           if action.kind == "tool_call":
               new_state.tool_results.append(observation["output"])
           return new_state

Provide an environment
----------------------

:class:`~forgeloop.agents.core.Environment` is a ``runtime_checkable`` protocol:
any object with ``step(action) -> dict`` satisfies it. The environment is the
one place a :class:`~forgeloop.agents.core.ToolCall` becomes a real effect, so it
is where validation belongs. Here a minimal environment dispatches on
``tool_name``; the typed tool registry that validates arguments against a schema
arrives in :doc:`05_tools_as_typed_actions`.

.. code-block:: python

   def classify_complaint(message: str) -> dict:
       return {"category": "billing_dispute"}

   class ToolEnvironment:
       def __init__(self, tools):
           self.tools = tools

       def step(self, action):
           output = self.tools[action.tool_name](**action.arguments)
           return {"output": output}

Run the loop and inspect it
---------------------------

:func:`~forgeloop.agents.core.run_loop` is a generator. Each
:class:`~forgeloop.agents.core.StepRecord` exposes ``step``, ``state_before``,
``action``, ``observation`` and ``state_after``, so nothing about a step is
hidden from a caller, a logger or a test.

.. code-block:: python

   from forgeloop.agents.core import run_loop

   agent = ClassifyThenFinish()
   env = ToolEnvironment({"classify_complaint": classify_complaint})

   for record in run_loop(agent, env, state, max_steps=4):
       print(f"step {record.step}: {record.action.kind:10s} "
             f"-> status {record.state_after.status}")

   # step 0: tool_call  -> status running
   # step 1: finish     -> status done

Termination
-----------

The loop stops when the agent proposes :class:`~forgeloop.agents.core.Finish` or
:class:`~forgeloop.agents.core.Escalate`, when ``state.status`` leaves
``"running"``, or when ``max_steps`` is reached. Passing a
:class:`~forgeloop.agents.core.BudgetTracker` adds a fourth exit: when any axis
is exhausted the loop yields a synthetic ``Escalate`` with
``context={"source": "budget"}`` and sets ``status="failed"`` — the subject of
:doc:`07_cost_latency_budgets`.

Where governance attaches
--------------------------

``run_loop`` has no built-in authorize step: the proposer proposes, the
environment executes. Governance is inserted as a gate that screens a proposed
action *before* it reaches the environment, so a rejected proposal never runs.
:mod:`forgeloop.agents.governance` provides that layer —
:class:`~forgeloop.agents.governance.Gate` returning a
:class:`~forgeloop.agents.governance.GateResult`, with concrete gates such as
:class:`~forgeloop.agents.governance.PolicyGate` and
:class:`~forgeloop.agents.governance.StateInvariantGate`. Because a gate checks
the proposed next state rather than the action label, replacing a rule-based
proposer with a model does not weaken it. This is the pattern the rest of the
part develops: the model proposes, a deterministic substrate approves. See
:doc:`12_runtime_governance`.

See also
--------

- :doc:`/3-api-reference/modules/core/index` — full signatures for these types.
- :doc:`04_tasks_state_actions` and :doc:`05_tools_as_typed_actions` — the task,
  state and typed-tool layers in depth.
- :doc:`/4-notebook-examples/agents/index` — the same loop run end to end.
