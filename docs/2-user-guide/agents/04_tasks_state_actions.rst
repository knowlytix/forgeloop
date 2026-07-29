Chapter 4 — Typing the Task, State and Actions
==============================================

This page shows how to replace the three loose parts of an early loop, a task
that is a bare string, an action that is a free-form dict and state kept in
module variables, with the typed objects in :mod:`forgeloop.agents.core`. The
behavior is the loop already built in :doc:`01_what_is_an_agent`; what changes is
that every step now serializes to JSON and reconstructs from it, which is the
precondition for the audit log and replay used in :doc:`12_runtime_governance`.

The pieces
----------

.. list-table::
   :header-rows: 1

   * - Type
     - Role
   * - :class:`~forgeloop.agents.core.TaskSpec`
     - Goal, inputs, expected outputs, constraints, validation.
   * - :class:`~forgeloop.agents.core.ValidationRule`
     - A named criterion a downstream check reads.
   * - :class:`~forgeloop.agents.core.AgentState`
     - Serializable state carried across steps.
   * - :class:`~forgeloop.agents.core.ToolCall`,
     - The four-member discriminated union of actions,
   * - :class:`~forgeloop.agents.core.AskUser`,
     - each tagged by a ``kind`` ``Literal``
   * - :class:`~forgeloop.agents.core.Finish`,
     - (with :class:`~forgeloop.agents.core.ActionKind`).
   * - :class:`~forgeloop.agents.core.Escalate`
     - 
   * - :func:`~forgeloop.agents.core.parse_action`
     - Rebuilds a typed action from a logged dict.

Define a task
-------------

A :class:`~forgeloop.agents.core.TaskSpec` gathers the goal with its inputs,
expected outputs, constraints and a list of
:class:`~forgeloop.agents.core.ValidationRule`. Only the goal is required and it
cannot be blank, so a malformed request is rejected before the agent runs.
Constraints stay plain strings because the same text travels to the model prompt
and to the enforcement gates.

.. code-block:: python

   from forgeloop.agents.core import TaskSpec, ValidationRule

   task = TaskSpec(
       goal="Classify and summarize a customer complaint",
       inputs={"message": "I was charged a $35 overdraft fee."},
       expected_outputs=["category", "summary"],
       constraints=["Do not invent facts", "Do not expose PII"],
       validation=[ValidationRule(name="schema", description="must match output schema")],
   )

The blank-goal case raises ``pydantic.ValidationError`` at construction.

.. code-block:: python

   from pydantic import ValidationError

   try:
       TaskSpec(goal="   ")
   except ValidationError as e:
       print("rejected empty goal:", e.errors()[0]["msg"])

Round-trip the state
--------------------

:class:`~forgeloop.agents.core.AgentState` serializes with
:meth:`~forgeloop.agents.core.AgentState.to_dict` and reconstructs with
:meth:`~forgeloop.agents.core.AgentState.from_dict`. The serialized form is the
unit of audit, replay and persistence, and it round-trips without loss. Scratchpad
entries and tool results are held as opaque dicts, so the reasoning and tool
sub-packages evolve without changing the schema the audit log depends on.

.. code-block:: python

   from forgeloop.agents.core import AgentState

   state = AgentState(task=task, step=2, messages=[{"role": "user", "content": "hi"}])
   back = AgentState.from_dict(state.to_dict())
   print("round-trip equal:", back == state)

Construct the four actions
--------------------------

The decision is a discriminated union of four frozen types. The ``kind`` field is
a ``Literal`` on each subclass, so a value tagged with the wrong kind fails at
construction rather than at dispatch.

.. code-block:: python

   from forgeloop.agents.core import ToolCall, AskUser, Finish, Escalate

   actions = [
       ToolCall(tool_name="search", arguments={"q": "overdraft"}),
       AskUser(question="Did you authorize this transaction?"),
       Finish(output={"category": "complaint"}),
       Escalate(reason="UDAAP risk", context={"flags": ["fee", "overdraft"]}),
   ]
   for a in actions:
       print(f"{a.kind:<10} {type(a).__name__}")

   try:
       ToolCall(kind="ask_user", tool_name="x")
   except ValidationError as e:
       print("wrong kind rejected:", e.errors()[0]["msg"])

Replay actions from a log
-------------------------

:func:`~forgeloop.agents.core.parse_action` reads a saved dict, dispatches on its
``kind`` and returns the matching subclass. It is the single entry point back from
logged data to a live action, which is how the audit log in
:doc:`12_runtime_governance` replays a run.

.. code-block:: python

   from forgeloop.agents.core import parse_action

   for a in actions:
       rebuilt = parse_action(a.model_dump())
       assert type(rebuilt) is type(a)

An unrecognized ``kind`` raises ``ValueError`` rather than returning a partially
built object.

Drive the typed loop
--------------------

The typed parts drop into the loop unchanged.
:func:`~forgeloop.agents.core.run_loop` yields one
:class:`~forgeloop.agents.core.StepRecord` per step, and each record's
``state_after`` serializes with the same
:meth:`~forgeloop.agents.core.AgentState.to_dict`, so a completed run is an
audit-ready sequence of dicts.

.. code-block:: python

   from forgeloop.agents.core import BaseAgent, run_loop

   class GreedyAgent(BaseAgent):
       def __init__(self, target):
           self.target = target
       def propose_action(self, state):
           if state.step >= self.target:
               return Finish(output={"pings": state.step})
           return ToolCall(tool_name="ping", arguments={})

   class PingEnv:
       def step(self, action):
           return {"pong": True}

   records = list(run_loop(GreedyAgent(2), PingEnv(), AgentState(task=task), max_steps=10))
   for r in records:
       print(r.step, r.action.kind, r.state_after.status)

The final record carries a :class:`~forgeloop.agents.core.Finish` action and a
``state_after`` that reconstructs identically through
:meth:`~forgeloop.agents.core.AgentState.from_dict`, which is the property the
audit chain and trajectory evaluation depend on.

See also
--------

- :doc:`/3-api-reference/modules/core/index` — full signatures for these types.
- :doc:`01_what_is_an_agent` — the loop these types are refitted into.
- :doc:`05_tools_as_typed_actions` — validating a
  :class:`~forgeloop.agents.core.ToolCall` against a schema.
- :doc:`/4-notebook-examples/agents/index` — the same refit run end to end.
