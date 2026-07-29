Chapter 8 — Planning, Decomposition and Replanning
==================================================

This page shows how to produce a plan with the three planner families in
:mod:`forgeloop.agents.planning`, split a plan into subtasks with ``decompose``,
trigger a replan when a step fails, and reject an incoherent plan before any tool
runs with :class:`~forgeloop.agents.gms_backend.GMSPlanGate`. A plan here is a
typed list of :class:`~forgeloop.agents.planning.PlanStep` values, not a
paragraph, so the budget tracker can count the intended calls, the audit log can
record what was committed to, and the evaluator can compare steps taken against
steps planned.

The pieces
----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Type
     - Role
   * - :class:`~forgeloop.agents.planning.Plan`, :class:`~forgeloop.agents.planning.PlanStep`
     - A plan and its typed, ordered steps.
   * - :class:`~forgeloop.agents.planning.Planner`
     - Protocol: ``plan(task) -> Plan``.
   * - :class:`~forgeloop.agents.planning.WorkflowPlanner`
     - Returns a fixed list of steps every time.
   * - :class:`~forgeloop.agents.planning.LMPlanner`
     - Asks a model for a JSON array of steps, or raises.
   * - :class:`~forgeloop.agents.planning.GraphSearchPlanner`
     - Breadth-first search over a state-transition graph.
   * - ``decompose`` / ``replan`` / ``should_replan``
     - Split a plan, and re-plan after a failure.
   * - :class:`~forgeloop.agents.gms_backend.GMSPlanGate`
     - Rejects a plan whose coherence drift exceeds a budget.

Plan a fixed workflow
---------------------

:class:`~forgeloop.agents.planning.WorkflowPlanner` returns the same steps for any
task, which is the reliable choice when the procedure is settled.
:meth:`~forgeloop.agents.planning.WorkflowPlanner.plan` takes a
:class:`~forgeloop.agents.core.TaskSpec` and returns a
:class:`~forgeloop.agents.planning.Plan`.

.. code-block:: python

   from forgeloop.agents.core import TaskSpec
   from forgeloop.agents.planning import PlanStep, WorkflowPlanner

   task = TaskSpec(
       goal='Handle a customer complaint about overdraft fees',
       inputs={'message': 'I was charged $35 unfairly', 'start': 'start'},
       expected_outputs=['drafted'],
   )
   workflow = WorkflowPlanner([
       PlanStep(id='classify', description='classify the complaint', action_hint='classify'),
       PlanStep(id='retrieve', description='retrieve relevant policy', action_hint='search'),
       PlanStep(id='draft',    description='draft a response',        action_hint='compose'),
   ])
   for s in workflow.plan(task).steps:
       print(f'  {s.id}: {s.description}')

Two alternative planners
------------------------

:class:`~forgeloop.agents.planning.LMPlanner` prompts a language model for a JSON
array of steps and parses the reply.
:meth:`~forgeloop.agents.planning.LMPlanner.plan` raises ``ValueError`` on a reply
that is not valid JSON or is not a JSON array, so a malformed plan reaches the
audit log rather than becoming a silent bad plan. It needs a model with a
``complete(prompt) -> str`` method, so the block below is illustrative.

.. code-block:: python

   from forgeloop.agents.models import QwenAdapter
   from forgeloop.agents.planning import LMPlanner

   lm_plan = LMPlanner(QwenAdapter()).plan(task)
   for s in lm_plan.steps:
       print(f'  {s.id}: {s.description[:60]}')

:class:`~forgeloop.agents.planning.GraphSearchPlanner` runs breadth-first search
over a graph mapping ``state -> {action: next_state}``, from ``task.inputs['start']``
to the first expected output, which is exact when every state can be enumerated.

.. code-block:: python

   from forgeloop.agents.planning import GraphSearchPlanner

   graph = {
       'start':      {'classify': 'classified'},
       'classified': {'search':   'searched'},
       'searched':   {'compose':  'drafted'},
   }
   for s in GraphSearchPlanner(graph).plan(task).steps:
       print(f'  {s.id}: {s.action_hint}')

Decompose a plan into subtasks
------------------------------

``decompose`` turns a plan into one :class:`~forgeloop.agents.core.TaskSpec` per
step, each taking the step description as its goal and carrying the parent goal
and step id in its inputs under ``_parent_goal`` and ``_step_id``, so a subtask
correlates back to the plan.

.. code-block:: python

   from forgeloop.agents.planning import decompose

   for sub in decompose(task, workflow.plan(task)):
       print(f'  {sub.goal!r}  parent={sub.inputs["_parent_goal"]!r}  step={sub.inputs["_step_id"]!r}')

Replan after a failure
----------------------

``should_replan`` inspects the last tool result and returns a
:class:`~forgeloop.agents.planning.ReplanTrigger` when the result reports failure,
or ``None`` otherwise. ``replan`` annotates the task with the trigger reason and
asks the planner again. The trigger's ``reason`` is a
:class:`~forgeloop.agents.planning.ReplanReason`; a failed call yields
``TOOL_FAILURE``.

.. code-block:: python

   from forgeloop.agents.planning import replan, should_replan

   trigger = should_replan({'success': False, 'error': 'rate limit'})
   print(trigger.reason)                    # ReplanReason.TOOL_FAILURE
   new_plan = replan(workflow, task, trigger)
   print('new plan length:', len(new_plan))

Because ``should_replan`` returns ``None`` on a success, a replan loop terminates;
a caller bounds it with a maximum count and escalates once the count is reached
rather than replanning indefinitely.

Validate a plan before it runs
------------------------------

A plan is a hypothesis, so it is checked before any tool fires.
:class:`~forgeloop.agents.gms_backend.GMSPlanGate` scores every consecutive
transition ``(step_i, has_enables, step_{i+1})`` against a trained store, averages
the geodesic distances into a mean coherence drift, and returns a
:class:`~forgeloop.agents.gms_backend.PlanVerdict` whose ``admissible`` flag is
false when ``drift`` exceeds ``drift_budget``. The block below requires
``knowlytix`` and ``torch`` and is not executed on this page.

.. code-block:: python

   import torch
   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
   from forgeloop.agents.gms_backend import GMSPlanGate

   store = GMSExpertStore(
       DocGMSConfig(store_path='data/gms_banking_store'),
       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
   store.load()
   gate = GMSPlanGate(store, drift_budget=0.4)

   for label, plan in [
       ('drafting path',   ['classify', 'extract', 'search_policy', 'draft_response']),
       ('escalation path', ['classify', 'extract', 'flag_regulatory', 'escalate']),
       ('skips two steps', ['classify', 'draft_response']),
   ]:
       v = gate.validate(plan)
       print(f'  {label:<16} drift={v.drift:.3f}  admissible={v.admissible}')

A plan that skips or reorders steps raises the mean transition distance and is
refused with no state change, whereas the same illegal step checked at execution
time surfaces only after the earlier steps had already acted on the world. A
refusal triggers a replan rather than ending the run.

Where per-step and whole-plan checks differ
--------------------------------------------

:class:`~forgeloop.agents.gms_backend.GMSPlausibilityGate` from
:doc:`06_safe_tool_execution` scores a single transition at execution time;
:class:`~forgeloop.agents.gms_backend.GMSPlanGate` scores the whole plan at plan
time. The plan gate reads ``PlanVerdict.transitions`` for the ordered
``(from_step, to_step, score)`` list, so the transition that contributed most to
the drift is visible. Scoring the plan before execution is why a bad plan costs
nothing.

See also
--------

- :doc:`/3-api-reference/modules/planning/index` — the planners, ``decompose``, ``replan`` and ``should_replan``.
- :doc:`/3-api-reference/modules/gms_backend/index` — :class:`~forgeloop.agents.gms_backend.GMSPlanGate` and :class:`~forgeloop.agents.gms_backend.PlanVerdict`.
- :doc:`07_cost_latency_budgets` — the budget tracker that counts a plan's intended calls.
- :doc:`/4-notebook-examples/agents/index` — the planners and plan gate run end to end.
