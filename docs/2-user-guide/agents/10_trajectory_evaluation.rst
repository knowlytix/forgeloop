Trajectory Evaluation and Metrics
=================================

The final answer of an agent run underdetermines whether the run was correct,
because the same answer can be reached by a path that consulted the right policy
and escalated the regulated step or by one that guessed and skipped the gate. The
object that distinguishes the two is the record of the run, and this page shows
how to build that record with :func:`~forgeloop.agents.evaluation.collect` and
score it with the pure functions in :mod:`forgeloop.agents.evaluation`. Every
metric is a function of a :class:`~forgeloop.agents.evaluation.Trajectory` alone,
so a check re-runs later from a stored trajectory without the live agent.

The evaluation surface
----------------------

The metrics divide into two groups: functions over a whole trajectory and
functions over the claims in a final answer.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symbol
     - Role
   * - :class:`~forgeloop.agents.evaluation.Trajectory`
     - The run record: ``task``, ``records``, timing.
   * - :func:`~forgeloop.agents.evaluation.collect`
     - Drains ``run_loop`` step records into a trajectory.
   * - :func:`~forgeloop.agents.evaluation.summarize`
     - Aggregates status, step and tool counts and terminal flags.
   * - :func:`~forgeloop.agents.evaluation.task_success`, :func:`~forgeloop.agents.evaluation.escalated`, :func:`~forgeloop.agents.evaluation.failed`
     - Single-purpose trajectory metrics that compose into a suite report.
   * - :func:`~forgeloop.agents.evaluation.extract_claims`, :func:`~forgeloop.agents.evaluation.groundedness_report`, :func:`~forgeloop.agents.evaluation.coverage`
     - Answer groundedness: split, check and aggregate against an evidence map.

Collecting trajectories
-----------------------

:func:`~forgeloop.agents.evaluation.collect` drains the
:class:`~forgeloop.agents.core.StepRecord` generator returned by
:func:`~forgeloop.agents.core.run_loop` into a
:class:`~forgeloop.agents.evaluation.Trajectory` and stamps ``ended_at``. Three
runs that terminate differently give the metrics more than the success path to
report on: a clean finish, an escalation and a budget exhaustion. Each action
runs through the :doc:`06_safe_tool_execution` executor and a real tool, so an
observation carries a genuine ``success`` field rather than a canned value.

.. code-block:: python

   from pydantic import BaseModel
   from forgeloop.agents.core import (
       AgentState, BaseAgent, Budget, BudgetTracker, Escalate, Finish, TaskSpec,
       ToolCall, run_loop,
   )
   from forgeloop.agents.evaluation import collect
   from forgeloop.agents.tools import GovernedToolExecutor, RiskLevel, Tool, ToolRegistry

   class PingNTimes(BaseAgent):
       def __init__(self, n): self.n = n
       def propose_action(self, state):
           if state.step >= self.n:
               return Finish(output="done")
           return ToolCall(tool_name="ping", arguments={})

   class AlwaysEscalate(BaseAgent):
       def propose_action(self, state):
           return Escalate(reason="needs human")

   class PingIn(BaseModel): pass
   class PingOut(BaseModel): ok: bool

   registry = ToolRegistry()
   registry.register(Tool(name="ping", description="a no-op tool",
                          input_schema=PingIn, output_schema=PingOut,
                          risk=RiskLevel.LOW, fn=lambda: {"ok": True}))

   class ToolEnv:
       def __init__(self, executor): self._ex = executor
       def step(self, action):
           r = self._ex.execute(action)
           return {"success": r.success, "output": r.output, "error": r.error}

   env = ToolEnv(GovernedToolExecutor(registry))
   task = TaskSpec(goal="trajectory demo")

   happy   = collect(task, run_loop(PingNTimes(3), env, AgentState(task=task), max_steps=10))
   esc     = collect(task, run_loop(AlwaysEscalate(), env, AgentState(task=task), max_steps=10))
   starved = collect(task, run_loop(PingNTimes(10), env, AgentState(task=task), max_steps=10,
                                    budget_tracker=BudgetTracker(Budget(tool_calls=2))))

The budget-limited run exhausts its two-call allowance before ``PingNTimes(10)``
proposes ``Finish``, so :func:`~forgeloop.agents.core.run_loop` yields a synthetic
:class:`~forgeloop.agents.core.Escalate` and leaves the run in ``status="failed"``
(see :doc:`07_cost_latency_budgets`).

Composable metrics
------------------

:func:`~forgeloop.agents.evaluation.summarize` returns the standard fields in one
dict; the individual metrics are used where a report needs only part of that.

.. code-block:: python

   from forgeloop.agents.evaluation import (
       summarize, task_success, finished_cleanly, tool_failure_count,
   )

   for name, traj in [("happy", happy), ("escalated", esc), ("budget-failed", starved)]:
       s = summarize(traj)
       print(f'{name:>14}: status={s["status"]:<10} steps={s["steps"]:<2} '
             f'tool_calls={s["tool_calls"]:<2} escalated={s["escalated"]} failed={s["failed"]}')

Because each metric is a plain function of a trajectory, a suite metric is the
same functions averaged over many trajectories, with no separate reporting
machinery.

.. code-block:: python

   def suite_report(trajs):
       n = len(trajs)
       return {
           "success_rate":    sum(task_success(t) for t in trajs) / n,
           "escalation_rate": sum(escalated(t) for t in trajs) / n,
           "clean_rate":      sum(finished_cleanly(t) for t in trajs) / n,
       }

Note that :func:`~forgeloop.agents.evaluation.finished_cleanly` counts an
escalation as a clean termination, whereas
:func:`~forgeloop.agents.evaluation.task_success` counts only ``status="done"``,
so an agent that escalates a case it should not have handled is not scored as a
success.

Groundedness of an answer
--------------------------

A trajectory metric reports how the run ended; groundedness reports whether the
final answer is supported by observed evidence.
:func:`~forgeloop.agents.evaluation.extract_claims` splits an answer into
sentence-level :class:`~forgeloop.agents.evaluation.Claim` values,
:func:`~forgeloop.agents.evaluation.groundedness_report` classifies each against
an evidence map by content-term overlap, and
:func:`~forgeloop.agents.evaluation.coverage` reports the supported fraction. A
claim with no evidence match is classified UNSUPPORTED rather than dropped.

.. code-block:: python

   from forgeloop.agents.evaluation import extract_claims, groundedness_report, coverage

   answer = (
       "The overdraft fee is thirty-five dollars per occurrence. "
       "Customers are notified within one business day. "
       "Dinosaurs were vegetarian."
   )
   evidence = {
       "overdraft": "Overdraft fees apply at thirty-five dollars per occurrence.",
       "notice":    "Customers are notified within one business day of the posted overdraft.",
   }

   report = groundedness_report(extract_claims(answer), evidence)
   for r in report:
       print(f"  [{r.verdict.value:<11}] {r.claim!r}  evidence={r.evidence_id}")
   print(f"coverage: {coverage(report) * 100:.0f}%")

The unsupported dinosaur sentence lowers coverage below one, which a ship-gate
reads directly: admit the answer when ``coverage(report) >= threshold``, else
return the unsupported claims for review.

Deterministic verification against a store
------------------------------------------

The overlap check in
:func:`~forgeloop.agents.evaluation.check_groundedness` is a term-matching
baseline. A stricter verifier reduces a claim to a triple and scores it against a
knowledge store as a geodesic distance, where a smaller distance is stronger
support. :class:`forgeloop.agents.gms_backend.GMSMemory` wraps a
``GMSExpertStore`` and exposes ``score_triple``. The call below requires
``torch`` and ``knowlytix`` and a built store, so it is shown as it runs rather
than executed here.

.. code-block:: python

   import torch
   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
   from forgeloop.agents.gms_backend import GMSMemory

   store = GMSExpertStore(
       DocGMSConfig(store_path="data/gms_banking_store"),
       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
   )
   store.load()
   mem = GMSMemory(store)

   for claim, (h, r, t) in [("overdraft fee is $35", ("overdraft", "has_fee_amount", "35.0")),
                            ("overdraft fee is $50", ("overdraft", "has_fee_amount", "50.0"))]:
       print(f"  {claim:<22} groundedness={mem.score_triple(h, r, t):.3f}")

The score is a function of a frozen store, so it replays byte-for-byte, which a
second-model judge does not.

See also
--------

- :doc:`/3-api-reference/modules/evaluation/index` — full signatures for the metrics,
  the trajectory type and the groundedness functions.
- :doc:`01_what_is_an_agent` — the loop and the ``StepRecord`` that ``collect``
  drains.
- :doc:`11_failure_modes_doe` — generating the trajectories a suite metric scores.
- :doc:`12_runtime_governance` — the gates whose firing these metrics report on.
- :doc:`/4-notebook-examples/agents/index` — the three runs collected and scored
  end to end.
