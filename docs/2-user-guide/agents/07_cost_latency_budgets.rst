Bounding a Run with a Budget
============================

This page shows how to cap what a single run may consume and have
:func:`~forgeloop.agents.core.run_loop` halt with an ordinary terminal record
when a cap is reached. A tool that returns a transient error and an agent that
retries it produces a loop that is not unsafe in the sense governance guards, yet
never terminates and accrues cost on every call. A budget is a hard limit on the
run, checked by the loop itself, not a value the agent reasons about.

The pieces
----------

.. list-table::
   :header-rows: 1

   * - Type
     - Role
   * - :class:`~forgeloop.agents.core.Budget`
     - Per-axis caps: tokens, seconds, tool calls, dollars; ``None`` is unlimited.
   * - :class:`~forgeloop.agents.core.Consumption`
     - Accumulated usage across the same four axes.
   * - :class:`~forgeloop.agents.core.BudgetTracker`
     - Accumulates a ``Consumption`` and reports the first axis to exhaust.
   * - :func:`~forgeloop.agents.core.run_loop`
     - Reads the tracker at the top of each step and halts on exhaustion.

Define a budget
---------------

A :class:`~forgeloop.agents.core.Budget` caps four axes independently. An unset
axis is left unlimited, so a budget can constrain tool calls while leaving tokens
and time open.

.. code-block:: python

   from forgeloop.agents.core import Budget

   generous = Budget(tool_calls=100)
   tight    = Budget(tool_calls=5)
   hostile  = Budget(tool_calls=1)

Attach a tracker to the loop
----------------------------

A :class:`~forgeloop.agents.core.BudgetTracker` wraps a budget and is passed to
:func:`~forgeloop.agents.core.run_loop` as ``budget_tracker``. The loop records a
tool call each time it steps a :class:`~forgeloop.agents.core.ToolCall` and reads
:meth:`~forgeloop.agents.core.BudgetTracker.exhausted` at the top of each
iteration. The run here drives a real tool through the governed executor of
:doc:`06_safe_tool_execution` and collects the trajectory with
:func:`~forgeloop.agents.evaluation.collect` and
:func:`~forgeloop.agents.evaluation.summarize`.

.. code-block:: python

   from pydantic import BaseModel
   from forgeloop.agents.core import (
       AgentState, BaseAgent, BudgetTracker, Finish, TaskSpec, ToolCall, run_loop,
   )
   from forgeloop.agents.evaluation import collect, summarize
   from forgeloop.agents.tools import GovernedToolExecutor, RiskLevel, Tool, ToolRegistry

   class PingAgent(BaseAgent):
       def propose_action(self, state):
           if state.step >= 5:
               return Finish(output="done")
           return ToolCall(tool_name="ping", arguments={})

   class PingIn(BaseModel): pass
   class PingOut(BaseModel): pong: bool

   registry = ToolRegistry()
   registry.register(Tool(name="ping", description="a no-op tool",
                          input_schema=PingIn, output_schema=PingOut,
                          risk=RiskLevel.LOW, fn=lambda: {"pong": True}))

   class ToolEnv:
       def __init__(self, executor):
           self._ex = executor
       def step(self, action):
           return {"success": self._ex.execute(action).success}

   env = ToolEnv(GovernedToolExecutor(registry))
   task = TaskSpec(goal="do 5 pings")

   def run_with(budget: Budget) -> dict:
       tracker = BudgetTracker(budget)
       traj = collect(task, run_loop(PingAgent(), env, AgentState(task=task),
                                     max_steps=20, budget_tracker=tracker))
       s = summarize(traj)
       s["budget_reason"] = tracker.reason_exhausted()
       return s

Under the generous budget the loop reaches :class:`~forgeloop.agents.core.Finish`
and the summary reports ``status="done"``; under the hostile budget it halts early
and reports ``status="failed"``.

.. code-block:: python

   for name, b in [("generous", generous), ("tight", tight), ("hostile", hostile)]:
       s = run_with(b)
       print(f"{name:>10}: status={s['status']:<8} "
             f"tool_calls={s['tool_calls']:<2} reason={s['budget_reason']!r}")

The synthetic terminal record
------------------------------

An exhausted axis does not break out of the loop. The loop yields one more
:class:`~forgeloop.agents.core.StepRecord` whose action is an
:class:`~forgeloop.agents.core.Escalate` carrying ``context={"source": "budget"}``,
and sets ``status="failed"``. A budget failure therefore has the same record shape
as any terminal step, so the audit chain and trajectory evaluation need no special
case.

.. code-block:: python

   tracker = BudgetTracker(Budget(tool_calls=1))
   traj = collect(task, run_loop(PingAgent(), env, AgentState(task=task),
                                 max_steps=20, budget_tracker=tracker))
   last = traj.records[-1]
   print("final action kind:", last.action.kind)          # escalate
   print("context:", getattr(last.action, "context", None))  # {'source': 'budget'}

Because the tracker is read at the top of each iteration, a budget equal to the
work still fails: five pings reach the cap of five before the
:class:`~forgeloop.agents.core.Finish` step runs, and a cap of six leaves room for
it.

.. code-block:: python

   print("tool_calls=5:", run_with(Budget(tool_calls=5))["status"])  # failed
   print("tool_calls=6:", run_with(Budget(tool_calls=6))["status"])  # done

Tokens and dollars from measured usage
---------------------------------------

Tokens are the axis a run exhausts most often, and the model adapter is the only
component that knows the true count. :class:`~forgeloop.agents.models.QwenAdapter`
reports ``last_input_tokens`` and ``last_output_tokens`` after each
:meth:`~forgeloop.agents.models.QwenAdapter.complete` call, the real counts rather
than an estimate from prompt length. The dollar axis is then those tokens times a
rate fixed by the deployment contract. The following runs a real model and is
reproduced from the notebook rather than executed on this page.

.. code-block:: python

   from forgeloop.agents.models import QwenAdapter

   lm = QwenAdapter()
   lm.complete("warm up")                      # the first call loads the model
   lm.complete("Summarize the overdraft fee policy in one sentence.")

   price_per_1k_tokens = 0.0005                # the deployment's rate, in USD
   tokens = lm.last_input_tokens + lm.last_output_tokens
   print(f"tokens: {tokens}")
   print(f"cost: ${tokens / 1000 * price_per_1k_tokens:.6f}")

The budget is enforced in the loop and never passed to the agent, so the limit
stays outside the object it is meant to contain.

See also
--------

- :doc:`/3-api-reference/modules/core/index` — full signatures for
  :class:`~forgeloop.agents.core.Budget`,
  :class:`~forgeloop.agents.core.BudgetTracker` and
  :class:`~forgeloop.agents.core.Consumption`.
- :doc:`01_what_is_an_agent` — the loop termination conditions this adds a fourth
  exit to.
- :doc:`06_safe_tool_execution` — the governed executor driven above.
- :doc:`/4-notebook-examples/agents/index` — the budgeted run end to end.
