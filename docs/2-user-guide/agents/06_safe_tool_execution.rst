Safe Tool Execution and Tool Testing
====================================

This page shows how to run a tool call through an ordered stack of gates with
:class:`~forgeloop.agents.tools.GovernedToolExecutor`, write a custom policy gate,
replace the default plausibility check with a geometric one, and stress a tool's
schema with a fuzzer. Each gate covers a different class of failure and returns a
:class:`~forgeloop.agents.tools.GateResult`; a deny or escalate at any layer stops
the call, and every result is recorded whether or not the tool runs.

The pieces
----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Type
     - Role
   * - :class:`~forgeloop.agents.tools.GovernedToolExecutor`
     - Runs the gates in order, then executes the tool's ``fn``.
   * - :class:`~forgeloop.agents.tools.Gate`
     - Protocol: ``check(action, state, registry) -> GateResult``.
   * - :class:`~forgeloop.agents.tools.GateResult`, :class:`~forgeloop.agents.tools.GateDecision`
     - A decision (``ALLOW`` / ``DENY`` / ``ESCALATE``) with a reason.
   * - :class:`~forgeloop.agents.tools.SyntaxGate`
     - Denies unknown tools and arguments failing schema validation.
   * - :class:`~forgeloop.agents.tools.PolicyGate`
     - Aggregates ``PolicyCheck`` functions of ``(action, state)``.
   * - :class:`~forgeloop.agents.tools.PlausibilityGate`
     - Rejects non-serializable or oversized arguments.
   * - :class:`~forgeloop.agents.tools.ToolResult`
     - Output or error, the gate results and a success flag.

Execute a call through the default gates
----------------------------------------

A tool carries its implementation in ``fn``.
:class:`~forgeloop.agents.tools.GovernedToolExecutor` defaults to three gates,
:class:`~forgeloop.agents.tools.SyntaxGate`, :class:`~forgeloop.agents.tools.PolicyGate`
and :class:`~forgeloop.agents.tools.PlausibilityGate`, run in that order.
:meth:`~forgeloop.agents.tools.GovernedToolExecutor.execute` takes a
:class:`~forgeloop.agents.core.ToolCall` and returns a
:class:`~forgeloop.agents.tools.ToolResult`. A call with an ill-typed argument is
denied by the syntax gate, and the denial carries the gate results that produced
it.

.. code-block:: python

   from pydantic import BaseModel
   from forgeloop.agents.core import ToolCall
   from forgeloop.agents.tools import GovernedToolExecutor, RiskLevel, Tool, ToolRegistry

   class AdderIn(BaseModel):
       x: int
   class AdderOut(BaseModel):
       y: int

   def _add_one(x: int) -> dict:
       return {'y': x + 1}

   adder = Tool(name='adder', description='add one to x',
                input_schema=AdderIn, output_schema=AdderOut,
                risk=RiskLevel.LOW, fn=_add_one)
   registry = ToolRegistry()
   registry.register(adder)

   executor = GovernedToolExecutor(registry)   # uses the three default gates
   ok = executor.execute(ToolCall(tool_name='adder', arguments={'x': 5}))
   print('success:', ok.success, 'output:', ok.output)

   bad = executor.execute(ToolCall(tool_name='adder', arguments={'x': 'not-an-int'}))
   print('success:', bad.success, 'error:', bad.error)
   print('gates:', [(g.gate_name, g.decision.value) for g in bad.gate_results])

Express policy as code
----------------------

A ``PolicyCheck`` is a function from ``(action, state)`` to a
:class:`~forgeloop.agents.tools.GateResult`, so a reviewer reads the rule and the
audit log together. Passing an explicit ``gates``
list replaces the defaults. A check that returns ``ESCALATE`` routes the call to a
human rather than denying it outright.

.. code-block:: python

   from forgeloop.agents.tools import (
       GovernedToolExecutor, PlausibilityGate, PolicyGate, SyntaxGate,
   )
   from forgeloop.agents.tools.executor import GateDecision, GateResult

   def small_x_only(action, state):
       if action.tool_name == 'adder' and action.arguments.get('x', 0) > 100:
           return GateResult(GateDecision.DENY, 'small_x_only', 'x must be <= 100')
       return GateResult(GateDecision.ALLOW, 'small_x_only')

   policy_exec = GovernedToolExecutor(registry,
       gates=[SyntaxGate(), PolicyGate(policies=[small_x_only]), PlausibilityGate()])
   blocked = policy_exec.execute(ToolCall(tool_name='adder', arguments={'x': 1000}))
   print('blocked:', blocked.error)

The default :class:`~forgeloop.agents.tools.PlausibilityGate` is a size bound: it
rejects arguments that are not JSON-serializable or exceed ``max_args_size``,
which catches a call whose payload has grown implausibly large.

.. code-block:: python

   tight = GovernedToolExecutor(registry, gates=[SyntaxGate(), PlausibilityGate(max_args_size=50)])
   result = tight.execute(ToolCall(tool_name='adder', arguments={'x': int('1' * 60)}))
   print('blocked:', result.error)

A geometric plausibility gate
-----------------------------

The size bound does not know whether a well-formed, permitted call arrives in the
right order. :class:`~forgeloop.agents.gms_backend.GMSPlausibilityGate` scores the
``(context, relation, tool)`` triple against a trained store and denies a call
above a calibrated threshold ``theta``, catching a step that is syntactically and
policy-clean yet out of sequence. The block below requires ``knowlytix`` and
``torch`` and is not executed on this page; ``theta`` is read from the store's
persisted calibration.

.. code-block:: python

   import json, torch
   from pathlib import Path
   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
   from forgeloop.agents.gms_backend import GMSPlausibilityGate

   store = GMSExpertStore(
       DocGMSConfig(store_path='data/gms_banking_store'),
       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
   store.load()
   theta = json.loads(Path('data/gms_banking_store/calibration.json').read_text()
                      )['plausibility_gate']['threshold']

   gms_gate = GMSPlausibilityGate(store, theta=theta,
                                  context='classify', relation='has_enables')
   for tool in ('extract', 'search_policy', 'draft_response', 'wire_international'):
       verdict = gms_gate.check(ToolCall(tool_name=tool, arguments={}), None, registry)
       print(f'  classify -> {tool:<18} {verdict.decision.value}')

The gate returns the ``on_missing`` decision when the store has no information for
the triple, so an untrained pair does not silently pass as plausible.

Contract-test a tool with a fuzzer
----------------------------------

``fuzz_tool`` (from :mod:`forgeloop.agents.tools.testing`) supplies many generated
inputs to a tool and reports how many were accepted, rejected cleanly as a
``ValidationError`` or crashed. A well-behaved tool reports zero crashes, since
every input should either parse and run or raise a validation error. A non-zero
crash count marks an uncovered input path to close in the schema.

.. code-block:: python

   from forgeloop.agents.tools.testing import fuzz_tool

   report = fuzz_tool(adder, num_cases=200, seed=1)
   print('accepted:', report['accepted'],
         'rejected_clean:', report['rejected_clean'],
         'crashed:', report['crashed'])

Why the order is fixed
----------------------

The gates run cheapest and most objective first.
:meth:`~forgeloop.agents.tools.GovernedToolExecutor.execute` returns on the first
gate that denies or escalates and appends every result it reached to the
:class:`~forgeloop.agents.tools.ToolResult`, so a denial is recorded as faithfully
as a success. Because each later gate assumes the earlier ones passed, no single
gate has to be perfect; syntax rejects the malformed, policy rejects the
forbidden and plausibility rejects the anomalous. This is defense in depth applied
to tool use, and the same :class:`~forgeloop.agents.tools.Gate` protocol carries
the richer gates built in :doc:`12_runtime_governance`.

See also
--------

- :doc:`/3-api-reference/modules/tools/index` — :class:`~forgeloop.agents.tools.GovernedToolExecutor`, the gates and ``fuzz_tool``.
- :doc:`/3-api-reference/modules/gms_backend/index` — :class:`~forgeloop.agents.gms_backend.GMSPlausibilityGate`.
- :doc:`05_tools_as_typed_actions` — the tool, registry and routing layer these gates screen.
- :doc:`/4-notebook-examples/agents/index` — the gate stack and fuzzer run end to end.
