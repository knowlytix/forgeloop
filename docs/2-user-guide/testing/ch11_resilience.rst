Chapter 11 — Resilience Under Tool Faults
=========================================

This page shows how to probe whether a governed agent fails loud when a tool it
relies on fails. In production a tool failure is a certainty, and a governed
agent is required to escalate rather than proceed on a corrupted result. The
probe injects a fault into one tool at a time, runs the suite and measures the
detection rate: the fraction of runs in which the agent escalated rather than
continuing silently.

The measured quantity
---------------------

A run counts as detected when the faulted tool was reached and the agent failed
loud rather than finishing with a corrupted result. A correct agent has a
detection rate of one and a silent-failure count of zero on every tool.

.. code-block:: python

   def agent(tool_ok):
       # a governed agent escalates the moment a tool result is unusable
       return 'escalated' if not tool_ok else 'answered'

   runs = [agent(tool_ok=False) for _ in range(8)]     # tool faulted on every run
   detected = sum(r == 'escalated' for r in runs)
   print(f'detected {detected}/{len(runs)}  silent {len(runs) - detected}')

Injecting a real fault
----------------------

:meth:`CapstoneTestHarness.fault_injection
<forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.fault_injection>`
installs a knowlytix ``ToolGateway`` on the agent's
:class:`~forgeloop.agents.tools.executor.GovernedToolExecutor` through its
:class:`~forgeloop.agents.tools.executor.ToolHooks` contract. It forces one tool
at a time to fail under a ``FaultProfile``, runs the scenarios and records
whether each run detected the fault. It returns a
:class:`~forgeloop.agents.testing.capstone_harness.FaultInjectionResult` carrying
the per-tool detection rate.

.. code-block:: python

   from forgeloop.agents.testing import CapstoneTestHarness

   harness = CapstoneTestHarness()
   result = harness.fault_injection(fault='error', limit=8)

   for tool, agg in result.per_tool.items():
       print(f"{tool:18s} detection_rate={agg['detection_rate']} "
             f"silent={agg['silent']}")

The gateway is installed for the duration of one tool's injection and uninstalled
afterward, so each tool is probed in isolation and the agent is otherwise
unmodified.

The fault taxonomy
------------------

The faults order by how much work detection takes. The ``fault`` argument selects
which profile the gateway installs.

.. list-table::
   :header-rows: 1

   * - Fault
     - How it is detected
   * - ``error``
     - The tool returns nothing usable, so the agent must escalate.
   * - ``latency``
     - A delayed call, caught by the budget or timeout path.
   * - ``stale``
     - A well-formed but out-of-date result, harder to catch than an error.

The hardest fault is a plausible-but-wrong structured result, of the right type
and shape but the wrong content, which passes any syntactic check and is caught
only downstream by a verifier that compares it against the substrate. A
value-polarity verifier catches a drafted reply that reverses the stance of a
policy fact by scoring the asserted stance against the store's, so the resilience
probe pairs with the verifiers rather than standing alone.

See also
--------

- :doc:`/3-api-reference/modules/tools/index` — ``GovernedToolExecutor`` and
  ``ToolHooks``.
- :doc:`/3-api-reference/modules/agents_testing/index` — ``fault_injection`` and
  ``FaultInjectionResult``.
- :doc:`ch12_capstone` — the resilience stage of the end-to-end campaign.
- :doc:`/4-notebook-examples/testing/index` — the detection probe worked through.
