Chapter 12 — Testing a Governed Agent End to End
================================================

This page shows how to drive the whole apparatus against one system: the governed
banking complaint agent of the companion volume *Beyond Prompt and Pray*. The
agent is a fixed five-step workflow under a stack of gates, with an escalation
path and a hash-chained audit log. It is not modified. The framework wraps it as
an external system under test with
:class:`~forgeloop.apps.complaint_sut.AgentSUT` and runs it through the stages the
earlier pages built.

Wrapping the agent as a SUT
---------------------------

:class:`~forgeloop.apps.complaint_sut.AgentSUT` runs one customer message through
the real governed harness and maps the resulting trajectory to a
:class:`~forgeloop.testing.evaluate.SUTResult`: the draft is the answer, the
status reflects escalation, the components carry the classification and the
trajectory carries the tool order. That is the exact interface
:func:`~forgeloop.testing.evaluate.run` consumes.

.. code-block:: python

   from forgeloop.apps.complaint_sut import AgentSUT
   import forgeloop.testing as gt
   from forgeloop.testing.evaluate import run, summary, attribute, weak_link

   sut = AgentSUT()

   cat = gt.Catalog.load()
   items = gt.SeedCaseSource(cases).items()          # 20 labeled complaint cases
   suite = gt.resolve(cat, ['exact_recall'],
                      ['clarity', 'entity_aliasing', 'reasoning_cue'], mode='embedded')
   scns = gt.compose(suite, items, n_runs=120, seed=42)

   rows = run(scns, sut, workflow_order=[
       'classify_complaint', 'extract_facts', 'search_policy',
       'flag_regulatory', 'draft_response'])
   print(summary(rows))                              # accuracy, workflow_adherence

The stages and the API that performs each
------------------------------------------

.. list-table::
   :header-rows: 1

   * - Stage
     - API
   * - Test enrichment
     - :func:`~forgeloop.testing.resolve.resolve`, :func:`~forgeloop.testing.compose.compose`
   * - Ground-truth precondition
     - :meth:`CapstoneTestHarness.substrate_test <forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.substrate_test>`
   * - Gate stack
     - :func:`~forgeloop.apps.complaint_sut.probe_policy_gate`, :func:`~forgeloop.apps.complaint_sut.probe_plausibility_gate`
   * - Policy search
     - :func:`~forgeloop.apps.complaint_sut.retrieval_benchmark`
   * - System attribution
     - :func:`~forgeloop.testing.evaluate.attribute`, :func:`~forgeloop.testing.evaluate.weak_link`
   * - Resilience
     - :meth:`CapstoneTestHarness.fault_injection <forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.fault_injection>`, :func:`~forgeloop.apps.complaint_sut.probe_value_polarity`

Ground truth before scores
---------------------------

A score is trusted only when the answer key is sound. The precondition is that
the GEODE store answers its own generated questions at a baseline of exactly
``1.0``; the substrate stage checks this before any agent number is read. The
policy-search stage grades ``search_policy`` against the graph ground truth
through parse, bind and retrieve rather than a top-k label, so a retrieval miss
is charged to the correct stage.

.. code-block:: python

   from forgeloop.apps.complaint_sut import probe_policy_gate, probe_plausibility_gate

   pol = probe_policy_gate(cases)      # refuse-path and admit-path of the Policy gate
   pla = probe_plausibility_gate()     # the plausibility gate's deny path over transitions

Attribution and resilience
---------------------------

The system stage regresses the per-query decision on the presentation factors
with :func:`~forgeloop.testing.evaluate.attribute` and charges each failure to a
component with :func:`~forgeloop.testing.evaluate.weak_link`, excluding the seed
case, which was a blocking factor rather than a property under test. The
operational stage reads workflow adherence and confirms the audit log verifies.
The resilience stage injects a tool fault and confirms the agent escalates rather
than continuing, then grades the value-polarity verifier on stance reversals.

On the pinned campaign the agent classifies at ``0.90``, holds workflow adherence
at ``1.0`` with a verifying audit log, and detects every injected tool fault
(detection rate ``1.0``, zero silent failures). The per-query decision accuracy,
which requires both the classification and the escalation to be right, is
``0.675``. Every figure is produced by a reproducible run rather than an
illustration.

See also
--------

- :doc:`/3-api-reference/modules/apps/index` — ``AgentSUT`` and the component
  probes.
- :doc:`/3-api-reference/modules/agents_testing/index` — ``CapstoneTestHarness``
  and its result types.
- :doc:`ch09_agentic`, :doc:`ch10_analysis` and :doc:`ch11_resilience` — the
  stages in isolation.
- :doc:`/4-notebook-examples/testing/index` — the capstone campaign worked
  through.
