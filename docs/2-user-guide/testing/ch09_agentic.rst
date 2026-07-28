Chapter 9 — Agentic Systems and What to Test in Them
====================================================

This page shows how to run composed scenarios through an agent and score the run
at three levels. When the system under test is an agent rather than a single
model call, the endpoint is not enough: an agent can reach the right final
decision by the wrong route, escalate for a spurious reason or proceed on a
corrupted tool result. The unit of observation is the trajectory, and
:func:`~forgeloop.testing.evaluate.run` scores it.

The system-under-test interface
--------------------------------

The framework treats the agent as an interface. The minimal contract is a
callable mapping a query and a context to an answer, which is enough to score the
outcome. Returning a :class:`~forgeloop.testing.evaluate.SUTResult` instead of a
bare value unlocks the remaining levels, because the result carries the
trajectory and the per-component predictions.

.. list-table::
   :header-rows: 1

   * - ``SUTResult`` field
     - What it records
   * - ``answer``
     - The final answer.
   * - ``components``
     - Per-tool predicted outputs, keyed by tool name.
   * - ``trajectory``
     - Ordered workflow steps, each a dict with ``tool`` and ``ok``.
   * - ``status``
     - ``"ok"``, ``"escalated"`` or ``"failed"``.
   * - ``escalation_trigger``
     - The trigger that caused escalation, or ``None``.

.. code-block:: python

   from forgeloop.testing.evaluate import SUTResult

   workflow = ['classify', 'extract', 'search', 'flag', 'draft']

   def sut(query, context=''):
       bad = '[clarity=Misleading]' in query          # this SUT slips on misleading framing
       return SUTResult(
           answer='OK' if not bad else 'WRONG',
           components={'classification': 'complaint' if not bad else 'other'},
           trajectory=[{'tool': t, 'ok': True} for t in workflow],
           status='ok' if not bad else 'failed',
       )

Running the suite and scoring three levels
-------------------------------------------

:func:`~forgeloop.testing.evaluate.run` executes every scenario and returns one
row per scenario. Passing ``workflow_order`` adds the trajectory-adherence check;
:func:`~forgeloop.testing.evaluate.summary` aggregates the rows.

.. code-block:: python

   import forgeloop.testing as gt
   from forgeloop.testing.evaluate import run, summary

   cat = gt.Catalog.load()
   items = [gt.QAItem(qid=f's{i}', query='[clarity=Clear] msg', answer='OK',
                      components={'expected_classification': 'complaint',
                                  'expected_escalation': False}) for i in range(3)]
   scns = gt.compose(gt.resolve(cat, ['exact_recall'], ['clarity'], mode='cross'),
                     items, n_runs=2, seed=1)

   rows = run(scns, sut, workflow_order=workflow)
   print(summary(rows))     # {'n': ..., 'accuracy': ..., 'workflow_adherence': 1.0}

Each row scores three things. The outcome check compares the answer or the
labeled decisions to ground truth. The trajectory check reads the tool sequence
against ``workflow_order`` and records ``workflow_adherent``. The process signals
are the step count, the tool-call count and the tool-failure count read from the
trajectory. A row also carries per-component correctness for each key in the
scenario's ``components``, so a failure can later be charged to the component
that produced it.

Attributing a failure to a component
-------------------------------------

:func:`~forgeloop.testing.evaluate.weak_link` walks the workflow order and
charges each end-to-end failure to the first component that erred. This is the
component-level counterpart to the input-level attribution of :doc:`ch10_analysis`:
one asks which tool is responsible, the other asks which input condition drives
failure.

See also
--------

- :doc:`/3-api-reference/modules/testing/index` — ``run``, ``summary``,
  ``weak_link`` and ``SUTResult``.
- :doc:`/3-api-reference/modules/agents_testing/index` — the harnesses that drive
  a real governed agent as a SUT.
- :doc:`ch10_analysis` — logistic attribution over the scored rows.
- :doc:`/4-notebook-examples/testing/index` — the SUT interface and scoring
  worked through.
