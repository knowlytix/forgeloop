Chapter 10 — Identifying Weakness: Logistic Attribution of Failure
==================================================================

This page shows how to turn a table of binary outcomes and factor levels into a
ranked statement about which conditions cause failure. A run over the base and
enrichment catalogs produces one row per scenario with a zero-one ``correct``
column. :func:`~forgeloop.testing.evaluate.attribute` fits a logistic model to
that column and ranks the factors by the deviance they explain.

Fitting the attribution
------------------------

The rows carry the outcome and the factor levels under the ``f_`` prefix that
:func:`~forgeloop.testing.evaluate.run` emits.
:func:`~forgeloop.testing.evaluate.attribute` takes the rows and the factor
names, fits logistic regression through the knowlytix ``DOEAnalyzer`` and returns
one record per factor.

.. code-block:: python

   import random
   import forgeloop.testing as gt
   from forgeloop.testing.evaluate import attribute

   cat = gt.Catalog.load()
   suite = gt.resolve(cat, ['multi_hop'],
                      ['clarity', 'entity_aliasing', 'reasoning_cue'], mode='embedded')
   scns = gt.compose(suite,
                     [gt.QAItem(qid=f's{i}', query='m', answer='ok') for i in range(6)],
                     n_runs=120, seed=7)

   rng = random.Random(7)
   rows = [{'correct': 0 if rng.random() < (0.6 if s.factor_levels['clarity'] == 'Misleading'
                                            else 0.15) else 1,
            **{f'f_{k}': v for k, v in s.factor_levels.items()}} for s in scns]

   for t in attribute(rows, suite.factor_names):
       print(f"{t['factor']:16s} p_adj={t['p_value_adj']:.4f} "
             f"R2={t['pseudo_r2']:.3f} sig={t['significant_adj']}")

The synthetic outcome above fails more often under misleading clarity, so
``clarity`` comes back significant after correction while the other factors do
not. On a real run the rows come from :func:`~forgeloop.testing.evaluate.run`
over a system under test.

What the record reports
-----------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``factor``
     - The factor the row summarizes.
   * - ``pseudo_r2``
     - McFadden's pseudo-:math:`R^2`: the share of uncertainty the factor explains.
   * - ``p_value_adj``
     - Analysis-of-deviance p-value after Benjamini-Hochberg correction.
   * - ``significant_adj``
     - Whether the factor is significant at the corrected level.

The response is a Bernoulli outcome, so the model is logistic rather than
analysis of variance, which would degenerate into a linear probability model on a
zero-one response. Significance is the likelihood-ratio test comparing a null
model against one adding the factor's indicators, referred to a chi-squared
distribution, and the family of per-factor tests is corrected for multiple
comparisons. The ranked drivers are also reachable through
:meth:`~forgeloop.agents.testing.harness.FactorAttribution.top_drivers` when the
attribution comes from a harness.

Charging failure to a component
-------------------------------

Attribution answers which input condition drives failure.
:func:`~forgeloop.testing.evaluate.weak_link` answers the complementary question:
walking the workflow order, it charges each end-to-end failure to the first
component that erred, which assigns failures to system components rather than to
inputs.

.. code-block:: python

   from forgeloop.testing.evaluate import weak_link

   blame = weak_link(rows_with_components, workflow_order=[
       'classify_complaint', 'extract_facts', 'search_policy',
       'flag_regulatory', 'draft_response'])

See also
--------

- :doc:`/3-api-reference/modules/testing/index` — ``attribute``, ``summary`` and
  ``weak_link``.
- :doc:`/3-api-reference/modules/agents_testing/index` —
  ``FactorAttribution.top_drivers`` and the harness-level tables.
- :doc:`ch09_agentic` — producing the scored rows.
- :doc:`/4-notebook-examples/testing/index` — the attribution run end to end.
