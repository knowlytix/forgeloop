Why Average Accuracy Is Not Enough
==================================

This page shows how a single aggregate score hides a concentrated failure, and
how :mod:`forgeloop.testing` reframes evaluation as a designed experiment that
estimates the effect of input conditions on the probability of failure. The
reframing rests on one separation the package encodes as two data types: a
:class:`~forgeloop.testing.catalogs.BaseSpec` fixes what is asked and what is
true, and a :class:`~forgeloop.testing.catalogs.FactorSpec` fixes how the
question is presented without disturbing its answer.

The aggregate conceals the weak condition
------------------------------------------

Take a hundred graded items under two presentation conditions. The system is
right on most clear items and wrong on most misleading ones. The overall rate
looks healthy while one condition is failing.

.. code-block:: python

   items = [('clear', 1)] * 78 + [('clear', 0)] * 2 \
         + [('misleading', 1)] * 8 + [('misleading', 0)] * 12
   overall = sum(o for _, o in items) / len(items)
   by_cond = {c: sum(o for cc, o in items if cc == c) / sum(cc == c for cc, _ in items)
              for c in ('clear', 'misleading')}
   print(f'overall accuracy : {overall:.2f}')       # 0.86
   print('by condition     :', {k: round(v, 2) for k, v in by_cond.items()})
   # {'clear': 0.97, 'misleading': 0.40}

The 86% aggregate averages a 0.97 condition with a 0.40 condition. A benchmark
reports the 0.86. A designed experiment reports the gap and attributes it to the
condition that produced it.

The two requirements a benchmark does not supply
------------------------------------------------

Estimating the effect of a condition needs a ground truth that is correct by
construction rather than by annotation, so a failure charged to a condition is
real and not an artifact of an ambiguous key, and a factor space whose
conditions vary independently of the items. The package supplies both through a
decomposition of every test item into a base and a set of enrichment factors.

.. list-table::
   :header-rows: 1

   * - Type
     - Role
   * - :class:`~forgeloop.testing.catalogs.BaseSpec`
     - One base question type: what is asked and the substrate primitive that computes its ground truth.
   * - :class:`~forgeloop.testing.catalogs.FactorSpec`
     - One presentation dimension whose levels leave the ground truth unchanged (``gt_invariant``).
   * - :class:`~forgeloop.testing.catalogs.Catalog`
     - The loaded base, factor and profile catalogs, keyed by name.

A base carries a question and the value a substrate primitive computes for it. A
factor carries a single presentation dimension and one requirement: varying it
leaves the answer alone. That requirement is recorded on the factor as
``gt_invariant`` and is why a handful of bases expands into a large,
factor-balanced suite in which every item remains automatically scorable.

.. code-block:: python

   import forgeloop.testing as gt

   cat = gt.Catalog.load()
   print(cat.summary())
   # 18 base categories in 5 families, 40 factors in 10 groups, ... profiles

   base = cat.bases['exact_recall']
   print(base.answer_type, '<-', base.ground_truth)   # str <- lookup_enm

   factor = cat.factors['clarity']
   print(factor.gt_invariant)                         # True

From the decomposition to attribution
--------------------------------------

The base and factor catalogs feed the pipeline the rest of this part builds.
:func:`~forgeloop.testing.resolve.resolve` expands a base and factor selection
into a suite, :func:`~forgeloop.testing.compose.compose` crosses the bases with a
space-filling design, and :func:`~forgeloop.testing.evaluate.attribute` fits a
logistic model that ranks which factor levels drive failure. The toy table above
is the first row of that pipeline: a binary outcome under a controlled condition,
which is what logistic attribution consumes.

See also
--------

- :doc:`/3-api-reference/modules/testing/index` — the catalog, resolution,
  composition and evaluation modules.
- :doc:`ch05_base_taxonomy` and :doc:`ch06_enrichment` — the base and factor
  catalogs in depth.
- :doc:`/4-notebook-examples/testing/index` — the same example run end to end.
