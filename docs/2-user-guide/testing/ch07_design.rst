Experimental Design and Composition
===================================

This page shows how to turn a resolved suite into concrete test scenarios: how a
profile bundles a selection, how the two composition modes arrange a base against
a design, and which design generator to use. A fixed budget must cover a factor
space that is high-dimensional and of mixed cardinality, so the design is
space-filling rather than full-factorial, and the base enters it in one of two
arrangements.

Profiles
--------

A :class:`~forgeloop.testing.catalogs.Profile` is a named bundle of a base
selection, a factor selection and a mode.
:func:`~forgeloop.testing.resolve.resolve_profile` resolves it into a
:class:`~forgeloop.testing.resolve.ResolvedSuite`, applying the ``applies_to``
filtering of :doc:`ch06_enrichment`.

.. code-block:: python

   import forgeloop.testing as gt

   cat = gt.Catalog.load()
   for name, p in cat.profiles.items():
       print(f'{name:20s} mode={p.mode:9s} bases={p.bases}')

   suite = gt.resolve_profile(cat, 'numeric_screen')
   print(suite.summary())

The two composition modes
-------------------------

:func:`~forgeloop.testing.compose.compose` builds
:class:`~forgeloop.testing.compose.Scenario` rows from a suite and a list of base
items. The mode carried on the suite selects the arrangement.

.. list-table::
   :header-rows: 1

   * - Mode
     - Arrangement
     - Suite size
   * - ``cross``
     - Each base item is paired with every design row.
     - ``n_base_items * n_runs``
   * - ``embedded``
     - The base is promoted to a factor in the design.
     - ``n_runs``

In the crossed arrangement the design covers the presentation factors alone, and
every base item sees every design row, giving exhaustive per-item coverage at a
cost that grows with the number of items. In the embedded arrangement the base is
varied jointly with the presentation factors, giving a fixed-size suite whose
rows are space-filled over the joint space, and because the base is a factor its
own items can be attributed later.

.. code-block:: python

   from collections import Counter
   from forgeloop.testing.compose import graphdoe_design   # knowlytix Sobol+refine

   items = [gt.QAItem(qid=f'q{i}', query=f'metric {i}?', answer=str(i)) for i in range(5)]

   cross = gt.compose(gt.resolve(cat, ['exact_recall'], ['clarity'], mode='cross'),
                      items, n_runs=8, seed=1, design_fn=graphdoe_design)
   emb   = gt.compose(gt.resolve(cat, ['exact_recall'], ['clarity'], mode='embedded'),
                      items, n_runs=8, seed=1, design_fn=graphdoe_design)

   print('cross   ', len(cross))                            # 5 * 8 = 40
   print('embedded', len(emb),                              # 8
         'base balance', dict(Counter(s.base.qid for s in emb)))

Choosing a design generator
---------------------------

The design generator is pluggable. :func:`~forgeloop.testing.compose.simple_design`
is dependency-light and deterministic, which suits reproducible small runs and
offline tests. :func:`~forgeloop.testing.compose.graphdoe_design` calls the
knowlytix Sobol-plus-refine generator, which minimizes the :math:`\phi_p`
space-filling criterion and supplies the coverage a full campaign needs.

In embedded mode the presentation factors are space-filled by the generator while
the base question is assigned as a balanced blocking factor by default
(``balance_base=True``), so every base is exercised equally even at small
``n_runs``. Setting ``balance_base=False`` places the base into the design matrix
as a joint factor, which is preferable only at large ``n_runs``.

See also
--------

- :doc:`/3-api-reference/modules/testing/index` — ``resolve_profile``,
  ``compose`` and the design functions.
- :doc:`ch08_training` and :doc:`ch09_agentic` — the two consumers of the
  composed scenarios.
- :doc:`/4-notebook-examples/testing/index` — profiles and both modes worked
  through.
