Chapter 6 — The Enrichment Design Space
=======================================

This page shows how to read the enrichment catalog, how a factor's ``applies_to``
specification restricts it to the bases it is meaningful for, and how factor
groups bundle a selection. The catalog holds forty factors in ten groups. Each
factor is a categorical variable whose levels are presentation conditions, and
each is ground-truth invariant: it changes how a question is presented, never
what its answer is.

Reading the factors
-------------------

Every factor is a :class:`~forgeloop.testing.catalogs.FactorSpec` with a level
vocabulary and a ``gt_invariant`` flag. The invariance holds across the whole
catalog, which is the precondition for expanding a base into a factor-balanced
suite that stays scorable.

.. code-block:: python

   from collections import Counter
   import forgeloop.testing as gt

   cat = gt.Catalog.load()
   print('factors        :', len(cat.factors))                         # 40
   print('all invariant  :', all(f.gt_invariant for f in cat.factors.values()))
   print('groups         :', Counter(f.section for f in cat.factors.values()))

Relevance: enrichment enriches the base
---------------------------------------

A factor carries an ``applies_to`` specification naming the base families and
answer types it acts on. When a selection is resolved,
:func:`~forgeloop.testing.resolve.resolve` keeps only the factors that apply to
at least one selected base and records the rest in ``dropped_factors``. A factor
that renders a number is meaningful for a numeric answer and dropped for a
set-valued one, because there is no number for it to act on.

.. code-block:: python

   # cross_reference answers a set -> numeric_format does not apply
   s1 = gt.resolve(cat, ['cross_reference'], ['numeric_format', 'clarity'], mode='embedded')
   print('kept   :', s1.factor_names)         # ['clarity']
   print('dropped:', s1.dropped_factors)      # [('numeric_format', '... no selected base')]

   # exact_recall answers a float -> numeric_format applies
   s2 = gt.resolve(cat, ['exact_recall'], ['numeric_format', 'clarity'], mode='embedded')
   print('kept   :', s2.factor_names)         # ['numeric_format', 'clarity']

The filtering is what makes the enrichment of a base specific to that base rather
than uniform across the catalog. It runs automatically inside ``resolve``;
:func:`~forgeloop.testing.resolve.applicable_factors` performs the same check
directly on a factor and base list.

Factor groups
-------------

A factor group is a named bundle of factor names, so a common selection is a
single token. The catalog ships groups for a quick screen, an adversarial pass
and a comprehensive sweep.

.. code-block:: python

   for g in ('quick_screen', 'adversarial', 'comprehensive'):
       print(f'{g:14s}', cat.factor_groups[g])

A group name is resolved the same way as a factor name, so
``gt.resolve(cat, bases, ['quick_screen'])`` expands the group and then filters
it against the selected bases.

Overriding a factor's levels
-----------------------------

The catalog stays domain-neutral: selecting a factor is data, rendering its
levels is code. An application supplies its own level vocabulary through
``level_overrides`` on :func:`~forgeloop.testing.compose.compose`, without
touching the factor definition. For example, an ``entity_aliasing`` factor can be
overridden to ``["canonical", "alias"]`` for a specific corpus while the catalog
entry is unchanged.

See also
--------

- :doc:`/3-api-reference/modules/testing/index` — ``FactorSpec``, ``resolve``
  and ``applicable_factors``.
- :doc:`ch05_base_taxonomy` — the bases a factor is filtered against.
- :doc:`ch07_design` — composing a resolved suite into scenarios.
- :doc:`/4-notebook-examples/testing/index` — the factor catalog and
  ``applies_to`` filtering worked through.
