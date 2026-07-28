Chapter 5 — The Base Taxonomy
=============================

This page shows how to load the base catalog, read what one category requires,
and turn categories into scorable questions through the three base sources. The
base catalog names eighteen question categories in five families; each category
records the substrate primitive that establishes its ground truth, so a
generated question is scorable by construction rather than by annotation.

Loading the catalog
--------------------

:meth:`Catalog.load <forgeloop.testing.catalogs.Catalog.load>` reads the base,
factor and profile YAML shipped as package data. Each base is a
:class:`~forgeloop.testing.catalogs.BaseSpec` recording the graph capability it
requires, the primitive that computes its ground truth, its answer type and a
build status.

.. code-block:: python

   import forgeloop.testing as gt

   cat = gt.Catalog.load()
   print(cat.summary())
   # 18 base categories in 5 families, 40 factors in 10 groups, ... profiles

   for fam, names in cat.families.items():
       print(f'{fam:24s} {len(names):2d}  {", ".join(names)}')

   b = cat.bases['exact_recall']
   for f in ('family', 'requires', 'generator', 'answer_type', 'ground_truth', 'status'):
       print(f'{f:14s}', getattr(b, f))
   # ground_truth  lookup_enm

A category is admissible for a corpus only when the store built from that corpus
exposes the capability the base records in ``requires``, which is what makes the
taxonomy portable across documents.

The three base sources
-----------------------

A base category becomes concrete questions through a
:class:`~forgeloop.testing.sources.BaseSource`. The three implementations differ
in where the ground truth comes from.

.. list-table::
   :header-rows: 1

   * - Source
     - Ground truth
   * - :class:`~forgeloop.testing.sources.UserBaseSource`
     - Supplied ``(query, answer)`` pairs; no GMS is used.
   * - :class:`~forgeloop.testing.sources.SeedCaseSource`
     - Hand-labeled cases carrying per-component truth.
   * - :class:`~forgeloop.testing.sources.CatalogBaseSource`
     - Mined from a GMS store; truth is graph-derived.

All three yield :class:`~forgeloop.testing.sources.QAItem` values, so the
enrichment layer treats them identically.

.. code-block:: python

   user = gt.UserBaseSource([
       {'query': 'What is the overdraft fee?', 'answer': '35'},
       {'query': 'How long to dispute a charge?', 'answer': '60 days'},
   ]).items()
   print(user[0].base)          # None — user-supplied carries no catalog base

   cases = [{'id': 'case-001',
             'message': 'I was charged a $35 overdraft fee and want it removed.',
             'expected_classification': 'complaint', 'expected_escalation': False,
             'expected_issue': 'overdraft_fee'}]
   seed = gt.SeedCaseSource(cases).items()[0]
   print(seed.components)       # per-tool ground truth carried on the item

Mining a base from the graph
-----------------------------

:class:`~forgeloop.testing.sources.CatalogBaseSource` runs the selected
categories' knowlytix generators against a loaded store. The ground truth for
each question is read from the graph, so no labeling is required. Selection names
the categories; the graph vetoes any category whose structure it cannot fill.

.. code-block:: python

   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
   import torch

   store = GMSExpertStore(
       DocGMSConfig(store_path=str(STORE), ingest_mode="regex"),
       device=torch.device("cpu"))
   assert store.load()

   suite = gt.resolve(cat, ['exact_recall', 'counting', 'contradiction'], [], mode='cross')
   items = gt.CatalogBaseSource(store, max_per_category=3).items(suite)
   for it in items[:3]:
       print(it.base, '|', it.query[:56], '-> ', it.answer)

A category that passes its ``requires`` check can still mine nothing when the
structure is absent: a clean store holds no contradiction, so ``contradiction``
resolves but yields no items. The ground truth a mined item carries re-derives
from the graph exactly, which is the property the functionality check in
:doc:`ch08_training` verifies.

See also
--------

- :doc:`/3-api-reference/modules/testing/index` — the catalog and the three base
  sources.
- :doc:`ch06_enrichment` — the factors that enrich a base.
- :doc:`/4-notebook-examples/testing/index` — the base catalog and a graph-mined
  base worked through.
