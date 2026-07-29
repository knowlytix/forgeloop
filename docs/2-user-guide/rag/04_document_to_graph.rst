Chapter 4 — From Document to Graph
==================================

This page turns a markdown document into a trained, queryable
:class:`~knowlytix.knowledge.store.GMSExpertStore`. One call ingests the
document, runs GEODE self-correction, trains the geometric model, populates
Exact Numerical Memory and saves the store to disk. Ingestion runs in regex
mode, a deterministic parser with no language model in the loop, so no
model-extracted number can enter the graph, and every authoritative figure is
parsed once into ENM rather than left as a string a model re-reads at query
time. The worked corpus is the Northwind Industries annual report shipped under
``data/annual_report.md``.

Configure the build
-------------------

:class:`~knowlytix.knowledge.config.DocGMSConfig` carries the store path, the
ingest mode and the geometry and training hyperparameters. ``ingest_mode="regex"``
selects the deterministic parser; ``loss_mode="cap"`` selects the cap geometry.
The geometry and training blocks are :class:`~knowlytix.core.config.GeometryConfig`
and :class:`~knowlytix.core.config.TrainConfig`.

.. code-block:: python

   import os, torch
   from knowlytix.core.config import GeometryConfig, TrainConfig
   from knowlytix.knowledge.config import DocGMSConfig
   from knowlytix.knowledge.geode.rag import build_rag_store
   from knowlytix.knowledge.geode.loop import make_default_trainer

   MD = os.path.join("data", "annual_report.md")
   STORE = os.path.join("data", "gms_annual_report_store")

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   config = DocGMSConfig(
       store_path=STORE,
       ingest_mode="regex",   # deterministic: no model-extracted numbers
       loss_mode="cap",
       geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32),
       train=TrainConfig(epochs=150, batch_size=64, neg_samples=16,
                         lr=5e-3, lr_riemannian=2e-3),
   )

Build the store
---------------

:func:`~knowlytix.knowledge.geode.rag.build_rag_store` composes ingestion, the
GEODE correction loop, training and persistence. The GEODE stage takes its own
trainer, constructed by :func:`~knowlytix.knowledge.geode.loop.make_default_trainer`.
The call returns a :class:`~knowlytix.knowledge.geode.rag.RagBuildResult` whose
fields report the graph size and the audit.

.. code-block:: python

   res = build_rag_store(
       MD, config, device=device,
       geode_trainer=make_default_trainer(device, epochs=80),
   )
   store = res.store

   print(f"converged={res.converged}  iterations={res.iterations}")
   print(f"entities={res.n_entities}  triples={res.n_triples}  enm={res.n_enm}")
   print(f"relations={len(store.adapter.relation_to_idx)}")
   print(f"GEODE corrections:      {len(res.corrections)}")
   print(f"GEODE anchor violations:{len(res.anchor_violations)}")

On the shipped corpus the build reports ``converged=True``, ``iterations=1``,
21 ENM entries and 10 relations (``has_amount``, ``has_division``,
``has_fy2024``, ``has_fy2025``, ``has_head``, ``has_headcount``, ``has_region``,
``has_revenue``, ``has_value``, ``in_section``). The corpus carries no
contradiction and no broken sum, so GEODE reports zero corrections and zero
anchor violations; :doc:`05_geode_self_correction` corrupts a figure to exercise
the loop.

Reload and query
----------------

The store is a set of files under ``store_path``. A fresh
:class:`~knowlytix.knowledge.store.GMSExpertStore` reloads it with ``load()`` and
no rebuild, no GEODE and no training. ENM is the authoritative numeric channel:
``lookup_enm`` returns the byte-exact value, and ``query_triples`` pattern-matches
an asserted edge.

.. code-block:: python

   from knowlytix.knowledge.store import GMSExpertStore

   reloaded = GMSExpertStore(config, device)
   assert reloaded.load(), f"no store at {STORE}"

   cloud_rev = reloaded.lookup_enm("segment_performance",
                                   "Cloud Platform/Technology/Revenue")
   total_rev = reloaded.lookup_enm("income_statement", "Revenue/FY2025")
   print(f"Cloud Platform revenue = {cloud_rev}")   # 120.0
   print(f"Total FY2025 revenue   = {total_rev}")   # 355.0

   div = reloaded.query_triples(head="cloud platform", relation="has_division")
   print("cloud platform division:", div)
   # [('cloud platform', 'has_division', 'technology')]

The reloaded store answers identically to the freshly built one: the figures
match the report's Segment Performance and Income Statement tables, and the
``cloud platform -> technology`` edge is present for the multi-hop chains built
in later chapters.

:mod:`forgeloop.rag` wraps this reload path for the shipped store.
:func:`forgeloop.rag.store_config` returns the exact
:class:`~knowlytix.knowledge.config.DocGMSConfig` the store was built with, and
:func:`forgeloop.rag.load_store` returns the trained store ready to query.

.. code-block:: python

   from forgeloop.rag import load_store, store_config

   store = load_store()             # trained store, no rebuild
   config = store_config()          # the DocGMSConfig it was built with

Why the numbers stay exact
--------------------------

Two decisions keep the store's figures byte-stable. Regex ingestion admits no
language model at ingestion, so a number in the graph came from a parsed table
cell, not a model's re-reading of one. Every authoritative figure is parsed once
into ENM and read back through ``lookup_enm``, so a query returns the stored
value rather than a string re-parsed at answer time. The consequence is a graph,
not a guarantee of completeness: regex ingestion extracts from tables and the
declared schema, not free-form prose, so the report's narrative sections produce
no triples, and a question answerable only from prose abstains rather than
answers. Coverage and abstention are the subject of :doc:`13_abstention_and_coverage`.

See also
--------

- :doc:`03_provenance` — the span-to-triple map over the same report.
- :doc:`05_geode_self_correction` — the correction loop ``build_rag_store`` runs.
- :doc:`/3-api-reference/modules/knowlytix/geode/index` and
  :doc:`/3-api-reference/modules/knowlytix/store/index` — signatures for these calls.
- :doc:`/4-notebook-examples/rag/index` — the same build run end to end.
