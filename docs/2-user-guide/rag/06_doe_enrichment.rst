Generating Data with a Designed Experiment
==========================================

The trained store serves as an oracle: every fact it holds is a known,
graph-derived answer, so the store supplies the data that trains and tests the
rest of the pipeline. This page mines the store for questions whose answers are
fixed by the graph, enriches each across a designed space of presentation
conditions and emits one corpus that feeds both the encoder tuning
(:doc:`07_embedding_sft`) and the evaluation (:doc:`15_evaluation`). The suite
API lives in ``knowlytix.harness.suite``.

The organizing principle is ground-truth-invariant enrichment. A question about
Cloud Platform's revenue has one answer fixed by the graph; whether it is asked
tersely or verbosely, plainly or hesitantly, the answer does not move. What is
asked, a base taxonomy the store can answer, is separated from how it is
presented, a layer of factors that vary the surface without touching the truth.

Load the store as an oracle
---------------------------

The base source reads answers from the store built in
:doc:`04_document_to_graph`. Reload it with
:class:`~knowlytix.knowledge.store.GMSExpertStore`.

.. code-block:: python

   import os, torch
   from knowlytix.knowledge.config import DocGMSConfig
   from knowlytix.knowledge.store import GMSExpertStore

   STORE = os.path.join("data", "gms_annual_report_store")
   dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   store = GMSExpertStore(DocGMSConfig(store_path=STORE), device=dev)
   assert store.load(), "build the store first (Ch4)"

Mine base questions
-------------------

:class:`~knowlytix.harness.suite.catalogs.Catalog` loads the taxonomy through
``Catalog.load``, and :func:`~knowlytix.harness.suite.resolve.resolve` selects the base
types and the presentation factors into a suite. The two content-retrieval types
the report supports are ``exact_recall`` and ``multi_hop``; the five presentation
factors are ``clarity``, ``style``, ``length``, ``expertise`` and
``paraphrase_depth``. :class:`~knowlytix.harness.suite.sources.CatalogBaseSource`
mines the store with the benchmark generators, and each mined item carries its
graph-derived ground truth, so no hand labeling is involved.

.. code-block:: python

   from knowlytix.harness.suite import Catalog, resolve, CatalogBaseSource

   CAT = Catalog.load()
   suite = resolve(CAT, ["exact_recall", "multi_hop"],
                   ["clarity", "style", "length", "expertise", "paraphrase_depth"],
                   mode="embedded")
   items = CatalogBaseSource(store, max_per_category=8, seed=42).items(suite)
   print(len(items), "base questions, each graph-derived")

Design the experiment
---------------------

:func:`~knowlytix.harness.suite.compose.compose` turns the base items and the
factor definitions into scenarios under a design. In ``embedded`` mode the base
question is a factor, so the number of scenarios equals ``n_runs``; the
presentation factors are space-filled by a Sobol sequence through
:func:`~knowlytix.harness.suite.compose.graphdoe_design`, and the base question
is a balanced blocking factor when ``balance_base=True``. The design covers the
presentation space with far fewer runs than an exhaustive grid.

.. code-block:: python

   from functools import partial
   from knowlytix.harness.suite import compose, graphdoe_design

   scns = compose(suite, items, n_runs=150, seed=42,
                  design_fn=partial(graphdoe_design, method="sobol+refine"),
                  balance_base=True)
   print(len(scns), "scenarios over", suite.factor_names)

Materialize and emit
--------------------

Each scenario is rendered into a natural-language question by a batched, guarded
local rewrite that must stay on topic and must not invent a number, year or
company. The rewrite emits the cohort, the v-space SFT rows, the u-space groups
and the draft pairs; its body is in ``scripts/enrich_data.py``. The cohort is
persisted as JSON and read back for inspection.

.. code-block:: python

   import json
   cohort = json.load(open(os.path.join("data", "enrichment", "rag_cohort.json")))
   print(len(cohort), "cohort cases; sample:", cohort[0]["question"][:70])

A self-check confirms the two invariants the design is built to hold: every case
carries a graph-derived answer, and the factor levels are balanced.

.. code-block:: python

   import collections
   assert all(c["expected_answer"] is not None for c in cohort)
   clar = collections.Counter(c["_factors"]["clarity"] for c in cohort)
   assert min(clar.values()) >= 30
   print("OK: designed, ground-truthed, balanced")

The corpus is only as broad as the generators that mine it: the cohort spans the
phrasings the design covers, not phrasings far outside the factor space, a limit
:doc:`07_embedding_sft` and :doc:`15_evaluation` carry forward.

See also
--------

- :doc:`04_document_to_graph` — the store this page mines as an oracle.
- :doc:`07_embedding_sft` and :doc:`15_evaluation` — the encoder tuning and
  evaluation that consume the emitted corpus.
- :doc:`/3-api-reference/modules/knowlytix/harness/index` — the suite API
  (``Catalog``, ``resolve``, ``CatalogBaseSource``, ``compose``,
  ``graphdoe_design``).
- :doc:`/4-notebook-examples/rag/index` — the enrichment run end to end.
