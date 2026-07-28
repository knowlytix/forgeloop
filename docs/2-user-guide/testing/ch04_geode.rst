Chapter 4 — Building the Oracle
===============================

This page shows how GEODE turns a raw document into the trained, calibrated store
the primitives read. The primitives in :doc:`ch03_gms_primitives` assume a store
exists; ``build_rag_store`` is the call that produces one, running ingest,
self-correction and training in a single closed loop so the oracle does not
inherit the errors of its own extraction.

Running GEODE on a document
---------------------------

``build_rag_store`` takes a document path, a ``DocGMSConfig`` and a trainer, and
returns a result carrying the built store and the loop's diagnostics. The
``ingest_mode="regex"`` backbone parses tables and prose with rules, which
recovers exact figures byte-for-byte; a hybrid mode adds a language model at a
few decision points to recover facts the rules miss while preserving the numbers.

.. code-block:: python

   import tempfile, torch
   from pathlib import Path
   from knowlytix.knowledge.geode import build_rag_store, make_default_trainer
   from knowlytix.knowledge.config import DocGMSConfig

   doc = Path(tempfile.mktemp(suffix='.md'))
   doc.write_text(
       "# Fee Schedule\n\n"
       "| product | fee_amount | type |\n| --- | --- | --- |\n"
       "| overdraft | 35.00 | per_occurrence |\n"
       "| wire_international | 45.00 | per_transaction |\n"
   )

   cfg = DocGMSConfig(store_path=tempfile.mkdtemp(), ingest_mode='regex')
   cfg.train.epochs = 60                         # small, for a fast build
   res = build_rag_store(
       doc, cfg, device=torch.device('cpu'),
       geode_trainer=make_default_trainer(torch.device('cpu'), epochs=60),
       max_iters=3,
   )
   print(f'converged={res.converged} iters={res.iterations} '
         f'entities={res.n_entities} triples={res.n_triples} '
         f'enm={res.n_enm} corrections={len(res.corrections)}')

The result exposes the same store interface the primitives read, so the freshly
built store answers ``query_triples`` and ``score_triple`` immediately.

.. code-block:: python

   print(res.store.query_triples(head='overdraft'))
   print(round(res.store.score_triple('overdraft', 'has_fee_amount', '35.0'), 3))

What the loop does
------------------

Ingest emits candidate triples for the graph and numeric values for the exact
register. A self-correction pass scores each candidate against the emerging
geometry and repairs the ones it finds implausible or contradictory; the count of
repairs is ``res.corrections``. A final training-and-calibration step turns the
surviving facts into the geometric memory. ``res.converged`` reports whether the
loop reached a fixed point within ``max_iters``.

Binding is the other half. A query arrives in a customer's words, and the tuned
encoder is what carries that phrasing to the stored entity. GEODE tunes the
encoder in the same loop, seeding the labeling from the alias and name edges the
document states and extending the seed geometrically, then fine-tuning until the
label set stops growing. The tuned encoder is written into the store, so the
agent under test and its grader read one encoder.

Loading a production store
--------------------------

A full build is loaded the same way as any store, and the relations recovered
from prose are present alongside those read from tables.

.. code-block:: python

   from knowlytix.knowledge.query import GMSExpertStore

   store = GMSExpertStore(
       DocGMSConfig(store_path=str(STORE), ingest_mode="regex"),
       device=torch.device("cpu"),
   )
   assert store.load()
   for r in ['has_fee_amount', 'has_filing_window_days']:
       print(r, store.query_triples(relation=r)[:1])

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/geode/index` — ``build_rag_store``,
  ``make_default_trainer`` and the self-correction loop.
- :doc:`/3-api-reference/modules/gms_backend/index` — the forgeloop adapters
  that expose a built store to agents.
- :doc:`ch03_gms_primitives` — the primitives the built store answers.
- :doc:`/4-notebook-examples/testing/index` — a small GEODE build and the full
  production store.
