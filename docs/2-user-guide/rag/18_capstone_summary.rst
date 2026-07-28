Chapter 18 — Capstone: The Assembled Pipeline
=============================================

Each preceding page built one stage of a governed RAG over a single annual
report, run on the same corpus so the stages compose. This page introduces no
new mechanism. It assembles the stages into one pipeline, answers through it,
and reads back the head-to-head verdict the designed experiment produced.

The stages
----------

The build turns the report into typed triples, cleans the graph, and trains and
persists the geometric model. The store then serves as its own oracle for
tuning and calibration. The online answer path parses a question, binds it,
retrieves asserted facts with provenance, synthesizes a grounded answer, and
gates it at a calibrated operating point.

.. list-table::
   :header-rows: 1

   * - Stage
     - Page
   * - retrieve a cell, not a chunk
     - :doc:`08_triple_mediated_retrieval`
   * - bind the query to graph vocabulary
     - :doc:`09_binding`
   * - answer through the graph
     - :doc:`10_answering_through_the_gms`
   * - synthesize a grounded answer
     - :doc:`11_grounded_synthesis`
   * - verify with geometry
     - :doc:`12_self_verification`
   * - abstain on a coverage blind spot
     - :doc:`13_abstention_and_coverage`
   * - calibrate every gate
     - :doc:`14_calibration`
   * - score against the store as oracle
     - :doc:`15_evaluation`

Assemble the pipeline
---------------------

The store loads from disk with :func:`forgeloop.rag.load_store`; the answer
backend is a local :class:`~knowlytix.knowledge.llm_backend.LocalTransformersBackend`.
The bank-grade :class:`~knowlytix.knowledge.rag.config.RagConfig` reads the
store's tuned encoders and persisted calibration rather than hand-set
thresholds: query parsing and the relevance accept run on the tuned v-encoder,
the relevance veto on the tuned contradiction encoder, and the accept operating
point comes from the calibration file. The dense fallback is off, answers are
self-verified, and a contradicted claim abstains.

.. code-block:: python

   from forgeloop.rag import load_store
   from knowlytix.knowledge.rag import RagConfig, RagPipeline
   from knowlytix.knowledge.llm_backend import LocalTransformersBackend
   from knowlytix.knowledge.geode import QWEN_3B
   from knowlytix.embedding import FineTunedEmbedding

   store = load_store()
   qwen = LocalTransformersBackend(QWEN_3B, device="cuda")

   sp = store.store_path
   v_ft = FineTunedEmbedding.load(f"{sp}/tuned_encoder")          # v-encoder (14)
   u_ft = FineTunedEmbedding.load(f"{sp}/contradiction_encoder")  # u-encoder (14)

   cfg = RagConfig(
       llm=qwen,
       encoder=v_ft.encode,            # tuned query/relevance v-encoder
       binding="fuzzy",                # 09_binding
       ground_extraction=True,         # 10_answering_through_the_gms
       relevance_gate=True,            # geometric: v accepts, u vetoes
       relevance_mode="geometric",
       relevance_u_encoder=u_ft.encode,
       verify_llm_output=True,         # 12_self_verification
       on_verify_fail="abstain",       # 13_abstention_and_coverage
       dense_fallback=False,           # 16_pluggable_llms_and_dense_fallback
       accept_threshold=0.0,           # else the calibrated value (14_calibration)
   )
   pipe = RagPipeline.from_store(store, cfg)

Answer through it
-----------------

The assembled :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline` answers a
factual question, follows a relation across the graph, and declines a
prose-only question. Each :class:`~knowlytix.knowledge.rag.pipeline.RagAnswer`
reports its ``decision`` alongside the answer.

.. code-block:: python

   for q in ["What is Cloud Platform revenue?",
             "Which region runs the division that contains Cloud Platform?",
             "What is management's outlook for fiscal 2026?"]:
       a = pipe.query(q)
       print(f"{a.decision:8} {q[:52]:52} -> {a.answer[:40]}")

The verdict
-----------

The designed experiment (:doc:`15_evaluation`) scores the geometric system and a
chunk-and-pray baseline on the same question cohort, with the store as oracle,
and writes the comparison to disk. The geometric system reports higher precision
and correctness, and only it abstains; the baseline cannot decline.

.. code-block:: python

   import json
   d = json.load(open("data/enrichment/rag_doe_compare.json"))
   g, b = d["geode"], d["baseline"]
   print(f"precision   GEODE {g['precision_at_k']:.2f} vs baseline {b['precision_at_k']:.2f}")
   print(f"correctness GEODE {g['correctness']:.2f} vs baseline {b['correctness']:.2f}")
   print(f"abstention  GEODE {g['abstention_rate']:.2f} vs baseline {b['abstention_rate']:.2f}")

Scope
-----

The system was shown on one small, clean, single-document report and is not
established to scale unchanged to long, noisy, multi-document corpora. For
exploratory search over prose with no figure to get wrong, the chunk-and-pray
baseline is simpler and adequate.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — the pipeline, config and
  answer types.
- :doc:`16_pluggable_llms_and_dense_fallback` and
  :doc:`17_external_persistence_kal` — swapping backends and persisting the
  graph to an external store.
- :doc:`/4-notebook-examples/rag/index` — the full pipeline run end to end.
