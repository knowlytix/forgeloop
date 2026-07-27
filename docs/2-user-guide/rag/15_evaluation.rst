Evaluating the RAG with a Designed Experiment
=============================================

A single accuracy number rewards a confident guess as much as a verified answer.
This page shows how to score the assembled RAG against the designed-experiment
cohort with the store as oracle, reading four metrics off every answer:
precision and recall at ``k``, correctness and completeness. Because the answers
are checked against ground truth the store already holds, no human labeling is
involved. The store is loaded by :func:`forgeloop.rag.load_store`, and two
knowlytix evaluators supply the grounded checks,
:class:`knowlytix.harness.testing.hallucination.HallucinationOracle` and
:class:`knowlytix.harness.testing.completeness.CompletenessEvaluator`.

The four metrics
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Metric
     - Question
     - Check
   * - precision@k
     - How many of the top-``k`` retrieved units carry the value?
     - value-equality, not string overlap
   * - recall@k
     - Is the ground-truth value among the top-``k``?
     - value-equality
   * - correctness
     - Is the asserted value grounded for the asked relation?
     - ``HallucinationOracle``, calibrated per-relation cut
   * - completeness
     - How many store-grounded atoms did the answer cover?
     - ``CompletenessEvaluator``

Load the store and the evaluators
---------------------------------

The oracle and the completeness evaluator both read the same store the pipeline
answers from, and the cohort is the designed-experiment question set built
earlier (:doc:`06_doe_enrichment`).

.. code-block:: python

   import json, os
   from forgeloop.rag import load_store
   from knowlytix.knowledge.llm_backend import LocalTransformersBackend
   from knowlytix.knowledge.rag import RagConfig, RagPipeline
   from knowlytix.knowledge.geode import QWEN_3B
   from knowlytix.harness.testing.hallucination import HallucinationOracle
   from knowlytix.harness.testing.completeness import CompletenessEvaluator

   store = load_store()
   pipe = RagPipeline.from_store(store, RagConfig(llm=LocalTransformersBackend(QWEN_3B)))
   oracle = HallucinationOracle(store=store)
   comp = CompletenessEvaluator(store)

   cohort = json.load(open(os.path.join(REPO, "data", "enrichment", "rag_cohort.json")))

   def gtv(e):
       return str(e[1]) if isinstance(e, list) else str(e)

   def golden(c):
       g = gtv(c["expected_answer"])
       return [(h, r, str(t)) for h, r, t in store.triples if str(t) == g][:1]

Score every answer
-------------------

Each question is run through the pipeline. Precision and recall are read from
the top-``k`` sources by value-equality; correctness and completeness are read
only for committed answers, so an abstention is counted separately rather than
scored as a wrong answer. Correctness extracts the numeric span the answer
asserts and asks the oracle whether that value is grounded for the golden head
and relation.

.. code-block:: python

   import re
   P = R = C = CO = AB = N = NA = 0
   for c in cohort:
       a = pipe.query(c["question"])
       g = gtv(c["expected_answer"])
       N += 1
       top = a.sources[:3]
       hit = [f for f in top if str(f.tail) == g]
       P += len(hit) / max(1, len(top))
       R += 1.0 if hit else 0.0
       if a.decision != "accept":
           AB += 1
           continue
       NA += 1
       gold = golden(c)
       if gold:
           h, r, _ = gold[0]
           m = re.findall(r"\d[\d,]*\.?\d*", (a.answer or "").replace(",", ""))
           CO += 1.0 if (m and oracle.assess_claim(h, r, m[0]).passed) else 0.0
           C += comp.evaluate(a.answer or "", [g], {"head": h}).score
   print(f"precision@3={P/N:.3f} recall@3={R/N:.3f} "
         f"correctness={CO/max(1,NA):.3f} completeness={C/max(1,NA):.3f} "
         f"abstention={AB/N:.3f}")

Correctness scores a real-but-wrong figure, such as a prior-year number, as
fabricated, because ``oracle.assess_claim`` grounds the asserted value against
the store's calibrated per-relation cut rather than against string proximity.

Compare against the baseline
----------------------------

The same cohort and metrics run against the chunk-and-pray baseline through the
comparison script, which writes both metric sets to disk for side-by-side
reading. The cohort is answerable by construction, so it measures retrieval on
in-scope questions, not out-of-scope abstention.

.. code-block:: python

   import subprocess, sys
   subprocess.run([sys.executable, os.path.join(REPO, "scripts", "rag_doe_compare.py"),
                   "--limit", "150", "--k", "3"], check=True)

   d = json.load(open(os.path.join(REPO, "data", "enrichment", "rag_doe_compare.json")))
   for m in ["precision_at_k", "recall_at_k", "correctness", "completeness", "abstention_rate"]:
       print(f"{m:16}{d['geode'][m]:>8.3f}{d['baseline'][m]:>10.3f}")

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — the evaluation and calibration
  API, including :class:`~knowlytix.knowledge.rag.eval.EvalCase`,
  :func:`~knowlytix.knowledge.rag.eval.benchmark_retrieval` and
  :func:`~knowlytix.knowledge.rag.eval.calibrate_accept_threshold`.
- :doc:`06_doe_enrichment` — the designed-experiment cohort used as the test set.
- :doc:`13_abstention_and_coverage` and :doc:`14_calibration` — the abstention
  behavior these metrics separate the two architectures on, and the calibrated
  cut correctness reads.
- :doc:`/4-notebook-examples/rag/index` — the evaluation and the baseline
  comparison run end to end.
