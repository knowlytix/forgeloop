Chapter 14 — Abstention and Coverage
====================================

This page shows how to make a triple-mediated pipeline decline a question it
cannot ground, and how to measure the document regions the pipeline cannot
reach. Two calls carry the work:
:func:`knowlytix.knowledge.rag.coverage.coverage_report` walks a store section by
section and reports which sections have body text but zero triples, and
:class:`knowlytix.knowledge.rag.pipeline.RagPipeline` abstains when a query binds
to no known entity or relation. Both draw on the store loaded by
:func:`forgeloop.rag.load_store`.

Two measurements
----------------

Coverage and abstention answer different questions and run on different paths.

.. list-table::
   :header-rows: 1

   * - Measurement
     - Call
     - Path
   * - Section coverage
     - ``coverage_report(store)``
     - CPU only; reads markdown plus triples.
   * - Query-time abstention
     - ``pipe.query(q)``
     - GPU; loads the store and the LLM.

Measure coverage
----------------

Coverage is computed from the markdown plus the store's triple list, so it needs
no model forward pass. The trained store ships its triples as ``triples.json``;
a minimal view carrying ``.markdown`` and ``.triples`` is all
:func:`~knowlytix.knowledge.rag.coverage.coverage_report` reads, and in
production the real store is passed in its place.

.. code-block:: python

   import json, os
   from dataclasses import dataclass, field
   from forgeloop.rag import load_store, CORPUS, STORE
   from knowlytix.knowledge.rag import coverage_report

   with open(CORPUS) as f:
       markdown = f.read()
   with open(os.path.join(STORE, "triples.json")) as f:
       TRIPLES = [tuple(t) for t in json.load(f)]

   @dataclass
   class _StoreView:
       markdown: str
       triples: list = field(default_factory=list)
       store_path: str = ""

   store_view = _StoreView(markdown=markdown, triples=TRIPLES, store_path=STORE)
   report = coverage_report(store_view)

   print(f"coverage_ratio = {report.coverage_ratio:.2f}")
   for r in report.regions:
       mark = "x" if r.covered else " "
       flag = "  <-- BLIND SPOT" if r.blind_spot else ""
       print(f"[{mark}] {r.title} ({r.triple_count} triples, "
             f"{r.body_lines} body lines){flag}")

On the Northwind FY2025 report the coverage ratio is 0.56. The fact-bearing
tables (Segment Performance, Income Statement, Balance Sheet) carry triples; the
qualitative prose sections are named as blind spots.

.. code-block:: text

   coverage_ratio = 0.56
   [x] 1. Segment Performance (15 triples, 9 body lines)
   [x] 3. Income Statement (8 triples, 7 body lines)
   [ ] 6. Management Discussion and Analysis (0 triples, 9 body lines)  <-- BLIND SPOT
   [ ] 7. Risk Factors (0 triples, 8 body lines)  <-- BLIND SPOT
   [ ] 8. Outlook (0 triples, 5 body lines)  <-- BLIND SPOT

The ratio is not a defect to drive to 1.0. Those prose sections carry no
authoritative figure the report itself relies on, and the coverage monitor
names and counts the gap rather than hiding it.

Abstain on an ungrounded question
----------------------------------

A question that targets a blind-spot section binds to no entity or relation, so
the bind-check fires and the pipeline declines with a notice rather than
summarizing prose it cannot verify. The following loads the trained store and
runs the local LLM on the GPU; the calls are shown as written in the notebook
and are not executed here.

.. code-block:: python

   from knowlytix.knowledge.llm_backend import LocalTransformersBackend
   from knowlytix.knowledge.rag import RagConfig, RagPipeline
   from knowlytix.knowledge.geode import QWEN_3B

   store = load_store()
   qwen = LocalTransformersBackend(QWEN_3B)
   rag = RagConfig(llm=qwen)            # abstain-not-guess defaults, dense off
   pipe = RagPipeline.from_store(store, rag)

   ans = pipe.query("What are the company's main risk factors?")
   print("decision:", ans.decision)    # abstain
   print("route:   ", ans.route)       # triple
   print("notice:  ", ans.notice)      # did not bind to known entities/relations

The answer is the refusal string, not a paraphrase of the Risk Factors prose.

Read the audit record
----------------------

Every :class:`~knowlytix.knowledge.rag.pipeline.RagAnswer`, accepted or
abstained, exposes a structured audit record through ``ans.audit()``: the
decision, the route, the query triples, what bound, the source facts and their
provenance spans, and the verification verdicts. An abstention records
``verified=True``, because declining on no grounded fact is a verified outcome
rather than an error.

.. code-block:: python

   import json
   audit = ans.audit()
   print(json.dumps(audit, indent=2, default=str))
   assert audit["decision"] == "abstain"
   assert audit["route"] == "triple"
   assert audit["verified"] is True

The record for an abstention carries an empty ``sources`` list; there was no
grounded fact to cite, which is why the pipeline declined. To capture every
query automatically, wire the record to ``RagConfig.audit_sink``.

How anchoring gates coverage
----------------------------

A blind spot clears only when a triple's provenance resolves into that section,
which the coverage monitor cannot be talked into otherwise. The
:class:`~knowlytix.knowledge.geode.provenance.ProvenanceLedger` resolves table
cells, section headers and schema bullets; a triple invented out of thin air
resolves to line ``-1`` and counts toward nothing. An anchored triple keyed to
a real header does resolve.

.. code-block:: python

   from knowlytix.knowledge.geode.provenance import ProvenanceLedger
   ledger = ProvenanceLedger.from_text(markdown)

   fake = ("cloud platform", "has_outlook", "continued investment")
   print(ledger.resolve(*fake).line_no)             # -1 (unaligned)

   outlook_triple = ("outlook", "in_section", "outlook")
   print(ledger.resolve(*outlook_triple).line_no)   # resolves to the header

   augmented = _StoreView(markdown=markdown,
                          triples=TRIPLES + [outlook_triple],
                          store_path=STORE)
   after = coverage_report(augmented, exclude_relations=())

Because :func:`~knowlytix.knowledge.rag.coverage.coverage_report` excludes
structural relations by default, ``exclude_relations=()`` is passed so the
anchored ``in_section`` triple registers. Only real evidence moves the ratio;
the remedy for a blind spot is more anchored triples, not distrusted dense
retrieval.

Limits
------

The monitor measures triple presence per section, not answer quality. A section
with triples can still miss the specific attribute a question asks for, which is
the relevance gate's concern (:doc:`11_answering_through_the_gms`), and abstention
here is binary: the pipeline does not rank how close a query came to binding.
Setting the bind threshold so paraphrases resolve while off-topic queries still
abstain is the subject of :doc:`15_calibration`.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — signatures for
  :func:`~knowlytix.knowledge.rag.coverage.coverage_report`,
  :func:`~knowlytix.knowledge.rag.coverage.graph_coverage`,
  :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline` and
  :class:`~knowlytix.knowledge.rag.pipeline.RagAnswer`.
- :doc:`09_binding` and :doc:`15_calibration` — the bind-check and its
  calibrated operating point.
- :doc:`/4-notebook-examples/rag/index` — the same coverage and abstention run
  end to end.
