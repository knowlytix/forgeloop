Chapter 1 — Why "Chunk and Pray" Fails
======================================

This page reproduces the default retrieval recipe on a single numeric question,
then runs the same question through the triple-mediated store so the two answers
can be compared side by side. The recipe splits a document into chunks, embeds
each chunk into a vector and, at query time, returns the top-k chunks nearest the
question. The failure it exhibits is structural, so a deterministic bag-of-words
encoder reproduces it without a GPU or a language model. The contrast uses
:func:`forgeloop.rag.load_store` and :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline`,
which load the trained store and a local synthesizer.

The question is "What was total revenue?", whose answer, 355.0, is the Revenue
row of the income statement.

Four failure modes
-------------------

The recipe fails in four ways on this question, and the four follow from one
cause: a lexical ranker treats "total" in "total revenue" as matching "Total
Assets" and "Total Liabilities" as strongly as the revenue row.

.. list-table::
   :header-rows: 1

   * - Failure
     - Cause in the recipe
   * - Hallucinated retrieval
     - top-k ranks the balance-sheet block above the revenue row.
   * - No provenance
     - a chunk spans many rows and points to no single cell.
   * - Silently wrong numbers
     - the figure is parsed out of retrieved text, unchecked.
   * - Unverifiable
     - the only available check is a second model grading the first.

Load the corpus
---------------

The annual report is plain markdown. Every authoritative number lives in a
table; the MD&A, Risk and Outlook sections are prose with no figures. The
report path is exported by the :mod:`forgeloop.rag` helper as ``CORPUS``.

.. code-block:: python

   from forgeloop.rag import CORPUS

   report = open(CORPUS).read()
   lines = report.split("\n")

Build a vanilla vector RAG
--------------------------

The recipe is chunk, embed, rank by cosine, take top-k. Chunks are blank-line
paragraphs paired with their 1-based start line; the encoder is a deterministic
bag-of-words vector standing in for a dense encoder.

.. code-block:: python

   import math, re
   from collections import Counter

   def chunk(md: str) -> list[tuple[int, str]]:
       """Blank-line paragraph chunks, paired with their 1-based start line."""
       chunks, buf, start = [], [], None
       for i, ln in enumerate(md.split("\n"), start=1):
           if ln.strip() == "":
               if buf:
                   chunks.append((start, "\n".join(buf)))
               buf, start = [], None
           else:
               start = i if start is None else start
               buf.append(ln)
       if buf:
           chunks.append((start, "\n".join(buf)))
       return chunks

   def embed(text: str) -> Counter:
       return Counter(re.findall(r"[a-z]+", text.lower()))

   def cosine(a: Counter, b: Counter) -> float:
       shared = set(a) & set(b)
       dot = sum(a[t] * b[t] for t in shared)
       na = math.sqrt(sum(v * v for v in a.values()))
       nb = math.sqrt(sum(v * v for v in b.values()))
       return dot / (na * nb) if na and nb else 0.0

   class VanillaRAG:
       def __init__(self, md: str):
           self.chunks = chunk(md)
           self.index = [embed(c) for _, c in self.chunks]

       def retrieve(self, q: str, top_k: int = 1):
           qv = embed(q)
           scored = [(cosine(qv, cv), self.chunks[i])
                     for i, cv in enumerate(self.index)]
           scored.sort(key=lambda x: x[0], reverse=True)
           return scored[:top_k]

   rag = VanillaRAG(report)

Run the numeric question
------------------------

The top chunk is the balance-sheet table, whose figures are 540.0, 210.0 and
330.0 and do not include the answer. The income-statement Revenue cell, 355.0, is
out-ranked, so the recipe hands the wrong table to the answer layer with no
cell-level span to cite.

.. code-block:: python

   question = "What was total revenue?"
   hits = rag.retrieve(question, top_k=3)
   for rank, (score, (line_no, text)) in enumerate(hits, start=1):
       snippet = text.replace("\n", " ")[:80]
       print(f"#{rank}  score={score:.3f}  line={line_no}  {snippet!r}")

   top_score, (top_line, top_text) = hits[0]
   has_answer = bool(re.search(r"\b355(?:\.0)?\b", top_text))
   numbers = re.findall(r"\d+\.\d+", top_text)
   print(f"top chunk contains the answer (355): {has_answer}")
   print(f"numbers in the top chunk (ambiguous): {numbers}")

Retrieving the right table would not repair this. Asked for net income (70.0),
the top hit is the income-statement table, but that chunk holds eight numbers
across four rows and two years, so no single provenance-backed figure is
recoverable. Retrieving the right table is not the same as answering the
question.

The same question through the store
------------------------------------

The triple-mediated route inverts each failure. The figure is read byte-exact
from Exact Numerical Memory rather than parsed from prose, and the retrieved
triple carries a ``file:line:char`` span back to the source cell. This listing
loads the trained store and a local Qwen synthesizer, so it needs a GPU and the
licensed store; the values shown are those the notebook asserts.

.. code-block:: python

   from forgeloop.rag import load_store
   from knowlytix.knowledge.llm_backend import LocalTransformersBackend
   from knowlytix.knowledge.rag import RagConfig, RagPipeline
   from knowlytix.knowledge.geode import QWEN_3B

   store = load_store()

   # ENM gives the authoritative number directly, byte-exact, no parsing.
   enm_total = store.lookup_enm("income_statement", "Revenue/FY2025")
   print(f"lookup_enm -> {enm_total}")   # 355.0

   qwen = LocalTransformersBackend(QWEN_3B, device=str(store.device))
   pipe = RagPipeline.from_store(store, RagConfig(llm=qwen))
   ans = pipe.query("What was total revenue?")
   print("answer  :", ans.answer)
   print("decision:", ans.decision, "| route:", ans.route,
         "| verified:", ans.verified)
   for f in ans.sources:
       print(f"  fact: ({f.head}, {f.relation}, {f.tail})  "
             f"src={f.source}  @ {f.location}")

Where triples exist, the route answers with a source; where they do not, it
abstains. The report's four prose sections carry no triples, so a question
answerable only from prose returns an abstention rather than a guess. This is the
trade the rest of the part develops: an answer that can be checked against a span,
or no answer at all. The distrusted dense index becomes an opt-in fallback
(:doc:`16_pluggable_llms_and_dense_fallback`), not the default retriever.

See also
--------

- :doc:`02_geometric_memory` — the store primitives the triple-mediated route calls.
- :doc:`03_provenance` — how a retrieved triple resolves to its ``file:line:char`` span.
- :doc:`/3-api-reference/modules/knowlytix/store/index` — :class:`~knowlytix.knowledge.store.GMSExpertStore` and its methods.
- :doc:`/4-notebook-examples/rag/index` — the same comparison run end to end.
