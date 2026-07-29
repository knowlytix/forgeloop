Chapter 8 — Triple-Mediated Retrieval
=====================================

A chunk-and-pray pipeline hands a question verbatim to a vector index and trusts
the model to read an answer from the nearest chunks. This page takes a different
first step. A question is translated into query triples, patterns of the form
``(head, relation, tail)`` in which the asked-for value is the bare variable
``?``, and those triples are later bound to the graph's vocabulary
(:doc:`09_binding`) and answered through the geometric memory
(:doc:`10_answering_through_the_gms`). The surface form of the question never
touches an opaque similarity search.

The query-triple vocabulary
---------------------------

Three symbols from :mod:`knowlytix.knowledge.rag.query_triples` define the
extraction layer.

.. list-table::
   :header-rows: 1

   * - Symbol
     - Role
   * - :class:`~knowlytix.knowledge.rag.query_triples.QueryTriple`
     - The ``(head, relation, tail)`` pattern.
   * - :data:`~knowlytix.knowledge.rag.query_triples.ASKED`
     - The bare ``?`` marking the asked slot.
   * - :class:`~knowlytix.knowledge.rag.query_triples.QueryTripleExtractor`
     - Turns NL into triples over an injectable LLM.

:func:`~knowlytix.knowledge.rag.query_triples.is_var` tests whether a slot is a
variable, and :func:`~knowlytix.knowledge.rag.query_triples.schema_from_store`
reads the graph's real names for grounding.

.. code-block:: python

   from knowlytix.knowledge.rag.query_triples import (
       ASKED, QueryTriple, QueryTripleExtractor, is_var, schema_from_store,
   )
   from knowlytix.knowledge.llm_backend import LLMBackend

   print(is_var(ASKED))            # True  — the asked slot
   print(is_var("?x"))             # True  — an intermediate variable
   print(is_var("cloud platform")) # False — a bound term

Inject an LLM backend
---------------------

:class:`~knowlytix.knowledge.rag.query_triples.QueryTripleExtractor` takes any
:class:`~knowlytix.knowledge.llm_backend.LLMBackend`. The production backend is a
local model (below); a deterministic backend that replays fixed JSON makes the
extraction reproducible without a GPU. The extractor code is identical, only the
backend swaps.

.. code-block:: python

   class ScriptedBackend(LLMBackend):
       """Replays canned JSON keyed by question — stands in for the model."""

       def __init__(self, script: dict[str, str], default: str = "[]"):
           self._script = script
           self._default = default

       def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
           question = user.split("\n\nNote:")[0].strip()
           return self._script.get(question, self._default)

       @property
       def model_name(self) -> str:
           return "scripted/qwen2.5-3b-instruct"

   SCRIPT = {
       "How many people work in Logistics?":
           '[{"head":"logistics","relation":"has_headcount","tail":"?"}]',
       "In which region is Cloud Platform's division?":
           '[{"head":"cloud platform","relation":"has_division","tail":"?x"},'
           '{"head":"?x","relation":"has_region","tail":"?"}]',
   }
   extractor = QueryTripleExtractor(ScriptedBackend(SCRIPT))

``ScriptedBackend`` is a notebook fixture, not a knowlytix symbol; it illustrates
the injection seam that
:class:`~knowlytix.knowledge.llm_backend.LLMBackend` defines.

Extract triples from a question
-------------------------------

``extract`` returns a list of :class:`~knowlytix.knowledge.rag.query_triples.QueryTriple`.
Factual, numeric and multi-hop questions all reduce to triple patterns, and each
extraction exposes exactly one asked slot. A numeric question has the same shape
as a factual lookup; exactness is enforced later by Exact Numerical Memory, not
by parsing prose.

.. code-block:: python

   qts = extractor.extract("How many people work in Logistics?")
   asked = sum(t.tail == ASKED for t in qts)
   for t in qts:
       print(t.as_tuple())        # ('logistics', 'has_headcount', '?')
   assert asked == 1

A multi-hop question links two triples through a named intermediate variable, so
the hop structure is explicit and inspectable before any answering happens.

.. code-block:: python

   multi = extractor.extract("In which region is Cloud Platform's division?")
   assert len(multi) == 2
   assert multi[0].tail == "?x" and multi[1].head == "?x"   # the chain link
   assert multi[1].tail == ASKED                            # the asked value
   print("hop 1:", multi[0].as_tuple())  # ('cloud platform', 'has_division', '?x')
   print("hop 2:", multi[1].as_tuple())  # ('?x', 'has_region', '?')

Ground extraction in the store's schema
----------------------------------------

Schema grounding is the principal reliability lever for a small model. Left to
its own vocabulary a 3B model emits plausible but non-binding relation names such
as ``staff_count``; reading the graph's actual relation and entity names into the
prompt makes the output binding-compatible.
:func:`~knowlytix.knowledge.rag.query_triples.schema_from_store` returns a
``{"relations": [...], "entities": [...]}`` mapping, filtering out the
``in_section`` bookkeeping relation and the numeric value-entities, and is passed
to the extractor as ``vocab=``.

.. code-block:: python

   vocab = schema_from_store(store)          # e.g. {"relations": ["has_headcount", ...], ...}
   grounded = QueryTripleExtractor(backend, vocab=vocab)

   ungrounded = QueryTripleExtractor(ScriptedBackend(
       {"How many people work in Logistics?":
        '[{"head":"logistics","relation":"staff_count","tail":"?"}]'}))

   bad = ungrounded.extract("How many people work in Logistics?")[0]
   good = grounded.extract("How many people work in Logistics?")[0]
   print(bad.relation)   # staff_count  — not a graph relation, will not bind
   print(good.relation)  # has_headcount — a real relation, binds

Repair a mis-extraction
-----------------------

Grounding reduces but does not eliminate mis-extraction. The pipeline's
extract-and-bind step runs a query-only repair loop: when a non-variable slot
fails to bind, it re-asks the extractor once with a ``hint`` naming the terms
that did not match and the real relations to use instead. The repair corrects the
question's encoding and never edits the graph.

.. code-block:: python

   class RepairingBackend(LLMBackend):
       """First call mis-extracts; a follow-up with a hint corrects it."""

       def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
           if "Note:" in user:   # the repair retry carries a hint
               return '[{"head":"logistics","relation":"has_headcount","tail":"?"}]'
           return '[{"head":"logistics","relation":"headcount","tail":"?"}]'

       @property
       def model_name(self) -> str:
           return "scripted/repairing"

   rb = QueryTripleExtractor(RepairingBackend())
   first = rb.extract("How many people work in Logistics?")[0]
   repaired = rb.extract(
       "How many people work in Logistics?",
       hint="These terms did not match: ['headcount']. Re-express using ONLY: has_headcount.",
   )[0]
   print(first.relation)     # headcount     — does not bind
   print(repaired.relation)  # has_headcount  — binds

``RepairingBackend`` is a notebook fixture standing in for the two-pass behaviour
of the real extractor.

The production wiring
---------------------

The production path loads the trained store through
:func:`forgeloop.rag.load_store`, grounds the extractor with the live schema and
drives it with a local model via
:class:`~knowlytix.knowledge.llm_backend.LocalTransformersBackend`. This step
requires a GPU and a licensed knowlytix build.

.. code-block:: python

   import torch
   from forgeloop.rag import load_store
   from knowlytix.knowledge.llm_backend import LocalTransformersBackend
   from knowlytix.knowledge.rag.query_triples import (
       QueryTripleExtractor, schema_from_store, ASKED,
   )

   store = load_store("data/gms_annual_report_store")
   dev = "cuda" if torch.cuda.is_available() else "cpu"

   vocab = schema_from_store(store)
   qwen = LocalTransformersBackend("Qwen/Qwen3-4B-Instruct-2507", device=dev)
   extractor = QueryTripleExtractor(qwen, vocab=vocab)

   qts = extractor.extract("How many people work in Logistics?")
   assert any(t.tail == ASKED for t in qts)               # an asked slot
   assert any(t.relation == "has_headcount" for t in qts) # grounding bound it

Extraction produces a checkable form; it does not itself answer. An attribute the
graph never recorded yields abstention rather than a forced answer. The end-to-end
run through :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline` returns a
decision and provenance and is the subject of :doc:`10_answering_through_the_gms`.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — signatures for
  :class:`~knowlytix.knowledge.rag.query_triples.QueryTripleExtractor` and
  :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline`.
- :doc:`/3-api-reference/modules/knowlytix/llm/index` —
  :class:`~knowlytix.knowledge.llm_backend.LLMBackend` and
  :class:`~knowlytix.knowledge.llm_backend.LocalTransformersBackend`.
- :doc:`09_binding` — resolving the extracted terms to the store's vocabulary.
- :doc:`/4-notebook-examples/rag/index` — extraction run against the trained store.
