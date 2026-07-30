Chapter 11 — Answering Through the GMS
======================================

Binding turns a question's words into the graph's vocabulary and leaves a list
of bound query triples, each with the asked value as the bare variable ``?``
(:doc:`09_binding`). This page shows how to resolve those bound triples into
facts through the store, using
:class:`~knowlytix.knowledge.rag.retrieve.Retriever`. Retrieval is geometry
alone: no language model runs, an asserted edge is preferred over a predicted
one, multi-hop questions chain through a variable environment, and every fact
carries a ``file:line:char`` provenance span.

Load the store and its provenance ledger
-----------------------------------------

:func:`forgeloop.rag.load_store` reloads the trained store built by the
tutorial's ``build_store.py``, rebuilding the model with the exact build-time
geometry so the state dict loads cleanly. The provenance spans live in a
:class:`~knowlytix.knowledge.geode.provenance.ProvenanceLedger` parsed from the
store's source markdown.

.. code-block:: python

   from forgeloop.rag import load_store
   from knowlytix.knowledge.geode.provenance import ProvenanceLedger

   store = load_store()
   ledger = ProvenanceLedger.from_text(store.markdown)
   print("entities:", store.adapter.num_entities,
         " relations:", store.adapter.num_relations)

Single-hop retrieval with provenance
-------------------------------------

A single-hop question binds to one triple with a known head and relation and an
unknown tail. :class:`~knowlytix.knowledge.rag.binding.TripleBinder` resolves
each non-variable slot to real graph vocabulary; the
:class:`~knowlytix.knowledge.rag.retrieve.Retriever` returns the asserted edge
rather than a parsed number, and attaches the source span.

.. code-block:: python

   from knowlytix.knowledge.rag import Retriever, TripleBinder
   from knowlytix.knowledge.rag.query_triples import QueryTriple

   binder = TripleBinder(store)
   retriever = Retriever(store, ledger)

   bt = binder.bind(QueryTriple("cloud platform", "has_revenue", "?"))
   assert bt.bound   # every non-variable slot resolved to graph vocabulary

   result = retriever.retrieve([bt])
   for f in result.facts:
       print(f"{f.head} {f.relation} {f.tail}  "
             f"[{f.source} conf={f.confidence:.2f}]  @ {f.location}")
   print("answers:", result.answers)

Each returned :class:`~knowlytix.knowledge.rag.retrieve.RetrievedFact` exposes
``head``, ``relation``, ``tail``, ``source``, ``confidence``, ``location`` and
``raw``. For the asserted edge the source is ``"triple"`` at confidence
``1.00``, and the tail ``120.0`` is the byte-exact figure carried by the store,
not a number scraped from prose.

Multi-hop through a variable environment
----------------------------------------

A multi-hop question resolves through a variable environment. An intermediate
variable ``?x`` is bound in the first hop and read in the second; the bare
``?`` marks the asked value. The retriever resolves ``?x`` from hop one and
feeds it into hop two.

.. code-block:: python

   chain = [
       binder.bind(QueryTriple("cloud platform", "has_division", "?x")),
       binder.bind(QueryTriple("?x", "has_region", "?")),
   ]
   assert all(b.bound for b in chain)

   res = retriever.retrieve(chain)
   for f in res.facts:
       print(f"  {f.head} {f.relation} {f.tail}  [{f.source}]  @ {f.location}")
   print("answer:", res.answers)

Both hops come back as separate provenance-bearing facts, and that list is the
audit trail: a reader can follow segment to division to region and check each
edge against its source span. Longer chains extend the same pattern; a three-hop
query reusing ``?x`` for two later hops resolves the same way.

Asserted edges preferred over prediction
-----------------------------------------

The retriever's tail query checks the asserted graph before it calls link
prediction. An edge the graph holds is returned as a hard fact at source
``"triple"``; :meth:`~knowlytix.knowledge.store.GMSExpertStore.link_predict`
fills only a genuinely missing edge and is scored lower.

.. code-block:: python

   asserted = store.query_triples(head="cloud platform", relation="has_revenue")
   print("asserted in graph:", asserted)

   # A missing edge falls back to a ranked, lower-confidence guess:
   guess = store.link_predict("cloud platform", "has_region", top_k=3)
   print("link_predict (no asserted edge):", guess)

The segment-to-region link is not asserted directly, because region lives on the
division, which is why the region question must be answered multi-hop rather
than by a single-edge guess.

Mechanism
---------

Retrieval answers only what the graph asserts or can chain. It follows one path
for an intermediate variable, skips a pattern with two unknown slots and does
not rank tails by world knowledge. Preferring an asserted edge over a prediction
keeps a held fact from being displaced by a guess, and
:meth:`~knowlytix.knowledge.store.GMSExpertStore.query_triples` gates the
fallback to :meth:`~knowlytix.knowledge.store.GMSExpertStore.link_predict`. When
the bound triples do not resolve, ``result.facts`` is empty and
``result.answers`` is empty; that emptiness is what drives abstention
(:doc:`14_abstention_and_coverage`). The retrieved facts and their spans are the
input to grounded synthesis (:doc:`12_grounded_synthesis`).

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — signatures for
  :class:`~knowlytix.knowledge.rag.retrieve.Retriever`,
  :class:`~knowlytix.knowledge.rag.retrieve.RetrievedFact`,
  :class:`~knowlytix.knowledge.rag.binding.TripleBinder` and
  :class:`~knowlytix.knowledge.rag.query_triples.QueryTriple`.
- :doc:`08_triple_mediated_retrieval` and :doc:`09_binding` — the retrieval
  design and the binding step that precedes this one.
- :doc:`/4-notebook-examples/rag/index` — the same retrieval run end to end.
