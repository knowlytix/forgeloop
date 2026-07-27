Binding Query Terms to the Graph
================================

Query triples are written in the user's vocabulary, not the graph's. A user
typing "topline" addresses a store that holds ``has_revenue``; a user saying
"consumer hardware" addresses a graph that knows ``devices``. Binding resolves
each non-variable slot to a term the store actually contains, or, when the term
is genuinely ambiguous, refuses rather than guessing. A silent mis-resolution
returns a confident, wrong, well-provenanced answer, which is the most damaging
failure because the provenance makes it appear trustworthy. An unbound slot is a
designed outcome that feeds abstention (:doc:`13_abstention_and_coverage`).

Binding is performed by
:class:`~knowlytix.knowledge.rag.binding.TripleBinder`, which reads two
vocabulary maps and an entity matcher from the store. The examples below use a
small stand-in store so the behaviour is deterministic and CPU-only; the trained
store loaded by :func:`forgeloop.rag.load_store` exposes the same interface.

.. code-block:: python

   from knowlytix.knowledge.store import GMSExpertStore

   class _Adapter:
       def __init__(self, entities, relations):
           self.entity_to_idx = {e: i for i, e in enumerate(entities)}
           self.relation_to_idx = {r: i for i, r in enumerate(relations)}

   class FakeStore:
       def __init__(self):
           self.adapter = _Adapter(
               entities=["cloud platform", "devices", "logistics", "retail",
                         "total", "technology", "operations", "revenue"],
               relations=["has_revenue", "has_headcount", "has_division",
                          "has_region", "has_head", "in_section"],
           )
       # reuse the shipped matcher verbatim
       fuzzy_match_entity = GMSExpertStore.fuzzy_match_entity

   store = FakeStore()

``FakeStore`` and ``_Adapter`` are notebook fixtures exposing exactly what
:class:`~knowlytix.knowledge.rag.binding.TripleBinder` consumes:
``adapter.entity_to_idx``, ``adapter.relation_to_idx`` and
``fuzzy_match_entity``.

Fuzzy mode: substrings resolve, paraphrases do not
--------------------------------------------------

``TripleBinder(store, mode="fuzzy")`` resolves entities by unique substring and
relations by a ``has_<slug>`` convention that maps a column heading to its
relation. A substring nickname binds; a true paraphrase such as "topline" shares
no substring with any graph term and returns unbound. A bound triple needs every
non-variable slot resolved, and the ``?`` slot is a variable that never counts
against binding.

.. code-block:: python

   from knowlytix.knowledge.rag import TripleBinder
   from knowlytix.knowledge.rag.query_triples import QueryTriple

   fuzzy = TripleBinder(store, mode="fuzzy")

   ok = fuzzy.bind(QueryTriple("cloud", "revenue", "?"))
   print(ok.head, ok.relation, ok.bound)   # cloud platform has_revenue True

   top = fuzzy.bind(QueryTriple("cloud platform", "topline", "?"))
   print(top.relation, top.bound)           # None False

   syn = fuzzy.bind(QueryTriple("consumer hardware", "headcount", "?"))
   print(syn.head, syn.bound)               # None False

Embedding mode: an injectable encoder resolves paraphrases
----------------------------------------------------------

``mode="embedding"`` keeps exact and fuzzy matches authoritative and adds a
nearest-neighbor cosine match over an injectable encoder as the fallback. The
encoder is any callable ``list[str] -> (N, d)`` tensor. In production the default
is the repo MiniLM encoder; a deterministic encoder makes the example
reproducible. "topline" reaches ``has_revenue`` and "consumer hardware" reaches
``devices`` through a shared semantic axis, so embedding binding succeeds exactly
where fuzzy failed.

.. code-block:: python

   import torch

   def fake_encode(texts: list[str]) -> torch.Tensor:
       rows = []
       for t in texts:
           tl = t.lower()
           rows.append([
               1.0 if ("revenue" in tl or "topline" in tl or "sales" in tl) else 0.0,
               1.0 if ("devices" in tl or "hardware" in tl or "gadget" in tl) else 0.0,
               1.0 if ("cloud" in tl or "platform" in tl) else 0.0,
               1.0 if ("headcount" in tl or "staff" in tl or "employees" in tl) else 0.0,
           ])
       return torch.tensor(rows)

   emb = TripleBinder(store, mode="embedding", encoder=fake_encode)
   bt = emb.bind(QueryTriple("consumer hardware", "topline", "?"))
   print(bt.head, bt.relation, bt.bound)   # devices has_revenue True

The embedding space is consulted only when string matching returns nothing.
Omitting ``encoder=`` makes :class:`~knowlytix.knowledge.rag.binding.TripleBinder`
lazily wrap the repo MiniLM encoder, which downloads on first use and requires a
GPU-capable environment.

Refuse on ambiguity
-------------------

Two safeguards constrain the match. A similarity below ``bind_threshold`` is
rejected as out-of-vocabulary, and a top-two pair within ``bind_margin`` is
refused as a near tie rather than choosing one plausible term. Both feed
bind-check abstention.

.. code-block:: python

   # (a) out-of-vocabulary term encodes to the zero vector, cosine below threshold
   oov = emb.bind(QueryTriple("the unit", "revenue", "?"))
   print(oov.head, oov.bound)      # None False

   # (b) a term equidistant from two relations is refused as a near tie
   def tie_encode(texts):
       rows = []
       for t in texts:
           tl = t.lower()
           if "growth" in tl:
               rows.append([0.7, 0.7, 0.0, 0.0])   # ambiguous query
           else:
               rows.append([1.0 if "revenue" in tl else 0.0,
                            1.0 if "headcount" in tl else 0.0, 0.0, 0.0])
       return torch.tensor(rows)

   tie = TripleBinder(store, mode="embedding", encoder=tie_encode, bind_margin=0.1)
   amb = tie.bind(QueryTriple("cloud platform", "growth", "?"))
   print(amb.relation, amb.bound)  # None False

An unbound query triple means the pipeline declines to answer rather than
fabricate a binding.

Where the thresholds come from
------------------------------

Binding decides whether a term resolves, not whether the thresholds suit a
corpus. ``bind_threshold`` and ``bind_margin`` are set under calibration:
:func:`~knowlytix.knowledge.rag.eval.calibrate_bind_threshold` fits the operating
point from labeled data (:doc:`14_calibration`). Binding also does not check
whether a bound relation applies to its bound entity; that relevance check occurs
downstream in :doc:`10_answering_through_the_gms`.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — signatures for
  :class:`~knowlytix.knowledge.rag.binding.TripleBinder` and
  :func:`~knowlytix.knowledge.rag.eval.calibrate_bind_threshold`.
- :doc:`08_triple_mediated_retrieval` — producing the query triples the binder
  consumes.
- :doc:`14_calibration` — fitting ``bind_threshold`` and ``bind_margin``.
- :doc:`/4-notebook-examples/rag/index` — binding run against the trained store.
