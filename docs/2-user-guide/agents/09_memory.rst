Chapter 9 — Memory: Types, Retrieval and Hybrid Stores
======================================================

This page shows how to build the memory stores in
:mod:`forgeloop.agents.memory`, one per question an agent asks about what it
knows, and combine them under :class:`~forgeloop.agents.memory.HybridMemory`. What
is known about an entity is resolved by relevance, how one entity relates to
another by following relationships, and what just happened by recency, so the
means of storage and retrieval differ by question. A fourth tier on the GMS
substrate closes the two failure modes a flat store cannot.

The pieces
----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Type
     - Role
   * - :class:`~forgeloop.agents.memory.MemoryItem`, :class:`~forgeloop.agents.memory.MemoryKind`
     - A stored item and its taxonomy label.
   * - :class:`~forgeloop.agents.memory.ShortTermMemory`
     - Bounded queue with substring query, resolved by recency.
   * - :class:`~forgeloop.agents.memory.VectorMemory`, ``chunk_text``
     - Chunk, embed and retrieve by cosine similarity.
   * - :class:`~forgeloop.agents.memory.GraphMemory`, :class:`~forgeloop.agents.memory.Triple`
     - Triple store walked breadth-first for structure.
   * - :class:`~forgeloop.agents.memory.HybridMemory`
     - Weighted combination of the three stores.
   * - :class:`~forgeloop.agents.gms_backend.GMSMemory`
     - Exact numeric recall and a contradiction check.

Items and their kind
--------------------

Every store holds :class:`~forgeloop.agents.memory.MemoryItem` values. Each item
carries a :class:`~forgeloop.agents.memory.MemoryKind` that travels with it for
evaluation and audit; the kind labels the item and does not alter retrieval. The
:class:`~forgeloop.agents.memory.Memory` protocol is ``add(item)`` and
``query(q, k)``, so a store is any object with those two methods.

Recency, relevance and structure
--------------------------------

:class:`~forgeloop.agents.memory.ShortTermMemory` is a bounded ring buffer that
drops the oldest item once full and matches on substrings of recent content.

.. code-block:: python

   from forgeloop.agents.memory import MemoryItem, MemoryKind, ShortTermMemory

   short = ShortTermMemory(maxlen=4)
   for msg in ['hi', 'I was charged a fee', 'specifically an overdraft fee', 'help me dispute it']:
       short.add(MemoryItem(content=msg, kind=MemoryKind.SHORT_TERM))
   print(short.query('fee'))

:class:`~forgeloop.agents.memory.VectorMemory` chunks documents with
``chunk_text``, embeds each chunk and returns the top-k by cosine similarity. It
takes the embedder as a constructor argument and ships no default, since retrieval
is only as good as the embeddings. The embedder needs an ``embed(texts) ->
list[list[float]]`` method; the notebook supplies one over the local GMS encoder,
which requires ``knowlytix``.

.. code-block:: python

   from forgeloop.agents.memory import VectorMemory, chunk_text
   from knowlytix.core.graph.encoders import encode_texts

   class GMSEmbedder:
       dim = 384
       def embed(self, texts):
           return encode_texts(list(texts), model_name='sentence-transformers/all-MiniLM-L6-v2').tolist()

   vec = VectorMemory(GMSEmbedder())
   for i, chunk in enumerate(chunk_text(open('data/policies/overdraft.txt').read(), window=200, overlap=20)):
       vec.add(MemoryItem(content=chunk, kind=MemoryKind.POLICY, metadata={'chunk': i}))
   for h in vec.query('overdraft fee policy', k=3):
       print(f'  {h.content[:80]!r}')

:class:`~forgeloop.agents.memory.GraphMemory` stores
:class:`~forgeloop.agents.memory.Triple` facts and walks them with
:meth:`~forgeloop.agents.memory.GraphMemory.neighbors`, answering relational and
provenance queries that similarity cannot. Traversal follows edges in both
directions, so a forward-only support chain filters the neighbors to those whose
subject is the current node.

.. code-block:: python

   from forgeloop.agents.memory import GraphMemory, Triple

   g = GraphMemory()
   g.add_triple(Triple('overdraft_policy', 'governs', 'overdraft_fee'))
   g.add_triple(Triple('overdraft_fee', 'reversible_via', 'fee_reversal_policy'))
   g.add_triple(Triple('fee_reversal_policy', 'requires', 'manager_approval'))
   for t in g.neighbors('overdraft_policy', hops=2):
       print(f'  {t.subject} --[{t.relation}]--> {t.object}')

Combine the stores
------------------

:class:`~forgeloop.agents.memory.HybridMemory` routes a write to all three stores
and combines reads into one ranking. A vector hit contributes ``alpha``, a graph
hit ``beta`` and a short-term hit ``gamma`` scaled by exponential recency decay,
summed per item id before ranking. A ``ValueError`` from the vector store on empty
content is suppressed, so a write never fails on one backend.

.. code-block:: python

   from forgeloop.agents.memory import HybridMemory

   hybrid = HybridMemory(
       vector=VectorMemory(GMSEmbedder()),
       graph=GraphMemory(),
       short_term=ShortTermMemory(maxlen=8),
   )
   hybrid.add(MemoryItem(content='Overdraft fees are $35 per occurrence.', kind=MemoryKind.POLICY))
   for h in hybrid.query('overdraft fee', k=3):
       print(f'  {h.content[:80]!r}')

The GMS tier
------------

Two failure modes remain where a banking agent cannot be wrong: a model in the
retrieval path reconstructs a number instead of reading it back, and a flat vector
store silently accepts a fact that contradicts what it holds.
:class:`~forgeloop.agents.gms_backend.GMSMemory` wraps a trained store and closes
both. :meth:`~forgeloop.agents.gms_backend.GMSMemory.lookup_enm` recalls a number
byte-exactly from the numeric register, and
:meth:`~forgeloop.agents.gms_backend.GMSMemory.score_triple` scores a candidate
fact so a contradiction on a functional relation scores far above the committed
value. The block below requires ``knowlytix`` and ``torch``.

.. code-block:: python

   import torch
   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
   from forgeloop.agents.gms_backend import GMSMemory

   store = GMSExpertStore(
       DocGMSConfig(store_path='data/gms_banking_store'),
       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
   store.load()
   mem = GMSMemory(store)

   print(mem.lookup_enm('fee_schedule', 'overdraft/per_occurrence'))       # exact, never parsed
   print(mem.score_triple('representative', 'has_max_reversal', '35.0'))   # committed
   print(mem.score_triple('representative', 'has_max_reversal', '100.0'))  # contradicts

Why one store per question
--------------------------

The stores differ because the questions differ, and the hybrid does not collapse
them into one index; it keeps each backend's strength and weights the evidence.
Similarity retrieval generalizes across phrasing, triple traversal recovers a
chain that is stored nowhere as a single fact, and the recency buffer surfaces the
active turn. The GMS tier admits a write only if the new fact survives the same
geometric check an action would, so a contradiction above the calibrated threshold
is rejected rather than stored. That store is built in :doc:`02_knowledge_graphs`.

See also
--------

- :doc:`/3-api-reference/modules/memory/index` — the stores, :class:`~forgeloop.agents.memory.MemoryItem` and ``chunk_text``.
- :doc:`/3-api-reference/modules/gms_backend/index` — :class:`~forgeloop.agents.gms_backend.GMSMemory`.
- :doc:`02_knowledge_graphs` — the triple store and the GMS substrate the fourth tier reads.
- :doc:`/4-notebook-examples/agents/index` — the four tiers and the hybrid run end to end.
