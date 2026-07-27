Knowledge Graphs and the Geometric Memory Substrate
===================================================

This page shows how to record domain facts as typed triples and query them,
first with the in-process triple store :class:`~forgeloop.agents.memory.GraphMemory`
and then with the Geometric Memory System (GMS) reached through
:class:`~forgeloop.agents.gms_backend.GMSMemory`. A triple is
``(subject, relation, object)``, so a fact is typed and can be checked by pattern
or scored by geometry rather than retrieved by similarity alone. The plain store
answers structural questions over facts held in memory; the GMS substrate adds a
continuous plausibility score and lossless numeric recall over a store trained
offline.

The pieces
----------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Type
     - Role
   * - :class:`~forgeloop.agents.memory.Triple`
     - A ``(subject, relation, object)`` fact with optional ``source``.
   * - :class:`~forgeloop.agents.memory.GraphMemory`
     - In-process triple store with breadth-first ``neighbors`` traversal.
   * - :class:`~forgeloop.agents.gms_backend.GMSMemory`
     - Adapter exposing the GMS primitives over a trained store.
   * - ``score_triple`` / ``link_predict``
     - Geodesic plausibility of a triple, and ranked tails for a gap.
   * - ``lookup_enm``
     - Exact numeric recall from the store's numeric register.
   * - ``query_triples``
     - Structural pattern match ``(head, relation, tail)``.

Record facts as triples and traverse them
------------------------------------------

:class:`~forgeloop.agents.memory.GraphMemory` holds :class:`~forgeloop.agents.memory.Triple`
records and walks them with :meth:`~forgeloop.agents.memory.GraphMemory.neighbors`,
which expands from an entity by breadth-first search in both edge directions and
visits each connected entity once. A multi-hop call recovers a chain that is
stored nowhere as a single fact.

.. code-block:: python

   from forgeloop.agents.memory import GraphMemory, Triple

   g = GraphMemory()
   g.add_triple(Triple('overdraft_policy', 'governs', 'overdraft_fee'))
   g.add_triple(Triple('overdraft_fee', 'reversible_via', 'fee_reversal_policy'))
   g.add_triple(Triple('fee_reversal_policy', 'requires', 'manager_approval'))

   for t in g.neighbors('overdraft_policy', hops=2):
       print(f'  {t.subject} --[{t.relation}]--> {t.object}')

The store also answers a substring query through
:meth:`~forgeloop.agents.memory.GraphMemory.query`, which returns the matching
triples wrapped as :class:`~forgeloop.agents.memory.MemoryItem` values of kind
:class:`~forgeloop.agents.memory.MemoryKind`. That form of retrieval matches the
literal surface of a field, so it is exact but does not generalize across
phrasing.

The geometric substrate
------------------------

A trained GMS store places each entity on a sphere and treats each relation as an
operator on it, so a triple maps to a configuration whose plausibility is a
geodesic distance. The store is built once from a document with the GMS library
and loaded thereafter. The build below requires ``knowlytix`` and ``torch`` and
is not executed on this page; it is the exact sequence the notebook runs.

.. code-block:: python

   import torch
   from knowlytix.knowledge.geode import build_rag_store, make_default_trainer
   from knowlytix.core.config import GeometryConfig, TrainConfig, CapLossConfig
   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   cfg = DocGMSConfig(store_path='data/gms_ch2_store', ingest_mode='regex', loss_mode='cap',
                      geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32),
                      cap=CapLossConfig(n_boundary=4),
                      train=TrainConfig(epochs=400, batch_size=64, neg_samples=8,
                                        lr=5e-3, lr_riemannian=2e-3))
   store = GMSExpertStore(cfg, device=device)
   store.load()                       # built once by build_rag_store, loaded thereafter

Query the triples by pattern
----------------------------

:meth:`~forgeloop.agents.gms_backend.GMSMemory.query_triples` matches on any of
``head``, ``relation`` or ``tail`` and returns raw ``(head, relation, tail)``
tuples. Composing two pattern matches walks a relation chain, so an evidence flag
resolves to the escalation it triggers even though the two are separated by an
intermediate node.

.. code-block:: python

   from forgeloop.agents.gms_backend import GMSMemory
   mem = GMSMemory(store)

   def heads_with(relation, tail):
       return [h for h, r, t in mem.query_triples(relation=relation, tail=tail)]
   def tails_of(head, relation):
       return [t for h, r, t in mem.query_triples(head=head, relation=relation)]

   evidence = 'unfair_or_abusive_fee'
   for flag in heads_with('has_applies_when', evidence):       # hop 1: evidence -> flag
       target = tails_of(flag, 'has_escalates_to')             # hop 2: flag -> escalation
       print(f'{evidence} -> {flag} -> escalate to {target}')

Score a triple and recall exact values
--------------------------------------

:meth:`~forgeloop.agents.gms_backend.GMSMemory.score_triple` returns the geodesic
distance of a triple, lower for a triple that fits the store. A committed fact
scores below a wrong one, and a legitimate workflow transition scores below a
step that skips ahead, so a single distance separates the two.

.. code-block:: python

   print(mem.score_triple('overdraft', 'has_fee_amount', '35.0'))    # committed value
   print(mem.score_triple('overdraft', 'has_fee_amount', '45.0'))    # wrong (wire fee)
   print(mem.score_triple('classify', 'has_enables', 'extract'))     # legal next step
   print(mem.score_triple('classify', 'has_enables', 'draft_response'))  # skips ahead

Exact values such as fees are not scored. They are held in a numeric register and
recalled byte-exactly through
:meth:`~forgeloop.agents.gms_backend.GMSMemory.lookup_enm`, which returns ``None``
when the value is absent. :meth:`~forgeloop.agents.gms_backend.GMSMemory.link_predict`
ranks the plausible tails for a gap ``(head, relation, ?)``, and the store's
``fuzzy_match_entity`` resolves a surface phrase to a canonical node.

.. code-block:: python

   print(mem.lookup_enm('fee_schedule', 'overdraft/per_occurrence'))  # 35.0, lossless
   print(mem.link_predict('overdraft', 'has_fee_amount', top_k=4))    # committed value first
   print(store.fuzzy_match_entity('overdraft fee'))                   # -> 'overdraft'

Why the geometry matters
-------------------------

The plausibility distance is continuous, so a threshold can be calibrated to a
known error rate; it is deterministic, so an auditor reproduces the verdict; and
it is not produced by a language model, so it does not share the failure modes of
the text it judges. That property is what lets the same store back a plausibility
gate on a tool call (:doc:`06_safe_tool_execution`), a whole-plan coherence check
(:doc:`08_planning`) and a contradiction check on a memory write
(:doc:`09_memory`). Exact numbers bypass the geometry entirely through the numeric
register, so a fee is recalled rather than reconstructed.

See also
--------

- :doc:`/3-api-reference/modules/memory/index` — :class:`~forgeloop.agents.memory.GraphMemory` and :class:`~forgeloop.agents.memory.Triple`.
- :doc:`/3-api-reference/modules/gms_backend/index` — :class:`~forgeloop.agents.gms_backend.GMSMemory` and its primitives.
- :doc:`09_memory` — the four memory tiers and where the GMS substrate sits among them.
- :doc:`/4-notebook-examples/agents/index` — the store built, queried and scored end to end.
