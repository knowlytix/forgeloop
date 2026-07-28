Chapter 2 — Knowledge Graphs and Exact Memory
=============================================

This page shows how the ground truth the testing method reads is drawn from a
knowledge graph, and why the geometric memory store replaces binary membership
with a graded distance. It starts from a graph small enough to write by hand,
then loads the trained ``GMSExpertStore`` the later pages test against. The store
is the licensed knowlytix backend; the testing package consumes it as an oracle.

The baseline graph
------------------

A knowledge graph is a set of triples, each an ordered statement ``(head,
relation, tail)``. A pattern query returns the triples matching a partial
pattern. Membership is binary: a triple is either asserted or it is not.

.. code-block:: python

   triples = [
       ('overdraft', 'has_fee_amount',  '35'),
       ('overdraft', 'governed_by',     'reg_e'),
       ('dispute',   'has_window_days', '60'),
   ]

   def query(h=None, r=None, t=None):
       return [tr for tr in triples
               if (h is None or tr[0] == h) and (r is None or tr[1] == r)
               and (t is None or tr[2] == t)]

   print(query(h='overdraft'))
   # membership is binary: yes/no, not "how wrong"
   print(('overdraft', 'has_fee_amount', '35') in triples)   # True
   print(('overdraft', 'has_fee_amount', '50') in triples)   # False

For retrieval this suffices. For grounding a system under test it has a gap: it
reports that ``50`` is not the fee but not that ``50`` is closer to the truth
than ``5000``. A binary predicate cannot grade how wrong an alternative tail is,
and grading is what a failure-attribution study needs.

Loading the trained store
--------------------------

The geometric memory store embeds the graph's entities and relations on a
learned manifold. It keeps the asserted triples and answers ``query_triples``
exactly, and it adds ``score_triple``, which returns a distance rather than a
Boolean. Loading the store is a ``GMSExpertStore`` over a ``DocGMSConfig``
pointing at a built store directory.

.. code-block:: python

   import torch
   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

   store = GMSExpertStore(
       DocGMSConfig(store_path=str(STORE), ingest_mode="regex"),
       device=torch.device("cpu"),
   )
   assert store.load(), "store failed to load"
   print(len(store.adapter.relation_to_idx), "relations,",
         len(store.adapter.entity_to_idx), "entities")

``STORE`` is the path to a store built by GEODE (:doc:`ch04_geode`). The
banking-policy store the notebooks use ships with the companion volume as
``gms_policy_store_geode``.

Membership becomes a graded distance
------------------------------------

The asserted edges are still read exactly. The committed tail scores a smaller
distance than an uncommitted one, so an alternative can be ranked by how far it
sits from the truth.

.. code-block:: python

   print(store.query_triples(head='overdraft')[:3])       # asserted edges, exact

   d_true  = store.score_triple('overdraft', 'has_fee_amount', '35.0')
   d_false = store.score_triple('overdraft', 'has_fee_amount', '50.0')
   print(f'35.0 -> {d_true:.3f}  (committed)')
   print(f'50.0 -> {d_false:.3f}  (not committed)')
   assert d_true < d_false                                 # graded, not binary

The graded distance is the raw material the primitives in :doc:`ch03_gms_primitives`
turn into calibrated decisions, and the ground truth the base taxonomy in
:doc:`ch05_base_taxonomy` reads for each question category.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/store/index` — ``GMSExpertStore``,
  ``DocGMSConfig`` and the store query surface.
- :doc:`/3-api-reference/modules/gms_backend/index` — the forgeloop adapters
  that expose the store to agents and gates.
- :doc:`/4-notebook-examples/testing/index` — the baseline graph and the store
  loaded side by side.
