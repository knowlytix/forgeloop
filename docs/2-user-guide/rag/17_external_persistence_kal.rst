Chapter 17 — Persisting to an External Store (KAL / Postgres)
=============================================================

A trained store is a directory on local disk: a model checkpoint, an adapter,
an ENM table and the source markdown. To serve a fleet the graph — triples plus
provenance plus GEODE verification — must live in a real database. KAL, the
Knowledge Adapter Layer, is the backend-agnostic knowledge-graph store, and
``knowlytix.knowledge.rag.kal_sink`` is the one-way bridge from a built RAG
store into it. This page converts a store's triples to KAL records, persists
them through the offline mock adapter, reads them back, and shows the gated
Postgres path. The trained geometry is not sent; only the graph is persisted.

What the sink reads
-------------------

:func:`~knowlytix.knowledge.rag.kal_sink.store_to_kal_triples` touches three
attributes of a built store.

.. list-table::
   :header-rows: 1

   * - Attribute
     - Role
   * - ``store.triples``
     - iterable of ``(head, relation, tail)`` tuples
   * - ``store.markdown``
     - source document, for provenance recovery
   * - ``store.store_path``
     - default ``source`` for the persisted triples

A store loaded with :func:`forgeloop.rag.load_store` supplies all three, so the
conversion code is identical whether the source is the real store or an offline
fixture with the same attribute surface.

Convert the store's triples
---------------------------

Each ``(head, relation, tail)`` maps to a KAL triple. A numeric tail becomes a
typed literal carrying the exact ENM value; an entity tail becomes a node. KAL's
model validator enforces exactly one of the two, so a malformed triple cannot be
constructed. Provenance is recovered from ``store.markdown``, and the
``in_section`` relation is dropped by the default ``exclude_relations``.

.. code-block:: python

   from knowlytix.knowledge.rag.kal_sink import store_to_kal_triples

   kal_triples = store_to_kal_triples(store, extractor="geode")

   # A numeric (literal) triple vs an entity (node) triple.
   rev = next(t for t in kal_triples
              if t.subject.normalized_name == "cloud platform"
              and t.predicate == "has_revenue")
   div = next(t for t in kal_triples
              if t.subject.normalized_name == "cloud platform"
              and t.predicate == "has_division")

   assert rev.object is None                       # numeric tail
   assert rev.object_literal.value == "120.0"      # exact ENM value, byte-for-byte
   assert rev.object_literal.datatype == "number"
   assert div.object.name == "technology"          # entity tail -> node
   assert div.object_literal is None

Persist through the mock adapter and read back
----------------------------------------------

:class:`~knowlytix.kal.adapters.mock.MockKnowledgeAdapter` is KAL's in-memory
adapter, the offline persistence path with no Postgres and no network.
:func:`~knowlytix.knowledge.rag.kal_sink.persist_store_to_kal` is the async
bridge; a notebook already runs inside an event loop, so it is awaited directly.
The :func:`~knowlytix.knowledge.rag.kal_sink.persist_store_to_kal_sync` wrapper
is for plain scripts, where its ``asyncio.run`` has no running loop to nest
inside. Reading back through a :class:`~knowlytix.kal.types.KALQuery` returns the
graph with its structure intact.

.. code-block:: python

   from knowlytix.kal import KALQuery
   from knowlytix.kal.adapters.mock import MockKnowledgeAdapter
   from knowlytix.knowledge.rag.kal_sink import persist_store_to_kal

   # Seed the mock with what will be persisted, so reads return them.
   adapter = MockKnowledgeAdapter("geode-mock", fixed_triples=kal_triples)

   inserted = await persist_store_to_kal(adapter, store, tenant_id="northwind")
   result = await adapter.query_triples(KALQuery(), tenant_id="northwind")

   assert inserted == len(kal_triples)
   assert len(result.triples) == len(kal_triples)   # round trip, count intact

Stamp GEODE verification
------------------------

Passing ``confidence`` to
:func:`~knowlytix.knowledge.rag.kal_sink.store_to_kal_triples` attaches a
verification record to every triple, with status ``verified`` and verifier
``geode``. A downstream consumer then filters on confidence without re-running
GEODE. Without ``confidence`` the verification is ``None``.

.. code-block:: python

   verified = store_to_kal_triples(store, extractor="geode", confidence=0.97)
   v_adapter = MockKnowledgeAdapter("geode-verified", fixed_triples=verified)
   v_result = await v_adapter.query_triples(KALQuery())

   vm = v_result.triples[0].verification
   assert vm.confidence == 0.97
   assert vm.verification_status == "verified"
   assert vm.verifier == "geode"

   assert kal_triples[0].verification is None   # unstamped default

The Postgres path
-----------------

:func:`~knowlytix.knowledge.rag.kal_sink.kal_postgres_adapter` builds a
pgvector-backed adapter. It requires a live database with KAL's migrations
applied, so the following is gated on a connection-string environment variable
and is a no-op in offline CI. The conversion and persist calls are identical to
the mock path; only the adapter differs.

.. code-block:: python

   import os
   from knowlytix.knowledge.rag.kal_sink import (
       kal_postgres_adapter, persist_store_to_kal)

   PG = os.environ.get("KAL_PG_DSN")   # host=...;db=...;user=...;pw=...
   if PG:
       parts = dict(kv.split("=", 1) for kv in PG.split(";"))
       pg_adapter = kal_postgres_adapter(
           host=parts["host"], database=parts["db"],
           username=parts["user"], password=parts["pw"])
       n = await persist_store_to_kal(pg_adapter, store,
                                      tenant_id="northwind", confidence=0.97)
   else:
       print("KAL_PG_DSN not set -- skipping live Postgres persist")

Mechanism
---------

The bridge is one-way and lossy by design. The trained
:class:`~knowlytix.knowledge.store.GMSExpertStore` model cannot be reconstructed
from KAL, because only the graph crosses the sink. The mock adapter is not a
database and does not survive a process restart; it exercises the same adapter
protocol the Postgres backend implements, so the two paths are byte-identical
apart from the adapter. A stamped confidence is the value GEODE produced at
extraction time, not a fresh check.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/kal/index` — the adapter contract and KAL
  types.
- :doc:`03_provenance` and :doc:`12_self_verification` — the provenance and
  verification records that ride along with each triple.
- :doc:`/4-notebook-examples/rag/index` — the round trip run end to end.
