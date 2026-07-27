Geometric Memory: the Store Primitives
=======================================

The alternative to similarity search over chunks is a trained triple register,
the GMS store. It is a small geometric model that has learned a knowledge graph,
exposes a set of typed operations over it and keeps every authoritative number in
a separate exact-recall memory with a cryptographic integrity check. This page
exercises the six operations at author altitude, over the Northwind Industries
FY2025 store, and describes what each output means rather than the geometry
beneath it. The store class is
:class:`~knowlytix.knowledge.store.GMSExpertStore`; its geometry and training are
fixed by :class:`~knowlytix.knowledge.config.DocGMSConfig` and
:class:`~knowlytix.core.config.GeometryConfig`.

The six operations
------------------

Two operations are exact lookups over the asserted graph; four read the trained
manifold.

.. list-table::
   :header-rows: 1

   * - Operation
     - What it returns
   * - :meth:`~knowlytix.knowledge.store.GMSExpertStore.lookup_enm`
     - A number byte-exact from Exact Numerical Memory, keyed by category and id.
   * - :meth:`~knowlytix.knowledge.store.GMSExpertStore.query_triples`
     - Asserted edges matching a pattern, with a wildcard for any unset slot.
   * - :meth:`~knowlytix.knowledge.store.GMSExpertStore.score_triple`
     - A geodesic distance for one triple; lower is more plausible.
   * - :meth:`~knowlytix.knowledge.store.GMSExpertStore.link_predict`
     - Type-valid tails for an open slot, ranked by distance.
   * - :meth:`~knowlytix.knowledge.store.GMSExpertStore.check_holonomy`
     - The holonomy defect of a composed path against a direct edge; zero is consistent.
   * - :meth:`~knowlytix.knowledge.store.GMSExpertStore.tension_energy`
     - The relationship between two entities on a 0..2 scale; zero agrees, two contradicts.

Load the trained store
----------------------

A store is reopened with the same geometry and loss configuration it was trained
under: a 64/64/32/32 geometry with cap-admissibility loss.
:func:`forgeloop.rag.load_store` builds that configuration through
:func:`forgeloop.rag.store_config` and calls
:meth:`~knowlytix.knowledge.store.GMSExpertStore.load`, which rebuilds the model,
the entity/relation adapter, the document graph and Exact Numerical Memory from
disk.

.. code-block:: python

   from forgeloop.rag import load_store

   store = load_store()
   print("relations:", sorted(store.adapter.relation_to_idx))

Exact recall against a parse
----------------------------

Every authoritative number lives in Exact Numerical Memory under a
``(category, id)`` key with SHA-256 integrity;
:meth:`~knowlytix.knowledge.store.GMSExpertStore.lookup_enm` returns the value
byte-exact. Parsing the same figure out of retrieved text is fragile: the same
digits recur across sentences, and word order alone can select the wrong number.

.. code-block:: python

   import re

   # Authoritative path: byte-exact read from Exact Numerical Memory.
   revenue_fy2025 = store.lookup_enm("income_statement", "Revenue/FY2025")
   print("ENM income_statement/Revenue/FY2025 =", revenue_fy2025)   # 355.0

   # Chunk-and-pray path: parse a figure out of a retrieved sentence.
   retrieved = ("Total revenue grew to $355M in FY2025, up from $320M, while the "
                "Cloud Platform segment alone contributed $120M.")
   first = float(re.search(r"\$(\d+)M", retrieved).group(1))   # 355 by luck of order

   reordered = ("The Cloud Platform segment contributed $120M as total revenue "
                "grew to $355M in FY2025.")
   wrong = float(re.search(r"\$(\d+)M", reordered).group(1))   # now 120, the segment
   assert wrong != revenue_fy2025

The regex returns 355 on the first sentence and 120 on the reordered one; the ENM
read is order-invariant and exact.

A true fact against a false one
-------------------------------

:meth:`~knowlytix.knowledge.store.GMSExpertStore.score_triple` returns a geodesic
distance on the trained manifold where lower is more plausible. A fact the store
was trained on sits closer than a fabricated tail. Tails are canonicalized
numeric strings, matching the triples the store was built from.

.. code-block:: python

   true_d = store.score_triple("cloud platform", "has_headcount", "340.0")
   false_d = store.score_triple("cloud platform", "has_headcount", "520.0")  # retail's
   assert true_d < false_d
   rho = store.cap_radius("has_headcount")   # the accept radius for this relation

The asserted headcount (340.0) scores strictly closer than the swapped tail
(520.0). The distance is relative geometry, not a calibrated probability;
:meth:`~knowlytix.knowledge.store.GMSExpertStore.cap_radius` reports the accept
radius for a relation, and turning a gap into an accept or abstain decision is the
subject of :doc:`14_calibration`.

Asserted edges and ranked tails
-------------------------------

:meth:`~knowlytix.knowledge.store.GMSExpertStore.query_triples` is exact
pattern-matching over the document graph; leaving a slot ``None`` wildcards it.
This is the path the retriever prefers, because an asserted edge carries
provenance.
:meth:`~knowlytix.knowledge.store.GMSExpertStore.link_predict` ranks type-valid
tails when no edge is asserted, and returns a prediction, not an assertion.

.. code-block:: python

   cloud_edges = store.query_triples(head="cloud platform")
   for h, r, t in cloud_edges:
       print(f"{h:>16} {r:>14} {t}")

   ranked = store.link_predict("cloud platform", "has_division", top_k=3)
   for tail, dist in ranked:
       print(f"{tail:>14}  d={dist:.4f}")

Where both are available the retriever takes the asserted
:meth:`~knowlytix.knowledge.store.GMSExpertStore.query_triples` edge over the
:meth:`~knowlytix.knowledge.store.GMSExpertStore.link_predict` guess
(:doc:`08_triple_mediated_retrieval`).

Path consistency and contradiction
-----------------------------------

:meth:`~knowlytix.knowledge.store.GMSExpertStore.check_holonomy` measures how far
composing a relation path drifts from a direct edge, where zero is consistent; it
is the backbone of multi-hop retrieval.
:meth:`~knowlytix.knowledge.store.GMSExpertStore.tension_energy` reads the
relationship between two entities on a 0..2 scale, where zero is agreement and two
is contradiction, so a functional-fact clash is a geometric signal rather than a
string compare.

.. code-block:: python

   defect = store.check_holonomy(["has_division", "has_region"], "has_region")
   tau = store.config.verify.tau_path   # a defect at or below tau_path is consistent

   te_heads = store.tension_energy("dana cole", "sam reyes")  # two distinct heads
   te_self = store.tension_energy("dana cole", "dana cole")   # identical -> agree
   assert te_self <= te_heads

Mechanism and scope
-------------------

The store does only what it was trained on. The manifold distances are relative
geometry, not calibrated probabilities, so an accept or abstain decision requires
a calibrated operating point. The primitives operate only over the entities and
relations the ingest populated, so a prose-only section returns no triples, and
:meth:`~knowlytix.knowledge.store.GMSExpertStore.lookup_enm` is exact only for
values that reached Exact Numerical Memory during ingest.

See also
--------

- :doc:`01_why_chunk_and_pray_fails` — the failure these primitives answer.
- :doc:`03_provenance` — resolving a retrieved triple to its source span.
- :doc:`/3-api-reference/modules/knowlytix/store/index` — :class:`~knowlytix.knowledge.store.GMSExpertStore` and its methods.
- :doc:`/3-api-reference/modules/knowlytix/config/index` — :class:`~knowlytix.knowledge.config.DocGMSConfig`, :class:`~knowlytix.core.config.GeometryConfig` and :class:`~knowlytix.core.config.TrainConfig`.
- :doc:`/4-notebook-examples/rag/index` — the six primitives exercised in a notebook.
