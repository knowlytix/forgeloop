GEODE: Self-Correcting Extraction
=================================

Building a store by trusting the extractor sends whatever triples the ingest
produced straight into the graph. GEODE screens them first. Before a triple is
indexed, a closed loop lets the geometry judge the extraction: a composition
critic catches relational triples that contradict the rest of the graph, and
external anchors, a declared sum or a duplicate value, catch numeric errors the
geometry is structurally blind to. This page corrupts the Northwind report on
purpose and shows the loop flag each break with provenance, then states the one
class of error neither mechanism can catch. All symbols come from
:mod:`knowlytix.knowledge.geode`.

Two forms of redundancy
-----------------------

The report provides two cross-checks. The segment table declares a ``Total`` row
whose revenue equals the sum of the four segment revenues, ``120 + 80 + 95 + 60
== 355``, which the sum anchor verifies by reading the exact values back through
the ENM register. The ``segment -> division -> region`` hierarchy is a relational
chain, which the composition critic verifies.

.. code-block:: python

   clean_triples = [
       ("cloud platform", "has_revenue", "120.0"),
       ("devices", "has_revenue", "80.0"),
       ("logistics", "has_revenue", "95.0"),
       ("retail", "has_revenue", "60.0"),
       ("total", "has_revenue", "355.0"),
       ("cloud platform", "has_division", "technology"),
       ("devices", "has_division", "technology"),
       ("logistics", "has_division", "operations"),
       ("retail", "has_division", "operations"),
       ("technology", "has_region", "north america"),
       ("operations", "has_region", "europe"),
   ]

Corrupt a segment figure: the sum anchor
----------------------------------------

Change Devices revenue from ``80.0`` to ``88.0``. The four segments now sum to
``363`` while the ``Total`` row still declares ``355.0``. No single triple is
internally wrong, so the geometry cannot see the break. The sum anchor can.
:func:`~knowlytix.knowledge.geode.anchor.enm_from_triples` builds the
integrity-checked exact register, and
:class:`~knowlytix.knowledge.geode.anchor.AnchorChecker` reads the values back
through it. ``auto_sum_constraints`` derives the constraint from the ``Total``
row the way a financial table already declares it, and a
:class:`~knowlytix.knowledge.geode.provenance.ProvenanceLedger` maps each break
to the cell it came from.

.. code-block:: python

   from knowlytix.knowledge.geode import (
       enm_from_triples, AnchorChecker, ProvenanceLedger,
   )
   from forgeloop.rag import CORPUS   # path to data/annual_report.md

   # Inject the sum break: Devices 80.0 -> 88.0.
   corrupted = [
       (h, r, "88.0") if (h, r) == ("devices", "has_revenue") else (h, r, t)
       for (h, r, t) in clean_triples
   ]

   enm = enm_from_triples(corrupted)
   ledger = ProvenanceLedger.from_text(open(CORPUS).read(), CORPUS)
   checker = AnchorChecker.from_enm(enm, ledger)

   for c in checker.auto_sum_constraints():
       print("constraint:", c.total, "= sum", [p[0] for p in c.parts])

   for v in checker.check_all(corrupted):
       print(v.kind, "|", v.message)
       for loc in v.locations:
           print("   provenance:", loc.location(), "raw:", repr(loc.raw))

``auto_sum_constraints`` derives ``('total','has_revenue') = sum of [cloud
platform, devices, logistics, retail]``. ``check_all`` returns a ``sum``
violation: declared ``355`` against ``sum(parts)=363``, off by ``8``, with a
provenance location pointing back into the segment table. The anchor detects the
break and names the candidate parts. It does not auto-rewrite a number, because
which part is wrong is a review decision; a sum constraint localizes to the set
of parts, not to the guilty one.

The full loop
-------------

:class:`~knowlytix.knowledge.geode.loop.GeodeLoop` runs the whole sequence over
the real document: regex ingest, train a small GMS, run the composition critic
to repair relational contradictions, resolve duplicate tails, then run the
external anchors for the numeric errors the geometry is blind to. The trainer is
injected with :func:`~knowlytix.knowledge.geode.loop.make_default_trainer`. This
call trains a GMS, so it requires a GPU.

.. code-block:: python

   import tempfile
   from knowlytix.knowledge.geode import GeodeLoop, make_default_trainer
   from forgeloop.rag import CORPUS, DEVICE

   source = open(CORPUS).read()
   corrupted_doc = source.replace(
       "| Devices | Technology | 80.0 | 210 |",
       "| Devices | Technology | 88.0 | 210 |",
   )
   assert corrupted_doc != source

   tmp = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
   tmp.write(corrupted_doc); tmp.close()

   loop = GeodeLoop(make_default_trainer(DEVICE, epochs=40))
   result = loop.run(tmp.name)

   print("converged:", result.converged, "| iterations:", result.iterations)
   for v in result.anchor_violations:
       print("anchor:", v["kind"], "|", v["message"], "| residual:", round(v["residual"], 3))

The composition critic finds no relational contradiction, since the
segment-to-division-to-region chain is intact, so the loop converges and runs the
anchors. ``result.anchor_violations`` contains a ``sum`` violation with residual
``8.0``, localized to the segment table. A relational error is different: the
graph carries a second source of truth in the composition path, so the critic
measures the geodesic gap between the composed path and the asserted tail and
rewrites the triple to the geometry's prediction. A lone corrupted number has no
redundant cross-check, so the sum anchor detects and localizes the break but
leaves which part is wrong to review.

The limit: a lone value
-----------------------

The boundary is a value with no redundancy. ``total_assets has_amount 540.0``
appears once, with no ``Total = sum of parts`` row over the balance-sheet line
items and no second statement of the figure. Corrupt it to ``999.0`` and neither
mechanism fires: the composition critic needs a redundant relational chain, and
the sum anchor needs declared parts.

.. code-block:: python

   lone = [("total assets", "has_amount", "999.0")]   # truth is 540.0

   checker_lone = AnchorChecker.from_enm(enm_from_triples(lone))
   print("derived sum constraints:", checker_lone.auto_sum_constraints())  # []
   print("anchor violations:", checker_lone.check_all(lone))               # []

Both return empty and the corrupted ``999.0`` passes silently. The remedy is
upstream, by declaring a constraint, adding a duplicate statement or carrying the
figure as an ENM-checked exact value sourced to its one cell. GEODE makes
confident-wrong relational and aggregate facts difficult to introduce; an
isolated corrupted figure remains the extractor's responsibility.

The duplicate anchor
--------------------

Asserting the same ``(entity, relation)`` twice with conflicting exact values
gives the anchor a cross-check it can localize precisely, unlike the sum case.
``check_duplicates`` fires on the conflict.

.. code-block:: python

   dup = clean_triples + [("total", "has_revenue", "350.0")]  # conflicts with 355.0

   checker_dup = AnchorChecker.from_enm(enm_from_triples(dup))
   for v in checker_dup.check_duplicates(dup):
       print(v.kind, "|", v.message, "| residual:", v.residual)

See also
--------

- :doc:`04_document_to_graph` — the build that runs this loop.
- :doc:`03_provenance` — the ledger the anchors cite through.
- :doc:`/3-api-reference/modules/knowlytix/geode/index` — signatures for
  ``AnchorChecker``, ``GeodeLoop``, ``enm_from_triples`` and ``ProvenanceLedger``.
- :doc:`/4-notebook-examples/rag/index` — the corruption run end to end.
