Chapter 3 — Provenance: Span to Triple
======================================

A retrieved fact is only as trustworthy as the source it can point to. The
chunk-and-pray recipe discards offsets at ingestion, so an answer can cite a
chunk but never the exact cell that carried a number. Geometric memory keeps the
reverse map: every asserted triple can be re-aligned to the precise span it came
from, given as ``file:line:char`` together with the raw substring, and a triple
with no structural anchor is reported as unaligned rather than guessed. This page
builds that map for the Northwind report with
:class:`~knowlytix.knowledge.geode.provenance.ProvenanceLedger` and checks a
resolved span with :func:`~knowlytix.knowledge.geode.provenance.is_consistent`.
Alignment is pure text-to-span work over the markdown source, so it needs neither
the trained store nor a language model.

Build the ledger
----------------

The ledger reads the raw markdown once. The report path is exported by the
:mod:`forgeloop.rag` helper as ``CORPUS``.

.. code-block:: python

   from forgeloop.rag import CORPUS
   from knowlytix.knowledge.geode.provenance import (
       ProvenanceLedger, Provenance, is_consistent,
   )

   ledger = ProvenanceLedger(CORPUS)

Resolve a table-cell triple
---------------------------

:meth:`~knowlytix.knowledge.geode.provenance.ProvenanceLedger.resolve` re-aligns a
triple post-hoc and returns a
:class:`~knowlytix.knowledge.geode.provenance.Provenance` carrying the 1-based
line, the absolute character span, the raw cell text and the alignment
``method``.

.. code-block:: python

   p = ledger.resolve("cloud platform", "has_revenue", "120.0")
   print("location:", p.location())   # data/annual_report.md:15:553-558
   print("method:  ", p.method)       # table_cell
   print("raw:     ", repr(p.raw))    # '120.0'

The ``method`` is ``table_cell``: the figure is anchored to a precise span, not
parsed loose from prose. The raw substring is the cell text, byte-exact 120.0,
the same value the store's Exact Numerical Memory returns, because both read the
same cell.

A prose triple is unaligned
---------------------------

The MD&A section says the Cloud Platform segment continued to lead the company's
topline. That sentence carries no table cell, schema bullet or section header for
a ``(cloud platform, leads, topline)`` triple, so structural alignment finds no
anchor and resolves it as ``unaligned`` with sentinel offsets.

.. code-block:: python

   prose = ledger.resolve("cloud platform", "leads", "topline")
   print("location:", prose.location())        # data/annual_report.md:-1:-1--1
   print("method:  ", prose.method)            # unaligned
   print("aligned? ", prose.method != "unaligned")   # False

The ``-1`` line and offsets are the signal that prose triples need spans captured
at extraction time. The ledger re-aligns regex-extracted structured triples,
tables, schema bullets and headers; it does not invent a span for an
LLM-extracted prose claim.

A coarse span for a prose section
---------------------------------

A prose section can still receive a span, not for a fine-grained fact but at
section granularity, through an ``in_section`` triple keyed on the section slug.
The ledger resolves it to the section-header line, giving the answer layer a
bounded region to cite where no table fact exists.

.. code-block:: python

   sec = ledger.resolve("md&a note", "in_section",
                        "management_discussion_and_analysis")
   print("location:", sec.location())   # data/annual_report.md:60:1527-1567
   print("method:  ", sec.method)       # section_header
   print("raw:     ", repr(sec.raw))    # '## 6. Management Discussion and Analysis'

This anchors a prose region to its header span. It does not make the prose
checkable against Exact Numerical Memory; fine numeric facts still come from
tables.

Verify that a span supports its triple
--------------------------------------

Resolving to a span is not sufficient; the span must support the triple.
:func:`~knowlytix.knowledge.geode.provenance.is_consistent` re-reads the raw
substring and checks it against the tail, numeric-aware for table cells and
slug-aware for headers and schema bullets.

.. code-block:: python

   good = ledger.resolve("total", "has_revenue", "355.0")
   print(good.location(), repr(good.raw), is_consistent(good))
   # data/annual_report.md:19:698-703 '355.0' True

   print(is_consistent(prose))   # False -- no span to read

The check parses 355.0 out of the raw cell and matches it to the tail within
``1e-6``. The unaligned prose triple has no span to read, so it is correctly
``False``: the ledger refuses to claim support it cannot point to.

Audit a batch of triples
------------------------

:meth:`~knowlytix.knowledge.geode.provenance.ProvenanceLedger.build` resolves a
list of triples at once, returning a provenance per triple that can be checked
with :func:`~knowlytix.knowledge.geode.provenance.is_consistent`.

.. code-block:: python

   triples = [
       ("cloud platform", "has_revenue", "120.0"),
       ("devices", "has_revenue", "80.0"),
       ("logistics", "has_revenue", "95.0"),
       ("retail", "has_revenue", "60.0"),
       ("total", "has_revenue", "355.0"),
   ]
   provs = ledger.build(triples)
   for key, pr in provs.items():
       flag = "OK " if is_consistent(pr) else "BAD"
       print(flag, key, "->", pr.location(), repr(pr.raw))

Each segment-revenue triple aligns to a table cell whose raw value equals the
tail, and the Total row aligns too, so a sum anchor has a span to cite if it ever
breaks.

Mechanism and scope
-------------------

The ledger re-aligns structured triples by re-reading the markdown; it is not a
general fact verifier and cannot anchor an LLM-extracted prose claim. Keys are
entity-relation pairs, so documents with recurring row labels can collide, and a
consistent span confirms that the span supports the tail, not that extraction
chose the right triple. This resolved span is the citation later carried onto a
retrieved answer (:doc:`10_answering_through_the_gms`).

See also
--------

- :doc:`01_why_chunk_and_pray_fails` — the missing-provenance failure this map repairs.
- :doc:`02_geometric_memory` — Exact Numerical Memory returns the same cell value.
- :doc:`/3-api-reference/modules/knowlytix/geode/index` — :class:`~knowlytix.knowledge.geode.provenance.ProvenanceLedger`, :class:`~knowlytix.knowledge.geode.provenance.Provenance` and :func:`~knowlytix.knowledge.geode.provenance.is_consistent`.
- :doc:`/4-notebook-examples/rag/index` — the span-to-triple map built end to end.
