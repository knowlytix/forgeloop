Chapter 14 — Calibrate, Don't Guess
===================================

The embedding binder turns on a single similarity cut: above it a paraphrase is
accepted as a graph entity, below it the term is refused and the query abstains
(:doc:`09_binding`, :doc:`13_abstention_and_coverage`). This page shows how to
set that cut from a labeled cohort with
:func:`knowlytix.knowledge.rag.eval.calibrate_bind_threshold` rather than by eye,
so the operating point is the output of a recorded sweep and re-calibration is a
re-run.

Build an embedding-mode binder
------------------------------

The binder is constructed over the store loaded by
:func:`forgeloop.rag.load_store`. In embedding mode fuzzy exact and substring
matches win first, and the embedding fallback fires only for terms fuzzy cannot
resolve. The notebook injects a deterministic four-axis encoder so the sweep
runs without the network MiniLM path; in production the real encoder is passed
in its place and the procedure is identical.

.. code-block:: python

   from forgeloop.rag import load_store
   from knowlytix.knowledge.rag import TripleBinder

   store = load_store()
   binder = TripleBinder(store, mode="embedding", encoder=fake_encode,
                         bind_threshold=0.5, bind_margin=0.05)
   print("starting bind_threshold:", binder.bind_threshold)

The four axes correspond to the four Northwind revenue segments (``cloud
platform``, ``devices``, ``logistics``, ``retail``); a paraphrase and its
segment share an axis, and unrelated terms land at the origin and bind to
nothing.

Label a cohort and sweep
------------------------

Positives are ``(term, expected_entity)`` paraphrases that should bind to a
segment; negatives are terms that should bind to nothing.
:func:`~knowlytix.knowledge.rag.eval.calibrate_bind_threshold` sweeps a grid,
picks the threshold that maximizes ``(true-positives + true-negatives) / total``
and writes it back onto the binder in place.

.. code-block:: python

   from knowlytix.knowledge.rag import calibrate_bind_threshold
   from knowlytix.knowledge.rag.query_triples import QueryTriple

   positives = [
       ("saas",     "cloud platform"),
       ("hardware", "devices"),
       ("shipping", "logistics"),
       ("stores",   "retail"),
   ]
   negatives = ["weather", "moon", "guitar"]

   best_th, acc = calibrate_bind_threshold(binder, positives, negatives)
   print(f"chosen bind_threshold = {best_th:.3f}")
   print(f"separation accuracy   = {acc:.3f}")
   print("binder now set to     =", binder.bind_threshold)

The returned threshold is the operating point: above it the binder accepts a
paraphrase as a segment match, below it the term is refused and feeds bind-check
abstention. A separation accuracy of ``1.0`` means the grid found a cut that
admits every paraphrase and refuses every unrelated term; the chapter's
self-check asserts ``acc == 1.0`` and that the binder was updated in place.

Confirm the calibrated behavior
--------------------------------

After the sweep, paraphrases resolve to their canonical segment while noise
resolves to ``None``.

.. code-block:: python

   def bind_head(term: str) -> str | None:
       return binder.bind(QueryTriple(term, "has_revenue", "?")).head

   for term in ["saas", "shipping", "stores"]:
       print(f"{term:10s} -> {bind_head(term)!r}")   # -> canonical segment
   for term in ["weather", "moon"]:
       print(f"{term:10s} -> {bind_head(term)!r}")   # -> None (refused)

Re-rank under a false-allow ceiling
-----------------------------------

The default sweep treats a false-allow (binding noise) the same as a
false-refuse. A bank-grade system pays more for a false-allow, so the grid can
be re-ranked to keep only thresholds whose false-allows fall under a hard
ceiling, then take the lowest such threshold to retain the most positives.

.. code-block:: python

   def calibrate_under_ceiling(binder, positives, negatives, *,
                               max_false_allow=0,
                               grid=tuple(i / 20 for i in range(1, 20))):
       feasible = []
       for th in grid:
           binder.bind_threshold = th
           false_allow = sum(binder.bind(QueryTriple(t, "_", "?")).head is not None
                             for t in negatives)
           if false_allow <= max_false_allow:
               tp = sum(binder.bind(QueryTriple(t, "_", "?")).head == exp
                        for t, exp in positives)
               feasible.append((th, tp))
       if not feasible:
           raise ValueError("no threshold meets the false-allow ceiling")
       best_th = max(feasible, key=lambda x: (x[1], -x[0]))[0]
       binder.bind_threshold = best_th
       return best_th

   strict_th = calibrate_under_ceiling(binder, positives, negatives,
                                       max_false_allow=0)

The operating point is a cohort plus a false-allow ceiling, persisted on the
binder, which is the form every gate in this project reads.

Limits
------

Calibration is only as good as the cohort. A threshold tuned on segment names
does not transfer to a relation band it never saw, so per-relation bands need
per-relation labels. A deterministic encoder makes the axes orthogonal by
construction; the real MiniLM space is messier, so a softer separation and a
non-trivial ``bind_margin`` are expected. Calibration sets where to draw the
line; it cannot manufacture signal the encoder does not carry, so a paraphrase
closer to the wrong segment in embedding space is a limitation of the encoder,
not of the sweep.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — signatures for
  :func:`~knowlytix.knowledge.rag.eval.calibrate_bind_threshold`,
  :func:`~knowlytix.knowledge.rag.eval.calibrate_accept_threshold`,
  :class:`~knowlytix.knowledge.rag.binding.TripleBinder` and
  :class:`~knowlytix.knowledge.rag.query_triples.QueryTriple`.
- :doc:`/3-api-reference/modules/knowlytix/store/index` — store-side calibration with
  :func:`~knowlytix.core.graph.admissibility.calibrate_cap_margins_per_head` and
  :func:`~knowlytix.core.graph.admissibility.calibrate_tension_threshold`.
- :doc:`09_binding` and :doc:`13_abstention_and_coverage` — where the calibrated
  cut is read at query time.
- :doc:`/4-notebook-examples/rag/index` — the sweep run end to end.
