---
name: gms-knowlytix-first
description: >-
  GMS/knowlytix engineering discipline for this repo. INVOKE BEFORE writing or
  proposing any new functionality — RAG gates, calibration, retrieval, binding,
  verification, extraction, evaluation, persistence, or any GMS/GEODE behavior.
  Two rules: (1) use what already exists in GMS-knowlytix before building
  anything new; (2) if new functionality is genuinely needed, build it INTO
  GMS-knowlytix (the library), not into tutorial scripts. Triggers whenever the
  task touches knowlytix.knowledge.*, GMS, GEODE, gates, thresholds, calibration,
  the store, or "add a function / write a helper / build a new" in this project.
---

# GMS-knowlytix first

This tutorial (`gms-rag-tutorial`, the *Chunk and Pray* book) is a **thin
consumer** of the GMS-knowlytix library. The library is the product; the book
demonstrates it. Two non-negotiable rules govern every change.

## Rule 1 — Use the library before building anything new

Before writing a new function, gate, threshold, scorer, parser, or helper, find
out whether GMS-knowlytix already does it. Hand-rolled logic in `scripts/` or a
notebook that duplicates library capability is a defect, not a shortcut.

How to check, in order:

1. **Read the API surface.** The branch ships docs — read them first:
   - `GMS-knowlytix/knowlytix/knowledge/rag/API_REFERENCE.md`
   - `GMS-knowlytix/knowlytix/knowledge/rag/USER_GUIDE.md`
   - `GMS-knowlytix/GEODE_RAG_DESIGN.md` (the gate/section design, e.g. §14 gates)
2. **Grep the branch source** for the capability before assuming it's missing:
   ```bash
   grep -rn "def <thing>\|class <Thing>" \
     /home/user/jupyterlab/GMS-knowlytix/knowlytix/
   ```
   Key modules: `knowlytix/knowledge/rag/` (config, pipeline, retrieve, binding,
   relevance, verify, eval, coverage, extraction, kal_sink), `knowlytix/knowledge/geode/`
   (loop, rag, provenance, canonicalize), `knowlytix/harness/testing/` (judge,
   hallucination), `knowlytix/core/` (geometry, graph, encoders).
3. **It runs from the branch, not the wheel.** `knowlytix` resolves to the
   `feat/cap-many-to-many` working tree at `/home/user/jupyterlab/GMS-knowlytix`
   via `scripts/_bootstrap.py` (`use_branch_library()` / `KNOWLYTIX_SRC`). The
   installed 0.2.0 wheel is stale — never reason from it. Notebooks copy the
   three bootstrap lines into their first cell.

If the capability exists, **call it**. Tutorial scripts and notebooks are thin
callers (`build_store.py`, `calibrate_gates.py` are the model: they wire and
verify, they do not implement decision logic).

## Rule 2 — New functionality goes INTO the library (carefully)

If a capability is genuinely missing, build it in GMS-knowlytix, not in the
tutorial. The tutorial then imports it. This keeps the book honest (everything
shown is real product) and grows the library.

**GMS-knowlytix is shared by MANY applications. A change that breaks an existing
signature, default, or return shape can break other apps. Modify it with extreme
care:**
- **Additive, backward-compatible only.** Add new functions/optional params with
  safe defaults; never change or remove an existing public signature, default
  value, return type, or behavior that callers depend on.
- **Verify before committing.** Run the library's own test suite (and any
  affected consumer) after a change; a green suite is the bar. If a change can't
  be made additively, STOP and raise it with the user rather than refactoring.
- Add the code to the right `knowlytix/` module, with its docstring, and update
  `API_REFERENCE.md` / `USER_GUIDE.md` / relevant `__init__.py` exports.
- **Commit locally in the GMS-knowlytix branch. NEVER push** (this repo too).
- The tutorial-side change is then just the import + call.
- Promote ad hoc wrappers upstream rather than leaving them in `scripts/`
  (precedent: `build_calibrated_rag_store`, `calibrate_accept_threshold`,
  `benchmark_query_parse`, `complete_batch`, `apply_thresholds`).

## Inviolable companions to these rules

- **Calibration is mandatory.** Every decision gate (accept/abstain,
  grounded/fabricated, relevance, plausibility) reads a calibrated operating
  point fit from a labeled cohort under a stated false-accept ceiling and
  persisted with the store. A default or hand-picked midpoint (e.g. a bare
  `accept_threshold=0.0` presented as calibrated) is wrong. A gate with no valid
  cohort **recuses** (neutral) — it does not silently default or hard-fail.
- **Public book, no IP leak.** Explain GMS at purpose/altitude only. Never reveal
  projection matrices / `PᵀP=I`, the SVD/learned-reduction algorithm, Cayley-rotor
  construction, cap/parallel-transport derivations, or the diagnostics.
- **Defaults:** Qwen2.5-3B-Instruct as the LLM; build geometry on GPU
  (GB10 is CPU-parity on torch 2.10+cu130, `cuda` if available); US English,
  no Oxford comma, declarative, concept-led prose.

## Quick decision flow

```
About to write GMS/RAG/gate/calibration logic?
  ├─ Does GMS-knowlytix already do it?  (docs → grep → branch source)
  │     ├─ yes → import and call it. done.
  │     └─ no  → is it genuinely new product capability?
  │               ├─ yes → build it IN knowlytix, export, doc, commit-local.
  │               │        then call it from the tutorial.
  │               └─ no  → you missed it; search again.
  └─ Is every decision gate reading a calibrated, persisted operating point?
        └─ if not, calibrate or recuse. no bare defaults.
```
