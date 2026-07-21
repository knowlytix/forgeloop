#!/usr/bin/env python
"""Build a GEODE self-corrected graph-RAG store from the banking policy.

This replaces the thin, regex-only ``gms_policy_store`` (6 relations, all numeric
facts flattened onto a generic ``has_value`` edge) with a dense, *typed* graph
that triple-mediated retrieval can actually bind to. The source doc
``data/banking_policy.md`` uses wide tables (Fee Schedule, Regulatory Flags,
Reversal Authority), so the regex ingester emits entity-headed, typed triples
(``overdraft -> has_fee_amount -> 35``; ``udaap -> has_threshold -> 500``;
``manager -> has_max_reversal -> 500``) rather than ``(field, has_value, n)``.

Pipeline (``knowlytix.knowledge.geode.rag.build_rag_store``):

  1. regex-ingest the markdown into initial triples + exact numeric registers,
  2. run the GEODE self-correction loop (propose -> diagnose contradictions via
     geometry / numeric anchors -> repair), with a Qwen-3B-locked actor raising
     confidence (the geometry stays authoritative),
  3. train a production ``GMSExpertStore`` on the corrected graph and persist it.

The geometry (loop trainer + production GMS) is trained on **CPU** on purpose:
the GB10 silently miscomputes solve-heavy geometry per-process, so we keep the
delicate training off the GPU. The Qwen actor (ordinary transformer inference)
stays on the GPU.

Run::

    python scripts/build_geode_rag_store.py
    python scripts/build_geode_rag_store.py --no-actor   # geometry-only correction
    python scripts/build_geode_rag_store.py --store-path data/gms_policy_store_geode
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
# The complete customer-policy source: wide entity-headed tables for ALL six
# policies (overdraft, disputes, fee_reversal, account_closure, pii_handling,
# regulatory_escalation) + fee schedule + regulatory flags + reversal authority
# + the customer-vocabulary alias table. (The older data/banking_policy.md held
# only fees + the agent-workflow spec, so search_policy abstained on dispute /
# closure / escalation complaints -- the facts simply were not in the store.)
_DEFAULT_DOC = _REPO_ROOT / "data" / "banking_policy_full.md"
_DEFAULT_STORE = _REPO_ROOT / "data" / "gms_policy_store_geode"

# Relations the regex ingester emits that are NOT policy facts: ``is_functional``
# is a schema declaration (single-valued constraint), and ``fails / passes /
# is_weak / is_strong`` are an opaque encoding of a boolean table column. They
# carry no answerable signal but pollute the queryable vocabulary, so the small
# query-triple extractor binds questions onto them (e.g. "when does Reg E apply"
# -> reg_e is_functional, instead of reg_e has_applies_when). Drop them so the
# graph holds only the typed policy facts triple-mediated retrieval should bind.
# The fact-only filter (organizational/schema/boolean relations -- incl.
# ``in_section``, whose section-header tail entities otherwise hijack query
# binding) is defined ONCE in knowlytix and reused here, so the policy is not a
# magic constant duplicated in the consumer. store_from_triples also applies it
# by default; we filter the loop output too so the build report counts match.
from knowlytix.knowledge.geode.rag import DEFAULT_NOISE_RELATIONS as _BASE_NOISE_RELATIONS

# Served-graph noise = the library defaults PLUS ``has_alias``. The trained store
# stays content-only (no alias-tail entities), so GMS/cap training is fast and the
# model entity set is the policy entities alone. The customer-alias table is still
# used for operator-native head binding, but it is injected into the served
# doc_graph at LOAD time (PolicyRagRetriever._inject_aliases) as has_alias NAMES --
# the parser reads them for head_candidates and excludes has_alias from content
# retrieval, so "social security number" binds pii_handling without alias tails
# ever becoming head candidates or model entities, and without the cost of
# training them. The reg aliases are de-collided in the corpus
# (``reg_e has_alias "regulation e"``) so there is no binding collision.
_NOISE_RELATIONS = frozenset(set(_BASE_NOISE_RELATIONS) | {"has_alias"})


def _rel_to_repo(p) -> str:
    """Path relative to the repo root, tolerant of relative ``--doc``/``--store-path``
    inputs (``Path.relative_to`` raises on a relative path; resolve first)."""
    try:
        return str(Path(p).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(p)


def _make_compat_trainer(device, *, epochs: int = 300, lambda_path: float = 1.0):
    """A GEODE-loop trainer compatible with the installed knowlytix 0.2.0.

    The branch's ``make_default_trainer`` calls ``train_gms(..., lambda_path=)``,
    a 0.5.0-only kwarg the installed 0.2.0 ``train_gms`` does not accept. This
    mirrors that trainer (small cap+path GMS for contradiction/admissibility
    diagnosis) but routes the path-consistency weight through ``LossConfig`` —
    the 0.2.0 way. Remove once geode/rag ship in the 0.2.x wheel.
    """
    def _trainer(triples):
        import torch
        from knowlytix.core.config import CapLossConfig, GeometryConfig, LossConfig
        from knowlytix.core.train_finstructbench import GraphToGMS, train_gms
        from knowlytix.knowledge.geode.anchor import enm_from_triples

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _G:
            def __init__(self, t):
                self.triples = list(t)

        adapter = GraphToGMS(_G(triples))
        model = train_gms(
            adapter, dev, epochs=epochs, batch_size=64, num_neg=16,
            geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32),
            loss=LossConfig(lambda_path=lambda_path),
            loss_mode="cap", cap=CapLossConfig(), seed=42,
        )
        model.eval()
        enm = enm_from_triples(triples)
        return model, adapter, enm

    return _trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default=str(_DEFAULT_DOC),
                        help="source markdown (default data/banking_policy.md)")
    parser.add_argument("--store-path", default=str(_DEFAULT_STORE),
                        help="output store dir (default data/gms_policy_store_geode)")
    parser.add_argument("--epochs", type=int, default=800,
                        help="production GMS training epochs (default 800; the "
                             "cap geometry converges by ~800, 300 is undertrained)")
    parser.add_argument("--max-iters", type=int, default=8,
                        help="GEODE self-correction loop iteration bound")
    parser.add_argument("--no-actor", action="store_true",
                        help="run the GEODE loop with geometry-only correction "
                             "(no Qwen actor)")
    parser.add_argument("--no-embed-sft", action="store_true",
                        help="skip the geometry-supervised iterative encoder SFT "
                             "(GeodeEmbedLoop) that warm-starts the store's v-space "
                             "from a document-tuned encoder")
    parser.add_argument("--actor-device", default="cuda",
                        help="device for the Qwen GEODE actor (default cuda)")
    parser.add_argument("--geo-device", default="cuda",
                        help="device for GMS geometry (default cuda; safe on "
                             "cu130, falls back to cpu if no GPU)")
    parser.add_argument("--min-coverage", type=float, default=0.8,
                        help="region coverage_ratio below which the build prints "
                             "a prominent warning listing blind-spot sections "
                             "(does not fail the build)")
    parser.add_argument("--canonicalize", action="store_true",
                        help="merge co-referent entities/relations in the loop "
                             "(geometry-near + logic-vetoed, abstain-on-ambiguity). "
                             "Off by default: it rewrites graph vocabulary, so "
                             "validate the proposed merges (see geode_build_report) "
                             "before shipping a canonicalized store")
    parser.add_argument("--canon-sim", type=float, default=0.86,
                        help="min cosine to propose a canonicalization merge")
    parser.add_argument("--canon-margin", type=float, default=0.05,
                        help="min gap to the runner-up; ambiguous surfaces abstain")
    parser.add_argument("--no-tail-aliases", action="store_true",
                        help="skip generating head+tail entity synonym aliases for "
                             "the v-space encoder SFT (synonym tolerance via "
                             "plausibility); aliases need the actor")
    parser.add_argument("--calibrate-relevance", action="store_true",
                        help="fit the embedding relevance-gate operating points "
                             "(off by default; retired under operator-native "
                             "head_facts retrieval, and ~O(rels^2) encoder calls)")
    args = parser.parse_args()

    import torch
    from knowlytix.core.config import CapLossConfig
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.geode.loop import GeodeLoop
    from knowlytix.knowledge.geode.rag import RagBuildResult, store_from_triples

    if not Path(args.doc).exists():
        print(f"FAIL: missing source doc {args.doc}", file=sys.stderr)
        return 1

    # GMS geometry on GPU: the GB10 cap/Cayley miscompute was an old-wheel issue,
    # resolved on torch 2.10+cu130 (CPU/GPU parity ~1e-6, cross-process stable,
    # GEODE training identical to CPU and ~8x faster). Override with --geo-device.
    geo_device = torch.device(args.geo_device if torch.cuda.is_available()
                              else "cpu")

    from knowlytix.knowledge.geode import QWEN_4B  # Qwen3-4B-Instruct-2507

    actor = None
    if not args.no_actor:
        from knowlytix.knowledge.geode.agent_llm import qwen_agent_callable
        dev = args.actor_device if torch.cuda.is_available() else "cpu"
        print(f"GEODE actor: {QWEN_4B} (locked) on {dev}")
        actor = qwen_agent_callable(QWEN_4B, device=dev, max_tokens=256)
    else:
        print("GEODE actor: none (geometry-only self-correction)")

    # loss_mode="cap" trains the relation-conditioned spherical-cap admissibility
    # geometry (paper sec.7); the core default is head-conditioned (many-to-many),
    # so retrieved/claimed triples can be gated by calibrated cap distance (legit
    # facts d~0.03, fabrications/cross-tail d~1.5). Round-trips through the store's
    # cap_enabled save/load + the rho_w load-shim.
    config = DocGMSConfig(ingest_mode="regex", store_path=args.store_path)
    config.train.epochs = args.epochs
    config.train.device = str(geo_device)
    config.loss_mode = "cap"
    config.cap = CapLossConfig()

    print(f"building GEODE RAG store from {args.doc}")
    print(f"  -> {args.store_path}  (geometry on {geo_device}, "
          f"{args.epochs} epochs, max_iters={args.max_iters})")

    # Full GEODE self-correction (propose -> diagnose -> repair), then drop the
    # non-fact noise relations from the corrected graph before training the
    # production store. build_rag_store() is inlined here so the filter sits
    # between the loop and store_from_triples.
    loop = GeodeLoop(
        _make_compat_trainer(geo_device, epochs=args.epochs),
        device=geo_device, llm=actor, max_iters=args.max_iters,
        canonicalize=args.canonicalize, canon_sim=args.canon_sim,
        canon_margin=args.canon_margin,
        # The loop runs before the embed-SFT block, so the document-tuned encoders
        # do not exist yet: relation-phrase similarity uses the default MiniLM and
        # entity merges use the trained GMS geometry. (Threading the tuned encoders
        # would require running the embed loop first.)
    )
    loop_res = loop.run(args.doc)
    if args.canonicalize:
        print(f"  canonicalization: {len(loop_res.canonicalizations)} merge(s), "
              f"{len(loop_res.canonicalize_flagged)} flagged ambiguous")
        for m in loop_res.canonicalizations:
            print(f"    merge[{m['kind']}] {m['merged']} -> {m['canonical']} "
                  f"(sim={m.get('max_similarity')})")
    kept = [(h, r, t) for h, r, t in loop_res.triples if r not in _NOISE_RELATIONS]
    dropped = len(loop_res.triples) - len(kept)
    print(f"  GEODE loop: converged={loop_res.converged} in {loop_res.iterations} "
          f"iter(s); dropped {dropped} noise triples ({sorted(_NOISE_RELATIONS)})")

    # Geometry-supervised iterative encoder SFT (GeodeEmbedLoop): tune a low-rank
    # adapter over MiniLM on the document's own alias/name structure (extended by
    # the GMS geometry), then warm-start the production store's v-space from that
    # document-tuned encoder via EmbeddingConfig Mode B. This makes customer
    # vocabulary ("chargeback", "NSF charge", "unfair fee") bind to the right
    # policy entity -- the binding the frozen MiniLM gets wrong on hard phrasings.
    if not args.no_embed_sft:
        from knowlytix.embedding import EmbeddingSFTConfig
        from knowlytix.knowledge.geode.embed_loop import (
            EmbedLoopConfig, GeodeEmbedLoop)
        from knowlytix.knowledge.geode.loop import make_default_trainer

        sft_cfg = EmbeddingSFTConfig(
            rank=8, mode="full", epochs=200, drift_weight=0.5,
            out_dim=config.geometry.d_v, device=str(geo_device))
        eloop = GeodeEmbedLoop(
            make_default_trainer(geo_device, epochs=150),
            EmbedLoopConfig(sft=sft_cfg, max_iters=2, use_geometry=True))
        ent_names = sorted({e for h, _, t in kept for e in (h, t)})
        # Synonym tolerance lives in V-SPACE: generate aliases for BOTH heads and
        # tails (a value tail like "forbidden" gets "prohibited"/"banned") and seed
        # the encoder SFT with them, so a synonym is placed near its entity and
        # plausibility (score_triple / cap) accepts a paraphrased value. Polarity
        # (opposite-stance contradiction) is handled separately in tuned u-space.
        seed_aliases = None
        if actor is not None and not args.no_tail_aliases:
            from knowlytix.knowledge.geode.alias_gen import generate_entity_aliases
            seed_aliases = generate_entity_aliases(
                ent_names, llm=actor, n_aliases=6, max_tokens=96,
                block_values=ent_names)
            n_al = sum(len(v) for v in seed_aliases.values())
            print(f"  v-space synonyms: generated {n_al} aliases over "
                  f"{len(ent_names)} entities (heads+tails)")
        eres = eloop.run(args.doc, seed_labels=seed_aliases)
        v_emb = eres.ft.export_vectors(ent_names)
        Path(args.store_path).mkdir(parents=True, exist_ok=True)
        vpath = Path(args.store_path) / "v_emb.pt"
        torch.save(v_emb, vpath)
        config.embedding.v_vectors_path = str(vpath)
        # Persist the tuned encoder itself so the retriever can use it for
        # query->entity binding AND the question<->attribute relevance gate
        # (RagConfig.encoder). That is where doc-tuning pays off: the gate stops
        # rejecting "overdraft fee" as an attribute overdraft does not hold, and
        # binds colloquial vocabulary the frozen MiniLM misses.
        eres.ft.save(Path(args.store_path) / "tuned_encoder")
        print(f"  embed-SFT (v-space): tuned encoder over {eres.iterations} iter(s) "
              f"(converged={eres.converged}); warm-started {len(v_emb)} entity "
              f"v-vectors + saved tuned_encoder/ for binding + v-accept relevance")

        # u-space contradiction SFT: the logical half of the relevance decision.
        # v-space accepts the nearest relation; u-space tension vetoes an
        # absent-but-adjacent attribute (interest rate != fee) -- a contradiction
        # v-space similarity cannot see. Supervision: same-relation phrasings
        # (consistent) vs cross-relation (contradictory). Phrasings are generated
        # once by the GEODE actor (LLM at BUILD time for data, never a runtime
        # judge); the runtime veto is pure geometry (tension). Falls back to the
        # relation name alone when no actor is configured.
        from knowlytix.knowledge.geode.embed_loop import contradiction_sft

        content_rels = sorted({
            r for _h, r, _t in kept
            if r.startswith("has_") and r not in ("has_alias",)})

        def _phrasings(rel: str) -> list[str]:
            name = (rel[4:] if rel.startswith("has_") else rel).replace("_", " ")
            variants = {name}
            if actor is not None:
                prompt = (f"List 4 short, distinct ways a bank customer might ask "
                          f"about a policy's '{name}'. One phrase per line, no "
                          f"numbering, no extra words.")
                try:
                    for ln in (actor(prompt) or "").splitlines():
                        s = ln.strip(" -*\t0123456789.").strip().lower()
                        if 2 <= len(s) <= 48:
                            variants.add(s)
                except Exception:  # noqa: BLE001 - phrasings are best-effort
                    pass
            return sorted(variants)[:5]

        phrasings = {r: _phrasings(r) for r in content_rels}
        u_enc = contradiction_sft(phrasings, epochs=400)
        u_enc.save(Path(args.store_path) / "contradiction_encoder")
        print(f"  contradiction-SFT (u-space): trained over {len(content_rels)} "
              f"relations; saved contradiction_encoder/ for u-veto relevance")

        # Calibrate the geometric relevance gate's operating points (v-accept
        # floor + per-relation u-veto cut) from the same phrasings -- every
        # decision gate is calibrated from the store, none hardcoded. Skipped by
        # default: the operator-native pipeline (head_facts) retires the relevance
        # gate, and this calibration calls the encoder ~O(rels^2) times (minutes).
        if args.calibrate_relevance:
            from knowlytix.knowledge.rag.relevance import (
                calibrate_relevance_thresholds)

            relcal = calibrate_relevance_thresholds(
                phrasings, eres.ft.encode, u_enc.encode)
            (Path(args.store_path) / "relevance_calibration.json").write_text(
                json.dumps(relcal, indent=2) + "\n")
            print(f"  relevance gate calibrated: tau_accept={relcal['tau_accept']}, "
                  f"default u-veto={relcal['default_tau_contra']}, "
                  f"per-relation cuts for {len(relcal['tau_contra_per_relation'])} "
                  f"relations ({len(relcal['overlap'])} with residual overlap)")

    store = store_from_triples(args.doc, kept, config, device=geo_device,
                               drop_relations=_NOISE_RELATIONS)
    result = RagBuildResult(
        store=store, converged=loop_res.converged, iterations=loop_res.iterations,
        n_entities=store.adapter.num_entities,
        n_triples=len(store.doc_graph.triples),
        n_enm=len(store.doc_graph.enm),
        corrections=loop_res.corrections,
        anchor_violations=loop_res.anchor_violations,
    )

    # --- diagnostics --------------------------------------------------------
    rels = collections.Counter(r for _, r, _ in store.doc_graph.triples)
    print("\n=== GEODE build result ===")
    print(f"  converged:         {result.converged} (after {result.iterations} iters)")
    print(f"  entities:          {result.n_entities}")
    print(f"  triples:           {result.n_triples}")
    print(f"  ENM registers:     {result.n_enm}")
    print(f"  relations ({len(rels)}):    {dict(rels)}")
    print(f"  GEODE corrections: {len(result.corrections)}")
    print(f"  anchor violations: {len(result.anchor_violations)}")
    for c in result.corrections[:8]:
        print(f"    correction: {c}")
    for v in result.anchor_violations[:8]:
        print(f"    anchor violation: {v}")

    # --- coverage report ----------------------------------------------------
    # Triple-mediated retrieval can only answer where the document was
    # triplified, so make the blind spots measurable rather than silent. Region
    # coverage = sections with >=1 fact triple; graph coverage = orphan entities,
    # single-fact relations, and relations declared (is_functional) but never
    # populated. An under-covered region is not an error -- a question about it
    # abstains -- but it must be surfaced, not hidden.
    from knowlytix.knowledge.rag import coverage_report, graph_coverage

    cov = coverage_report(store)
    # graph_coverage needs the FULL corrected graph (pre fact-only filter) so it
    # can see in_section/is_functional edges -- store_from_triples strips them, so
    # feed loop_res.triples (post-correction, pre-filter), not the served store.
    gcov = graph_coverage(loop_res.triples)
    (Path(args.store_path) / "coverage_report.json").write_text(
        json.dumps({**cov.as_dict(), "graph": gcov.as_dict()}, indent=2) + "\n")
    print("\n=== coverage ===")
    print(f"  region coverage:   {cov.coverage_ratio:.2f} "
          f"({len(cov.blind_spots)} blind-spot section(s), "
          f"{cov.unaligned_triples} unaligned triple(s))")
    print(f"  fact entities:     {gcov.n_fact_entities}/{gcov.n_subject_entities} "
          f"({len(gcov.orphan_entities)} orphan)")
    if gcov.singleton_relations:
        print(f"  single-fact rels:  {gcov.singleton_relations}")
    if gcov.unpopulated_declared:
        print(f"  declared but unpopulated: {gcov.unpopulated_declared}")
    print(f"  wrote {args.store_path}/coverage_report.json")

    if cov.coverage_ratio < args.min_coverage:
        print(f"\n  !! WARNING: region coverage {cov.coverage_ratio:.2f} < "
              f"--min-coverage {args.min_coverage:.2f}")
        print(f"  !! {len(cov.blind_spots)} section(s) have body text but no "
              f"triples -- questions about them will abstain:")
        for r in cov.blind_spots:
            print(f"  !!   - {r.title} (lines {r.line_start}-{r.line_end}, "
                  f"{r.body_lines} body lines)")
        print("  !! Consider re-ingesting in hybrid mode or enriching the source "
              "tables to lift coverage.")

    # Persist a human-readable build report alongside the store.
    report = {
        "doc": _rel_to_repo(args.doc),
        "store_path": _rel_to_repo(args.store_path),
        "converged": result.converged,
        "iterations": result.iterations,
        "n_entities": result.n_entities,
        "n_triples": result.n_triples,
        "n_enm": result.n_enm,
        "relations": dict(rels),
        "n_corrections": len(result.corrections),
        "n_anchor_violations": len(result.anchor_violations),
        "corrections": result.corrections,
        "anchor_violations": result.anchor_violations,
        "actor": None if args.no_actor else QWEN_4B,
        "epochs": args.epochs,
        "canonicalization": {
            "enabled": args.canonicalize,
            "merges": loop_res.canonicalizations,
            "flagged": loop_res.canonicalize_flagged,
        },
        "coverage": {
            "region_coverage_ratio": cov.coverage_ratio,
            "min_coverage": args.min_coverage,
            "below_min_coverage": cov.coverage_ratio < args.min_coverage,
            "blind_spots": [r.title for r in cov.blind_spots],
            "unaligned_triples": cov.unaligned_triples,
            "orphan_entities": gcov.orphan_entities,
            "singleton_relations": gcov.singleton_relations,
            "unpopulated_declared": gcov.unpopulated_declared,
        },
    }
    (Path(args.store_path) / "geode_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.store_path}/geode_build_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
