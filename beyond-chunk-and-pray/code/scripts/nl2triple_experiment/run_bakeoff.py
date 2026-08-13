# SPDX-License-Identifier: Apache-2.0
"""Binder Bake-Off head-to-head (spec Section 6), Arms A/B/C/D.

Wires the four binder configurations to the shared bake-off harness and runs the
factor-sliced crossover on the DOE eval cohort plus the Claude-materialized
real-language holdout. Only the parser + binding differ; the retriever, geometric
relevance gate, verifier, assembler, ENM numeric route and synthesis LLM are one
RagConfig across arms, so any delta is the binder's.

  Arm A  encoder-binding   query_parse_mode="llm"      + embedding bind (tuned v-encoder)
  Arm B  finetuned SLM     query_parse_mode="compiler" (LoRA compiler over the store)
  Arm C  frozen few-shot   frozen base LM + k-NN training exemplars (no adapter)
  Arm D  alias-table hybrid compiler relations + patchable alias-table entity resolver

Arms C and D reuse the models arms A and B already load (the frozen base LM and the
compiler), so the four-arm run has the same footprint as the A/B run. C and D are
injected as ``pipe.extractor`` on an otherwise-standard pipeline, so every post-parse
stage is shared. All decision/metric logic lives in knowlytix.knowledge.rag.bakeoff;
this script only builds the arms and calls bakeoff(). Run under the memory guard.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
from _bootstrap import load_store_geo  # noqa: E402  (local store-loader helper)

from forgeloop import data_path  # noqa: E402  (resolves the installed book data)

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True

from knowlytix.embedding import FineTunedEmbedding  # noqa: E402
from knowlytix.knowledge.geode import QWEN_3B  # noqa: E402
from knowlytix.knowledge.llm_backend import LocalTransformersBackend  # noqa: E402
from knowlytix.knowledge.rag import RagConfig, RagPipeline  # noqa: E402
from knowlytix.knowledge.rag.compiler.config import COMPILER_SUBDIR, QWEN_4B  # noqa: E402
from knowlytix.knowledge.rag.bakeoff import (Arm, BinderPrediction,  # noqa: E402
                                             AliasHybridParser, AliasTableResolver,
                                             ExemplarFewShotParser, ExemplarIndex,
                                             RealizedFactorScorer,
                                             calibrate_alias_resolver, bakeoff)

STORE = os.environ.get("GMS_STORE", str(data_path("gms_annual_report_store")))
COMP_DIR = os.path.join(STORE, COMPILER_SUBDIR)
DATA_DIR = os.path.join(COMP_DIR, "data")
HOLDOUT = str(data_path("enrichment", "holdout", "holdout_materialized.jsonl"))
EXEMPLARS = str(data_path("enrichment", "llm_extract_sft.jsonl"))
ALIAS_CAL = os.path.join(STORE, "alias_resolver_calibration.json")
OUT = str(data_path("enrichment", "bakeoff_ABCD.json"))
FEWSHOT_K = int(os.environ.get("FEWSHOT_K", "6"))

# Realized presentation factors to slice the crossover on (measured, not intended).
SLICE_FACTORS = ["clarity", "style", "length", "expertise", "paraphrase_depth"]


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rows(name):
    p = os.path.join(DATA_DIR, f"{name}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.isfile(p) else []


def _binder(pipe):
    """Wrap a RagPipeline as a harness binder: question -> BinderPrediction using
    the bound query triples and gate decision (no synthesis)."""
    def binder(questions):
        preds = []
        for q in questions:
            ans = pipe.query(q, generate=False)
            triples = getattr(ans, "bound_triples", []) or []
            # All-or-nothing: an arm commits to a chain only if EVERY hop bound.
            # Filtering out the unbound hops instead would report a 2-hop query
            # whose second hop failed as a valid 1-hop chain -- a chain the arm
            # never predicted. metric_report's abstention_quality calls
            # _would_be_correct() on that pred_answer, so a fabricated
            # subsequence answers a different question than the one asked and
            # silently scores the "was this abstention justified?" column.
            all_bound = bool(triples) and all(
                getattr(b, "bound", False) for b in triples)
            hops = [[b.head, b.relation] for b in triples] if all_bound else None
            preds.append(BinderPrediction(hops=hops,
                                          decision=getattr(ans, "decision", "accept")))
        return preds
    return binder


def _base_config(llm, v_encode, u_encode, relcal, **over):
    return RagConfig(
        llm=llm, encoder=v_encode, ground_extraction=True,
        relevance_gate=True, relevance_mode="geometric", relevance_u_encoder=u_encode,
        relevance_tau_accept=relcal.get("tau_accept", 0.30),
        relevance_tau_contra=relcal.get("default_tau_contra", 0.75),
        relevance_tau_contra_per_relation=relcal.get("tau_contra_per_relation", {}),
        verify_llm_output=True, verify_mode="geometric", on_verify_fail="abstain",
        dense_fallback=False, accept_threshold=0.0,
        # numeric_order MUST stay off here, even though it is a good default in
        # production. RagPipeline.query runs _try_numeric_order BEFORE extraction
        # and regardless of generate=False, and that route returns decision
        # "accept" with NO bound_triples. _binder would then emit hops=None, and
        # run_arm scores bound=False / pred_answer=None as `mis_bind` -- for every
        # superlative or comparison question, in every arm, even when the pipeline
        # answered it correctly. mis_bind_rate is the load-bearing ceiling in the
        # G4 decision, so that would silently indict whichever binder saw the most
        # superlatives. This is a binder bake-off: every question must go through
        # the binder, not around it.
        numeric_order=False, **over)


def build_arms(store, llm, base_4b, v_encode, u_encode, relcal):
    # Arm A: prompted schema-grounded extractor + embedding bind over the tuned v-encoder.
    a = RagPipeline.from_store(store, _base_config(
        llm, v_encode, u_encode, relcal,
        query_parse_mode="llm", binding="embedding", llm_extract=base_4b))
    # Arm B: the LoRA compiler front-end.
    b = RagPipeline.from_store(store, _base_config(
        llm, v_encode, u_encode, relcal,
        query_parse_mode="compiler", compiler_dir=COMP_DIR,
        compiler_base_model=QWEN_4B, compiler_device=str(store.device)))

    # Arms C and D share the compiler-style downstream (fuzzy bind + exact walk):
    # build a light geometric-parse pipeline, then inject the arm's own parser as
    # pipe.extractor so every post-parse stage matches. No new model is loaded --
    # C reuses the frozen base LM (base_4b), D reuses B's compiler (b.extractor).
    def _shared_pipe():
        return RagPipeline.from_store(store, _base_config(
            llm, v_encode, u_encode, relcal, query_parse_mode="geometric",
            binding="fuzzy"))

    # Arm C: frozen base LM prompted with the k nearest training exemplars.
    rows = [json.loads(l) for l in open(EXEMPLARS)] if os.path.isfile(EXEMPLARS) else []
    index = ExemplarIndex.from_rows(rows, v_encode)
    c = _shared_pipe()
    c.extractor = ExemplarFewShotParser(store, base_4b, index, v_encode, k=FEWSHOT_K)

    # Arm D: compiler relations + a patchable alias-table entity resolver with a
    # calibrated encoder fallback (persisted operating point; recuses below tau).
    cal = calibrate_alias_resolver(
        AliasTableResolver.from_store(store).entities, v_encode,
        far_ceiling=0.05, persist_path=ALIAS_CAL)
    resolver = AliasTableResolver.from_store(store, encoder=v_encode, tau=cal["tau"])
    d = _shared_pipe()
    d.extractor = AliasHybridParser(store, b.extractor, resolver)
    print(f"[bakeoff] arm C exemplars={len(index)} k={FEWSHOT_K} | "
          f"arm D alias_tau={cal['tau']} calibrated={cal['calibrated']}", flush=True)

    return [
        Arm(name="A_encoder", binder=_binder(a), patch_mechanism="adapter_retune"),
        Arm(name="B_compiler", binder=_binder(b), patch_mechanism="full_retrain"),
        Arm(name="C_fewshot", binder=_binder(c), patch_mechanism="alias_edit"),
        Arm(name="D_hybrid", binder=_binder(d), patch_mechanism="alias_edit"),
    ]


def build_cohort():
    """In-distribution eval = held-out presentation levels; novel = held-out facts."""
    cohort = []
    for r in _rows("test_heldout_level"):
        r["novel"] = False
        cohort.append(r)
    for r in _rows("test_heldout_fact"):
        r["novel"] = True
        cohort.append(r)
    return cohort


def main():
    dev = _dev()
    store = load_store_geo(STORE, dev)
    v_ft = FineTunedEmbedding.load(os.path.join(STORE, "tuned_encoder"))
    u_ft = FineTunedEmbedding.load(os.path.join(STORE, "contradiction_encoder"))
    relcal = json.load(open(os.path.join(STORE, "relevance_calibration.json")))
    llm = LocalTransformersBackend(QWEN_3B, device=str(dev))
    base_4b = LocalTransformersBackend(QWEN_4B, device=str(dev))

    cohort = build_cohort()
    assert cohort, "no eval cohort; run --stage datagen first"

    # Key slices on the REALIZED factor levels, not the requested ones.
    scorer = RealizedFactorScorer.default(encoder=v_ft.encode).calibrate_length(cohort)
    cohort = scorer.annotate(cohort, out_key="_realized")
    for r in cohort:
        r["_factors"] = {**(r.get("factors") or {}), **r["_realized"]}

    from knowlytix.knowledge.rag.bakeoff import load_materialized_holdout
    holdout = load_materialized_holdout(HOLDOUT) if os.path.isfile(HOLDOUT) else None

    arms = build_arms(store, llm, base_4b, v_ft.encode, u_ft.encode, relcal)
    print(f"[bakeoff] cohort={len(cohort)} holdout={len(holdout) if holdout else 0} "
          f"arms={[a.name for a in arms]}", flush=True)

    report = bakeoff(arms, cohort, store.doc_graph, factors=SLICE_FACTORS,
                     holdout=holdout, conformal_alpha=0.1)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== overall ===", flush=True)
    for name, e in report["arms"].items():
        o = e["overall"]
        line = (f"  {name:12s} acc={o['accuracy']:.3f} mis_bind={o['mis_bind_rate']:.3f} "
                f"fail_bind={o['fail_bind_rate']:.3f} oos_FAR={o['oos_false_accept']:.3f} "
                f"novel={e['novel_entity']['novel_entity_score']:.3f} patch_cost={e['patch_cost']}")
        if "holdout" in e:
            line += (f" | holdout_acc={e['holdout']['accuracy']:.3f} "
                     f"gap={e['synthetic_vs_real_gap']:.3f} "
                     f"conf_cov={e['conformal']['coverage']:.3f}/{e['conformal']['target']:.2f}")
        print(line, flush=True)
    print(f"\n[bakeoff] overall accuracy leader: {report['crossover']['overall_accuracy_leader']}")
    print(f"[bakeoff] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
