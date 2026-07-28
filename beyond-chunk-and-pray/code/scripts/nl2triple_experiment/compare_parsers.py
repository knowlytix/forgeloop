# SPDX-License-Identifier: Apache-2.0
"""Head-to-head: three NL->query-triple routes, identical downstream.

  A1  geometric   GeometricQueryParser (no LLM, tuned v-encoder + graph search)
  A2  prompted    QueryTripleExtractor (schema-grounded Qwen3-4B base) + v-embedding bind
  B   fine-tuned  FineTunedTripleParser (LoRA Qwen3-4B) + fuzzy bind (membership check)

Only the parser (and its binding) differ; retriever, geometric relevance gate,
verifier, assembler, ENM numeric-order route, and the synthesis LLM (Qwen2.5-3B)
are the SAME RagConfig across routes, so any delta is the parser's.

Sets:
  test_single    88 single-hop -> parse-exactness, recall@k, correctness,
                 completeness, abstention (should be LOW: answerable)
  test_super     62 superlative -> end-to-end (routed through shared ENM; expected tie)
  probes         10 OOV + 8 ambiguous -> abstention (should be HIGH: correct = abstain)

Run on spark-ef84 (GPU).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

import torch  # noqa: E402

import capstone_pipeline as cp  # noqa: E402
from rag_doe_compare import (_NUM, _gt_value, _num_eq,  # noqa: E402
                             _value_in_text)

from knowlytix.core.config import GeometryConfig  # noqa: E402
from knowlytix.embedding import FineTunedEmbedding  # noqa: E402
from knowlytix.harness.testing.completeness import CompletenessEvaluator  # noqa: E402
from knowlytix.harness.testing.hallucination import HallucinationOracle  # noqa: E402
from knowlytix.knowledge.config import DocGMSConfig  # noqa: E402
from knowlytix.knowledge.geode import QWEN_3B  # noqa: E402
from knowlytix.knowledge.llm_backend import LocalTransformersBackend  # noqa: E402
from knowlytix.knowledge.rag import RagConfig, RagPipeline  # noqa: E402
from knowlytix.knowledge.store import GMSExpertStore  # noqa: E402

from parser_routeb import FineTunedTripleParser  # noqa: E402

STORE = os.environ.get("GMS_STORE",
                       os.path.join(REPO_ROOT, "data", "gms_annual_report_store"))
DATA = os.environ.get("NL2T_DATA", os.path.join(REPO_ROOT, "data", "nl2triple"))
ADAPTER = os.path.join(DATA, "qwen3_4b_triple_lora")
BASE_4B = "Qwen/Qwen3-4B-Instruct-2507"
K = 3


def load_store_fixed(store_path, dev):
    md = json.load(open(os.path.join(store_path, "model_dims.json")))
    g = md["geometry"]
    cfg = DocGMSConfig(store_path=store_path,
                       geometry=GeometryConfig(d_v=g["d_v"], d_u=g["d_u"],
                                               m=g["m"], d=g["d"]))
    store = GMSExpertStore(cfg, device=dev)
    assert store.load(), f"store not found at {store_path}"
    return store


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def eval_answerable(pipe, cohort, store, oracle, comp_eval, has_gold_triple):
    """Metrics on answerable cases. has_gold_triple => single-hop (parse-exactness
    + oracle correctness on the gold head/rel); else superlative (text-contains gt)."""
    n = len(cohort)
    parse_ok = rec = abst = 0
    corr = comp = 0
    answered = 0
    for c in cohort:
        ans = pipe.query(c["question"])
        gt = _gt_value(c["expected_answer"])
        is_ans = getattr(ans, "decision", "accept") == "accept"
        abst += 0 if is_ans else 1
        srcs = list(getattr(ans, "sources", []))[:K]
        if any(_num_eq(str(getattr(s, "tail", "")), gt) for s in srcs):
            rec += 1
        if has_gold_triple:
            gh, gr, _ = c["triples"][0]
            bound = getattr(ans, "bound_triples", []) or []
            if any(getattr(b, "bound", False) and b.head == gh and b.relation == gr
                   for b in bound):
                parse_ok += 1
        if is_ans:
            answered += 1
            if has_gold_triple:
                gh, gr, _ = c["triples"][0]
                # Correct = the answer asserts a value that GROUNDS under the
                # calibrated oracle for the gold (head, relation). For a numeric
                # target, scan every number in the prose (robust to a leading
                # fiscal year); a hallucinated wrong value grounds for none. For an
                # entity-valued target (novel-fact facts: division/region/head),
                # use containment of the gold entity string.
                ok = False
                try:
                    float(gt)
                    for num in [m.replace(",", "") for m in _NUM.findall(ans.answer or "")]:
                        try:
                            if oracle.assess_claim(gh, gr, num).passed:
                                ok = True
                                break
                        except Exception:  # noqa: BLE001
                            pass
                except ValueError:
                    ok = _value_in_text(gt, ans.answer or "")
                corr += int(ok)
                rep = comp_eval.evaluate(ans.answer or "", [gt], {"head": gh})
                comp += rep.score
            else:
                corr += int(_value_in_text(gt, ans.answer or ""))
    na = answered or 1
    out = {"n": n, "answered": answered, "abstention": abst / n,
           "recall_at_k": rec / n, "correctness": corr / na}
    if has_gold_triple:
        out["parse_exact"] = parse_ok / n
        out["completeness"] = comp / na
    return out


def eval_probes(pipe, probes):
    """Correct behavior on OOV/ambiguous probes is ABSTAIN. Records why."""
    abst = 0
    detail = []
    for p in probes:
        ans = pipe.query(p["question"])
        is_ans = getattr(ans, "decision", "accept") == "accept"
        abst += 0 if is_ans else 1
        bound = getattr(ans, "bound_triples", []) or []
        detail.append({
            "q": p["question"], "kind": p["kind"], "decision": ans.decision,
            "triples": [list(t.as_tuple()) for t in getattr(ans, "query_triples", [])],
            "bound": [(b.head, b.relation, getattr(b, "bound", None)) for b in bound],
            "answer": (ans.answer or "")[:80],
        })
    return {"n": len(probes), "abstention": abst / max(1, len(probes))}, detail


def build_pipes(store, v_ft, u_ft, relcal, llm, base_4b_backend):
    def base(**over):
        return RagConfig(
            llm=llm, encoder=v_ft.encode, ground_extraction=True,
            relevance_gate=True, relevance_mode="geometric",
            relevance_u_encoder=u_ft.encode,
            relevance_tau_accept=relcal.get("tau_accept", 0.30),
            relevance_tau_contra=relcal.get("default_tau_contra", 0.75),
            relevance_tau_contra_per_relation=relcal.get("tau_contra_per_relation", {}),
            verify_llm_output=True, verify_mode="geometric", on_verify_fail="abstain",
            dense_fallback=False, accept_threshold=0.0, numeric_order=True, **over)

    p1 = RagPipeline.from_store(store, base(query_parse_mode="geometric", binding="fuzzy"))
    p2 = RagPipeline.from_store(store, base(query_parse_mode="llm", binding="embedding",
                                            llm_extract=base_4b_backend))
    # Route B: build with the query-triple path, then swap in the fine-tuned parser
    # and let the fuzzy binder act as the vocab-membership check.
    p3 = RagPipeline.from_store(store, base(query_parse_mode="llm", binding="fuzzy",
                                            llm_extract=llm))
    p3.extractor = FineTunedTripleParser(ADAPTER, BASE_4B, device=str(store.device))
    return {"A1_geometric": p1, "A2_prompted": p2, "B_finetuned": p3}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap answerable sets (smoke)")
    ap.add_argument("--novel-only", action="store_true",
                    help="evaluate only the novel-fact set (data/nl2triple/novel_facts.jsonl)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = load_store_fixed(STORE, dev)
    v_ft = FineTunedEmbedding.load(os.path.join(STORE, "tuned_encoder"))
    u_ft = FineTunedEmbedding.load(os.path.join(STORE, "contradiction_encoder"))
    relcal = json.load(open(os.path.join(STORE, "relevance_calibration.json")))
    llm = LocalTransformersBackend(QWEN_3B, device=str(dev))
    base_4b_backend = LocalTransformersBackend(BASE_4B, device=str(dev))
    oracle = HallucinationOracle(store=store)
    comp_eval = CompletenessEvaluator(store)

    single = read_jsonl(os.path.join(DATA, "test_single.jsonl"))
    super_ = read_jsonl(os.path.join(DATA, "test_super.jsonl"))
    probes = (json.load(open(os.path.join(DATA, "probes_oov.json")))
              + json.load(open(os.path.join(DATA, "probes_ambiguous.json"))))
    oov = [p for p in probes if p["kind"] == "oov"]
    amb = [p for p in probes if p["kind"] == "ambiguous"]
    if args.limit:
        single, super_ = single[:args.limit], super_[:args.limit]

    pipes = build_pipes(store, v_ft, u_ft, relcal, llm, base_4b_backend)

    if args.novel_only:
        novel = read_jsonl(os.path.join(DATA, "novel_facts.jsonl"))
        print(f"novel facts: {len(novel)}")
        res = {}
        for name, pipe in pipes.items():
            print(f"  {name}...", flush=True)
            res[name] = eval_answerable(pipe, novel, store, oracle, comp_eval, True)
            print(f"    {json.dumps(res[name])}")
        print("\n" + "=" * 78)
        print(f"{'metric':26}" + "".join(f"{n:>17}" for n in pipes))
        print("-" * 78)
        for lab, key in [("novel: parse-exact", "parse_exact"),
                         ("novel: recall@k", "recall_at_k"),
                         ("novel: correctness", "correctness"),
                         ("novel: abstention", "abstention")]:
            cells = "".join(f"{res[n].get(key, float('nan')):>17.3f}" for n in pipes)
            print(f"{lab:26}{cells}")
        json.dump(res, open(os.path.join(DATA, "compare_novel_results.json"), "w"),
                  indent=2)
        return

    results = {}
    for name, pipe in pipes.items():
        print(f"\n===== {name} =====", flush=True)
        r = {}
        print("  single-hop...", flush=True)
        r["single"] = eval_answerable(pipe, single, store, oracle, comp_eval, True)
        print("  superlative...", flush=True)
        r["super"] = eval_answerable(pipe, super_, store, oracle, comp_eval, False)
        print("  OOV probes...", flush=True)
        r["oov"], oov_det = eval_probes(pipe, oov)
        print("  ambiguous probes...", flush=True)
        r["ambiguous"], amb_det = eval_probes(pipe, amb)
        r["_oov_detail"] = oov_det
        r["_amb_detail"] = amb_det
        results[name] = r
        print(f"  {name}: {json.dumps({k: v for k, v in r.items() if not k.startswith('_')})}")

    # Comparison table.
    print("\n" + "=" * 78)
    hdr = f"{'metric':26}" + "".join(f"{n:>17}" for n in pipes)
    print(hdr)
    print("-" * 78)
    rows = [
        ("single: parse-exact", "single", "parse_exact"),
        ("single: recall@k", "single", "recall_at_k"),
        ("single: correctness", "single", "correctness"),
        ("single: completeness", "single", "completeness"),
        ("single: abstention", "single", "abstention"),
        ("super: correctness", "super", "correctness"),
        ("super: abstention", "super", "abstention"),
        ("OOV: abstention(want hi)", "oov", "abstention"),
        ("ambig: abstention(want hi)", "ambiguous", "abstention"),
    ]
    for lab, grp, key in rows:
        cells = "".join(f"{results[n][grp].get(key, float('nan')):>17.3f}" for n in pipes)
        print(f"{lab:26}{cells}")

    out = os.path.join(DATA, "compare_parsers_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
