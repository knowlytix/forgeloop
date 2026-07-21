# SPDX-License-Identifier: Apache-2.0
"""Build the NL->triple SFT experiment data (Route B: fine-tuned parser).

Reuses the book's DoE machinery (scripts/enrich_data.py helpers) to:

  1. TRAIN corpus: generate a surface-diverse cohort at a *different* seed than
     the shipped one, and for each single-hop (exact_recall) scenario recover the
     golden (head, relation, "?") query triple from the store. Superlative
     multi_hop scenarios are skipped (their correct triple is answer-dependent;
     they are measured end-to-end only).

  2. TEST set: the shipped data/enrichment/rag_cohort.json (seed 42). exact_recall
     cases get a recovered golden triple (parse-exactness + end-to-end);
     multi_hop cases are kept end-to-end only.

  3. PROBES: hand-authored OOV (attribute/entity the graph does not hold) and
     ambiguous (under-specified / near-tie) probes. Correct behavior on both is
     ABSTAIN -- the abstention comparison the experiment turns on.

GPU (Qwen rewrite for the training phrasings). Run on spark-ef84.
"""
from __future__ import annotations

import collections
import contextlib
import io
import json
import os
import re
import sys
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

import torch  # noqa: E402
import enrich_data as ed  # noqa: E402  (reuse _natural_target/_build_rewrite_prompt/SEMANTIC)

from knowlytix.core.config import GeometryConfig  # noqa: E402
from knowlytix.harness.suite import (  # noqa: E402
    Catalog, CatalogBaseSource, compose, graphdoe_design, resolve)
from knowlytix.knowledge.config import DocGMSConfig  # noqa: E402
from knowlytix.knowledge.geode import QWEN_3B  # noqa: E402
from knowlytix.knowledge.llm_backend import LocalTransformersBackend  # noqa: E402
from knowlytix.knowledge.store import GMSExpertStore  # noqa: E402

STORE = os.environ.get("GMS_STORE",
                       os.path.join(REPO_ROOT, "data", "gms_annual_report_store"))
COHORT = os.path.join(os.environ.get("GMS_ENRICH",
                      os.path.join(REPO_ROOT, "data", "enrichment")), "rag_cohort.json")
OUTDIR = os.environ.get("NL2T_DATA", os.path.join(REPO_ROOT, "data", "nl2triple"))

TRAIN_SEED = 7          # != shipped cohort seed (42): different phrasings
TRAIN_RUNS = 500
MAX_PER_CAT = 40        # enumerate the full single-hop fact space so training
                        # covers every fact the test set asks about (phrasing, not
                        # fact coverage, is the variable under test)


def load_store_fixed(store_path, dev):
    """Loader that respects the store's own geometry (model_dims.json). Works
    around the dev-branch DocGMSConfig default-geometry drift (store d=32 vs
    current default d=128); capstone_pipeline.load_store's bare-config path fails."""
    md = json.load(open(os.path.join(store_path, "model_dims.json")))
    g = md["geometry"]
    cfg = DocGMSConfig(store_path=store_path,
                       geometry=GeometryConfig(d_v=g["d_v"], d_u=g["d_u"],
                                               m=g["m"], d=g["d"]))
    store = GMSExpertStore(cfg, device=dev)
    assert store.load(), f"store not found at {store_path}"
    return store


def _numeq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip().lower() == str(b).strip().lower()


def golden_triple(trips, gt, attr):
    """Recover the unique (head, relation, '?') for an exact_recall case, matching
    rag_doe_compare._golden: the store triple whose tail == gt (and relation == attr
    when attr is a has_ relation). Returns (head, relation) or None if not unique."""
    heads = [(h, r) for h, r, t in trips
             if _numeq(t, gt) and (not attr.startswith("has_") or r == attr)]
    return heads[0] if len(heads) == 1 else None


def _gt_value(expected):
    if isinstance(expected, list) and len(expected) == 2:
        return str(expected[1])
    return str(expected)


def make_rewriter(dev):
    """The same batched greedy Qwen2.5-3B rewrite the book uses for phrasings."""
    backend = LocalTransformersBackend(QWEN_3B, device=str(dev))
    tok, model = backend._tokenizer, backend._model
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    @torch.no_grad()
    def rewrite_batch(prompts):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with contextlib.redirect_stderr(io.StringIO()):
            out = model.generate(**enc, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip().strip('"') for g in gen]

    return rewrite_batch


def _guarded(rw, target, anchors):
    """The book's on-topic + fabrication guard; falls back to a plain question."""
    norm = re.sub(r"[^a-z0-9 ]", " ", (rw or "").lower())
    on_topic = any(re.sub(r"[^a-z0-9 ]", " ", a.lower()).strip() in norm for a in anchors)
    allowed = set(re.findall(r"\d+", target))
    fabricated = any(n not in allowed for n in re.findall(r"\d+", rw or ""))
    if not rw or not on_topic or fabricated:
        return f"What is {target}?", True
    return rw, False


def build_train(store, trips, dev, batch_size=16):
    CAT = Catalog.load(os.path.join(use_branch_library(), "knowlytix",
                                    "harness", "suite", "catalogs"))
    suite = resolve(CAT, ed.BASE_TYPES, ed.FACTORS, mode="embedded")
    items = CatalogBaseSource(store, max_per_category=MAX_PER_CAT,
                              seed=TRAIN_SEED).items(suite)
    scns = compose(suite, items, n_runs=TRAIN_RUNS, seed=TRAIN_SEED,
                   design_fn=partial(graphdoe_design, method="sobol+refine"),
                   balance_base=True)
    print(f"[train] {len(items)} base -> {len(scns)} DoE scenarios (seed {TRAIN_SEED})")

    targets = [ed._natural_target(s.base) for s in scns]
    prompts = [ed._build_rewrite_prompt(t, s.factor_levels)
               for s, (t, _a, _anc) in zip(scns, targets)]
    rewrite_batch = make_rewriter(dev)
    rewrites = []
    for b in range(0, len(prompts), batch_size):
        rewrites.extend(rewrite_batch(prompts[b:b + batch_size]))
        print(f"  rewrote {min(b + batch_size, len(prompts))}/{len(prompts)}", flush=True)

    rows, skipped_super, skipped_nogold, fell_back = [], 0, 0, 0
    for s, rw, (target, attr, anchors) in zip(scns, rewrites, targets):
        if s.base.base != "exact_recall":
            skipped_super += 1
            continue
        rw, fb = _guarded(rw, target, anchors)
        fell_back += int(fb)
        gt = _gt_value(s.base.answer)
        g = golden_triple(trips, gt, attr)
        if g is None:
            skipped_nogold += 1
            continue
        head, rel = g
        rows.append({"question": rw, "triples": [[head, rel, "?"]],
                     "base": s.base.base, "attribute": attr,
                     "_factors": dict(s.factor_levels)})
    print(f"[train] {len(rows)} single-hop rows "
          f"(skipped {skipped_super} superlative, {skipped_nogold} no-unique-gold, "
          f"{fell_back} rewrites fell back)")
    return rows


def build_test(trips):
    cohort = json.load(open(COHORT))
    single, sup = [], []
    for c in cohort:
        gt = _gt_value(c["expected_answer"])
        if c["base"] == "exact_recall":
            g = golden_triple(trips, gt, c["attribute"])
            if g is None:
                continue
            head, rel = g
            single.append({"id": c["id"], "question": c["question"],
                           "triples": [[head, rel, "?"]],
                           "attribute": c["attribute"],
                           "expected_answer": c["expected_answer"],
                           "_factors": c.get("_factors", {})})
        else:
            sup.append({"id": c["id"], "question": c["question"],
                        "attribute": c["attribute"],
                        "expected_answer": c["expected_answer"],
                        "_factors": c.get("_factors", {})})
    print(f"[test] {len(single)} single-hop, {len(sup)} superlative (end-to-end only)")
    return single, sup


# Adversarial probes. Correct behavior on ALL of these is ABSTAIN.
# OOV: an attribute/entity the Northwind graph does not hold.
PROBES_OOV = [
    "What is Northwind's debt-to-equity ratio?",
    "How many patents does the Cloud Platform segment hold?",
    "What was the marketing budget for the Retail segment?",
    "What is the customer churn rate for Devices?",
    "How much did Northwind spend on research and development in FY2025?",
    "What dividend per share did Northwind pay this year?",
    "Who is Northwind's Chief Marketing Officer?",
    "What is the gross margin percentage for Cloud Platform?",
    "How many employees work in the Marketing division?",
    "What was Northwind's stock price at fiscal year end?",
]
# Ambiguous: under-specified referent or a term that ties between graph terms.
PROBES_AMBIGUOUS = [
    "What was the value?",
    "How much did the unit make?",
    "What's the number for the segment?",
    "Can you give me the figure?",
    "How big is it?",
    "Tell me about the revenue.",
    "What was the headcount?",
    "What is the total?",
]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = load_store_fixed(STORE, dev)
    trips = list(store.triples)
    print("store loaded; triples:", len(trips))

    os.makedirs(OUTDIR, exist_ok=True)
    test_single, test_super = build_test(trips)     # CPU, no leakage risk
    train_rows = build_train(store, trips, dev)

    # Dedup: drop any train question that appears verbatim in the test set.
    test_qs = {r["question"] for r in test_single} | {r["question"] for r in test_super}
    before = len(train_rows)
    train_rows = [r for r in train_rows if r["question"] not in test_qs]
    print(f"[dedup] dropped {before - len(train_rows)} train rows overlapping test")

    # 90/10 train/val split (val for training monitoring only).
    n_val = max(1, len(train_rows) // 10)
    val_rows, tr_rows = train_rows[:n_val], train_rows[n_val:]

    def dump(name, obj):
        p = os.path.join(OUTDIR, name)
        if name.endswith(".jsonl"):
            with open(p, "w") as fh:
                for r in obj:
                    fh.write(json.dumps(r) + "\n")
        else:
            json.dump(obj, open(p, "w"), indent=2)
        print(f"  wrote {name}: {len(obj)}")

    dump("train.jsonl", tr_rows)
    dump("val.jsonl", val_rows)
    dump("test_single.jsonl", test_single)
    dump("test_super.jsonl", test_super)
    dump("probes_oov.json", [{"question": q, "expect": "abstain", "kind": "oov"}
                             for q in PROBES_OOV])
    dump("probes_ambiguous.json", [{"question": q, "expect": "abstain", "kind": "ambiguous"}
                                   for q in PROBES_AMBIGUOUS])
    print("\nattribute balance (train):",
          dict(collections.Counter(r["attribute"] for r in tr_rows)))


if __name__ == "__main__":
    main()
