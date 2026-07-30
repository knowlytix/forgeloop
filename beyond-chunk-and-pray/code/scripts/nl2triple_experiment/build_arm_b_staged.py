# SPDX-License-Identifier: Apache-2.0
"""Binder Bake-Off, Arm B: build the finetuned SLM binder, one stage per process.

The one-call ``build_query_compiler`` co-resides up to three Qwen3-4B copies in a
single process (datagen backend, the SFT model, then the eval parser reloading the
base), which is the OOM hazard on the GB10's unified memory. This runner splits the
same library pipeline into three stages, each its own ``python`` invocation, so every
Qwen3-4B copy is reclaimed at process exit and the peak footprint is one model, not
three. Stages also checkpoint to disk: a crash in ``train`` does not lose ``datagen``.

Each stage is a thin caller of the knowlytix compiler library
(``knowlytix.knowledge.rag.compiler``); no pipeline logic is reimplemented here.

Run (spark-venv resolves the dev compiler via its branch overlay; official knowlytix
lacks ``rag.compiler`` and the SFT extras):

    source ~/cluster/spark-venv/bin/activate
    python scripts/nl2triple_experiment/build_arm_b_staged.py --stage datagen
    python scripts/nl2triple_experiment/build_arm_b_staged.py --stage train
    python scripts/nl2triple_experiment/build_arm_b_staged.py --stage eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
from _bootstrap import load_store_geo  # noqa: E402  (local store-loader helper)

from forgeloop import data_path  # noqa: E402  (resolves the installed book data)

# Reclaim freed unified memory eagerly and reduce fragmentation across stages.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from knowlytix.knowledge.rag.compiler import QWEN_4B  # noqa: E402
from knowlytix.knowledge.rag.compiler.config import (COMPILER_SUBDIR,  # noqa: E402
                                                     CompilerSFTConfig)

STORE = os.environ.get("GMS_STORE", str(data_path("gms_annual_report_store")))
COMP_DIR = os.path.join(STORE, COMPILER_SUBDIR)
DATA_DIR = os.path.join(COMP_DIR, "data")

# Corpus knobs shared by datagen and the manifest.
VARIANTS_PER_BASE = 12
MAX_MULTI_HOP = 150
NEGATIVES_PER_HEAD = 1
HELDOUT_FACT_FRAC = 0.15   # reserve novel (head,relation) facts for novel_entity_score
# Reserve an unseen presentation LEVEL of a training factor for the in-distribution
# held-out-level eval. Must name a factor that survives the training-factor
# restriction (the default entity_aliasing is a holdout factor, so it is excluded
# and would leave the split empty).
HELDOUT_LEVELS = {"clarity": "Misleading"}
SEED = 42

# The 4th corpus consumer (Section 3 of the bake-off spec): (nl, query_triples).
ENRICH_DIR = str(data_path("enrichment"))
LLM_EXTRACT_SFT = os.path.join(ENRICH_DIR, "llm_extract_sft.jsonl")


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _train_factor_names():
    """The 'comprehensive' DoE factors MINUS the holdout's disjoint factors, so
    Arm B never trains on the presentation factors the real-language holdout uses
    (disjoint on the factor axis as well as the materializer axis)."""
    from knowlytix.harness.graphdoe import DesignMatrix
    from knowlytix.knowledge.rag.bakeoff import DISJOINT_HOLDOUT_FACTOR_NAMES
    comp = list(DesignMatrix.from_catalog(group="comprehensive", n_runs=4,
                                          method="sobol", seed=1).generate().columns)
    held = set(DISJOINT_HOLDOUT_FACTOR_NAMES)
    return [f for f in comp if f not in held]


def _hops_to_triples(hops):
    """Convert a ``[[head, relation], ...]`` chain to QueryTriple dicts with
    ``?``-prefixed unknown slots (ASKED tail on the final hop, ``?x/?y/...``
    intermediates), the (nl, triples) shape the spec's 4th consumer stores."""
    inter = ["?x", "?y", "?z", "?w"]
    triples = []
    cur = hops[0][0] if hops else None
    for i, hop in enumerate(hops):
        rel = hop[1]
        last = i == len(hops) - 1
        tail = "?" if last else inter[i % len(inter)]
        triples.append({"head": cur, "relation": rel, "tail": tail})
        cur = tail
    return triples


def _emit_llm_extract_sft(splits):
    """Write the (nl, triples) corpus from the train split (the answerable,
    in-scope rows -- relation-absent probes have no gold triple to bind)."""
    os.makedirs(ENRICH_DIR, exist_ok=True)
    n = 0
    with open(LLM_EXTRACT_SFT, "w") as f:
        for row in splits.get("train", []):
            if row.get("category") == "relation_absent":
                continue
            rec = {"nl": row["question"], "triples": _hops_to_triples(row["hops"]),
                   "category": row.get("category"), "_factors": row.get("factors", {})}
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[datagen] llm_extract_sft: {n} (nl, triples) rows -> {LLM_EXTRACT_SFT}", flush=True)


def stage_datagen():
    """Load store + backend, generate the DoE corpus, write splits, exit."""
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    from knowlytix.knowledge.rag.compiler.datagen import build_compiler_dataset

    dev = _dev()
    store = load_store_geo(STORE, torch.device(dev))
    llm = LocalTransformersBackend(QWEN_4B, device=dev)
    factor_names = _train_factor_names()
    print(f"[datagen] store={STORE} dev={dev} triples={len(store.triples)} "
          f"train_factors={len(factor_names)} (holdout factors excluded)", flush=True)
    splits = build_compiler_dataset(
        store, llm, factor_names=factor_names, variants_per_base=VARIANTS_PER_BASE,
        max_multi_hop=MAX_MULTI_HOP, negatives_per_head=NEGATIVES_PER_HEAD,
        heldout_levels=HELDOUT_LEVELS, heldout_fact_frac=HELDOUT_FACT_FRAC,
        seed=SEED, out_dir=DATA_DIR)
    counts = {k: len(v) for k, v in splits.items()}
    print(f"[datagen] splits={counts} -> {DATA_DIR}", flush=True)
    assert counts.get("train", 0) > 0, "no training rows generated"
    _emit_llm_extract_sft(splits)


def stage_train():
    """LoRA-fine-tune the compiler on the datagen train split, save the adapter, exit."""
    from knowlytix.knowledge.rag.compiler.train import train_compiler

    train_path = os.path.join(DATA_DIR, "train.jsonl")
    assert os.path.isfile(train_path), f"run --stage datagen first ({train_path} missing)"
    cfg = CompilerSFTConfig(device=_dev())
    print(f"[train] base={cfg.base_model} r={cfg.lora_r} epochs={cfg.epochs} "
          f"bs={cfg.per_device_batch_size}x{cfg.grad_accum_steps} -> {COMP_DIR}", flush=True)
    out = train_compiler(train_path, COMP_DIR, config=cfg)
    print(f"[train] adapter saved: {out}", flush=True)


def stage_eval():
    """Load store + ONE compiler parser, score accept-cohort + both holdouts, write manifest."""
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    from knowlytix.knowledge.rag.compiler.evaluate import evaluate_compiler
    from knowlytix.knowledge.rag.compiler.model import CompiledQueryParser

    dev = _dev()
    store = load_store_geo(STORE, torch.device(dev))
    # One backend for synthesis (never actually called by evaluate) and one parser
    # loaded once; evaluate reuses it rather than reloading the base model.
    llm = LocalTransformersBackend(QWEN_4B, device=dev)
    parser = CompiledQueryParser(store, COMP_DIR, device=dev)

    def _rows(name):
        p = os.path.join(DATA_DIR, f"{name}.jsonl")
        return [json.loads(l) for l in open(p)] if os.path.isfile(p) else []

    heldout_level = _rows("test_heldout_level")
    heldout_fact = _rows("test_heldout_fact")

    metrics = evaluate_compiler(store, llm, parser=parser,
                                heldout=heldout_level or None, seed=SEED)
    # novel_entity_score: raw graph-recovery on facts whose head was unseen in training.
    if heldout_fact:
        nov = evaluate_compiler(store, llm, parser=parser, heldout=heldout_fact, seed=SEED)
        metrics["novel_fact_graph_recovery"] = nov.get("heldout_graph_recovery")
        metrics["n_novel_fact"] = nov.get("n_heldout")

    manifest = {
        "compiler_dir": COMP_DIR,
        "base_model": QWEN_4B,
        "n_train": len(_rows("train")),
        "n_heldout_level": len(heldout_level),
        "n_heldout_fact": len(heldout_fact),
        "variants_per_base": VARIANTS_PER_BASE,
        "heldout_fact_frac": HELDOUT_FACT_FRAC,
        "seed": SEED,
        "metrics": metrics,
    }
    with open(os.path.join(COMP_DIR, "compiler_meta.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[eval] manifest:")
    print(json.dumps(manifest, indent=2), flush=True)


STAGES = {"datagen": stage_datagen, "train": stage_train, "eval": stage_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    args = ap.parse_args()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
