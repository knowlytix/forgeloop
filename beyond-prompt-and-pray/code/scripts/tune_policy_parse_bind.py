#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Stage B (Chunk-and-Pray Ch7): document-tune the banking parse-bind encoder
with the GEODE embed loop, seeded by the curated policy aliases AND Qwen3-4B
generated aliases.

The banking corpus declares customer aliases in its Policy Aliases table (the
``has_alias`` edges). We use those as the supervision seed and EXTEND them with
aliases the local Qwen3-4B proposes, then tune a low-rank adapter over the frozen
base encoder with ``GeodeEmbedLoop`` (aliases extended by the GMS geometry). The
generated aliases are saved for audit and never written into the served graph --
only the encoder learns from them. The resulting ``FineTunedEmbedding`` replaces
``<store>/tuned_encoder`` so the retriever binds colloquial customer vocabulary
to the right policy entity.

Run on spark-ef84:
    PYTHONPATH=$HOME/GMS-knowlytix \
      python scripts/tune_policy_parse_bind.py \
        --corpus data/banking_policy_full.md \
        --out data/gms_policy_store_cap/tuned_encoder
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import torch

from knowlytix.benchmark.ingest import ingest_markdown
from knowlytix.embedding import EmbeddingSFTConfig
from knowlytix.knowledge.geode import QWEN_4B
from knowlytix.knowledge.geode.alias_gen import generate_entity_aliases
from knowlytix.knowledge.geode.embed_loop import EmbedLoopConfig, GeodeEmbedLoop
from knowlytix.knowledge.geode.loop import make_default_trainer


def corpus_facts(md_path: str):
    """Canonical heads worth aliasing (``has_*`` fact edges), tail values, and the
    curated alias seed read off the ``has_alias`` edges."""
    tr = list(ingest_markdown(md_path, mode="regex").triples)
    heads = sorted({h for h, r, _t in tr
                    if r.startswith("has_") and r != "has_alias"})
    values = {t for _h, _r, t in tr}
    curated = collections.defaultdict(list)
    for h, r, t in tr:
        if r == "has_alias":
            curated[h].append(t)          # head canonical, tail customer alias
    return heads, values, dict(curated)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/banking_policy_full.md")
    ap.add_argument("--out", default="data/gms_policy_store_cap/tuned_encoder")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--gms-epochs", type=int, default=120)
    ap.add_argument("--alias-model", default=QWEN_4B)
    ap.add_argument("--n-aliases", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    print(f"device={dev}  corpus={args.corpus}")

    heads, values, curated = corpus_facts(args.corpus)
    ctx = open(args.corpus).read()
    print(f"aliasing {len(heads)} canonical entities; {len(curated)} have curated "
          f"aliases. generating up to {args.n_aliases} more each with "
          f"{args.alias_model} ...")
    generated = generate_entity_aliases(
        heads, args.alias_model, context=ctx, n_aliases=args.n_aliases,
        block_values=values, device=str(dev))

    # Seed = curated (Policy Aliases table) UNION generated (Qwen3-4B). Dedup,
    # keep curated first. Only heads that actually carry fact edges are aliased.
    seed = {}
    for h in heads:
        merged = list(dict.fromkeys(list(curated.get(h, [])) + list(generated.get(h, []))))
        if merged:
            seed[h] = merged
    json.dump({"curated": curated, "generated": generated, "seed": seed},
              open(os.path.join(args.out, "generated_aliases.json"), "w"), indent=2)
    print("  alias seed:")
    for h, a in seed.items():
        print(f"    {h}: {a}")

    sft = EmbeddingSFTConfig(rank=args.rank, mode=args.mode, epochs=args.epochs,
                             val_split=0.0, device=str(dev))
    loop = GeodeEmbedLoop(
        make_default_trainer(dev, epochs=args.gms_epochs),
        EmbedLoopConfig(sft=sft, max_iters=args.iters, use_geometry=True,
                        ingest_mode="regex"))
    res = loop.run(args.corpus, seed_labels=seed)
    print(f"\nconverged={res.converged} iters={res.iterations} "
          f"canonicals={len(res.canonicals)} "
          f"labels={sum(len(v) for v in res.labels.values())}")
    for h in res.history:
        print("  ", h)
    res.ft.save(args.out)
    json.dump({c: sorted(v) for c, v in res.labels.items()},
              open(os.path.join(args.out, "labels.json"), "w"), indent=2)
    print(f"\nsaved tuned encoder -> {args.out}")


if __name__ == "__main__":
    main()
