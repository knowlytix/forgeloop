#!/usr/bin/env python
"""Benchmark the query-parsing step: geometric vs LLM (Qwen 3B / 7B).

Thin CLI over knowlytix ``rag.eval.benchmark_query_parse`` (the generic parse->bind
comparison). This script only builds the parsers to compare -- the deployed store's
geometric parser and a local Qwen LLM parser at each model size -- and hands them to
knowlytix, which reverses gold questions from the store, binds with one shared
binder and scores recovery + latency.

We default ``query_parse_mode`` to geometric because a small instruct model emits
attribute-as-head / generic relations that fail to bind; this quantifies that with
real models and shows whether 7B closes the gap. Note: explicit, gold-anchored
questions contain the relation words, the LLM's favorable case.

Usage:
  python scripts/benchmark_query_parse_llms.py            # geometric + 3B + 7B
  python scripts/benchmark_query_parse_llms.py --n 12     # quick smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=_DEFAULT_MODELS,
                    help="HF model ids for the LLM query parser (default: Qwen 3B + 7B)")
    ap.add_argument("--n", type=int, default=0, help="cap questions (0 = all facts)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(_REPO / "data" / "benchmark_query_parse.json"))
    args = ap.parse_args()

    import torch
    from agentlab.capstone.policy_rag import PolicyRagRetriever
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    from knowlytix.knowledge.rag import benchmark_query_parse
    from knowlytix.knowledge.rag.query_triples import (
        GeometricQueryParser, QueryTripleExtractor, schema_from_store)

    # Load the deployed store + its tuned encoder + the embedding binder as the
    # agent does; the binder is held constant inside benchmark_query_parse so only
    # the parser varies.
    print("loading deployed policy store ...")
    r = PolicyRagRetriever()
    store, binder = r.store, r.pipe.binder
    vocab = schema_from_store(store)

    # The parsers to compare: the agent's geometric parser + a Qwen LLM parser per
    # model size (same vocab-grounded prompt).
    parsers = {"geometric": r.pipe.extractor}
    assert isinstance(parsers["geometric"], GeometricQueryParser)
    llms = []  # keep refs alive for the run
    for model in args.models:
        llm = LocalTransformersBackend(model, device=args.device)
        llms.append(llm)
        parsers[f"llm:{model}"] = QueryTripleExtractor(llm, vocab=vocab)

    payload = benchmark_query_parse(store, parsers, binder=binder, n=args.n)
    payload["store"] = Path(store.config.store_path).name
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n=== query-parse benchmark ({payload['n_questions']} questions) ===")
    hdr = f"{'parser':<34}{'parse':>8}{'bind':>8}{'recover':>9}{'lat(ms)':>10}"
    print(hdr); print("-" * len(hdr))
    for name, m in payload["results"].items():
        print(f"{name:<34}{m['parse_rate']:>8.2f}{m['bind_rate']:>8.2f}"
              f"{m['recover_rate']:>9.2f}{m['latency_ms_mean']:>10.1f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
