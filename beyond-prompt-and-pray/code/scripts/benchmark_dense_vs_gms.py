#!/usr/bin/env python
"""Benchmark: conventional dense-embedding RAG vs the GMS/GEODE retriever.

Both systems answer the SAME questions over the SAME policy corpus with the SAME
LLM (Qwen2.5-3B-Instruct); both are scored by the SAME oracle
(``CapstoneTestHarness.rag_test`` -- questions reversed from the GEODE graph,
answers decomposed into typed claims and verified against the GMS). The only
variable is the retrieval engine:

  * ``gms``          -- ``PolicyRagRetriever``: triple-mediated GEODE retrieval
                        with calibrated cap / tension / relevance gates and
                        abstention (the system under comparison).
  * ``dense:fixed``  -- ``DenseRagRetriever``: frozen all-MiniLM-L6-v2 over
                        fixed-size char windows, cosine top-k, always answers.
  * ``dense:section``-- same, chunked per markdown ``##`` section.

So a gap between ``gms`` and the dense rows is attributable to triple-mediation +
the geometric gates, not to a different corpus, chunker LLM or question set.

Run::

    python scripts/benchmark_dense_vs_gms.py
    python scripts/benchmark_dense_vs_gms.py --n-runs 3 --systems gms dense:section
    python scripts/benchmark_dense_vs_gms.py --out data/dense_vs_gms_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agentlab.testing import CapstoneTestHarness

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_JSON = _REPO_ROOT / "data" / "dense_vs_gms_benchmark.json"

_ALL_SYSTEMS = ("gms", "dense:fixed", "dense:section")


def _make_retriever(system: str):
    """Build the retriever for one system label. ``gms`` returns None so
    ``rag_test`` constructs its default ``PolicyRagRetriever``."""
    if system == "gms":
        return None
    if system == "dense:fixed":
        from agentlab.capstone.dense_rag import DenseRagRetriever
        return DenseRagRetriever(chunking="fixed", chunk_size=512, chunk_overlap=64)
    if system == "dense:section":
        from agentlab.capstone.dense_rag import DenseRagRetriever
        return DenseRagRetriever(chunking="section")
    raise ValueError(f"unknown system {system!r}; choose from {_ALL_SYSTEMS}")


def _row(system: str, res) -> dict:
    d = asdict(res)
    correctness = (res.claims_verified / res.n_claims) if res.n_claims else 0.0
    return {"system": system, "correctness": round(correctness, 3), **d}


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("system", "system", 14),
        ("n_questions", "n_q", 5),
        ("n_claims", "claims", 7),
        ("claims_verified", "verif", 6),
        ("correctness", "correct", 8),
        ("mean_completeness", "complete", 9),
        ("mean_confidence", "conf", 6),
    ]
    header = "  ".join(label.ljust(w) for _, label, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(key, "")).ljust(w) for key, _, w in cols))
    print("\nfailure codes per system:")
    for r in rows:
        print(f"  {r['system']:<14} {r.get('failure_codes', {})}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=list(_ALL_SYSTEMS),
                        help=f"systems to benchmark (default: {' '.join(_ALL_SYSTEMS)})")
    parser.add_argument("--n-runs", type=int, default=2,
                        help="DoE replications per generator (default 2)")
    parser.add_argument("--max-per-category", type=int, default=3,
                        help="questions per generator category (default 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(_OUT_JSON))
    args = parser.parse_args()

    rows: list[dict] = []
    for system in args.systems:
        print(f"\n=== benchmarking: {system} ===", flush=True)
        # One harness per system, same seed -> identical question set across runs.
        harness = CapstoneTestHarness(seed=args.seed)
        res = harness.rag_test(
            n_runs=args.n_runs,
            max_per_category=args.max_per_category,
            retriever=_make_retriever(system),
        )
        rows.append(_row(system, res))
        print(f"    correctness={rows[-1]['correctness']} "
              f"completeness={res.mean_completeness} "
              f"verified={res.claims_verified}/{res.n_claims}", flush=True)

    _print_table(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "n_runs": args.n_runs,
                               "results": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
