#!/usr/bin/env python
"""Protocol 2: the shipped strict per-tool number for the GMS-based RAG.

Runs the governed complaint agent over the DoE design (`harness.run()`) and
decomposes per-tool correctness (`harness.tool_breakdown()`) -- the established
protocol, not a hand-rolled scoring loop. `search_policy` is scored strictly:
`expected_policy in returned ids`. Comparing this to protocol 1's claim-
verification correctness (rag_test, ~1.0) tells us how much of the low shipped
search_policy number is strict-scorer divergence vs real enrichment collapse.

Writes to a fresh JSON (not the shipped capstone_tool_breakdown.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentlab.testing import CapstoneTestHarness

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT = _REPO_ROOT / "data" / "gms_rag_protocol2_breakdown.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=120,
                   help="design size (shipped scale = 120)")
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N scenarios (smoke)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(_OUT))
    args = p.parse_args()

    harness = CapstoneTestHarness(n_runs=args.n_runs, seed=args.seed,
                                  rephrase_method="qwen")
    result = harness.run(limit=args.limit)
    breakdown = harness.tool_breakdown(result)

    print(f"\nran {result.n_runs} scenarios")
    print(f"  outcome accuracy:    {result.summary['accuracy']:.3f}")
    print(f"  trajectory accuracy: {result.summary['trajectory_accuracy']:.3f}")
    print("\nper-tool correctness (strict; scored where reached + GT defined):")
    for tool, s in breakdown["per_tool"].items():
        acc = f"{s['accuracy']:.4f}" if s["accuracy"] is not None else "n/a"
        print(f"   {tool:<18} accuracy={acc} ({s['correct']}/{s['scored']})")
    print(f"  weak-link blame: {breakdown['weak_link_counts']}")

    out = Path(args.out)
    out.write_text(json.dumps({
        "n_runs": result.n_runs,
        "summary": result.summary,
        "tool_breakdown": breakdown,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
