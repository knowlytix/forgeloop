#!/usr/bin/env python
"""Parity + GPU-correctness check for the product-augmented GEODE store (v2).

Runs the authoritative `rag_test` oracle with two retrievers -- the live
baseline store and v2 -- against the SAME oracle (built from the policy doc), so
the only variable is the store. If the core policy retrieval verifies identically
(claims_verified / n_claims / failure_codes), then (a) adding the product layer
did not perturb search_policy, and (b) the GPU-trained v2 geometry matches the
CPU-built baseline -> GB10 GPU compute is sound for this build.
"""
from __future__ import annotations

import json
from pathlib import Path

from agentlab.testing import CapstoneTestHarness
from agentlab.capstone.policy_rag import PolicyRagRetriever

_REPO = Path(__file__).resolve().parents[1]
_V2 = _REPO / "data" / "gms_policy_store_geode_v2"


def _summ(r):
    return {
        "n_claims": r.n_claims,
        "claims_verified": r.claims_verified,
        "verified_rate": round(r.claims_verified / max(r.n_claims, 1), 4),
        "completeness": round(getattr(r, "completeness", float("nan")), 4),
        "failure_codes": dict(getattr(r, "failure_codes", {}) or {}),
    }


def main() -> int:
    h = CapstoneTestHarness(n_runs=2, seed=42)
    print("=== BASELINE (live store) ===")
    base = h.rag_test()  # default PolicyRagRetriever on the live store
    print(json.dumps(_summ(base), indent=2))

    print("\n=== V2 (product-augmented, GPU-built) ===")
    v2 = h.rag_test(retriever=PolicyRagRetriever(store_path=_V2))
    print(json.dumps(_summ(v2), indent=2))

    bs, vs = _summ(base), _summ(v2)
    parity = (bs["n_claims"] == vs["n_claims"]
              and bs["claims_verified"] == vs["claims_verified"]
              and bs["failure_codes"] == vs["failure_codes"])
    print(f"\nPARITY: {'PASS' if parity else 'DIFFER'} "
          f"(baseline {bs['claims_verified']}/{bs['n_claims']} vs "
          f"v2 {vs['claims_verified']}/{vs['n_claims']})")
    (_REPO / "data" / "parity_check_v2.json").write_text(
        json.dumps({"baseline": bs, "v2": vs, "parity": parity}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
