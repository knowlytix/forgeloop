"""Capture the Response Draft groundedness component test and pin it for
notebook 12.5 (component testing).

The draft component is tested by groundedness-as-distance: the draft's verifiable
dollar claim is aligned to a store triple and scored by geodesic distance, then
mapped to a tier (grounded / distortion / fabrication) by per-relation bands
CALIBRATED FROM THE STORE itself. This script derives those bands and scores a
handful of representative drafts -- a grounded one, a distorted amount and a
fabricated reversal cap -- so the notebook reads the result rather than loading a
model and a store. Writes `draft_groundedness` into `data/capstone_companions.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "data" / "capstone_companions.json").exists())
sys.path.insert(0, str(ROOT.parent if (ROOT.parent / "agentlab").exists() else ROOT))

from agentlab.testing.capstone_harness import CapstoneTestHarness  # noqa: E402

# Representative drafts. Each carries one store-checkable dollar claim; the tier is
# decided by the store, not by this list.
DRAFTS = [
    ("grounded overdraft fee",
     "Under our overdraft policy a $35 fee applies to the transaction.",
     "overdraft_fee"),
    ("distorted overdraft fee",
     "Under our overdraft policy a $50 fee applies to the transaction.",
     "overdraft_fee"),
    ("grounded reversal cap",
     "A representative can approve a goodwill reversal of up to $35.",
     None),
    ("fabricated reversal cap",
     "A representative can approve a goodwill reversal of up to $100.",
     None),
]


def main() -> None:
    h = CapstoneTestHarness()
    bands = h.calibrate_groundedness()   # per-relation, derived from the store

    examples = []
    for label, draft, issue in DRAFTS:
        triple, geodesic, tier = h.judge_draft(draft, issue)
        examples.append({
            "label": label,
            "draft": draft,
            "triple": triple,
            "geodesic": round(float(geodesic), 4) if geodesic is not None else None,
            "tier": tier,
        })
        print(f"{label:24s} {triple:38s} d={geodesic!s:8.8} -> {tier}")

    out = {
        "bands": bands,
        "examples": examples,
        "_note": "Response Draft groundedness for 12.5: store-calibrated per-relation "
                 "bands + judge_draft on representative drafts. Bands are deterministic "
                 "from the store; built by scripts/capture_draft_groundedness.py.",
    }

    comp_path = ROOT / "data" / "capstone_companions.json"
    C = json.loads(comp_path.read_text())
    C["draft_groundedness"] = out
    comp_path.write_text(json.dumps(C, indent=2) + "\n")
    print("wrote draft_groundedness into", comp_path)


if __name__ == "__main__":
    main()
