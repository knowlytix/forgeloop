"""Materialize the Chapter 16 DoE clarity test set once, to a jsonl.

This is the external, held-out evaluation set for the complaint classifier: the
capstone DoE suite (seeded from data/eval_cases, NOT from the training seeds, so
there is no leakage), with the clarity factor realized by Qwen exactly as the
Ch16 testing harness does. Each row is {message, label, clarity, seed_case};
rows whose seed case has no expected_classification are skipped.

    python scripts/make_doe_eval_set.py --n-runs 120 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_OUT = Path(__file__).resolve().parents[1] / "data" / "eval_cases" / "doe_clarity_eval.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default="sobol")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    from agentlab.testing import CapstoneTestHarness

    h = CapstoneTestHarness(n_runs=args.n_runs, method=args.method, seed=args.seed,
                            rephrase_method="qwen")
    rows = h.design().to_dict("records")
    out = []
    for i, row in enumerate(rows):
        case = h._cases_by_id[row["seed_case"]]
        gold = case.get("expected_classification")
        if not gold:
            continue
        msg = h.materialize(row)
        out.append({"message": msg, "label": gold,
                    "clarity": row["clarity"], "seed_case": row["seed_case"]})
        if (i + 1) % 20 == 0:
            print(f"  materialized {i + 1}/{len(rows)}", flush=True)

    Path(args.out).write_text("\n".join(json.dumps(r) for r in out) + "\n")
    import collections
    print(f"wrote {len(out)} DoE eval rows -> {args.out}")
    print(f"  by clarity: {dict(collections.Counter(r['clarity'] for r in out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
