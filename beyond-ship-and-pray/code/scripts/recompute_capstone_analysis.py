"""Recompute the capstone analysis from the pinned per-row table -- WITHOUT
re-running the agent.

`capstone_run.py` persists every run's per-row table to `data/capstone_rows.json`.
After changing the attribution or scoring math (e.g. how per-component failures are
reported), run this to rebuild every row-derived block -- per-tool, per-component
attribution, the overall conjunction, weak-link, balances, adherence -- and merge in
the deterministic gate stack + audit flag preserved from the existing
`capstone_run.json`. Cheap: no model, no store, no 120 agent passes.

    PYTHONPATH=. python scripts/recompute_capstone_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capstone_run import build_row_analysis, OUT, ROWS  # noqa: E402


def main() -> int:
    if not ROWS.exists():
        raise SystemExit(f"{ROWS} not found -- run scripts/capstone_run.py once to pin it")
    data = json.loads(ROWS.read_text())
    rows, factors = data["rows"], data["factors"]
    analysis = build_row_analysis(rows, factors)

    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    result = {
        "n_runs": len(rows),
        "workflow_adherence": analysis["workflow_adherence"],
        "audit_verifies": prev.get("audit_verifies"),   # deterministic; from the live run
        "gate_stack": prev.get("gate_stack"),            # deterministic; from the live run
        "seed_balance": analysis["seed_balance"],
        "clarity_balance": analysis["clarity_balance"],
        "attribution_by_component": analysis["attribution_by_component"],
        "overall": analysis["overall"],
        "weak_link": analysis["weak_link"],
        "per_tool": analysis["per_tool"],
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"recomputed {OUT.name} from {ROWS.name} ({len(rows)} rows); "
          f"gate_stack + audit preserved from the live run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
