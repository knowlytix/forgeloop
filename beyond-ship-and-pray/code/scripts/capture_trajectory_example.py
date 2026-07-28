"""Capture a few real capstone-agent trajectories and pin them for the Chapter 9
companion notebook (`notebooks/09_agentic_capstone.ipynb`).

The notebook reads the pinned artifact rather than running the agent, exactly as
the Chapter 12 section notebooks read `capstone_run.json`. This script is the
"run separately" path that produced it: it drives the REAL governed agent
(`build_complaint_harness`) on two representative cases and records, per step, the
tool call, its success and the gate decisions -- the trajectory as the unit of
observation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "data" / "eval_cases" / "cases.json").exists())
TUT = next(p for p in [Path.cwd(), *Path.cwd().parents]
           if (p / "apps" / "complaint_sut.py").exists())
sys.path.insert(0, str(TUT))
sys.path.insert(0, str(ROOT))

from apps.complaint_sut import AgentSUT  # noqa: E402
from agentlab.testing.capstone_harness import (  # noqa: E402
    _tool_sequence, did_escalate, _classification_of, _escalation_trigger,
)


def step_view(traj) -> list[dict]:
    """One row per tool call: the tool, whether it succeeded, and any gate that
    did not ALLOW the call (decision + which gate)."""
    rows = []
    for r in traj.records:
        if r.action.kind != "tool_call":
            continue
        obs = r.observation or {}
        gates = [
            {"gate": g.get("gate"), "decision": g.get("decision")}
            for g in obs.get("gate_results", [])
            if g.get("decision") and g.get("decision") != "allow"
        ]
        rows.append({
            "tool": r.action.tool_name,
            "ok": bool(obs.get("success", True)),
            "gates_not_allowed": gates,
        })
    return rows


def main() -> None:
    cases = {c["id"]: c for c in json.loads((ROOT / "data" / "eval_cases" / "cases.json").read_text())}
    picks = ["case-001", "case-004"]   # complaint->draft, complaint->escalate

    sut = AgentSUT()
    out = []
    for cid in picks:
        case = cases[cid]
        msg = case.get("message") or case.get("query")
        traj = sut.run_traj(msg)
        res = sut.result_from_traj(traj)
        out.append({
            "case": cid,
            "message": msg,
            "expected_classification": case.get("expected_classification"),
            "expected_escalate": bool(case.get("expected_escalate",
                                               case.get("expected_escalation", False))),
            "sut_result": {
                "classification": res.components.get("classification"),
                "status": res.status,
                "escalation_trigger": res.escalation_trigger,
                "tool_sequence": _tool_sequence(traj),
                "answer_excerpt": (res.answer or "")[:240],
            },
            "steps": step_view(traj),
        })
        print(f"{cid}: status={res.status} class={res.components.get('classification')} "
              f"seq={'>'.join(_tool_sequence(traj))} trigger={res.escalation_trigger!r}")

    dest = TUT / "data" / "capstone_trajectory_example.json"
    dest.write_text(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
