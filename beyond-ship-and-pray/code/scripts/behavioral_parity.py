#!/usr/bin/env python
"""Behavioral parity: the gmstest SCORER vs the CapstoneTestHarness scorer, on
the SAME real-agent trajectories.

Each scenario's message is run through the governed complaint agent exactly
ONCE; that single Trajectory is then scored two ways:

  reference : the harness's own logic — correct = (classification == expected)
              and (escalated == expected_escalation), via the shipped helpers.
  gmstest   : evaluate._outcome_correct on the SUTResult mapped from the SAME
              trajectory.

Scoring one trajectory with both scorers isolates *scorer parity* (the new
piece) from *agent run-to-run nondeterminism* (a property of the agent, not the
framework, and shared by both paths in production). A correct framework agrees
on every scenario.

Run from the tutorial dir with the venv active (uses GPU/Qwen):
    PYTHONPATH=. python scripts/behavioral_parity.py --limit 6
    PYTHONPATH=. python scripts/behavioral_parity.py --limit 120   # full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gmstest as gt
from gmstest.evaluate import _outcome_correct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--rephrase", default="template", choices=["template", "qwen"])
    args = ap.parse_args()

    from agentlab.testing import CapstoneTestHarness
    from agentlab.testing.capstone_harness import _classification_of, did_escalate
    from apps.complaint_sut import AgentSUT

    harness = CapstoneTestHarness(n_runs=120, seed=42, rephrase_method=args.rephrase)
    cases_by_id = {c["id"]: c for c in harness._cases}
    records = harness.design().to_dict("records")[: args.limit]
    print(f"Behavioral parity on N={len(records)} (rephrase={args.rephrase})\n")

    sut = AgentSUT()
    n = len(records)
    correct_match = esc_match = cls_match = 0
    ref_correct = gms_correct = 0

    for i, drow in enumerate(records):
        case = cases_by_id[drow["seed_case"]]
        msg = harness.materialize(drow)
        scn = gt.Scenario(
            sid=f'{case["id"]}~{i:04d}',
            base=gt.QAItem(qid=case["id"], query=msg, answer=None, answer_type="decision",
                           components={"expected_classification": case["expected_classification"],
                                       "expected_escalation": case["expected_escalation"]}),
            factor_levels={k: drow[k] for k in ("clarity", "entity_aliasing", "reasoning_cue")},
            mode="embedded")

        traj = sut.run_traj(msg)                    # ONE agent run
        res = AgentSUT.result_from_traj(traj)

        # gmstest scorer
        g_ok = bool(_outcome_correct(scn, res))
        # reference scorer (harness logic on the same trajectory)
        r_cls, r_esc = _classification_of(traj), did_escalate(traj)
        r_ok = (r_cls == case["expected_classification"]) and (r_esc == case["expected_escalation"])

        gms_correct += int(g_ok)
        ref_correct += int(r_ok)
        correct_match += int(g_ok == r_ok)
        cls_match += int(res.components["classification"] == r_cls)
        esc_match += int((res.status == "escalated") == r_esc)
        flag = "" if g_ok == r_ok else "   <-- MISMATCH"
        print(f"  {case['id']}  ref={int(r_ok)} gms={int(g_ok)} "
              f"cls={r_cls} esc={r_esc}{flag}")

    print("\n" + "=" * 60)
    print(f"reference accuracy : {ref_correct / n:.3f}")
    print(f"gmstest   accuracy : {gms_correct / n:.3f}")
    print(f"per-row outcome-scorer match   : {correct_match}/{n}")
    print(f"per-row classification match   : {cls_match}/{n}")
    print(f"per-row escalation match       : {esc_match}/{n}")
    print(f"audit verifies (agent)         : {sut.audit_verifies()}")
    print("=" * 60)
    ok = correct_match == n and cls_match == n and esc_match == n
    print("PARITY PASS" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
