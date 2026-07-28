#!/usr/bin/env python
"""Run the banking complaint agent against the synthetic eval set.

Usage:
    python scripts/run_complaint_agent.py
    python scripts/run_complaint_agent.py --limit 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentlab.capstone import build_complaint_harness
from agentlab.core import Budget, BudgetTracker, TaskSpec
from agentlab.evaluation import summarize


def load_json(path: Path):
    return json.loads(path.read_text())


def did_escalate(traj) -> bool:
    if traj.final_state.status == "escalated":
        return True
    if traj.final_state.status == "failed":
        return True
    output = traj.final_state.final_output or {}
    if isinstance(output, dict) and output.get("recommended_action") == "escalate":
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/complaint_agent.json")
    parser.add_argument("--policies", default=None, help="override policies_dir from config")
    parser.add_argument("--cases", default=None, help="override cases path from config")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 1
    config = load_json(config_path)

    policies_dir = Path(args.policies or config.get("policies_dir", "data/policies"))
    cases_path = Path(args.cases or config.get("eval_cases", "data/eval_cases/cases.json"))
    cases = load_json(cases_path)
    if args.limit:
        cases = cases[: args.limit]

    harness, _ = build_complaint_harness(policies_dir=policies_dir)
    budget_cfg = config.get("budget", {})
    max_steps = int(config.get("max_steps", 16))

    # Warm up the tool models once (untimed) so the per-request budgets below are
    # not billed for cold model loading. Otherwise the FIRST case alone loads ~5
    # models, exceeds the seconds budget mid-load, and is recorded as a failure --
    # a harness artifact, not an agent error (production loads models at startup).
    warm = TaskSpec(goal="handle complaint",
                    inputs={"message": "I was charged a fee and would like it reviewed."})
    harness.run(warm, max_steps=max_steps, budget_tracker=BudgetTracker(Budget()))

    classify_correct = 0
    escalation_correct = 0
    total_steps = 0
    for case in cases:
        task = TaskSpec(goal="handle complaint", inputs={"message": case["message"]})
        tracker = BudgetTracker(Budget(**budget_cfg))
        traj = harness.run(task, max_steps=max_steps, budget_tracker=tracker)
        summary = summarize(traj)
        output = traj.final_state.final_output or {}
        actual_class = output.get("classification") if isinstance(output, dict) else None
        if actual_class is None:
            for rec in traj.records:
                if rec.action.kind == "tool_call" and rec.observation.get("success"):
                    out = rec.observation.get("output") or {}
                    if isinstance(out, dict) and "category" in out:
                        actual_class = out["category"]
                        break
        actual_escalated = did_escalate(traj)

        if actual_class == case.get("expected_classification"):
            classify_correct += 1
        if actual_escalated == case.get("expected_escalation"):
            escalation_correct += 1
        total_steps += summary["steps"]

        if args.verbose:
            print(
                f"[{case['id']}] status={traj.final_state.status} "
                f"class={actual_class} escalated={actual_escalated} steps={summary['steps']}"
            )

    n = len(cases)
    if n == 0:
        print("no cases to evaluate")
        return 0
    print(f"\nResults across {n} cases")
    print(f"  classification accuracy: {classify_correct}/{n} ({100*classify_correct/n:.0f}%)")
    print(f"  escalation accuracy:     {escalation_correct}/{n} ({100*escalation_correct/n:.0f}%)")
    print(f"  total steps:             {total_steps}")
    print(f"  audit chain verifies:    {harness.audit.verify()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
