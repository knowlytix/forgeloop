#!/usr/bin/env python
"""Capstone DoE driver: design a complaint suite, run the Ch15 agent, judge, persist.

DEPRECATED as the testing-chapter driver. The Testing-the-Capstone-Agent chapter and
its notebook now apply the unified ``gmstest`` framework via ``apps.complaint_sut``
(gms-testing-tutorial): the campaign runs through ``scripts/capstone_run.py`` and the
chapter notebook is built by ``scripts/build_nb_16_testing_agents.py`` from the pinned
artifacts (data/capstone_run.json, capstone_retrieval.json, capstone_companions.json).
This ad-hoc harness is retained only for the dev scripts and tests that still import it.

Pipeline (mirrors the testing chapter's coverage / judgment / attribution):

  1. Design a factor-balanced suite of complaint scenarios over three
     presentation factors (clarity, entity_aliasing, reasoning_cue) crossed
     with the 20 labeled seed cases (a blocking factor), via knowlytix
     ``DesignMatrix``.
  2. Materialize each design row into a concrete complaint message
     (deterministic templated transforms; ``--rephrase qwen`` for richer phrasing).
  3. Run each scenario through the real governed complaint agent
     (``build_complaint_harness``) and collect the trajectory outcome.
  4. Judge: binary correctness vs the case labels (classification + escalation)
     and a per-draft groundedness tier scored against the GMS substrate
     (``score_triple``).
  5. Attribute residual failures to factor levels (logistic regression with
     BH-corrected p-values) and persist one CSV the Ch16 notebook and chapter
     prose load deterministically.

Run end-to-end (regenerates the book's CSV)::

    python scripts/run_capstone_doe.py

Fast smoke run (fewer rows, model-light tools)::

    AGENTLAB_USE_LLM_EXTRACT=0 AGENTLAB_USE_LLM_FLAG=0 \
        python scripts/run_capstone_doe.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentlab.testing import CapstoneTestHarness

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_CSV = _REPO_ROOT / "data" / "capstone_doe_results.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=120,
                        help="size of the balanced design (default 120)")
    parser.add_argument("--method", default="sobol", help="sobol | lhs | grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rephrase", default="qwen",
                        choices=["template", "qwen"],
                        help="materialization method (default qwen: rich, greedy-reproducible)")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N scenarios (smoke test)")
    parser.add_argument("--out", default=str(_OUT_CSV))
    parser.add_argument("--fault-limit", type=int, default=8,
                        help="scenarios per tool for fault injection (0 to skip)")
    parser.add_argument("--skip-substrate", action="store_true",
                        help="skip the native-mode substrate test")
    parser.add_argument("--batch-materialize", action="store_true",
                        help="materialize all scenario messages in one batched "
                             "Qwen pass (faster; batched decoding yields slightly "
                             "different phrasings than the published CSV)")
    args = parser.parse_args()

    harness = CapstoneTestHarness(
        n_runs=args.n_runs,
        method=args.method,
        seed=args.seed,
        rephrase_method=args.rephrase,
    )

    bands = harness.persist_calibration()
    print(f"groundedness bands (from store): {bands}")

    print(f"designing {args.n_runs}-row complaint suite ({args.method}, seed={args.seed})...")
    result = harness.run(limit=args.limit, batch_materialize=args.batch_materialize)
    harness.to_csv(result, args.out)

    print(f"\nran {result.n_runs} scenarios")
    print(f"  outcome accuracy:      {result.summary['accuracy']:.3f}  "
          "(classification + escalation labels)")
    print(f"  trajectory accuracy:   {result.summary['trajectory_accuracy']:.3f}  "
          "(+ workflow order, clean finish, right escalation path)")
    print(f"  workflow adherence:    {result.summary['workflow_adherence']:.3f}")
    print(f"  escalation-path acc.:  {result.summary['trigger_accuracy']:.3f}")
    print(f"  audit chain verifies:  {result.summary['audit_verifies']}")
    tiers: dict[str, int] = {}
    for row in result.rows:
        tiers[row["draft_tier"]] = tiers.get(row["draft_tier"], 0) + 1
    print(f"  draft groundedness tiers: {tiers}")

    attribution = harness.analyze(result)
    print("\nfactor attribution (logistic deviance, BH-corrected):")
    for row in attribution.logistic_table:
        flag = "*" if row.get("significant_adj") else " "
        print(f" {flag} {row['factor']:<16} "
              f"p_adj={row.get('p_value_adj', row.get('p_value')):.4f}  "
              f"pseudo_r2={row.get('pseudo_r2', 0):.3f}")
    print("\ntop failure (factor, level):")
    for row in attribution.top_failures(k=5):
        print(f"   {row['factor']:<16} {row['level']:<14} "
              f"failure_rate={row['failure_rate']:.2f} (n={row['n_total']})")
    print(f"\nwrote {args.out}")

    # --- A: per-tool correctness decomposition (the weak-link analysis) ----
    breakdown = harness.tool_breakdown(result)
    print("\nper-tool correctness (scored where reached + ground truth defined):")
    for tool, s in breakdown["per_tool"].items():
        acc = f"{s['accuracy']:.2f}" if s["accuracy"] is not None else "n/a"
        print(f"   {tool:<18} accuracy={acc} ({s['correct']}/{s['scored']})")
    print(f"  weak-link blame (failures attributed to first erring tool): {breakdown['weak_link_counts']}")
    (_REPO_ROOT / "data" / "capstone_tool_breakdown.json").write_text(
        json.dumps(breakdown, indent=2) + "\n")

    # --- B: per-tool fault injection via the real knowlytix ToolGateway ----
    if args.fault_limit > 0:
        print(f"\nfault injection (real knowlytix ToolGateway, {args.fault_limit} scenarios/tool):")
        fault = harness.fault_injection(fault="error", limit=args.fault_limit)
        for tool, s in fault.per_tool.items():
            rate = f"{s['detection_rate']:.2f}" if s["detection_rate"] is not None else "n/a"
            print(f"   {tool:<18} detected={s['detected']}/{s['reached']} reached "
                  f"(silent={s['silent']}) detection_rate={rate}")
        (_REPO_ROOT / "data" / "capstone_fault_injection.json").write_text(
            json.dumps(fault.per_tool, indent=2) + "\n")

    # --- Substrate test: full knowlytix platform, native mode (ship/no-ship)
    if not args.skip_substrate:
        print("\nnative-mode substrate test (DOEGMSBenchmark + Qwen RAG + release gate):")
        sub = harness.substrate_test()
        print(f"   baseline acc (ground truth): {sub.baseline_accuracy}")
        print(f"   Qwen-RAG substrate acc:      {sub.evaluator_accuracy} ({sub.n_questions} q)")
        print(f"   typed verdicts: {sub.verdict_summary}")
        for g in sub.gates:
            print(f"   GATE {g['tier']:<14} passed={g['passed']} "
                  f"(acc {g['actual']} vs threshold {g['threshold']})")
        (_REPO_ROOT / "data" / "capstone_substrate_test.json").write_text(
            json.dumps({"baseline_accuracy": sub.baseline_accuracy,
                        "evaluator_accuracy": sub.evaluator_accuracy,
                        "n_questions": sub.n_questions,
                        "verdict_summary": sub.verdict_summary,
                        "gates": sub.gates}, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
