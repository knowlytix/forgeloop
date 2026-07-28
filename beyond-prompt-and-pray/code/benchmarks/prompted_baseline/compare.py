#!/usr/bin/env python
"""Benchmark the engineered capstone agent against the prompted baseline.

Three agents over the same eval set (``data/eval_cases/cases.json``):

  1. engineered (Qwen)   --- agentlab.capstone.build_complaint_harness: trained
     classifier head, hybrid extractor, Graph RAG, GMS regulatory guard, LoRA
     drafter + verifier, governance gates.
  2. prompted (Qwen-3B)  --- the prompt-and-pray baseline on the *same* local
     model, so the only difference from (1) is the engineering.
  3. prompted (Sonnet)   --- the same baseline on a frontier model, so the only
     difference from (2) is model size.

Each is scored two ways:

  - Deterministic, against the ground-truth labels in cases.json:
    classification accuracy and escalation accuracy.
  - LLM-as-judge (default: Sonnet), on the drafted reply: a 1-5 quality score
    and safety flags (unauthorized fee-waiver promise, off-topic, empty).

We also report the judge's *own* escalation accuracy against ground truth, to
show that an LLM judge is itself a prompted model with the same blind spots ---
not a replacement for the engineered guard.

Run from the repo root (so `agentlab` and `benchmarks` import):

    python -m benchmarks.prompted_baseline.compare
    python -m benchmarks.prompted_baseline.compare --limit 5 --verbose
    python -m benchmarks.prompted_baseline.compare --no-sonnet   # Qwen only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agentlab.capstone import build_complaint_harness
from agentlab.core import TaskSpec
from agentlab.evaluation import summarize

from benchmarks.prompted_baseline.chat_backends import (
    AnthropicChatBackend,
    QwenChatBackend,
)
from benchmarks.prompted_baseline.judge import LLMJudge
from benchmarks.prompted_baseline.prompted_agent import build_prompted_harness

REPO_ROOT = Path(__file__).resolve().parents[2]


def did_escalate(traj) -> bool:
    if traj.final_state.status in ("escalated", "failed"):
        return True
    output = traj.final_state.final_output or {}
    return isinstance(output, dict) and output.get("recommended_action") == "escalate"


def extract_classification(traj) -> str | None:
    output = traj.final_state.final_output or {}
    if isinstance(output, dict) and output.get("classification") is not None:
        return output.get("classification")
    # Escalated runs may not compile a final output; recover from the trajectory.
    for rec in traj.records:
        if rec.action.kind == "tool_call" and rec.observation.get("success"):
            out = rec.observation.get("output") or {}
            if isinstance(out, dict) and "category" in out:
                return out["category"]
    return None


def extract_draft(traj) -> str:
    output = traj.final_state.final_output or {}
    if isinstance(output, dict):
        return output.get("draft_response", "") or ""
    return ""


# Cases whose correct handling is to escalate at the *input gate* (PII / prompt
# injection) before any classification. The engineered agent screens these out
# and never classifies them --- that is correct behavior, not a classification
# miss --- so classification accuracy is measured only over the classifiable
# cases. The escalation metric (over all cases) is what credits catching them.
_GATE_REGULATORY = {"PII", "prompt_injection"}


def is_classifiable(case) -> bool:
    return case.get("factors", {}).get("regulatory") not in _GATE_REGULATORY


@dataclass
class AgentScore:
    label: str
    n: int = 0
    classify_total: int = 0
    classify_correct: int = 0
    escalate_correct: int = 0
    steps: int = 0
    audit_ok: bool = True
    draft_scores: list[int] = field(default_factory=list)
    unauthorized_promises: int = 0
    off_topic: int = 0
    empty_drafts: int = 0
    drafts_judged: int = 0

    def add_draft(self, verdict) -> None:
        self.draft_scores.append(verdict.score)
        self.drafts_judged += 1
        self.unauthorized_promises += int(verdict.unauthorized_promise)
        self.off_topic += int(verdict.off_topic)
        self.empty_drafts += int(verdict.empty)

    @property
    def classify_acc(self) -> float:
        return self.classify_correct / self.classify_total if self.classify_total else 0.0

    @property
    def escalate_acc(self) -> float:
        return self.escalate_correct / self.n if self.n else 0.0

    @property
    def avg_draft(self) -> float:
        return sum(self.draft_scores) / len(self.draft_scores) if self.draft_scores else 0.0


def run_agent(label, harness, cases, judge, max_steps, verbose) -> AgentScore:
    score = AgentScore(label=label, n=len(cases))
    for case in cases:
        task = TaskSpec(goal="handle complaint", inputs={"message": case["message"]})
        traj = harness.run(task, max_steps=max_steps)
        summary = summarize(traj)

        actual_class = extract_classification(traj)
        actual_escalate = did_escalate(traj)
        # Classification is scored only on cases that legitimately reach the
        # classifier (not screened out at the input gate).
        if is_classifiable(case):
            score.classify_total += 1
            if actual_class == case.get("expected_classification"):
                score.classify_correct += 1
        if actual_escalate == case.get("expected_escalation"):
            score.escalate_correct += 1
        score.steps += summary["steps"]

        # Judge the draft only when the agent actually produced one (responded).
        draft = extract_draft(traj)
        if not actual_escalate and judge is not None:
            verdict = judge.judge_draft(case["message"], draft, case.get("factors", {}).get("regulatory", ""))
            score.add_draft(verdict)

        if verbose:
            mark = "ok " if actual_escalate == case.get("expected_escalation") else "MISS"
            print(
                f"  [{label}] {case['id']} {mark} class={actual_class} "
                f"escalate={actual_escalate} (gold={case.get('expected_escalation')}) "
                f"steps={summary['steps']}"
            )
    score.audit_ok = harness.audit.verify()
    return score


def run_judge_escalation(cases, judge) -> tuple[int, int]:
    """The judge's own escalation verdicts vs ground truth (agent-independent)."""
    correct = 0
    for case in cases:
        v = judge.judge_escalation(case["message"])
        if v.should_escalate == case.get("expected_escalation"):
            correct += 1
    return correct, len(cases)


def print_table(scores: list[AgentScore], judge_name: str | None) -> None:
    print("\n" + "=" * 78)
    print("RESULTS  (deterministic accuracy vs. ground-truth labels)")
    print("=" * 78)
    header = f"{'agent':24s} {'class':>11s} {'escal':>7s} {'steps':>6s} {'audit':>6s}"
    print(header)
    print("-" * 78)
    for s in scores:
        cls = f"{s.classify_acc*100:.0f}% ({s.classify_correct}/{s.classify_total})"
        print(
            f"{s.label:24s} {cls:>11s} {s.escalate_acc*100:6.0f}% "
            f"{s.steps:6d} {('ok' if s.audit_ok else 'FAIL'):>6s}"
        )
    print("\n  class = accuracy over classifiable cases only (PII / prompt-injection")
    print("  cases are correctly screened at the input gate before classification).")
    print("  escal = escalation accuracy over all cases.")
    if judge_name:
        print("\n" + "=" * 78)
        print(f"LLM-AS-JUDGE  (judge = {judge_name}; drafts scored only on responded cases)")
        print("=" * 78)
        jh = f"{'agent':24s} {'n':>4s} {'avg':>5s} {'promise':>8s} {'offtop':>7s} {'empty':>6s}"
        print(jh)
        print("-" * 78)
        for s in scores:
            print(
                f"{s.label:24s} {s.drafts_judged:4d} {s.avg_draft:5.2f} "
                f"{s.unauthorized_promises:8d} {s.off_topic:7d} {s.empty_drafts:6d}"
            )
        print("\n  avg = mean draft quality 1-5 | promise = unauthorized fee-waiver/refund")
        print("  offtop = off-topic replies | empty = blank/nonsensical replies")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(REPO_ROOT / "data" / "eval_cases" / "cases.json"))
    parser.add_argument("--policies", default=str(REPO_ROOT / "data" / "policies"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--no-engineered", action="store_true", help="skip the engineered agent")
    parser.add_argument("--no-sonnet", action="store_true", help="skip the Sonnet baseline")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    parser.add_argument("--judge", choices=["sonnet", "qwen"], default="sonnet",
                        help="judge backend (falls back to qwen if Sonnet unavailable)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    if args.limit:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases from {args.cases}")

    # One shared local Qwen backend (also the engineered agent's internal model).
    qwen = QwenChatBackend.shared()
    sonnet = None if args.no_sonnet else AnthropicChatBackend.from_env()
    if not args.no_sonnet and sonnet is None:
        print("NOTE: Sonnet unavailable (no ANTHROPIC_API_KEY or SDK error); "
              "skipping the Sonnet baseline.")

    # Judge selection.
    judge = None
    judge_name = None
    if not args.no_judge:
        if args.judge == "sonnet" and sonnet is not None:
            judge = LLMJudge(sonnet)
        elif args.judge == "sonnet" and sonnet is None:
            print("NOTE: requested Sonnet judge but it is unavailable; "
                  "falling back to the Qwen-3B judge (weaker, expect noisier verdicts).")
            judge = LLMJudge(qwen)
        else:
            judge = LLMJudge(qwen)
        judge_name = judge.name

    scores: list[AgentScore] = []

    if not args.no_engineered:
        print("\nRunning engineered agent (Qwen + GMS + guards)...")
        eng_harness, _ = build_complaint_harness(policies_dir=args.policies)
        scores.append(run_agent("engineered (qwen)", eng_harness, cases, judge,
                                 args.max_steps, args.verbose))

    print("\nRunning prompted baseline (Qwen-3B)...")
    pq_harness, _ = build_prompted_harness(qwen)
    scores.append(run_agent("prompted (qwen-3b)", pq_harness, cases, judge,
                            args.max_steps, args.verbose))

    if sonnet is not None:
        print("\nRunning prompted baseline (Sonnet)...")
        ps_harness, _ = build_prompted_harness(sonnet)
        scores.append(run_agent("prompted (sonnet)", ps_harness, cases, judge,
                                args.max_steps, args.verbose))

    print_table(scores, judge_name)

    # The judge's own escalation accuracy vs ground truth (agent-independent).
    if judge is not None:
        jc, jn = run_judge_escalation(cases, judge)
        print("\n" + "=" * 78)
        print("JUDGE SANITY CHECK  (can the judge itself recover the escalation label?)")
        print("=" * 78)
        print(f"  LLM judge ({judge_name}) escalation accuracy vs ground truth: "
              f"{jc}/{jn} ({100*jc/jn:.0f}%)")
        print("  -> an LLM judge is another prompted model; compare this to the")
        print("     engineered agent's escalation accuracy above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
