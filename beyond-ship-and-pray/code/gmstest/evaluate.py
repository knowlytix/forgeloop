"""Test / evaluation consumer (Chapter 16).

Runs composed scenarios through a System Under Test (SUT) and scores the run at
three levels — generalizing CapstoneTestHarness:

  Outcome     did the final answer / labeled decisions match ground truth?
  Trajectory  did the workflow run in order and escalate by the right path?
  Process     step count, tool calls, failures, clean termination.

Plus optional groundedness (a draft's claim scored by a judge_fn, e.g.
store.score_triple) and DoE factor attribution via knowlytix DOEAnalyzer.

A SUT is any callable `sut(query, context="") -> answer | SUTResult`. Returning
a bare value scores Outcome only; returning a SUTResult unlocks trajectory and
per-component scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .compose import Scenario


@dataclass
class SUTResult:
    answer: Any = None
    components: dict[str, Any] = field(default_factory=dict)   # tool -> predicted
    trajectory: list[dict[str, Any]] = field(default_factory=list)  # ordered steps
    status: str = "ok"                                          # ok|escalated|failed
    escalation_trigger: str | None = None


JudgeFn = Callable[[str], tuple[str, float]]  # draft -> (tier, score)


def _as_result(raw: Any) -> SUTResult:
    return raw if isinstance(raw, SUTResult) else SUTResult(answer=raw)


def _outcome_correct(scn: Scenario, res: SUTResult) -> bool:
    comps = scn.base.components
    # Labeled seed-case style: classification AND escalation must both match.
    if "expected_classification" in comps:
        cls_ok = res.components.get("classification") == comps["expected_classification"]
        esc_key = "expected_escalation" if "expected_escalation" in comps else "expected_escalate"
        if esc_key in comps:
            esc = res.status == "escalated"
            esc_ok = esc == bool(comps[esc_key])
            return cls_ok and esc_ok
        return cls_ok
    # Generic: compare the answer to the base ground truth.
    if scn.base.answer is not None:
        return res.answer == scn.base.answer
    return False


def run(
    scenarios: list[Scenario],
    sut: Callable[..., Any],
    judge_fn: JudgeFn | None = None,
    workflow_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute every scenario and return one result row per scenario."""
    rows = []
    for scn in scenarios:
        res = _as_result(sut(scn.base.query, context=scn.base.metadata.get("context", "")))
        row: dict[str, Any] = {
            "sid": scn.sid,
            "seed": scn.base.qid,
            "base": scn.base.base,
            **{f"f_{k}": v for k, v in scn.factor_levels.items()},
            "correct": int(_outcome_correct(scn, res)),
            "status": res.status,
        }
        # Trajectory structure.
        if res.trajectory:
            tools = [s.get("tool") for s in res.trajectory]
            row["tool_sequence"] = ">".join(t for t in tools if t)
            row["n_steps"] = len(res.trajectory)
            row["tool_failures"] = sum(1 for s in res.trajectory if s.get("ok") is False)
            if workflow_order:
                seen = [t for t in tools if t in workflow_order]
                expected = [t for t in workflow_order if t in seen]
                row["workflow_adherent"] = int(seen == expected)
        # Per-component correctness.
        for k, truth in scn.base.components.items():
            comp = k.replace("expected_", "")
            if comp in res.components:
                row[f"c_{comp}"] = int(res.components[comp] == truth)
        # Groundedness of a textual answer.
        if judge_fn and isinstance(res.answer, str) and res.answer:
            tier, score = judge_fn(res.answer)
            row["draft_tier"] = tier
            row["draft_score"] = score
        rows.append(row)
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    out = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / n}
    if any("workflow_adherent" in r for r in rows):
        wa = [r["workflow_adherent"] for r in rows if "workflow_adherent" in r]
        out["workflow_adherence"] = sum(wa) / (len(wa) or 1)
    return out


def weak_link(rows: list[dict[str, Any]], workflow_order: list[str]) -> dict[str, int]:
    """Attribute each failure to the first tool, in workflow order, that erred."""
    counts: dict[str, int] = {}
    for r in rows:
        if r["correct"]:
            continue
        for tool in workflow_order:
            key = f"c_{tool}"
            if key in r and r[key] == 0:
                counts[tool] = counts.get(tool, 0) + 1
                break
    return counts


def attribute(
    rows: list[dict[str, Any]],
    factor_names: list[str],
    metric: str = "correct",
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Logistic factor attribution (+ Benjamini-Hochberg) via knowlytix DOEAnalyzer."""
    import pandas as pd
    from knowlytix.harness.graphdoe import DOEAnalyzer

    df = pd.DataFrame(rows)
    cols = {f"f_{f}": f for f in factor_names if f"f_{f}" in df.columns}
    df = df.rename(columns=cols)
    analyzer = DOEAnalyzer.from_dataframe(df, factors=list(cols.values()))
    table = analyzer.run_logistic(metric=metric)
    table = analyzer.apply_fdr_correction(table, alpha=alpha)
    return table.to_dict(orient="records") if hasattr(table, "to_dict") else table


# ---------------------------------------------------------------------------
class MockSUT:
    """Deterministic SUT for tests: answers correctly unless a configured factor
    level is present, in which case it errs. Emits a simple trajectory."""

    def __init__(self, fail_on: dict[str, str] | None = None,
                 workflow: list[str] | None = None):
        self.fail_on = fail_on or {}
        self.workflow = workflow or [
            "classify_complaint", "extract_facts", "search_policy",
            "flag_regulatory", "draft_response"]

    def __call__(self, query: str, context: str = "", scenario: Scenario | None = None) -> SUTResult:
        # The harness calls sut(query, context=...); we cannot see factor_levels
        # here, so failure is keyed off markers the materializer leaves in text.
        fail = any(f"[{k}={v}]" in query for k, v in self.fail_on.items())
        traj = [{"tool": t, "ok": True} for t in self.workflow]
        if fail:
            return SUTResult(answer="WRONG", components={"classification": "other"},
                             trajectory=traj, status="failed")
        return SUTResult(answer="OK", components={"classification": "complaint"},
                         trajectory=traj, status="ok")
