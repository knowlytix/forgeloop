#!/usr/bin/env python
"""End-to-end capstone campaign: drive the governed complaint agent through the
framework and produce the full analysis (balance, factor attribution, weak-link
and per-tool decomposition). Overall accuracy is the per-query DECISION accuracy
(right classification AND right escalate/don't-escalate); per-tool components and
graph-truth retrieval are scored separately, not folded into it. Writes
data/capstone_run.json.

    PYTHONPATH=. python scripts/capstone_run.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gmstest as gt
from gmstest.compose import graphdoe_design
from gmstest.evaluate import attribute, weak_link

TUT = Path(__file__).resolve().parents[1]
OUT = TUT / "data" / "capstone_run.json"
ROWS = TUT / "data" / "capstone_rows.json"
TESTSET = TUT / "data" / "capstone_testset.json"
WORKFLOW = ["classify_complaint", "extract_facts", "search_policy",
            "flag_regulatory", "draft_response"]


def _load_or_build_testset(scns, harness, factors):
    """Return the 120 fixed customer messages, one per composed scenario.

    The test set is GENERATED ONCE (materializing the clarity dimension with Qwen,
    the only nondeterministic step) and PINNED to capstone_testset.json. Every rerun
    then LOADS the pinned messages and runs the agent over them, so the test is fixed
    and reruns are comparable -- only the SUT varies, not the inputs. The pinned set
    is reused only when it matches the composed design row-for-row (same seed case and
    factor levels); otherwise it is rebuilt, so a design change can never silently
    score against stale inputs."""
    design = [{"row": i, "seed_case": scn.base.qid,
               **{f: scn.factor_levels[f] for f in factors}}
              for i, scn in enumerate(scns)]
    if TESTSET.exists():
        pinned = json.loads(TESTSET.read_text()).get("scenarios", [])
        matches = len(pinned) == len(design) and all(
            p.get("seed_case") == d["seed_case"]
            and all(p.get(f) == d[f] for f in factors)
            for p, d in zip(pinned, design))
        if matches:
            print(f"loaded pinned test set ({len(pinned)} scenarios) from {TESTSET.name}")
            return [p["message"] for p in pinned]
        print(f"{TESTSET.name} does not match the composed design; re-materializing")
    messages = [harness.materialize({"seed_case": d["seed_case"],
                                     **{f: d[f] for f in factors}}) for d in design]
    TESTSET.write_text(json.dumps({
        "_note": "Pinned 120-scenario test set for the capstone campaign: the "
                 "materialized customer messages + factor levels for the canonical "
                 "balanced design (gt.compose balance_base=True, seed=42). The "
                 "campaign loads THIS and runs the agent over it; regenerate only "
                 "deliberately (delete this file).",
        "source": "materialized once from the canonical balanced design",
        "n": len(design),
        "scenarios": [{**d, "message": m} for d, m in zip(design, messages)],
    }, indent=2) + "\n")
    print(f"materialized and pinned {len(messages)} messages to {TESTSET.name}")
    return messages


def _component_attribution(rows, factors):
    """Per-component failure analysis by EMPIRICAL per-level failure rate.

    The end-to-end outcome (`correct`) is a conjunction that folds several components
    together; attributing it answers nothing actionable. Instead each model-backed
    component is analyzed on its OWN correctness, on the runs where it executed against
    defined ground truth, so the result says which input phrasing drives THAT
    component's errors.

    Because failures are sparse here (10-18 graded runs per tool), a logistic fit
    separates and its odds ratios blow up; the robust, honest report is the EMPIRICAL
    failure rate at each factor level -- failed / graded -- which never diverges. A
    deviance test is added as a SECONDARY signal only for a factor whose fit does not
    separate (every level carries both a pass and a fail); otherwise it is omitted and
    the rates stand on their own. A component with no failures has nothing to analyze."""
    out = {}
    for col, tool in (("c_classify_complaint", "classify_complaint"),
                      ("c_search_policy", "search_policy"),
                      ("c_flag_regulatory", "flag_regulatory")):
        sub = [{**r, col: int(r[col])} for r in rows if r.get(col) is not None]
        n = len(sub)
        fails = sum(1 for r in sub if r[col] == 0)
        entry = {"scored": n, "failures": fails, "by_factor": {}}
        for f in factors:
            fcol = f"f_{f}"
            levels = {}
            for lvl in sorted({r[fcol] for r in sub}):
                at = [r for r in sub if r[fcol] == lvl]
                failed = sum(1 for r in at if r[col] == 0)
                levels[lvl] = {"failed": failed, "n": len(at),
                               "rate": round(failed / len(at), 3) if at else None}
            entry["by_factor"][f] = levels

        # Secondary deviance test, kept only for non-separable factors (every level
        # has both a pass and a fail) -- otherwise the logistic fit is unreliable.
        if 0 < fails < n:
            try:
                table = attribute(sub, factors, metric=col)
                dev = {}
                for t in table:
                    f = t["factor"]
                    lv = entry["by_factor"][f]
                    separable = any(v["failed"] in (0, v["n"]) for v in lv.values())
                    if not separable:
                        dev[f] = {"G2": round(t["deviance"], 2),
                                  "p_adj": round(t["p_value_adj"], 4),
                                  "pseudo_r2": round(t["pseudo_r2"], 3),
                                  "significant_adj": bool(t["significant_adj"])}
                if dev:
                    entry["deviance"] = dev
            except Exception:
                pass
        out[tool] = entry
    return out


def _fmt_attr(table):
    """Format a knowlytix attribution table (used for the overall conjunction, which
    has enough rows -- 120 -- for the logistic odds ratios to be stable)."""
    return [{"factor": t["factor"], "G2": round(t["deviance"], 2),
             "p_adj": round(t["p_value_adj"], 4), "pseudo_r2": round(t["pseudo_r2"], 3),
             "significant_adj": bool(t["significant_adj"]),
             "odds_ratios": {k: round(v, 2) for k, v in t["odds_ratios"].items()}}
            for t in table]


def _per_tool(rows):
    """Per-component correctness: classify/flag as a plain rate, search as recall@k
    with coverage and an abstention count (an abstention is not a wrong answer)."""
    per_tool = {}
    for col, tool in (("c_classify_complaint", "classify_complaint"),
                      ("c_search_policy", "search_policy"),
                      ("c_flag_regulatory", "flag_regulatory")):
        scored = [r[col] for r in rows if r[col] is not None]
        if scored:
            per_tool[tool] = {"scored": len(scored), "correct": int(sum(scored)),
                              "accuracy": round(sum(scored) / len(scored), 3)}
    if "search_policy" in per_tool:
        abst = sum(1 for r in rows if r.get("c_search_abstained") is True)
        committed = per_tool["search_policy"]["scored"]
        per_tool["search_policy"]["abstained"] = abst
        per_tool["search_policy"]["coverage"] = (
            round(committed / (committed + abst), 3) if (committed + abst) else None)
    return per_tool


def build_row_analysis(rows, factors):
    """Every analysis block that is derived from the per-row table -- everything
    except the gate stack and the audit, which need the live agent. Shared by the
    campaign and the offline recompute script, so a change to the attribution math
    never requires re-running 120 agents."""
    n = len(rows)
    return {
        "workflow_adherence": round(sum(r["workflow_adherent"] for r in rows) / n, 3),
        "seed_balance": dict(sorted(Counter(r["seed"] for r in rows).items())),
        "clarity_balance": dict(Counter(r["f_clarity"] for r in rows)),
        # (1) PER-COMPONENT: which input breaks WHICH tool -- the actionable diagnosis.
        "attribution_by_component": _component_attribution(rows, factors),
        # (2) OVERALL conjunction: whether the whole run erred at any step -- a useful
        #     end-to-end summary, read as a whole, NOT a substitute for the per-component view.
        "overall": {
            "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
            "definition": "fraction of test queries correctly decisioned: the right "
                          "classification AND the right escalate/don't-escalate decision. "
                          "Per-tool components are scored separately, not folded in here.",
            "attribution": _fmt_attr(attribute(rows, factors)),
        },
        "weak_link": weak_link(rows, WORKFLOW),    # first erring step on a wrong run
        "per_tool": _per_tool(rows),
    }


def main() -> int:
    from agentlab.testing import CapstoneTestHarness
    from agentlab.testing.capstone_harness import (
        _classification_of, _workflow_adherent, did_escalate,
    )
    from apps.complaint_sut import (
        AgentSUT, score_components, probe_policy_gate, probe_plausibility_gate,
        retrieval_benchmark,
    )

    cat = gt.Catalog.load()
    app = yaml.safe_load((TUT / "apps/complaint_agent.yaml").read_text())
    pc = app["parity_ch16"]                      # clarity, entity_aliasing, reasoning_cue
    overrides = pc["level_overrides"]            # domain level vocabulary
    goldens = json.loads((TUT.parents[1] / "beyond-prompt-and-pray" / "code" / app["data"]["goldens"]).read_text())

    harness = CapstoneTestHarness(n_runs=120, seed=42, rephrase_method="qwen")
    cases_by_id = {c["id"]: c for c in harness._cases}
    items = gt.SeedCaseSource(harness._cases).items()

    # Design: balanced blocking over the seed cases, Sobol+refine over factors.
    suite = gt.resolve(cat, ["conditional_rule"], pc["factors"], mode="embedded")
    refine = partial(graphdoe_design, method="sobol+refine")
    scns = gt.compose(suite, items, n_runs=120, seed=42, design_fn=refine,
                      balance_base=True, level_overrides=overrides)
    factors = suite.factor_names

    # Fixed test inputs: materialize once, pin, then reuse on every rerun.
    messages = _load_or_build_testset(scns, harness, factors)

    sut = AgentSUT()
    rows = []
    for scn, msg in zip(scns, messages):
        case = cases_by_id[scn.base.qid]
        traj = sut.run_traj(msg)
        classification = _classification_of(traj)
        escalated = did_escalate(traj)
        comps = score_components(traj, case, goldens)
        # Overall = per-query DECISION accuracy: the query is correct iff the agent
        # reached the right end decision -- the right classification AND the right
        # escalate/don't-escalate. This is what "correctly decisioned" means, counted
        # over the individual test queries. The per-tool components (classification,
        # retrieval, flag) are scored SEPARATELY as diagnostics -- they are not folded
        # into this number (a run can bind the right policy yet mis-decide, or decide
        # right while a downstream tool erred); weak_link below attributes a
        # wrongly-decisioned query to its first erring step.
        correct = int(classification == case["expected_classification"]
                      and escalated == case["expected_escalation"])
        rows.append({
            "seed": scn.base.qid,
            "correct": correct,
            "workflow_adherent": int(_workflow_adherent(traj)),
            **{f"f_{k}": v for k, v in scn.factor_levels.items()},
            "c_classify_complaint": comps.get("c_classify_complaint"),
            "c_search_policy": comps.get("c_search_policy"),
            "c_search_abstained": comps.get("c_search_abstained"),
            "c_flag_regulatory": comps.get("c_flag_regulatory"),
        })

    analysis = build_row_analysis(rows, factors)

    # Gate stack (DETERMINISTIC, decoupled from the nondeterministic campaign): a
    # gate decides whether a tool fires, so its decisions are graded too -- but both
    # probes drive the DEPLOYED gates directly, not the campaign rows, so the grade
    # is reproducible. The Policy gate is run on the canonical case messages (PII ->
    # escalate, injection -> deny, benign -> allow); the plausibility gate is swept
    # over every workflow transition against the store's has_enables DAG.
    pol = probe_policy_gate(list(cases_by_id.values()), harness=sut._harness)
    pla = probe_plausibility_gate(harness=sut._harness)
    gate_stack = {
        "policy_gate": {
            "refuse_path": pol["refuse_path"],
            "allow_path": pol["allow_path"],
            "errors": pol["errors"],
        },
        "plausibility_gate": {
            "theta": pla["theta"],
            "legal": pla["legal"],
            "illegal": pla["illegal"],
            "errors": pla["errors"],
        },
    }

    # Retrieval quality graded against the GRAPH GROUND TRUTH (Beyond Chunk and
    # Pray): query -> bind to a triple -> retrieve, hit iff the retrieved value
    # matches the ground-truth value, misses localized to parse/bind/retrieve, with
    # a dense baseline on the same encoder. This is the authoritative search_policy
    # metric; the per-tool top-k policy-label recall below is a coarse secondary.
    retrieval = retrieval_benchmark()

    result = {
        "n_runs": len(rows),
        "workflow_adherence": analysis["workflow_adherence"],
        "audit_verifies": sut.audit_verifies(),
        "retrieval_benchmark": retrieval,
        "gate_stack": gate_stack,
        "seed_balance": analysis["seed_balance"],
        "clarity_balance": analysis["clarity_balance"],
        "attribution_by_component": analysis["attribution_by_component"],
        "overall": analysis["overall"],
        "weak_link": analysis["weak_link"],
        "per_tool": analysis["per_tool"],
    }
    OUT.write_text(json.dumps(result, indent=2))
    # Persist the per-row table so attribution / per-tool math can be recomputed
    # offline (scripts/recompute_capstone_analysis.py) WITHOUT re-running 120 agents.
    ROWS.write_text(json.dumps({"factors": factors, "rows": rows}, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote per-row table to {ROWS.name} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
