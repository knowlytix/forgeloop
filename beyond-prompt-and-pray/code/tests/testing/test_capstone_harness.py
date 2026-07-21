"""Tests for the CapstoneTestHarness DoE test stand.

The model-free tests (design, materialize, attribution on synthetic rows) run
anywhere. The groundedness test loads the GMS banking store (cheap --- no Qwen)
and is skipped if knowlytix or the store is unavailable. The full agent run is
exercised by `scripts/run_capstone_doe.py`, not here, to keep the suite fast.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentlab.testing import CapstoneTestHarness, CapstoneTestResult
from agentlab.testing.capstone_harness import (
    _escalation_trigger,
    _expected_trigger,
    _tool_sequence,
    _workflow_adherent,
)


def _traj(tool_names, escalate_reason=None):
    """Build a minimal stand-in trajectory: one record per tool call, plus a
    terminal escalate record when a reason is given."""
    recs = [SimpleNamespace(action=SimpleNamespace(kind="tool_call", tool_name=t))
            for t in tool_names]
    if escalate_reason is not None:
        recs.append(SimpleNamespace(action=SimpleNamespace(kind="escalate",
                                                           reason=escalate_reason)))
    return SimpleNamespace(records=recs)

REPO_ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = REPO_ROOT / "data" / "gms_banking_store"


def _knowlytix_available() -> bool:
    try:
        import knowlytix.harness.graphdoe  # noqa: F401
    except ImportError:
        return False
    return True


requires_knowlytix = pytest.mark.skipif(
    not _knowlytix_available(), reason="knowlytix harness not installed"
)
requires_store = pytest.mark.skipif(
    not STORE_PATH.exists(), reason="GMS banking store not built"
)


# --- model-free design / materialize -------------------------------------

@requires_knowlytix
def test_design_is_balanced():
    h = CapstoneTestHarness(n_runs=90, seed=42)
    df = h.design()
    assert len(df) == 90
    assert set(df.columns) == {"clarity", "entity_aliasing", "reasoning_cue", "seed_case"}
    # Each level of each presentation factor should be well represented (a
    # balanced Sobol design, not a degenerate one).
    for col in ("clarity", "entity_aliasing", "reasoning_cue"):
        counts = df[col].value_counts()
        assert len(counts) == 3
        assert counts.min() >= 15


@requires_knowlytix
def test_materialize_varies_and_preserves_signal():
    # template path: deterministic + model-free, so the assertions are exact.
    # (Qwen is the default at runtime; here we pin template for a fast offline test.)
    h = CapstoneTestHarness(n_runs=30, seed=1, rephrase_method="template")
    row = {"seed_case": "case-001", "clarity": "clear",
           "entity_aliasing": "canonical", "reasoning_cue": "none"}
    clear = h.materialize(row)
    assert clear == h._cases_by_id["case-001"]["message"]
    # Each factor transform changes the message.
    misleading = h.materialize({**row, "clarity": "misleading"})
    typo = h.materialize({**row, "entity_aliasing": "typo"})
    cue = h.materialize({**row, "reasoning_cue": "misleading_cue"})
    assert misleading != clear and typo != clear and cue != clear
    # The dollar amount survives every transform (it is the verifiable claim).
    assert "35" in misleading and "35" in typo and "35" in cue


# --- trajectory-level helpers (model-free) --------------------------------

def test_workflow_adherence_prefix_and_violation():
    full = ["classify_complaint", "extract_facts", "search_policy",
            "flag_regulatory", "draft_response"]
    assert _workflow_adherent(_traj(full))                       # full workflow
    assert _workflow_adherent(_traj(full[:2]))                   # early-escalation prefix
    assert _tool_sequence(_traj(full[:3])) == full[:3]
    # out-of-order / skipped steps are not adherent
    assert not _workflow_adherent(_traj(["classify_complaint", "draft_response"]))
    assert not _workflow_adherent(_traj(["search_policy", "classify_complaint"]))


def test_escalation_trigger_classification():
    assert _escalation_trigger(_traj(["classify_complaint"],
                                     "tool failure at step 0: PII")) == "gate_refusal"
    assert _escalation_trigger(_traj(["classify_complaint", "extract_facts",
                                      "search_policy", "flag_regulatory"],
                                     "regulatory risk flagged: ['UDAAP']")) == "regulatory"
    assert _escalation_trigger(_traj([], "draft verification: waiver")) == "draft_verifier"
    assert _escalation_trigger(_traj(["classify_complaint"])) == ""   # no escalation


def test_expected_trigger_from_factors():
    assert _expected_trigger({"factors": {"regulatory": "PII"}}) == "gate_refusal"
    assert _expected_trigger({"factors": {"regulatory": "prompt_injection"}}) == "gate_refusal"
    assert _expected_trigger({"factors": {"regulatory": "UDAAP"}}) == "regulatory"
    assert _expected_trigger({"factors": {"regulatory": "none"}}) == ""


# --- attribution on synthetic rows (no agent run) -------------------------

def _synthetic_result(n: int = 90) -> CapstoneTestResult:
    """Rows where `misleading` clarity drives failure; everything else passes.
    Deterministic, no model. Cycles the factor levels for balance."""
    clar = ["clear", "ambiguous", "misleading"]
    alias = ["canonical", "alias", "typo"]
    cue = ["none", "cot", "misleading_cue"]
    rows = []
    for i in range(n):
        c = clar[i % 3]
        rows.append({
            "row": i, "seed_case": f"case-{(i % 20) + 1:03d}",
            "clarity": c, "entity_aliasing": alias[i % 3], "reasoning_cue": cue[i % 3],
            "correct": 0 if c == "misleading" else 1,
            "draft_tier": "n/a",
        })
    return CapstoneTestResult(rows=rows, n_runs=n)


@requires_knowlytix
def test_analyze_attributes_failure_to_clarity():
    h = CapstoneTestHarness(seed=42)
    attr = h.analyze(_synthetic_result(90))
    assert attr.logistic_table  # non-empty
    factors = {r["factor"] for r in attr.logistic_table}
    assert {"clarity", "entity_aliasing", "reasoning_cue"} <= factors
    # clarity is the planted driver and should be the top significant factor.
    drivers = attr.top_drivers(k=1)
    assert drivers and drivers[0]["factor"] == "clarity"
    # the misleading level is the worst failure.
    top = attr.top_failures(k=1)
    assert top and top[0]["level"] == "misleading"


# --- per-tool decomposition + fault profiles (model-free) -----------------

@requires_knowlytix
def test_tool_breakdown_and_weak_link():
    h = CapstoneTestHarness()
    rows = [
        {"tool_classify_ok": True, "tool_extract_ok": False, "tool_search_ok": None,
         "tool_flag_ok": True, "weak_link": "extract_facts"},
        {"tool_classify_ok": False, "tool_extract_ok": None, "tool_search_ok": True,
         "tool_flag_ok": None, "weak_link": "classify_complaint"},
        {"tool_classify_ok": True, "tool_extract_ok": True, "tool_search_ok": True,
         "tool_flag_ok": None, "weak_link": ""},
    ]
    bd = h.tool_breakdown(CapstoneTestResult(rows=rows, n_runs=len(rows)))
    # classify scored on all 3 rows, 2 correct -> 2/3; extract scored on 2, 1 -> 0.5
    assert bd["per_tool"]["classify_complaint"]["accuracy"] == pytest.approx(2 / 3)
    assert bd["per_tool"]["extract_facts"]["accuracy"] == 0.5
    assert bd["per_tool"]["search_policy"]["accuracy"] == 1.0
    assert bd["per_tool"]["flag_regulatory"]["accuracy"] == 1.0  # scored once, correct
    assert bd["weak_link_counts"] == {"extract_facts": 1, "classify_complaint": 1}


@requires_knowlytix
def test_fault_profiles():
    h = CapstoneTestHarness()
    err = h._fault_profile("search_policy", "error")
    assert err.tool_pattern == "search_policy" and err.error_rate == 1.0
    lat = h._fault_profile("search_policy", "latency")
    assert lat.latency_ms > 0
    stale = h._fault_profile("search_policy", "stale")
    assert stale.mock_response is not None


# --- groundedness-as-distance (store only, no Qwen) -----------------------

@requires_knowlytix
@requires_store
def test_groundedness_tiers_separate():
    h = CapstoneTestHarness()
    bands = h.calibrate_groundedness()
    assert bands["has_fee_amount"]["grounded_score"] < bands["has_fee_amount"]["fabricated_score"]
    assert bands["has_max_reversal"]["grounded_score"] < bands["has_max_reversal"]["fabricated_score"]
    # The committed overdraft fee ($35) is grounded; a wrong amount fabricates.
    _, geo_ok, tier_ok = h.judge_draft("The overdraft fee is $35 per occurrence.", "overdraft_fee")
    _, geo_bad, tier_bad = h.judge_draft("The overdraft fee is $50 per occurrence.", "overdraft_fee")
    assert tier_ok == "grounded"
    assert tier_bad == "fabrication"
    assert geo_ok < geo_bad
    # No verifiable fee claim -> n/a (not a false fabrication).
    assert h.judge_draft("Here is how to update your address.", "account_issue")[2] == "n/a"
    assert h.judge_draft("We will review your overdraft.", "overdraft_fee")[2] == "n/a"
