"""Unit tests for the gmstest core + consumers (no GMS / no GPU required)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import gmstest as gt  # noqa: E402
from gmstest.evaluate import MockSUT, SUTResult, run, summary, weak_link  # noqa: E402


@pytest.fixture(scope="module")
def catalog():
    return gt.Catalog.load()


# -- catalogs ---------------------------------------------------------------
def test_catalog_loads(catalog):
    assert len(catalog.bases) == 18
    assert len(catalog.factors) == 40
    assert len(catalog.families) == 5
    assert "smoke" in catalog.profiles


def test_every_factor_is_gt_invariant(catalog):
    assert all(f.gt_invariant for f in catalog.factors.values())


def test_no_name_collision(catalog):
    assert not (set(catalog.bases) & set(catalog.factors))


# -- resolution + applies_to ------------------------------------------------
def test_resolve_expands_families_and_groups(catalog):
    suite = gt.resolve(catalog, ["retrieval_recall"], ["query_core"], mode="cross")
    assert {b.name for b in suite.bases} == set(catalog.families["retrieval_recall"])
    assert suite.factors  # query_core expanded to its factors


def test_applies_to_filters_numeric_factors_without_numeric_base(catalog):
    # A purely set-valued base (cross_reference -> set_str) must drop numeric_format.
    suite = gt.resolve(catalog, ["cross_reference"], ["numeric_format", "clarity"],
                       mode="embedded")
    names = suite.factor_names
    assert "clarity" in names               # applies to all
    assert "numeric_format" not in names    # numeric-only -> dropped
    assert any(d[0] == "numeric_format" for d in suite.dropped_factors)


def test_numeric_factor_kept_with_numeric_base(catalog):
    suite = gt.resolve(catalog, ["exact_recall"], ["numeric_format"], mode="embedded")
    assert "numeric_format" in suite.factor_names


def test_resolve_profile(catalog):
    suite = gt.resolve_profile(catalog, "numeric_screen")
    assert suite.profile == "numeric_screen"
    assert suite.mode == "embedded"
    assert suite.bases and suite.factors


# -- sources ----------------------------------------------------------------
def test_user_base_source_no_gms():
    src = gt.UserBaseSource([
        {"query": "What is the overdraft fee?", "answer": "35"},
        {"query": "How long to dispute?", "answer": "60 days", "answer_type": "str"},
    ])
    items = src.items()
    assert len(items) == 2
    assert items[0].base is None          # user-supplied, not catalog-derived
    assert items[0].answer == "35"


def test_seed_case_source_components():
    cases = [{"id": "case-001", "message": "I was charged a $35 fee",
              "expected_classification": "complaint", "expected_escalate": False}]
    items = gt.SeedCaseSource(cases).items()
    assert items[0].qid == "case-001"
    assert items[0].components["expected_classification"] == "complaint"


# -- composition: cross vs embedded ----------------------------------------
def _items(n):
    return [gt.QAItem(qid=f"q{i}", query=f"[clarity=clear] q{i}", answer="OK") for i in range(n)]


def test_cross_size_is_bases_times_runs(catalog):
    suite = gt.resolve(catalog, ["exact_recall"], ["clarity"], mode="cross")
    items = _items(3)
    scns = gt.compose(suite, items, n_runs=5, seed=1)
    assert len(scns) == 3 * 5
    assert all(s.mode == "cross" for s in scns)


def test_embedded_size_is_runs(catalog):
    suite = gt.resolve(catalog, ["exact_recall"], ["clarity"], mode="embedded")
    items = _items(3)
    scns = gt.compose(suite, items, n_runs=12, seed=1)
    assert len(scns) == 12
    assert all(s.mode == "embedded" for s in scns)
    assert all(s.base.qid in {"q0", "q1", "q2"} for s in scns)


def test_compose_preserves_ground_truth(catalog):
    suite = gt.resolve(catalog, ["exact_recall"], ["clarity", "length"], mode="cross")
    items = _items(2)
    for s in gt.compose(suite, items, n_runs=4):
        assert s.base.answer == "OK"  # enrichment never changes the answer


# -- emit (Ch15 SFT) --------------------------------------------------------
def test_emit_classifier_sft_label_preserving(catalog):
    suite = gt.resolve(catalog, ["exact_recall"], ["clarity"], mode="cross")
    items = [gt.QAItem(qid="s0", query="msg", answer="complaint")]
    scns = gt.compose(suite, items, n_runs=3)
    recs = gt.emit_classifier_sft(scns)
    assert len(recs) == 3
    assert all(r["label"] == "complaint" for r in recs)  # invariant


def test_emit_draft_sft_golden_validation(catalog):
    suite = gt.resolve(catalog, ["exact_recall"], ["clarity"], mode="cross")
    items = [gt.QAItem(qid="p0", query="I was charged $35", answer=None,
                       metadata={"policy_id": "overdraft"})]
    scns = gt.compose(suite, items, n_runs=2)
    goldens = {"overdraft": {"required_keywords": ["overdraft"],
                             "required_numbers": [35],
                             "forbidden_phrases": ["we will waive"]}}
    # Good response cites keyword + number; bad omits the number.
    def good(s): return "Your overdraft fee of $35 may be reversed once per year."
    kept, dropped = gt.emit_draft_sft(scns, good, goldens=goldens)
    assert len(kept) == 2 and not dropped

    def bad(s): return "We will waive your fee."
    kept2, dropped2 = gt.emit_draft_sft(scns, bad, goldens=goldens)
    assert not kept2 and len(dropped2) == 2  # forbidden phrase + missing number


def test_to_jsonl_roundtrip(tmp_path, catalog):
    recs = [{"message": "a", "label": "complaint"}]
    n = gt.to_jsonl(recs, tmp_path / "out.jsonl")
    assert n == 1
    assert (tmp_path / "out.jsonl").read_text().strip().startswith("{")


# -- evaluate (Ch16) --------------------------------------------------------
def test_evaluate_mock_sut_and_attribution_signal(catalog):
    suite = gt.resolve(catalog, ["exact_recall"], ["clarity"], mode="embedded")
    # base queries carry a factor marker the MockSUT reads (stand-in materializer)
    items = [gt.QAItem(qid=f"s{i}", query="msg", answer="OK",
                       components={"expected_classification": "complaint",
                                   "expected_escalate": False})
             for i in range(4)]
    scns = gt.compose(suite, items, n_runs=30, seed=7)
    # Stamp the clarity level into the query text so MockSUT can see it.
    for s in scns:
        s.base = gt.QAItem(qid=s.base.qid,
                           query=f"[clarity={s.factor_levels.get('clarity','clear')}] msg",
                           answer=s.base.answer, components=s.base.components)
    sut = MockSUT(fail_on={"clarity": "Misleading"})  # catalog levels are capitalized
    rows = run(scns, sut, workflow_order=sut.workflow)
    s = summary(rows)
    assert s["n"] == 30
    assert s["workflow_adherence"] == 1.0
    # misleading rows fail; clear/ambiguous pass -> accuracy strictly between 0 and 1
    assert 0.0 < s["accuracy"] < 1.0


def test_weak_link_blames_failing_component():
    rows = [
        {"correct": 0, "c_classify_complaint": 0, "c_extract_facts": 1},
        {"correct": 0, "c_classify_complaint": 1, "c_extract_facts": 0},
        {"correct": 1, "c_classify_complaint": 1, "c_extract_facts": 1},
    ]
    counts = weak_link(rows, ["classify_complaint", "extract_facts"])
    assert counts == {"classify_complaint": 1, "extract_facts": 1}


def test_sut_result_bare_value_scores_outcome():
    scn = gt.Scenario(sid="x", base=gt.QAItem(qid="q", query="hi", answer="OK"),
                      factor_levels={}, mode="cross")
    rows = run([scn], lambda q, context="": "OK")
    assert rows[0]["correct"] == 1
