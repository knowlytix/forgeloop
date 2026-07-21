from agentlab.evaluation import (
    Claim,
    ClaimVerdict,
    check_groundedness,
    coverage,
    extract_claims,
    groundedness_report,
)


def test_extract_claims_splits_on_punctuation():
    claims = extract_claims("The fee is thirty-five dollars. Customers must be notified.")
    assert len(claims) == 2
    assert "fee" in claims[0].text


def test_extract_claims_empty():
    assert extract_claims("") == []


def test_supported_when_terms_overlap():
    claim = Claim(text="The overdraft fee is thirty-five dollars per transaction")
    evidence = {
        "doc-1": "Overdraft fees apply at thirty-five dollars for each transaction on consumer accounts.",
    }
    result = check_groundedness(claim, evidence)
    assert result.verdict == ClaimVerdict.SUPPORTED
    assert result.evidence_id == "doc-1"


def test_unsupported_when_terms_disjoint():
    claim = Claim(text="Cats produce milk regularly every Tuesday morning")
    evidence = {
        "doc-1": "Banks charge fees on overdraft transactions.",
    }
    result = check_groundedness(claim, evidence)
    assert result.verdict == ClaimVerdict.UNSUPPORTED


def test_uncertain_when_claim_is_only_stopwords():
    claim = Claim(text="the a an is to of")
    evidence = {"doc-1": "anything"}
    result = check_groundedness(claim, evidence)
    assert result.verdict == ClaimVerdict.UNCERTAIN


def test_groundedness_report_and_coverage():
    claims = [
        Claim(text="overdraft fees apply on consumer accounts daily"),
        Claim(text="dinosaurs walked the planet in the Cretaceous"),
    ]
    evidence = {"doc-1": "overdraft fees apply on consumer accounts daily."}
    report = groundedness_report(claims, evidence)
    assert len(report) == 2
    assert coverage(report) == 0.5


def test_coverage_empty():
    assert coverage([]) == 0.0
