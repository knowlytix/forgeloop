import pytest
from pydantic import ValidationError

from agentlab.reasoning import Entry, EntryType, Scratchpad, TrustLevel


def test_add_claim_with_evidence():
    s = Scratchpad()
    e = s.add_claim("the sky is blue", evidence="doc-1", trust=TrustLevel.MEDIUM)
    assert e.kind == EntryType.CLAIM
    assert e.evidence == "doc-1"
    assert e.trust == TrustLevel.MEDIUM


def test_entry_text_must_be_nonempty():
    with pytest.raises(ValidationError):
        Entry(kind=EntryType.CLAIM, text="   ")


def test_entry_is_frozen():
    e = Entry(kind=EntryType.QUESTION, text="why")
    with pytest.raises(ValidationError):
        e.text = "changed"


def test_observation_requires_source():
    with pytest.raises(ValidationError):
        Entry(kind=EntryType.OBSERVATION, text="x")


def test_trust_above_low_requires_evidence():
    with pytest.raises(ValidationError):
        Entry(kind=EntryType.CLAIM, text="x", trust=TrustLevel.HIGH)


def test_by_type_filters():
    s = Scratchpad()
    s.add_claim("a", evidence="e")
    s.add_assumption("b")
    s.add_question("c")
    s.add_observation("d", source="doc")
    assert len(s.by_type(EntryType.CLAIM)) == 1
    assert len(s.by_type(EntryType.ASSUMPTION)) == 1
    assert len(s.by_type(EntryType.QUESTION)) == 1
    assert len(s.by_type(EntryType.OBSERVATION)) == 1


def test_unsupported_claims_are_detected():
    s = Scratchpad()
    s.add_claim("supported", evidence="doc-1")
    s.add_claim("unsupported")
    bad = s.unsupported_claims()
    assert len(bad) == 1
    assert bad[0].text == "unsupported"


def test_assert_all_claims_have_evidence_raises():
    s = Scratchpad()
    s.add_claim("no evidence")
    with pytest.raises(AssertionError):
        s.assert_all_claims_have_evidence()


def test_assert_all_claims_have_evidence_passes():
    s = Scratchpad()
    s.add_claim("with evidence", evidence="e")
    s.assert_all_claims_have_evidence()


def test_render_table_list_of_dicts():
    s = Scratchpad()
    s.add_claim("hi", evidence="e")
    rows = s.render_table()
    assert isinstance(rows, list)
    assert rows[0]["text"] == "hi"
    assert rows[0]["evidence"] == "e"


def test_render_table_string():
    s = Scratchpad()
    s.add_claim("hi", evidence="e")
    text = s.render_table(as_string=True)
    assert isinstance(text, str)
    assert "hi" in text


def test_render_table_empty():
    s = Scratchpad()
    assert s.render_table(as_string=True) == "(empty scratchpad)"


def test_entries_property_is_immutable_tuple():
    s = Scratchpad()
    s.add_question("q")
    entries = s.entries
    assert isinstance(entries, tuple)
