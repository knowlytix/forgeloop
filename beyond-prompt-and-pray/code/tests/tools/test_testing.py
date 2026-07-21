from pydantic import BaseModel

from agentlab.tools import RiskLevel, Tool
from agentlab.tools.testing import assert_schema_matches, fuzz_tool


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


def _adder(x: int) -> dict:
    return {"y": x + 1}


def _tool() -> Tool:
    return Tool(
        name="adder",
        description="",
        input_schema=_In,
        output_schema=_Out,
        risk=RiskLevel.LOW,
        fn=_adder,
    )


def test_fuzz_reports_three_counters():
    report = fuzz_tool(_tool(), num_cases=20, seed=1)
    assert {"accepted", "rejected_clean", "crashed"} <= report.keys()
    assert report["accepted"] + report["rejected_clean"] + report["crashed"] == 20


def test_fuzz_well_behaved_tool_has_no_crashes():
    # An int-typed schema should reject everything that's not {"x": int} via ValidationError
    report = fuzz_tool(_tool(), num_cases=50, seed=2)
    assert report["crashed"] == 0


def test_assert_schema_matches_ok():
    assert_schema_matches(_tool(), {"x": 1}, {"x": "nope"})


def test_assert_schema_matches_raises_when_invalid_passes():
    import pytest

    with pytest.raises(AssertionError):
        assert_schema_matches(_tool(), {"x": 1}, {"x": 2})
