import pytest
from pydantic import BaseModel, ValidationError

from agentlab.tools import RiskLevel, Tool, ToolRegistry


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


def _tool(name: str = "adder") -> Tool:
    return Tool(
        name=name,
        description="add one to x",
        input_schema=_In,
        output_schema=_Out,
        risk=RiskLevel.LOW,
    )


def test_register_and_get():
    r = ToolRegistry()
    t = _tool()
    r.register(t)
    assert r.get("adder") is t


def test_duplicate_name_rejected():
    r = ToolRegistry()
    r.register(_tool())
    with pytest.raises(ValueError):
        r.register(_tool())


def test_invalid_name_rejected():
    r = ToolRegistry()
    with pytest.raises(ValueError):
        r.register(_tool(name="bad name"))


def test_validate_success():
    r = ToolRegistry()
    r.register(_tool())
    parsed = r.validate("adder", {"x": 1})
    assert parsed.x == 1


def test_validate_failure():
    r = ToolRegistry()
    r.register(_tool())
    with pytest.raises(ValidationError):
        r.validate("adder", {"x": "not-an-int"})


def test_unknown_tool_raises():
    r = ToolRegistry()
    with pytest.raises(KeyError):
        r.get("nope")


def test_all_returns_registered():
    r = ToolRegistry()
    r.register(_tool("a"))
    r.register(_tool("b"))
    names = sorted(t.name for t in r.all())
    assert names == ["a", "b"]
