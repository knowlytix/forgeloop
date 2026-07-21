import pytest
from pydantic import ValidationError

from agentlab.core import TaskSpec, ValidationRule


def test_task_minimal():
    t = TaskSpec(goal="say hello")
    assert t.goal == "say hello"
    assert t.inputs == {}
    assert t.expected_outputs == []


def test_task_full():
    t = TaskSpec(
        goal="classify complaint",
        inputs={"message": "they charged me twice"},
        expected_outputs=["category", "summary"],
        constraints=["no PII in output"],
        validation=[ValidationRule(name="schema", description="must match output schema")],
    )
    assert t.expected_outputs == ["category", "summary"]
    assert t.validation[0].name == "schema"


def test_empty_goal_rejected():
    with pytest.raises(ValidationError):
        TaskSpec(goal="   ")


def test_roundtrip():
    t = TaskSpec(goal="g", constraints=["c1"])
    d = t.model_dump()
    t2 = TaskSpec.model_validate(d)
    assert t == t2
