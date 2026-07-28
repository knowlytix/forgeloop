from agentlab.evaluation.failure_modes import (
    ALL_INJECTORS,
    FailureMode,
    inject,
    prompt_injection,
    wrong_tool,
)


def test_every_failure_mode_has_injector():
    assert set(ALL_INJECTORS.keys()) == set(FailureMode)


def test_prompt_injection_appends_payload():
    s = prompt_injection().apply({"user_message": "hello"})
    assert "user_message" in s
    assert s["user_message"] != "hello"
    assert len(s["user_message"]) > len("hello")


def test_wrong_tool_injector_adds_hint():
    s = wrong_tool().apply({})
    assert "hints" in s
    assert isinstance(s["hints"], list)
    assert len(s["hints"]) >= 1


def test_inject_multiple_modes_composes():
    s = inject({}, FailureMode.PROMPT_INJECTION, FailureMode.WRONG_TOOL)
    assert "user_message" in s
    assert "hints" in s


def test_injection_does_not_mutate_original():
    original = {"user_message": "hi"}
    inject(original, FailureMode.PROMPT_INJECTION)
    assert original == {"user_message": "hi"}
