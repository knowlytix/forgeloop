from agentlab.evaluation import DEFAULT_FACTORS, generate_test_cases


def test_generate_count():
    cases = generate_test_cases(12)
    assert len(cases) == 12


def test_each_case_has_task_message_and_behavior():
    for c in generate_test_cases(5):
        assert c.task is not None
        assert c.user_message
        assert c.expected_behavior
        assert isinstance(c.factors, dict)
        assert c.id.startswith("case-")


def test_factor_keys_match():
    cases = generate_test_cases(3)
    for c in cases:
        assert set(c.factors.keys()) == set(DEFAULT_FACTORS.keys())


def test_adversarial_intent_appears_in_message():
    factors = {
        "task_complexity": ["single_step"],
        "tool_risk": ["read_only"],
        "context_quality": ["clean"],
        "user_intent": ["adversarial"],
        "memory_state": ["relevant"],
        "evidence": ["available"],
        "budget": ["loose"],
        "policy": ["no_conflict"],
    }
    cases = generate_test_cases(1, factors=factors)
    assert "fee waiver" in cases[0].user_message.lower() or "ignore" in cases[0].user_message.lower()


def test_escalation_policy_changes_expectation():
    factors = {
        "task_complexity": ["single_step"],
        "tool_risk": ["read_only"],
        "context_quality": ["clean"],
        "user_intent": ["benign"],
        "memory_state": ["relevant"],
        "evidence": ["available"],
        "budget": ["loose"],
        "policy": ["requires_escalation"],
    }
    cases = generate_test_cases(1, factors=factors)
    assert "escalate" in cases[0].expected_behavior.lower()
