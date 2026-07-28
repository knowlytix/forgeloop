from agentlab.models import BaseLM, MockLM


def test_default():
    lm = MockLM(default="hello")
    assert lm.complete("anything") == "hello"


def test_scripted_responses():
    lm = MockLM(responses={"q1": "a1", "q2": "a2"}, default="fallback")
    assert lm.complete("q1") == "a1"
    assert lm.complete("q2") == "a2"
    assert lm.complete("q3") == "fallback"


def test_token_count():
    lm = MockLM()
    assert lm.token_count("one two three four") == 4


def test_protocol_conformance():
    assert isinstance(MockLM(), BaseLM)
