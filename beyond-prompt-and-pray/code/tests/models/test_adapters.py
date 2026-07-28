import pytest

from agentlab.models.adapters import DEFAULT_PRICING, AnthropicAdapter


def test_token_count_static_works_without_sdk():
    assert AnthropicAdapter.token_count("hello world") >= 1


def test_pricing_has_default_models():
    assert "claude-opus-4-6" in DEFAULT_PRICING
    for name, price in DEFAULT_PRICING.items():
        assert "input" in price and "output" in price
        assert price["input"] >= 0 and price["output"] >= 0


def test_init_raises_cleanly_without_sdk():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="anthropic"):
            AnthropicAdapter()
        return
    # If anthropic is installed in the env, just check construction succeeds with a fake key
    adapter = AnthropicAdapter(api_key="sk-fake")
    assert adapter is not None
