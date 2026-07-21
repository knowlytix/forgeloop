"""Production adapters.

The Anthropic SDK is imported lazily so the rest of the library remains
usable without it. Token counts are reported per call; rough dollar cost
is estimated with a model-pricing table you can override.
"""

from __future__ import annotations

import os
from typing import Any


# Per-million-token prices in USD. Override for your account.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}


class AnthropicAdapter:
    """Thin wrapper over the Anthropic SDK that conforms to BaseLM and reports usage."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        api_key: str | None = None,
        pricing: dict[str, dict[str, float]] | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic  # noqa: I001 - lazy import is the point
        except ImportError as e:
            raise ImportError(
                "anthropic SDK is required for AnthropicAdapter. Install with `pip install anthropic`."
            ) from e
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model
        self._pricing = pricing or DEFAULT_PRICING
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_dollars = 0.0

    def complete(self, prompt: str, **kwargs: Any) -> str:
        max_tokens = int(kwargs.get("max_tokens", 1024))
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage
        self.last_input_tokens = usage.input_tokens
        self.last_output_tokens = usage.output_tokens
        self.last_dollars = self._estimate_dollars(usage.input_tokens, usage.output_tokens)
        return response.content[0].text

    def _estimate_dollars(self, input_tokens: int, output_tokens: int) -> float:
        price = self._pricing.get(self._model)
        if price is None:
            return 0.0
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    @staticmethod
    def token_count(text: str) -> int:
        return max(1, len(text) // 4)
