"""Deterministic mock LM with optional latency and token tracking."""

from __future__ import annotations

import time


class MockLM:
    """Deterministic LM. Scripted by `responses`, falls back to `default`.

    `latency_s` sleeps before returning to simulate slow models. Last call's
    input and output token counts are exposed for budget testing.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "",
        latency_s: float = 0.0,
    ) -> None:
        self._responses = responses or {}
        self._default = default
        self._latency_s = latency_s
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def complete(self, prompt: str, **kwargs) -> str:
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        response = self._responses.get(prompt, self._default)
        self.last_input_tokens = self.token_count(prompt)
        self.last_output_tokens = self.token_count(response)
        return response

    def token_count(self, text: str) -> int:
        return len(text.split())
