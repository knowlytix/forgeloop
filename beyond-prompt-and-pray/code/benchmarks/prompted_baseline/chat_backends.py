"""Pluggable chat backends for the prompted baseline.

A backend is anything with ``chat(system, user) -> str`` and ``chat_json(...)``.
Two are provided:

  - :class:`QwenChatBackend` --- the local Qwen3-4B-Instruct. To keep the
    comparison against the engineered agent fair (same model, only the
    engineering differs) it reuses the model and tokenizer that
    ``agentlab.models.qwen_extractor`` already loads, so it costs no extra VRAM
    and no extra download. Greedy decoding, for reproducibility.

  - :class:`AnthropicChatBackend` --- a larger frontier model (Sonnet by
    default) via the Anthropic API, ``temperature=0`` for near-determinism. This
    answers the separate question of how far a stronger model alone closes the
    gap, with none of the engineered backstops.

Both ``chat`` methods return ``""`` on any failure and ``chat_json`` returns
``None`` when there is no parseable JSON object, so the agent --- not the
backend --- decides how a prompt failure degrades.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


def first_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first balanced-looking JSON object out of model text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


@runtime_checkable
class ChatBackend(Protocol):
    name: str

    def chat(self, system: str, user: str, max_new_tokens: int = 256) -> str: ...

    def chat_json(
        self, system: str, user: str, max_new_tokens: int = 256
    ) -> dict[str, Any] | None: ...


# --------------------------------------------------------------------------- #
# Local Qwen3-4B backend (reuses the extractor's loaded weights)
# --------------------------------------------------------------------------- #


@dataclass
class QwenChatBackend:
    model: Any
    tokenizer: Any
    device: Any
    name: str = "qwen2.5-3b"

    @classmethod
    def shared(cls) -> "QwenChatBackend":
        """Build from the extractor's already-loaded base model."""
        from agentlab.models.qwen_extractor import get_default_extractor

        ex = get_default_extractor()
        return cls(model=ex.model, tokenizer=ex.tokenizer, device=ex.device)

    def chat(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        if not user or not user.strip():
            return ""
        import torch

        try:
            with torch.no_grad():
                chat = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                inputs = self.tokenizer.apply_chat_template(
                    chat,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(self.device)
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                gen = out[0, inputs["input_ids"].shape[1]:]
                return self.tokenizer.decode(gen, skip_special_tokens=True)
        except Exception:
            return ""

    def chat_json(
        self, system: str, user: str, max_new_tokens: int = 256
    ) -> dict[str, Any] | None:
        return first_json_object(self.chat(system, user, max_new_tokens))


# --------------------------------------------------------------------------- #
# Anthropic (Sonnet) backend
# --------------------------------------------------------------------------- #

_DEFAULT_ANTHROPIC_MODEL = os.environ.get("AGENTLAB_BENCH_ANTHROPIC_MODEL", "claude-sonnet-4-6")


@dataclass
class AnthropicChatBackend:
    client: Any
    model: str = _DEFAULT_ANTHROPIC_MODEL
    name: str = "sonnet"

    @classmethod
    def from_env(cls, model: str | None = None) -> "AnthropicChatBackend | None":
        """Construct from ``ANTHROPIC_API_KEY``; return None if unavailable so the
        benchmark can skip this backend with a clear message rather than crash."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        except Exception:
            return None
        return cls(client=client, model=model or _DEFAULT_ANTHROPIC_MODEL)

    def chat(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        if not user or not user.strip():
            return ""
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_new_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return "".join(parts)
        except Exception:
            return ""

    def chat_json(
        self, system: str, user: str, max_new_tokens: int = 256
    ) -> dict[str, Any] | None:
        return first_json_object(self.chat(system, user, max_new_tokens))
