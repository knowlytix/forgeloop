"""Local open-weight LM adapter, conforming to ``BaseLM``.

This is the book's default language model: a small open-weight model
(Qwen3-4B-Instruct) loaded locally, so every example runs on a real model
--- with real token counts and real latency --- without an API key and
without a mock. Use it anywhere the book needs an LM.

The model is loaded once per process (a lazy, per-(model, device) singleton)
and shared across adapter instances, so constructing a ``QwenAdapter`` is
cheap and a notebook that makes several is not penalized. ``complete`` is
greedy (``do_sample=False``), so output is deterministic for a fixed prompt
and model --- the reproducibility a mock used to provide, now from a real
model. ``last_input_tokens`` / ``last_output_tokens`` are the real token
counts of the most recent call, which the budget tracker (Chapter 7) reads.
"""

from __future__ import annotations

from typing import Any

import torch

from forgeloop.agents.models.constants import DEFAULT_QWEN_MODEL

_DEFAULT_MODEL = DEFAULT_QWEN_MODEL
_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _load(model_name: str, device: torch.device) -> tuple[Any, Any]:
    key = (model_name, str(device))
    if key not in _CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device.type if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device)
        model.eval()
        _CACHE[key] = (tokenizer, model)
    return _CACHE[key]


class QwenAdapter:
    """BaseLM adapter over a local Qwen instruct model.

    Parameters
    ----------
    model : str
        Hugging Face model id. Defaults to ``Qwen/Qwen3-4B-Instruct-2507``.
    device : str | torch.device | None
        Defaults to CUDA when available, else CPU.
    max_new_tokens : int
        Generation cap; override per call with ``complete(prompt, max_tokens=...)``.
    system : str | None
        Optional system message prepended to every call.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: str | torch.device | None = None,
        max_new_tokens: int = 512,
        system: str | None = None,
    ) -> None:
        self._model_name = model or _DEFAULT_MODEL
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._max_new_tokens = int(max_new_tokens)
        self._system = system
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    @torch.no_grad()
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Greedy-decode a single prompt and record the input and output token counts.

        Args:
            prompt: User prompt text.
            **kwargs: Accepts ``max_tokens`` to override the generation cap.

        Returns:
            The decoded completion text.
        """
        tokenizer, model = _load(self._model_name, self._device)
        messages: list[dict[str, str]] = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": prompt})
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=int(kwargs.get("max_tokens", self._max_new_tokens)),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        generated = out[0, prompt_len:]
        self.last_input_tokens = int(prompt_len)
        self.last_output_tokens = int(generated.shape[0])
        return tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def complete_batch(
        self, prompts: list[str], *, max_tokens: int | None = None, batch_size: int = 16, **kwargs: Any
    ) -> list[str]:
        """Greedy-decode many prompts in batched ``model.generate`` calls.

        Same decoding as ``complete`` (greedy -> deterministic), but processes
        ``batch_size`` prompts per forward pass instead of one at a time. Uses
        left-padding so each sequence's generated tokens start at a consistent
        offset. Returns one decoded string per prompt, in order.
        """
        tokenizer, model = _load(self._model_name, self._device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        max_new = int(max_tokens if max_tokens is not None else kwargs.get("max_tokens", self._max_new_tokens))
        outputs: list[str] = []
        try:
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start:start + batch_size]
                texts = [
                    tokenizer.apply_chat_template(
                        ([{"role": "system", "content": self._system}] if self._system else [])
                        + [{"role": "user", "content": p}],
                        add_generation_prompt=True, tokenize=False,
                    )
                    for p in chunk
                ]
                enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
                gen = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                prompt_len = enc["input_ids"].shape[1]
                for row in gen:
                    outputs.append(tokenizer.decode(row[prompt_len:], skip_special_tokens=True))
        finally:
            tokenizer.padding_side = prev_side
        return outputs

    def token_count(self, text: str) -> int:
        """Count tokens with the model's own tokenizer.

        Args:
            text: Text to encode.

        Returns:
            The number of tokens.
        """
        tokenizer, _ = _load(self._model_name, self._device)
        return len(tokenizer.encode(text))
