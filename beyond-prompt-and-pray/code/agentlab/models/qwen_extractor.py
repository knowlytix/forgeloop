"""Open-weight LLM fact extractor for the banking complaint agent.

This is the model half of a *hybrid* extractor. An instruction-tuned Qwen
model reads the free-text customer message and proposes ``product``,
``issue``, ``urgency`` and ``sentiment``. The deterministic half --- the
normalization onto the agent's enumerable taxonomy, plus the rule-based
fallback --- lives in ``agentlab.capstone.banking_tools``, which owns the
domain taxonomy. This module only knows how to load the model and turn a
message into a small parsed dict.

Why an LLM here: the keyword extractor it replaces misses any phrasing
outside its substring ladder. "Reverse this $35 charge on my checking
account" never contains the word *overdraft*, so the rule-based path
labels it ``account_issue`` and the UDAAP regulatory flag --- which keys
on ``overdraft_fee`` --- never fires. Qwen generalizes over the phrasing
and returns ``overdraft_fee``, so the escalation path stays correct.

Determinism: generation is greedy (``do_sample=False``), so extraction is
reproducible for a fixed input and model. ``extract`` never raises --- on a
load error, a generation error, or unparseable output it returns ``None``
and the caller falls back to the rule-based extractor, so the pipeline
degrades safely and stays usable with no GPU.

The default model is Qwen3-4B-Instruct (small, fast, already a fit for
structured extraction); override with the ``AGENTLAB_EXTRACTOR_MODEL``
environment variable or the ``model`` argument to ``load``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import torch

_DEFAULT_MODEL = os.environ.get("AGENTLAB_EXTRACTOR_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

_SYSTEM_PROMPT = (
    "You extract structured facts from a retail-bank customer message. "
    "Return ONLY a JSON object with exactly these keys: product, issue, urgency, sentiment.\n"
    'product must be one of ["checking_account","credit_card","mortgage","loan","unknown"].\n'
    'urgency must be one of ["high","normal"].\n'
    'sentiment must be one of ["negative","neutral"].\n'
    "issue is a short snake_case phrase describing the core problem. "
    "If the message is about a fee charged on a checking or overdraft account "
    "(for example an overdraft fee, or a charge the customer wants reversed on "
    "their checking account), set issue to overdraft_fee.\n"
    "Do not add any text before or after the JSON object."
)

# Keys we require to be present before we trust a parse.
_REQUIRED_KEYS = ("product", "issue", "urgency", "sentiment")


@dataclass
class QwenFactExtractor:
    model: Any
    tokenizer: Any
    device: torch.device

    @classmethod
    def load(
        cls,
        model: str | None = None,
        device: str | torch.device | None = None,
    ) -> QwenFactExtractor:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = model or _DEFAULT_MODEL
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        lm = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device.type if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            lm = lm.to(device)
        lm.eval()
        return cls(model=lm, tokenizer=tokenizer, device=lm.device)

    @torch.no_grad()
    def extract(self, message: str, max_new_tokens: int = 96) -> dict[str, str] | None:
        """Return a parsed {product, issue, urgency, sentiment} dict, or None on failure."""
        if not message or not message.strip():
            return None
        try:
            chat = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": message},
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
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
        except Exception:
            return None
        return _parse_json_object(text)


def _parse_json_object(text: str) -> dict[str, str] | None:
    """Pull the first JSON object out of the model's text and validate its keys."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not all(k in obj for k in _REQUIRED_KEYS):
        return None
    # Coerce everything to plain strings; the taxonomy normalization happens
    # downstream in banking_tools where the enum knowledge lives.
    return {k: str(obj[k]).strip().lower() for k in _REQUIRED_KEYS}


_DEFAULT_EXTRACTOR: QwenFactExtractor | None = None


def get_default_extractor() -> QwenFactExtractor:
    """Lazy singleton accessor; loads the model once per process."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = QwenFactExtractor.load()
    return _DEFAULT_EXTRACTOR
