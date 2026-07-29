"""Open-weight LLM regulatory flagger for the banking complaint agent.

Model half of the *hybrid* flag_regulatory tool. An instruction-tuned Qwen
model reads the customer message (plus the extracted product/issue) and
proposes a set of regulatory flags drawn from a fixed taxonomy. The
deterministic half --- an evidence guard that refuses any flag the message
text does not corroborate, plus a rule-based safety floor --- lives in
``agentlab.capstone.banking_tools``.

Why an LLM here: the rule-based flagger only fired UDAAP on the literal
co-occurrence of "fee" and "overdraft", and Reg X on "mortgage"/"loan". It
missed unauthorized-transaction complaints (Reg E), credit-card billing
disputes (Reg Z) and credit-report disputes (FCRA), and any UDAAP phrasing
that did not say "overdraft". Qwen generalizes over the phrasing; the
downstream evidence guard keeps every escalation auditable.

Determinism and safety mirror ``qwen_extractor``: greedy decoding, and
``propose`` never raises --- on any failure it returns ``None`` so the tool
falls back to the deterministic rule baseline.

Allowed flags:
    UDAAP  - unfair/deceptive/abusive acts (fees, misleading terms)
    Reg_X  - RESPA: mortgage servicing, escrow
    Reg_E  - EFTA: unauthorized electronic transfers, debit/ATM fraud
    Reg_Z  - TILA: credit-card billing errors, APR/interest disputes
    FCRA   - credit-report / credit-bureau disputes
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import torch

from agentlab.models.constants import DEFAULT_QWEN_MODEL

_DEFAULT_MODEL = os.environ.get("AGENTLAB_FLAGGER_MODEL", DEFAULT_QWEN_MODEL)

ALLOWED_FLAGS = ("UDAAP", "Reg_X", "Reg_E", "Reg_Z", "FCRA")

_SYSTEM_PROMPT = (
    "You are a bank compliance assistant. Read the customer message and decide "
    "which US consumer-finance regulations may be implicated. "
    "Return ONLY a JSON object with one key, flags, whose value is a list "
    "drawn from this set (use an empty list if none apply):\n"
    'UDAAP  - unfair, deceptive or abusive acts (fees, misleading terms)\n'
    'Reg_X  - mortgage servicing or escrow (RESPA)\n'
    'Reg_E  - unauthorized electronic transfer, debit-card or ATM fraud (EFTA)\n'
    'Reg_Z  - credit-card billing error, APR or interest dispute (TILA)\n'
    'FCRA   - credit-report or credit-bureau dispute\n'
    "Flag only what the message actually evidences. "
    "Do not add any text before or after the JSON object."
)


@dataclass
class QwenRegulatoryFlagger:
    """Instruction-tuned Qwen wrapper that reads a complaint and proposes regulatory flags from the fixed ALLOWED_FLAGS taxonomy.

    Attributes:
        model: The loaded causal language model.
        tokenizer: Tokenizer paired with the model.
        device: Torch device the model runs on.
    """

    model: Any
    tokenizer: Any
    device: torch.device

    @classmethod
    def load(
        cls,
        model: str | None = None,
        device: str | torch.device | None = None,
    ) -> "QwenRegulatoryFlagger":
        """Load the flagger model and tokenizer.

        Args:
            model: Model id to load; defaults to the AGENTLAB_FLAGGER_MODEL setting.
            device: Torch device or device string; defaults to CUDA when available.

        Returns:
            A ready-to-use QwenRegulatoryFlagger.
        """
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
    def propose(
        self,
        message: str,
        product: str = "",
        issue: str = "",
        max_new_tokens: int = 48,
    ) -> list[str] | None:
        """Return a list of proposed flags from ALLOWED_FLAGS, or None on failure."""
        if not message or not message.strip():
            return None
        user = message
        if product or issue:
            user = f"{message}\n\n(product={product}; issue={issue})"
        try:
            chat = [
                {"role": "system", "content": _SYSTEM_PROMPT},
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
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
        except Exception:
            return None
        return _parse_flags(text)


def _parse_flags(text: str) -> list[str] | None:
    """Pull {"flags": [...]} out of the model's text; keep only allowed flags."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "flags" not in obj:
        return None
    raw = obj["flags"]
    if not isinstance(raw, list):
        return None
    # Normalize casing/aliases onto the canonical taxonomy.
    canon = {f.lower(): f for f in ALLOWED_FLAGS}
    canon.update({"regx": "Reg_X", "rege": "Reg_E", "regz": "Reg_Z"})
    out: list[str] = []
    for item in raw:
        key = str(item).strip().lower().replace(" ", "_").replace("-", "_")
        if key in canon and canon[key] not in out:
            out.append(canon[key])
    return out


_DEFAULT_FLAGGER: QwenRegulatoryFlagger | None = None


def get_default_flagger() -> QwenRegulatoryFlagger:
    """Lazy singleton accessor; loads the model once per process."""
    global _DEFAULT_FLAGGER
    if _DEFAULT_FLAGGER is None:
        _DEFAULT_FLAGGER = QwenRegulatoryFlagger.load()
    return _DEFAULT_FLAGGER
