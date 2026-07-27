"""LoRA-backed intent classifier for the Policy gate (prompt injection / prohibited
advice).

The zero-shot Qwen guard in ``semantic_guard`` reads injection intent by prompting;
the design-of-experiments testing (Chapter 16) showed it is brittle on obfuscated
or misleading wording --- it both misses real injections and over-fires on
misleading-benign text. This is the same gap ``classify_complaint`` closed by
climbing from a frozen/zero-shot classifier to a LoRA SFT: a Qwen3-4B
``SEQ_CLS`` head fine-tuned (LoRA) on a clarity-augmented
``{none, prompt_injection, prohibited_advice}`` corpus (with misleading-benign hard
negatives) is markedly more robust (injection recall 0.31 -> 0.85 on the held-out
DoE rows, fewer benign false-fires).

Degrade-safe: if the adapter is absent or ``AGENTLAB_USE_LORA_INJECTION=0``,
:func:`get_injection_classifier` returns ``None`` and the policies fall back to the
zero-shot semantic guard.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from forgeloop.agents._paths import data_path

_DEFAULT_DIR = data_path("injection_classifier_lora")
_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass
class LoraInjectionClassifier:
    """Qwen-3B + LoRA SEQ_CLS classifier over {none, prompt_injection, prohibited_advice}.

    Attributes:
        model: PEFT-wrapped Qwen sequence-classification model in eval mode.
        tok: Tokenizer paired with the model.
        labels: Intent label names indexed by the model's logits.
        device: Torch device the model runs on.
        _cache: Memo mapping input text to its predicted label.
    """

    model: Any
    tok: Any
    labels: list[str]
    device: Any
    _cache: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path | None = None, device=None) -> "LoraInjectionClassifier":
        """Load the base Qwen model and LoRA adapter from disk.

        Args:
            path: Directory holding the adapter and ``labels.json``; defaults to
                the packaged ``injection_classifier_lora`` data path.
            device: Torch device to place the model on; defaults to CUDA when
                available, else CPU.

        Returns:
            A classifier ready for inference.
        """
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path) if path is not None else _DEFAULT_DIR
        labels = json.loads((path / "labels.json").read_text())
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(_MODEL_ID)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_ID, num_labels=len(labels),
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, str(path)).to(device).eval()
        return cls(model=model, tok=tok, labels=labels, device=device)

    @torch.no_grad()
    def classify(self, text: str) -> str:
        """Return the predicted intent label (``none`` when the text is empty)."""
        if not text or not text.strip():
            return "none"
        if text in self._cache:
            return self._cache[text]
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        idx = int(self.model(**enc).logits.argmax(-1).item())
        label = self.labels[idx]
        self._cache[text] = label
        return label


_DEFAULT: LoraInjectionClassifier | None = None
_LOAD_FAILED = False


def get_injection_classifier() -> LoraInjectionClassifier | None:
    """Cached LoRA intent classifier, or ``None`` (toggle off / adapter absent)."""
    global _DEFAULT, _LOAD_FAILED
    if os.environ.get("AGENTLAB_USE_LORA_INJECTION", "1") == "0":
        return None
    if _DEFAULT is None and not _LOAD_FAILED:
        try:
            _DEFAULT = LoraInjectionClassifier.load()
        except Exception:
            _LOAD_FAILED = True
            return None
    return _DEFAULT
