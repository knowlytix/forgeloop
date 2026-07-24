"""LoRA-backed polarity classifier for the output-disclosure gate (Chapter 13).

The geometric value-polarity gate (knowlytix ``ValuePolarityChecker``) decides
whether an asserted stance agrees with, contradicts, or is unrelated to the value a
policy holds. On short value tokens it is accurate, but on the full prose an agent
actually produces its u-space tension is out of distribution and it degrades (Ch13
held-out DoE test: contradiction F1 0.62, and 0.41 3-class on sentences). A
Qwen3-4B ``SEQ_CLS`` head fine-tuned (LoRA) on GMS-labeled, DoE-varied data reads
the prose directly and is markedly stronger (contradiction F1 0.83, recall 1.0,
robust across phrasing) -- the same climb from a zero-shot/geometric gate to a LoRA
SFT that ``classify_complaint`` and the injection classifier take.

The classifier's input pairs the policy reference with the claim
(``"Policy: <attribute> is <stance>. Claim: <text>"``), so the relative judgment
has both sides. It backs the knowlytix ``ClassifierDisclosureGuard``, which the
``GovernedRetriever`` uses to scan an answer against admitted policy stances.

Degrade-safe: if the adapter is absent or ``AGENTLAB_USE_LORA_POLARITY=0``,
:func:`get_polarity_classifier` returns ``None`` and the caller falls back to the
geometric guard.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
# The NL (premise+claim) adapter: it reads prose, which is what a disclosure
# scanner receives. The tuple adapter exists for the Chapter 13 ablation.
_DEFAULT_DIR = _REPO_ROOT / "data" / "polarity_classifier_qwen_nl"
_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Reference phrasing for each stance relation -- must match the premise the
# classifier was trained on (scripts/build_polarity_doe_dataset.py).
RELATION_PHRASE = {
    "has_unencrypted_channel_pii": "sending personal information over an unencrypted channel",
    "has_redaction": "redaction of personal information in tickets and logs",
    "has_identity_verification": "identity verification before an account is closed",
    "has_fraud_notice_exception": "the exception to the advance-notice rule when fraud is confirmed",
    "has_provisional_credit": "provisional credit to the customer while a dispute is investigated",
}


@dataclass
class LoraPolarityClassifier:
    """Qwen-4B + LoRA SEQ_CLS over {contradicted, supported, uncertain}."""

    model: Any
    tok: Any
    labels: list[str]
    device: Any
    _cache: dict[str, tuple[str, float]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path | None = None, device=None) -> LoraPolarityClassifier:
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
    def classify(self, text: str) -> tuple[str, float]:
        """Return ``(label, confidence)`` for a reference+claim string."""
        if not text or not text.strip():
            return "uncertain", 0.0
        if text in self._cache:
            return self._cache[text]
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=96).to(self.device)
        probs = torch.softmax(self.model(**enc).logits[0].float(), dim=-1)
        idx = int(probs.argmax().item())
        out = (self.labels[idx], float(probs[idx].item()))
        self._cache[text] = out
        return out


_DEFAULT: LoraPolarityClassifier | None = None
_LOAD_FAILED = False


def get_polarity_classifier() -> LoraPolarityClassifier | None:
    """Cached LoRA polarity classifier, or ``None`` (toggle off / adapter absent)."""
    global _DEFAULT, _LOAD_FAILED
    if os.environ.get("AGENTLAB_USE_LORA_POLARITY", "1") == "0":
        return None
    if _DEFAULT is None and not _LOAD_FAILED:
        try:
            _DEFAULT = LoraPolarityClassifier.load()
        except Exception:
            _LOAD_FAILED = True
            return None
    return _DEFAULT


def make_disclosure_guard(tau: float = 0.0, probes=(), path: Path | None = None):
    """Build a knowlytix ``ClassifierDisclosureGuard`` backed by this classifier.

    Returns ``None`` when the classifier is unavailable, so the caller can fall
    back to the geometric ``PolarityDisclosureGuard``.
    """
    from knowlytix.knowledge.rag.governed import ClassifierDisclosureGuard

    clf = LoraPolarityClassifier.load(path) if path is not None else get_polarity_classifier()
    if clf is None:
        return None
    return ClassifierDisclosureGuard(
        classify=clf.classify, relation_phrase=RELATION_PHRASE, tau=tau, probes=probes)
